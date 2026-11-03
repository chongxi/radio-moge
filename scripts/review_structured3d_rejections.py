"""Reproduce native Structured3D rejection evidence from retained ZIP members.

No label generation, threshold changes, exclusions or source acceptance occur.
Member SHA/CRC and original RGB are checked; whole archives are not rehashed.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from audit_structured3d_outcomes import rejection_kind
from indoor_download import write_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import canonical_json, publish_json, sha_file
from radio_moge.structured3d_depth import convert_structured3d, parse_errata
from visualize_structured3d_outcomes import depth_panel


MEMBERS = ('rgb_rawlight.png', 'depth.png', 'normal.png', 'semantic.png',
           'camera_pose.txt', 'layout.json')


def read_members(record, archive, archive_path, archive_sha256):
    row = record['image']
    member_path = PurePosixPath(row['source_member'])
    if (member_path.is_absolute() or '..' in member_path.parts
            or member_path.name != 'rgb_rawlight.png'
            or row['source_archive'] != archive_path.name):
        raise ValueError('Unexpected original source member identity')
    native = {}
    for name in MEMBERS:
        member = str(member_path.parent / name)
        proofs = [p for p in record['source_proofs'] if p.get('member') == member]
        if (len(proofs) != 1 or proofs[0]['source'] != str(archive_path)
                or proofs[0]['archive_sha256'] != archive_sha256):
            raise ValueError('Exact unique source proof required')
        info = archive.getinfo(member)
        data = archive.read(member)  # ZIP decoding also verifies CRC.
        if (info.CRC != proofs[0]['crc32']
                or hashlib.sha256(data).hexdigest() != proofs[0]['sha256']):
            raise ValueError('Retained source member changed')
        native[name] = data
    return native


def image_array(data):
    with Image.open(io.BytesIO(data)) as image:
        return np.array(image)


def reproduce(record, native):
    row = record['image']
    if (row['dataset'] != 'structured3d' or row['split'] != 'train'
            or record['reason'] != 'insufficient_verified_support'
            or hashlib.sha256(native['rgb_rawlight.png']).hexdigest() != row['sha256']
            or sha_file(row['image']) != row['sha256']
            or row.get('image_rotation_cw', 0)):
        raise ValueError('Unchanged original training RGB and rejection required')
    with Image.open(io.BytesIO(native['rgb_rawlight.png'])) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError('Unexpected native RGB orientation')
        rgba = np.array(image.convert('RGBA'))
    raw, semantic = image_array(native['depth.png']), image_array(native['semantic.png'])
    if raw.shape != (row['height'], row['width']) or rgba.shape[:2] != raw.shape:
        raise ValueError('Native grids differ from the frozen original RGB')
    _, valid, _, stats = convert_structured3d(raw, image_array(native['normal.png'])[..., :3],
        semantic, np.loadtxt(io.StringIO(native['camera_pose.txt'].decode())),
        json.loads(native['layout.json']), rgba[..., 3])
    if canonical_json(stats) != canonical_json(record['stats']) or valid.mean() >= .05:
        raise ValueError('Recomputed rejection statistics differ')
    return rgba, raw, semantic, valid, stats


def selection(records):
    """Bounded extremes across both configurations; not a prevalence estimate."""
    selected = {}
    for variant in ('empty', 'full'):
        group = [x for x in records if x['image']['configuration'] == variant and 'stats' in x]
        for flagged in (False, True):
            candidates = [x for x in group if x['stats']['source_needs_review'] is flagged]
            criteria = ([('candidate_fraction_before_image_gate', False),
                         ('candidate_fraction_before_image_gate', True),
                         ('layout_conflicting_pixels', True), ('layout_verified_pixels', True)]
                        if flagged else [('valid_fraction', False), ('valid_fraction', True)])
            for field, reverse in criteria:
                for value in sorted(candidates, key=lambda x: (x['stats'][field], x['image']['image']),
                                    reverse=reverse)[:2]:
                    selected.setdefault(value['image']['sha256'], []).append(
                        f'{variant}:flagged={flagged}:{field}:max={reverse}')
    return selected


def page(row, ordinal, rgba, raw, semantic, valid, stats):
    native = (raw > 0) & (raw < 65535) & (rgba[..., 3] == 255)
    # Semantics are categorical display colors, not a confidence or depth map.
    palette = np.array([[(i * 71) % 256, (i * 139) % 256, (i * 199) % 256]
                        for i in range(256)], np.uint8)
    panels = [Image.fromarray(rgba[..., :3]), depth_panel(raw.astype(np.float32) / 1000, native, .025, 64),
              Image.fromarray(palette[semantic]), Image.fromarray((valid * 255).astype(np.uint8)).convert('RGB')]
    width, height = 384, round(384 * row['height'] / row['width'])
    result = Image.new('RGB', (4 * width, height + 90), 'white')
    draw = ImageDraw.Draw(result)
    lines = [f"#{ordinal} {row['group']} room {row['room']} {row['configuration']} view {row['view']}",
        'Original RGB | native camera-Z log 0.025--64m | categorical semantics | unchanged final valid support',
        f"Native {stats['native_valid_fraction']:.5f}; before image gate {stats['candidate_fraction_before_image_gate']:.5f}; final {stats['valid_fraction']:.5f}",
        f"Layout checked/verified/conflicting {stats['layout_checked_pixels']}/{stats['layout_verified_pixels']}/{stats['layout_conflicting_pixels']}; source flag {stats['source_needs_review']}"]
    for i, line in enumerate(lines):
        draw.text((4, 4 + 20 * i), line, fill='black')
    for i, panel in enumerate(panels):
        result.paste(panel.resize((width, height), Image.Resampling.LANCZOS if i == 0
                                 else Image.Resampling.NEAREST), (i * width, 90))
    return result


def run(config, mode, attempt, preflight):
    root = Path(config['root'])
    audit_ref = file_ref(root / 'structured3d-outcomes-01/report.json')
    audit = checked_json(audit_ref)
    scope_ref = audit['unresolved_manifest']
    scope = [json.loads(line) for line in checked_path(scope_ref).read_text().splitlines()]
    cache = root / 'structured3d-native-cache-02'
    rejection_ref, policy_ref = [file_ref(cache / p) for p in ('rejections.json', 'packs/policy.json')]
    records, policy = checked_json(rejection_ref), checked_json(policy_ref)
    if (audit['stage'] != 'native_outcomes_accounted_not_accepted'
            or audit['manifest_sha256'] != config['manifest_sha256']
            or policy['manifest_sha256'] != config['manifest_sha256']
            or policy['minimum_image_valid_fraction'] != .05 or policy['dataset'] != 'structured3d'
            or policy['native_holes_filled'] is not False or policy['label_kind'] != 'synthetic_renderer'
            or not len(scope) == len(records) == audit['outcome_counts']['rejected_needs_disposition']):
        raise ValueError('Complete frozen native outcomes and unchanged policy required')
    repo = Path(__file__).resolve().parents[1]
    for name in ('src/radio_moge/structured3d_depth.py', 'src/radio_moge/geometry_sources.py'):
        if sha_file(repo / name) != policy['code_sha256'][name]:
            raise ValueError('Converter differs from the completed producer')
    bindings = {x['image']['sha256']: x for x in scope}
    if len(bindings) != len(scope) or len({x['image']['sha256'] for x in records}) != len(records):
        raise ValueError('Duplicate original rejection identity')
    errata_path = root / 'native-source/structured3d-metadata/errata.txt'
    errata_ref = file_ref(errata_path)
    if errata_ref['sha256'] != policy['upstream_errata']['sha256']:
        raise ValueError('Bound upstream errata changed')
    errata = parse_errata(errata_path.read_text())
    for item in records:
        binding = bindings.get(item['image']['sha256'])
        kind, _ = rejection_kind(item, errata, policy['upstream_errata'])
        if (binding is None or binding['image'] != item['image']
                or binding['native_reason'] != kind or binding['evidence'] != rejection_ref):
            raise ValueError('Rejection differs from frozen source outcome')
    generator_ref = file_ref(Path(__file__))
    if mode == 'census':
        previous = checked_json(file_ref(root / preflight / 'report.json'))
        if (previous['stage'] != 'rejection_reproduction_complete_not_reviewed'
                or previous['mode'] != 'preflight' or previous['generator'] != generator_ref
                or previous['outcomes'] != audit_ref or previous['rejections'] != rejection_ref
                or previous['reproduced_images'] != previous['expected_reproductions']):
            raise ValueError('Matching successful source reproduction preflight required')
    chosen = selection(records)
    if mode == 'preflight':
        probes = []
        for variant in ('empty', 'full'):
            for flagged in (False, True):
                candidates = [x for x in records if x['image']['configuration'] == variant
                              and 'stats' in x and x['stats']['source_needs_review'] is flagged]
                if candidates:
                    probes.append(min(candidates, key=lambda x: x['image']['image']))
        records = probes
        chosen = {x['image']['sha256']: ['preflight:configuration_and_flag'] for x in records}
    records = sorted(records, key=lambda x: (x['image']['source_archive'], x['image']['image']))
    output = root / attempt
    output.mkdir(exist_ok=False)
    status = dict(stage='reproducing_rejections', mode=mode, started_at=time.time(), processed=0,
        reproduced_images=0, expected_reproductions=sum('stats' in x for x in records),
        training_admission=False, optimizer_updates=0, source_quality_accepted=False)
    write_json(output / 'status.json', status)
    counts, pages, archives, archive = Counter(), [], {}, None
    active = None
    try:
        with (output / 'records.jsonl.incoming').open('x') as journal:
            for record in records:
                row = record['image']
                ordinal = bindings[row['sha256']]['corpus_ordinal']
                kind, _ = rejection_kind(record, errata, policy['upstream_errata'])
                value = dict(corpus_ordinal=ordinal, image=row, native_reason=kind,
                             source_proofs=record['source_proofs'], geometry_disposition='pending_review')
                if 'stats' in record:
                    name = row['source_archive']
                    if active != name:
                        if archive is not None:
                            archive.close()
                        path = Path(config['dataset_root']) / 'structured3d-perspective-v1-20261022/archives' / name
                        receipt_ref = file_ref(path.parent.parent / 'receipts' / (name + '.json'))
                        receipt = checked_json(receipt_ref)
                        if path.stat().st_size != receipt['bytes']:
                            raise ValueError('Retained archive size changed')
                        archives[name] = dict(receipt=receipt_ref, archive_sha256=receipt['sha256'],
                                              whole_archive_rehashed=False)
                        archive, active = zipfile.ZipFile(path), name
                    native = read_members(record, archive, path, receipt['sha256'])
                    rgba, raw, semantic, valid, stats = reproduce(record, native)
                    value.update(reproduced_statistics_exact=True, stats=stats)
                    status['reproduced_images'] += 1
                    if row['sha256'] in chosen:
                        target = output / f'{ordinal}.jpg'
                        temp = target.with_suffix('.jpg.incoming')
                        page(row, ordinal, rgba, raw, semantic, valid, stats).save(temp, format='JPEG', quality=95)
                        os.rename(temp, target)
                        pages.append(dict(corpus_ordinal=ordinal, page=file_ref(target),
                                          selection_reasons=chosen[row['sha256']]))
                else:
                    value['errata_identity_rechecked'] = True
                journal.write(canonical_json(value).decode() + '\n')
                counts[kind] += 1
                status['processed'] += 1
                if status['processed'] % 16 == 0:
                    status['updated_at'] = time.time()
                    write_json(output / 'status.json', status)
        if archive is not None:
            archive.close()
        if status['reproduced_images'] != status['expected_reproductions']:
            raise ValueError('Incomplete raw source reproduction')
        for ref in (audit_ref, rejection_ref, policy_ref, scope_ref, errata_ref):
            checked_path(ref)
        os.rename(output / 'records.jsonl.incoming', output / 'records.jsonl')
        report = dict(status, stage='rejection_reproduction_complete_not_reviewed', completed_at=time.time(),
            outcomes=audit_ref, rejections=rejection_ref, policy=policy_ref, generator=generator_ref,
            errata=errata_ref, category_counts=dict(counts), full_rejection_scope_checked=mode == 'census',
            pages=pages, archives=archives, records=file_ref(output / 'records.jsonl'),
            selection_method='Extremes by configuration and source flag; not a prevalence estimate',
            source_member_hashes_rechecked=True, archive_payload_hashes_rechecked=False,
            original_rgb_rechecked_for_reproduced_images=True, errata_rgb_rechecked=False,
            native_holes_filled=False, thresholds_changed=False,
            teacher_inference_performed=False, automatic_exclusions=0, network_requests=0,
            independent_physical_calibration=False)
        publish_json(output / 'report.json', report)
        write_json(output / 'status.json', report)
        print(json.dumps(dict(stage=report['stage'], processed=status['processed'],
                             reproduced_images=status['reproduced_images'], pages=len(pages))), flush=True)
        return report
    except BaseException as error:
        if archive is not None:
            archive.close()
        write_json(output / 'status.json', dict(status, stage='failed', error=type(error).__name__,
                                                message=str(error)[:1000]))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'census'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--preflight', default='structured3d-rejection-preflight-01')
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.attempt, args.preflight)):
        parser.error('Use attempt basenames')
    run(json.loads(args.config.read_text()), args.mode, args.attempt, args.preflight)
