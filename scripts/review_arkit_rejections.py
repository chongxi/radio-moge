"""Reproduce ARKit numeric failures and retain publisher ZIP absence evidence.

Numeric conversion reads retained bytes only. Missing exact timestamps are
checked against publisher central directories, one archive at a time; no depth
payload is downloaded, substituted or regenerated. No admission is issued.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import requests

from audit_arkit_outcomes import rejection_kind
from cache_arkit_geometry import convert, load_image
from indoor_download import RangeFile, write_json
from radio_moge.arkit_depth import native_depth
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json, sha_file
from visualize_arkit_faro_probe import RetainedBundle
from visualize_arkit_outcomes import aligned_display, depth_colors


KINDS = ('lowres_depth', 'confidence', 'lowres_wide', 'lowres_wide_intrinsics', 'vga_wide_intrinsics')


class DirectoryTail(io.BytesIO):
    """A ZIP directory at its original offsets, with no member payload access."""
    def __init__(self, data, start, size):
        if start < 0 or size != start + len(data):
            raise ValueError('Invalid retained directory extent')
        super().__init__(data)
        self.start, self.size = start, size

    def tell(self):
        return self.start + super().tell()

    def seek(self, offset, whence=0):
        position = offset + {0: 0, 1: self.tell(), 2: self.size}[whence]
        if position < self.start:
            raise ValueError('Attempt to read archive payload outside retained directory')
        super().seek(position - self.start)
        return position


def directory_absences(data, start, size, expected):
    if len(set(expected)) != len(expected) or not expected:
        raise ValueError('Distinct missing native members required')
    with zipfile.ZipFile(DirectoryTail(data, start, size)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not names or len(names) > 200000:
            raise ValueError('Ambiguous or unexpectedly large publisher directory')
        if archive.start_dir != start or set(expected) & set(names):
            raise ValueError('Claimed missing member exists or directory offset changed')
        return dict(member_count=len(names), listing_sha256=hashlib.sha256(
            canonical_json(names)).hexdigest(), absent_members=sorted(expected))


def missing_request(record):
    row = record['image']
    if (rejection_kind(record) != 'missing_exact_timestamp_native_member'
            or row['dataset'] != 'arkitscenes' or row['split'] != 'train'):
        raise ValueError('Bound training missing-timestamp outcome required')
    member = record['stats']['detail']
    stem = PurePosixPath(row['source_member']).stem
    if member != f'lowres_depth/{stem}.png':
        raise ValueError('Unexpected missing native member')
    base = row['source_url'].rsplit('/', 1)[0]
    if not base.startswith('https://docs-assets.developer.apple.com/') or '/Training/' not in base:
        raise ValueError('Only exact training publisher archive allowed')
    return base + '/lowres_depth.zip', member


def numeric_sources(record, source_root):
    row = record['image']
    if rejection_kind(record) not in ('low_registered_support', 'image_registration_failed'):
        raise ValueError('Numeric native rejection required')
    base, stem = row['source_url'].rsplit('/', 1)[0], PurePosixPath(row['source_member']).stem
    proofs = []
    for kind in KINDS:
        url = base + '/' + kind + '.zip'
        suffix = '.pincam' if kind.endswith('intrinsics') else '.png'
        path = source_root / hashlib.sha256(url.encode()).hexdigest()[:16] / kind / (stem + suffix)
        proofs.append(json.loads(path.with_name(path.name + '.receipt.json').read_text()))
    bundle = RetainedBundle(source_root, proofs)
    depth, valid, kd, kr, actual_proofs, stats = convert(row, bundle)
    if (canonical_json(stats) != canonical_json(record['stats']) or valid.mean() >= .05
            or actual_proofs != proofs):
        raise ValueError('Reproduced native rejection differs from producer statistics')
    return depth, valid, kd, kr, proofs, stats, bundle


def selection(records):
    chosen = {}
    for turn in (0, 90, 180, 270):
        for passed in (False, True):
            rows = [x for x in records if 'image_registration_pass' in x['stats']
                    and x['image']['image_rotation_cw'] == turn
                    and x['stats']['image_registration_pass'] is passed]
            field = 'valid_fraction' if passed else 'paired_rgb_mae'
            for key, reverse in ((field, False), (field, True), ('native_confident_stable_pixels', True)):
                if rows:
                    item = sorted(rows, key=lambda x: (x['stats'][key], x['image']['image']), reverse=reverse)[0]
                    chosen.setdefault(item['image']['image'], []).append(f'cw{turn}:registered={passed}:{key}:max={reverse}')
    return chosen


def render(row, ordinal, reproduced):
    depth, valid, _, _, _, stats, bundle = reproduced
    base, stem = row['source_url'].rsplit('/', 1)[0], PurePosixPath(row['source_member']).stem
    paths = {kind: bundle.get(base, kind, stem)[0] for kind in KINDS}
    low, rgb_camera = [np.loadtxt(paths[k]).tolist() for k in ('lowres_wide_intrinsics', 'vga_wide_intrinsics')]
    turn = row['image_rotation_cw']
    rgb, low_rgb, sampled, error, _, _ = aligned_display(load_image(row['image']),
        load_image(paths['lowres_wide']), low, rgb_camera, turn, stats)
    before_depth, before_valid, _ = native_depth(load_image(paths['lowres_depth']),
        load_image(paths['confidence']), low, turn)
    if int(before_valid.sum()) != stats['native_confident_stable_pixels']:
        raise ValueError('Display support differs before image gate')
    heat = (np.clip(error / .08, 0, 1)[..., None] * [255, 110, 0]).astype(np.uint8)
    panels = [(rgb, 'Original upright VGA RGB'), (low_rgb, 'Retained native lowres RGB'),
        ((sampled * 255).clip(0, 255).astype(np.uint8), 'VGA projected via K to native grid'),
        (depth_colors(before_depth, before_valid), 'Stable/confident native Z BEFORE RGB gate'),
        (depth_colors(depth, valid), 'Final valid Z; rejected below 5% support'),
        (heat, 'RGB MAE: black=0, orange>=0.08')]
    page = Image.new('RGB', (1200, 1200), 'white')
    draw = ImageDraw.Draw(page)
    lines = [f'#{ordinal} {row["group"]} | {stem} | CW {turn}',
        f'Registration pass={stats["image_registration_pass"]}; native stable={before_valid.mean():.3%}; final={valid.mean():.3%}',
        f'RGB MAE={stats["paired_rgb_mae"]:.5f}; p90={stats["paired_rgb_error_p90"]:.5f}; exact producer statistics reproduced.',
        'Depth: fixed log 0.125-8m; black invalid. Before-gate panel is diagnostic ONLY, never an accepted label.']
    for i, line in enumerate(lines):
        draw.text((8, 8 + 20 * i), line, fill='black')
    for i, (array, label) in enumerate(panels):
        x, y = i % 3 * 400, 100 + i // 3 * 550
        draw.text((x + 5, y), label, fill='black')
        panel = ImageOps.contain(Image.fromarray(array), (392, 520),
            Image.Resampling.LANCZOS if i < 3 else Image.Resampling.NEAREST)
        page.paste(panel, (x + (400 - panel.width) // 2, y + 22))
    return page


def run(config, mode, attempt, preflight):
    root = Path(config['root'])
    audit_ref = file_ref(root / 'arkit-outcomes-01/report.json')
    audit = checked_json(audit_ref)
    records, policy = checked_json(audit['rejections']), checked_json(audit['policy'])
    code = Path(__file__).parents[1]
    if (audit['stage'] != 'native_outcomes_accounted_not_accepted'
            or audit['manifest_sha256'] != config['manifest_sha256']
            or policy['minimum_image_valid_fraction'] != .05 or policy['native_holes_filled'] is not False
            or len(records) != audit['outcome_counts']['rejected']
            or hashlib.sha256(canonical_json(policy)).hexdigest() != audit['policy_sha256']
            or sha_file(code / 'scripts/cache_arkit_geometry.py') != policy['script_sha256']):
        raise ValueError('Completed original outcomes and unchanged producer policy required')
    for name in ('src/radio_moge/arkit_depth.py', 'src/radio_moge/geometry_sources.py', 'scripts/indoor_download.py'):
        if sha_file(code / name) != policy['code_sha256'][name]:
            raise ValueError('Original source conversion or range reader changed')
    with readonly_db(checked_path(audit['index'])) as db:
        bindings = {image: dict(ordinal=ordinal, row=json.loads(encoded), reason=reason,
                               evidence=json.loads(evidence)) for ordinal, image, encoded, reason, evidence in
            db.execute("SELECT ordinal,image,row_json,reason,evidence_json FROM images WHERE outcome='rejected'")}
    if len(bindings) != len(records) or len({x['image']['image'] for x in records}) != len(records):
        raise ValueError('Rejected scope is not unique')
    for item in records:
        b = bindings[item['image']['image']]
        if b['row'] != item['image'] or b['reason'] != rejection_kind(item) or b['evidence'] != audit['rejections']:
            raise ValueError('Rejection differs from frozen native outcome')
    generator = file_ref(Path(__file__))
    if mode == 'census':
        previous = checked_json(file_ref(root / preflight / 'report.json'))
        if (previous['stage'] != 'rejection_reproduction_complete_not_reviewed'
                or previous['mode'] != 'preflight' or previous['generator'] != generator
                or previous['outcomes'] != audit_ref):
            raise ValueError('Matching source reproduction preflight required')
    else:
        probes = {}
        for item in records:
            key = (item['image']['image_rotation_cw'], item['stats'].get('image_registration_pass', 'missing'))
            probes.setdefault(key, item)
        records = list(probes.values())
    chosen = selection(records)
    if len(chosen) > 32:
        raise ValueError('Unbounded visual selection')
    output = root / attempt
    output.mkdir(exist_ok=False)
    status = dict(stage='reproducing_rejections', mode=mode, started_at=time.time(), processed=0,
        numeric_reproduced=0, missing_members_verified=0, expected=len(records),
        training_admission=False, optimizer_updates=0)
    write_json(output / 'status.json', status)
    numeric = [x for x in records if x['reason'] == 'insufficient_registered_native_support']
    missing = [x for x in records if x['reason'] == 'missing_exact_timestamp_native_member']
    journal_path = output / 'records.jsonl.incoming'
    pages, directories, counts = [], [], Counter()
    try:
        with journal_path.open('x') as journal:
            for item in numeric:
                row = item['image']
                b = bindings[row['image']]
                reproduced = numeric_sources(item, root / 'native-source/arkitscenes')
                value = dict(corpus_ordinal=b['ordinal'], image=row, kind=b['reason'],
                    stats=reproduced[5], source_proofs=reproduced[4], statistics_exactly_reproduced=True)
                if row['image'] in chosen:
                    path = output / f'{b["ordinal"]}.png'
                    temp = path.with_suffix('.png.incoming')
                    render(row, b['ordinal'], reproduced).save(temp, format='PNG')
                    temp.rename(path)
                    pages.append(dict(corpus_ordinal=b['ordinal'], page=file_ref(path),
                                      selection_reasons=chosen[row['image']]))
                journal.write(canonical_json(value).decode() + '\n')
                counts[b['reason']] += 1
                status['processed'] += 1
                status['numeric_reproduced'] += 1
                if status['processed'] % 32 == 0:
                    status['updated_at'] = time.time()
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
            by_archive = {}
            for item in missing:
                url, _ = missing_request(item)
                by_archive.setdefault(url, []).append(item)
            with requests.Session() as session:
                for url, group in sorted(by_archive.items()):
                    expected = [missing_request(x)[1] for x in group]
                    # Only central-directory/EOCD bytes are retained. Never read a member.
                    remote = RangeFile(url, session)
                    try:
                        with zipfile.ZipFile(remote) as archive:
                            start = archive.start_dir
                        size = remote.size
                        if not 0 < size - start <= 16 * 1024**2:
                            raise ValueError('Unexpected publisher central-directory extent')
                        data = remote.fetch(start, size - start)
                        stats = directory_absences(data, start, size, expected)
                        received = remote.bytes_received
                    finally:
                        remote.close()
                    directory = output / ('directory-' + hashlib.sha256(url.encode()).hexdigest()[:16] + '.bin')
                    with directory.with_suffix('.bin.incoming').open('xb') as file:
                        file.write(data)
                    directory.with_suffix('.bin.incoming').rename(directory)
                    proof = dict(source=url, archive_bytes=size, directory_start=start,
                        retained_directory=file_ref(directory), fetched_bytes=received,
                        checked_at=time.time(), payload_members_downloaded=0, **stats)
                    directories.append(proof)
                    for item in group:
                        row = item['image']
                        b = bindings[row['image']]
                        if sha_file(row['image']) != row['sha256']:
                            raise ValueError('Missing-depth frame original RGB changed')
                        journal.write(canonical_json(dict(corpus_ordinal=b['ordinal'], image=row,
                            kind=b['reason'], producer_detail=item['stats']['detail'],
                            publisher_directory=proof, exact_member_absence_verified=True)).decode() + '\n')
                        counts[b['reason']] += 1
                        status['processed'] += 1
                        status['missing_members_verified'] += 1
                    status['updated_at'] = time.time()
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        if status['processed'] != len(records) or (mode == 'census' and dict(counts) != audit['rejection_reasons']):
            raise ValueError('Reproductions do not cover every frozen rejection')
        checked_path(audit_ref)
        checked_path(audit['rejections'])
        journal_path.rename(output / 'records.jsonl')
        report = dict(**status, completed_at=time.time(), generator=generator, outcomes=audit_ref,
            rejections=audit['rejections'], policy=audit['policy'], policy_sha256=audit['policy_sha256'],
            records=file_ref(output / 'records.jsonl'), counts=dict(counts), pages=pages,
            directories=directories, archive_count=len(directories), source_members_per_numeric_image=5,
            publisher_directory_bytes_fetched=sum(x['fetched_bytes'] for x in directories),
            gpu_used=False, native_payload_network_downloads=0, source_quality_accepted=False,
            automatic_exclusions=0, no_threshold_changes=True, full_archive_rehashed=False)
        report['stage'] = 'rejection_reproduction_complete_not_reviewed'
        publish_json(output / 'report.json', report)
        write_json(output / 'status.json', report)
        print(json.dumps({k:report[k] for k in ('stage','processed','counts','archive_count')}), flush=True)
        return report
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'census'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--preflight', default='arkit-rejection-preflight-01')
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.attempt, args.preflight)):
        parser.error('Use new attempt basenames')
    run(json.loads(args.config.read_text()), args.mode, args.attempt, args.preflight)
