"""Account completed COCO/NYU teacher sources and select bounded visual evidence.

No quality decision or exclusion is automatic. Full producer readback is a
prerequisite; this repeats receipt identity checks and selected payload reads.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_irs_teacher_outcomes import teacher_metrics
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json, read_geometry_member
from radio_moge.geometry_sampling import source_supervision


def rgb(row):
    data = Path(row['image']).read_bytes()
    if hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('Frozen RGB bytes changed')
    with Image.open(io.BytesIO(data)) as source:
        if source.getexif().get(274, 1) != 1 or row.get('image_rotation_cw', 0):
            raise ValueError('Unexpected RGB orientation')
        return source.convert('RGB')


def take_identity(originals, row):
    ordinal = row.get('corpus_ordinal')
    original = originals.pop(ordinal, None)
    if original is None or {k: v for k, v in row.items() if k != 'corpus_ordinal'} != original:
        raise ValueError('Changed, duplicate or outside-scope frozen teacher image')
    return ordinal


def select_views(records, dataset, preflight_ordinals):
    chosen = defaultdict(list)
    if dataset == 'coco2017':
        bins = [[r for r in records if r['stratum'] == name]
                for name in sorted({r['stratum'] for r in records})]
        for values in bins:
            for sort_key, reason in ((lambda r: (-r['p99_over_median'], r['corpus_ordinal']), 'aspect_top_three_p99'),
                                    (lambda r: (r['valid_fraction'], r['corpus_ordinal']), 'aspect_least_support_three')):
                for r in sorted(values, key=sort_key)[:3]:
                    chosen[r['corpus_ordinal']].append(reason)
        for r in records:
            if r['corpus_ordinal'] in preflight_ordinals:
                chosen[r['corpus_ordinal']].append('every_numeric_preflight_view')
        for r in sorted(records, key=lambda r: r['sha256']):
            if r['corpus_ordinal'] not in chosen:
                chosen[r['corpus_ordinal']].append('eight_additional_low_hash_views')
                if sum('eight_additional_low_hash_views' in v for v in chosen.values()) == 8:
                    break
    else:
        for sort_key, count, reason in ((lambda r: (-r['p99_over_median'], r['corpus_ordinal']), 6, 'top_six_p99'),
                                       (lambda r: (r['valid_fraction'], r['corpus_ordinal']), 4, 'least_support_four')):
            for r in sorted(records, key=sort_key)[:count]:
                chosen[r['corpus_ordinal']].append(reason)
    return [dict(r, selection_reasons=chosen[r['corpus_ordinal']])
            for r in records if r['corpus_ordinal'] in chosen]


def render(selected, output, policy_hash):
    output.mkdir()
    records = []
    for number, chosen in enumerate(selected):
        receipt = checked_json(chosen['receipt'])
        entry, = [e for e in receipt['entries'] if e['key'] == chosen['key']]
        pack = Path(chosen['receipt']['path']).parent / receipt['tar']
        if (pack.parent != Path(chosen['receipt']['path']).parent
                or pack.stat().st_size != receipt['bytes'] or receipt['policy_sha256'] != policy_hash
                or entry['sha256'] != chosen['member_sha256']):
            raise ValueError('Selected pack identity differs')
        depth, valid, _, meta = read_geometry_member(pack, entry)
        image = rgb(entry['image'])
        if (depth.shape != (image.height, image.width)
                or entry['image']['corpus_ordinal'] != chosen['corpus_ordinal']
                or source_supervision(meta) != 'relative_pseudo'):
            raise ValueError('Selected RGB/geometry identity or shape differs')
        q = np.quantile(depth[valid], [0, .01, .5, .99, 1])
        rel = q / q[2]
        np.testing.assert_allclose([valid.mean(), rel[1], rel[3], rel[4]],
            [chosen[k] for k in ('valid_fraction', 'p01_over_median', 'p99_over_median', 'max_over_median')],
            rtol=1e-6, atol=1e-12)
        colors = []
        for lo, hi in ((q[1], q[3]), (.25 * q[2], 16 * q[2])):
            value = np.zeros(depth.shape, np.float32)
            value[valid] = np.clip((np.log(depth[valid]) - np.log(lo)) / max(1e-6, np.log(hi / lo)), 0, 1)
            color = (np.stack((value, 1 - np.abs(2 * value - 1), 1 - value), -1) * 255).astype(np.uint8)
            color[~valid] = 0
            colors.append(Image.fromarray(color))
        panels = [image, *colors, Image.fromarray((valid * 255).astype(np.uint8)).convert('RGB')]
        page = Image.new('RGB', (4 * image.width, image.height + 76), 'white')
        draw = ImageDraw.Draw(page)
        for y, line in enumerate((f"#{number:03d} corpus {chosen['corpus_ordinal']} | {chosen['group']} | {meta['source_id']}",
                'RGB | per-image log p01-p99 | shared relative Z/median palette [0.25,16] | unchanged support',
                f"support={valid.mean():.6f}; p99/median={rel[3]:.4f}; max/median={rel[4]:.4f}. Blue near, red far, black invalid.",
                'Pseudo geometry only. Per-image colors are not comparable across views; no calibrated scale or confidence claim.')):
            draw.text((4, 3 + 18 * y), line, fill='black')
        for column, panel in enumerate(panels):
            page.paste(panel, (column * image.width, 76))
        path = output / f'{number:03d}.png'
        temporary = path.with_suffix('.png.incoming')
        page.save(temporary, format='PNG')
        os.rename(temporary, path)
        records.append(dict(chosen, page=file_ref(path), relative_quantiles=rel.tolist(),
                            selected_member_payload_rechecked=True, original_rgb_rechecked=True))
    publish_json(output / 'report.json', dict(stage='selected_teacher_evidence_rendered_not_reviewed',
        records=records, rendered_images=len(records), shared_relative_palette=[.25, 16],
        per_image_palette_also_shown=True, image_resampling=False, depth_or_mask_changed=False,
        pack_payload_hashes_rechecked=False, training_admission=False, optimizer_updates=0))


def run(config, dataset, cache_name, preflight_name, attempt):
    if dataset not in ('coco2017', 'nyuv2'):
        raise ValueError('This audit is limited to COCO and NYU teacher sources')
    root, expected = Path(config['root']), config['expected_counts'][dataset]
    source_ref = file_ref(root / 'inventory.json')
    source = checked_json(source_ref)
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    checked_path(inventory_ref)
    if (source['stage'] != 'inventory_complete' or source['manifest_sha256'] != config['manifest_sha256']
            or source['dataset_counts'] != config['expected_counts']):
        raise ValueError('Matching full inventory required')
    cache = root / cache_name
    refs = {name: file_ref(cache / path) for name, path in
            [('generation', 'report.json'), ('packs', 'packs/packs.json'),
             ('policy', 'packs/policy.json'), ('rejections', 'rejections.json')]}
    gen, packs, policy, rejects = [checked_json(refs[n]) for n in ('generation', 'packs', 'policy', 'rejections')]
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if (gen['stage'] != 'complete' or gen['mode'] != 'generate' or gen['completed'] != expected
            or gen['verified_cache_members'] != expected or packs['count'] != expected
            or gen['rejected'] != 0 or rejects or gen['teacher_frozen'] is not True
            or gen['training_admission'] is not False or gen['training_started'] is not False
            or gen['optimizer_updates'] != 0 or gen['policy_sha256'] != policy_hash
            or packs['policy_sha256'] != policy_hash or policy['dataset'] != dataset
            or policy['teacher'] != config['teacher'] or policy['manifest_sha256'] != config['manifest_sha256']
            or policy['native_metric_gt'] is not False or policy['label_kind'] != 'teacher_pseudo'):
        raise ValueError('Complete zero-rejection teacher producer readback required')
    preflight_ref = file_ref(root / preflight_name / 'report.json')
    preflight = checked_json(preflight_ref)
    if preflight['stage'] != 'complete' or preflight['passed'] is not True or preflight['policy_sha256'] != policy_hash:
        raise ValueError('Matching numeric preflight required')
    probes = {r['ordinal'] for r in preflight['examples']} if dataset == 'coco2017' else set()
    originals = {}
    with readonly_db(inventory_ref['path']) as db:
        for ordinal, image, encoded in db.execute(
                'SELECT ordinal,image,row_json FROM images WHERE dataset=? ORDER BY ordinal', (dataset,)):
            row = json.loads(encoded)
            row['image'] = image
            if row['dataset'] != dataset or row['split'] != 'train':
                raise ValueError('Frozen training source differs')
            originals[ordinal] = row
    if len(originals) != expected or not probes <= set(originals):
        raise ValueError('Frozen training scope count differs')
    records, keys = [], set()
    for packet in packs['packs']:
        receipt_ref = dict(path=str(cache / 'packs' / packet['receipt']), sha256=packet['sha256'])
        receipt = checked_json(receipt_ref)
        tar = cache / 'packs' / receipt['tar']
        if (Path(receipt_ref['path']).parent != cache / 'packs' or tar.parent != cache / 'packs'
                or tar.stat().st_size != receipt['bytes'] or receipt['policy_sha256'] != policy_hash
                or len(receipt['entries']) != packet['count']):
            raise ValueError('Pack receipt identity differs')
        for entry in receipt['entries']:
            row, meta = entry['image'], entry['geometry_metadata']
            ordinal = take_identity(originals, row)
            if entry['key'] in keys or source_supervision(meta) != 'relative_pseudo':
                raise ValueError('Duplicate key or incorrect supervision')
            keys.add(entry['key'])
            image_shape = None
            if 'height' not in row or 'width' not in row:
                image = rgb(row)
                image_shape = (image.height, image.width)
            support, low, high, maximum = teacher_metrics(entry, policy_hash,
                config['teacher']['revision'], dataset, image_shape)
            h, w = meta['shape']
            stratum = row['scene_type'] if dataset == 'nyuv2' else (
                'landscape' if w / h > 1.05 else 'portrait' if w / h < .95 else 'square')
            records.append(dict(corpus_ordinal=ordinal, image=row['image'], sha256=row['sha256'], group=row['group'],
                stratum=stratum, key=entry['key'], member_sha256=entry['sha256'], receipt=receipt_ref,
                valid_fraction=support, p01_over_median=low, p99_over_median=high, max_over_median=maximum))
    if originals or len(records) != expected:
        raise ValueError('Not every frozen source row was accounted once')
    output = root / attempt
    output.mkdir(exist_ok=False)
    records.sort(key=lambda r: r['corpus_ordinal'])
    incoming = output / 'outcomes.jsonl.incoming'
    with incoming.open('x') as stream:
        for row in records:
            stream.write(canonical_json(row).decode() + '\n')
    os.rename(incoming, output / 'outcomes.jsonl')
    selected = select_views(records, dataset, probes)
    publish_json(output / 'selection.json', selected)
    probabilities = [0, .01, .5, .9, .99, 1]
    distributions = {key: np.quantile([r[key] for r in records], probabilities).tolist()
                     for key in ('valid_fraction', 'p01_over_median', 'p99_over_median', 'max_over_median')}
    render(selected, output / 'visuals', policy_hash)
    for ref in [source_ref, preflight_ref, *refs.values()]:
        checked_path(ref)
    report = dict(stage='completed_teacher_source_accounted_not_accepted', dataset=dataset,
        policy_sha256=policy_hash, manifest_sha256=config['manifest_sha256'], expected_scope=expected,
        candidate_images=len(records), rejected_images=0, groups=len({r['group'] for r in records}),
        strata=dict(Counter(r['stratum'] for r in records)), probabilities=probabilities,
        relative_statistics=distributions, full_support_images=sum(r['valid_fraction'] == 1 for r in records),
        below_five_percent_support=sum(r['valid_fraction'] < .05 for r in records),
        inventory=inventory_ref, source_report=source_ref, preflight=preflight_ref, **refs,
        outcomes=file_ref(output / 'outcomes.jsonl'), selection=file_ref(output / 'selection.json'),
        visual_evidence=file_ref(output / 'visuals/report.json'), producer_array_readback_complete=True,
        independent_full_array_readback_performed=False, selected_member_payloads_rechecked=True,
        source_quality_accepted=False, automatic_exclusions=0, training_admission=False, optimizer_updates=0,
        audit_code=file_ref(Path(__file__)), metrics_helper=file_ref(Path(__file__).with_name('audit_irs_teacher_outcomes.py')))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(dataset=dataset, candidate_images=len(records), selected=len(selected),
                         relative_statistics=distributions)), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', choices=('coco2017', 'nyuv2'), required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--preflight', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.cache, args.preflight, args.attempt)):
        parser.error('Use attempt basenames')
    run(json.loads(args.config.read_text()), args.dataset, args.cache, args.preflight, args.attempt)
