"""Display bounded extremes from completed Hypersim teacher routes, without admission.

Select the six largest relative p99 values and four least-supported views from
all teacher routes. These are diagnostic ranks, never exclusion thresholds.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, read_geometry_member, sha_file


def run(config, attempt):
    root = Path(config['root'])
    audit_ref = file_ref(root / 'hypersim-combined-routes-01/report.json')
    audit = checked_json(audit_ref)
    generation = checked_json(audit['generation_report'])
    if (audit['stage'] != 'source_routes_accounted_not_accepted' or audit['dataset'] != 'hypersim'
            or audit['manifest_sha256'] != config['manifest_sha256']
            or generation['stage'] != 'complete' or generation['mode'] != 'generate'
            or generation['completed'] != generation['verified_cache_members']
            or generation['policy_sha256'] != audit['policy_sha256']
            or audit['producer_array_readback_complete'] is not True):
        raise ValueError('Complete bound Hypersim route accounting and producer readback required')
    routes = []
    with checked_path(audit['routes']).open() as file:
        for line in file:
            row = json.loads(line)
            if row['route'] == 'teacher_candidate':
                routes.append(row)
    if (len(routes) != generation['completed'] or len(routes) < 10
            or len({r['corpus_ordinal'] for r in routes}) != len(routes)):
        raise ValueError('Every distinct teacher route must be present')
    selection, reasons = {}, {}
    for name, chosen in (
        ('top_six_relative_p99', sorted(routes, key=lambda r: (-r['p99_over_median'], r['corpus_ordinal']))[:6]),
        ('least_support_four', sorted(routes, key=lambda r: (r['valid_fraction'], r['corpus_ordinal']))[:4])):
        for row in chosen:
            ordinal = row['corpus_ordinal']
            selection[ordinal] = row
            reasons.setdefault(ordinal, []).append(name)
    output = root / attempt
    output.mkdir(exist_ok=False)
    records = []
    for number, (ordinal, chosen) in enumerate(sorted(selection.items())):
        receipt = checked_json(chosen['evidence'])
        entries = [e for e in receipt['entries'] if e['key'] == chosen['entry_key']]
        if len(entries) != 1 or receipt['policy_sha256'] != audit['policy_sha256']:
            raise ValueError('Selected teacher receipt identity differs')
        entry = entries[0]
        row = entry['image']
        pack = Path(chosen['evidence']['path']).parent / receipt['tar']
        if (pack.parent != Path(chosen['evidence']['path']).parent
                or pack.stat().st_size != receipt['bytes'] or row['corpus_ordinal'] != ordinal
                or any(row[k] != chosen[k] for k in ('image', 'sha256', 'group'))):
            raise ValueError('Selected teacher payload differs from its frozen route')
        depth, valid, _, meta = read_geometry_member(pack, entry)
        if (meta['label_kind'] != 'teacher_pseudo' or meta['metric_calibrated'] is not False
                or meta['camera_kind'] != 'teacher_estimated' or meta['dataset'] != 'hypersim'
                or meta['teacher_revision'] != config['teacher']['revision']
                or meta['policy_sha256'] != audit['policy_sha256']
                or meta['native_source_review_unresolved'] is not True
                or meta['native_reason'] != chosen['native_reason']
                or row['split'] != 'train' or row.get('image_rotation_cw', 0)
                or sha_file(row['image']) != row['sha256']):
            raise ValueError('Original RGB and flagged nonmetric teacher provenance required')
        with Image.open(row['image']) as source:
            if source.mode != 'RGB' or source.getexif().get(274, 1) != 1:
                raise ValueError('Unexpected original RGB format or orientation')
            rgb = source.copy()
        if depth.shape != (rgb.height, rgb.width):
            raise ValueError('Original RGB and teacher grids differ')
        quantiles = np.quantile(depth[valid], [0, .01, .5, .99, 1])
        relative = quantiles / quantiles[2]
        np.testing.assert_allclose([valid.mean(), relative[1], relative[3], relative[4]],
            [chosen[k] for k in ('valid_fraction', 'p01_over_median', 'p99_over_median', 'max_over_median')],
            rtol=1e-6, atol=1e-12)
        value = np.zeros(depth.shape, np.float32)
        # The observed full-source p99/median maximum is about 111. A shared
        # upper color limit of 128 exposes those backgrounds instead of hiding
        # every value above 16 in one saturated color. This is display only.
        value[valid] = np.clip(np.log2(depth[valid] / quantiles[2] / .25) / 9, 0, 1)
        colors = (np.stack((value, 1 - np.abs(2 * value - 1), 1 - value), -1) * 255).astype(np.uint8)
        colors[~valid] = 0
        page = Image.new('RGB', (3 * rgb.width, rgb.height + 84), 'white')
        draw = ImageDraw.Draw(page)
        draw.text((6, 5), f"corpus {ordinal} | {row['group']} | {meta['source_id']}", fill='black')
        draw.text((6, 23), f"RGB | relative teacher Z / median | unchanged valid support {valid.mean():.6f}; p99/median={relative[3]:.3f}", fill='black')
        draw.text((6, 41), 'One palette for every view: blue <=0.25, green 5.657, red >=128; black=invalid. NOT calibrated meters.', fill='black')
        draw.text((6, 59), 'No resizing, depth editing, confidence claim or native-source repair. Native reason: ' + chosen['native_reason'], fill='black')
        page.paste(rgb, (0, 84))
        page.paste(Image.fromarray(colors), (rgb.width, 84))
        page.paste(Image.fromarray((valid * 255).astype(np.uint8)).convert('RGB'), (2 * rgb.width, 84))
        target = output / f'{number:03d}.png'
        temporary = target.with_suffix('.png.incoming')
        page.save(temporary, format='PNG')
        os.rename(temporary, target)
        records.append(dict(corpus_ordinal=ordinal, group=row['group'], source_id=meta['source_id'],
            image=dict(path=row['image'], sha256=row['sha256']), receipt=chosen['evidence'],
            entry_key=entry['key'], member_sha256=entry['sha256'], member_payload_rechecked=True,
            page=file_ref(target), selection_reasons=reasons[ordinal], relative_quantiles=relative.tolist(),
            valid_fraction=float(valid.mean()), native_reason=chosen['native_reason']))
    checked_path(audit_ref)
    checked_path(audit['routes'])
    publish_json(output / 'report.json', dict(stage='teacher_extremes_rendered_not_reviewed',
        dataset='hypersim', audit=audit_ref, routes=audit['routes'], teacher_images=len(routes),
        full_support_images=sum(r['valid_fraction'] == 1 for r in routes),
        minimum_valid_fraction=min(r['valid_fraction'] for r in routes),
        maximum_p99_over_median=max(r['p99_over_median'] for r in routes),
        diagnostic_p99_ratio_counts={str(t): sum(r['p99_over_median'] > t for r in routes)
                                     for t in (4, 8, 16, 32, 64)},
        records=records, rendered_images=len(records), relative_color_limits=[.25, 128],
        image_resampling=False, pack_payload_hashes_rechecked=False, selected_member_payloads_rechecked=True,
        source_depth_or_mask_changed=False, exclusions_created=0, native_source_flags_cleared=False,
        source_quality_accepted=False, training_admission=False, optimizer_updates=0,
        generator=file_ref(Path(__file__))))
    print(json.dumps(dict(images=len(records), output=str(output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use an attempt basename')
    run(json.loads(args.config.read_text()), args.attempt)
