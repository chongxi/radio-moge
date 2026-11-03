"""Re-render explicitly flagged IRS views with one shared relative-depth palette.

The earlier sheets scale colors by each image's p01/p99. Those colors cannot
establish disagreement between views. This display uses the same fixed range
for every depth/median image, without fitting cameras, geometry, or metric scale.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, read_geometry_member, sha_file


def run(review_path, output):
    review_ref = file_ref(review_path)
    review = checked_json(review_ref)
    index_ref = review['evidence'][0]
    index = checked_json(index_ref)
    if (review['stage'] != 'selected_extremes_visual_review_complete' or review['dataset'] != 'irs'
            or not 1 <= len(review['specific_review_flags']) <= 32
            or index['stage'] != 'visual_evidence_rendered_not_reviewed'
            or index['selected_member_payloads_rechecked'] is not True):
        raise ValueError('Bounded explicit IRS quality flags and original display index required')
    by_id = {v['corpus_ordinal']: v for v in index['records']}
    output.mkdir(exist_ok=False)
    records, seen = [], set()
    for flag in review['specific_review_flags']:
        ordinal = flag['corpus_ordinal']
        if ordinal in seen:
            raise ValueError('Repeated flagged view')
        seen.add(ordinal)
        chosen = by_id[ordinal]
        if flag['image'] != chosen['image'] or flag['source_id'] != chosen['source_id']:
            raise ValueError('Quality flag changed original RGB identity')
        receipt = checked_json(chosen['receipt'])
        matching = [e for e in receipt['entries'] if e['key'] == chosen['key']]
        if len(matching) != 1:
            raise ValueError('Exact cached member identity required')
        entry = matching[0]
        row = entry['image']
        pack = Path(chosen['receipt']['path']).parent / receipt['tar']
        if (str(pack) != chosen['pack']['path'] or pack.stat().st_size != receipt['bytes']
                or entry['sha256'] != chosen['member_sha256'] or row['corpus_ordinal'] != ordinal
                or row['image'] != chosen['image']['path'] or row['sha256'] != chosen['image']['sha256']):
            raise ValueError('Flagged member differs from original reviewed payload')
        depth, valid, _, metadata = read_geometry_member(pack, entry)
        if (metadata['label_kind'] != 'teacher_pseudo' or metadata['metric_calibrated'] is not False
                or metadata['camera_kind'] != 'teacher_estimated' or row['dataset'] != 'irs'
                or row['split'] != 'train' or row.get('image_rotation_cw', 0)
                or sha_file(row['image']) != row['sha256']):
            raise ValueError('Unchanged training RGB and nonmetric teacher required')
        with Image.open(row['image']) as source:
            if (source.mode not in ('RGB', 'RGBA') or source.getexif().get(274, 1) != 1
                    or (source.mode == 'RGBA' and source.getextrema()[3] != (255, 255))):
                raise ValueError('Unexpected RGB mode/orientation')
            # These publisher PNGs have an opaque alpha channel; match the
            # producer's RGB conversion without compositing transparency.
            rgb = source.convert('RGB')
        if depth.shape != (rgb.height, rgb.width):
            raise ValueError('Original image and geometry grids differ')
        median = float(np.median(depth[valid]))
        relative = np.zeros(depth.shape, np.float32)
        relative[valid] = depth[valid] / median
        value = np.zeros(depth.shape)
        value[valid] = np.clip(np.log2(relative[valid] / .25) / 6, 0, 1)
        colors = (np.stack((value, 1 - np.abs(2 * value - 1), 1 - value), -1) * 255).astype(np.uint8)
        colors[~valid] = 0
        page = Image.new('RGB', (2 * rgb.width, rgb.height + 84), 'white')
        draw = ImageDraw.Draw(page)
        draw.text((6, 5), f"#{flag['display_index']} | corpus {ordinal} | {flag['source_id']}", fill='black')
        draw.text((6, 23), 'Original RGB | unchanged teacher Z / valid median; same log palette for every view', fill='black')
        draw.text((6, 41), 'Shared palette: blue <=0.25, green 2, red >=16; black=invalid. Values are NOT calibrated meters.', fill='black')
        draw.text((6, 59), 'Display saturation only; no depth/mask edits, pose fitting, or multiview accuracy claim.', fill='black')
        page.paste(rgb, (0, 84))
        page.paste(Image.fromarray(colors), (rgb.width, 84))
        target = output / f"{flag['display_index']:03d}.png"
        temporary = target.with_suffix('.png.incoming')
        page.save(temporary, format='PNG')
        os.rename(temporary, target)
        records.append(dict(corpus_ordinal=ordinal, original_display_index=flag['display_index'],
            source_id=flag['source_id'], image=chosen['image'], receipt=chosen['receipt'],
            entry_key=entry['key'], member_sha256=entry['sha256'], member_payload_rechecked=True,
            page=file_ref(target), median_nominal_uncalibrated=median,
            relative_quantiles=np.quantile(relative[valid], [0, .01, .5, .99, 1]).tolist(),
            fraction_display_above_16=float((relative[valid] > 16).mean()),
            fraction_display_below_quarter=float((relative[valid] < .25).mean()),
            valid_fraction=float(valid.mean()), original_flag=flag['reason']))
    checked_path(review_ref)
    checked_path(index_ref)
    report = dict(stage='shared_palette_evidence_rendered_not_reviewed', dataset='irs',
        original_review=review_ref, original_index=index_ref, records=records, rendered_images=len(records),
        relative_color_limits=[.25, 16], normalization='each image divided by its valid depth median',
        image_resampling=False, camera_or_pose_fitting=False, source_depth_or_mask_changed=False,
        calibrated_depth_claim=False, independent_geometry_check=False, pack_payload_hashes_rechecked=False,
        source_quality_accepted=False, training_admission=False, optimizer_updates=0,
        generator=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(stage=report['stage'], images=len(records), output=str(output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.review.resolve(), args.output.resolve())
