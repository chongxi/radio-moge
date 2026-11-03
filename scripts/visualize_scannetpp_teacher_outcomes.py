"""Render bounded DSLR teacher outcome extremes with the original masks.

Both per-image and common depth/median palettes are shown. Display resizing
does not edit supervision; original anonymity masks and selected payloads are
checked again. This is evidence rendering, not a source-quality pass.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import binary_erosion

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, read_geometry_member


def depth_display(depth, valid, low, high):
    values = np.zeros_like(depth)
    values[valid] = np.clip((np.log(depth[valid]) - math.log(low)) / max(math.log(high / low), 1e-8), 0, 1)
    colors = np.stack((values, 1 - np.abs(2 * values - 1), 1 - values), -1)
    colors[~valid] = 0
    return Image.fromarray((colors * 255).astype(np.uint8))


def render_entry(pack, entry, chosen, ordinal):
    depth, valid, _, metadata = read_geometry_member(pack, entry)
    row = entry['image']
    if (metadata.get('native_source_review_unresolved') is not True
            or metadata['label_kind'] != 'teacher_pseudo' or metadata['metric_calibrated'] is not False
            or metadata['camera_kind'] != 'teacher_estimated' or row['dataset'] != 'scannetpp_dslr'
            or row['split'] != 'train' or row.get('image_rotation_cw', 0)):
        raise ValueError('Only unchanged nonmetric teacher candidates with retained native flags are displayed')
    rgb_bytes, mask_bytes = Path(row['image']).read_bytes(), Path(row['mask']).read_bytes()
    if (hashlib.sha256(rgb_bytes).hexdigest() != row['sha256']
            or hashlib.sha256(mask_bytes).hexdigest() != metadata['source_quality']['anonymous_mask_sha256']):
        raise ValueError('Original RGB or anonymity mask changed')
    with Image.open(io.BytesIO(rgb_bytes)) as source:
        if source.getexif().get(274, 1) != 1:
            raise ValueError('Undocumented RGB orientation')
        rgb = source.convert('RGB')
    with Image.open(io.BytesIO(mask_bytes)) as source:
        anonymity = np.asarray(source)
    if (depth.shape != (rgb.height, rgb.width) or anonymity.shape != depth.shape
            or anonymity.dtype != np.uint8
            or np.any(valid & ~binary_erosion(anonymity == 255, structure=np.ones((5, 5), bool), border_value=0))):
        raise ValueError('Native support exceeds its exact eroded anonymity mask or original RGB grid')
    quantiles = np.quantile(depth[valid], [0, .01, .5, .99, 1])
    relative = quantiles / quantiles[2]
    for actual, expected in ((valid.mean(), chosen['valid_fraction']),
            (relative[1], chosen['p01_over_median']), (relative[3], chosen['p99_over_median']),
            (relative[4], chosen['max_over_median'])):
        if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError('Selected payload statistics differ from the full receipt audit')
    panels = [rgb, depth_display(depth, valid, quantiles[1], quantiles[3]),
              depth_display(depth / quantiles[2], valid, .25, 128),
              Image.fromarray((valid * 255).astype(np.uint8)).convert('RGB')]
    tile = Image.new('RGB', (1440, 332), 'white')
    draw = ImageDraw.Draw(tile)
    draw.text((5, 5), f"#{ordinal:03d} corpus {chosen['corpus_ordinal']} | {row['scene']} | {row['view']} "
              f"| {', '.join(chosen['selection_reasons'])}", fill='black')
    draw.text((5, 23), f"Support {valid.mean():.5f}; depth/median p01={relative[1]:.3f}, p99={relative[3]:.3f}, "
              f"max={relative[4]:.3f}; native quality flag retained; NOT metric GT", fill='black')
    titles = ('Original RGB', 'Per-image log Z: p01-p99', 'Common log Z/median: 0.25-128', 'Unchanged valid support')
    for col, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        draw.text((col * 360 + 5, 45), title, fill='black')
        resized = ImageOps.contain(panel, (352, 258),
            Image.Resampling.LANCZOS if col == 0 else Image.Resampling.NEAREST)
        tile.paste(resized, (col * 360 + (360 - resized.width) // 2, 66))
    identity = dict(corpus_ordinal=chosen['corpus_ordinal'], group=row['group'], scene=row['scene'], view=row['view'],
        image=dict(path=row['image'], sha256=row['sha256']), mask=dict(path=row['mask'],
        sha256=metadata['source_quality']['anonymous_mask_sha256']), valid_fraction=float(valid.mean()),
        grid=list(depth.shape), relative_depth_quantiles=relative.tolist(), cache_key=entry['key'],
        policy_sha256=metadata['policy_sha256'], member_sha256=entry['sha256'],
        native_source_review_unresolved=True, selected_member_payload_rechecked=True,
        original_anonymity_mask_checked=True, support_within_eroded_anonymity_mask=True,
        selection_reasons=chosen['selection_reasons'])
    return tile, identity


def render(audit_path, output):
    audit_ref = file_ref(audit_path)
    audit = checked_json(audit_ref)
    selected = checked_json(audit['review_candidates'])
    if (audit['stage'] != 'teacher_outcomes_accounted_not_accepted' or audit['dataset'] != 'scannetpp_dslr'
            or audit['producer_array_readback_complete'] is not True
            or audit['native_source_flags_cleared'] is not False
            or not 1 <= len(selected) == audit['review_candidates_count'] <= 128
            or len({r['corpus_ordinal'] for r in selected}) != len(selected)
            or len({r['group'] for r in selected}) != audit['scenes']):
        raise ValueError('Complete bounded all-scene teacher outcome selection required')
    output.mkdir(exist_ok=False)
    pages, records = [], []
    receipts = {}
    for start in range(0, len(selected), 3):
        tiles, identities = [], []
        for ordinal in range(start, min(start + 3, len(selected))):
            chosen = selected[ordinal]
            proof = chosen['receipt']
            if proof['path'] not in receipts:
                receipts[proof['path']] = checked_json(proof)
            receipt = receipts[proof['path']]
            entries = [e for e in receipt['entries'] if e['key'] == chosen['key']]
            if (len(entries) != 1 or receipt['policy_sha256'] != chosen['policy_sha256']
                    or receipt['policy_sha256'] not in audit['candidate_policy_counts']):
                raise ValueError('Selected teacher member does not belong to an audited policy')
            entry = entries[0]
            if (any(entry['image'][k] != chosen[k] for k in ('image','sha256','group','corpus_ordinal'))
                    or entry['geometry_metadata']['policy_sha256'] != receipt['policy_sha256']):
                raise ValueError('Selected RGB differs from the bound outcome')
            pack = Path(proof['path']).parent / receipt['tar']
            if pack.parent != Path(proof['path']).parent or pack.stat().st_size != receipt['bytes']:
                raise ValueError('Selected pack size or location changed')
            tile, identity = render_entry(pack, entry, chosen, ordinal)
            identity.update(receipt=proof, pack=dict(path=str(pack), producer_sha256=receipt['sha256'], rehashed=False))
            tiles.append(tile)
            identities.append(identity)
        page = Image.new('RGB', (1440, sum(tile.height for tile in tiles)), '#cccccc')
        offset = 0
        for tile, identity in zip(tiles, identities, strict=True):
            page.paste(tile, (0, offset))
            identity.update(page=f'{len(pages):03d}.jpg', page_y=offset, page_height=tile.height)
            offset += tile.height
        path = output / f'{len(pages):03d}.jpg'
        temporary = path.with_suffix('.jpg.incoming')
        with temporary.open('xb') as file:
            page.save(file, format='JPEG', quality=95)
            file.flush()
            os.fsync(file.fileno())
        temporary.rename(path)
        pages.append(file_ref(path))
        records.extend(identities)
    checked_path(audit_ref)
    publish_json(output / 'index.json', dict(stage='visual_evidence_rendered_not_reviewed',audit=audit_ref,
        selection=audit['review_candidates'],rendered_images=len(records),scenes=len({r['scene'] for r in records}),
        pages=pages,records=records,all_selected_views_included=True,selected_payloads_and_rgb_rechecked=True,
        original_anonymity_masks_rechecked=True,common_relative_depth_range=[.25,128],display_resizing_only=True,
        full_pack_payloads_rehashed=False,native_flags_cleared=False,source_quality_accepted=False,
        model_inference_performed=False,training_admission=False,optimizer_updates=0,generator=file_ref(Path(__file__))))
    print(json.dumps(dict(rendered_images=len(records),pages=len(pages),output=str(output))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    render(args.audit.resolve(),args.output.resolve())
