"""Bounded native DSLR evidence after completed outcome accounting.

Read selected cached arrays and original RGB/masks only. Depth/support stay on
their native 640-wide grid; RGB alone is sampled for display. No mesh rerender,
teacher inference, physical-calibration claim or automatic source acceptance.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import binary_erosion, map_coordinates

from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import publish_json, read_geometry_member
from radio_moge.geometry_sources import normalized_opencv_intrinsics, pinhole_rays
from visualize_scannetpp_teacher_outcomes import depth_display


def anonymity_support(mask, original_shape, shape):
    if (mask.shape != tuple(original_shape) or mask.ndim != 2 or mask.dtype != np.uint8
            or min(mask.shape) < min(shape)):
        raise ValueError('Original anonymity mask grid/type differs')
    # Same float area-support and default cross-shaped erosion as the producer.
    averaged = np.asarray(Image.fromarray((mask == 255).astype(np.float32)).resize(
        (shape[1], shape[0]), Image.Resampling.BOX))
    return binary_erosion(averaged >= 1-1e-7, iterations=2)


def aligned_rgb(rgb, kd, kr, shape):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Original RGB type differs')
    rays = pinhole_rays(kd, *shape)
    uvz = rays @ np.asarray(kr).T
    uv = uvz[..., :2]/uvz[..., 2:]
    if (not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv > 1)):
        raise ValueError('Native rays project outside the original RGB')
    xy = uv*np.array([rgb.shape[1], rgb.shape[0]])-.5
    sampled = np.stack([map_coordinates(rgb[..., c].astype(np.float64),
        [xy[..., 1], xy[..., 0]], order=1, mode='nearest') for c in range(3)], -1)
    return sampled.clip(0, 255).astype(np.uint8), uv


def checked_display(entry, cached, rgb, mask, chosen):
    depth, valid, kd, metadata = cached
    row = entry['image']
    frozen_row = dict(row)
    ordinal = frozen_row.pop('corpus_ordinal', chosen['corpus_ordinal'])
    expected_k = normalized_opencv_intrinsics(*row['camera_width_height_fx_fy_cx_cy'])
    shape = (round(row['height']*640/row['width']), 640)
    if (metadata != entry['geometry_metadata'] or frozen_row != chosen['image']
            or ordinal != chosen['corpus_ordinal']
            or metadata['policy_sha256'] != chosen['policy_sha256']
            or metadata['dataset'] != 'scannetpp_dslr' or row['split'] != 'train'
            or metadata['label_kind'] != 'laser_mesh_derived' or metadata['camera_kind'] != 'calibrated_sensor'
            or metadata['metric_calibrated'] is not True or metadata['source_needs_review'] is not False
            or metadata['not_direct_per_dslr_pixel_measurements'] is not True
            or metadata['depth_convention'] != 'camera_z' or metadata['units'] != 'meters'
            or metadata['camera_pixel_convention'] != 'normalized_half_center'
            or depth.shape != shape or valid.shape != shape or valid.dtype != np.bool_ or not valid.any()
            or list(depth.shape) != metadata['shape'] or row.get('image_rotation_cw', 0)
            or rgb.shape != (row['height'], row['width'], 3)
            or metadata['rgb_shape'] != [row['height'], row['width']]
            or not np.isfinite(kd).all() or not np.allclose(kd, expected_k, rtol=0, atol=1e-12)
            or not np.allclose(metadata['rgb_intrinsics_normalized'], expected_k, rtol=0, atol=1e-12)
            or not np.array_equal(metadata['depth_to_rgb_uv_transform'], np.eye(3))):
        raise ValueError('Selected native camera, source meaning or RGB/depth grid differs')
    allowed = anonymity_support(mask, rgb.shape[:2], shape)
    if np.any(valid & ~allowed):
        raise ValueError('Cached support exceeds original conservative anonymity support')
    q = np.quantile(depth[valid], [0, .01, .5, .99, 1])
    if (not np.array_equal(q, entry['depth_quantiles_m'])
            or int(valid.sum()) != metadata['valid_pixels']
            or int(valid.sum()) != metadata['source_quality']['verified_pixels']
            or not np.isclose(valid.mean(), chosen['metrics']['valid_fraction'], rtol=0, atol=1e-12)
            or not np.isclose(q[2], chosen['metrics']['median_depth_m'], rtol=0, atol=1e-12)
            or not np.isclose(q[4], chosen['metrics']['maximum_depth_m'], rtol=0, atol=1e-12)):
        raise ValueError('Selected payload statistics differ from audited receipt')
    sampled, uv = aligned_rgb(rgb, kd, expected_k, shape)
    x, y = np.meshgrid((np.arange(shape[1])+.5)/shape[1], (np.arange(shape[0])+.5)/shape[0])
    if not np.allclose(uv, np.stack((x, y), -1), rtol=0, atol=1e-12):
        raise ValueError('Native/RGB normalized UV identity differs')
    return sampled, allowed, q


def render_entry(pack, entry, chosen):
    cached = read_geometry_member(pack, entry)
    row, meta = entry['image'], entry['geometry_metadata']
    rgb_bytes, mask_bytes = Path(row['image']).read_bytes(), Path(row['mask']).read_bytes()
    mask_proof = [p for p in meta['source_proofs'] if p['source'] == row['mask']]
    if (hashlib.sha256(rgb_bytes).hexdigest() != row['sha256'] or len(mask_proof) != 1
            or hashlib.sha256(mask_bytes).hexdigest() != mask_proof[0]['sha256']):
        raise ValueError('Original RGB or native mask byte identity differs')
    with Image.open(io.BytesIO(rgb_bytes)) as source:
        if source.getexif().get(274, 1) != 1:
            raise ValueError('Unexpected original RGB orientation')
        rgb = np.asarray(source.convert('RGB'))
    with Image.open(io.BytesIO(mask_bytes)) as source:
        mask = np.asarray(source)
    sampled, allowed, q = checked_display(entry, cached, rgb, mask, chosen)
    depth, valid, _, _ = cached
    fixed = depth_display(depth, valid, .025, 64)
    overlay = sampled.copy()
    overlay[valid] = (.55*sampled[valid]+.45*np.asarray(fixed)[valid]).astype(np.uint8)
    panels = [(Image.fromarray(rgb), 'Original undistorted DSLR RGB'),
        (fixed, 'Camera Z: fixed log 0.025-64 meters'),
        (depth_display(depth, valid, q[1], q[3]), 'Camera Z: per-image log p01-p99'),
        (Image.fromarray(overlay), 'RGB sampled through K + unchanged depth'),
        (Image.fromarray((allowed*255).astype(np.uint8)), 'Original mask: float BOX + cross erosion'),
        (Image.fromarray((valid*255).astype(np.uint8)), 'Unchanged final valid support')]
    page = Image.new('RGB', (1440, 1000), 'white')
    draw = ImageDraw.Draw(page)
    lines = [f"#{chosen['corpus_ordinal']} {row['group']} | {row['view']}",
        f"Support {valid.mean():.5%}; p01/median/p99 = {q[1]:.5f}/{q[2]:.5f}/{q[3]:.5f} m",
        'Laser-mesh-derived perspective camera Z; not direct per-pixel measurements or independent calibration.',
        'Display-only RGB sampling/resizing. Depth and support unchanged. '+', '.join(chosen['selection_reasons'])]
    for i, line in enumerate(lines):
        draw.text((6, 5+20*i), line, fill='black')
    for i, (panel, label) in enumerate(panels):
        x, y = (i % 3)*480, 90+(i//3)*450
        draw.text((x+5, y), label, fill='black')
        thumb = ImageOps.contain(panel, (470, 410), Image.Resampling.LANCZOS if i == 0 else Image.Resampling.NEAREST)
        page.paste(thumb, (x+(480-thumb.width)//2, y+25))
    identity = dict(corpus_ordinal=chosen['corpus_ordinal'], image=dict(path=row['image'], sha256=row['sha256']),
        mask=dict(path=row['mask'], sha256=mask_proof[0]['sha256']), group=row['group'], view=row['view'],
        entry_key=entry['key'], member_sha256=entry['sha256'], policy_sha256=chosen['policy_sha256'],
        grid=list(depth.shape), rgb_grid=list(rgb.shape[:2]), depth_quantiles_m=q.tolist(),
        valid_fraction=float(valid.mean()), selection_reasons=chosen['selection_reasons'])
    return page, identity


def render(audit_path, output):
    audit_ref = file_ref(audit_path)
    audit = checked_json(audit_ref)
    selected = checked_json(audit['review_candidates'])
    if (audit['stage'] != 'native_outcomes_accounted_not_accepted' or audit['dataset'] != 'scannetpp_dslr'
            or audit['producer_array_readback_complete'] is not True or audit['source_flags_preserved'] is not True
            or audit['training_admission'] is not False or audit['source_quality_accepted'] is not False
            or not 1 <= len(selected) == audit['review_candidate_count'] <= 30
            or len({r['corpus_ordinal'] for r in selected}) != len(selected)):
        raise ValueError('Completed native outcome audit and bounded unique selection required')
    index = checked_path(audit['index'])
    output.mkdir(exist_ok=False)
    records = []
    with readonly_db(index) as db:
        for chosen in selected:
            receipt = checked_json(chosen['receipt'])
            found = [e for e in receipt['entries'] if e['key'] == chosen['entry_key']]
            row = db.execute('SELECT row_json,outcome,policy,entry_key,evidence_json,details_json FROM images '
                'WHERE ordinal=?', (chosen['corpus_ordinal'],)).fetchone()
            if (len(found) != 1 or row is None or row[1] != 'native_candidate'
                    or json.loads(row[0]) != chosen['image'] or row[2] != chosen['policy_sha256']
                    or row[3] != chosen['entry_key'] or json.loads(row[4]) != chosen['receipt']
                    or json.loads(row[5]) != chosen['metrics'] or receipt['policy_sha256'] != row[2]
                    or row[2] not in audit['candidate_policy_counts']):
                raise ValueError('Selected member differs from bound native outcome')
            pack = Path(chosen['receipt']['path']).parent/receipt['tar']
            if pack.parent != Path(chosen['receipt']['path']).parent or pack.stat().st_size != receipt['bytes']:
                raise ValueError('Selected pack path or size differs')
            page, identity = render_entry(pack, found[0], chosen)
            path = output/f'{len(records):02d}-{chosen["corpus_ordinal"]}.png'
            incoming = path.with_suffix('.png.incoming')
            with incoming.open('xb') as stream:
                page.save(stream, format='PNG')
            incoming.rename(path)
            records.append(dict(**identity, receipt=chosen['receipt'], page=file_ref(path)))
            print(json.dumps(dict(rendered=len(records), corpus_ordinal=chosen['corpus_ordinal'])), flush=True)
    checked_path(audit_ref)
    checked_path(audit['review_candidates'])
    checked_path(audit['index'])
    report = dict(stage='native_visual_evidence_rendered_not_reviewed', audit=audit_ref,
        selection=audit['review_candidates'], records=records, rendered_images=len(records),
        selected_payloads_rechecked=True, pack_payload_hashes_rechecked=False, original_masks_rechecked=True,
        normalized_camera_and_support_checked=True, fixed_depth_limits_m=[.025, 64], display_only_resizing=True,
        depth_or_mask_edited=False, source_flags_cleared=False, mesh_rerendered=False,
        independent_physical_accuracy_checked=False, source_quality_accepted=False, training_admission=False,
        optimizer_updates=0, gpu_used=False, renderer=file_ref(Path(__file__)),
        palette_helper=file_ref(Path(__file__).with_name('visualize_scannetpp_teacher_outcomes.py')),
        payload_reader=file_ref(Path(__file__).parents[1]/'src/radio_moge/geometry_cache.py'))
    publish_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    render(args.audit.resolve(), args.output.resolve())
