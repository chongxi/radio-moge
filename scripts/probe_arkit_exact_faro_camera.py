"""Replace FARO camera hypotheses with exact publisher wide-camera intrinsics.

Download only the nine already reviewed training frames' wide RGB and pincam
members. Retain the original AppleDepth, confidence, mask and FARO bytes. The
publisher's wide assets exist even though its default downloader omits them.
This is a bounded diagnostic, not source admission or a physical error bound.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import map_coordinates

from cache_arkit_geometry import NativeBundle, convert, load_image
from radio_moge.arkit_depth import rotate_camera
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json
from radio_moge.geometry_sources import normalized_opencv_intrinsics, pinhole_rays
from visualize_arkit_faro_probe import RetainedBundle, depth_colors, display_fields, error_colors


def exact_camera(values, shape):
    camera = np.asarray(values, dtype=np.float64)
    if (camera.shape != (6,) or not np.isfinite(camera).all()
            or tuple(camera[:2]) != (shape[1], shape[0])
            or min(camera[2:4]) <= 0 or not 0 <= camera[4] < camera[0]
            or not 0 <= camera[5] < camera[1]):
        raise ValueError('Exact wide-camera intrinsics do not describe the released FARO grid')
    return camera.tolist()


def rgb_correspondence(depth_camera, shape, wide_camera, wide, low, valid):
    if low.shape != (*shape, 3) or valid.shape != shape or wide.shape[-1] != 3:
        raise ValueError('Source RGB/depth grids differ')
    kw = normalized_opencv_intrinsics(*wide_camera)
    uv = pinhole_rays(depth_camera, *shape) @ kw.T
    uv = uv[..., :2] / uv[..., 2:]
    inside = ((uv[..., 0] >= 0) & (uv[..., 0] <= 1)
              & (uv[..., 1] >= 0) & (uv[..., 1] <= 1))
    xy = uv * [wide.shape[1], wide.shape[0]] - .5
    sampled = np.stack([map_coordinates(wide[..., c].astype(float) / 255,
        [xy[..., 1], xy[..., 0]], order=1, mode='nearest') for c in range(3)], -1)
    error = np.abs(sampled - low.astype(float) / 255).mean(-1)
    mask = valid & inside
    if not mask.any():
        raise ValueError('Exact camera has no observed native overlap')
    return error, mask, dict(compared_native_pixels=int(mask.sum()),
        mean_absolute_rgb_difference=float(error[mask].mean()),
        rgb_difference_quantiles=np.quantile(error[mask], [.5, .9, .95, .99, 1]).tolist(),
        fraction_rgb_difference_le_008=float((error[mask] <= .08).mean()))


def run(root, output):
    probe_ref = file_ref(root / 'arkit-faro-quality-probe-01/report.json')
    review_ref = file_ref(root / 'arkit-faro-visuals-01/visual-review-01.json')
    probe, review = checked_json(probe_ref), checked_json(review_ref)
    source_ref = file_ref(root / 'inventory.json')
    source = checked_json(source_ref)
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    checked_path(inventory_ref)
    if (probe['stage'] != 'diagnostic_complete' or len(probe['records']) != 9
            or review['images_reviewed'] != 9 or probe['manifest_sha256'] != source['manifest_sha256']
            or probe['inventory_sha256'] != inventory_ref['sha256']):
        raise ValueError('Existing reviewed exact nine training frames are required')
    ordinals = {r['image']['sha256']: r['corpus_ordinal'] for r in review['records']}
    output.mkdir(exist_ok=False)
    assets = NativeBundle(output / 'publisher-source')
    results = []
    try:
        with readonly_db(inventory_ref['path']) as db:
            for number, original in enumerate(probe['records']):
                ordinal = ordinals[original['image_sha256']]
                found = db.execute('SELECT image,row_json FROM images WHERE ordinal=?', (ordinal,)).fetchone()
                row = dict(json.loads(found[1]), image=found[0])
                if (row['split'] != 'train' or row['dataset'] != 'arkitscenes'
                        or row['sha256'] != original['image_sha256']
                        or row['timestamp'] != original['timestamp'] or row['video_id'] != original['video']):
                    raise ValueError('Original training image identity changed')
                bundle = RetainedBundle(root / 'arkit-faro-quality-probe-01/native-source', original['source_proofs'])
                depth, valid, kd, _, native_proofs, quality = convert(row, bundle)
                base, stem = row['source_url'].rsplit('/', 1)[0], PurePosixPath(row['source_member']).stem
                faro_path, faro_proof = bundle.get(base, 'highres_depth', stem)
                low_path, _ = bundle.get(base, 'lowres_wide', stem)
                if ([*native_proofs, faro_proof] != original['source_proofs']
                        or canonical_json(quality) != canonical_json(original['native_quality'])):
                    raise ValueError('Unchanged native source, support and camera must reproduce exactly')
                camera_path, camera_proof = assets.get(base, 'wide_intrinsics', stem)
                wide_path, wide_proof = assets.get(base, 'wide', stem)
                faro, wide = load_image(faro_path), load_image(wide_path)
                camera = exact_camera(np.loadtxt(camera_path), faro.shape)
                if wide.shape != (*faro.shape, 3):
                    raise ValueError('Exact wide RGB and FARO grids differ')
                turn = row['image_rotation_cw']
                wide_upright = np.rot90(wide, -turn // 90).copy()
                low_upright = np.rot90(load_image(low_path), -turn // 90).copy()
                z, supported, difference, compared, stats = display_fields(depth, valid, kd, faro, camera, turn, False)
                stats['mapping'] = 'exact_publisher_wide_intrinsics_no_fit'
                photo_error, photo_mask, photo = rgb_correspondence(kd, depth.shape,
                    rotate_camera(camera, turn), wide_upright, low_upright, valid)
                vga = np.asarray(row['native_camera_width_height_fx_fy_cx_cy'], dtype=float)
                scale = camera[0] / vga[0]
                integer = vga * scale
                half = integer.copy()
                half[4:] += (scale - 1) / 2
                hypotheses = dict(integer_scaled=(np.asarray(camera) - integer).tolist(),
                                  half_center_scaled=(np.asarray(camera) - half).tolist())
                page = Image.new('RGB', (1200, 1240), 'white')
                draw = ImageDraw.Draw(page)
                lines = [f"Frozen ordinal {ordinal}, {stem}, CW {turn}: EXACT publisher wide camera",
                    'Original native depth/mask unchanged; exact timestamp, no intrinsic/depth fit.',
                    'Depth log 0.25-8m; signed native-FARO error blue -10cm / white zero / red +10cm.',
                    f"{stats['compared_pixels']} pixels; median signed {stats['median_signed_difference_m']:.4f}m; "
                    f"absolute p95 {stats['abs_error_m_quantiles'][2]:.4f}m; within5cm {stats['fraction_within_5cm']:.2%}",
                    f"Exact wide vs native RGB: MAE {photo['mean_absolute_rgb_difference']:.5f}; "
                    f"p90 {photo['rgb_difference_quantiles'][1]:.5f}; no extra supervision mask applied.",
                    'Publisher-filtered subset; residuals may include sensor, mesh, visibility and timing effects.']
                for i, line in enumerate(lines):
                    draw.text((8, 8 + 19 * i), line, fill='black')
                photo_colors = np.stack((np.minimum(photo_error / .16, 1),) * 3, -1)
                photo_colors[~photo_mask] = 0
                panels = [(np.rot90(load_image(row['image']), -turn // 90), 'Original frozen VGA RGB'),
                    (wide_upright, 'Exact timestamp publisher wide RGB'),
                    (depth_colors(depth, valid), 'Unchanged native AppleDepth / mask'),
                    (depth_colors(z, supported), 'FARO mesh depth with native support'),
                    (error_colors(difference, compared), 'Signed error using EXACT released camera'),
                    ((255 * photo_colors).astype(np.uint8), 'Wide/native RGB error: black0 / white>=0.16')]
                for i, (array, title) in enumerate(panels):
                    x, y = i % 3 * 400, 136 + i // 3 * 550
                    draw.text((x + 5, y), title, fill='black')
                    thumb = ImageOps.contain(Image.fromarray(array), (392, 520),
                        Image.Resampling.LANCZOS if i in (0, 1) else Image.Resampling.NEAREST)
                    page.paste(thumb, (x + (400 - thumb.width) // 2, y + 22))
                target = output / f'{number:02d}-{ordinal}.png'
                incoming = target.with_suffix('.png.incoming')
                with incoming.open('xb') as file:
                    page.save(file, format='PNG')
                incoming.rename(target)
                results.append(dict(corpus_ordinal=ordinal, image=file_ref(row['image']), video=row['video_id'],
                    timestamp=row['timestamp'], rotation_cw=turn, exact_camera=camera,
                    exact_minus_hypothesis_camera=hypotheses, exact_comparison=stats,
                    prior_hypothesis_comparisons=original['comparisons'], exact_wide_rgb_comparison=photo,
                    publisher_members=[camera_proof, wide_proof],
                    publisher_receipts=[file_ref(p.with_name(p.name + '.receipt.json')) for p in (camera_path, wide_path)],
                    retained_source_receipts=bundle.receipts, original_native_quality=quality,
                    page=file_ref(target), original_native_arrays_and_masks_unchanged=True))
                print(json.dumps(dict(processed=len(results), ordinal=ordinal, camera_difference=hypotheses,
                    signed_median_m=stats['median_signed_difference_m'], rgb_mae=photo['mean_absolute_rgb_difference'])), flush=True)
    finally:
        for archive in assets.archives.values():
            archive.zip.close()
            archive.file.close()
        assets.session.close()
    report = dict(stage='exact_publisher_camera_diagnostic_complete_not_reviewed', completed_at=time.time(),
        images=len(results), videos=len({r['video'] for r in results}), records=results,
        original_probe=probe_ref, previous_visual_review=review_ref, source_report=source_ref,
        inventory=inventory_ref, source_flags_cleared=False, training_admission=False,
        optimizer_updates=0, gpu_used=False, source_quality_accepted=False,
        calibration_fit_performed=False, original_native_arrays_and_masks_unchanged=True,
        exact_highres_intrinsics_now_obtained=True, source_accuracy_guarantee_claimed=False,
        correction='The earlier claim that exact high-resolution K was not released was incorrect: wide_intrinsics.zip exists on the publisher server, although the default downloader omits it.',
        limitations=['These nine frames still cover only three videos and rotations0/90.',
            'The publisher selected the high-resolution subset using depth disagreement; no population accuracy claim.',
            'Known exact intrinsics do not resolve sensor/mesh registration, visibility, reflections or timing uncertainty.'],
        documentation_urls=['https://github.com/apple/ARKitScenes/blob/main/raw/README.md',
                            'https://github.com/apple/ARKitScenes/blob/main/download_data.py'],
        generator=file_ref(Path(__file__)), display_helper=file_ref(Path(__file__).with_name('visualize_arkit_faro_probe.py')))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(report=file_ref(output / 'report.json'), images=len(results))), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
