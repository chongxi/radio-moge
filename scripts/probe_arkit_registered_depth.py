"""Train-only ARKit exact-time, native-grid RGB/depth/camera registration probe."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from indoor_download import write_json
from probe_native_depth import Evidence, unpack_image
from radio_moge.arkit_depth import native_depth, registration_uv, rotate_camera
from radio_moge.geometry_cache import sha_file


def probe(config, attempt):
    root = Path(config['root'])
    inventory = json.loads((root / 'inventory.json').read_text())
    if (inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']):
        raise ValueError('Frozen inventory changed')
    evidence = Evidence(root / attempt)
    result = dict(stage='probing', training_admission=False, optimizer_updates=0, records=[])
    selected, seen = [], set()
    with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
        db.execute('PRAGMA cache_size=-32768')
        for turn in (0, 90, 180, 270):
            candidates = db.execute("SELECT row_json,image FROM images WHERE dataset='arkitscenes' "
                "AND json_extract(row_json,'$.image_rotation_cw')=? ORDER BY ordinal", (turn,))
            count = 0
            for encoded, path in candidates:
                row = json.loads(encoded)
                if row['video_id'] in seen:
                    continue
                seen.add(row['video_id'])
                row['image'] = path
                selected.append(row)
                count += 1
                if count == 3:
                    break
    for row in selected:
        if row['split'] != 'train' or sha_file(row['image']) != row['sha256']:
            raise ValueError('Original ARKit training image changed')
        video, turn = str(row['video_id']), row['image_rotation_cw']
        stem = PurePosixPath(row['source_member']).stem
        if stem.split('_')[0] != video or float(stem.split('_')[1]) != row['timestamp']:
            raise ValueError('Manifest timestamp does not match native RGB filename')
        base = row['source_url'].rsplit('/', 1)[0]
        record = dict(video=video, timestamp=row['timestamp'], rotation_cw=turn,
                      exact_timestamp=True, training_admission=False)
        paths = {}
        try:
            for kind in ('lowres_depth', 'confidence', 'lowres_wide', 'lowres_wide_intrinsics',
                         'vga_wide_intrinsics'):
                suffix = '.pincam' if kind.endswith('intrinsics') else '.png'
                member = f'{kind}/{stem}{suffix}'
                paths[kind] = evidence.zip_member(f'{video}/{kind}/{stem}{suffix}',
                                                   base + '/' + kind + '.zip', member)
        except KeyError:
            # Missing timestamp is a native-data exclusion, not a nearest-neighbor substitution.
            record.update(stage='missing_exact_time_asset')
            result['records'].append(record)
            write_json(evidence.root / 'report.json', result)
            continue
        low_camera, rgb_camera = (np.loadtxt(paths[kind]).tolist() for kind in
                                  ('lowres_wide_intrinsics', 'vga_wide_intrinsics'))
        if not np.allclose(rgb_camera, row['native_camera_width_height_fx_fy_cx_cy'], atol=1e-8, rtol=0):
            raise ValueError('Native VGA intrinsics differ from training manifest')
        if not np.allclose(rotate_camera(rgb_camera, turn),
                           row['camera_width_height_fx_fy_cx_cy'], atol=1e-8, rtol=0):
            raise ValueError('Upright camera does not match frozen image rotation')
        depth, valid, oriented_camera = native_depth(unpack_image(paths['lowres_depth'].read_bytes()),
            unpack_image(paths['confidence'].read_bytes()), low_camera, turn)
        uv, _, _ = registration_uv(low_camera, rgb_camera)
        with Image.open(row['image']) as file:
            if file.getexif().get(274, 1) != 1:
                raise ValueError('Unexpected ARKit native RGB EXIF orientation')
            rgb = np.asarray(file.convert('RGB')).astype(float) / 255
        low_rgb = unpack_image(paths['lowres_wide'].read_bytes()).astype(float) / 255
        if low_rgb.shape != (*uv.shape[:2], 3) or rgb.shape[:2] != (rgb_camera[1], rgb_camera[0]):
            raise ValueError('RGB stream dimensions disagree with cameras')
        xy = uv * np.array([rgb.shape[1], rgb.shape[0]]) - .5
        sampled = np.stack([map_coordinates(rgb[..., c], [xy[..., 1], xy[..., 0]],
                              order=1, mode='nearest') for c in range(3)], -1)
        error = np.abs(sampled - low_rgb).mean(-1)
        flipped_error = np.abs(sampled - low_rgb[:, ::-1]).mean(-1)
        identity_error = np.abs(sampled - np.roll(low_rgb, 16, axis=1)).mean(-1)
        record.update(stage='diagnostic_complete', native_depth_shape=list(depth.shape),
            valid_fraction=float(valid.mean()), depth_camera=oriented_camera,
            rgb_camera=rotate_camera(rgb_camera, turn),
            paired_rgb_mae=float(error.mean()), flipped_rgb_mae=float(flipped_error.mean()),
            wrong_shift_rgb_mae=float(identity_error.mean()),
            pixel_error_quantiles=np.quantile(error, [.5, .9, .99, 1]).tolist(),
            depth_quantiles_m=np.quantile(depth[valid], [0, .5, .99, 1]).tolist() if valid.any() else None,
            source_kind='measured_sensor_AppleDepth_LiDAR_not_FARO_mesh',
            note='Photometric diagnostic plus source calibration; not independent physical depth accuracy')
        result['records'].append(record)
        # Review exact sampling alignment at native resolution; no depth upsampling.
        h, w = low_rgb.shape[:2]
        panel = Image.new('RGB', (w * 3, h))
        panels = [low_rgb, sampled, np.repeat(error[..., None], 3, axis=-1)]
        for i, item in enumerate(panels):
            panel.paste(Image.fromarray(np.clip(item * 255, 0, 255).astype(np.uint8)), (i * w, 0))
        buffer = io.BytesIO()
        panel.save(buffer, format='PNG')
        evidence.save(f'visual-qc/{video}.png', buffer.getvalue(), 'computed-calibrated-RGB-grid-diagnostic')
        # Bound live remote ZIP indices to a single video's streams.
        for archive in evidence.archives.values():
            archive.zip.close()
            archive.file.close()
        evidence.archives.clear()
        write_json(evidence.root / 'report.json', result)
    result.update(stage='complete', expected_examples=len(selected),
                  manifest_sha256=config['manifest_sha256'],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(evidence.root / 'report.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a new attempt basename')
    probe(json.loads(args.config.read_text()), args.attempt)
