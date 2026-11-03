"""Render an existing train-only FARO diagnostic from retained source bytes.

No network, new sample selection, calibration fit or source admission. The two
pixel-origin hypotheses remain separate. Display arrays reproduce the original
statistics before any resizing, and all source masks stay unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import maximum_filter, minimum_filter

from cache_arkit_geometry import convert, load_image
from probe_arkit_faro_quality import compare
from radio_moge.arkit_depth import rotate_camera
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json, sha_file
from radio_moge.geometry_sources import normalized_opencv_intrinsics, pinhole_rays


class RetainedBundle:
    def __init__(self, root, proofs):
        self.root, self.proofs, self.receipts = Path(root), proofs, []

    def get(self, base, kind, stem):
        suffix = '.pincam' if kind.endswith('intrinsics') else '.png'
        member, url = f'{kind}/{stem}{suffix}', base + '/' + kind + '.zip'
        matches = [p for p in self.proofs if p['source'] == url and p['member'] == member]
        if (not url.startswith('https://docs-assets.developer.apple.com/') or len(matches) != 1
                or Path(member).is_absolute() or '..' in Path(member).parts):
            raise ValueError('Exact unique publisher member proof required')
        proof = matches[0]
        path = self.root / hashlib.sha256(url.encode()).hexdigest()[:16] / member
        receipt = path.with_name(path.name + '.receipt.json')
        data = path.read_bytes()  # Missing retained bytes must fail, never download.
        if (json.loads(receipt.read_text()) != proof or len(data) != proof['bytes']
                or hashlib.sha256(data).hexdigest() != proof['sha256']
                or zlib.crc32(data) != proof['crc32']):
            raise ValueError('Retained source bytes or receipt changed')
        self.receipts.append(file_ref(receipt))
        return path, proof


def display_fields(depth, valid, kd, faro, camera, turn, half_center):
    """Reproduce original nearest-pixel diagnostic for display, not supervision."""
    w, h, fx, fy, cx, cy = camera
    sx, sy = faro.shape[1] / w, faro.shape[0] / h
    if faro.dtype != np.uint16 or faro.shape != (1440, 1920) or sx != sy:
        raise ValueError('Unexpected FARO grid')
    shift = (sx - 1) / 2 if half_center else 0
    kf = normalized_opencv_intrinsics(*rotate_camera(
        [1920, 1440, fx * sx, fy * sy, cx * sx + shift, cy * sy + shift], turn))
    oriented = np.rot90(faro, -turn // 90).copy()
    supported = (oriented > 0) & (oriented < 65535)
    z = oriented.astype(np.float64) / 1000
    low = minimum_filter(np.where(supported, z, -np.inf), 5, mode='constant', cval=-np.inf)
    high = maximum_filter(np.where(supported, z, np.inf), 5, mode='constant', cval=np.inf)
    stable = supported & np.isfinite(low) & np.isfinite(high) & (high - low <= .03 + .01 * z)
    uv = pinhole_rays(kd, *depth.shape) @ kf.T
    ij = np.rint(uv[..., :2] / uv[..., 2:] * [oriented.shape[1], oriented.shape[0]] - .5).astype(int)
    inside = ((ij[..., 0] >= 2) & (ij[..., 0] < oriented.shape[1] - 2)
              & (ij[..., 1] >= 2) & (ij[..., 1] < oriented.shape[0] - 2))
    yy, xx = ij[..., 1].clip(0, oriented.shape[0] - 1), ij[..., 0].clip(0, oriented.shape[1] - 1)
    mask = valid & inside & stable[yy, xx]
    difference = depth - z[yy, xx]
    stats = compare(depth, valid, kd, faro, camera, turn, half_center)
    if (int(mask.sum()) != stats['compared_pixels']
            or float(np.median(difference[mask])) != stats['median_signed_difference_m']
            or np.quantile(np.abs(difference[mask]), [.5, .9, .95, .99, 1]).tolist()
            != stats['abs_error_m_quantiles']):
        raise ValueError('Display correspondence differs from original diagnostic')
    return z, supported, difference, mask, stats


def depth_colors(depth, valid):
    value = np.zeros(depth.shape)
    value[valid] = np.clip(np.log(depth[valid] / .25) / np.log(32), 0, 1)
    colors = (255 * np.stack((value, 1 - np.abs(2 * value - 1), 1 - value), -1)).astype(np.uint8)
    colors[~valid] = 0
    return colors


def error_colors(difference, mask):
    value = np.clip(difference / .1, -1, 1)
    colors = np.stack((1 + np.minimum(value, 0), 1 - np.abs(value),
                       1 - np.maximum(value, 0)), -1)
    colors[~mask] = 0
    return (255 * colors).astype(np.uint8)


def render(config, probe_name, attempt):
    root = Path(config['root'])
    probe = root / probe_name
    probe_ref = file_ref(probe / 'report.json')
    report = checked_json(probe_ref)
    source_ref = file_ref(root / 'inventory.json')
    source = checked_json(source_ref)
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=report['inventory_sha256'])
    checked_path(inventory_ref)
    probe_code = Path(__file__).with_name('probe_arkit_faro_quality.py')
    if (report['stage'] != 'diagnostic_complete' or report['training_admission'] is not False
            or report['optimizer_updates'] != 0 or source['inventory_sha256'] != inventory_ref['sha256']
            or source['manifest_sha256'] != report['manifest_sha256']
            or report['manifest_sha256'] != config['manifest_sha256']
            or report['code_sha256'] != sha_file(probe_code) or not 1 <= len(report['records']) <= 32):
        raise ValueError('Matching completed bounded probe and frozen inventory required')
    output = root / attempt
    output.mkdir(exist_ok=False)
    records, seen = [], set()
    with readonly_db(inventory_ref['path']) as db:
        for number, record in enumerate(report['records']):
            found = db.execute('SELECT ordinal,image,row_json FROM images WHERE sha256=?',
                               (record['image_sha256'],)).fetchall()
            if len(found) != 1 or record['stage'] != 'diagnostic_complete':
                raise ValueError('Exact unique completed training frame required')
            ordinal, path, encoded = found[0]
            row = dict(json.loads(encoded), image=path)
            if (ordinal in seen or row['dataset'] != 'arkitscenes' or row['split'] != 'train'
                    or row['group'] != record['group'] or row['video_id'] != record['video']
                    or row['timestamp'] != record['timestamp']):
                raise ValueError('Frozen RGB identity differs')
            seen.add(ordinal)
            bundle = RetainedBundle(probe / 'native-source', record['source_proofs'])
            depth, valid, kd, _, proofs, quality = convert(row, bundle)
            base, stem = row['source_url'].rsplit('/', 1)[0], PurePosixPath(row['source_member']).stem
            faro_path, faro_proof = bundle.get(base, 'highres_depth', stem)
            if canonical_json(quality) != canonical_json(record['native_quality']) or [*proofs, faro_proof] != record['source_proofs']:
                raise ValueError('Reproduced native statistics or proofs differ')
            faro, turn = load_image(faro_path), row['image_rotation_cw']
            fields = [display_fields(depth, valid, kd, faro,
                row['native_camera_width_height_fx_fy_cx_cy'], turn, flag) for flag in (False, True)]
            comparisons = [x[4] for x in fields]
            if canonical_json(comparisons) != canonical_json(record['comparisons']):
                raise ValueError('Reproduced FARO comparisons differ')
            low_path, _ = bundle.get(base, 'lowres_wide', stem)
            panels = [(np.rot90(load_image(path), -turn // 90), 'Original upright VGA RGB'),
                (depth_colors(depth, valid), 'AppleDepth native grid, exact valid mask'),
                (depth_colors(fields[0][0], fields[0][1]), 'FARO mesh native grid (not direct scanner Z)'),
                (np.rot90(load_image(low_path), -turn // 90), 'Exact native lowres RGB'),
                (error_colors(fields[0][2], fields[0][3]), 'Native - FARO: integer-origin hypothesis'),
                (error_colors(fields[1][2], fields[1][3]), 'Native - FARO: half-center hypothesis')]
            page = Image.new('RGB', (1200, 1240), 'white')
            draw = ImageDraw.Draw(page)
            lines = [f"Frozen ordinal {ordinal}: {row['group']} | {stem} | CW {turn}",
                'Depth: common log 0.25-8 m, blue near / red far; black invalid. No input array changes.',
                'Signed error: blue <= -10cm, white zero, red >= +10cm; black = not compared.',
                'Two uncalibrated camera hypotheses; no fitting. Filtered subset is not a corpus accuracy certificate.']
            for flag, stats in zip(('Integer', 'Half-center'), comparisons):
                lines.append(f"{flag}: {stats['compared_pixels']} pixels; median signed {stats['median_signed_difference_m']:.3f} m; "
                    f"abs p95 {stats['abs_error_m_quantiles'][2]:.3f} m; within 5cm {stats['fraction_within_5cm']:.2%}")
            for i, line in enumerate(lines):
                draw.text((8, 8 + 19 * i), line, fill='black')
            for i, (array, title) in enumerate(panels):
                x, y = i % 3 * 400, 136 + i // 3 * 550
                draw.text((x + 5, y), title, fill='black')
                thumb = ImageOps.contain(Image.fromarray(array), (392, 520),
                    Image.Resampling.LANCZOS if i in (0, 3) else Image.Resampling.NEAREST)
                page.paste(thumb, (x + (400 - thumb.width) // 2, y + 22))
            target = output / f'{number:02d}-{ordinal}.png'
            incoming = target.with_suffix('.png.incoming')
            with incoming.open('xb') as file:
                page.save(file, format='PNG')
            incoming.rename(target)
            records.append(dict(corpus_ordinal=ordinal, image=file_ref(path), group=row['group'],
                video=row['video_id'], timestamp=row['timestamp'], image_rotation_cw=turn,
                native_quality=quality, comparisons=comparisons, receipts=bundle.receipts,
                page=file_ref(target), source_bytes_reproduced=True))
    checked_path(probe_ref)
    publish_json(output / 'report.json', dict(stage='faro_visual_evidence_rendered_not_reviewed',
        records=records, images=len(records), videos=len({r['video'] for r in records}),
        probe=probe_ref, inventory=inventory_ref, source_report=source_ref,
        original_comparison_code=file_ref(probe_code), renderer=file_ref(Path(__file__)),
        conversion_code=file_ref(Path(__file__).with_name('cache_arkit_geometry.py')),
        caveats=report['caveats'], source_quality_accepted=False, registration_calibration_accepted=False,
        source_flags_cleared=False, training_admission=False, optimizer_updates=0,
        network_used=False, gpu_used=False, display_only_resizing=True))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--probe', default='arkit-faro-quality-probe-01')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.probe, args.attempt)):
        parser.error('Use attempt basenames')
    result = render(json.loads(args.config.read_text()), args.probe, args.attempt)
    print(json.dumps(dict(rendered_images=len(result))), flush=True)
