"""Package cached-calibration snapshots into an offline 3D comparison page."""

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from PIL import Image


NAMES = ['fresh', 'gpu_10', 'gpu_30', 'gpu_first']


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.new')
    with temporary.open('xb') as stream:
        stream.write(data)
    os.replace(temporary, path)


def ply(depth, k, mask, rgb):
    height, width = depth.shape
    x, y = np.meshgrid((np.arange(width) + .5) / width, (np.arange(height) + .5) / height)
    points = (np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(k).T) * depth[..., None]
    points = points[mask].astype('<f4') * np.array([1, -1, -1], np.float32)
    assert np.isfinite(points).all() and len(points) > 1000
    vertices = np.empty(len(points), dtype=[('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])
    vertices['xyz'], vertices['rgb'] = points, rgb[mask]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment predicted scale, fixed first-frame K, common current reference mask; x right y up z backward\n'
              f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    return header.encode() + vertices.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.root
    report = json.loads((root / 'report.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    quality = json.loads((root / 'quality-records.json').read_text())
    assert report['completed'] and len(quality) == 360
    assert report['plan_sha256'] == hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest()
    data = {'names': NAMES, 'camera': report['fixed_camera'], 'timings': report['timings'],
            'curves': [[r['modes'][m]['abs_rel'] for r in quality] for m in NAMES[1:]], 'samples': []}
    export = root / 'pointclouds-01'
    export.mkdir(exist_ok=False)
    zip_path = export / 'full-resolution-3d.zip'
    zip_tmp = zip_path.with_name(zip_path.name + '.new')
    manifest = {'reference_is_ground_truth': False, 'camera': report['fixed_camera'],
                'support': 'Common current-frame reference mask for comparison; cached runtime uses finite-depth validity',
                'frames': [59, 179, 359], 'methods': NAMES, 'files': []}
    with zipfile.ZipFile(zip_tmp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for index in plan['snapshots']:
            with np.load(root / 'snapshots' / f'frame-{index:03d}.npz') as z:
                rgb = z['rgb'].copy()
                mask = z['current_reference_mask'].copy()
                camera = z['intrinsics'].copy()
                np.testing.assert_array_equal(camera, np.array(report['fixed_camera'], dtype=camera.dtype))
                maps = [z[name].copy() for name in NAMES]
            for depth in maps:
                assert np.isfinite(depth[mask]).all() and (depth[mask] > 0).all()
            payload = b''.join(d[::4, ::4].astype('<f4').tobytes() for d in maps)
            payload += mask[::4, ::4].astype('u1').tobytes() + rgb[::4, ::4].tobytes()
            assert len(payload) == 160 * 120 * 20
            jpeg = io.BytesIO()
            Image.fromarray(rgb).resize((320, 240)).save(jpeg, format='JPEG', quality=85)
            data['samples'].append({'frame': index, 'payload': base64.b64encode(gzip.compress(payload)).decode(),
                                    'rgb': 'data:image/jpeg;base64,' + base64.b64encode(jpeg.getvalue()).decode(),
                                    'range': np.quantile(maps[0][mask], [.02, .98]).tolist(),
                                    'metrics': quality[index]['modes']})
            if index in manifest['frames']:
                for name, depth in zip(NAMES, maps):
                    content = ply(depth, camera, mask, rgb)
                    filename = f'frame-{index:03d}-{name}.ply'
                    archive.writestr(filename, content)
                    manifest['files'].append({'file': filename, 'vertices': int(mask.sum()),
                                               'sha256': hashlib.sha256(content).hexdigest()})
        archive.writestr('manifest.json', json.dumps(manifest, indent=2) + '\n')
    os.replace(zip_tmp, zip_path)
    write(export / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    repo = Path(__file__).resolve().parents[1]
    bundle = export / 'viewer.bundle.js'
    subprocess.run(['bun', 'build', str(repo / 'web/ppd-cached-geometry.js'), '--target=browser', '--minify',
                    '--outfile=' + str(bundle)], check=True)
    html = (repo / 'web/ppd-cached-geometry.html').read_text()
    html = html.replace('__DATA__', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
    html = html.replace('__BUNDLE__', bundle.read_text().replace('</script', '<\\/script'))
    write(args.output, html.encode())
    write(export / 'index.html', html.encode())
    print(json.dumps({'page': str(args.output), 'bytes': args.output.stat().st_size,
                       'full_resolution_zip': str(zip_path), 'ply_files': len(manifest['files'])}))


if __name__ == '__main__':
    main()
