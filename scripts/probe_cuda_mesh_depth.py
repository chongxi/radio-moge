"""Compare the CUDA rasterizer to independent CPU ray intersections; no training."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import binary_erosion, maximum_filter, minimum_filter
import torch
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from indoor_download import write_json
from probe_native_depth import quantiles
from probe_scannetpp_mesh import colmap_pose, sha_file, source_rows
from radio_moge.geometry_sources import normalized_opencv_intrinsics, pinhole_rays
from radio_moge.mesh_depth import CudaMeshDepth


def cpu_depth(mesh, k, pose, h, w):
    rays = pinhole_rays(k, h, w).reshape(-1, 3) @ pose[:3, :3].T
    origins = np.broadcast_to(pose[:3, 3], rays.shape).copy()
    locations, indices, _ = RayMeshIntersector(mesh).intersects_location(
        origins, rays, multiple_hits=False)
    depth = np.zeros(h * w, np.float32)
    depth[indices] = ((locations - pose[:3, 3]) @ pose[:3, :3])[:, 2]
    return depth.reshape(h, w)


def compare(cpu, gpu, valid):
    cpu_valid = np.isfinite(cpu) & (cpu > .01) & (cpu < 100)
    support_agreement = float((cpu_valid == valid).mean())
    # Quantify all common pixels AND a stable interior; don't hide edge errors.
    common = valid & cpu_valid
    stable = binary_erosion(common, iterations=2)
    stable &= (maximum_filter(cpu, 3) - minimum_filter(cpu, 3)) < .02 * cpu
    differences = np.abs(cpu - gpu)
    return dict(support_agreement=support_agreement, common_pixels=int(common.sum()),
                stable_pixels=int(stable.sum()), all_common_error_m=quantiles(differences[common]),
                stable_error_m=quantiles(differences[stable]))


def probe(config, attempt):
    root = Path(config['root'])
    output = root / attempt
    if output.exists():
        raise FileExistsError('Preserve old rasterizer evidence; use a new attempt')
    output.mkdir()
    result = dict(stage='running', training_admission=False, training_started=False,
                  started_at=time.time(), torch=torch.__version__, cuda=torch.version.cuda,
                  rasterizer_version=importlib.metadata.version('nvdiffrast'),
                  rasterizer_commit='253ac4fcea7de5f396371124af597e6cc957bfae', scenes=[])
    write_json(output / 'report.json', result)
    for row in source_rows(config, 3):
        scene = Path(config['dataset_root']) / 'scannetpp-v2-full/data' / row['scene']
        mesh_path = scene / 'scans/mesh_aligned_0.05.ply'
        mesh = trimesh.load_mesh(mesh_path, process=False)
        pose = colmap_pose(scene / 'dslr/colmap/images.txt', row['view'])
        k = normalized_opencv_intrinsics(*row['camera_width_height_fx_fy_cx_cy'])
        h, w = round(row['height'] * 640 / row['width']), 640
        reference = cpu_depth(mesh, k, pose, h, w)
        renderer = CudaMeshDepth(mesh.vertices, mesh.faces)
        depth, valid, _ = renderer.render(k, pose, h, w)
        metrics = compare(reference, depth, valid)
        times = []
        for _ in range(5):
            start = time.monotonic()
            renderer.render(k, pose, h, w)
            times.append(time.monotonic() - start)
        record = dict(scene=row['scene'], view=row['view'], mesh_sha256=sha_file(mesh_path),
                      render_size=[h, w], **metrics, render_seconds=times)
        stable = metrics['stable_error_m']
        record['parity_pass'] = bool(stable and metrics['stable_pixels'] > 1000
            and metrics['support_agreement'] > .999 and stable['p50'] < 1e-4
            and stable['p99'] < .001)
        result['scenes'].append(record)
        del renderer
        write_json(output / 'report.json', result)
    result.update(stage='complete', completed_at=time.time(),
                  parity_pass=all(x['parity_pass'] for x in result['scenes']))
    write_json(output / 'report.json', result)
    print(json.dumps(dict(stage='complete', parity_pass=result['parity_pass'])), flush=True)
    if not result['parity_pass']:
        raise ValueError('CUDA/CPU parity failed; do not generate a corpus')


if __name__ == '__main__':
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', default='cuda-mesh-probe-01')
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('attempt must be a new directory basename')
    probe(json.loads(args.config.read_text()), args.attempt)
