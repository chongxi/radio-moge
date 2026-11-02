"""Strict CPU/CUDA parity on GPU-only visibility masks, zero optimizer steps.

Retains every failed view in a new report. A source audit and full cache/QC are
still required; passing software parity is not physical source accuracy.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from indoor_download import write_json
from probe_cuda_mesh_depth import compare, cpu_depth
from probe_native_depth import quantiles
from probe_scannetpp_mesh import colmap_pose
from radio_moge.geometry_cache import sha_file
from radio_moge.geometry_sources import normalized_opencv_intrinsics
from radio_moge.mesh_depth import CudaMeshDepth, verify_cpu_intersections


def selected(root):
    with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
        query = "SELECT DISTINCT json_extract(row_json,'$.scene') FROM images WHERE dataset='scannetpp_dslr' ORDER BY 1"
        scenes = [r[0] for r in db.execute(query)]
        chosen = {scenes[i] for i in np.linspace(0, len(scenes) - 1, 12).astype(int)}
        # Preserve the first parity probe, and include two source-review scenes.
        chosen.update(('00777c41d4', '00a231a370', '0271889ec0', '033d0b9343', '04c23ccd1d'))
        for scene in sorted(chosen):
            values = db.execute("SELECT row_json FROM images WHERE dataset='scannetpp_dslr' "
                "AND json_extract(row_json,'$.scene')=? ORDER BY ordinal", (scene,)).fetchall()
            for index in sorted({0, len(values) - 1}):
                row = json.loads(values[index][0])
                if row['split'] != 'train':
                    raise ValueError('Only frozen training views allowed')
                yield row


def probe(config, attempt, verify_cpu=False):
    root = Path(config['root'])
    output = root / attempt
    output.mkdir(exist_ok=False)
    result = dict(stage='running', started_at=time.time(), training_admission=False,
        training_started=False, optimizer_updates=0, torch=torch.__version__,
        cuda=torch.version.cuda, manifest_sha256=config['manifest_sha256'],
        cpu_verification_of_every_candidate=verify_cpu,
        code_sha256=sha_file(Path(__file__).parents[1] / 'src/radio_moge/mesh_depth.py'),
        thresholds=dict(retained_cpu_support_fraction=1., retained_max_error_m=.002,
                        retained_p99_error_m=.001, minimum_retained_pixels=1000), records=[])
    write_json(output / 'report.json', result)
    active, renderer, mesh = None, None, None
    try:
        for row in selected(root):
            scene = Path(config['dataset_root']) / 'scannetpp-v2-full/data' / row['scene']
            mesh_path = scene / 'scans/mesh_aligned_0.05.ply'
            if active != row['scene']:
                del renderer, mesh
                mesh = trimesh.load_mesh(mesh_path, process=False)
                renderer = CudaMeshDepth(mesh.vertices, mesh.faces)
                intersector = RayMeshIntersector(mesh)
                active, mesh_hash = row['scene'], sha_file(mesh_path)
            pose = colmap_pose(scene / 'dslr/colmap/images.txt', row['view'])
            k = normalized_opencv_intrinsics(*row['camera_width_height_fx_fy_cx_cy'])
            h, w = round(row['height'] * 640 / row['width']), 640
            reference = cpu_depth(mesh, k, pose, h, w)
            original, original_valid, _ = renderer.render(k, pose, h, w)
            start = time.monotonic()
            depth, valid, _, stats = renderer.render_conservative(k, pose, h, w)
            seconds = time.monotonic() - start
            cpu_valid = np.isfinite(reference) & (reference > .01) & (reference < 100)
            common = valid & cpu_valid
            errors = np.abs(depth - reference)
            retained = quantiles(errors[common])
            passed = bool(valid.sum() >= 1000 and not (valid & ~cpu_valid).any()
                          and retained['max'] <= .002 and retained['p99'] <= .001)
            # Explicit worst-pixel coordinates aid review; no hidden quantile-only pass.
            worst = np.unravel_index(np.where(common, errors, -1).argmax(), errors.shape)
            record = dict(scene=row['scene'], view=row['view'], mesh_sha256=mesh_hash,
                render_size=[h, w], original=compare(reference, original, original_valid),
                **stats, retained_cpu_missing_pixels=int((valid & ~cpu_valid).sum()),
                retained_error_m=retained, worst_pixel_yx=[int(x) for x in worst],
                worst_depths_cpu_gpu_m=[float(reference[worst]), float(depth[worst])],
                render_mask_seconds=seconds, passed=passed)
            if verify_cpu:
                start = time.monotonic()
                verified, support, verification = verify_cpu_intersections(intersector, k, pose, depth, valid)
                verified_error = np.abs(verified - reference)[support]
                record.update(gpu_only_passed=passed, verification=verification,
                    verified_error_vs_full_cpu_m=quantiles(verified_error),
                    cpu_verification_seconds=time.monotonic() - start,
                    passed=bool(support.sum() >= 1000 and not (support & ~cpu_valid).any()
                                and verified_error.max() <= 1e-5))
            result['records'].append(record)
            write_json(output / 'report.json', result)
            print(json.dumps(record), flush=True)
        result.update(stage='complete', completed_at=time.time(),
                      passed=all(r['passed'] for r in result['records']))
        write_json(output / 'report.json', result)
    except BaseException as error:
        result.update(stage='failed', error=type(error).__name__, message=str(error))
        write_json(output / 'report.json', result)
        raise


if __name__ == '__main__':
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--verify-cpu', action='store_true')
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a new attempt basename')
    probe(json.loads(args.config.read_text()), args.attempt, args.verify_cpu)
