"""Cross-check exact pinhole-frame conversion on the most tilted training cameras."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np

from indoor_download import write_json
from probe_native_depth import Evidence, hdf5_array, hypersim_probe, quantiles
from radio_moge.geometry_sources import hypersim_pinhole_frame, pinhole_rays, range_to_z


def probe(config, attempt):
    root = Path(config['root'])
    metadata_path = root / 'native-probe-01/hypersim/metadata_camera_parameters.csv'
    cameras = {}
    with metadata_path.open() as file:
        for row in csv.DictReader(file):
            matrix = np.array([[float(row[f'M_cam_from_uv_{i}{j}']) for j in range(3)] for i in range(3)])
            cameras[row['scene_name']] = matrix
    inventory = json.loads((root / 'inventory.json').read_text())
    groups = [x['group'] for x in inventory['source_groups'] if x['dataset'] == 'hypersim']
    ordered = sorted(groups, key=lambda group: -np.linalg.norm(cameras[group.split(':')[1]][2, :2]))
    evidence = Evidence(root / attempt)
    rows = []
    with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
        for group in ordered[:3]:
            rows.append(json.loads(db.execute('SELECT row_json FROM images WHERE scene_group=?'
                ' ORDER BY ordinal LIMIT 1', (group,)).fetchone()[0]))
    initial = hypersim_probe(evidence, rows)
    reports = []
    for row, source_check in zip(rows, initial, strict=True):
        scene = row['group'].split(':')[1]
        location = evidence.root / f'hypersim/{scene}-{row["camera"]}-{row["frame_id"]:04d}'
        matrix = cameras[scene]
        k, rotation = hypersim_pinhole_frame(matrix)
        radial, world = hdf5_array(location / 'range.hdf5'), hdf5_array(location / 'position.hdf5')
        indices = hdf5_array(location / 'camera_keyframe_frame_indices.hdf5').reshape(-1)
        index = np.flatnonzero(indices == row['frame_id']).item()
        pose_r = hdf5_array(location / 'camera_keyframe_orientations.hdf5')[index]
        pose_t = hdf5_array(location / 'camera_keyframe_positions.hdf5')[index]
        with (location / 'metadata_scene.csv').open() as file:
            scale = float(next(x for x in csv.DictReader(file)
                if x['parameter_name'] == 'meters_per_asset_unit')['parameter_value'])
        cam = (world - pose_t) @ pose_r * scale
        cam *= np.array([1, -1, -1])
        truth = cam @ rotation.T
        z, valid = range_to_z(radial, pinhole_rays(k, *radial.shape),
                             np.isfinite(truth).all(-1) & (truth[..., 2] > 0))
        error = np.abs(truth[..., 2] - z)
        relative = error[valid] / truth[..., 2][valid]
        report = dict(scene=scene, intrinsics=k.tolist(), frame_rotation=rotation.tolist(),
                      source_check=source_check, z_vs_world_m=quantiles(error[valid]),
                      relative_z_error=quantiles(relative), training_admission=False,
                      preserves_original_rgb=True, passes=bool(np.quantile(relative, .99) < .005))
        reports.append(report)
    # All frozen training scene matrices must be decomposable, not just the probes.
    all_cameras = {group: hypersim_pinhole_frame(cameras[group.split(':')[1]])[0].tolist()
                   for group in groups}
    result = dict(stage='complete', training_admission=False, probe_scenes=reports,
                  exact_camera_decompositions=all_cameras,
                  passes=all(x['passes'] for x in reports), source_scope='training only')
    write_json(evidence.root / 'report.json', result)
    print(json.dumps(dict(stage=result['stage'], passes=result['passes'], cameras=len(all_cameras))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', default='hypersim-tilt-probe-01')
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('attempt must be a new directory basename')
    probe(json.loads(args.config.read_text()), args.attempt)
