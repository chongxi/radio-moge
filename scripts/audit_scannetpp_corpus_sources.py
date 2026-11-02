"""Audit ALL frozen DSLR scene sources against ALL available raw scans.

Systematic subsampling of each panorama, not a claim of full per-ray inspection
or independent scanner accuracy. Keeps every scene failure visible. No training.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import time

import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from indoor_download import write_json
from probe_scannetpp_mesh import dslr_check, scanner_mesh_check, sha_file
from radio_moge.geometry_cache import publish_json


def audit(config, attempt):
    root, data = Path(config['root']), Path(config['dataset_root'])
    inventory = json.loads((root / 'inventory.json').read_text())
    if (inventory['stage'] != 'inventory_complete'
            or inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']):
        raise ValueError('Frozen inventory identity differs')
    groups = [entry for entry in inventory['source_groups'] if entry['dataset'] == 'scannetpp_dslr']
    output = root / attempt
    if output.exists():
        raise FileExistsError('Audit exists; do not implicitly restart or overwrite')
    output.mkdir()
    status = dict(stage='auditing', started_at=time.time(), expected_scenes=len(groups),
                  scenes_completed=0, scenes_needing_review=0, scans_checked=0,
                  sampled_rays=0, manifest_sha256=config['manifest_sha256'],
                  training_started=False, training_admission=False, scenes=[])
    publish_json(output / 'policy.json', dict(raw_scan_stride=128, neighboring_pixels=3,
        stable_range_relative_difference=.03, anonymous_mask_value=255,
        diagnostic_review_thresholds=dict(minimum_eligible_rays=100,
            minimum_mesh_hit_fraction=.95, minimum_within_5cm_fraction=.95),
        thresholds_are_diagnostics_not_training_admission=True,
        scope='all frozen DSLR scenes/all raw scans, systematic sampled rays only',
        source_of_mesh='same laser acquisition; not independent absolute accuracy',
        script_sha256=sha_file(Path(__file__)),
        helper_sha256=sha_file(Path(__file__).with_name('probe_scannetpp_mesh.py'))))
    write_json(output / 'status.json', status)
    with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
        db.execute('PRAGMA cache_size=-32768')
        for group in groups:
            scene_name = group['group'].split(':')[-1]
            scene = data / 'scannetpp-v2-full/data' / scene_name
            record = dict(scene=scene_name, images_in_training_manifest=group['images'],
                          started_at=time.time(), training_admission=False)
            try:
                encoded, image = db.execute('SELECT row_json,image FROM images WHERE scene_group=?'
                    ' ORDER BY ordinal LIMIT 1', (group['group'],)).fetchone()
                row = json.loads(encoded)
                row['image'] = image
                if row['scene'] != scene_name or row['split'] != 'train':
                    raise ValueError('Scene or training split differs')
                mesh_path = scene / 'scans/mesh_aligned_0.05.ply'
                mesh = trimesh.load_mesh(mesh_path, process=False)
                intersector = RayMeshIntersector(mesh)
                record.update(mesh_sha256=sha_file(mesh_path), vertices=len(mesh.vertices),
                              faces=len(mesh.faces))
                record['scanner_vs_mesh'] = scanner_mesh_check(
                    scene, root / 'scanner-poses', intersector, scan_limit=None, stride=128)
                record['dslr'] = dslr_check(row, scene, mesh, intersector, output)
                record['review_scans'] = [x['scan'] for x in record['scanner_vs_mesh']
                    if x['eligible_sampled_rays'] < 100 or x['mesh_hit_fraction'] < .95
                    or x['within_5cm_fraction'] is None or x['within_5cm_fraction'] < .95]
                record['stage'] = 'needs_review' if record['review_scans'] else 'source_checks_passed'
                status['scans_checked'] += len(record['scanner_vs_mesh'])
                status['sampled_rays'] += sum(x['eligible_sampled_rays'] for x in record['scanner_vs_mesh'])
                del mesh, intersector
            except Exception as error:
                record.update(stage='needs_review', error=type(error).__name__, message=str(error)[:1000])
            record['completed_at'] = time.time()
            publish_json(output / f'{scene_name}.json', record)
            status['scenes_completed'] += 1
            status['scenes_needing_review'] += record['stage'] == 'needs_review'
            status['scenes'].append(dict(scene=scene_name, report=f'{scene_name}.json',
                                         sha256=sha_file(output / f'{scene_name}.json'), stage=record['stage']))
            status['updated_at'] = time.time()
            write_json(output / 'status.json', status)
            print(json.dumps({k: v for k, v in status.items() if k != 'scenes'}), flush=True)
    status.update(stage='complete', completed_at=time.time())
    write_json(output / 'status.json', status)
    publish_json(output / 'report.json', status)


if __name__ == '__main__':
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', default='scannetpp-source-audit-01')
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('attempt must be a new directory basename')
    audit(json.loads(args.config.read_text()), args.attempt)
