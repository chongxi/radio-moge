"""Bind sixteen preselected confirmation views to the frozen source policy.

Checksums existing scanner files sequentially; does not decode ranges or RGB,
render a confirmation view, start an optimizer, or authorize execution itself.
"""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from probe_scannetpp_triangle_scan_support import scanner_pose_diagnostics
from probe_scannetpp_v2_source_pilot import read
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.mesh_clip import MeshClipContract
from radio_moge.scanner_support import ScannerSupportContract


def freeze(root, output):
    policy_ref = file_ref(root/'scannetpp-native-v2-conditional-policy-plan-01/report.json')
    selection_ref = file_ref(root/'scannetpp-native-v2-validation-01/report.json')
    if (policy_ref['sha256'] != '3216cac67ac2a03854858b174a6362f8bddbb73940c8018b610719de0ac23d75'
            or selection_ref['sha256'] != '37b957e897013bf3e58d9636b279d70ad1c1598c4f90f60bd7a89cae8d87c322'):
        raise ValueError('Original sealed selection and frozen conditional policy required')
    policy, selection = read(policy_ref, root), read(selection_ref, root)
    if policy['policy'] != asdict(ScannerSupportContract()):
        raise ValueError('Source quality policy changed')
    all_rows = read(selection['selection'], root)
    rows = [r for r in all_rows if r['role'] == 'confirmation']
    other_groups = {r['group'] for r in all_rows if r['role'] != 'confirmation'}
    if (len(rows) != 16 or len({r['group'] for r in rows}) != 16
            or len({r['corpus_ordinal'] for r in rows}) != 16
            or any(r['group'] in other_groups for r in rows)):
        raise ValueError('Sixteen original disjoint confirmation scenes required')
    bound = []
    encoded_bytes = 0
    for chosen in rows:
        row = chosen['image']
        if (row['dataset'] != 'scannetpp_dslr' or row['split'] != 'train'
                or row.get('image_rotation_cw', 0) != 0 or row['official_split'] != 'nvs_sem_train'
                or chosen['group'] != 'scannetpp:'+row['scene']):
            raise ValueError('Original perspective training RGB required')
        scene = row['scene']
        base = Path(row['image']).parents[2]
        pose_ref = file_ref(root/'scanner-poses/data'/scene/'scans/scanner_poses.json')
        poses = np.asarray(read(pose_ref, root), np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
            raise ValueError('Two published scanner poses required; never substitute')
        if np.array_equal(poses[0], poses[1]):
            raise ValueError('Distinct retained scanner transforms required')
        scans = []
        for scan in (0, 1):
            scanner_pose_diagnostics(poses[scan])
            files = {}
            for kind, folder in (('range', 'depth'), ('mask', 'anon_mask'), ('azim', 'azim'), ('elev', 'elev')):
                path = base/'panocam'/folder/f'{scan}.png'
                files[kind] = file_ref(path)
                encoded_bytes += path.stat().st_size
            scans.append(dict(scan=str(scan), inputs=files, pose=poses[scan].tolist()))
        h, w = round(row['height']*640/row['width']), 640
        bound.append(dict(original_selection=chosen, corpus_ordinal=chosen['corpus_ordinal'],
            scene=scene, role='confirmation', grid=[h, w], scanner_poses=pose_ref, scanners=scans,
            pixel_yx=[[int((y+.5)*h/8), int((x+.5)*w/12)] for y in range(8) for x in range(12)]))
        print(chosen['corpus_ordinal'], 'fixed source checksums bound; no decode', flush=True)
    output.mkdir(exist_ok=False)
    publish_json(output/'selection.json', bound)
    result = dict(stage='confirmation_inputs_and_policy_frozen_execution_not_yet_admitted',
        original_selection=selection_ref, source_policy=policy_ref, exclusions=policy['exclusions'],
        selection=file_ref(output/'selection.json'), scanner_source_files=128, encoded_source_bytes_hashed=encoded_bytes,
        policy=asdict(ScannerSupportContract()), clip_contract=MeshClipContract().as_dict(),
        images=16, distinct_scenes=16, DSLR_slots=1536, station_slots=3072,
        maximum_nine_direction_source_queries=27648, maximum_exact_target_queries=3072,
        numerical_rule='Existing clipped CPU/GPU support intersect public renderer consensus; retain unchanged CPU values only.2mm remains numerical,5percent numerical imagefloor unchanged.',
        source_rule='Exact frozen conditional two-station policy. Both fixed0/1 scanners, no replacements. Include unsupported/missing/occluded points in1536/3072denominators.',
        outcome='Evaluate frozen conditional sampled support and inspect all16pages. No dense-mask/source-wide admission from samples.',
        full_dense_future_requirement='Every full-image grid slot evaluated and conditional mask subset numerical;>=5percent image pixels and>=32points. No interpolation from96probes.',
        required_before_execution=['Implement dedicated confirmation runner binding this manifest, all source hashes, frozen policy and exact implementation.',
            'Fresh actual CUDA fixtures for current float32 clip guard and unchanged numerical thresholds; bind to exact new numerical implementation.',
            'Test confirmation scope/identity/all-slot aggregation and strict policy; do not relabel old development fixtures as current.',
            'Use originalGPUlock, one image/scanner at a time,maththreads1, preserved viewer; no optimizer,batchregeneration,automaticretry or thresholdtuning.'],
        confirmation_execution_allowed=False, confirmation_images_read=0,
        decoded_scanner_range_images=0, RGB_payload_reads=0,
        physical_uncertainty_budget_calibrated=False, independent_physical_calibration=False,
        source_quality_accepted=False, training_admission=False, optimizer_updates=0,
        no_new_user_confirmation_required=True, implementation=file_ref(Path(__file__)))
    publish_json(output/'report.json', result)
    print(file_ref(output/'report.json'), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    freeze(args.root.resolve(), args.output.resolve())
