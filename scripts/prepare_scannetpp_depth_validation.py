"""Prepare an isolated 120-view LiDAR validation subset, never a training input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from radio_moge.scannetpp_depth import read_selected_depth
from radio_moge.scannetpp_validation import (
    aligned_view, internal_holdout, read_video_frame, registered_frames, select_spaced_frames,
)


def atomic(path, data):
    incoming = path.with_name(path.name + '.incoming')
    incoming.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    incoming.replace(path)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def completed_assets(data, control, scene, cutoff):
    evidence = {}
    for name in ['rgb.mkv', 'rgb_mask.zip', 'depth.zip', 'colmap.zip', 'pose_intrinsic_imu.zip']:
        relative = f'data/{scene}/iphone/{name}'
        receipt = json.loads((control/'receipts'/(sha(relative.encode())+'.json')).read_text())
        if (receipt['path'] != relative or receipt['completed_at'] > cutoff
                or not receipt['bytes'] > 0 or len(receipt['sha256']) != 64):
            raise ValueError('Incomplete or mismatched asset receipt')
        row = {k: receipt[k] for k in ['path', 'bytes', 'sha256', 'completed_at']}
        if name.endswith('.zip'):
            path = control/'extractions'/(sha(str(data/relative).encode())+'.json')
            extraction = json.loads(path.read_text())
            if (extraction.get('stage') != 'passed' or not extraction.get('crc_checked')
                    or extraction['completed_at'] > cutoff):
                raise ValueError('Asset extraction/CRC is incomplete')
            row['extraction'] = {k: extraction[k] for k in ['stage', 'crc_checked', 'completed_at']}
        elif (data/relative).stat().st_size != receipt['bytes']:
            raise ValueError('Video size differs from its verified transfer receipt')
        evidence[name] = row
    return evidence


def capture(path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened() or cap.getBackendName() != 'FFMPEG':
        cap.release()
        raise ValueError('Existing bounded CPU video backend could not open asset')
    if cap.get(cv2.CAP_PROP_ORIENTATION_META) != 0:
        cap.release()
        raise ValueError('Unexpected video orientation')
    return cap


def prepare(data, control, snapshot, output, cutoff):
    cv2.setNumThreads(1)
    frozen = json.loads((snapshot/'snapshot.json').read_text())
    verified = json.loads((snapshot/'verification.json').read_text())
    if verified['stage'] != 'passed' or output.exists():
        raise ValueError('Require a verified snapshot and an unused output directory')
    scenes = sorted(frozen['internal_validation_scenes'])
    if len(scenes) != 30 or len(set(scenes)) != 30 or not all(map(internal_holdout, scenes)):
        raise ValueError('Expected the fixed 30 permanent internal heldout scenes')
    split_relative = 'splits/nvs_sem_train.txt'
    split_bytes = (data/split_relative).read_bytes()
    inventory = json.loads((control/'inventory.json').read_text())
    if sha(split_bytes) != inventory['metadata_sha256'][split_relative]:
        raise ValueError('Official scene split differs from frozen download inventory')
    official_train = set(split_bytes.decode().split())
    if not set(scenes) <= official_train:
        raise ValueError('Unexpected official split in heldout validation')
    output.mkdir(mode=0o700)
    started, rows, scene_evidence, rejected = time.time(), [], {}, []

    def status(stage, **extra):
        atomic(output/'status.json', dict(stage=stage, pid=os.getpid(), started_at=started,
               updated_at=time.time(), completed_scenes=len(scene_evidence),
               prepared_views=len(rows), **extra))

    status('preparing')
    try:
        for scene in scenes:
            status('preparing', scene=scene)
            evidence = completed_assets(data, control, scene, cutoff)
            source = data/'data'/scene/'iphone'
            registered = registered_frames(source/'colmap/cameras.txt', source/'colmap/images.txt')
            selected = select_spaced_frames(registered)
            poses = json.loads((source/'pose_intrinsic_imu.json').read_text())
            depths, decoder = read_selected_depth(source/'depth.bin', selected)
            video, mask_video = capture(source/'rgb.mkv'), capture(source/'rgb_mask.mkv')
            try:
                metadata = []
                for cap in [video, mask_video]:
                    values = [cap.get(key) for key in [cv2.CAP_PROP_FRAME_WIDTH,
                              cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS]]
                    if values[:2] != [1920, 1440] or values[2] <= max(selected) or abs(values[3]-60) > .001:
                        raise ValueError('Video dimensions, frame population or FPS differ')
                    metadata.append(values)
                if metadata[0] != metadata[1]:
                    raise ValueError('RGB/mask video timelines differ')
                for index in selected:
                    name = f'frame_{index:06d}'
                    pose = poses[name]
                    intrinsic = np.asarray(pose['intrinsic'])
                    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
                        raise ValueError('Invalid per-frame ARKit RGB intrinsics')
                    for key in ['pose', 'aligned_pose']:
                        matrix = np.asarray(pose[key])
                        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                            raise ValueError('Invalid per-frame pose')
                    image, depth, alignment = aligned_view(read_video_frame(video, index),
                        read_video_frame(mask_video, index), depths[index], registered[index]['camera'])
                    if alignment['rgb_valid_fraction'] < .90 or alignment['depth_valid_fraction'] < .50:
                        rejected.append(dict(scene=scene, frame=index, **alignment))
                        raise ValueError('Fixed validation frame has insufficient valid support')
                    folder = output/scene
                    folder.mkdir(exist_ok=True)
                    image_path, depth_path = folder/(name+'.png'), folder/(name+'-depth.npy')
                    if not cv2.imwrite(str(image_path), image):
                        raise ValueError('RGB output failed')
                    np.save(depth_path, depth, allow_pickle=False)
                    row = dict(id=f'scannetpp-lidar:{scene}:{index}', image=str(image_path),
                               depth=str(depth_path), depth_scale=1.0, dataset='scannetpp_iphone_lidar',
                               group='scannetpp:'+scene, scene=scene, split='val', frame_index=index,
                               sha256=sha(image_path.read_bytes()), depth_sha256=sha(depth_path.read_bytes()),
                               depth_source='measured_iPhone_LiDAR', training_admission=False,
                               source_rgb=str(source/'rgb.mkv'), source_depth=str(source/'depth.bin'),
                               source_mask=str(source/'rgb_mask.mkv'), timestamp=pose['timestamp'],
                               source_arkit_intrinsic=pose['intrinsic'], camera=registered[index],
                               alignment=alignment)
                    rows.append(row)
            finally:
                video.release()
                mask_video.release()
            scene_evidence[scene] = dict(assets=evidence, selected_frame_indices=selected,
                                         decoder=decoder, video_metadata=metadata)
            atomic(output/'scene-evidence.json', scene_evidence)
        if len(rows) != 120 or len({r['id'] for r in rows}) != 120:
            raise ValueError('Expected 120 unique heldout views')
        manifest = ''.join(json.dumps(row, sort_keys=True, allow_nan=False)+'\n' for row in rows)
        temporary = output/'val-scannetpp-lidar.jsonl.incoming'
        temporary.write_text(manifest)
        temporary.replace(output/'val-scannetpp-lidar.jsonl')
        atomic(output/'verification.json', dict(stage='passed', completed_at=time.time(),
            scenes=30, views=120, views_per_scene=4, permanent_heldouts=scenes,
            source_snapshot=str(snapshot), cutoff=cutoff, manifest_sha256=sha(manifest.encode()),
            train_admission=False, current_training_configuration_changed=False,
            gpu_used=False, environment_changed=False, cpu_threads=1,
            rgb_depth_mask_frame_ids_match=True, all_assets_receipt_crc_gated=True,
            depth_units='metres', native_depth_hw=[192, 256], output_hw=[576, 768],
            depth_sampling='nearest; scaled RGB intrinsics; shared OPENCV undistortion map',
            anonymous_and_sensor_invalid_depth_zero=True, min_rgb_valid_fraction=.90,
            min_depth_valid_fraction=.50, model_metric_selection=False,
            current_dslr_validation_unchanged=True, model_evaluation_performed=False,
            alignment_scope='Official co-registration plus identical frame IDs and common calibrated pixel maps; numerical and synthetic geometry tests, not independent physical sensor calibration.'))
        status('complete')
    except Exception as error:
        atomic(output/'rejected-views.json', rejected)
        status('failed', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['dataset', 'control', 'snapshot', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cutoff', type=float, required=True)
    args = parser.parse_args()
    prepare(args.dataset, args.control, args.snapshot, args.output, args.cutoff)
