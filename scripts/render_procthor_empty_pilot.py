"""Render a bounded RGB-only ProcTHOR pilot with CPU Vulkan and pinned houses."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image


BUILD = 'f0825767cd50d69f666c7f282e54abfe58f1e917'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    incoming = path.with_name(path.name + '.incoming')
    with incoming.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(incoming, path)


def main(root, output):
    if importlib.metadata.version('ai2thor') != '5.0.0':
        raise ValueError('Use the isolated pinned AI2THOR environment')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Pilot must leave the training GPU unused')
    icd = Path(os.environ['VK_ICD_FILENAMES'])
    if 'libvulkan_lvp.so' not in json.loads(icd.read_text())['ICD']['library_path']:
        raise ValueError('Only the private CPU lavapipe ICD is admitted for this pilot')
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    build = json.loads((root / 'procthor-unity-build-01/report.json').read_text())
    executable = Path(build['executable'])
    if build['commit'] != BUILD or sha(executable) != build['executable_sha256']:
        raise ValueError('Pinned Unity executable differs')
    from ai2thor.controller import Controller
    from ai2thor.fifo_server import FifoServer
    from ai2thor.platform import CloudRendering

    class BoundedController(Controller):
        def unity_command(self, width, height, headless):
            return super().unity_command(width, height, headless) + [
                '-force-vulkan', '-batchmode', '-job-worker-count', '1',
                '-logFile', str(output / 'unity.log')]

    if hasattr(os, 'sched_getaffinity'):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[-2:])
    houses = []
    for ordinal in range(4):
        name = f'train-house-{ordinal:05d}.json'
        original_path = root / 'procthor-source-01' / name
        empty_path = root / 'procthor-empty-candidates-01' / name
        original = json.loads(original_path.read_text())
        empty = json.loads(empty_path.read_text())
        expected = dict(original, objects=[])
        if empty != expected:
            raise ValueError('Only the object list may differ from the released house')
        houses.append((ordinal, empty, original_path, empty_path))
    records, controller = [], None
    publish(output / 'status.json', dict(stage='starting_cpu_renderer', completed=0))
    try:
        controller = BoundedController(
            platform=CloudRendering, local_executable_path=str(executable),
            server_class=FifoServer, quality='High', scene=houses[0][1],
            width=640, height=480, fieldOfView=70, snapToGrid=False,
            renderDepthImage=False, renderInstanceSegmentation=False,
            renderSemanticSegmentation=False, renderNormalsImage=False,
            server_start_timeout=180., server_timeout=120.)
        initial_seconds = time.monotonic() - started
        for ordinal, house, original_path, empty_path in houses:
            house_started = time.monotonic()
            event = controller.last_event if ordinal == 0 else controller.reset(scene=house)
            if not event.metadata['lastActionSuccess']:
                raise RuntimeError(event.metadata.get('errorMessage', 'CreateHouse failed'))
            if event.metadata.get('objects'):
                # Structural doors/windows may have metadata; retain it for review.
                publish(output / f'house-{ordinal:05d}-objects.json', event.metadata['objects'])
            agent = house['metadata']['agent']
            for view in range(4):
                requested = dict(position=agent['position'],
                    rotation=dict(x=0, y=(agent['rotation']['y'] + 90 * view) % 360, z=0),
                    horizon=0, standing=True)
                view_started = time.monotonic()
                event = controller.step(action='TeleportFull', **requested, raise_for_failure=True)
                rgb = event.frame
                if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                    raise ValueError('Expected ordinary perspective RGB at 640 by 480')
                if not np.isfinite(rgb).all() or float(rgb.std()) == 0:
                    raise ValueError('Empty or invalid RGB output')
                target = output / f'house-{ordinal:05d}-view-{view:02d}.png'
                incoming = target.with_name(target.name + '.incoming')
                Image.fromarray(rgb).save(incoming, format='PNG')
                os.rename(incoming, target)
                record = dict(house_id=f'procthor10k/train/{ordinal:05d}', split='train',
                    source=dict(path=str(original_path), sha256=sha(original_path)),
                    empty_house=dict(path=str(empty_path), sha256=sha(empty_path)),
                    image=dict(path=str(target), sha256=sha(target)),
                    projection='perspective', width=640, height=480, vertical_fov_degrees=70,
                    requested_pose=requested, actual_agent=event.metadata['agent'],
                    actual_camera_position=event.metadata.get('cameraPosition'),
                    image_std=float(rgb.std()), seconds=time.monotonic() - view_started,
                    room_count=len(house['rooms']), wall_count=len(house['walls']),
                    door_count=len(house.get('doors', [])), window_count=len(house.get('windows', [])),
                    object_list_empty=True, rendered_depth=False, training_admission=False)
                records.append(record)
                publish(output / 'status.json', dict(stage='rendering_rgb', completed=len(records),
                    house=ordinal, elapsed_seconds=time.monotonic() - started))
                print(json.dumps(dict(completed=len(records), house=ordinal, view=view,
                                      seconds=record['seconds'])), flush=True)
            publish(output / f'house-{ordinal:05d}-report.json', dict(
                house_id=f'procthor10k/train/{ordinal:05d}',
                elapsed_seconds=time.monotonic() - house_started,
                records=records[-4:]))
        publish(output / 'report.json', dict(stage='rendered_not_visually_reviewed',
            completed=len(records), houses=len(houses), records=records,
            initial_seconds=initial_seconds, elapsed_seconds=time.monotonic() - started,
            cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_vulkan_icd=str(icd),
            build=build, ai2thor='5.0.0', generator=dict(path=str(Path(__file__)),
            sha256=sha(Path(__file__))), training_admission=False, geometry_loss=False,
            description='Empty structural scenes; original materials, lights, walls, openings retained. '
                        'Four headings at one known valid position per house; RGB-only pilot, not bulk diversity.'))
        publish(output / 'status.json', dict(stage='rendered_not_visually_reviewed', completed=len(records)))
    except Exception as error:
        publish(output / 'status.json', dict(stage='failed', completed=len(records),
            error=type(error).__name__, message=str(error)))
        raise
    finally:
        if controller is not None:
            controller.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve())
