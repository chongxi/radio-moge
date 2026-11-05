"""Generate bounded, house-separated empty-room RGB with distributed perspective poses."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from render_procthor_empty_pilot import create_controller, publish, sha


TRAIN_SHA = 'ee3c4aa14b4d8f0895fecfb5fdaca59395427ca1018b2f9aeeedbc61e5824587'


def inside(point, polygon):
    x, z = point
    found = False
    for (ax, az), (bx, bz) in zip(polygon, polygon[1:] + polygon[:1]):
        if (az > z) != (bz > z) and x < (bx - ax) * (z - az) / (bz - az) + ax:
            found = not found
    return found


def clearance(point, polygon):
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(polygon, dtype=np.float64)
    b = np.roll(a, -1, axis=0)
    v = b - a
    denominator = (v * v).sum(axis=1)
    t = np.clip(((p - a) * v).sum(axis=1) / np.maximum(denominator, 1e-20), 0, 1)
    return float(np.linalg.norm(p - a - t[:, None] * v, axis=1).min())


def room_positions(room, wall_margin=0.5):
    polygon = [(float(p['x']), float(p['z'])) for p in room['floorPolygon']]
    xs, zs = zip(*polygon)
    candidates = []
    for x in np.arange(min(xs) + 0.25, max(xs), 0.5):
        for z in np.arange(min(zs) + 0.25, max(zs), 0.5):
            point = (float(x), float(z))
            if inside(point, polygon):
                distance = clearance(point, polygon)
                if distance >= wall_margin:
                    candidates.append((point, distance))
    if not candidates:
        return []
    chosen = [max(candidates, key=lambda v: (v[1], v[0]))[0]]
    for _ in range(3):
        other = [(p, min(math.dist(p, q) for q in chosen)) for p, _ in candidates if p not in chosen]
        if not other:
            break
        point, distance = max(other, key=lambda v: (v[1], v[0]))
        if distance < 0.5:
            break
        chosen.append(point)
    return chosen


def camera_slots(house, count):
    rooms, omitted = [], []
    for room in sorted(house['rooms'], key=lambda r: r['id']):
        positions = room_positions(room)
        if positions:
            rooms.append((room, positions))
        else:
            omitted.append(dict(room=room['id'], reason='No camera grid point at least 0.5m inside floor boundary'))
    slots = []
    for index in range(count):
        if not rooms:
            break
        room, points = rooms[index % len(rooms)]
        cycle = index // len(rooms)
        point = points[cycle % len(points)]
        center = points[0]
        yaw = (math.degrees(math.atan2(center[0] - point[0], center[1] - point[1]))
               if point != center else 45.)
        yaw = (yaw + 45 * (cycle // len(points))) % 360
        slots.append(dict(room=room['id'], index=index,
            position=dict(x=point[0], y=float(house['metadata']['agent']['position']['y'])
                          + float(room['floorPolygon'][0]['y']), z=point[1]),
            rotation=dict(x=0, y=yaw, z=0),
            horizon=(0, 10, -10)[(cycle // len(points) + cycle % len(points)) % 3], standing=True))
    return slots, omitted


def houses_from(archive, start, count):
    with gzip.open(archive, 'rt') as stream:
        for ordinal, line in enumerate(stream):
            if ordinal < start:
                continue
            if ordinal >= start + count:
                break
            yield ordinal, json.loads(line), hashlib.sha256(line.encode()).hexdigest()


def run(root, output, start, count, views, maximum_seconds):
    if not 0 <= start < 10000 or not 1 <= count <= 500 or start + count > 10000:
        raise ValueError('Use a bounded batch within the official 10000 training houses')
    if views != 20 or not 60 <= maximum_seconds <= 7200:
        raise ValueError('This version pins 20 views per house and at most two hours')
    archive = root / 'procthor-source-01/train.jsonl.gz'
    if sha(archive) != TRAIN_SHA:
        raise ValueError('Official pinned source archive changed')
    output.mkdir(exist_ok=False)
    for name in ('rgb', 'houses', 'house-reports'):
        (output / name).mkdir()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    started = time.monotonic()
    controller = None
    totals = dict(houses_closed=0, completed=0, failed_house_actions=0,
                  failed_camera_actions=0, uniform_frames=0, duplicate_frames=0)
    next_house = start
    seen, house_summaries = set(), []
    manifest = output / 'rgb-candidates.jsonl'
    publish(output / 'status.json', dict(stage='starting_cpu_renderer', **totals))
    try:
        with manifest.open('x') as records:
            for ordinal, original, line_hash in houses_from(archive, start, count):
                if (output / 'STOP_AFTER_HOUSE').exists() or time.monotonic() - started >= maximum_seconds:
                    break
                house_started = time.monotonic()
                house = dict(original, objects=[])
                house_path = output / 'houses' / f'train-{ordinal:05d}.json'
                publish(house_path, house)
                fov = (55, 70, 85)[ordinal % 3]
                if controller is None:
                    controller = create_controller(root, output, house, field_of_view=fov)
                    event = controller.last_event
                else:
                    event = controller.reset(scene=house, fieldOfView=fov)
                if controller.initialization_parameters.get('makeAgentsVisible') is not False:
                    raise ValueError('Simulator agent must stay hidden after every house reset')
                summary = dict(house_id=f'procthor10k/train/{ordinal:05d}',
                    archive_sha256=TRAIN_SHA, source_line_sha256=line_hash,
                    empty_house_sha256=sha(house_path), original_objects=len(original.get('objects', [])),
                    rooms=len(house['rooms']), vertical_fov_degrees=fov, images=[], failed_actions=[],
                    uniform_slots=[], duplicate_slots=[])
                if not event.metadata['lastActionSuccess']:
                    totals['failed_house_actions'] += 1
                    summary['create_house_error'] = event.metadata.get('errorMessage')
                else:
                    slots, omitted = camera_slots(house, views * 3)
                    summary['rooms_without_safe_camera_grid'] = omitted
                    for slot in slots:
                        if len(summary['images']) >= views:
                            break
                        pose = {k: slot[k] for k in ('position', 'rotation', 'horizon', 'standing')}
                        event = controller.step(action='TeleportFull', **pose)
                        if not event.metadata['lastActionSuccess']:
                            totals['failed_camera_actions'] += 1
                            summary['failed_actions'].append(dict(slot=slot,
                                message=event.metadata.get('errorMessage')))
                            continue
                        rgb = event.frame
                        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                            raise ValueError('Unexpected RGB framebuffer')
                        spatial_std = float(rgb.astype(np.float32).std(axis=(0, 1)).mean())
                        if spatial_std < 0.3:
                            totals['uniform_frames'] += 1
                            summary['uniform_slots'].append(slot)
                            continue
                        pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                        if pixel_hash in seen:
                            totals['duplicate_frames'] += 1
                            summary['duplicate_slots'].append(slot)
                            continue
                        seen.add(pixel_hash)
                        image = output / 'rgb' / f'train-{ordinal:05d}-slot-{slot["index"]:02d}.png'
                        incoming = image.with_suffix('.incoming')
                        Image.fromarray(rgb).save(incoming, format='PNG')
                        os.rename(incoming, image)
                        row = dict(image=str(image), sha256=sha(image), decoded_rgb_sha256=pixel_hash,
                            dataset='procthor_empty', split='train', group=summary['house_id'],
                            width=640, height=480, projection='perspective', vertical_fov_degrees=fov,
                            source_archive_sha256=TRAIN_SHA, source_line_sha256=line_hash,
                            house=dict(path=str(house_path), sha256=summary['empty_house_sha256']),
                            room=slot['room'], requested_pose=pose,
                            actual_agent=event.metadata['agent'],
                            actual_camera_position=event.metadata.get('cameraPosition'),
                            spatial_rgb_std=spatial_std, agent_visible=False,
                            geometry_label=None, training_admission=False)
                        records.write(json.dumps(row, allow_nan=False) + '\n')
                        summary['images'].append(dict(image=str(image), sha256=row['sha256'],
                                                      room=slot['room'], slot=slot['index']))
                        totals['completed'] += 1
                records.flush()
                os.fsync(records.fileno())
                summary['elapsed_seconds'] = time.monotonic() - house_started
                publish(output / 'house-reports' / f'train-{ordinal:05d}.json', summary)
                house_summaries.append(dict(ordinal=ordinal, images=len(summary['images']),
                    rooms=summary['rooms'], elapsed_seconds=summary['elapsed_seconds']))
                totals['houses_closed'] += 1
                next_house = ordinal + 1
                status = dict(stage='rendering_rgb_candidates', next_house=next_house,
                    elapsed_seconds=time.monotonic() - started, **totals)
                publish(output / 'status.json', status)
                print(json.dumps(status), flush=True)
                if totals['failed_house_actions'] >= 5:
                    raise RuntimeError('Five house creation errors; inspect before continuing')
        report = dict(stage='bounded_rgb_batch_rendered_review_pending', **totals,
            start_house=start, requested_houses=count, requested_views_per_house=views,
            next_house=next_house, all_requested_houses_closed=next_house == start + count,
            elapsed_seconds=time.monotonic() - started, maximum_seconds=maximum_seconds,
            manifest=dict(path=str(manifest), sha256=sha(manifest)), houses=house_summaries,
            source_archive_sha256=TRAIN_SHA, generator_sha256=sha(Path(__file__)),
            cpu_only=True, geometry_labels=False, training_admission=False, agent_visible=False,
            camera_selection='0.5m interior floor margin; farthest-spread poses and round-robin rooms; '
                             'engine checks every teleport; 55/70/85-degree pinhole vertical FOV',
            unique_rgb_pixels_in_batch=len(seen), dataset_split='Official training houses only')
        publish(output / 'report.json', report)
        publish(output / 'status.json', dict(stage=report['stage'], next_house=next_house, **totals))
    except Exception as error:
        publish(output / 'status.json', dict(stage='failed_inspect_partial', next_house=next_house,
            **totals, error=type(error).__name__, message=str(error)))
        raise
    finally:
        if controller is not None:
            controller.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-house', type=int, default=0)
    parser.add_argument('--houses', type=int, default=500)
    parser.add_argument('--views', type=int, default=20)
    parser.add_argument('--maximum-seconds', type=int, default=7200)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.start_house,
        args.houses, args.views, args.maximum_seconds)
