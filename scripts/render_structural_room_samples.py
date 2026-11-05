"""Small, explicitly labelled camera comparison for closed structural rooms.

Uses the existing pinned ProcTHOR Unity environment and runtime mesh API.
These are new parameterized scenes, not unmodified official ProcTHOR houses.
No depth, model inference, training admission, or bulk generation occurs here.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from render_procthor_empty_pilot import BUILD, create_controller, publish, sha


POLICY = 'closed-structural-room-camera-preview-v1'


def vec(values):
    return dict(zip(('x', 'y', 'z'), map(float, values)))


def box_mesh(size):
    """Closed box, separate face vertices for sharp, planar face normals."""
    half = np.asarray(size, dtype=float) / 2
    if half.shape != (3,) or not np.isfinite(half).all() or (half <= 0).any():
        raise ValueError('Positive finite box dimensions required')
    vertices, normals, triangles, uvs = [], [], [], []
    for axis in range(3):
        for sign in (-1, 1):
            normal = np.eye(3)[axis] * sign
            other = [i for i in range(3) if i != axis]
            face = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = normal * half
                p[other] = [a * half[other[0]], b * half[other[1]]]
                face.append(p)
            if np.dot(np.cross(face[1] - face[0], face[2] - face[0]), normal) < 0:
                face.reverse()
            start = len(vertices)
            vertices.extend(face)
            normals.extend([normal] * 4)
            triangles.extend([start, start + 1, start + 2, start, start + 2, start + 3])
            uvs.extend([dict(x=x, y=y) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))])
    return dict(vertices=list(map(vec, vertices)), normals=list(map(vec, normals)),
                triangles=triangles, uvs=uvs)


def cylinder_mesh(radius, length, axis='y', segments=64):
    """Capped cylinder: smooth sides, flat cap normals, actual curved silhouette."""
    if axis not in ('x', 'y') or segments < 12 or min(radius, length) <= 0:
        raise ValueError('Invalid cylinder parameters')
    vertices, normals, triangles, uvs = [], [], [], []

    def rotate(p):
        x, y, z = p
        return np.asarray((y, -x, z) if axis == 'x' else (x, y, z), dtype=float)

    for i in range(segments + 1):
        angle = i * math.tau / segments
        for y in (-length / 2, length / 2):
            vertices.append(rotate((radius * math.cos(angle), y, radius * math.sin(angle))))
            normals.append(rotate((math.cos(angle), 0, math.sin(angle))))
            uvs.append(dict(x=i / segments, y=float(y > 0)))
    for i in range(segments):
        a = 2 * i
        triangles.extend([a, a + 1, a + 3, a, a + 3, a + 2])
    for sign in (-1, 1):
        center = len(vertices)
        vertices.append(rotate((0, sign * length / 2, 0)))
        normals.append(rotate((0, sign, 0)))
        uvs.append(dict(x=.5, y=.5))
        for i in range(segments):
            angle = i * math.tau / segments
            vertices.append(rotate((radius * math.cos(angle), sign * length / 2,
                                    radius * math.sin(angle))))
            normals.append(rotate((0, sign, 0)))
            uvs.append(dict(x=.5 + .5 * math.cos(angle), y=.5 + .5 * math.sin(angle)))
        for i in range(segments):
            a, b = center + 1 + i, center + 1 + (i + 1) % segments
            triangles.extend([center, b, a] if sign == 1 else [center, a, b])
    return dict(vertices=list(map(vec, vertices)), normals=list(map(vec, normals)),
                triangles=triangles, uvs=uvs)


def material(color, name='PureWhite'):
    return dict(name=name, color=dict(zip(('r', 'g', 'b'), color)),
                metallic=0., smoothness=.1)


def room(variant):
    """Four continuous walls, ceiling, floor; zero windows, doors or external views."""
    palettes = ((.83, .82, .78), (.74, .78, .82), (.87, .81, .72))
    color = palettes[variant % len(palettes)]
    width, depth, height = 10., 14., 4.2
    floor = [vec(p) for p in ((0, 0, 0), (0, 0, depth), (width, 0, depth), (width, 0, 0))]
    walls = []
    for i, (a, b) in enumerate(zip(floor, floor[1:] + floor[:1])):
        walls.append(dict(id=f'wall|preview-{variant}|{i}', roomId='room|0',
            polygon=[copy.deepcopy(a), copy.deepcopy(b), dict(a, y=height), dict(b, y=height)],
            material=material(color, 'WallDrywallWhite3'), thickness=.15))
    pose = dict(position=vec((5, .95, 2)), rotation=vec((0, 0, 0)), horizon=0, standing=True)
    lights = []
    for i, (position, intensity) in enumerate((((2.2, 3.8, 8.), 1.0),
                                              ((7.5, 3.6, 11.5), .7),
                                              ((5., 3.8, 3.), .5))):
        lights.append(dict(id=f'light_{i}', type='point', position=vec(position),
            intensity=intensity, range=18, rgb=dict(r=1., g=.96, b=.92),
            shadow=dict(type='Soft', strength=.8, normalBias=0., bias=.03,
                        nearPlane=.2, resolution='FromQualitySettings')))
    return dict(id=f'structural-preview-{variant:03d}',
        rooms=[dict(id='room|0', roomType='LivingRoom', floorPolygon=floor,
                    floorMaterial=material((.62, .61, .59), 'LightWoodCounters2'), ceilings=[])],
        walls=walls, windows=[], doors=[], objects=[],
        proceduralParameters=dict(ceilingMaterial=material((.84, .84, .82), 'WallDrywallWhite3'),
            # Required by pinned CreateHouse even in a closed room. The six room
            # surfaces hide it; preview cameras additionally use a solid sentinel.
            skyboxId='SkyGarden', floorColliderThickness=1., receptacleHeight=.7,
            reflections=[], lights=lights),
        metadata=dict(schema='1.0.0', agent=pose, agentPoses=dict(default=pose)))


def structures(variant):
    """Grounded pillars and connected spanning members, kept away from camera."""
    z = 10.5 + .4 * variant
    colors = ((.69, .68, .64), (.66, .71, .75), (.76, .70, .61))
    color = colors[variant % len(colors)]
    data = [
        dict(name='square_column', kind='box', size=(.65, 4.2, .65), center=(3.2, 2.1, z)),
        dict(name='round_column', kind='cylinder_y', radius=.36, length=4.2,
             center=(6.8, 2.1, z)),
        dict(name='rectangular_beam', kind='box', size=(4.32, .45, .62),
             center=(5., 3.5, z)),
        dict(name='horizontal_round_member', kind='cylinder_x', radius=.15 + .03 * variant,
             length=3.6, center=(5., 1.25 + .3 * variant, z)),
    ]
    if variant == 1:
        data.append(dict(name='rear_square_column', kind='box', size=(.5, 4.2, .5),
                         center=(5.1, 2.1, 13.2)))
    if variant == 2:
        data.extend([
            dict(name='rear_round_column', kind='cylinder_y', radius=.32, length=4.2,
                 center=(4.5, 2.1, 13.2)),
            dict(name='depth_beam', kind='box', size=(.4, .4, 3.1),
                 center=(6.8, 3.6, 12.1)),
        ])
    return [dict(x, color=color) for x in data]


def camera_requests():
    base = dict(position=vec((5., 2.1, 2.)), rotation=vec((0, 0, 0)),
                fieldOfView=32., nearClippingPlane=.05, farClippingPlane=50.,
                skyboxColor='#ff00ff')
    return [
        ('frontal_far_perspective', dict(base, orthographic=False)),
        ('orthographic_comparison_only', dict(base, orthographic=True, orthographicSize=2.0)),
    ]


def mesh_for(item):
    if item['kind'] == 'box':
        return box_mesh(item['size'])
    return cylinder_mesh(item['radius'], item['length'], item['kind'][-1])


def save_image(path, array):
    incoming = path.with_name(path.name + '.incoming')
    Image.fromarray(array).save(incoming, format='PNG')
    incoming.rename(path)


def render(root, output):
    output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    controller, records = None, []
    publish(output / 'status.json', dict(stage='starting', policy=POLICY))
    try:
        for variant in range(3):
            house = room(variant)
            house_path = output / f'room-{variant:02d}.json'
            publish(house_path, house)
            if controller is None:
                controller = create_controller(root, output, house, field_of_view=32)
                event = controller.last_event
            else:
                event = controller.reset(scene=house, fieldOfView=32)
            if not event.metadata['lastActionSuccess']:
                raise RuntimeError(event.metadata.get('errorMessage', 'CreateHouse failed'))
            if controller.initialization_parameters.get('makeAgentsVisible') is not False:
                raise ValueError('Simulator agent must be hidden')
            object_records = []
            for item in structures(variant):
                mesh = mesh_for(item)
                texture = output / f'{variant:02d}-{item["name"]}-material.png'
                texture_rgb = np.broadcast_to(np.rint(np.asarray(item['color']) * 255)
                                               .astype(np.uint8), (8, 8, 3)).copy()
                save_image(texture, texture_rgb)
                name = f'structural_preview_{variant}_{item["name"]}'
                points = np.asarray([list(v.values()) for v in mesh['vertices']])
                collider = box_mesh(points.max(axis=0) - points.min(axis=0))
                request = dict(action='CreateObjectPrefab', name=name, **mesh,
                    texturePath=str(texture), visibilityPoints=[vec((0, 0, 0))],
                    colliders=[{k: collider[k] for k in ('vertices', 'triangles')}],
                    physicalProperties=dict(useGravity=False, isKinematic=True),
                    annotations=dict(objectType='Undefined', primaryProperty='Undefined',
                                     secondaryProperties=[]))
                mesh_path = output / f'{name}.json'
                publish(mesh_path, dict(specification=item, action=request))
                controller.step(**request, raise_for_failure=True)
                spawn = controller.step(action='SpawnAsset', assetId=name, generatedId=name,
                    position=vec(item['center']), rotation=vec((0, 0, 0)), raise_for_failure=True)
                object_records.append(dict(name=name, specification=item,
                    mesh=dict(path=str(mesh_path), sha256=sha(mesh_path)),
                    texture=dict(path=str(texture), sha256=sha(texture)),
                    spawn_result=spawn.metadata.get('actionReturn')))
            for view, (label, request) in enumerate(camera_requests()):
                if view == 0:
                    event = controller.step(action='AddThirdPartyCamera', **request,
                                             raise_for_failure=True)
                else:
                    event = controller.step(action='UpdateThirdPartyCamera', thirdPartyCameraId=0,
                                             **request, raise_for_failure=True)
                rgb = event.third_party_camera_frames[0]
                if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or rgb.std() < 1:
                    raise ValueError('Invalid structural RGB frame')
                # Closed room should entirely hide the deliberately magenta background.
                background = (rgb[..., 0] > 240) & (rgb[..., 1] < 15) & (rgb[..., 2] > 240)
                if background.any():
                    save_image(output / f'invalid-exterior-{variant}-{view}.png', rgb)
                    raise ValueError('Exterior/background visible')
                path = output / f'room-{variant:02d}-{label}.png'
                save_image(path, rgb)
                record = dict(scene=variant, view=label, policy=POLICY,
                    source_role='custom_parameterized_preview_not_official_house',
                    image=dict(path=str(path), sha256=sha(path),
                               pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()),
                    house=dict(path=str(house_path), sha256=sha(house_path)),
                    objects=object_records, requested_camera=request,
                    actual_cameras=event.metadata.get('thirdPartyCameras'),
                    wall_distance_m=12., nearest_structure_distance_m=float(
                        min(x['center'][2] for x in structures(variant)) - 2. - .36),
                    window_count=0, door_count=0, magenta_background_pixels=int(background.sum()),
                    agent_visible=False, rgb_only=True, geometry_labels=None,
                    training_admission=False, projection_decision_pending=True)
                records.append(record)
                publish(output / 'status.json', dict(stage='rendering', completed=len(records),
                    elapsed_seconds=time.monotonic() - started))
                print(json.dumps(dict(completed=len(records), scene=variant, view=label)), flush=True)
            publish(output / f'room-{variant:02d}-runtime.json', event.metadata)
        if len({r['image']['pixel_sha256'] for r in records}) != len(records):
            raise ValueError('Duplicate sample pixels')
        publish(output / 'report.json', dict(stage='rendered_not_visually_reviewed',
            policy=POLICY, renderer_build=BUILD, completed=len(records), scenes=3,
            records=records, elapsed_seconds=time.monotonic() - started,
            geometry_loss=False, training_admission=False))
    finally:
        if controller is not None:
            controller.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.output)
