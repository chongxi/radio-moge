"""Separate Hypersim range agreement from camera-ray agreement, without repairs.

Reads a snapshot of closed receipts and already downloaded native assets only.
The publisher derives range from world position, so norm agreement alone is
not independent evidence of correct image rays. This diagnostic never changes
a label, removes a source-review flag, downloads data, or admits training.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from cache_hypersim_geometry import hdf5
from indoor_download import atomic_write, write_json
from radio_moge.geometry_cache import publish_json, read_geometry_member, sha_file
from radio_moge.geometry_sources import hypersim_pinhole_frame, hypersim_pose_rotation, pinhole_rays
from radio_moge.hypersim_depth import convert_verified_range, half_ulp


def reference(path):
    return dict(path=str(path.resolve()), sha256=sha_file(path))


def snapshot(packs):
    counts, flags, first, worst, controls, receipts = Counter(), Counter(), {}, {}, {}, []
    for path in sorted(packs.glob('pack-*.json')):
        receipts.append(reference(path))
        receipt = json.loads(path.read_text())
        for entry in receipt['entries']:
            group = entry['image']['group']
            counts[group] += 1
            item = dict(receipt=receipts[-1], tar=str(path.parent / receipt['tar']), entry=entry)
            metadata = entry['geometry_metadata']
            if metadata.get('source_needs_review'):
                flags[group] += 1
                first.setdefault(group, item)
                quality = metadata['source_quality']
                fraction = quality['inconsistent_supported_pixels'] / quality['supported_pixels']
                if group not in worst or fraction > worst[group][0]:
                    worst[group] = (fraction, item)
            elif len(controls) < 4:
                controls.setdefault(group, item)
    groups = {group for group, _ in flags.most_common(12)}
    groups |= {group for group in flags if group == 'hypersim:ai_009_003'}
    selected = {}
    for group in sorted(groups):
        for role, item in (('first_flagged', first[group]), ('most_inconsistent', worst[group][1])):
            key = item['entry']['key']
            selected.setdefault(key, dict(**item, selection_roles=[]))['selection_roles'].append(role)
    for item in controls.values():
        selected[item['entry']['key']] = dict(**item, selection_roles=['unflagged_control'])
    if not selected or len(selected) > 30:
        raise ValueError('Expected a nonempty, bounded diagnostic selection')
    return dict(receipts=receipts, receipt_only_counts=dict(counts), receipt_only_flags=dict(flags)), list(selected.values())


def native_arrays(root, entry):
    result, evidence = {}, []
    for proof in entry['geometry_metadata']['source_proofs']:
        if 'member' not in proof:
            continue
        path = root / 'native-source/hypersim' / hashlib.sha256(proof['source'].encode()).hexdigest()[:16] / proof['member']
        if sha_file(path) != proof['sha256']:
            raise ValueError('Downloaded native source changed')
        evidence.append(reference(path))
        if path.suffix == '.hdf5':
            result[path.name] = hdf5(path)
        else:
            with path.open() as file:
                result['scale'] = float(next(x for x in csv.DictReader(file)
                    if x['parameter_name'] == 'meters_per_asset_unit')['parameter_value'])
    return result, evidence


def quantiles(values, support):
    return np.quantile(values[support], [0, .5, .99, 1]).tolist() if support.any() else None


def diagnose(root, item, matrix):
    entry = item['entry']
    cached_depth, cached_valid, cached_k, _ = read_geometry_member(item['tar'], entry)
    native, evidence = native_arrays(root, entry)
    frame = entry['image']['frame_id']
    index = np.flatnonzero(native['camera_keyframe_frame_indices.hdf5'].reshape(-1) == frame).item()
    radial = native[f'frame.{frame:04d}.depth_meters.hdf5']
    world = native[f'frame.{frame:04d}.position.hdf5']
    orientation = native['camera_keyframe_orientations.hdf5'][index]
    position, scale = native['camera_keyframe_positions.hdf5'][index], native['scale']
    depth, valid, k, stats = convert_verified_range(radial, world, orientation, position, scale, matrix)
    if (not np.array_equal(depth, cached_depth) or not np.array_equal(valid, cached_valid)
            or not np.array_equal(k, cached_k) or stats != entry['geometry_metadata']['source_quality']):
        raise ValueError('Native reproduction differs from the immutable candidate')
    _, frame_rotation = hypersim_pinhole_frame(matrix)
    rotation, _ = hypersim_pose_rotation(orientation)
    world_ray = (world.astype(np.float64) - position) * scale
    truth = (world_ray @ rotation * np.array([1, -1, -1])) @ frame_rotation.T
    norm = np.linalg.norm(world_ray, axis=-1)
    radial_error = np.abs(norm - radial.astype(np.float64))
    bound = 2 * (np.linalg.norm(half_ulp(world), axis=-1) * scale + half_ulp(radial)) + 1e-4
    support = np.isfinite(truth).all(-1) & (truth[..., 2] > 0) & np.isfinite(radial) & (radial > 0)
    bad = support & ~valid
    rays = pinhole_rays(k, *radial.shape)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    cosine = (truth * rays).sum(-1) / np.maximum(np.linalg.norm(truth, axis=-1), 1e-30)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    result = dict(image=entry['image'], key=entry['key'], receipt=item['receipt'],
        selection_roles=item['selection_roles'], native_evidence=evidence,
        source_needs_review=entry['geometry_metadata']['source_needs_review'],
        native_reproduction_bitwise_equal=True, existing_quality=stats,
        range_dtype=str(radial.dtype), position_dtype=str(world.dtype),
        inconsistent_fraction=float(bad.mean()),
        norm_consistent_inconsistent_pixels=int((bad & (radial_error <= bound)).sum()),
        inconsistent_pixels=int(bad.sum()),
        norm_error_bad_m=quantiles(radial_error, bad),
        angle_error_bad_degrees=quantiles(angle, bad),
        angle_error_retained_degrees=quantiles(angle, valid),
        source_review_flag_removed=False)
    return result, valid, bad, support, angle


def contact_sheet(output, row, valid, bad, support, angle):
    if sha_file(row['image']['image']) != row['image']['sha256']:
        raise ValueError('Frozen RGB changed')
    with Image.open(row['image']['image']) as file:
        rgb = np.asarray(file.convert('RGB'))
    if rgb.shape[:2] != valid.shape:
        raise ValueError('Native RGB and geometry grids differ')
    overlay = rgb.copy()
    overlay[bad] = (.35 * overlay[bad] + .65 * np.array([255, 0, 0])).astype(np.uint8)
    mask = np.zeros_like(rgb)
    mask[valid] = [0, 200, 0]
    mask[bad] = [255, 0, 0]
    intensity = np.clip(np.nan_to_num(angle) / 5, 0, 1)
    heat = np.stack([intensity, 1 - intensity, np.zeros_like(intensity)], -1)
    heat[~support] = 0
    panels = [rgb, overlay, mask, (heat * 255).astype(np.uint8)]
    image = Image.new('RGB', (1536, 328), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((4, 3), row['image']['source_member'], fill='black')
    labels = ['original RGB', 'red = source XYZ/ray mismatch', 'green retained / red rejected / black unsupported', 'angular error: green 0 deg -> red >=5 deg']
    for i, panel in enumerate(panels):
        draw.text((i * 384 + 4, 20), labels[i], fill='black')
        image.paste(Image.fromarray(panel).resize((384, 288), Image.Resampling.NEAREST), (i * 384, 40))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    atomic_write(output, buffer.getvalue())


def run(config, attempt):
    root = Path(config['root'])
    packs = root / 'hypersim-native-cache-02/packs'
    policy = json.loads((packs / 'policy.json').read_text())
    source = Path(__file__).parents[1]
    for name in ('src/radio_moge/hypersim_depth.py', 'src/radio_moge/geometry_sources.py'):
        if sha_file(source / name) != policy['code_sha256'][name]:
            raise ValueError('Diagnostic must use the exact candidate geometry implementation')
    if policy['manifest_sha256'] != config['manifest_sha256']:
        raise ValueError('Candidate scope differs')
    camera_file = root / 'native-probe-01/hypersim/metadata_camera_parameters.csv'
    if sha_file(camera_file) != policy['camera_metadata_sha256']:
        raise ValueError('Native camera metadata changed')
    with camera_file.open() as file:
        matrices = {row['scene_name']: np.array([[float(row[f'M_cam_from_uv_{i}{j}'])
            for j in range(3)] for i in range(3)]) for row in csv.DictReader(file)}
    output = root / attempt
    output.mkdir(exist_ok=False)
    overview, selected = snapshot(packs)
    publish_json(output / 'snapshot.json', overview)
    report = dict(stage='running', started_at=time.time(), training_admission=False,
        optimizer_updates=0, code=reference(Path(__file__)), policy=reference(packs / 'policy.json'),
        snapshot=reference(output / 'snapshot.json'), selected_images=len(selected), records=[],
        interpretation='Norm agreement checks publisher-derived distance, not independent image-ray alignment.',
        source_review_flags_changed=False, source_assets_changed=False,
        reference='https://github.com/apple/ml-hypersim/blob/main/code/python/tools/generate_hdf5_from_vrimg.py')
    try:
        for index, item in enumerate(selected):
            scene = item['entry']['image']['group'].split(':')[1]
            record, valid, bad, support, angle = diagnose(root, item, matrices[scene])
            path = output / f'{index:03d}.png'
            contact_sheet(path, record, valid, bad, support, angle)
            record['contact_sheet'] = reference(path)
            report['records'].append(record)
            write_json(output / 'status.json', {k: v for k, v in report.items() if k != 'records'}
                       | dict(completed=len(report['records'])))
        report.update(stage='diagnostic_complete', completed_at=time.time())
        publish_json(output / 'report.json', report)
    except BaseException as error:
        report.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', {k: v for k, v in report.items() if k != 'records'})
        raise
    write_json(output / 'status.json', {k: v for k, v in report.items() if k != 'records'}
               | dict(completed=len(report['records'])))
    print(json.dumps(dict(stage=report['stage'], views=len(report['records']),
                         training_admission=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a new attempt basename')
    run(json.loads(args.config.read_text()), args.attempt)
