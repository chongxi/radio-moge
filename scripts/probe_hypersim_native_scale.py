"""Read retained native assets behind specific Hypersim scale-review findings.

This records source units and reproduces existing conversion without changing
labels or interpreting renderer consistency as independent physical calibration.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from audit_completed_rgb_teachers import rgb
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json, read_geometry_member
from radio_moge.hypersim_depth import convert_verified_range


def run(root, output):
    audit_ref = file_ref(root / 'hypersim-outcomes-01/report.json')
    audit = checked_json(audit_ref)
    receipts, rows = {}, []
    with readonly_db(checked_path(audit['index'])) as db:
        records = list(db.execute('SELECT ordinal,row_json,outcome,evidence_json,entry_key FROM images '
            'WHERE scene_group=? OR ordinal IN (?,?) ORDER BY ordinal',
            ('hypersim:ai_013_001', 210303, 251659)))
    if len(records) != 202 or any(r[2] != 'unflagged_cache' for r in records):
        raise ValueError('The specific scale review scope changed')
    output.mkdir(exist_ok=False)
    selected = {210303, 211112, 211162, 211211, 211212, 211262, 211311, 251659}
    scopes, probes = defaultdict(list), []
    matrix_ref = file_ref(root / 'native-probe-01/hypersim/metadata_camera_parameters.csv')
    matrices = {r['scene_name']: np.array([[float(r[f'M_cam_from_uv_{i}{j}'])
        for j in range(3)] for i in range(3)]) for r in csv.DictReader(checked_path(matrix_ref).open())}
    for ordinal, encoded, _, receipt_json, key in records:
        receipt_ref = json.loads(receipt_json)
        if receipt_ref['path'] not in receipts:
            receipt = checked_json(receipt_ref)
            receipts[receipt_ref['path']] = receipt, {e['key']: e for e in receipt['entries']}
        receipt, entries = receipts[receipt_ref['path']]
        entry = entries[key]
        row, meta = entry['image'], entry['geometry_metadata']
        original = dict(row)
        original.pop('corpus_ordinal', None)
        if (original != json.loads(encoded) or meta.get('source_needs_review')
                or hashlib.sha256(canonical_json(meta)).hexdigest() != key):
            raise ValueError('Frozen native identity or flag differs')
        record = dict(corpus_ordinal=ordinal, image=row['image'], sha256=row['sha256'],
            group=row['group'], camera=row['camera'], frame_id=row['frame_id'],
            entry_key=key, member_sha256=entry['sha256'], receipt=receipt_ref,
            depth_quantiles_m=entry['depth_quantiles_m'], valid_fraction=entry['valid_fraction'])
        rows.append(record)
        scopes[(row['group'], row['camera'])].append(record)
        if ordinal not in selected:
            continue
        assets, proofs = {}, []
        for proof in meta['source_proofs']:
            if 'member' not in proof:
                continue
            path = root / 'native-source/hypersim' / hashlib.sha256(
                proof['source'].encode()).hexdigest()[:16] / proof['member']
            checked_path(dict(path=str(path), sha256=proof['sha256']))
            proofs.append(dict(path=str(path), sha256=proof['sha256']))
            name = path.name
            if name.endswith('.hdf5'):
                with h5py.File(path) as file:
                    assets[name] = np.asarray(file['dataset'])
            elif name == 'metadata_scene.csv':
                assets['scale'] = float(next(x for x in csv.DictReader(io.StringIO(path.read_text()))
                    if x['parameter_name'] == 'meters_per_asset_unit')['parameter_value'])
        index, = np.flatnonzero(assets['camera_keyframe_frame_indices.hdf5'].reshape(-1) == row['frame_id'])
        radial = assets[f'frame.{row["frame_id"]:04d}.depth_meters.hdf5']
        world = assets[f'frame.{row["frame_id"]:04d}.position.hdf5']
        position = assets['camera_keyframe_positions.hdf5'][index]
        orientation = assets['camera_keyframe_orientations.hdf5'][index]
        depth, valid, k, stats = convert_verified_range(radial, world, orientation, position,
            assets['scale'], matrices[row['group'].split(':')[1]])
        pack = Path(receipt_ref['path']).parent / receipt['tar']
        old_depth, old_valid, old_k, _ = read_geometry_member(pack, entry)
        np.testing.assert_array_equal(depth, old_depth)
        np.testing.assert_array_equal(valid, old_valid)
        np.testing.assert_allclose(k, old_k, rtol=1e-7, atol=1e-7)
        image = rgb(row)
        if image.size != (depth.shape[1], depth.shape[0]):
            raise ValueError('Original RGB dimensions differ')
        positive = np.isfinite(radial) & (radial > 0)
        derived = np.linalg.norm(world.astype(np.float64) - position, axis=-1) * assets['scale']
        support = positive & np.isfinite(derived)
        info = dict(record, meters_per_asset_unit=assets['scale'], source_assets=proofs,
            radial_m_quantiles=np.quantile(radial[positive].astype(float), [.01, .5, .99]).tolist(),
            world_distance_m_quantiles=np.quantile(derived[support], [.01, .5, .99]).tolist(),
            raw_norm_relative_error_quantiles=np.quantile(
                np.abs(derived[support] - radial[support]) / radial[support], [.5, .99, 1]).tolist(),
            source_dtypes=dict(radial=str(radial.dtype), world=str(world.dtype)),
            reproduced_cached_depth_and_mask_exactly=True, source_quality=stats)
        page = Image.new('RGB', (image.width, image.height + 64), 'white')
        draw = ImageDraw.Draw(page)
        for y, text in enumerate((f'corpus {ordinal} {row["group"]} {row["camera"]} frame {row["frame_id"]}',
                f'Original RGB; source meters/asset={assets["scale"]:.10g}; no fitted scale or geometry changes',
                f'Cached Z median={entry["depth_quantiles_m"][2]:.6g}m; raw radial median={info["radial_m_quantiles"][1]:.6g}m')):
            draw.text((4, 3 + 18 * y), text, fill='black')
        page.paste(image, (0, 64))
        path = output / f'{ordinal}.png'
        temporary = path.with_suffix('.png.incoming')
        page.save(temporary, format='PNG')
        os.rename(temporary, path)
        info['page'] = file_ref(path)
        probes.append(info)
    incoming = output / 'scope.jsonl.incoming'
    with incoming.open('x') as file:
        for row in rows:
            file.write(canonical_json(row).decode() + '\n')
        file.flush()
        os.fsync(file.fileno())
    os.rename(incoming, output / 'scope.jsonl')
    summary = [dict(group=g, camera=c, images=len(v),
        median_depth_m_range=[min(x['depth_quantiles_m'][2] for x in v), max(x['depth_quantiles_m'][2] for x in v)],
        p99_depth_m_range=[min(x['depth_quantiles_m'][3] for x in v), max(x['depth_quantiles_m'][3] for x in v)])
        for (g, c), v in sorted(scopes.items())]
    checked_path(audit_ref)
    publish_json(output / 'report.json', dict(stage='native_scale_probe_complete_not_accepted',
        source_outcomes=audit_ref, matrix=matrix_ref, scope=file_ref(output / 'scope.jsonl'),
        scope_images=len(rows), camera_statistics=summary, probes=probes,
        observation='Source radial distances and world positions share renderer provenance. Agreement checks conversion/storage consistency, not independent physical scale.',
        native_flags_cleared=False, source_quality_accepted=False, geometry_changed=False,
        training_admission=False, optimizer_updates=0, code=file_ref(Path(__file__))))
    print(json.dumps(dict(scope_images=len(rows), probes=len(probes), camera_statistics=summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
