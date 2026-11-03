"""Real three-way sanity views from the frozen first-stage pack snapshot.

Use the already resident Gradio inference service, serially. No model training,
source acceptance, changed labels or evaluation-set depth is involved.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

import numpy as np
from PIL import Image

from indoor_download import write_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, read_geometry_member
from radio_moge.geometry_sampling import camera_matrix, centered_view, select_observations, warp_rgb


def select_rows(database, per_source=6):
    groups, flagged = defaultdict(dict), {}
    with sqlite3.connect(database) as db:
        db.execute('PRAGMA cache_size=-32768')
        for values in db.execute('SELECT i.dataset,i.scene_group,i.ordinal,i.sha256,c.entry_key,'
                'c.pack_id,c.native_flag,c.external_flag,c.excluded,c.label_kind FROM candidates c '
                'JOIN images i ON i.sha256=c.image_sha256'):
            dataset, group, ordinal, checksum, key, pack, native, external, excluded, kind = values
            item = dict(dataset=dataset, group=group, ordinal=ordinal, sha256=checksum, key=key,
                        pack_id=pack, native_flag=bool(native), external_flag=bool(external),
                        excluded=bool(excluded), label_kind=kind)
            if native or external or excluded:
                if dataset not in flagged or checksum < flagged[dataset]['sha256']:
                    flagged[dataset] = item
            else:
                group_key = (kind, group)
                if group_key not in groups[dataset] or checksum < groups[dataset][group_key]['sha256']:
                    groups[dataset][group_key] = item
        selected = []
        for dataset in sorted(groups):
            rows = list(groups[dataset].values())
            kinds = sorted({r['label_kind'] for r in rows})
            chosen = []
            for kind in kinds:
                chosen.append(min((r for r in rows if r['label_kind'] == kind),
                    key=lambda r: hashlib.sha256(r['group'].encode()).hexdigest()))
            order = sorted(rows, key=lambda r: hashlib.sha256(r['group'].encode()).hexdigest())
            for item in order:
                if len(chosen) >= per_source:
                    break
                if item['ordinal'] not in {r['ordinal'] for r in chosen}:
                    chosen.append(item)
            if len(chosen) != per_source:
                raise ValueError('Not enough independent frozen groups for bounded selection')
            for item in chosen:
                item['selection'] = 'deterministic distinct scene groups; selected before inference'
                selected.append(item)
            if dataset in flagged:
                item = flagged[dataset]
                item['selection'] = 'additional explicitly quarantined/excluded diagnostic'
                selected.append(item)
        for item in selected:
            source, receipt, tar = db.execute('SELECT source,receipt_json,tar_json FROM packs WHERE id=?',
                                            (item['pack_id'],)).fetchone()
            item.update(source=source, receipt=json.loads(receipt), tar=json.loads(tar))
    return selected


def prepare(entry):
    """Display flagged candidates without clearing flags or calling a training loader."""
    depth, valid, kd, metadata = read_geometry_member(entry['pack'], entry['entry'])
    row = entry['entry']['image']
    original = checked_path(dict(path=row['image'], sha256=row['sha256'])).read_bytes()
    with Image.open(io.BytesIO(original)) as photo:
        if photo.getexif().get(274, 1) != 1:
            raise ValueError('Unexpected EXIF orientation')
        photo = photo.convert('RGB')
    rotation = row.get('image_rotation_cw', 0)
    if rotation != metadata.get('image_rotation_cw', 0) or rotation not in (0, 90, 180, 270):
        raise ValueError('Unbound image orientation')
    if rotation:
        photo = photo.transpose({90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
                                270: Image.Transpose.ROTATE_90}[rotation])
    rgb = np.array(photo)
    if 'height' in row and tuple(rgb.shape[:2]) != (row['height'], row['width']):
        raise ValueError('Original RGB dimensions changed')
    if 'rgb_intrinsics_normalized' in metadata:
        kr = camera_matrix(metadata['rgb_intrinsics_normalized'])
        if (metadata['rgb_shape'] != list(rgb.shape[:2])
                or not np.allclose(metadata['depth_to_rgb_uv_transform'], kr @ np.linalg.inv(kd),
                                   rtol=0, atol=1e-8)):
            raise ValueError('Native depth/RGB calibration mapping changed')
    else:
        if depth.shape != rgb.shape[:2]:
            raise ValueError('Different grids require explicit calibration')
        kr = camera_matrix(kd)
    shape = (640, 480) if rgb.shape[0] > rgb.shape[1] else (400, 640)
    view = centered_view(kr, rgb.shape[:2], shape)
    target = select_observations(depth, valid, kd, view)
    rgb = warp_rgb(rgb, view)
    points = target['points'][target['valid']].astype(np.float64)
    uv = target['query_uv'][target['valid']].astype(np.float64)
    projected = points @ view['intrinsics'].T
    error = float(np.max(np.abs(projected[:, :2] / projected[:, 2:] - uv)))
    if error > 1e-6 or len(points) < 32:
        raise ValueError('Reference projection or support failed sanity preflight')
    return rgb, points, uv, view, metadata, error


def bilinear(array, uv):
    h, w = array.shape[:2]
    xy = np.clip(uv * [w, h] - .5, [0, 0], [w - 1, h - 1])
    lo = np.floor(xy).astype(int)
    hi = np.minimum(lo + 1, [w - 1, h - 1])
    t = xy - lo
    if array.ndim == 3:
        t = t[..., None]
    tx, ty = t[:, 0], t[:, 1]
    # Prediction interpolation only. Native observations are never interpolated.
    return ((1 - ty) * ((1 - tx) * array[lo[:, 1], lo[:, 0]] + tx * array[lo[:, 1], hi[:, 0]])
            + ty * ((1 - tx) * array[hi[:, 1], lo[:, 0]] + tx * array[hi[:, 1], hi[:, 0]]))


def encode_cloud(points, colors, max_points=30000):
    good = np.isfinite(points).all(-1) & (points[:, 2] > 0)
    points, colors = points[good], colors[good]
    full_count = len(points)
    ids = np.linspace(0, full_count - 1, min(max_points, full_count), dtype=int)
    points, colors = points[ids], colors[ids]
    raw = np.asarray(points, dtype='<f4').tobytes() + np.asarray(colors, dtype=np.uint8).tobytes()
    return raw, dict(count=len(points), full_count=full_count,
        median_z=float(np.median(points[:, 2])),
        bounds=np.quantile(points, [.02, .98], axis=0).tolist())


def export(snapshot_path, output, api):
    from gradio_client import Client, handle_file

    snapshot_ref = file_ref(snapshot_path)
    report = checked_json(snapshot_ref)
    if report['stage'] != 'candidate_snapshot_metadata_complete_not_accepted' or report['processed_packs'] != 4098:
        raise ValueError('Use the requested fixed 4098-pack snapshot')
    output.mkdir(exist_ok=False)
    public = output / 'public'
    public.mkdir()
    (output / 'arrays').mkdir()
    with tempfile.TemporaryDirectory(prefix='stage1-sanity-') as folder:
        local_db = Path(folder) / 'candidates.sqlite'
        shutil.copyfile(checked_path(report['index']), local_db)
        selected = select_rows(local_db)
        if len({r['dataset'] for r in selected}) != 8:
            raise ValueError('All eight original sources must be represented')
        publish_json(output / 'selection.json', dict(snapshot=snapshot_ref, selected=selected,
            selection_before_inference=True, training_admission=False, optimizer_updates=0))
        state = dict(stage='exporting_sanity_views', completed=0, total=len(selected),
            started_at=time.time(), optimizer_updates=0, training_admission=False)
        write_json(output / 'status.json', state)
        client = Client(api, verbose=False)
        samples = []
        for item in selected:
            receipt = checked_json(item['receipt'])
            matches = [e for e in receipt['entries'] if e['key'] == item['key']]
            if len(matches) != 1 or matches[0]['image']['sha256'] != item['sha256']:
                raise ValueError('Selected snapshot member changed')
            entry = matches[0]
            pack = Path(item['tar']['path'])
            if pack.stat().st_size != item['tar']['bytes']:
                raise ValueError('Published selected pack size changed')
            rgb, gt, uv, view, meta, error = prepare(dict(pack=pack, entry=entry))
            sample_id = f"{item['dataset']}-{item['ordinal']}"
            input_path = Path(folder) / 'input.png'
            Image.fromarray(rgb).save(input_path)
            h, w = rgb.shape[:2]
            response = client.predict(handle_file(str(input_path)), f'{h}×{w}', api_name='/compare')
            info = response[4]
            if info['adapter_step'] != 232288 or info['processed_hw'] != [h, w]:
                raise ValueError('Inference checkpoint or image shape differs')
            with np.load(response[5], allow_pickle=False) as arrays:
                results = {k: arrays[k].copy() for k in arrays.files}
            if not np.array_equal(results['rgb'], rgb):
                raise ValueError('Inference service recropped or changed the prepared RGB')
            np.savez_compressed(output / 'arrays' / f'{sample_id}.npz', **results,
                reference_points=gt, reference_uv=uv, reference_intrinsics=view['intrinsics'])
            Image.fromarray(rgb).save(public / f'{sample_id}.jpg', quality=94)
            pix = np.minimum((uv * [w, h]).astype(int), [w - 1, h - 1])
            gt_colors = rgb[pix[:, 1], pix[:, 0]]
            clouds, metrics = {}, {}
            for name in ('reference', 'moge2', 'adapter'):
                if name == 'reference':
                    full_points, full_colors, fixed_points = gt, gt_colors, gt
                else:
                    points = results[name + '_points']
                    depth = results[name + '_depth']
                    mask = results[name + '_mask'].astype(bool)
                    k = results[name + '_intrinsics']
                    yy, xx = np.meshgrid((np.arange(h) + .5) / h,
                                         (np.arange(w) + .5) / w, indexing='ij')
                    expected = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(k).T
                    expected *= depth[..., None]
                    valid = mask & np.isfinite(points).all(-1) & (points[..., 2] > 0)
                    if not valid.any() or not np.allclose(expected[valid], points[valid], rtol=3e-5, atol=3e-5):
                        raise ValueError('Model point/depth/camera consistency failed')
                    full_points, full_colors = points[valid], rgb[valid]
                    fixed_points = bilinear(points, uv)
                    z = fixed_points[:, 2]
                    good = np.isfinite(z) & (z > 0)
                    ratio = float(np.median(gt[good, 2]) / np.median(z[good]))
                    metrics[name] = dict(reference_observations=len(gt), finite_positive=int(good.sum()),
                        raw_abs_rel=float(np.mean(np.abs(z[good] - gt[good, 2]) / gt[good, 2])),
                        raw_rmse=float(np.sqrt(np.mean((z[good] - gt[good, 2])**2))),
                        median_scale_gt_over_prediction=ratio,
                        scale_only_abs_rel=float(np.mean(np.abs(ratio * z[good] - gt[good, 2]) / gt[good, 2])))
                clouds[name] = {}
                for mode, points, colors in [('full', full_points, full_colors),
                                              ('reference', fixed_points, gt_colors)]:
                    raw, stats = encode_cloud(points, colors)
                    filename = f'{sample_id}-{name}-{mode}.bin.gz'
                    with (public / filename).open('xb') as file:
                        file.write(gzip.compress(raw, compresslevel=4, mtime=0))
                    clouds[name][mode] = dict(file=filename, **stats)
            sample = dict(id=sample_id, ordinal=item['ordinal'], dataset=item['dataset'],
                group=item['group'], label_kind=item['label_kind'], metric_calibrated=meta['metric_calibrated'],
                native_flag=item['native_flag'], external_flag=item['external_flag'], excluded=item['excluded'],
                selection=item['selection'], image=f'{sample_id}.jpg', width=w, height=h,
                rgb_sha256=item['sha256'], member_sha256=entry['sha256'], receipt_sha256=item['receipt']['sha256'],
                geometry_key=item['key'], policy_sha256=meta['policy_sha256'],
                clouds=clouds, metrics=metrics, reference_projection_error=error,
                inference=info, reference_camera=view['intrinsics'].tolist(),
                adapter_camera=results['adapter_intrinsics'].tolist(), moge2_camera=results['moge2_intrinsics'].tolist())
            publish_json(output / 'arrays' / f'{sample_id}.json', dict(sample=sample, source=item,
                member_payload_verified=True, original_rgb_hash_verified=True,
                exact_input_rgb_parity=True, training_admission=False, optimizer_updates=0))
            samples.append(sample)
            state.update(completed=len(samples), updated_at=time.time(), last=sample_id)
            write_json(output / 'status.json', state)
            print(json.dumps(state), flush=True)
        display = dict(stage='complete', snapshot_sha256=snapshot_ref['sha256'],
            snapshot_packs=4098, snapshot_rgb=report['unique_rgb'], adapter_step=232288,
            sources=sorted({r['dataset'] for r in samples}), samples=samples,
            geometry_training_updates=0, diagnostic_not_acceptance=True)
        publish_json(public / 'manifest.json', display)
        checked_path(snapshot_ref)
        state.update(stage='complete', finished_at=time.time(), samples=len(samples),
            manifest=file_ref(public / 'manifest.json'), snapshot=snapshot_ref,
            selection=file_ref(output / 'selection.json'), code=file_ref(Path(__file__)),
            exact_input_rgb_parity=True, selected_member_payloads_verified=True)
        publish_json(output / 'report.json', state)
        write_json(output / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--api', default='http://127.0.0.1:7861')
    args = parser.parse_args()
    export(args.snapshot, args.output, args.api)
