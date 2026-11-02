"""Generate pinned MoGe-2 COCO pseudo depth once; never performs optimizer updates.

COCO has no native depth in this corpus. Other datasets are deliberately not
eligible for this script: native-label problems need a separate reviewed routing
decision, not an automatic fallback that hides missing measured coverage.
"""
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import time

from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import torch

from indoor_download import write_json
from radio_moge.geometry_cache import (GeometryPackWriter, canonical_json, publish_json,
                                       read_geometry_member, sha_file)
from radio_moge.geometry_sources import check_point_projection
from radio_moge.upstream import load_moge


def source_rows(root):
    with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
        db.execute('PRAGMA cache_size=-32768')
        for ordinal, image, encoded in db.execute(
                'SELECT ordinal,image,row_json FROM images WHERE dataset=? ORDER BY ordinal',
                ('coco2017',)):
            row = json.loads(encoded)
            if row['split'] != 'train' or row['official_split'] != 'train2017':
                raise ValueError('Non-training COCO row')
            row.update(image=image, corpus_ordinal=ordinal)
            yield row


def choose_probe_rows(root):
    # Stable low-hash selection, retaining landscape, portrait, and square views.
    bins = {'landscape': [], 'portrait': [], 'square': []}
    regression = None
    for row in source_rows(root):
        aspect = row['width'] / row['height']
        group = 'landscape' if aspect > 1.05 else 'portrait' if aspect < .95 else 'square'
        bins[group].append(row)
        if row['source_member'] == 'train2017/000000043393.jpg':
            regression = row
    result = []
    for group in bins:
        result.extend(sorted(bins[group], key=lambda x: x['sha256'])[:4])
    if len(result) != 12:
        raise ValueError('Incomplete aspect-stratified preflight')
    if regression is None or regression in result:
        raise ValueError('Missing or duplicated far-depth numeric regression image')
    result.append(regression)
    return result


def generation_policy(config, checkpoint_hash):
    return dict(schema_version=1, dataset='coco2017', label_kind='teacher_pseudo',
        native_metric_gt=False, teacher=config['teacher'], checkpoint_sha256=checkpoint_hash,
        num_tokens=2500, use_fp16=True, force_projection=True, apply_mask=False,
        support='official teacher mask AND finite positive Z; no filling',
        camera='teacher_estimated_normalized_half_center',
        input='original decoded RGB, original size, no crop/stretch/EXIF transform',
        geometry='float32 camera-Z in teacher nominal meters, NOT calibrated sensor scale',
        projection_check='3D vector-relative error <= 2e-5, not ill-conditioned coordinate-relative error',
        supervision_constraint='no calibrated-metric GT claim or native-hole filling',
        manifest_sha256=config['manifest_sha256'],
        generator_sha256=sha_file(Path(__file__)),
        pack_format_sha256=sha_file(Path(__file__).parents[1] / 'src/radio_moge/geometry_cache.py'),
        geometry_helpers_sha256=sha_file(Path(__file__).parents[1] / 'src/radio_moge/geometry_sources.py'))


def infer_one(model, row):
    data = Path(row['image']).read_bytes()
    if hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('COCO bytes differ from the frozen manifest')
    with Image.open(io.BytesIO(data)) as file:
        if file.getexif().get(274, 1) != 1 or file.size != (row['width'], row['height']):
            raise ValueError('COCO orientation/size changed')
        rgb = np.array(file.convert('RGB'), copy=True)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to('cuda') / 255
    with torch.inference_mode():
        predicted = model.infer(tensor, num_tokens=2500, force_projection=True,
                                apply_mask=False, use_fp16=True)
    depth, k, mask, points = (predicted[key].detach().float().cpu().numpy()
                             for key in ('depth', 'intrinsics', 'mask', 'points'))
    if depth.shape != rgb.shape[:2] or mask.shape != depth.shape or points.shape != rgb.shape:
        raise ValueError('Teacher geometry/image dimensions differ')
    valid = (mask > .5) & np.isfinite(depth) & (depth > 0) & np.isfinite(points).all(-1)
    if not valid.any():
        return None, dict(reason='teacher_has_no_valid_depth', valid_fraction=0.)
    stats = dict(valid_fraction=float(valid.mean()), **check_point_projection(depth, k, points, valid),
                 valid_pixels=int(valid.sum()), total_pixels=int(valid.size))
    return (depth, valid, k), stats


def run(config, mode, attempt, preflight_name):
    root = Path(config['root'])
    inventory = json.loads((root / 'inventory.json').read_text())
    if (inventory['stage'] != 'inventory_complete'
            or inventory['manifest_sha256'] != config['manifest_sha256']
            or inventory['dataset_counts'] != config['expected_counts']):
        raise ValueError('Frozen corpus inventory is not complete')
    if sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']:
        raise ValueError('Inventory bytes changed')
    output = root / attempt
    if output.exists():
        raise FileExistsError('Preserve prior cache attempts; no implicit resume')
    output.mkdir()
    status = dict(stage='loading_teacher', mode=mode, started_at=time.time(), completed=0, rejected=0,
                  training_started=False, optimizer_updates=0, training_admission=False)
    write_json(output / 'status.json', status)
    checkpoint = Path(hf_hub_download(config['teacher']['repo'], 'model.pt',
                                      revision=config['teacher']['revision'], local_files_only=True))
    checkpoint_hash = sha_file(checkpoint)
    policy = generation_policy(config, checkpoint_hash)
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if mode == 'generate':
        preflight = json.loads((root / preflight_name / 'report.json').read_text())
        if (preflight.get('stage') != 'complete' or not preflight.get('passed')
                or preflight['policy_sha256'] != policy_hash):
            raise ValueError('Matching verified teacher preflight required')
    source = Path('vendor/MoGe').resolve()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    if dirty.strip():
        raise ValueError('Pinned upstream MoGe source is dirty')
    torch.set_num_threads(1)
    torch.manual_seed(43)
    model, _ = load_moge(dict(moge_checkpoint=str(checkpoint), moge_source=str(source),
                              moge_repo=config['teacher']['repo'],
                              moge_revision=config['teacher']['revision']), 'cuda')
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Teacher must be frozen in evaluation mode')
    versions = [parameter._version for parameter in model.parameters()]
    rows = choose_probe_rows(root) if mode == 'preflight' else source_rows(root)
    writer = GeometryPackWriter(output / 'packs', policy, max_entries=256)
    assert writer.policy_hash == policy_hash
    examples, rejections = [], []
    try:
        for row in rows:
            start = time.monotonic()
            result, stats = infer_one(model, row)
            if result is None:
                rejections.append(dict(image=row, **stats))
                status['rejected'] += 1
            else:
                depth, valid, k = result
                provenance = dict(image_sha256=row['sha256'], dataset='coco2017',
                    source_id=row['source_member'], label_kind='teacher_pseudo',
                    camera_kind='teacher_estimated', metric_calibrated=False,
                    conversion='pinned_moge2_infer_forced_projection', policy_sha256=policy_hash,
                    teacher_revision=config['teacher']['revision'],
                    source_proofs=[dict(source=config['teacher']['repo'] + '/model.pt',
                                        sha256=checkpoint_hash),
                                   dict(source=row['image'], sha256=row['sha256'])])
                writer.add(row, depth, valid, k, provenance)
                status['completed'] += 1
            if mode == 'preflight':
                examples.append(dict(ordinal=row['corpus_ordinal'], source_id=row['source_member'],
                                     seconds=time.monotonic() - start, **stats))
            if (status['completed'] + status['rejected']) % 64 == 0:
                status.update(stage='generating_cache', updated_at=time.time())
                write_json(output / 'status.json', status)
                print(json.dumps(status), flush=True)
        if versions != [parameter._version for parameter in model.parameters()]:
            raise ValueError('Teacher parameters changed during cache generation')
        writer.close()
        publish_json(output / 'rejections.json', rejections)
        verified = 0
        # Full offline reread, hash, decoding and schema check, not just a write counter.
        for pack in writer.packs:
            receipt = json.loads((output / 'packs' / pack['receipt']).read_text())
            if sha_file(output / 'packs' / pack['receipt']) != pack['sha256']:
                raise ValueError('Completed pack receipt changed')
            if sha_file(output / 'packs' / receipt['tar']) != receipt['sha256']:
                raise ValueError('Completed pack bytes changed')
            for entry in receipt['entries']:
                read_geometry_member(output / 'packs' / receipt['tar'], entry)
                verified += 1
        expected = 13 if mode == 'preflight' else config['expected_counts']['coco2017']
        if status['completed'] + status['rejected'] != expected or verified != status['completed']:
            raise ValueError('Cache did not account for every authorized image')
        status.update(stage='complete', completed_at=time.time(), verified_cache_members=verified,
                      policy_sha256=policy_hash, torch=torch.__version__, cuda=torch.version.cuda,
                      teacher_frozen=True, examples=examples,
                      passed=(mode == 'preflight' and verified == expected and not rejections))
        write_json(output / 'status.json', status)
        publish_json(output / 'report.json', status)
        print(json.dumps({k: v for k, v in status.items() if k != 'examples'}), flush=True)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'generate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--preflight', default='coco-teacher-preflight-02')
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt or Path(args.preflight).name != args.preflight:
        parser.error('attempt must be a new directory basename')
    cfg = json.loads(args.config.read_text())
    lock_path = (Path(cfg['root']).parent / 'data-expansion-attention-20261022'
                 / 'attention-24h/.training.lock')
    if not lock_path.is_file():
        raise FileNotFoundError('Established project GPU lock is missing; do not create another lock')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(cfg, args.mode, args.attempt, args.preflight)
