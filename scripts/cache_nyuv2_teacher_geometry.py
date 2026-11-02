"""Separate, nonmetric NYU teacher candidates while native registration is unresolved.

Never reads native depth or changes RGB. No nearest raw-frame pairing, image-fit
correction, native-hole filling, or evaluation-image inclusion. Full generation
is a separate explicit invocation after the stratified preflight is inspected.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time

from huggingface_hub import hf_hub_download
from PIL import Image
import torch

from cache_coco_teacher_geometry import generation_policy, infer_one
from indoor_download import write_json
from radio_moge.corpus_admission import checked_path
from radio_moge.geometry_cache import (GeometryPackWriter, canonical_json, publish_json,
                                       read_geometry_member, sha_file)
from radio_moge.upstream import load_moge


def select_preflight(records):
    """Two stable scene-distinct examples per scene type, plus raw-probe regressions."""
    bins = defaultdict(list)
    for row in records:
        bins[row['scene_type']].append(row)
    selected = {}
    for name in sorted(bins):
        groups = set()
        for row in sorted(bins[name], key=lambda r: r['sha256']):
            if row['group'] in groups:
                continue
            selected[row['id']] = row
            groups.add(row['group'])
            if len(groups) == 2:
                break
    regressions = {'nyuv2_' + n for n in ('0213', '0214', '0005', '0006', '0010', '0011')}
    for row in records:
        if row['id'] in regressions:
            selected[row['id']] = row
    return sorted(selected.values(), key=lambda r: r['corpus_ordinal'])


def source_rows(root):
    with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
        values = db.execute("SELECT ordinal,image,row_json FROM images WHERE dataset='nyuv2' ORDER BY ordinal")
        result = []
        for ordinal, image, encoded in values:
            row = json.loads(encoded)
            if (row['dataset'] != 'nyuv2' or row['split'] != 'train'
                    or row.get('image_rotation_cw', 0) != 0 or not row.get('scene_type')
                    or row['group'] != 'nyuv2:' + row['scene']):
                raise ValueError('Only frozen original NYU training RGB is eligible')
            row.update(image=image, corpus_ordinal=ordinal)
            result.append(row)
    return result


def run(config, mode, attempt, preflight):
    root = Path(config['root'])
    route_path = root / 'nyuv2-teacher-candidate-routing-20261102.json'
    route = json.loads(route_path.read_text())
    inventory = json.loads((root / 'inventory.json').read_text())
    if (inventory['stage'] != 'inventory_complete' or inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']
            or route['manifest_sha256'] != config['manifest_sha256'] or route['dataset'] != 'nyuv2'
            or route['scope_images'] != config['expected_counts']['nyuv2']
            or route['label_kind'] != 'teacher_pseudo' or route['metric_calibrated'] is not False
            or route['training_admission'] is not False or route['fill_native_holes'] is not False):
        raise ValueError('Frozen inventory and explicit nonmetric teacher routing required')
    for ref in route['evidence']:
        checked_path(ref)
    all_rows = source_rows(root)
    if len(all_rows) != config['expected_counts']['nyuv2']:
        raise ValueError('NYU training scope differs from the frozen inventory')
    probes = select_preflight(all_rows)
    output = root / attempt
    output.mkdir(exist_ok=False)
    (output / 'rejected').mkdir()
    status = dict(stage='loading_teacher', mode=mode, started_at=time.time(), completed=0, rejected=0,
                  training_started=False, training_admission=False, optimizer_updates=0)
    write_json(output / 'status.json', status)
    try:
        if str(torch.__version__) != '2.8.0+cu128' or torch.version.cuda != '12.8':
            raise ValueError('Use the established Torch2.8cu128 model environment, not native prep env')
        checkpoint = Path(hf_hub_download(config['teacher']['repo'], 'model.pt',
            revision=config['teacher']['revision'], local_files_only=True))
        checkpoint_hash = sha_file(checkpoint)
        policy = generation_policy(config, checkpoint_hash)
        policy.update(dataset='nyuv2', generator_sha256=sha_file(Path(__file__)),
            reused_inference_generator_sha256=sha_file(Path(__file__).with_name('cache_coco_teacher_geometry.py')),
            candidate_routing_sha256=sha_file(route_path), native_depth_decision=route['decision'],
            preflight_image_sha256=[r['sha256'] for r in probes],
            native_depth_read=False, torch=str(torch.__version__), cuda=torch.version.cuda,
            dimensions='decode hash-bound original RGB; do not alter frozen row fields')
        policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
        if mode == 'generate':
            previous = json.loads((root / preflight / 'report.json').read_text())
            if previous['stage'] != 'complete' or not previous['passed'] or previous['policy_sha256'] != policy_hash:
                raise ValueError('Matching NYU stratified preflight required')
        torch.set_num_threads(1)
        torch.manual_seed(43)
        source = Path('vendor/MoGe').resolve()
        if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Pinned upstream MoGe source is dirty')
        model, _ = load_moge(dict(moge_checkpoint=str(checkpoint), moge_source=str(source),
            moge_repo=config['teacher']['repo'], moge_revision=config['teacher']['revision']), 'cuda')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Teacher must be frozen in evaluation mode')
        versions = [p._version for p in model.parameters()]
        writer = GeometryPackWriter(output / 'packs', policy)
        status.update(stage='generating_cache', policy_sha256=policy_hash)
        write_json(output / 'status.json', status)
        examples, rejections = [], []
        rows = probes if mode == 'preflight' else all_rows
        for row in rows:
            # Only inspect the immutable RGB. The original row's `depth` points
            # to unresolved native data and is deliberately never opened here.
            with Image.open(row['image']) as file:
                width, height = file.size
            result, stats = infer_one(model, dict(row, width=width, height=height))
            if result is None:
                record = dict(image=row, **stats)
                rejections.append(record)
                publish_json(output / 'rejected' / (row['sha256'] + '.json'), record)
                status['rejected'] += 1
            else:
                depth, valid, k = result
                provenance = dict(image_sha256=row['sha256'], dataset='nyuv2', source_id=row['id'],
                    label_kind='teacher_pseudo', camera_kind='teacher_estimated', metric_calibrated=False,
                    conversion='pinned_moge2_infer_forced_projection', policy_sha256=policy_hash,
                    teacher_revision=config['teacher']['revision'], source_quality=stats,
                    source_proofs=[dict(source=config['teacher']['repo'] + '/model.pt', sha256=checkpoint_hash),
                                   dict(source=row['image'], sha256=row['sha256']),
                                   dict(source=str(route_path), sha256=sha_file(route_path))])
                writer.add(row, depth, valid, k, provenance)
                status['completed'] += 1
            if mode == 'preflight':
                examples.append(dict(source_id=row['id'], group=row['group'], scene_type=row['scene_type'], **stats))
            if (status['completed'] + status['rejected']) % 8 == 0:
                status['updated_at'] = time.time()
                write_json(output / 'status.json', status)
                print(json.dumps(status), flush=True)
        if versions != [p._version for p in model.parameters()]:
            raise ValueError('Teacher parameters changed')
        writer.close()
        publish_json(output / 'rejections.json', rejections)
        verified = 0
        for item in writer.packs:
            receipt_path = output / 'packs' / item['receipt']
            receipt = json.loads(receipt_path.read_text())
            tar = receipt_path.parent / receipt['tar']
            if sha_file(receipt_path) != item['sha256'] or sha_file(tar) != receipt['sha256']:
                raise ValueError('Teacher cache changed')
            for entry in receipt['entries']:
                read_geometry_member(tar, entry)
                verified += 1
        if status['completed'] + status['rejected'] != len(rows) or verified != status['completed']:
            raise ValueError('Teacher cache does not account for the exact selected scope')
        status.update(stage='complete', completed_at=time.time(), expected_scope=len(rows),
            verified_cache_members=verified, torch=torch.__version__, cuda=torch.version.cuda,
            teacher_frozen=True, examples=examples, scene_types=sorted({r['scene_type'] for r in rows}),
            passed=bool(mode == 'preflight' and not rejections and verified == len(probes)))
        write_json(output / 'status.json', status)
        publish_json(output / 'report.json', status)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'generate'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--preflight', default='nyuv2-teacher-preflight-01')
    parser.add_argument('--wait-for-gpu-lock', action='store_true')
    args = parser.parse_args()
    if any(Path(name).name != name for name in (args.attempt, args.preflight)):
        parser.error('Use a new attempt basename')
    config = json.loads(args.config.read_text())
    lock_path = Path(config['root']).parent / 'data-expansion-attention-20261022/attention-24h/.training.lock'
    if not lock_path.is_file():
        raise FileNotFoundError('Established GPU lock is missing')
    with lock_path.open('a+') as lock:
        if args.wait_for_gpu_lock:
            print(json.dumps(dict(stage='waiting_for_existing_gpu_lock', optimizer_updates=0,
                                  attempt=args.attempt, mode=args.mode, time=time.time())), flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_gpu_lock else fcntl.LOCK_NB))
        run(config, args.mode, args.attempt, args.preflight)
