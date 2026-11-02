"""Separate nonmetric teacher candidates for exactly the unresolved native rows.

Only original RGB enters inference. Native arrays, camera poses and masks are
not read, repaired, filled or overwritten. Generation needs inspected preflight.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import hf_hub_download
import torch

from cache_coco_teacher_geometry import generation_policy, infer_one
from indoor_download import write_json
from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import (GeometryPackWriter, canonical_json, publish_json,
                                      read_geometry_member, sha_file)
from radio_moge.native_teacher_scope import review_scope, select_preflight, require_visual_preflight
from radio_moge.upstream import load_moge


def run(config, mode, attempt, preflight):
    root = Path(config['root'])
    route_ref = file_ref(root / 'hypersim-review-teacher-routing-20261102.json')
    route = checked_json(route_ref)
    inventory = json.loads((root / 'inventory.json').read_text())
    if (inventory['stage'] != 'inventory_complete' or inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']
            or inventory['dataset_counts'] != config['expected_counts']):
        raise ValueError('Complete frozen inventory required')
    all_rows, bindings = review_scope(root, route, config)
    probes = select_preflight(all_rows, bindings)
    output = root / attempt
    output.mkdir(exist_ok=False)
    (output / 'rejected').mkdir()
    status = dict(stage='loading_teacher', mode=mode, started_at=time.time(), completed=0, rejected=0,
                  training_started=False, training_admission=False, optimizer_updates=0)
    write_json(output / 'status.json', status)
    try:
        if str(torch.__version__) != '2.8.0+cu128' or torch.version.cuda != '12.8':
            raise ValueError('Use the established Torch2.8cu128 model environment')
        checkpoint = Path(hf_hub_download(config['teacher']['repo'], 'model.pt',
            revision=config['teacher']['revision'], local_files_only=True))
        checkpoint_hash = sha_file(checkpoint)
        policy = generation_policy(config, checkpoint_hash)
        policy.update(dataset='hypersim', generator_sha256=sha_file(Path(__file__)),
            reused_inference_generator_sha256=sha_file(Path(__file__).with_name('cache_coco_teacher_geometry.py')),
            scope_helper_sha256=sha_file(Path(__file__).parents[1] / 'src/radio_moge/native_teacher_scope.py'),
            candidate_routing_sha256=route_ref['sha256'], native_depth_decision=route['decision'],
            preflight_image_sha256=[r['sha256'] for r in probes], native_geometry_read=False,
            minimum_valid_fraction=.05, torch=str(torch.__version__), cuda=torch.version.cuda)
        policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
        if mode == 'generate':
            require_visual_preflight(root / preflight / 'report.json', policy_hash, route['scene_counts'])
        source = Path('vendor/MoGe').resolve()
        if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Pinned upstream MoGe source is dirty')
        torch.set_num_threads(1)
        torch.manual_seed(43)
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
            native = bindings[row['corpus_ordinal']]
            result, stats = infer_one(model, row)
            if result is not None and stats['valid_fraction'] < .05:
                result = None
                stats['reason'] = 'insufficient_teacher_support'
            if result is None:
                record = dict(image=row, **stats)
                rejections.append(record)
                publish_json(output / 'rejected' / (row['sha256'] + '.json'), record)
                status['rejected'] += 1
            else:
                depth, valid, k = result
                provenance = dict(image_sha256=row['sha256'], dataset='hypersim',
                    source_id=f'{row["group"]}/{row["camera"]}/{row["frame_id"]}',
                    label_kind='teacher_pseudo', camera_kind='teacher_estimated', metric_calibrated=False,
                    conversion='pinned_moge2_infer_forced_projection', policy_sha256=policy_hash,
                    teacher_revision=config['teacher']['revision'], source_quality=stats,
                    native_source_review_unresolved=True, native_outcome=native['native_outcome'],
                    native_reason=native['native_reason'], source_proofs=[
                        dict(source=config['teacher']['repo'] + '/model.pt', sha256=checkpoint_hash),
                        dict(source=row['image'], sha256=row['sha256']),
                        dict(source=route_ref['path'], sha256=route_ref['sha256'])])
                writer.add(row, depth, valid, k, provenance)
                status['completed'] += 1
            if mode == 'preflight':
                examples.append(dict(group=row['group'], camera=row['camera'], frame=row['frame_id'],
                    corpus_ordinal=row['corpus_ordinal'], native_reason=native['native_reason'], **stats))
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
            raise ValueError('Candidate output does not cover the exact selected scope')
        status.update(stage='complete', completed_at=time.time(), expected_scope=len(rows),
            verified_cache_members=verified, torch=str(torch.__version__), cuda=torch.version.cuda,
            teacher_frozen=True, examples=examples, scenes=sorted(route['scene_counts']),
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
    parser.add_argument('--preflight', default='hypersim-review-teacher-preflight-01')
    parser.add_argument('--wait-for-gpu-lock', action='store_true')
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.attempt, args.preflight)):
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
