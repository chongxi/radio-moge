"""Separate IRS teacher fallback candidates while native disparity is unresolved.

This does not discard native labels or fill their holes. Final corpus routing
must explicitly select an accepted source; teacher estimated scale is not GT.
Reuses the exact tested original-RGB MoGe inference path from the COCO producer.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time

from huggingface_hub import hf_hub_download
import torch

from cache_coco_teacher_geometry import generation_policy, infer_one
from indoor_download import write_json
from radio_moge.geometry_cache import (GeometryPackWriter, canonical_json, publish_json,
                                       read_geometry_member, sha_file)
from radio_moge.upstream import load_moge


def rows(root, preflight):
    with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
        db.execute('PRAGMA cache_size=-32768')
        if preflight:
            pairs = db.execute("SELECT MIN(ordinal),MAX(ordinal) FROM images WHERE dataset='irs' "
                "GROUP BY scene_group,json_extract(row_json,'$.eye') ORDER BY 1").fetchall()
            indices = sorted({i for pair in pairs for i in pair})
            if len(pairs) != 26 or len(indices) != 52:
                raise ValueError('Require both eyes, first/last frozen frames, all 13 IRS families')
            query = 'SELECT ordinal,image,row_json FROM images WHERE ordinal IN (' + ','.join('?' * 52) + ') ORDER BY ordinal'
            values = db.execute(query, indices)
        else:
            values = db.execute("SELECT ordinal,image,row_json FROM images WHERE dataset='irs' ORDER BY ordinal")
        for ordinal, image, encoded in values:
            row = json.loads(encoded)
            if (row['dataset'] != 'irs' or row['split'] != 'train' or row['eye'] not in ('left', 'right')
                    or row.get('image_rotation_cw', 0) != 0):
                raise ValueError('Only frozen, original-orientation IRS training RGB permitted')
            row.update(image=image, corpus_ordinal=ordinal)
            yield row


def run(config, mode, attempt, preflight_name):
    root = Path(config['root'])
    inventory = json.loads((root / 'inventory.json').read_text())
    route_path = root / 'irs-teacher-candidate-routing-20261102.json'
    route = json.loads(route_path.read_text())
    if (inventory['stage'] != 'inventory_complete'
            or inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']
            or route['dataset'] != 'irs' or route['scope_images'] != config['expected_counts']['irs']
            or route['manifest_sha256'] != config['manifest_sha256']
            or route['label_kind'] != 'teacher_pseudo' or route['training_admission'] is not False):
        raise ValueError('Frozen inventory and explicit teacher-candidate routing required')
    output = root / attempt
    output.mkdir(exist_ok=False)
    (output / 'rejected').mkdir()
    status = dict(stage='loading_teacher', mode=mode, started_at=time.time(), completed=0, rejected=0,
                  training_started=False, optimizer_updates=0, training_admission=False)
    write_json(output / 'status.json', status)
    try:
        checkpoint = Path(hf_hub_download(config['teacher']['repo'], 'model.pt',
            revision=config['teacher']['revision'], local_files_only=True))
        checkpoint_hash = sha_file(checkpoint)
        policy = generation_policy(config, checkpoint_hash)
        policy.update(dataset='irs', generator_sha256=sha_file(Path(__file__)),
            reused_inference_generator_sha256=sha_file(Path(__file__).with_name('cache_coco_teacher_geometry.py')),
            candidate_routing_sha256=sha_file(route_path),
            native_depth_decision='unresolved; these are separate pseudo candidates, not replacement measured GT',
            final_routing_requires_global_acceptance=True)
        policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
        if mode == 'generate':
            previous = json.loads((root / preflight_name / 'report.json').read_text())
            if previous['stage'] != 'complete' or not previous['passed'] or previous['policy_sha256'] != policy_hash:
                raise ValueError('Matching full-family/both-eye teacher preflight required')
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
        writer = GeometryPackWriter(output / 'packs', policy, max_entries=256)
        status.update(stage='generating_cache', policy_sha256=policy_hash)
        write_json(output / 'status.json', status)
        examples, rejections, groups = [], [], set()
        for row in rows(root, mode == 'preflight'):
            result, stats = infer_one(model, row)
            groups.add((row['group'], row['eye']))
            if result is None:
                record = dict(image=row, **stats)
                rejections.append(record)
                publish_json(output / 'rejected' / (row['sha256'] + '.json'), record)
                status['rejected'] += 1
            else:
                depth, valid, k = result
                provenance = dict(image_sha256=row['sha256'], dataset='irs', source_id=row['source_member'],
                    label_kind='teacher_pseudo', camera_kind='teacher_estimated', metric_calibrated=False,
                    conversion='pinned_moge2_infer_forced_projection', policy_sha256=policy_hash,
                    teacher_revision=config['teacher']['revision'], source_quality=stats,
                    source_proofs=[dict(source=config['teacher']['repo'] + '/model.pt', sha256=checkpoint_hash),
                                   dict(source=row['image'], sha256=row['sha256']),
                                   dict(source=str(route_path), sha256=sha_file(route_path))])
                writer.add(row, depth, valid, k, provenance)
                status['completed'] += 1
            if mode == 'preflight':
                examples.append(dict(source_id=row['source_member'], group=row['group'], eye=row['eye'], **stats))
            if (status['completed'] + status['rejected']) % 32 == 0:
                status.update(updated_at=time.time(), family_eye_pairs_seen=len(groups))
                write_json(output / 'status.json', status)
                print(json.dumps(status), flush=True)
        if versions != [p._version for p in model.parameters()]:
            raise ValueError('Teacher parameters changed during inference')
        writer.close()
        publish_json(output / 'rejections.json', rejections)
        status.update(stage='verifying_cache', updated_at=time.time())
        write_json(output / 'status.json', status)
        verified = 0
        for item in writer.packs:
            receipt_path = output / 'packs' / item['receipt']
            receipt = json.loads(receipt_path.read_text())
            tar = receipt_path.parent / receipt['tar']
            if sha_file(receipt_path) != item['sha256'] or sha_file(tar) != receipt['sha256']:
                raise ValueError('Completed pseudo pack changed')
            for entry in receipt['entries']:
                read_geometry_member(tar, entry)
                verified += 1
        expected = 52 if mode == 'preflight' else config['expected_counts']['irs']
        if status['completed'] + status['rejected'] != expected or verified != status['completed']:
            raise ValueError('Pseudo cache did not account for the exact frozen scope')
        status.update(stage='complete', completed_at=time.time(), expected_scope=expected,
            verified_cache_members=verified, policy_sha256=policy_hash, torch=torch.__version__,
            cuda=torch.version.cuda, teacher_frozen=True, examples=examples, family_eye_pairs_seen=len(groups),
            passed=bool(mode == 'preflight' and verified == 52 and not rejections and len(groups) == 26))
        write_json(output / 'status.json', status)
        publish_json(output / 'report.json', status)
        print(json.dumps({k:v for k,v in status.items() if k != 'examples'}), flush=True)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'generate'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--preflight', default='irs-teacher-preflight-01')
    args = parser.parse_args()
    if any(Path(name).name != name for name in (args.attempt, args.preflight)):
        parser.error('Use a new attempt basename')
    config = json.loads(args.config.read_text())
    lock_path = Path(config['root']).parent / 'data-expansion-attention-20261022/attention-24h/.training.lock'
    if not lock_path.is_file():
        raise FileNotFoundError('Established GPU lock is missing; do not create a new one')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(config, args.mode, args.attempt, args.preflight)
