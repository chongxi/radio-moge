"""Separate nonmetric teacher candidates for exactly the quarantined DSLR scenes.

Original native audits, labels and masks remain unchanged. Only original RGB and
its anonymity mask enter inference; laser depths/poses never repair the teacher.
Bulk generation requires both matching numeric and explicit visual preflight.
"""
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import torch

from cache_coco_teacher_geometry import generation_policy, infer_one
from indoor_download import write_json
from radio_moge.corpus_admission import checked_json, checked_path, readonly_db
from radio_moge.geometry_cache import (GeometryPackWriter, canonical_json, publish_json,
                                      read_geometry_member, sha_file)
from radio_moge.upstream import load_moge


def review_scope(root, route, manifest):
    audit = checked_json(route['source_audit'])
    if (audit['stage'] != 'complete' or audit['manifest_sha256'] != manifest
            or audit['scenes_completed'] != audit['expected_scenes']
            or len(audit['scenes']) != audit['expected_scenes']):
        raise ValueError('Complete matching native source audit is required')
    selected = [s for s in audit['scenes'] if s['stage'] == 'needs_review']
    if (any(s['stage'] not in ('needs_review', 'source_checks_passed') for s in audit['scenes'])
            or len(selected) != audit['scenes_needing_review']
            or len({s['scene'] for s in audit['scenes']}) != len(audit['scenes'])):
        raise ValueError('Unresolved or duplicated source-audit entries')
    records, probes, counts = [], [], {}
    with readonly_db(root / 'inventory.sqlite') as db:
        for item in sorted(selected, key=lambda s: s['scene']):
            path = Path(route['source_audit']['path']).parent / item['report']
            value = checked_json(dict(path=str(path), sha256=item['sha256']))
            if value['scene'] != item['scene'] or value['stage'] != 'needs_review':
                raise ValueError('Native review identity changed')
            scene_rows = []
            # Scene identifiers already include their dataset namespace. Query
            # that indexed key directly; the extra dataset predicate makes
            # SQLite rescan all 841k DSLR rows once for every scene. The row's
            # dataset, scene and split are still checked independently below.
            for ordinal, image, encoded in db.execute('SELECT ordinal,image,row_json FROM images '
                    'WHERE scene_group=? ORDER BY ordinal', ('scannetpp:' + item['scene'],)):
                row = json.loads(encoded)
                if (row['dataset'] != 'scannetpp_dslr' or row['split'] != 'train'
                        or row['official_split'] != 'nvs_sem_train' or row['scene'] != item['scene']
                        or row['group'] != 'scannetpp:' + row['scene']
                        or row.get('image_rotation_cw', 0) != 0):
                    raise ValueError('Only exact frozen original training DSLR RGB is eligible')
                row.update(image=image, corpus_ordinal=ordinal)
                scene_rows.append(row)
            if len(scene_rows) < 2 or len(scene_rows) != value['images_in_training_manifest']:
                raise ValueError('Source-audit image scope differs from the inventory')
            counts[item['scene']] = len(scene_rows)
            records.extend(scene_rows)
            probes.extend([scene_rows[0], scene_rows[-1]])
    if counts != route['scene_counts'] or len(records) != route['scope_images']:
        raise ValueError('Explicit candidate routing differs from the full reviewed-scene scope')
    return records, probes


def anonymized_support(row, valid):
    data = Path(row['mask']).read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        mask = np.asarray(image)
    if mask.dtype != np.uint8 or mask.shape != valid.shape or valid.shape != (row['height'], row['width']):
        raise ValueError('Original anonymity mask must match the unchanged teacher/RGB grid')
    support = valid & binary_erosion(mask == 255, structure=np.ones((5, 5), bool), border_value=0)
    return support, hashlib.sha256(data).hexdigest()


def run(config, mode, attempt, preflight):
    root = Path(config['root'])
    route_path = root / 'scannetpp-review-teacher-routing-20261102.json'
    route = json.loads(route_path.read_text())
    inventory = json.loads((root / 'inventory.json').read_text())
    if (inventory['stage'] != 'inventory_complete' or inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']
            or route['manifest_sha256'] != config['manifest_sha256'] or route['dataset'] != 'scannetpp_dslr'
            or route['label_kind'] != 'teacher_pseudo' or route['metric_calibrated'] is not False
            or route['training_admission'] is not False or route['fill_native_holes'] is not False):
        raise ValueError('Frozen inventory and explicit nonmetric candidate routing required')
    all_rows, probes = review_scope(root, route, config['manifest_sha256'])
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
        policy.update(dataset='scannetpp_dslr', generator_sha256=sha_file(Path(__file__)),
            reused_inference_generator_sha256=sha_file(Path(__file__).with_name('cache_coco_teacher_geometry.py')),
            candidate_routing_sha256=sha_file(route_path), native_depth_decision=route['decision'],
            preflight_image_sha256=[r['sha256'] for r in probes], native_geometry_read=False,
            anonymity='original mask==255 AND 5x5 erosion AND teacher support; no resizing/filling',
            minimum_valid_fraction=.05, torch=str(torch.__version__), cuda=torch.version.cuda)
        policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
        if mode == 'generate':
            previous_path = root / preflight / 'report.json'
            previous = json.loads(previous_path.read_text())
            review = json.loads((root / preflight / 'visual-review.json').read_text())
            if (previous['stage'] != 'complete' or not previous['passed']
                    or previous['policy_sha256'] != policy_hash or review.get('passed') is not True
                    or review.get('stage') != 'visual_review_complete'
                    or review.get('preflight_report_sha256') != sha_file(previous_path)
                    or review.get('policy_sha256') != policy_hash
                    or set(review.get('reviewed_scenes', [])) != set(route['scene_counts'])
                    or not review.get('evidence')):
                raise ValueError('Matching numeric and all-scene visual preflight required')
            for ref in review['evidence']:
                checked_path(ref)
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
        examples, rejections = [], []
        rows = probes if mode == 'preflight' else all_rows
        for row in rows:
            result, stats = infer_one(model, row)
            if result is not None:
                depth, valid, k = result
                valid, mask_hash = anonymized_support(row, valid)
                stats.update(teacher_valid_fraction=stats['valid_fraction'], valid_fraction=float(valid.mean()),
                             valid_pixels=int(valid.sum()), anonymous_mask_sha256=mask_hash)
                if valid.mean() < .05:
                    result = None
                    stats['reason'] = 'insufficient_teacher_and_anonymous_support'
            if result is None:
                record = dict(image=row, **stats)
                rejections.append(record)
                publish_json(output / 'rejected' / (row['sha256'] + '.json'), record)
                status['rejected'] += 1
            else:
                provenance = dict(image_sha256=row['sha256'], dataset='scannetpp_dslr',
                    source_id=row['scene'] + '/' + row['view'], label_kind='teacher_pseudo',
                    camera_kind='teacher_estimated', metric_calibrated=False,
                    conversion='pinned_moge2_infer_forced_projection_with_original_anonymity_mask',
                    policy_sha256=policy_hash, teacher_revision=config['teacher']['revision'],
                    source_quality=stats, native_source_review_unresolved=True,
                    source_proofs=[dict(source=config['teacher']['repo'] + '/model.pt', sha256=checkpoint_hash),
                        dict(source=row['image'], sha256=row['sha256']),
                        dict(source=row['mask'], sha256=mask_hash),
                        dict(source=str(route_path), sha256=sha_file(route_path))])
                writer.add(row, depth, valid, k, provenance)
                status['completed'] += 1
            if mode == 'preflight':
                examples.append(dict(scene=row['scene'], view=row['view'], **stats))
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
            raise ValueError('Candidate output does not account for the exact selected scope')
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
    parser.add_argument('--preflight', default='scannetpp-review-teacher-preflight-01')
    parser.add_argument('--wait-for-gpu-lock', action='store_true')
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.attempt, args.preflight)):
        parser.error('Use a new attempt basename')
    config = json.loads(args.config.read_text())
    lock_path = Path(config['root']).parent / 'data-expansion-attention-20261022/attention-24h/.training.lock'
    with lock_path.open('r+') as lock:
        if args.wait_for_gpu_lock:
            print(json.dumps(dict(stage='waiting_for_existing_gpu_lock', optimizer_updates=0,
                                  attempt=args.attempt, mode=args.mode, time=time.time())), flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_gpu_lock else fcntl.LOCK_NB))
        run(config, args.mode, args.attempt, args.preflight)
