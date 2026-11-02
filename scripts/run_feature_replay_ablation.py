"""Bounded GT-only versus 50% RGB feature-slot replay, with identical GT exposure."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.config import ROOT
from radio_moge.data import check_split, read_manifest
from radio_moge.feature_replay import (is_replay_batch, load_replay_records, make_replay_loader,
                                       replay_feature_losses)
from radio_moge.losses import FeatureStatistics, weighted_loss
from radio_moge.official_gt import OFFICIAL_GT_WEIGHTS, official_ground_truth_losses
from radio_moge.system import build_system
from radio_moge.train import make_loader, validate_config

if __package__:
    from .report_loss_weight_ablation import collect, compare, read, tensor_digest
    from .run_loss_weight_ablation import FEATURE_KEYS, SOURCE, digest, run
else:
    from report_loss_weight_ablation import collect, compare, read, tensor_digest
    from run_loss_weight_ablation import FEATURE_KEYS, SOURCE, digest, run


PARENT = Path('/home/buildhub/shared/radio-moge/chongxi/loss-weight-ablation-20261102')
REPLAY_SEED = 100043


def quotas(weights, count):
    total = sum(weights.values())
    exact = {key: count * value / total for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    order = sorted(weights, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:count - sum(result.values())]:
        result[key] += 1
    return result


def select_rgb_pool(manifest, weights, count, seed):
    """Index byte offsets, not decoded data; sample source -> scene -> frame.

    Exact original source quotas, uniform scene proposals, unique RGB paths.
    Labels are deliberately stripped, including NYU training depth filenames.
    """
    groups = defaultdict(lambda: defaultdict(list))
    source_hash = hashlib.sha256()
    offset = 0
    with Path(manifest).open('rb') as file:
        for line in file:
            source_hash.update(line)
            if line.strip():
                row = json.loads(line)
                if row.get('split') != 'train' or not row.get('group'):
                    raise ValueError('Replay source must be exclusively training scenes')
                groups[row['dataset']][row['group']].append(offset)
            offset += len(line)
    if set(groups) != set(weights):
        raise ValueError('Replay sources must match the original seven-source mixture')
    counts = quotas(weights, count)
    rng = random.Random(seed)
    rows = []
    with Path(manifest).open('rb') as file:
        for source in sorted(groups):
            scenes = [groups[source][key] for key in sorted(groups[source])]
            if counts[source] > sum(map(len, scenes)):
                raise ValueError('Replay quota exceeds available distinct training images')
            selected = set()
            while len(selected) < counts[source]:
                selected.add(rng.choice(rng.choice(scenes)))
            for byte_offset in sorted(selected):
                file.seek(byte_offset)
                row = json.loads(file.readline())
                rgb = {key: row[key] for key in ('image', 'dataset', 'group', 'split',
                                                'sha256', 'sha256_kind', 'image_rotation_cw')
                       if key in row}
                rgb['image'] = str((Path(manifest).parent / rgb['image']).resolve())
                rgb['replay_origin_byte_offset'] = byte_offset
                rows.append(rgb)
    if len({row['image'] for row in rows}) != count:
        raise ValueError('Duplicate RGB images in selected replay pool')
    return rows, dict(source_manifest=str(manifest), source_manifest_sha256=source_hash.hexdigest(),
                      seed=seed, selected_images=count, selected_sources=dict(Counter(
                          row['dataset'] for row in rows)), available_scenes={
                          source: len(scenes) for source, scenes in groups.items()},
                      sampling='Original source quotas; uniform scene/frame proposals without duplicate images')


def check_heldout_metadata(rows, manifests):
    """Inspect split metadata only; never open validation/test depth labels."""
    images = {row['image'] for row in rows}
    groups = {row['group'] for row in rows}
    hashes = {row['sha256'] for row in rows if row.get('sha256')}
    proof = {}
    for manifest in manifests:
        count, checksum = 0, hashlib.sha256()
        with Path(manifest).open('rb') as file:
            for line in file:
                checksum.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                image = str((Path(manifest).parent / row['image']).resolve())
                if (image in images or row.get('group') in groups
                        or (row.get('sha256') and row['sha256'] in hashes)):
                    raise ValueError(f'Replay overlaps heldout metadata: {manifest}')
                count += 1
        proof[str(manifest)] = dict(images=count, sha256=checksum.hexdigest(), overlap=0)
    return proof


def write_new_jsonl(rows, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as file:
            for row in rows:
                file.write(json.dumps(row, allow_nan=False) + '\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(root, parent, revision):
    if (root / 'plan.json').exists() or (root / 'replay-train-rgb.jsonl').exists():
        raise FileExistsError('Replay experiment already prepared; use a new destination')
    previous, previous_report = read(parent / 'plan.json'), read(parent / 'report-verified.json')
    assert read(parent / 'controller.json')['stage'] == previous_report['stage'] == 'complete'
    assert previous['source'] == str(SOURCE) and digest(SOURCE) == previous['source_sha256']
    template = read(previous['arms']['official-f1']['path'])
    assert template['geometry_supervision']['objective'] == 'moge2_core'
    assert all(template['loss'][key] == 1. for key in FEATURE_KEYS)
    assert template['train']['batch_size'] == 1 and template['train']['accumulation'] == 16
    original = read(ROOT / 'configs/joint-qknorm-full-day.json')['data']
    count = 250 * template['train']['batch_size'] * template['train']['accumulation'] // 2
    rows, pool_proof = select_rgb_pool(original['train_manifest'], original['mixture_weights'],
                                       count, REPLAY_SEED)
    fixed = [template['data']['val_manifest'], template['data']['geometry_val_manifest'],
             *(view['manifest'] for view in template['evaluation']['dataset_variants'].values())]
    source_dir = Path(original['train_manifest']).parent
    tests = sorted(source_dir.glob('test*.jsonl'))
    validations = sorted(source_dir.glob('val*.jsonl'))
    if not tests or not validations:
        raise ValueError('Original heldout-split metadata must be available for leakage checks')
    heldout = list(dict.fromkeys([*fixed, *(str(path) for path in validations),
                                 *(str(path) for path in tests)]))
    pool_proof['heldout_metadata_checks'] = check_heldout_metadata(rows, heldout)
    check_split(rows, read_manifest(template['data']['train_manifest']))
    manifest = root / 'replay-train-rgb.jsonl'
    write_new_jsonl(rows, manifest)
    # Resolve every selected RGB path before any training starts.
    assert len(read_manifest(manifest)) == count
    atomic_json(pool_proof, root / 'replay-pool.json')
    arms = {}
    for name in ('gt-only', 'rgb-replay50'):
        config = deepcopy(template)
        config['output_dir'] = str(root / name / 'train')
        config['train'].update(steps=250, max_seconds=3600, eval_at_start=False,
                               eval_every=250, save_every=250)
        config['tracking'].update(name=root.name + '-' + name, group=root.name,
                                   tags=['feature-replay-ablation', name], mode='offline')
        if name == 'rgb-replay50':
            config['feature_replay'] = dict(manifest=str(manifest), fraction=.5, seed=REPLAY_SEED)
        validate_config(config)
        path = root / name / 'config.json'
        atomic_json(config, path)
        arms[name] = dict(path=str(path), sha256=digest(path), objective='moge2_core',
                          feature_weight=1., feature_replay_fraction=.5 if name == 'rgb-replay50' else 0.)
    plan = dict(authorized=True, parent_experiment=str(parent), source=str(SOURCE),
                source_sha256=previous['source_sha256'], source_step=232288,
                revision=revision, steps_per_arm=250, seed=43, arms=arms,
                training_lock=previous['training_lock'], automatic_restart=False,
                maximum_seconds_per_arm=3600, prepared_at=time.time(),
                changed_factors=['Source of half the feature-supervision slots: GT RGB to original mixed RGB'],
                unchanged=['Same source adapter/statistics, fresh AdamW, seed43, LR1e-5',
                           '16 GT geometry images and 16 feature images per optimizer update',
                           '81 GT views, GT order/augmentation, all loss weights, 400x640 bucket',
                           'Frozen RADIO/MoGe backbones and decoder; fixed four validation views plus laser18'],
                replay_pool=pool_proof,
                assessment=['Compare fixed update250 against the rerun GT-only control with paired scene intervals',
                            'Assess both macro cosine metrics and NYU depth jointly; always report laser tradeoffs',
                            'Do not call it full retention unless both cosine metrics and NYU error recover '
                            'to the source232288 baseline without worse laser RMSE'],
                limits=previous['limits'] + ['RGB replay adds encoder/adapter compute, not optimizer updates',
                                             'Only 2000 sampled original RGB images; not a full large-scale replay run'],
                conditional_next=None)
    atomic_json(plan, root / 'plan.json')
    print(json.dumps(dict(stage='prepared', arms=list(arms), replay_pool=pool_proof), indent=2))


def preflight(root):
    plan = read(root / 'plan.json')
    if (root / 'preflight.json').exists():
        raise FileExistsError('Preflight already recorded')
    config = read(plan['arms']['rgb-replay50']['path'])
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = torch.load(plan['source'], map_location='cpu', weights_only=True, mmap=True)
        system, adapter = build_system(config, torch.device('cuda'))
        adapter.load_state_dict(state['adapter'])
        stats = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim']).cuda()
        stats.load_state_dict(state['statistics'])
        del state
        records = read_manifest(config['data']['train_manifest'])
        replay = load_replay_records(config, [read_manifest(config['data']['val_manifest'])])
        rgb_iterator = iter(make_replay_loader(replay, config, 0, 4))
        results = []
        def amp():
            return torch.autocast('cuda', dtype=torch.bfloat16)

        for cursor, batch in enumerate(make_loader(records, config, 0, 4)):
            image = batch['image'].cuda()
            target = {key: value.cuda() for key, value in batch.items() if key != 'image'}
            with torch.no_grad(), amp():
                features, teacher = system.encode(image, adapter.indices)
            with amp():
                predicted = adapter(features)
            raw = system.decoder(predicted, image.shape[-2:])
            gt_losses = official_ground_truth_losses(raw, target, seed=43 + cursor)
            feature_losses = stats.loss(predicted, teacher)
            if is_replay_batch(cursor):
                before = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                feature_losses = replay_feature_losses(
                    system, adapter, stats, next(rgb_iterator).cuda(), adapter.indices, amp,
                    REPLAY_SEED + cursor // 2)
                assert torch.equal(before[0], torch.get_rng_state())
                assert torch.equal(before[1], torch.cuda.get_rng_state())
            loss = weighted_loss({**feature_losses, **gt_losses}, config['loss'])
            assert torch.isfinite(loss)
            loss.backward()
            grads = [parameter.grad for parameter in adapter.parameters() if parameter.grad is not None]
            assert grads and all(torch.isfinite(gradient).all() for gradient in grads)
            assert any(gradient.abs().sum() > 0 for gradient in grads)
            assert all(parameter.grad is None for name in ('radio', 'encoder', 'decoder')
                       for parameter in getattr(system, name).parameters())
            adapter.zero_grad(set_to_none=True)
            results.append(dict(cursor=cursor, rgb_feature_slot=is_replay_batch(cursor),
                                feature_loss=weighted_loss(feature_losses, config['loss']).item(),
                                geometry_loss=weighted_loss(gt_losses, OFFICIAL_GT_WEIGHTS).item(),
                                total=loss.item()))
        atomic_json(dict(stage='passed', optimizer_updates=0, gt_microbatches=4,
                         rgb_feature_slots=2, rng_isolated=True, finite_adapter_gradients=True,
                         frozen_backbone_and_decoder_gradients=True, results=results), root / 'preflight.json')
        print(json.dumps(read(root / 'preflight.json'), indent=2))


def report(root):
    plan = read(root / 'plan.json')
    assert read(root / 'controller.json')['stage'] == 'complete'
    parent = Path(plan['parent_experiment'])
    old = read(parent / 'report-verified.json')
    metrics, paths, digests, identities = {}, {}, {}, {}
    for name, arm in plan['arms'].items():
        train_dir = root / name / 'train'
        stop = read(train_dir / 'stop-report.json')
        assert stop['step'] == 250 and stop['final_evaluation'] == 'complete'
        assert digest(arm['path']) == arm['sha256']
        paths[name] = train_dir / 'eval-0000250'
        metrics[name] = collect(paths[name])
        provenance = read(train_dir / 'provenance.json')
        assert provenance['execution_revision'] == plan['revision']
        digests[name] = provenance['data_digests']
        state = torch.load(train_dir / 'initial.pt', map_location='cpu', weights_only=True, mmap=True)
        identities[name] = tensor_digest(state)
        assert identities[name] == old['source_adapter_statistics_digest']
        assert state['step'] == state['cursor'] == 0 and not state['optimizer']['state']
        del state
        final = torch.load(train_dir / 'latest.pt', map_location='cpu', weights_only=True, mmap=True)
        assert final['step'] == 250 and final['cursor'] == 250 * 16
        del final
        if name == 'rgb-replay50':
            logs = [json.loads(line) for line in (train_dir / 'train.jsonl').read_text().splitlines()]
            assert logs[-1]['step'] == 250
            assert all(row['feature_replay_fraction'] == .5 for row in logs)
    replay_digest = dict(digests['rgb-replay50'])
    assert replay_digest.pop('feature_replay')
    old_digests = read(parent / 'official-f1/train/provenance.json')['data_digests']
    assert replay_digest == digests['gt-only'] == old_digests
    config = read(plan['arms']['gt-only']['path'])
    records = read_manifest(config['data']['val_manifest'])
    control_repeat = compare(parent / 'official-f1/train/eval-0000250', paths['gt-only'], records)
    result = dict(stage='complete', steps_per_arm=250, baseline=old['baseline'], arms=metrics,
                  comparisons=dict(gt_only_vs_replay=compare(paths['gt-only'], paths['rgb-replay50'], records),
                                   baseline_vs_replay=compare(parent / 'custom-f01/train/eval-0000000',
                                                              paths['rgb-replay50'], records),
                                   repeated_control=control_repeat),
                  same_gt_and_validation_data=True, identical_initial_adapter_statistics=identities,
                  control_repeat_max_abs_metric_difference=max(
                      abs(metrics['gt-only'][key] - old['arms']['official-f1'][key])
                      for key in ('spatial_cos', 'summary_cos', 'nyu_absrel', 'nyu_rmse',
                                  'laser_absrel', 'laser_rmse', 'laser_plane_rmse')),
                  replay_pool=plan['replay_pool'], limitations=plan['limits'],
                  selection='Fixed final update250, not best checkpoints')
    atomic_json(result, root / 'report-verified.json')
    print(json.dumps({key: result[key] for key in ('stage', 'baseline', 'arms')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'preflight', 'run', 'report'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--revision')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'prepare':
        if not args.revision:
            parser.error('prepare requires --revision')
        prepare(args.root, args.parent, args.revision)
    elif args.mode == 'preflight':
        preflight(args.root)
    elif args.mode == 'run':
        assert read(args.root / 'preflight.json')['stage'] == 'passed'
        run(args.root)
    else:
        report(args.root)
