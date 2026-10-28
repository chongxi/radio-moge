"""One bounded, sequential plane-loss/control experiment with scene-disjoint data."""

import argparse
from collections import defaultdict
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json
from radio_moge.data import check_split
from radio_moge.planes import plane_config_dict


def read_nyu(path):
    rows = []
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            if row.get('dataset', 'nyuv2') != 'nyuv2':
                continue
            row['dataset'] = 'nyuv2'
            row['group'] = 'nyuv2:' + row['scene']
            for key in ('image', 'depth'):
                if key in row:
                    row[key] = str((path.parent / row[key]).resolve())
            rows.append(row)
    return rows


def select_scenes(rows, scenes, per_scene, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['group']].append(row)
    eligible = sorted(k for k, values in grouped.items() if len(values) >= per_scene)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if len(eligible) < scenes:
        raise ValueError('Insufficient independent scenes')
    result = []
    for group in eligible[:scenes]:
        candidates = sorted(grouped[group], key=lambda r: r['id'])
        rng.shuffle(candidates)
        result.extend(candidates[:per_scene])
    return result


def write_manifest(path, rows):
    temporary = path.with_suffix('.incoming')
    temporary.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    temporary.replace(path)


def identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def prepare(base, root, revision, expanded=False):
    root.mkdir(exist_ok=False)
    data = base.parents[1] / 'datasets'
    source = data / 'attention-full-mix-v1-20261022'
    scene_count = 256 if expanded else 32
    train = select_scenes(read_nyu(source / 'train.jsonl'), scene_count, 2, 42)
    validation = select_scenes(read_nyu(source / 'val-selection.jsonl'), 16, 1, 142)
    screenshot = [r for r in read_nyu(data / 'nyuv2-labeled-s42/test.jsonl')
                  if r['id'] == 'nyuv2_0271']
    assert len(screenshot) == 1
    check_split(train, validation)
    check_split(train, screenshot)
    check_split(validation, screenshot)
    # One image from 16 training scenes: labelled as training diagnostics, never held-out scores.
    train_probe = train[:64:4]
    reference = base / 'teacher-plane-subset-pilot-20261028'
    if expanded:
        assert json.loads((reference / 'completion-verification.json').read_text())['stage'] == 'passed'
        for name, expected in [('train', train[:64]), ('validation', validation),
                               ('train-probe', train_probe), ('screenshot', screenshot)]:
            previous = [json.loads(s) for s in (reference / f'{name}.jsonl').read_text().splitlines()]
            assert previous == expected, f'Original {name} panel changed'
    for name, records in [('train', train), ('validation', validation),
                          ('train-probe', train_probe), ('screenshot', screenshot)]:
        write_manifest(root / f'{name}.jsonl', records)
    parent = base / 'attention-24h/final-review/last.pt'
    plan = dict(
        purpose=('Expanded scene coverage with fixed plane-loss weights and adapter' if expanded
                 else 'Small-subset test of teacher-selected plane constraints'), revision=revision,
        parent=str(parent), parent_step=83494,
        parent_sha256='54a3e5783148f20c0676c6235131b5850dea323200885a442601a4a96605fdef',
        max_updates=300, max_seconds_per_arm=1800, lr=1e-5,
        train_images=len(train), train_scenes=scene_count, validation_images=len(validation),
        validation_scenes=16, screenshot_ids=[r['id'] for r in screenshot],
        effective_batch=4, seed=42, augment=False,
        plane_weights=dict(plane_flatness=20.0, plane_normal=2.0, plane_anchor=.1),
        weight_policy='Prespecified moderate auxiliary weights; no validation tuning.',
        comparison='Plane arm first; control matches its actual completed updates, at most 300.',
        initialization='Same adapter/statistics; both reset Adam, scheduler, seed and sampler.',
        endpoint='Final checkpoint at matched updates, not validation-selected best.',
        primary_metric='Fixed teacher masks: per-image mean best-fit-plane RMS / spatial extent.',
        secondary_metrics=['plane normal angle', 'GT depth', 'outside-plane teacher depth',
                           'teacher log-depth edge gradients', 'feature cosine and magnitude'],
        limitations=['Small custom subset, not an official benchmark.',
                     'Teacher planes are pseudo-labels.',
                     'Frozen pretrained adapter has previously seen the training pool.',
                     'Screenshot is a separate qualitative test, not used for tuning.'])
    if expanded:
        plan.update(profile='coverage-512', reference_pilot=str(reference),
                    evaluation_steps=list(range(0, 301, 50)), original_64_retained=True,
                    diagnostic_images=16, nominal_train_passes=300 * 4 / len(train),
                    changed_factors=['Training coverage 64 images/32 scenes to 512/256',
                                     'Every-50-step validation and fixed training diagnostics'],
                    limitations=plan['limitations'] + [
                        'Same 300-update budget gives fewer passes per image than the 64-image run.',
                        'Fixed original 16-image validation is small; this remains a pilot.'])
    atomic_json(plan, root / 'plan.json')
    original = json.loads((base / 'attention-24h/config.json').read_text())
    for arm in ('plane', 'control'):
        config = deepcopy(original)
        config['code_revision'] = revision
        config['output_dir'] = str(root / arm)
        config['data'].update(train_manifest=str(root / 'train.jsonl'),
                              val_manifest=str(root / 'validation.jsonl'),
                              buckets=[[480, 640]], workers=0, augment=False)
        config['data'].pop('mixture_weights', None)
        config['train'].update(steps=300, max_seconds=1800, lr=1e-5, lr_schedule='constant',
                               batch_size=2, accumulation=2, warmup=0, log_every=10,
                               update_norm_every=10, save_every=100, eval_every=50 if expanded else 300,
                               eval_at_start=True, tensorboard=False)
        config['planes'] = plane_config_dict()
        if expanded:
            config['data']['train_probe_manifest'] = str(root / 'train-probe.jsonl')
        config['loss'].update({k: v if arm == 'plane' else 0.
                               for k, v in plan['plane_weights'].items()})
        config['evaluation'].update(planes=True, visualizations=2, limit=None,
                                    geometry_ids=[validation[0]['id'], validation[1]['id'],
                                                  train_probe[0]['id'], 'nyuv2_0271'])
        config['tracking'].update(name=f'{root.name}-{arm}', images=2,
                                  tags=['plane-pilot', arm, f'subset-{len(train)}', 'lr-1e-5'])
        atomic_json(config, root / f'{arm}-config.json')
    return plan


def assess(root):
    import torch
    from radio_moge.compare import fingerprint
    from radio_moge.data import read_manifest
    from radio_moge.evaluate import evaluate
    from radio_moge.losses import FeatureStatistics
    from radio_moge.system import build_system

    config = json.loads((root / 'plane-config.json').read_text())
    has_probe_curve = bool(config['data'].get('train_probe_manifest'))
    system, adapter = build_system(config, 'cuda')
    stats = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim'])
    stats.cuda()
    initial_hashes = {}
    for arm in ('plane', 'control'):
        state = torch.load(root / arm / 'initial.pt', map_location='cpu', weights_only=True)
        initial_hashes[arm] = fingerprint(state['adapter'])
        del state
    assert initial_hashes['plane'] == initial_hashes['control']
    atomic_json(initial_hashes, root / 'initial-adapter-verification.json')
    for name, path in [('initial', root / 'plane/initial.pt'),
                       ('control', root / 'control/latest.pt'),
                       ('plane', root / 'plane/latest.pt')]:
        state = torch.load(path, map_location='cpu', weights_only=True)
        adapter.load_state_dict(state['adapter'])
        stats.load_state_dict(state['statistics'])
        step = state['step']
        del state
        for partition in ('train-probe', 'screenshot'):
            if partition == 'train-probe' and has_probe_curve:
                # These exact weights/images were already evaluated in the training process.
                source = root / ('plane' if name == 'initial' else name) / f'train-probe-{step:07d}'
                destination = root / 'assessment' / name / partition
                destination.mkdir(parents=True, exist_ok=False)
                for item in source.iterdir():
                    temporary = destination / (item.name + '.incoming')
                    shutil.copyfile(item, temporary)
                    temporary.replace(destination / item.name)
                print(json.dumps(dict(assessment=name, partition=partition, step=step,
                                      reused_evaluation=str(source))), flush=True)
                continue
            result = evaluate(system, adapter, stats, read_manifest(root / f'{partition}.jsonl'),
                              config, root / 'assessment' / name / partition)
            print(json.dumps(dict(assessment=name, partition=partition, step=step,
                                  metrics=result)), flush=True)


def run(base, root, repo, revision):
    plan = json.loads((root / 'plan.json').read_text())
    assert plan['max_updates'] == 300 and plan['max_seconds_per_arm'] == 1800
    assert plan['revision'] == revision
    state = dict(stage='starting', pid=os.getpid(), started_at=time.time(), revision=revision)

    def record(**changes):
        state.update(changes, updated_at=time.time())
        atomic_json(state, root / 'status.json')

    with (base / 'attention-24h/.training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'status.json').exists():
            raise FileExistsError('Already launched; inspect existing status, do not relaunch')
        active = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError('GPU already has compute processes')
        parent = Path(plan['parent'])
        parent_identity = identity(parent)
        with parent.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        assert digest == plan['parent_sha256'] and identity(parent) == parent_identity
        record(parent_sha256=digest, parent_identity=parent_identity)
        env = dict(os.environ, PYTHONPATH=str(repo / 'src'), OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')

        def child(command, name):
            with (root / f'{name}-console.log').open('w') as log:
                p = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
                record(stage=name, child_pid=p.pid, command=command)
                while p.poll() is None:
                    time.sleep(10)
                    record()
            record(child_pid=None, child_exit=p.returncode)
            if p.returncode:
                raise RuntimeError(f'{name} failed: inspect its log before any action')

        try:
            matched_steps = plan['max_updates']
            for arm in ('plane', 'control'):
                command = [sys.executable, '-u', '-m', 'radio_moge.train', '--config',
                           str(root / f'{arm}-config.json'), '--init', str(parent),
                           '--execution-revision', revision, '--stop-after', str(matched_steps)]
                child(command, arm)
                stop = json.loads((root / arm / 'stop-report.json').read_text())
                assert stop['final_evaluation'] == 'complete' and 0 < stop['step'] <= 300
                if arm == 'plane':
                    matched_steps = stop['step']
                else:
                    assert stop['step'] == matched_steps, 'Unequal update counts; inspect'
                record(**{arm + '_stop': stop}, matched_steps=matched_steps)
            child([sys.executable, '-u', str(Path(__file__).resolve()),
                   '--root', str(root), '--assess'], 'assessment')
            assert identity(parent) == parent_identity
            record(stage='complete', finished_at=time.time(), parent_unchanged=True)
        except Exception as error:
            record(stage='needs_inspection', error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--revision')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--expanded', action='store_true', help='512 images from 256 NYU scenes')
    parser.add_argument('--assess', action='store_true')
    args = parser.parse_args()
    if args.assess:
        assess(args.root)
    elif args.prepare:
        print(json.dumps(prepare(args.base, args.root, args.revision, args.expanded)))
    else:
        run(args.base, args.root, args.repo, args.revision)
