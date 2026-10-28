"""Bounded original-versus-joint adapter trial; architecture changes start at identity."""

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json, atomic_save


ARMS = ('control', 'joint')
EXTENSION = dict(refinement='joint_attention', refinement_width=256,
                 refinement_blocks=2, refinement_heads=8, refinement_activation='gelu')
PARENT_SHA = '54a3e5783148f20c0676c6235131b5850dea323200885a442601a4a96605fdef'


def read(path):
    return json.loads(path.read_text())


def identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def prepare(base, root, revision):
    reference = base / 'teacher-plane-ownflat512-pilot-20261028'
    assert read(reference / 'completion-verification.json')['stage'] == 'passed'
    root.mkdir(exist_ok=False)
    manifest_hashes = {}
    for partition in ('train', 'validation', 'train-probe', 'screenshot'):
        payload = (reference / f'{partition}.jsonl').read_bytes()
        temporary = root / f'{partition}.incoming'
        temporary.write_bytes(payload)
        temporary.replace(root / f'{partition}.jsonl')
        manifest_hashes[partition] = hashlib.sha256(payload).hexdigest()
    plan = dict(profile='joint-cosine-512', revision=revision, arms=list(ARMS),
                purpose='Warm-start adapter refinement: original versus joint spatial/summary attention',
                parent=str(base / 'attention-24h/final-review/last.pt'), parent_step=83494,
                parent_sha256=PARENT_SHA, reference_pilot=str(reference),
                manifest_sha256=manifest_hashes, train_images=512, train_scenes=256,
                validation_images=16, diagnostic_images=16, effective_batch=4, seed=42,
                max_updates=600, max_seconds_per_arm=1200, lr=1e-5,
                evaluation_steps=list(range(0, 601, 100)), nominal_train_passes=600*4/512,
                arm_labels=dict(control='Original adapter', joint='Joint spatial + summary, 2 blocks'),
                joint_extension=EXTENSION,
                initialization='Same parent base weights/statistics; zero residual heads; fresh Adam/RNG/cursor.',
                training_loss='Feature only: normalized spatial MSE + summary MSE + 0.1 spatial cosine error.',
                primary_metrics=['spatial/cosine', 'summary/cosine'],
                speed_target_absolute_cosine_gain=.001,
                clocks=['optimizer_seconds excludes evaluation/loading', 'elapsed_seconds includes them'],
                secondary_metrics=['feature norm ratio', 'GT depth AbsRel/RMSE/delta1',
                                   'plane flatness', 'depth edges', 'adapter+decoder branch_ms'],
                endpoint='Matched final updates, never validation-selected best',
                order=list(ARMS), stop_policy='At most 600 updates or 1200s per arm; no automatic extension.',
                limitations=['Warm-start residual-extension comparison, not from-scratch architecture ranking.',
                             'Joint arm has more parameters; improvement alone cannot isolate token mixing.',
                             '512 NYU images and 16 validation images are a screening pilot, not a benchmark.',
                             'Training probe is a fixed diagnostic subset of training scenes.',
                             'Cosine is a proxy; depth, amplitude and edges decide practical utility.',
                             'Single seed; small differences need repeatability checks.'])
    atomic_json(plan, root / 'plan.json')
    original = read(base / 'attention-24h/config.json')
    template = read(reference / 'plane-config.json')
    for arm in ARMS:
        config = deepcopy(template)
        config['code_revision'] = revision
        config.pop('plane_flatness_mode', None)
        config['adapter'] = deepcopy(original['adapter'])
        if arm == 'joint':
            config['adapter'].update(EXTENSION)
        config['output_dir'] = str(root / arm)
        config['data'].update(train_manifest=str(root / 'train.jsonl'),
                              val_manifest=str(root / 'validation.jsonl'),
                              train_probe_manifest=str(root / 'train-probe.jsonl'))
        config['train'].update(steps=600, max_seconds=1200, eval_every=100,
                               record_cosine_speed=True, save_every=200)
        config['loss'] = {key: 0. for key in template['loss']}
        config['loss'].update(spatial=1., summary=1., cosine=.1)
        config['tracking'].update(name=f'{root.name}-{arm}', group=root.name,
                                  tags=['adapter-prior', 'joint-cosine', arm, 'subset-512'])
        atomic_json(config, root / f'{arm}-config.json')
    return plan


def make_adapter(parent, arm):
    import torch
    from radio_moge.adapters import GeometryAdapter

    torch.manual_seed(42)
    adapter = GeometryAdapter(**parent['adapter_config'], **(EXTENSION if arm == 'joint' else {}))
    missing, unexpected = adapter.load_state_dict(parent['adapter'], strict=False)
    assert not unexpected and all(k.startswith('refinement.') for k in missing)
    assert bool(missing) == (arm == 'joint')
    state = adapter.state_dict()
    assert all(torch.equal(state[k], value) for k, value in parent['adapter'].items())
    return adapter


def preflight(root):
    import torch
    from radio_moge.compare import fingerprint
    from radio_moge.data import read_manifest, TrainingImages
    from radio_moge.losses import FeatureStatistics, weighted_loss
    from radio_moge.system import build_system

    plan = read(root / 'plan.json')
    parent = torch.load(plan['parent'], map_location='cpu', weights_only=True, mmap=True)
    config = read(root / 'control-config.json')
    system, unused = build_system(config, 'cuda')
    del unused
    data = TrainingImages(read_manifest(root / 'validation.jsonl'), augment=False)
    images = torch.stack([data[(i, 42, 480, 640)] for i in range(2)]).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs, target = system.encode(images, config['adapter']['indices'],
                                       config['evaluation']['num_tokens'])
    statistics = FeatureStatistics(parent['adapter_config']['target_dim'],
                                   parent['adapter_config']['target_summary_dim']).cuda()
    statistics.load_state_dict(parent['statistics'])
    proof = dict(stage='passed', optimizer_updates=0, parent_step=parent['step'],
                 base_fingerprint=fingerprint(parent['adapter']), arms={}, images=2)
    baseline = None
    for arm in ARMS:
        adapter = make_adapter(parent, arm)
        save = {key: parent[key] for key in ('statistics', 'model_config', 'config', 'step')}
        save.update(adapter=adapter.state_dict(), adapter_config=adapter.config,
                    parent_sha256=PARENT_SHA, initialization='identity extension, optimizer reset')
        atomic_save(save, root / f'{arm}-init.pt')
        del save
        adapter.cuda().train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            predicted = adapter(inputs)
        if baseline is None:
            baseline = (predicted.spatial.detach().clone(), predicted.summary.detach().clone())
        differences = [float((value - ref).abs().max()) for value, ref in
                       zip((predicted.spatial, predicted.summary), baseline)]
        assert differences == [0., 0.], 'Initial function changed'
        loss = weighted_loss(statistics.loss(predicted, target), config['loss'])
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
        proof['arms'][arm] = dict(parameters=sum(p.numel() for p in adapter.parameters()),
                                 config=adapter.config, max_initial_output_difference=differences,
                                 finite_backward=True, feature_loss=float(loss.detach()))
        del predicted, adapter, loss
        torch.cuda.empty_cache()
    atomic_json(proof, root / 'preflight-verification.json')


def assess(root):
    import torch
    from radio_moge.adapters import GeometryAdapter
    from radio_moge.compare import fingerprint
    from radio_moge.data import read_manifest
    from radio_moge.evaluate import evaluate
    from radio_moge.losses import FeatureStatistics
    from radio_moge.system import build_system

    config = read(root / 'control-config.json')
    system, unused = build_system(config, 'cuda')
    del unused
    initial_base = {}
    for arm in ARMS:
        state = torch.load(root / arm / 'initial.pt', map_location='cpu', weights_only=True, mmap=True)
        initial_base[arm] = fingerprint({k: v for k, v in state['adapter'].items()
                                         if not k.startswith('refinement.')})
        del state
    assert len(set(initial_base.values())) == 1
    assert initial_base['control'] == read(root / 'preflight-verification.json')['base_fingerprint']
    atomic_json(dict(shared_base_fingerprints=initial_base, stage='passed',
                     full_architectures_identical=False), root / 'initial-adapter-verification.json')
    for name in ('initial', *ARMS):
        arm = 'control' if name == 'initial' else name
        state = torch.load(root / arm / ('initial.pt' if name == 'initial' else 'latest.pt'),
                           map_location='cpu', weights_only=True, mmap=True)
        adapter = GeometryAdapter(**state['adapter_config']).cuda()
        adapter.load_state_dict(state['adapter'])
        statistics = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim']).cuda()
        statistics.load_state_dict(state['statistics'])
        step = state['step']
        del state
        output = root / 'assessment' / name
        source = root / arm / f'train-probe-{step:07d}'
        shutil.copytree(source, output / 'train-probe')
        evaluate(system, adapter, statistics, read_manifest(root / 'screenshot.jsonl'),
                 config, output / 'screenshot')
        del adapter, statistics
        torch.cuda.empty_cache()


def run(base, root, repo, revision):
    plan = read(root / 'plan.json')
    assert plan['revision'] == revision and plan['arms'] == list(ARMS)
    assert plan['max_updates'] == 600 and plan['max_seconds_per_arm'] == 1200
    if (root / 'status.json').exists():
        raise FileExistsError('Already launched; inspect, never relaunch')
    state = dict(stage='starting', pid=os.getpid(), started_at=time.time(), revision=revision)

    def record(**changes):
        state.update(changes, updated_at=time.time())
        atomic_json(state, root / 'status.json')

    with (base / 'attention-24h/.training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                          '--format=csv,noheader'], text=True).strip()
        assert not active, 'GPU already busy'
        parent = Path(plan['parent'])
        parent_identity = identity(parent)
        with parent.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        assert digest == PARENT_SHA and identity(parent) == parent_identity
        record(parent_identity=parent_identity, parent_sha256=digest)
        env = dict(os.environ, PYTHONPATH=str(repo / 'src'), OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')

        def child(command, name):
            with (root / f'{name}-console.log').open('w') as log:
                p = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
                record(stage=name, child_pid=p.pid)
                while p.poll() is None:
                    time.sleep(10)
                    record()
            record(child_pid=None, child_exit=p.returncode)
            assert p.returncode == 0, f'{name} failed; inspect log; no automatic retry'

        try:
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root)]
            child(command + ['--preflight'], 'preflight')
            matched_steps = 600
            for arm in ARMS:
                child([sys.executable, '-u', '-m', 'radio_moge.train', '--config',
                       str(root / f'{arm}-config.json'), '--init', str(root / f'{arm}-init.pt'),
                       '--execution-revision', revision, '--stop-after', str(matched_steps)], arm)
                stop = read(root / arm / 'stop-report.json')
                assert stop['final_evaluation'] == 'complete' and 0 < stop['step'] <= matched_steps
                if arm == 'control':
                    matched_steps = stop['step']
                assert stop['step'] == matched_steps, 'Unequal updates; inspect'
                record(**{arm + '_stop': stop}, matched_steps=matched_steps)
            child(command + ['--assess'], 'assessment')
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
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--assess', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.base, args.root, args.revision)
    elif args.preflight:
        preflight(args.root)
    elif args.assess:
        assess(args.root)
    else:
        run(args.base, args.root, args.repo, args.revision)
