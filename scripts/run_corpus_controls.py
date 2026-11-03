"""Run a frozen four-arm control: all real-model gates first, no automatic retry.

Each arm initializes from the same archive with fresh AdamW. The actual trainer's
initial state and every consumed input are compared across arms. The original
GPU lock covers the whole experiment. An incomplete arm stops the controller.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.corpus_admission import (checked_json, checked_path, file_ref,
    implementation_hashes, open_training_gate, seal_acceptance, validate_pre_model_evidence)
from radio_moge.geometry_cache import publish_json
from radio_moge.train import validate_config
from verify_corpus_stage_startup import equal_finite


def verify_initial_state(old, new, config, first=None):
    if (old['step'] != 232288 or new['step'] != 0 or new['cursor'] != 0
            or new['config'] != config or new['optimizer']['state']
            or new['scheduler']['last_epoch'] != 0
            or new['adapter_config'] != old['adapter_config']
            or new['model_config'] != old['model_config']):
        raise ValueError('Control trainer did not initialize a fresh matching experiment')
    for section in ('adapter', 'statistics'):
        if not equal_finite(new[section], old[section]):
            raise ValueError('Actual initial weights/statistics differ from the archived baseline')
    if first is not None:
        for section in ('adapter', 'statistics', 'optimizer', 'scheduler', 'rng', 'revisions'):
            if not equal_finite(new[section], first[section]):
                raise ValueError(f'Control initial {section} differs between arms')
        if (new['data_digests'].keys() != first['data_digests'].keys()
                or any(v != first['data_digests'][k] for k,v in new['data_digests'].items()
                       if k not in ('train', 'geometry'))):
            raise ValueError('Replay/heldout identities changed between controls')


def inspect_initial(directory, source, config, first_initial):
    path = directory / 'training/initial.pt'
    old = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    new = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    first = torch.load(first_initial, map_location='cpu', weights_only=True, mmap=True) if first_initial else None
    verify_initial_state(old, new, config, first)
    result = dict(stage='actual_fresh_control_initial_state_verified', source_step=232288,
        step=0, cursor=0, optimizer_state_entries=0, adapter_statistics_bitwise_archive=True,
        optimizer_scheduler_rng_bitwise_first_arm=first is not None,
        first_arm_reference=first is None, initial=file_ref(path),
        first_initial=file_ref(first_initial) if first_initial else None,
        config=file_ref(directory / 'training-config.json'), independent_trainer_process=True,
        optimizer_updates_by_this_verifier=0, gpu_used=False, code=file_ref(Path(__file__)))
    publish_json(directory / 'initial-verification.json', result)
    return result


def inspect_completed(directory, config, expected_input_sha):
    training = directory / 'training'
    stop = checked_json(file_ref(training / 'stop-report.json'))
    records = [json.loads(line) for line in (training / 'train.jsonl').read_text().splitlines()]
    audit = checked_json(file_ref(training / 'control-input-audit.json'))
    checked_path(audit['records'])
    if (stop['step'] != 250 or stop['reason'] != 'step_limit' or stop['final_evaluation'] != 'complete'
            or [r['step'] for r in records] != list(range(1, 251))
            or any(not math.isfinite(v) for r in records for v in r.values() if isinstance(v, (int, float)))
            or audit['microbatches'] != 4000 or audit['replay_microbatches'] != 2000
            or (expected_input_sha and audit['records']['sha256'] != expected_input_sha)):
        raise ValueError('Incomplete control or different actual input sequence')
    disabled = config['corpus_loss_control']['geometry_multiplier'] == 0
    if any(r['gradient_norm'] <= 0 or r['feature_weighted'] <= 0
           or (disabled and r['geometry_weighted'] != 0) for r in records):
        raise ValueError('Control losses or gradients violate the frozen experiment')
    final = torch.load(training / 'latest.pt', map_location='cpu', weights_only=True, mmap=True)
    initial = torch.load(training / 'initial.pt', map_location='cpu', weights_only=True, mmap=True)
    if (final['step'] != 250 or final['cursor'] != 4000 or final['config'] != config
            or final['data_digests'] != initial['data_digests']
            or not equal_finite(final['statistics'], initial['statistics'])
            or {int(v['step']) for v in final['optimizer']['state'].values()} != {250}):
        raise ValueError('Final control checkpoint or optimizer counters differ')
    for state in [final['adapter'], *final['optimizer']['state'].values()]:
        if any(not bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v)):
            raise ValueError('Nonfinite control checkpoint')
    result = dict(stage='fixed_budget_control_completed', optimizer_updates=250,
        actual_input_audit=file_ref(training / 'control-input-audit.json'),
        input_sha256=audit['records']['sha256'], checkpoint=file_ref(training / 'latest.pt'),
        final_checkpoint_finite=True, all_training_records_finite=True,
        initial_verification=file_ref(directory / 'initial-verification.json'),
        stop_report=file_ref(training / 'stop-report.json'),
        training_log=file_ref(training / 'train.jsonl'), validation_log=file_ref(training / 'validation.jsonl'),
        run=file_ref(training / 'wandb-run.json'), config=file_ref(directory / 'training-config.json'),
        fixed_final_step_selected=True, evaluation_pairing_review_pending=True)
    publish_json(directory / 'completed.json', result)
    return result


def run(experiment_path, lock_path):
    experiment_ref = file_ref(experiment_path)
    experiment = checked_json(experiment_ref)
    root = experiment_path.parent
    plan = checked_json(experiment['plan'])
    code = Path(experiment['execution_root'])
    if (experiment['stage'] != 'four_control_nonmodel_gates_ready'
            or experiment['implementation_sha256'] != implementation_hashes()
            or code != Path(__file__).resolve().parents[1] or len(experiment['arms']) != 4
            or not lock_path.is_absolute() or not lock_path.is_file()):
        raise ValueError('Frozen four-control execution and original GPU lock required')
    status_path = root / 'controller-status.json'
    if status_path.exists():
        raise FileExistsError('Inspect the prior controller; automatic restart is forbidden')
    state = dict(stage='starting', training_started=False, started_at=time.time(),
                 experiment=experiment_ref, completed_arms=[], optimizer_updates=0)
    def status(stage, **values):
        state.update(stage=stage, updated_at=time.time(), **values)
        atomic_json(state, status_path)
        print(json.dumps(state), flush=True)
    env = dict(os.environ, PYTHONPATH=str(code / 'src') + ':' + str(code / 'scripts'),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    torch.set_num_threads(1)
    child = None
    try:
        with lock_path.open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepared, reference_probe = {}, None
            for name, arm in experiment['arms'].items():
                ready = checked_json(arm['readiness'])
                config = checked_json(arm['config'])
                directory = Path(arm['directory'])
                if (ready['initial_checkpoint'] != plan['common_initialization']['file']
                        or ready['source_step'] != 232288 or ready['accounting_report'] != plan['accounting']
                        or ready['training_config'] != arm['config'] or directory.parent != root):
                    raise ValueError('Control readiness changed its frozen source or scope')
                validate_config(config, corpus_preflight=True)
                validate_pre_model_evidence(ready['accounting_report'], ready['checks'])
                status('actual_model_gate_running', arm=name, gpu_used=True)
                subprocess.run([sys.executable, '-c',
                    'from pathlib import Path;import sys;from probe_corpus_model import run;'
                    'run(Path(sys.argv[1]),Path(sys.argv[2]))',
                    arm['readiness']['path'], str(directory / 'model-zero-update')],
                    cwd=code, env=env, check=True)
                model_ref = file_ref(directory / 'model-zero-update/report.json')
                proof = checked_json(model_ref)
                selection_ref = file_ref(directory / 'model-zero-update/selection.json')
                if reference_probe:
                    if (proof['state_before'] != reference_probe['state_before']
                            or selection_ref['sha256'] != reference_probe['selection_sha256']):
                        raise ValueError('Control model gates used different initial states or views')
                else:
                    reference_probe = dict(state_before=proof['state_before'], selection_sha256=selection_ref['sha256'])
                checks = dict(ready['checks'], model_zero_update=model_ref)
                gate_path = directory / 'acceptance.json'
                seal_acceptance(ready['accounting_report'], checks, config, gate_path)
                runtime = deepcopy(config)
                runtime['data']['geometry_corpus'] = file_ref(gate_path)
                publish_json(directory / 'training-config.json', runtime)
                open_training_gate(runtime, init=ready['initial_checkpoint']['path'])
                prepared[name] = dict(config=runtime, directory=directory, ready=ready, model=model_ref)
            publish_json(root / 'all-model-gates.json', dict(stage='all_four_actual_model_gates_complete',
                optimizer_updates=0, training_started=False, same_initial_model_states=True,
                same_selected_views=True, cases=sum(checked_json(x['model'])['cases'] for x in prepared.values()),
                models={name:x['model'] for name,x in prepared.items()}, experiment=experiment_ref))
            first_initial, expected_input_sha, completed = None, None, {}
            for name, arm in prepared.items():
                checked_path(experiment_ref)
                config, directory = arm['config'], arm['directory']
                status('training_control', arm=name, training_started=True, optimizer_updates=None)
                with (directory / 'trainer-console.log').open('x') as log:
                    child = subprocess.Popen([sys.executable, '-m', 'radio_moge.train', '--config',
                        str(directory / 'training-config.json'), '--init', arm['ready']['initial_checkpoint']['path']],
                        cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + config['train']['max_seconds'] + 900
                    initial = directory / 'training/initial.pt'
                    while not initial.exists() and child.poll() is None:
                        if time.monotonic() > deadline:
                            raise TimeoutError('Control did not initialize within its bound')
                        time.sleep(1)
                    if not initial.exists():
                        raise RuntimeError('Control trainer exited before initial checkpoint')
                    inspect_initial(directory, arm['ready']['initial_checkpoint']['path'], config, first_initial)
                    if first_initial is None:
                        first_initial = initial
                    code_result = child.wait(timeout=max(1., deadline - time.monotonic()))
                    if code_result:
                        raise RuntimeError(f'Control trainer exited with code {code_result}')
                    child = None
                result = inspect_completed(directory, config, expected_input_sha)
                if expected_input_sha is None:
                    expected_input_sha = result['input_sha256']
                completed[name] = file_ref(directory / 'completed.json')
                status('control_completed', completed_arms=list(completed), optimizer_updates=250 * len(completed))
            publish_json(root / 'report.json', dict(stage='four_fixed_budget_controls_completed',
                completed=completed, actual_input_sha256=expected_input_sha,
                experiment=experiment_ref, model_gates=file_ref(root / 'all-model-gates.json'),
                total_optimizer_updates=1000, updates_per_arm=250, evaluation_pairing_review_pending=True,
                code=file_ref(Path(__file__)), elapsed_seconds=time.time()-state['started_at']))
            status('complete', arm=None)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=30)
        status('failed', error=type(error).__name__, message=str(error)[:1000])
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--gpu-lock', required=True, type=Path)
    args = parser.parse_args()
    run(args.experiment.resolve(), args.gpu_lock.resolve())
