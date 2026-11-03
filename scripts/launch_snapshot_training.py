"""Wait without a GPU, then probe, seal, and train the requested frozen stage.

Readiness is a reviewed accounting plan, not a completion counter. Missing
evidence waits; invalid evidence fails. The existing GPU lock covers the actual
zero-update probe and trainer. The final training code is archived separately
before readiness publication, so preparation can continue without code drift.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def file_ref(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as file:
        for part in iter(lambda: file.read(8 << 20), b''):
            h.update(part)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(ref):
    if set(ref) != {'path', 'sha256'} or not Path(ref['path']).is_absolute() or file_ref(ref['path']) != ref:
        raise ValueError('Launch evidence changed or is not an absolute hash reference')
    return json.loads(Path(ref['path']).read_text())


def publish(path, value, *, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + '.incoming')
    with temporary.open('x') as file:
        json.dump(value, file, sort_keys=True, allow_nan=False)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def validate_bindings(request, ready, payload, snapshot, scope):
    if (request.get('training_authorized_after_checks') is not True
            or ready.get('training_config') != request['training_config']
            or ready.get('initial_checkpoint') != request['initial_checkpoint']
            or ready.get('source_step') != request['source_step']
            or ready.get('snapshot') != request['snapshot']
            or scope.get('snapshot') != request['snapshot']
            or scope.get('kind') != 'frozen_candidate_subset'
            or scope.get('expected_counts') != snapshot['dataset_counts']):
        raise ValueError('Readiness changed the requested stage, configuration, or checkpoint')
    if (payload.get('stage') != 'snapshot_payload_verified_not_accepted'
            or payload.get('mode') != 'census' or payload.get('snapshot') != request['snapshot']
            or payload.get('all_snapshot_packs_verified') is not True
            or payload.get('all_snapshot_members_verified') is not True
            or payload.get('failed_packs') != 0
            or payload.get('verified_packs') != snapshot['processed_packs']
            or payload.get('verified_members') != snapshot['processed_entries']):
        raise ValueError('Every requested snapshot payload must finish verification')
    if set(ready.get('checks', {})) != {'source_quality', 'coverage', 'heldout_isolation', 'loss_tests'}:
        raise ValueError('The four nonmodel gates are required before starting a model')


def run(request_path, *, once=False, poll_seconds=30):
    request_ref = file_ref(request_path)
    request = checked(request_ref)
    config = checked(request['training_config'])
    snapshot = checked(request['snapshot'])
    if ('geometry_corpus' in config['data'] or request['source_step'] != 232288
            or request.get('training_authorized_after_checks') is not True):
        raise ValueError('Use the unsealed requested stage configuration and archived feature checkpoint')
    output = Path(request['launch_directory'])
    output.mkdir(exist_ok=False)
    state = dict(stage='waiting_for_data_and_loss_checks', started_at=time.time(),
        training_started=False, training_admission=False, optimizer_updates=0,
        request=request_ref, gpu_used=False, readiness_plan=request['readiness_plan'])

    def status(stage, **extra):
        state.update(stage=stage, updated_at=time.time(), **extra)
        publish(output / 'status.json', state, replace=True)

    try:
        status('waiting_for_data_and_loss_checks')
        while not Path(request['readiness_plan']).is_file():
            if once:
                return state
            time.sleep(poll_seconds)
        if file_ref(request_path) != request_ref:
            raise ValueError('Launch request changed while waiting')
        ready_ref = file_ref(request['readiness_plan'])
        ready = checked(ready_ref)
        payload_ref = file_ref(request['payload_report'])
        payload = checked(payload_ref)
        accounting = checked(ready['accounting_report'])
        scope = checked(accounting['scope'])
        validate_bindings(request, ready, payload, snapshot, scope)
        code = Path(request['execution_root']).resolve(strict=True)
        sys.path[:0] = [str(code / 'src'), str(code / 'scripts')]
        from radio_moge.corpus_admission import (seal_acceptance, validate_pre_model_evidence,
                                                 open_training_gate)
        from radio_moge.train import validate_config
        validate_config(config, corpus_preflight=True)
        validate_pre_model_evidence(ready['accounting_report'], ready['checks'])
        lock_path = Path(request['gpu_lock'])
        if not lock_path.is_absolute() or not lock_path.is_file():
            raise ValueError('The original GPU lock must already exist')
        env = dict(os.environ, PYTHONPATH=str(code / 'src') + ':' + str(code / 'scripts'),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1')
        with lock_path.open('r+') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    status('waiting_for_existing_gpu_lock')
                    time.sleep(poll_seconds)
            # Recheck after waiting; a stale plan cannot start a model.
            checked(request_ref)
            checked(ready_ref)
            checked(payload_ref)
            checked(request['training_config'])
            validate_pre_model_evidence(ready['accounting_report'], ready['checks'])
            probe = output / 'model-zero-update'
            status('model_zero_update_running', gpu_used=True)
            # Parent owns the existing lock. Use the probe function in a child
            # so its model allocations are released before the trainer starts.
            subprocess.run([sys.executable, '-c',
                'from pathlib import Path; import sys; from probe_corpus_model import run; '
                'run(Path(sys.argv[1]), Path(sys.argv[2]))',
                str(Path(request['readiness_plan'])), str(probe)],
                cwd=code, env=env, check=True)
            checks = dict(ready['checks'], model_zero_update=file_ref(probe / 'report.json'))
            gate_path = output / 'acceptance.json'
            seal_acceptance(ready['accounting_report'], checks, config, gate_path)
            runtime = deepcopy(config)
            runtime['data']['geometry_corpus'] = file_ref(gate_path)
            config_path = output / 'training-config.json'
            publish(config_path, runtime)
            validate_config(runtime)
            open_training_gate(runtime, init=request['initial_checkpoint']['path'])
            status('training_running', training_started=True, training_admission=True,
                   optimizer_updates=None, training_config=file_ref(config_path))
            subprocess.run([sys.executable, '-m', 'radio_moge.train', '--config', str(config_path),
                '--init', request['initial_checkpoint']['path']], cwd=code, env=env, check=True)
            status('training_process_completed', optimizer_updates=None,
                   training_output=config['output_dir'])
        return state
    except BaseException as error:
        status('failed', error=type(error).__name__, message=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--once', action='store_true', help='Observe missing readiness once; never bypass it')
    args = parser.parse_args()
    run(args.request, once=args.once)
