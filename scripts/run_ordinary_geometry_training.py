"""Publish a separately bound admission, then run one explicitly bounded ordinary fork.

Preparation requires completed CPU and actual CUDA evidence. Run never retries,
changes the budget, or interprets a partial wall-clock endpoint as fixed250.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import time

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.ordinary_training import (CPU_REQUIREMENTS, MODEL_REQUIREMENTS, config_digest,
    digest, equal_state, implementation_digest, validate_ordinary_config, validate_ordinary_performance)
from radio_moge.train import train
from radio_moge.projection_fork import ordinary_projection_scope


def model_backend_scope(model):
    """Keep runtime forwards separate from the exact diagnostic backward."""
    modes = None
    if (model.get('default_backend_gradient_bitwise_equality_claimed') is not False
            or model.get('diagnostic_backend_used_for_training') is not False
            or not model.get('feature_cases')):
        raise ValueError('Runtime and diagnostic attention scopes are not explicit')
    for case in model['feature_cases']:
        current = case['runtime_attention_modes']
        if (set(current) != {'flash', 'mem_efficient', 'math', 'cudnn'}
                or any(type(v) is not bool for v in current.values())
                or not any(current.values())
                or case['runtime_attention_backend_restored'] is not True
                or case['runtime_forward']['runtime_feature_pixels_losses_exact'] is not True
                or case['runtime_forward']['backward_performed'] is not False
                or case['diagnostic_math_backward']['feature_pixels_losses_gradients_exact'] is not True
                or (modes is not None and current != modes)):
            raise ValueError('Actual model attention backend scope differs between cases')
        modes = dict(current)
    return modes


def require_runtime_backend(expected):
    actual = {key: getattr(torch.backends.cuda, key+'_sdp_enabled')() for key in expected}
    if actual != expected:
        raise ValueError('Runtime attention backend differs from the actual model gate')


def prepare(gate_dir, execution_revision):
    request = checked_json(file_ref(gate_dir/'request.json'))
    config, performance = checked_json(request['config']), checked_json(request['performance'])
    validate_ordinary_config(config)
    validate_ordinary_performance(config, performance)
    binding = request['binding']
    if (config_digest(config) != binding['config_sha256']
            or digest(performance) != binding['performance_sha256']
            or implementation_digest() != binding['implementation_sha256']):
        raise ValueError('Candidate config/profile/code differs from completed gate')
    refs = dict(cpu_evidence=file_ref(gate_dir/'cpu-report.json'),
                model_evidence=file_ref(gate_dir/'model-report.json'))
    for field, stage, requirements in (
            ('cpu_evidence', 'ordinary_production_loop_cpu_verified', CPU_REQUIREMENTS),
            ('model_evidence', 'ordinary_actual_model_zero_update_verified', MODEL_REQUIREMENTS)):
        proof = checked_json(refs[field])
        counter = 'production_optimizer_updates' if field == 'cpu_evidence' else 'optimizer_updates'
        if (proof['stage'] != stage or proof['binding'] != binding
                or proof[counter] != 0 or proof['training_admission'] is not False
                or any(proof.get(k) is not True for k in requirements)):
            raise ValueError('Incomplete or differently bound ordinary evidence')
    model = checked_json(refs['model_evidence'])
    runtime_modes = model_backend_scope(model)
    memory = model['memory']
    if (model['device'] != 'cuda' or model['observer'] != request['observer']
            or model['request'] != file_ref(gate_dir/'request.json')
            or memory['dense32_plus_future32_exercised'] is not True
            or memory['free_bytes_adjusted_to_peak_reservation'] < memory['minimum_required_free_bytes']):
        raise ValueError('Actual CUDA observer/memory proof incomplete')
    planned = checked_json(binding['planned_keys'])
    actual = [key for row in model['actual_scope'] if row['kind'] == 'actual_first_three_updates'
              for key in row['keys']]
    dense = [row for row in model['actual_scope'] if row['kind'] == 'diagnostic_repeated_dense32_not_training_stream']
    if (actual != planned[:96] or len(planned) != 8000 or len(dense) != 3
            or {tuple(row['keys'][0][2:]) for row in dense} != set(map(tuple, config['data']['buckets']))):
        raise ValueError('Actual model scope does not cover original stream and all full32 shapes')
    root = Path(config['output_dir']).parent
    if (root/'admission.json').exists() or (root/'run-request.json').exists():
        raise FileExistsError('Ordinary training preparation already published')
    root.mkdir(exist_ok=False)
    _, population = ordinary_projection_scope(binding)
    seal = dict(stage='ordinary_rgb_geometry_training_accepted', schema_version=1,
        training_admission=True, **binding, **refs, source_step=232288, optimizer_updates=250,
        rgb_images=population, rgb_digest=binding['data_digests']['train'],
        automatic_followup_training=False, no_new_baseline_inference=True,
        runtime_attention_modes=runtime_modes,
        exact_gradient_parity_backend='math_diagnostic_only',
        created_at=time.time(), publisher=file_ref(Path(__file__)))
    publish_json(root/'admission.json', seal)
    runtime = deepcopy(config)
    runtime['ordinary_geometry']['admission'] = file_ref(root/'admission.json')
    if config_digest(runtime) != binding['config_sha256']:
        raise ValueError('Publishing the admission changed the runtime objective')
    publish_json(root/'training-config.json', runtime)
    publish_json(root/'run-request.json', dict(config=file_ref(root/'training-config.json'),
        performance=request['performance'], admission=file_ref(root/'admission.json'),
        execution_revision=execution_revision, runner=file_ref(Path(__file__)),
        gate_request=file_ref(gate_dir/'request.json')))
    return file_ref(root/'run-request.json')


def verify_endpoint(stop, log, audit, initial, final, config):
    if (stop['step'] != 232538 or stop['reason'] != 'step_limit'
            or stop['final_evaluation'] != 'complete'):
        raise ValueError('Run did not complete its fixed250 endpoint; no automatic extension')
    if ([r['step'] for r in log] != list(range(232289, 232539))
            or any(not math.isfinite(v) for r in log for v in r.values() if isinstance(v, (int, float)))):
        raise ValueError('Actual update log is incomplete or nonfinite')
    if (audit['optimizer_updates'] != 250 or audit['cursor'] != 2000 or audit['presentations'] != 8000
            or audit['planned_keys_matched'] is not True or audit['unused_prefetch_excluded'] is not True
            or audit['actual_compute_groups_verified'] is not True):
        raise ValueError('Actual fixed250 input accounting differs')
    if (final['step'] != 232538 or final['cursor'] != 2000 or final['config'] != config
            or final['data_digests'] != initial['data_digests']
            or final['revisions'] != initial['revisions']
            or not equal_state(final['statistics'], initial['statistics'])
            or not final['optimizer']['state']
            or {int(v['step']) for v in final['optimizer']['state'].values()} != {232538}):
        raise ValueError('Actual final checkpoint/state contract differs')
    for section in (final['adapter'], *final['optimizer']['state'].values()):
        if any(not bool(torch.isfinite(v).all()) for v in section.values() if torch.is_tensor(v)):
            raise ValueError('Nonfinite final adapter/Adam state')


def run(request_path, lock_path):
    request = checked_json(file_ref(request_path))
    config, performance = checked_json(request['config']), checked_json(request['performance'])
    seal = checked_json(request['admission'])
    root, output = request_path.parent, Path(config['output_dir'])
    if (request['runner'] != file_ref(Path(__file__)) or output.parent != root
            or config['ordinary_geometry']['admission'] != request['admission']):
        raise ValueError('Ordinary run request path/runner differs')
    model = checked_json(seal['model_evidence'])
    if (model_backend_scope(model) != seal['runtime_attention_modes']
            or seal['exact_gradient_parity_backend'] != 'math_diagnostic_only'):
        raise ValueError('Published runtime/diagnostic backend scopes changed')
    require_runtime_backend(seal['runtime_attention_modes'])
    if (root/'status.json').exists() or output.exists():
        raise FileExistsError('Ordinary training already attempted; no automatic retry')
    status = dict(stage='loading_metadata_and_actual_initial_state', started_at=time.time(),
                  request=file_ref(request_path), optimizer_updates=0)
    atomic_json(status, root/'status.json')
    try:
        with lock_path.open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Reuse connected SDK authentication, but never a stale inherited SDK service.
            os.environ.pop('WANDB_SERVICE', None)
            final_path = train(config, fork_from=seal['source_checkpoint']['path'], data_fork=True,
                performance=performance, execution_revision=request['execution_revision'])
            stop = checked_json(file_ref(output/'stop-report.json'))
            audit = checked_json(file_ref(output/'actual-rgb-input-audit.json'))
            initial = torch.load(output/'initial.pt', map_location='cpu', weights_only=True, mmap=True)
            final = torch.load(final_path, map_location='cpu', weights_only=True, mmap=True)
            log = [json.loads(row) for row in (output/'train.jsonl').read_text().splitlines()]
            verify_endpoint(stop, log, audit, initial, final, config)
            linked = root/'fixed-final-step232538.pt'
            os.link(final_path, linked)
            status.update(stage='ordinary_fixed250_training_complete', optimizer_updates=250,
                final_checkpoint=file_ref(linked), actual_inputs=file_ref(output/'actual-rgb-input-audit.json'),
                actual_initial=file_ref(output/'ordinary-initial-state.json'),
                final_state_finite=True, fixed_statistics_unchanged=True,
                final_evaluation_review_pending=True, completed_at=time.time())
            publish_json(root/'completed.json', status)
            atomic_json(status, root/'status.json')
    except BaseException as error:
        status.update(stage='incomplete_no_automatic_retry', error=type(error).__name__, message=str(error),
                      failed_at=time.time(), optimizer_updates=None,
                      update_count_note='Inspect actual training log/checkpoint; failure does not imply zero updates.')
        atomic_json(status, root/'status.json')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--gate-dir', type=Path)
    parser.add_argument('--execution-revision')
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'prepare':
        if not args.gate_dir or not args.execution_revision:
            parser.error('Preparation needs completed gate directory and execution revision')
        print(json.dumps(prepare(args.gate_dir.resolve(), args.execution_revision)))
    else:
        if not args.request or not args.lock:
            parser.error('Run needs request and original GPU lock')
        run(args.request.resolve(), args.lock.resolve())
