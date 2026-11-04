"""One bounded original-feature reference, with synchronous pre-update checks.

Uses the ordinary RGB trainer, original prepared pipeline, archived Adam/RNG,
and a fixed endpoint. This is a diagnostic data fork, not a production resume.
No automatic retry or subsequent experiment is launched.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from radio_moge.checkpoint import atomic_json, rng_state, restore_rng
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.corpus_model_probe import module_digest, gradient_summary, assert_frozen
from radio_moge.data import TrainingImages
from radio_moge.data_fork import check_data_fork
from radio_moge.geometry_cache import publish_json
from radio_moge.losses import weighted_loss
from radio_moge.performance import grouped_window
from radio_moge.prefetch import normalize_uint8
from radio_moge.rgb_input_audit import RGBInputAudit, pixels_sha, load_reference_records
from radio_moge.train import train
from verify_corpus_stage_startup import equal_finite


def checkpoint_ref(plan):
    # The plan annotates this reference with payload_reread=False; checked_path
    # deliberately accepts only the two identity fields.
    return {key: plan['source_checkpoint'][key] for key in ('path', 'sha256')}


def verify_initial(request_path):
    request = checked_json(file_ref(request_path))
    plan = checked_json(request['plan'])
    config = checked_json(request['config'])
    initial_path = Path(config['output_dir']) / 'initial.pt'
    old = torch.load(checked_path(checkpoint_ref(plan)), map_location='cpu', weights_only=True, mmap=True)
    new = torch.load(initial_path, map_location='cpu', weights_only=True, mmap=True)
    if (old['step'] != 232288 or new['step'] != old['step'] or new['cursor'] != 0
            or new['config'] != config):
        raise ValueError('Reference actual initial state/cursor/config differs')
    for name in ('adapter', 'statistics', 'optimizer', 'rng', 'adapter_config', 'model_config', 'revisions'):
        if not equal_finite(old[name], new[name]):
            raise ValueError('Actual reference initial state differs: ' + name)
    if (new['data_digests'].keys() != old['data_digests'].keys()
            or new['data_digests']['train'] == old['data_digests']['train']
            or any(new['data_digests'][k] != v for k, v in old['data_digests'].items() if k != 'train')
            or {int(v['step']) for v in new['optimizer']['state'].values()} != {232288}):
        raise ValueError('Reference initial Adam counters or heldout identities differ')
    result = dict(stage='actual_reference_initial_state_verified', source_step=232288,
        cursor=0, optimizer_states=len(new['optimizer']['state']),
        equal_sections=['adapter', 'statistics', 'optimizer', 'rng', 'adapter_config', 'model_config', 'revisions'],
        all_tensors_finite=True, independent_verifier_process=True,
        trainer_waited_before_first_update=True, optimizer_updates=0,
        scheduler_policy='Constant LR wrapper reset intentionally; not claimed bitwise equal',
        initial=file_ref(initial_path), source=plan['source_checkpoint'],
        config=request['config'], unchanged_heldout_digests=True)
    publish_json(request_path.parent / 'initial-verification.json', result)


def observe_reference_model(system, adapter, statistics, records, config, planned, old_data_path):
    """Actual source/resolution backward checks plus first full prepared update.

    No optimizer step. All module parameters/buffers and the model RNG must be
    unchanged; gradients are discarded before any real update is permitted.
    """
    spec = importlib.util.spec_from_file_location('archived_reference_data', old_data_path)
    old_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_data)
    current = TrainingImages(records, True, compact_transport=True)
    original = old_data.TrainingImages(records, True, compact_transport=True)
    first_by_source = {}
    for index, row in enumerate(records):
        values = first_by_source.setdefault(row['dataset'], [])
        if len(values) < 4:
            values.append(index)
    if len(first_by_source) != 8 or any(len(x) != 4 for x in first_by_source.values()):
        raise ValueError('Reference probe needs all eight sources')
    keys = [[row['index'], row['augmentation_seed'], row['height'], row['width']]
            for row in planned[:32]]
    extra = [([index, 43+j, h, w], source) for source, values in sorted(first_by_source.items())
             for h, w in config['data']['buckets'] for j, index in enumerate(values)]
    key_groups = [(keys[i:i+4], 'first_actual_update') for i in range(0, 32, 4)]
    key_groups += [([key for key, _ in extra[i:i+4]], extra[i][1]) for i in range(0, len(extra), 4)]
    modules = dict(radio=system.radio, encoder=system.encoder, decoder=system.decoder,
                   adapter=adapter, statistics=statistics)
    assert_frozen(system)
    state_before = {name: module_digest(module) for name, module in modules.items()}
    random_before = rng_state()
    device = next(adapter.parameters()).device
    amp = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)
           ) if config['train']['precision'] == 'bf16' else nullcontext
    rows, batches, pixel_rows = [], [], []
    try:
        for group, source in key_groups:
            images = []
            for key in group:
                a, b = current[key], original[key]
                if not torch.equal(a, b):
                    raise ValueError('Actual augmented RGB differs from archived execution')
                row = records[key[0]]
                if file_ref(Path(row['image']))['sha256'] != row['sha256']:
                    raise ValueError('Probe original RGB bytes changed')
                images.append(a)
                pixel_rows.append(dict(key=key, dataset=row['dataset'], image=row['image'],
                                       sha256=row['sha256'], compact_rgb_sha256=pixels_sha(a)))
            batches.append((torch.stack(images), source))
        # The first 32 images use the exact original grouping and batch weights.
        windows = [(x, 'first_actual_update', 32) for x in grouped_window(
            iter([x for x, _ in batches[:8]]), 8, 32)]
        windows += [(x, source, 4) for x, source in batches[8:]]
        adapter.train()
        adapter.zero_grad(set_to_none=True)
        for number, (compact, source, denominator) in enumerate(windows):
            if source != 'first_actual_update':
                adapter.zero_grad(set_to_none=True)
            image = normalize_uint8(compact).to(device)
            with amp(), torch.no_grad():
                features, target = system.encode(image, adapter.indices)
            with amp():
                prediction = adapter(features)
            losses = statistics.loss(prediction, target)
            loss = weighted_loss(losses, config['loss'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite original-feature probe loss')
            (loss * (len(image) / denominator)).backward()
            gradients = gradient_summary(adapter.parameters())
            assert_frozen(system)
            rows.append(dict(case=number, source=source, shape=list(image.shape),
                losses={k: float(v.detach()) for k, v in losses.items()},
                total=float(loss.detach()), gradient=gradients))
            del image, features, target, prediction, losses, loss
        state_after = {name: module_digest(module) for name, module in modules.items()}
        if state_after != state_before:
            raise ValueError('Reference zero-update probe changed model state')
    finally:
        adapter.zero_grad(set_to_none=True)
        restore_rng(random_before)
    if not equal_finite(random_before, rng_state()):
        raise ValueError('Reference model RNG was not restored')
    return dict(stage='actual_reference_zero_update_passed', cases=len(rows), images=len(pixel_rows),
        rows=rows, actual_inputs=pixel_rows, original_execution_pixel_parity=True,
        state_before=state_before, state_after=state_after, rng_restored_bitwise=True,
        optimizer_updates=0, gradients_discarded=True, geometry_branch_executed=False)


class ReferenceAudit(RGBInputAudit):
    def __init__(self, request_path):
        self.request_path = Path(request_path)
        self.request = checked_json(file_ref(request_path))
        self.plan = checked_json(self.request['plan'])
        self.config = checked_json(self.request['config'])
        self.planned = checked_json(self.plan['planned_inputs'])
        keys = [[r['index'], r['augmentation_seed'], r['height'], r['width']] for r in self.planned]
        super().__init__(self.config['output_dir'], 232288, 4, 8, 32, keys)

    def load_records(self, path):
        if path != self.plan['manifest']['path']:
            raise ValueError('Reference metadata loader received another manifest')
        started = time.time()
        def progress(count):
            atomic_json(dict(stage='reading_original_ordered_rgb_metadata', rows=count,
                expected=self.plan['reference_images'], workers=4, started_at=started,
                updated_at=time.time(), optimizer_updates=0),
                self.request_path.parent / 'data-read-status.json')
        records = load_reference_records(path, progress)
        checked_path(self.plan['manifest'])
        if len(records) != self.plan['reference_images']:
            raise ValueError('Reference metadata scope changed')
        publish_json(self.request_path.parent / 'data-read-report.json', dict(
            stage='reference_rgb_metadata_complete', rows=len(records), digest=records.digest(),
            manifest=self.plan['manifest'], unchanged_original_record_order=True,
            actual_consumed_rgb_hash_verification=True, workers=4, seconds=time.time()-started))
        return records

    def check_fork(self, config, state, digests):
        if config != self.config:
            raise ValueError('Reference frozen runtime config changed')
        normalized = deepcopy(config)
        # Only this reference permits skipping duplicate baseline inference and
        # collecting each update. General data-fork policy stays unchanged.
        for key in ('eval_at_start', 'log_every'):
            normalized['train'][key] = state['config']['train'][key]
        check_data_fork(normalized, state, digests)

    def before_updates(self, system, adapter, statistics, records, config, output):
        if len(records) != self.plan['reference_images'] or config != self.config:
            raise ValueError('Actual reference data/config differs from plan')
        for item in self.planned:
            if records[item['index']] != item['image']:
                raise ValueError('Planned RGB identity differs from actual loader records')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'verify-initial',
                        '--request', str(self.request_path)], env=env, check=True, timeout=300)
        proof = observe_reference_model(system, adapter, statistics, records, config,
            self.planned, checked_path(self.plan['old_execution_data']))
        publish_json(self.request_path.parent / 'model-zero-update.json', proof)
        publish_json(self.request_path.parent / 'update-permission.json', dict(
            stage='reference_pre_update_gates_passed', config=self.request['config'],
            initial=file_ref(self.request_path.parent / 'initial-verification.json'),
            actual_model=file_ref(self.request_path.parent / 'model-zero-update.json'),
            maximum_new_updates=250, maximum_seconds=2700, automatic_next_experiment=False))


def validate_request(request_path):
    request = checked_json(file_ref(request_path))
    plan = checked_json(request['plan'])
    config = checked_json(request['config'])
    candidate = checked_json(plan['candidate_config'])
    expected = deepcopy(candidate)
    expected['output_dir'] = str(request_path.parent / 'training')
    expected['train'].update(eval_at_start=False, log_every=1)
    expected['tracking'].update(name='feature-original-pipeline-reference-250',
        tags=['diagnostic-reference', 'original-feature-pipeline', 'preserved-adam-rng',
              'effective32', 'three-resolutions', 'fixed250', 'no-geometry'])
    if (config != expected or request['execution_root'] != str(Path(__file__).resolve().parents[1])
            or request['user_requested_root_cause_investigation'] is not True
            or config['loss'] != dict(spatial=1., summary=1., cosine=1., summary_cosine=1.)
            or config['train']['steps'] != 232538 or config['train']['max_seconds'] != 2700
            or 'mixture_weights' in config['data'] or 'geometry_supervision' in config
            or 'feature_replay' in config or 'coverage_epochs' in config['data']):
        raise ValueError('Only the frozen bounded original-feature reference is authorized')
    checked_path(plan['manifest'])
    checked_path(plan['old_execution_data'])
    # Stateful mathematical components must remain byte-identical to old execution.
    old = Path(plan['old_execution_root']) / 'src/radio_moge'
    current = Path(__file__).resolve().parents[1] / 'src/radio_moge'
    for name in ('losses.py', 'system.py', 'adapters.py', 'architecture_screen.py',
                 'attention_tricks.py', 'feature_refinement.py', 'performance.py'):
        if file_ref(old / name)['sha256'] != file_ref(current / name)['sha256']:
            raise ValueError('Original feature implementation changed: ' + name)
    return request, plan, config


def run(request_path, lock_path):
    request, plan, config = validate_request(request_path)
    root = request_path.parent
    status_path = root / 'status.json'
    if status_path.exists():
        raise FileExistsError('Reference already attempted; no automatic retry')
    status = dict(stage='starting', optimizer_updates=0, request=file_ref(request_path), started_at=time.time())
    atomic_json(status, status_path)
    audit = None
    try:
        with lock_path.open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            perf = checked_json(plan['candidate_performance'])
            perf.update(authorized=True, original_started_at=time.time())
            publish_json(root / 'performance.json', perf)
            audit = ReferenceAudit(request_path)
            status.update(stage='loading_reference_and_running_pre_update_gates')
            atomic_json(status, status_path)
            final_path = train(config, fork_from=checked_path(checkpoint_ref(plan)),
                data_fork=True, performance=perf, execution_revision=request['execution_revision'],
                reference=audit)
            stop = checked_json(file_ref(Path(config['output_dir']) / 'stop-report.json'))
            if stop['step'] != 232538 or stop['reason'] != 'step_limit' or stop['final_evaluation'] != 'complete':
                raise ValueError('Reference did not complete its fixed250 endpoint')
            initial = torch.load(Path(config['output_dir']) / 'initial.pt', map_location='cpu', weights_only=True, mmap=True)
            final = torch.load(final_path, map_location='cpu', weights_only=True, mmap=True)
            if (final['cursor'] != 2000 or final['config'] != config
                    or final['data_digests'] != initial['data_digests']
                    or not equal_finite(final['statistics'], initial['statistics'])
                    or {int(v['step']) for v in final['optimizer']['state'].values()} != {232538}):
                raise ValueError('Final reference state differs from bounded contract')
            for section in (final['adapter'], *final['optimizer']['state'].values()):
                if any(not bool(torch.isfinite(v).all()) for v in section.values() if torch.is_tensor(v)):
                    raise ValueError('Nonfinite final reference state')
            log = [json.loads(x) for x in (Path(config['output_dir']) / 'train.jsonl').read_text().splitlines()]
            if ([r['step'] for r in log] != list(range(232289, 232539))
                    or any(not math.isfinite(v) for r in log for v in r.values() if isinstance(v, (int, float)))):
                raise ValueError('Reference actual update log is incomplete/nonfinite')
            linked = root / 'fixed-final-step232538.pt'
            os.link(final_path, linked)
            status.update(stage='fixed250_reference_training_complete', optimizer_updates=250,
                final_checkpoint=file_ref(linked), actual_inputs=file_ref(Path(config['output_dir']) / 'actual-rgb-input-audit.json'),
                final_state_finite=True, fixed_statistics_unchanged=True, final_evaluation_review_pending=True,
                completed_at=time.time())
            publish_json(root / 'completed.json', status)
            atomic_json(status, status_path)
    except BaseException as error:
        status.update(stage='failed_no_automatic_retry', error=type(error).__name__, message=str(error),
                      failed_at=time.time(), optimizer_updates=None,
                      confirmed_successful_updates=audit.committed if audit else 0,
                      update_count_note='Failure may precede or follow an uncommitted update; do not infer zero')
        atomic_json(status, status_path)
        raise
    finally:
        if audit:
            audit.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'verify-initial'))
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'verify-initial':
        verify_initial(args.request.resolve())
    else:
        if args.lock is None:
            parser.error('Original GPU lock required')
        run(args.request.resolve(), args.lock.resolve())
