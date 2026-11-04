"""Bounded actual CUDA observation; never launches an optimizer or admits training."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from torch.nn.attention import sdpa_kernel, SDPBackend

from radio_moge.checkpoint import atomic_json, rng_state, restore_rng
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.corpus_model_probe import assert_frozen
from radio_moge.data import DeterministicBatches, TrainingImages
from radio_moge.geometry_cache import publish_json
from radio_moge.losses import FeatureStatistics
from radio_moge.ordinary_geometry_data import GeometryAttachedImages, GeometrySidecar, collate_geometry_attached
from radio_moge.ordinary_model_probe import (compare_feature_window, compare_feature_forward,
                                            geometry_window, module_states, state_digest)
from radio_moge.ordinary_training import (config_digest, digest, equal_state, implementation_digest,
    validate_ordinary_config, validate_ordinary_performance, check_ordinary_data_fork)
from radio_moge.prefetch import PreparedWindows, DeviceWindows, FloatLoader, normalize_uint8
from radio_moge.rgb_input_audit import RGBInputAudit
from radio_moge.system import build_system
from radio_moge.train import make_loader
from verify_ordinary_rgb_pixels import selected_rows


class OriginalIndexRecords(dict):
    verify_on_read = True

    def __init__(self, selected, full_count):
        super().__init__(selected)
        self.full_count = full_count

    def __len__(self):
        return self.full_count


def attention_modes():
    return {key: getattr(torch.backends.cuda, key+'_sdp_enabled')()
            for key in ('flash', 'mem_efficient', 'math', 'cudnn')}


def paired_feature(system, adapter, statistics, plain, attached, config, same_path_check=False):
    original_modes = attention_modes()
    runtime = compare_feature_forward(system, adapter, statistics, plain, attached, config)
    # Exact gradient comparison uses the separately demonstrated repeatable
    # diagnostic backend. The actual trainer and G-on memory/backward retain
    # the original automatic backend; no tolerance is loosened.
    with sdpa_kernel(SDPBackend.MATH):
        diagnostic = compare_feature_window(system, adapter, statistics, plain, attached, config,
                                            same_path_check=same_path_check)
    if attention_modes() != original_modes:
        raise ValueError('Feature diagnostic changed runtime attention backend')
    return dict(runtime_forward=runtime, diagnostic_math_backward=diagnostic,
                runtime_attention_backend_restored=True, runtime_attention_modes=original_modes)


def loader_windows(records, config, performance, sidecar, output, keys):
    audit = RGBInputAudit(output, 232288, 4, 8, performance['compute_batch'], keys)
    cfg = deepcopy(config)
    # Match train.py's loader-only performance override; keep checkpoint config
    # workers unchanged for the narrow original-feature data fork.
    cfg['data']['workers'] = performance['workers']
    cfg['data']['compact_transport'] = performance['compact_transport']
    if sidecar is None:
        cfg.pop('ordinary_geometry', None)
    loader = make_loader(records, cfg, 0, 24, reference=audit, sidecar=sidecar)
    cpu = PreparedWindows(loader, 8, performance['compute_batch'], 3, pin=True, observer=audit.observe_window)
    return DeviceWindows(cpu, 'cuda'), audit


def diagnostic_groups(dataset, keys, config, performance, windows=1):
    # Fixed explicit diagnostic keys, NOT an alternative training sampler.
    logical = config['train']['batch_size']
    if len(keys) % (logical*windows):
        raise ValueError('Diagnostic keys must form complete logical batches')
    loader = DataLoader(dataset, batch_sampler=[keys[i:i+logical] for i in range(0, len(keys), logical)],
        num_workers=0, collate_fn=collate_geometry_attached, pin_memory=True,
        generator=torch.Generator().manual_seed(config['train']['seed']))
    cpu = PreparedWindows(FloatLoader(loader), len(keys)//(logical*windows),
                          performance['compute_batch'], windows, pin=True)
    return DeviceWindows(cpu, 'cuda')


def reserve_optimizer_memory(state, adapter):
    adam_bytes = sum(t.numel()*t.element_size() for v in state['optimizer']['state'].values()
                     for t in v.values() if isinstance(t, torch.Tensor))
    parameter_bytes = sum(p.numel()*p.element_size() for p in adapter.parameters() if p.requires_grad)
    # Conservative capacity reservation; this is not an executed Adam step.
    remaining = adam_bytes + 2*parameter_bytes
    buffers = []
    while remaining:
        size = min(256*1024*1024, remaining)
        buffers.append(torch.empty(size, dtype=torch.uint8, device='cuda'))
        remaining -= size
    return buffers, dict(source_adam_tensor_bytes=adam_bytes, step_workspace_reserved_bytes=2*parameter_bytes,
                         reservation_bytes=adam_bytes+2*parameter_bytes, actual_adam_step_executed=False)


def run(request_path, lock_path):
    request_ref = file_ref(request_path)
    request = checked_json(request_ref)
    output = request_path.parent
    status_path = output/'status.json'
    if status_path.exists():
        raise FileExistsError('Probe already attempted; no automatic retry')
    config, performance = checked_json(request['config']), checked_json(request['performance'])
    validate_ordinary_config(config)
    validate_ordinary_performance(config, performance)
    binding = request['binding']
    if (binding['config_sha256'] != config_digest(config)
            or binding['performance_sha256'] != digest(performance)
            or binding['implementation_sha256'] != implementation_digest()
            or request['observer'] != file_ref(Path(__file__))
            or binding['sidecar'] != config['ordinary_geometry']['sidecar']
            or binding['manifest']['path'] != config['data']['train_manifest']
            or config['train']['steps'] != 232538 or config['train']['max_seconds'] != 2700
            or config['train']['batch_size'] != 4 or performance['compute_batch'] != 32
            or performance['gpu_prefetch'] is not True or performance['workers'] != 4):
        raise ValueError('Frozen ordinary model-observer request differs')
    planned = checked_json(binding['planned_keys'])
    expected = [list(k) for batch in DeterministicBatches(1637908, 4, config['data']['buckets'],
                config['train']['seed'], 0, 2000) for k in batch]
    if planned != expected:
        raise ValueError('Full ordinary permutation presentation plan differs')
    numeric = {k: checked_json(v) for k, v in request['repeatability_evidence'].items()}
    if (set(numeric) != {'current', 'sdpa_math'}
            or any(p['stage'] != 'original_feature_repeatability_observed_not_admitted'
                   or p['mode'] != k or p['optimizer_updates'] != 0
                   or p['training_admission'] is not False or len(p['repetitions']) != 2
                   for k,p in numeric.items())
            or any(not all(r['components_exact'].values()) or not r['loss_rows_exact'] or not r['rng_exact']
                   for p in numeric.values() for r in p['repetitions'])
            or any(not r['gradient_difference']['exact'] for r in numeric['sdpa_math']['repetitions'])):
        raise ValueError('Separately recorded runtime/math repeatability diagnosis is incomplete')
    for proof in numeric.values():
        original_request = checked_json(proof['request'])
        if (original_request['binding']['config_sha256'] != binding['config_sha256']
                or original_request['binding']['source_checkpoint'] != binding['source_checkpoint']
                or [r['key'] for r in proof['actual_inputs']['logical_images']] != planned[:32]):
            raise ValueError('Repeatability diagnosis belongs to a different model/config/input stream')
    target_proof = checked_json(request['target_proof'])
    if target_proof['stage'] != 'ordinary_target_camera_cpu_verified_not_admitted':
        raise ValueError('Expected completed bounded target proof')
    sidecar_proof = checked_json(target_proof['input_proof'])
    if (sidecar_proof['job']['manifest'] != binding['manifest']
            or sidecar_proof['job']['historical_acceptance'] != binding['sidecar']
            or sidecar_proof['stage'] != 'ordinary_geometry_sidecar_cpu_verified_not_admitted'):
        raise ValueError('Target camera proof belongs to a different RGB/geometry scope')
    reference = checked_json(sidecar_proof['job']['reference_pixels'])
    expected_targets = {tuple(r['key']): r for r in target_proof['records']}
    diagnostic_keys = [r['key'] for r in target_proof['records'][32:]]
    chosen = [dict(key=k) for k in planned[:96]+diagnostic_keys]
    start = time.time()
    status = dict(stage='verifying_sources', started_at=start, optimizer_updates=0, training_admission=False)
    atomic_json(status, status_path)
    def progress(stage, **detail):
        status.update(stage=stage, updated_at=time.time(), **detail)
        atomic_json(status, status_path)
        print(json.dumps(dict(stage=stage, **detail)), flush=True)
    adapter, sidecar, initial_random = None, None, None
    try:
        with lock_path.open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not torch.cuda.is_available():
                raise ValueError('Actual observer requires CUDA')
            selected, count = selected_rows(binding['manifest'], chosen)
            if count != 1637908:
                raise ValueError('Changed original full RGB scope')
            records = OriginalIndexRecords(selected, count)
            for item in reference['actual_inputs']:
                row = records[item['key'][0]]
                if any(row[k] != item[k] for k in ('image', 'sha256', 'dataset')):
                    raise ValueError('Saved original reference RGB identity changed')
            progress('loading_sidecar')
            sidecar = GeometrySidecar(binding['sidecar'])
            progress('loading_checkpoint')
            state = torch.load(checked_path(binding['source_checkpoint']), map_location='cpu', weights_only=True, mmap=True)
            check_ordinary_data_fork(config, state, binding['data_digests'])
            if state['step'] != 232288 or {int(v['step']) for v in state['optimizer']['state'].values()} != {232288}:
                raise ValueError('Source Adam counters/checkpoint step differ')
            adam_before = state_digest(state['optimizer'])
            progress('loading_actual_model')
            with patch.object(torch.optim.Optimizer, '__init__', side_effect=AssertionError('No optimizer in observer')):
                system, adapter = build_system(config, torch.device('cuda'))
                if adapter.config != state['adapter_config'] or system.model_config != state['model_config']:
                    raise ValueError('Actual model or adapter architecture changed')
                adapter.load_state_dict(state['adapter'])
                statistics = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim']).cuda()
                statistics.load_state_dict(state['statistics'])
                if not equal_state(adapter.state_dict(), state['adapter']) or not equal_state(statistics.state_dict(), state['statistics']):
                    raise ValueError('Actual CUDA adapter/RMS initialization differs from source')
                restore_rng(state['rng'])
                if not equal_state(rng_state(), state['rng']):
                    raise ValueError('Actual initial RNG differs')
                adapter.train()
                assert_frozen(system)
                initial_random = rng_state()
                before = module_states(system, adapter, statistics)
                reservation, memory = reserve_optimizer_memory(state, adapter)
                torch.cuda.reset_peak_memory_stats()
                feature_rows, geometry_rows, scope = [], [], []
                plain_dataset = TrainingImages(records, True, compact_transport=True)
                dataset = GeometryAttachedImages(plain_dataset, sidecar)
                plain_cfg = deepcopy(config)
                plain_cfg.pop('ordinary_geometry')
                # Only one four-worker loader exists at a time. Keep three CPU
                # baseline windows; attached DeviceWindows still stages current+next.
                plain_windows, plain_audit = loader_windows(records, plain_cfg, performance, None, output, planned[:96])
                plain_cpu = []
                try:
                    for window in plain_windows:
                        plain_cpu.append([x.cpu() for x in window])
                    del window
                finally:
                    plain_windows.close()
                    plain_audit.close()
                attached_windows, attached_audit = loader_windows(records, config, performance, sidecar, output, planned[:96])
                try:
                    for i, attached in enumerate(attached_windows):
                        progress('actual_stream_model_probe', window=i, completed_feature_cases=len(feature_rows))
                        plain = [x.cuda() for x in plain_cpu[i]]
                        feature_rows.append(paired_feature(system, adapter, statistics,
                            plain, attached, config, same_path_check=i == 0))
                        geometry_rows.append(geometry_window(system, adapter, statistics, attached, config, 32, i*8,
                                                             expected_targets))
                        scope.append(dict(kind='actual_first_three_updates', keys=planned[i*32:(i+1)*32]))
                        del plain, attached
                finally:
                    attached_windows.close()
                    attached_audit.close()
                del plain_cpu
                for i in range(0, len(diagnostic_keys), 4):
                    keys = diagnostic_keys[i:i+4]
                    progress('source_model_probe', case=i//4, cases=len(diagnostic_keys)//4)
                    windows = diagnostic_groups(dataset, keys, config, performance)
                    try:
                        groups = next(windows)
                        plain = [normalize_uint8(torch.stack([plain_dataset[tuple(k)] for k in keys])).cuda()]
                        feature_rows.append(paired_feature(system, adapter, statistics, plain, groups, config))
                        geometry_rows.append(geometry_window(system, adapter, statistics, groups, config, 4,
                                                             expected_targets=expected_targets))
                        scope.append(dict(kind='source_resolution_diagnostic', keys=keys))
                        del groups, plain
                    finally:
                        windows.close()
                # Stress full32 at every allowed shape, with the most dense
                # already-reviewed target at that shape. Repeats are explicit.
                for shape in config['data']['buckets']:
                    candidates = [r for r in target_proof['records'] if r['key'][2:] == shape and r['geometry_available']]
                    candidate = max(candidates, key=lambda r:r['observations'])
                    keys = [candidate['key']]*32
                    progress('full32_geometry_memory_probe', shape=shape)
                    windows = diagnostic_groups(dataset, keys+keys, config, performance, windows=2)
                    try:
                        groups = next(windows)
                        # Retain the next complete CUDA window throughout the
                        # backward, instead of relying on asynchronous timing.
                        future = next(windows)
                        dense = geometry_window(system, adapter, statistics, groups, config, 32,
                                                expected_targets=expected_targets)
                        if dense['counts']['gt_supervised_images'] != 32:
                            raise ValueError('Dense32 memory case lost actual target support')
                        geometry_rows.append(dense)
                        scope.append(dict(kind='diagnostic_repeated_dense32_not_training_stream', keys=keys,
                                          staged_future_keys=keys))
                        del groups, future
                    finally:
                        windows.close()
                adapter.zero_grad(set_to_none=True)
                labels = {t['label_kind'] for g in geometry_rows for row in g['groups']
                          for t in row['targets'] if t and t['observations'] >= 32}
                if labels != {'measured_sensor', 'synthetic_renderer', 'teacher_pseudo'}:
                    raise ValueError('Actual geometry gradient coverage lacks a required label stratum')
                restore_rng(initial_random)
                after = module_states(system, adapter, statistics)
                if before != after or not equal_state(initial_random, rng_state()):
                    raise ValueError('Ordinary zero-update observation changed model state/RNG')
                if state_digest(state['optimizer']) != adam_before:
                    raise ValueError('Source Adam tensors changed during observation')
                free, total = torch.cuda.mem_get_info()
                peak_reserved = torch.cuda.max_memory_reserved()
                adjusted_free = free-max(0, peak_reserved-torch.cuda.memory_reserved())
                memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=peak_reserved, free_bytes_at_end=free,
                    free_bytes_adjusted_to_peak_reservation=adjusted_free,
                    device_total_bytes=total, minimum_required_free_bytes=8*1024**3,
                    extra_geometry_only_gradient_buffer=True, future_prefetch_exercised=True,
                    dense32_plus_future32_exercised=True,
                    external_gpu_memory_note='End-time external use plus this process peak reservation; other jobs may vary.')
                if adjusted_free < 8*1024**3:
                    raise ValueError('Insufficient CUDA headroom with optimizer/workspace reservation')
                del reservation
            # Constructor/gradient paths above were explicitly optimizer-disabled.
            report = dict(stage='ordinary_actual_model_zero_update_verified', binding=binding, device='cuda',
                observer=request['observer'], request=request_ref, actual_scope=scope,
                feature_cases=feature_rows, geometry_cases=geometry_rows, memory=memory,
                state_before=before, state_after=after, source_adam_digest=adam_before,
                cuda_transport_exact=True, initial_adapter_rms_rng_exact=True, source_adam_unchanged=True,
                runtime_feature_pixels_losses_exact=True, diagnostic_math_feature_gradients_exact=True,
                runtime_attention_backend_restored=True, geometry_gradients_finite_nonzero=True,
                repeatability_evidence=request['repeatability_evidence'],
                default_backend_gradient_bitwise_equality_claimed=False,
                diagnostic_backend_used_for_training=False,
                frozen_state_unchanged=True, rng_restored=True, gradients_discarded=True,
                optimizer_not_constructed=True, optimizer_updates=0, training_admission=False,
                initial_adam_note='Source moments/counters retained unchanged in CPU checkpoint; actual trainer restoration still requires its separate initial checkpoint comparison.',
                sparse_metadata_note='Selected original indices plus full manifest hash; archived full metadata digest reused, not recomputed from this bounded mapping.',
                elapsed_seconds=time.time()-start)
            sidecar.close()
            checked_path(request_ref)
            if binding['implementation_sha256'] != implementation_digest():
                raise ValueError('Probe runtime code changed')
            publish_json(output/'model-report.json', report)
            progress('complete', report=file_ref(output/'model-report.json'))
            return report
    except BaseException as error:
        progress('failed_no_admission', error=type(error).__name__, message=str(error))
        raise
    finally:
        if adapter is not None:
            adapter.zero_grad(set_to_none=True)
        if initial_random is not None:
            restore_rng(initial_random)
        if sidecar is not None:
            sidecar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--lock', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.request.resolve(), args.lock.resolve())
