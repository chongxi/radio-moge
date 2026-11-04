"""Quantify repeated original-feature CUDA backward; no geometry or optimizer.

This diagnosis does not admit training or relax the failed exact-parity gate.
Current and deterministic numerical modes are separate explicit observations.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch

from radio_moge.checkpoint import atomic_json, rng_state, restore_rng
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.losses import FeatureStatistics
from radio_moge.ordinary_model_probe import feature_window, capture_gradients, module_states, state_digest
from radio_moge.ordinary_training import equal_state, implementation_digest
from radio_moge.prefetch import PreparedWindows
from radio_moge.rgb_input_audit import RGBInputAudit
from radio_moge.system import build_system
from radio_moge.train import make_loader
from probe_ordinary_geometry_model import OriginalIndexRecords
from verify_ordinary_rgb_pixels import selected_rows


def gradient_difference(names, before, after):
    rows, total_squared, reference_squared = [], 0., 0.
    for name, a, b in zip(names, before, after, strict=True):
        if a is None or b is None:
            if a is not b:
                raise ValueError('Changed gradient connectivity: '+name)
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('Changed gradient shape/dtype: '+name)
        flat_a, flat_b = a.reshape(-1), b.reshape(-1)
        changed, maximum, squared, base, maximum_reference = 0, 0., 0., 0., 0.
        for start in range(0, a.numel(), 1048576):
            x, y = flat_a[start:start+1048576].double(), flat_b[start:start+1048576].double()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('Nonfinite repeated gradient')
            delta = x-y
            changed += int(torch.count_nonzero(delta))
            maximum = max(maximum, float(delta.abs().max()))
            maximum_reference = max(maximum_reference, float(x.abs().max()))
            squared += float(delta.square().sum())
            base += float(x.square().sum())
        total_squared += squared
        reference_squared += base
        rows.append(dict(parameter=name, elements=a.numel(), changed=changed, maximum_absolute=maximum,
            maximum_reference=maximum_reference, difference_l2=squared**.5, reference_l2=base**.5))
    return dict(exact=all(r['changed'] == 0 for r in rows), tensors=len(rows),
        changed_tensors=sum(r['changed'] > 0 for r in rows), changed_elements=sum(r['changed'] for r in rows),
        maximum_absolute=max(r['maximum_absolute'] for r in rows), difference_l2=total_squared**.5,
        reference_l2=reference_squared**.5,
        relative_l2=(total_squared/reference_squared)**.5 if reference_squared else None, parameters=rows)


def run(gate_dir, output, lock_path, mode):
    output.mkdir(exist_ok=False)
    request_ref = file_ref(gate_dir/'request.json')
    request = checked_json(request_ref)
    if implementation_digest() != request['binding']['implementation_sha256']:
        raise ValueError('Diagnosis must retain the failed observer runtime implementation')
    config = checked_json(request['config'])
    planned = checked_json(request['binding']['planned_keys'])[:32]
    status = dict(stage='preparing_original_rgb', started_at=time.time(), mode=mode,
                  optimizer_updates=0, training_admission=False)
    atomic_json(status, output/'status.json')
    adapter, initial_random, windows = None, None, None
    def progress(stage, **fields):
        status.update(stage=stage, updated_at=time.time(), **fields)
        atomic_json(status, output/'status.json')
        print(json.dumps(dict(stage=stage, **fields)), flush=True)
    try:
        with lock_path.open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            rows, count = selected_rows(request['binding']['manifest'], [dict(key=k) for k in planned])
            records = OriginalIndexRecords(rows, count)
            cfg = deepcopy(config)
            cfg.pop('ordinary_geometry')
            cfg['data'].update(workers=0, compact_transport=True)
            audit = RGBInputAudit(output, 232288, 4, 8, 32, planned)
            windows = PreparedWindows(make_loader(records, cfg, 0, 8, reference=audit), 8, 32, 1,
                                      observer=audit.observe_window)
            groups = [image.cuda() for image in next(windows)]
            windows.close()
            actual = audit.pending_windows[0]
            audit.close()
            target_proof = checked_json(request['target_proof'])
            sidecar_proof = checked_json(target_proof['input_proof'])
            reference = checked_json(sidecar_proof['job']['reference_pixels'])
            for current, expected in zip(actual['logical_images'], reference['actual_inputs'][:32], strict=True):
                if any(current[k] != expected[k] for k in ('key', 'image', 'sha256', 'dataset', 'compact_rgb_sha256')):
                    raise ValueError('Original diagnostic pixels differ from saved reference')
            progress('loading_original_model')
            state = torch.load(checked_path(request['binding']['source_checkpoint']), map_location='cpu',
                               weights_only=True, mmap=True)
            with patch.object(torch.optim.Optimizer, '__init__', side_effect=AssertionError('No optimizer')):
                system, adapter = build_system(config, torch.device('cuda'))
                adapter.load_state_dict(state['adapter'])
                statistics = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim']).cuda()
                statistics.load_state_dict(state['statistics'])
                if not equal_state(adapter.state_dict(), state['adapter']) or not equal_state(statistics.state_dict(), state['statistics']):
                    raise ValueError('Actual original initialization differs')
                restore_rng(state['rng'])
                initial_random = rng_state()
                adapter.train()
                before = module_states(system, adapter, statistics)
                observations = []
                trace = []
                original_encode = system.encode
                def encode(*args, **kwargs):
                    features, target = original_encode(*args, **kwargs)
                    trace.append(dict(encoded_radio=state_digest(vars(features)),
                                      encoded_teacher=state_digest(vars(target))))
                    return features, target
                def observe_adapter(module, inputs, prediction):
                    row = trace[-1]
                    row['adapter_output'] = state_digest(vars(prediction))
                    for name in ('spatial', 'summary'):
                        getattr(prediction, name).register_hook(
                            lambda gradient, key=name, record=row: record.__setitem__(key+'_output_gradient', state_digest(gradient)))
                hook = adapter.register_forward_hook(observe_adapter)
                with patch.object(system.decoder, 'forward', side_effect=AssertionError('G0 must not decode geometry')), \
                        patch.object(system, 'encode', encode):
                    for repetition in range(3):
                        progress('repeated_original_backward', repetition=repetition)
                        restore_rng(initial_random)
                        trace.clear()
                        value = feature_window(system, adapter, statistics, groups, config, 32)
                        gradients, after_random = capture_gradients(adapter), rng_state()
                        if repetition == 0:
                            first, first_gradients, first_random = value, gradients, after_random
                            first_trace = deepcopy(trace)
                        else:
                            observations.append(dict(repetition=repetition,
                                loss_rows_exact=equal_state(first['rows'], value['rows']),
                                rng_exact=equal_state(first_random, after_random),
                                component_hashes=deepcopy(trace),
                                components_exact={k:all(a[k] == b[k] for a,b in zip(first_trace, trace, strict=True))
                                                  for k in first_trace[0]},
                                loss_rows=value['rows'], gradient_summary=value['gradient'],
                                gradient_difference=gradient_difference([n for n,_ in adapter.named_parameters()],
                                                                        first_gradients, gradients)))
                        del gradients
                hook.remove()
                adapter.zero_grad(set_to_none=True)
                restore_rng(initial_random)
                after = module_states(system, adapter, statistics)
                if before != after or not equal_state(initial_random, rng_state()):
                    raise ValueError('Repeated backward changed frozen/model/RMS state or RNG')
                report = dict(stage='original_feature_repeatability_observed_not_admitted',
                    request=request_ref, script=file_ref(Path(__file__)), mode=mode,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    cuda_flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),
                    actual_inputs=actual, first=first, first_component_hashes=first_trace, repetitions=observations,
                    adapter_rms_initial_exact=True, model_state_before=before, model_state_after=after,
                    source_adam_read_only_digest=state_digest(state['optimizer']),
                    geometry_payloads_read=False, geometry_decoder_called=False, optimizer_constructed=False,
                    optimizer_updates=0, training_admission=False, gradients_discarded=True, rng_restored=True,
                    historical_regression_cause_established=False, seconds=time.time()-status['started_at'])
                publish_json(output/'report.json', report)
                progress('complete_observation_no_admission', report=file_ref(output/'report.json'))
    except BaseException as error:
        progress('diagnostic_failed_no_admission', error=type(error).__name__, message=str(error))
        raise
    finally:
        if windows is not None:
            windows.close()
        if adapter is not None:
            adapter.zero_grad(set_to_none=True)
        if initial_random is not None:
            restore_rng(initial_random)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--lock', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('current', 'deterministic', 'sdpa_math'))
    args = parser.parse_args()
    if args.mode == 'deterministic':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    # A separate isolated attention-backend diagnostic; do not alter global
    # deterministic flags or replace the original median/channel-cleaning path.
    from torch.nn.attention import sdpa_kernel, SDPBackend
    context = sdpa_kernel(SDPBackend.MATH) if args.mode == 'sdpa_math' else nullcontext()
    with context:
        run(args.gate_dir.resolve(), args.output.resolve(), args.lock.resolve(), args.mode)
