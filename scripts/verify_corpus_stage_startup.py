"""Read the actual trainer's initial checkpoint and verify exact continuation.

This is a CPU observation of a separate trainer invocation, not another launch
or optimizer update. It preserves a hard link to that invocation's initial file.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json


def equal_finite(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and torch.equal(a, b))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_finite(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_finite(x, y) for x, y in zip(a, b))
    return a == b


def run(request_path, output):
    request_ref = file_ref(request_path)
    request = checked_json(request_ref)
    if request['start_mode'] != 'corpus_data_fork':
        raise ValueError('Explicit reviewed corpus continuation required')
    launch = Path(request['launch_directory'])
    config_ref = file_ref(launch / 'training-config.json')
    config = checked_json(config_ref)
    acceptance_ref = config['data']['geometry_corpus']
    acceptance = checked_json(acceptance_ref)
    if acceptance['stage'] != 'accepted_for_training' or not acceptance['training_admission']:
        raise ValueError('New actual model acceptance required')
    initial = Path(config['output_dir']) / 'initial.pt'
    initial_ref = file_ref(initial)
    old = torch.load(checked_path(request['initial_checkpoint']), map_location='cpu', weights_only=True)
    new = torch.load(initial, map_location='cpu', weights_only=True)
    step = request['source_step']
    if old['step'] != step or new['step'] != step or new['cursor'] != 0 or new['config'] != config:
        raise ValueError('Actual initial step, new stream or frozen runtime config differs')
    for name in ('adapter', 'statistics', 'optimizer', 'rng', 'adapter_config', 'model_config', 'revisions'):
        if not equal_finite(old[name], new[name]):
            raise ValueError(f'Actual initial {name} differs from parent or contains nonfinite values')
    steps = sorted({int(v['step'].item()) for v in new['optimizer']['state'].values()})
    if steps != [step] or len(new['optimizer']['state']) != len(old['optimizer']['state']):
        raise ValueError('Adam update counters were reset')
    if (new['data_digests'].keys() != old['data_digests'].keys()
            or any(new['data_digests'][k] != v for k, v in old['data_digests'].items()
                   if k not in ('train', 'geometry'))
            or any(new['data_digests'][k] == old['data_digests'][k] for k in ('train', 'geometry'))):
        raise ValueError('Geometry selection must change and replay/validation identities must stay fixed')
    if file_ref(initial) != initial_ref:
        raise ValueError('Actual initial checkpoint changed during verification')
    output.mkdir(exist_ok=False)
    linked = output / f'initial-step{step}.pt'
    os.link(initial, linked)
    report = dict(stage='actual_continuation_initial_state_verified', step=step,
        parent_cursor=old['cursor'], new_cursor=0, optimizer_states=len(new['optimizer']['state']),
        optimizer_steps=steps, adapter_bitwise_equal=True, feature_statistics_bitwise_equal=True,
        optimizer_state_bitwise_equal=True, rng_state_bitwise_equal=True, all_tensors_finite=True,
        replay_and_validation_digests_unchanged=True, accepted_geometry_digest_changed=True,
        independent_real_trainer_invocation_observed=True, weights_modified_by_this_verification=False,
        gpu_used=False, initial_checkpoint=dict(path=str(linked), sha256=initial_ref['sha256']),
        parent_checkpoint=request['initial_checkpoint'], parent_finish=request['parent_finish'],
        new_config=config_ref, new_acceptance=acceptance_ref, request=request_ref,
        fork_state=file_ref(Path(config['output_dir']) / 'fork-state.json'),
        verified_at=time.time(), verification_code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(step=step, optimizer_states=report['optimizer_states'],
        new_cursor=0, report=file_ref(output / 'report.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.request.resolve(), args.output.resolve())
