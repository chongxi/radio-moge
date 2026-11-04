"""Preserve a budget-limited filtered B endpoint without accepting fixed250.

Read existing checkpoints, committed inputs and saved evaluations only. This
never changes the fixed-endpoint reviewer or runs a model/optimizer.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.ordinary_training import equal_state
from review_ordinary_geometry_training import compare_input_rows
from review_projection_pair_arm import VIEWS, checked_json as mirrored_json, compare_view
from review_projection_filtered_metrics import identities, normalized_manifest


def partial_scope(stop, log, audit):
    updates = stop['step'] - 232288
    if (not 0 < updates < 250 or stop['reason'] != 'wall_clock_budget'
            or stop['final_evaluation'] != 'complete' or stop['max_seconds'] != 2700
            or stop['optimizer_stopped_elapsed_seconds'] < stop['max_seconds']):
        raise ValueError('Only an evaluated budget-limited partial endpoint is in scope')
    if ([r['step'] for r in log] != list(range(232289, stop['step']+1))
            or any(not math.isfinite(v) for r in log for v in r.values()
                   if isinstance(v, (int, float)))
            or any(r['effective_batch'] != 32 or r['logical_data_cursor'] != (i+1)*8
                   for i, r in enumerate(log))):
        raise ValueError('Partial update log is missing, changed or nonfinite')
    if (audit['optimizer_updates'] != updates or audit['cursor'] != updates*8
            or audit['presentations'] != updates*32
            or audit.get('source_counts', {}).get('coco2017', 0) != 0
            or any(audit[k] is not True for k in ('planned_keys_matched',
                'unused_prefetch_excluded', 'actual_compute_groups_verified'))):
        raise ValueError('Partial committed input scope differs')
    return updates


def run(root, output):
    if (root/'completed.json').exists():
        raise ValueError('Completed endpoints belong to the fixed250 reviewer')
    training, corpus = root/'training', root.parent
    request_ref, stop_ref, audit_ref = [file_ref(p) for p in
        (root/'run-request.json', training/'stop-report.json', training/'actual-rgb-input-audit.json')]
    request, stop, audit = map(checked_json, (request_ref, stop_ref, audit_ref))
    config = checked_json(request['config'])
    if (Path(config['output_dir']) != training or config['ordinary_geometry']['multiplier'] != 1
            or set(config['loss'].values()) != {1.0}):
        raise ValueError('Unexpected partial run configuration')
    log = [json.loads(s) for s in (training/'train.jsonl').read_text().splitlines()]
    updates = partial_scope(stop, log, audit)
    initial_proof_ref = file_ref(training/'ordinary-initial-state.json')
    initial_proof = checked_json(initial_proof_ref)
    if not initial_proof['adapter_rms_adam_rng_exact'] or initial_proof['source_step'] != 232288:
        raise ValueError('Existing actual initial proof is incomplete')
    initial = torch.load(training/'initial.pt', map_location='cpu', weights_only=True, mmap=True)
    final_path = training/'latest.pt'
    before = final_path.stat()
    final = torch.load(final_path, map_location='cpu', weights_only=True, mmap=True)
    if (final['step'] != stop['step'] or final['cursor'] != updates*8 or final['config'] != config
            or final['data_digests'] != initial['data_digests']
            or any(not equal_state(final[k], initial[k]) for k in
                ('statistics', 'adapter_config', 'model_config', 'revisions', 'performance',
                 'execution_revision', 'fork_provenance', 'runtime_limits'))
            or final['execution_revision'] != request['execution_revision']
            or final['runtime_limits'] != dict(steps=232538, max_seconds=2700)
            or final['scheduler']['last_epoch'] != updates
            or any(g['lr'] != config['train']['lr'] for g in final['optimizer']['param_groups'])
            or len(final['optimizer']['state']) != 100
            or {int(v['step']) for v in final['optimizer']['state'].values()} != {stop['step']}):
        raise ValueError('Partial checkpoint metadata, counters or fixed statistics differ')
    for section in (final['adapter'], *final['optimizer']['state'].values()):
        if any(not bool(torch.isfinite(v).all()) for v in section.values() if torch.is_tensor(v)):
            raise ValueError('Nonfinite partial adapter or Adam state')
    provenance_ref = file_ref(training/'provenance.json')
    provenance = checked_json(provenance_ref)
    if (provenance['data_digests'] != final['data_digests']
            or provenance['revisions'] != final['revisions']):
        raise ValueError('Checkpoint and recorded provenance differ')
    a_root = corpus/'projection-filtered-feature1-geometry0-02'
    a_provenance = checked_json(file_ref(a_root/'training/provenance.json'))
    if any(provenance[k] != a_provenance[k] for k in ('data_digests', 'revisions', 'adapter')):
        raise ValueError('A/B model or data identity differs')
    current = [json.loads(s) for s in checked_path(audit['records']).read_text().splitlines()]
    reference = [json.loads(s) for s in (a_root/'training/actual-rgb-inputs.jsonl').read_text().splitlines()]
    compare_input_rows(reference[:updates], current, updates)
    saved_ref = file_ref(corpus/'projection-filtered-metrics-01/report.json')
    saved = mirrored_json(saved_ref)
    selection = mirrored_json(saved['index'])
    kept = identities(normalized_manifest(selection['new_selection']))
    validation_ref = file_ref(training/'validation.jsonl')
    validation = [json.loads(s) for s in checked_path(validation_ref).read_text().splitlines()]
    if len(validation) != 1 or validation[0]['step'] != stop['step']:
        raise ValueError('One saved evaluation of the actual partial endpoint is required')
    views = {}
    for view in VIEWS:
        directory = training/f"eval-{stop['step']:07d}"
        if view != 'original':
            directory /= view
        refs = {k: file_ref(directory/(k+'.json')) for k in ('metrics', 'per_image', 'protocol')}
        actual = {k: checked_json(v) for k, v in refs.items()}
        old = saved['evidence'][view]['archive']
        compared = compare_view(mirrored_json(old['per_image']), actual['per_image'],
            mirrored_json(old['protocol']), actual['protocol'], actual['metrics'], view, kept)
        prefix = '' if view == 'original' else ('resolution/native-640x400/' if view == 'native-640x400'
            else 'dataset/'+view.removeprefix('dataset-')+'/')
        if any(validation[0].get(prefix+k) != v for k, v in actual['metrics'].items()):
            raise ValueError('Partial saved metrics and validation log differ')
        views[view] = dict(**compared, files=refs)
    output.mkdir(exist_ok=False)
    linked = output/f"partial-step{stop['step']}.pt"
    os.link(final_path, linked)
    checkpoint_ref = file_ref(linked)
    after = final_path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Partial checkpoint changed while reviewing')
    report = dict(stage='filtered_partial_endpoint_preserved_not_fixed250', completed_at=time.time(),
        request=request_ref, stop=stop_ref, actual_updates=updates, step=stop['step'],
        checkpoint=checkpoint_ref, adapter_adam_finite=True, adam_states=100,
        scheduler_epoch=updates, original_runtime_limits_retained=True,
        fixed_statistics_unchanged=True, existing_initial_proof=initial_proof_ref,
        initial_source_state_reverification=False, actual_inputs=audit_ref,
        actual_rgb_presentations=updates*32, A_committed_rgb_prefix_exact=True,
        provenance=provenance_ref, validation=validation_ref, saved_archive_scope=saved_ref,
        views=views, fixed250_completed=False, valid_fixed250_pair_comparison=False,
        historical_cause_established=False, inference_performed=False, optimizer_updates_by_reviewer=0,
        interpretation='Descriptive partial-endpoint metrics only; no equal-budget A/B or predeclared joint success claim.',
        reviewer=file_ref(Path(__file__)))
    publish_json(output/'report.json', report)
    print(json.dumps(dict(report=file_ref(output/'report.json'), updates=updates,
        checkpoint=report['checkpoint'], fixed250_completed=False)), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.arm.resolve(), args.output.resolve())
