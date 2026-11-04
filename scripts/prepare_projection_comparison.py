"""Bind a common filtered RGB plan for independent F1/G0 and F1/G1 forks.

Metadata only: no image/model/checkpoint reads, admission or automatic launch.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(value):
    raw = Path(value['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != value['sha256']:
        raise ValueError('Bound metadata changed: ' + value['path'])
    return raw


def publish(path, value):
    incoming = path.with_suffix(path.suffix + '.incoming')
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.rename(incoming, path)


def run(root, output):
    hold_ref = dict(path=str(root / 'projection-filtered-candidates-01/report.json'),
        sha256='2e4602f59ad8ab8e63fc06094a7d36a3f99095e9eb27deacc72041ce386a8e52')
    evaluation_ref = dict(path=str(root / 'projection-filtered-metrics-01/report.json'),
        sha256='f4f6c6d98c0e04296b720238b7aec5f3017121da9eea9786e07424b0480abb1c')
    heldout_ref = dict(path=str(root / 'projection-heldout-audit-02/index.json'),
        sha256='ff7afc40163b302975f1a750e694d5efdb51f55389f421a9b6fef8e780f8329e')
    old_ref = dict(path=str(root / 'feature-pipeline-reference-plan-01/plan.json'),
        sha256='006e536863d9ba5a5d5e6724fc3588e911b9c1b998f5e291707152a20e774338')
    hold, evaluation, heldout, old = [json.loads(checked(v)) for v in
                                     (hold_ref, evaluation_ref, heldout_ref, old_ref)]
    if (hold['stage'] != 'projection_hold_materialized_not_training_accepted'
            or hold['rgb']['retained_rows'] != 1597911
            or hold['geometry']['selected_images'] != 558314
            or hold['rgb']['held_counts'] != {'coco2017': 39997}
            or heldout['selection_remaining_images'] != 1012
            or evaluation['posthoc_diagnostic'] is not True):
        raise ValueError('Completed filtered-scope materialization and saved metric review required')
    original = json.loads(checked(old['archive_config']))
    if (original['loss'] != dict(cosine=1., spatial=1., summary=1., summary_cosine=1.)
            or original['data'].get('mixture_weights')):
        raise ValueError('Original F1 permutation settings required')
    data_source = checked(old['old_execution_data']).decode()
    node = next(n for n in ast.parse(data_source).body
                if isinstance(n, ast.ClassDef) and n.name == 'DeterministicBatches')
    namespace = dict(Sampler=object, random=random, math=math)
    exec(compile(ast.Module(body=[node], type_ignores=[]), old['old_execution_data']['path'], 'exec'), namespace)
    population = hold['rgb']['retained_rows']
    sampler = namespace['DeterministicBatches'](population, 4, original['data']['buckets'], 43, 0, 2000)
    keys = [dict(cursor=i, index=index, augmentation_seed=seed, height=h, width=w)
            for i, batch in enumerate(sampler) for index, seed, h, w in batch]
    needed = {v['index'] for v in keys}
    if len(keys) != 8000 or len(needed) != 8000:
        raise ValueError('The fixed250/effective32 plan must contain 8000 distinct images')
    found, counts, digest, previous_ordinal, count = {}, Counter(), hashlib.sha256(), -1, 0
    with Path(hold['rgb']['output']['path']).open('rb') as source:
        for raw in source:
            digest.update(raw)
            row = json.loads(raw)
            ordinal = row['corpus_ordinal']
            if (row['dataset'] == 'coco2017' or row['split'] != 'train'
                    or ordinal == 210812 or ordinal <= previous_ordinal):
                raise ValueError('Held source, white image, split or original ordering differs')
            previous_ordinal = ordinal
            if count in needed:
                found[count] = row
            counts[row['dataset']] += 1
            count += 1
    if (count != population or dict(counts) != hold['rgb']['retained_counts']
            or digest.hexdigest() != hold['rgb']['output']['sha256'] or set(found) != needed):
        raise ValueError('Filtered manifest identity/population or planned identity coverage differs')
    planned = [dict(k, image=found[k['index']]) for k in keys]
    output.mkdir(exist_ok=False)
    publish(output / 'planned-first8000-inputs.json', planned)
    archive = evaluation['views']['original']['archive']['filtered']['macro']
    plan = dict(stage='filtered_matched_comparison_planned_not_admitted',
        hold=hold_ref, heldout=heldout_ref, saved_evaluation=evaluation_ref, prior_reference_plan=old_ref,
        original_sampler=old['old_execution_data'], manifest=hold['rgb']['output'],
        candidate_geometry_index=hold['geometry']['output'], validation=heldout['new_selection'],
        source_checkpoint=old['source_checkpoint'], source_step=232288, endpoint_step=232538,
        arms=[dict(name='feature1_geometry0', geometry_multiplier=0),
              dict(name='feature1_geometry1', geometry_multiplier=1)],
        both_arms_fork_same_parent_independently=True, second_arm_resumes_first=False,
        common=dict(feature_weights=original['loss'], population=population, sampling='DeterministicBatches permutation',
            seed=43, logical_batch=4, effective_batch=32, actual_accumulation=8,
            resolutions=original['data']['buckets'], crop='ordinary aspect crop/flip/PIL resize',
            optimizer='archived Adam moments/counters, constant LR1e-5',
            state='same archived adapter/RMS/model RNG; new manifest cursor0',
            updates=250, presentations=8000, max_wall_seconds_each=2700,
            replay_branch=False, best_checkpoint_selection=False, eval_at_start=False),
        planned_inputs=ref(output / 'planned-first8000-inputs.json'),
        planned_source_counts=dict(Counter(v['image']['dataset'] for v in planned)),
        planned_resolution_counts=dict(Counter(f"{v['height']}x{v['width']}" for v in planned)),
        plan_is_actual_consumption=False, manifest_payload_sha_checked=True,
        rgb_payloads_read=False, geometry_payloads_read=False, model_or_checkpoint_payloads_read=False,
        optimizer_updates=0, training_admission=False, automatic_launch=False,
        runtime_config_publication_pending=True,
        primary_joint_rule=dict(spatial_cosine_gt=archive['spatial/cosine'],
            summary_cosine_gt=archive['summary/cosine'],
            lidar_absrel_lt=evaluation['views']['dataset-scannetpp-lidar']['archive']['unchanged_lidar_abs_rel'],
            endpoint='Both fixed final250; incomplete arms are not comparable',
            interpretation='Directional screen only; any gain needs replication and no causal certainty'),
        comparisons=['Each arm vs the same six-source reaggregated archive baseline.',
                     'G1 vs G0 on identical actual8000 inputs; geometry contribution conditional on this filtered stream.'],
        limitations=['The archive has historical COCO exposure; this is filtered continuation, not clean pretraining.',
            'The regenerated permutation changes image identities versus prior runs; prior-run differences are not a pure panorama-only intervention.',
            'A null or tiny250-update result does not establish a model ceiling.',
            'No new ProcTHOR/ZInD images or new loss are introduced into this comparison.'],
        required_before_optimizer=['New scope/accounting and sidecar gate must handle projection_hold explicitly.',
            'Replace old population literals with verified-scope values and narrowly bind changed validation identities.',
            'Publish exact runtime configs and effective32 performance profile; do not use historical acceptance.',
            'New CPU and actual CUDA zero-update proofs bound to filtered data/code/config, including G0 no-call parity.',
            'Actual independent initial adapter/RMS/Adam/RNG equality and actual8000 RGB/key/group equality between arms.',
            'Run under original GPU lock; preserve existing native producer/viewer; no retry or budget extension.'],
        generator=ref(__file__))
    publish(output / 'plan.json', plan)
    print(json.dumps(dict(plan=ref(output / 'plan.json'), planned_source_counts=plan['planned_source_counts'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
