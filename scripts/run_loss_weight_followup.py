"""One predeclared, bounded weight-3 follow-up after the complete 2x2 screen."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.data import read_manifest
from radio_moge.train import validate_config

if __package__:
    from .report_loss_weight_ablation import collect, compare, read, tensor_digest
    from .run_loss_weight_ablation import FEATURE_KEYS, digest, run
else:
    from report_loss_weight_ablation import collect, compare, read, tensor_digest
    from run_loss_weight_ablation import FEATURE_KEYS, digest, run


def select_candidate(report):
    """Screen on validation only; do not pretend this selection is a blind test."""
    candidates = []
    for family in ('custom', 'official'):
        low, high = report['arms'][family + '-f01'], report['arms'][family + '-f1']
        if (high['spatial_cos'] - low['spatial_cos'] >= .0001
                and high['summary_cos'] - low['summary_cos'] >= .0001
                and low['nyu_absrel'] - high['nyu_absrel'] >= .002):
            candidates.append(family + '-f1')
    return min(candidates, key=lambda name: report['arms'][name]['nyu_absrel']) if candidates else None


def followup(root):
    report = read(root / 'report-verified.json')
    parent = read(root / 'plan.json')
    assert report['stage'] == 'complete' and read(root / 'controller.json')['stage'] == 'complete'
    candidate = select_candidate(report)
    destination = root / 'feature-weight3'
    if destination.exists():
        raise FileExistsError('Weight-3 decision/run already exists; do not auto-restart')
    decision = dict(selected=candidate,
                    rule='Both cosine gains >=0.0001 and NYU AbsRel improvement >=0.002 when '
                         'raising feature weight 0.1 to1; choose eligible weight1 arm with lowest NYU AbsRel.',
                    maximum_additional_updates=250, maximum_additional_runs=1,
                    limits='Adaptive validation screening, single seed. Not proof of globally optimal weight.')
    atomic_json(decision, destination / 'decision.json')
    if candidate is None:
        print(json.dumps(decision))
        return
    arm_name = candidate.replace('-f1', '-f3')
    config = read(parent['arms'][candidate]['path'])
    config['output_dir'] = str(destination / arm_name / 'train')
    config['train']['eval_at_start'] = False
    config['loss'].update({key: 3. for key in FEATURE_KEYS})
    config['tracking']['name'] = root.name + '-' + arm_name
    config['tracking']['tags'] += ['conditional-weight3']
    validate_config(config)
    config_path = destination / arm_name / 'config.json'
    atomic_json(config, config_path)
    plan = deepcopy(parent)
    plan.update(arms={arm_name: dict(path=str(config_path), sha256=digest(config_path),
                                    objective=parent['arms'][candidate]['objective'], feature_weight=3.)},
                selected_parent_arm=candidate, changed_factors=['four feature weights 1 to3'],
                parent_experiment=str(root), conditional_next=None)
    atomic_json(plan, destination / 'plan.json')
    run(destination)
    step = parent['steps_per_arm']
    low = root / candidate / 'train' / f'eval-{step:07d}'
    high = destination / arm_name / 'train' / f'eval-{step:07d}'
    records = read_manifest(config['data']['val_manifest'])
    initial = torch.load(destination / arm_name / 'train/initial.pt', map_location='cpu',
                         weights_only=True, mmap=True)
    initial_hash = tensor_digest(initial)
    assert initial_hash == report['source_adapter_statistics_digest']
    assert (read(root / candidate / 'train/provenance.json')['data_digests']
            == read(destination / arm_name / 'train/provenance.json')['data_digests'])
    result = dict(stage='complete', selected=candidate, candidate=arm_name, steps=step,
                  baseline=report['baseline'], weight1=report['arms'][candidate], weight3=collect(high),
                  paired_weight1_vs_weight3=compare(low, high, records),
                  paired_baseline_vs_weight3=compare(root / 'custom-f01/train/eval-0000000', high, records),
                  initial_adapter_statistics_digest=initial_hash, decision=decision)
    atomic_json(result, destination / 'result.json')
    print(json.dumps({k: result[k] for k in ('stage', 'selected', 'candidate', 'weight1', 'weight3')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    torch.set_num_threads(1)
    followup(parser.parse_args().root)
