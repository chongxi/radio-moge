"""Freeze four matched controls using existing accepted data and new loss tests.

This publishes configurations and nonmodel readiness, never model acceptance.
No source review, sampling census, inference or optimizer update is repeated.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import time

from radio_moge.corpus_admission import (checked_json, checked_path, file_ref,
    implementation_hashes, validate_pre_model_evidence)
from radio_moge.geometry_cache import publish_json
from radio_moge.official_gt import OFFICIAL_GT_WEIGHTS
from radio_moge.train import validate_config


FEATURE_KEYS = ('spatial', 'summary', 'cosine', 'summary_cosine')


def configurations(plan, reference, output):
    if (plan.get('stage') != 'controlled_experiment_plan_not_launched'
            or plan['common_initialization']['step'] != 232288
            or plan['optimizer_step_origin'] != 0 or plan['statistics_refit'] is not False
            or plan['fresh_optimizer_all_arms'] is not True
            or plan['data_config'] != reference['data']
            or plan['feature_replay'] != reference['feature_replay']
            or plan['source_mixture'] != reference['data']['mixture_weights']
            or plan['base_geometry_loss'] != OFFICIAL_GT_WEIGHTS
            or plan['updates_per_arm'] != plan['train']['steps'] or plan['train']['steps'] != 250
            or plan['train']['batch_size'] != 1 or plan['train']['accumulation'] != 16
            or plan['train']['seed'] != 43 or plan['train']['lr'] != 1e-5
            or plan['train']['max_seconds'] != 2700):
        raise ValueError('Controlled plan changed the frozen data, initialization or matched budget')
    factors, configs = set(), {}
    for name, variant in plan['variants'].items():
        coefficients, multiplier = variant['feature_coefficients'], variant['geometry_multiplier']
        if (Path(name).name != name or set(coefficients) != set(FEATURE_KEYS)
                or len(set(coefficients.values())) != 1
                or next(iter(coefficients.values())) not in (.1, 1.)
                or type(multiplier) is not int or multiplier not in (0, 1)):
            raise ValueError('Exactly the predefined feature/geometry factorial is required')
        factor = (next(iter(coefficients.values())), multiplier)
        if factor in factors:
            raise ValueError('Duplicate controlled factor')
        factors.add(factor)
        config = deepcopy(reference)
        config['data'].pop('geometry_corpus')
        config['train'] = deepcopy(plan['train'])
        config['loss'] = dict(coefficients, **{k: v * multiplier for k, v in OFFICIAL_GT_WEIGHTS.items()})
        config['corpus_loss_control'] = dict(geometry_multiplier=multiplier, record_inputs=True)
        config['output_dir'] = str(output / name / 'training')
        config['tracking'].update(name='corpus-control-' + name, group='corpus-controls-20261103',
            tags=['matched-corpus-control', 'archive232288', 'fresh-adam', 'fixed250', name])
        # Only the fixed final checkpoint enters the experiment comparison.
        # Remove parent-stage best-selection baselines, preserving evaluation protocols.
        for spec in config['evaluation'].get('checkpoint_metrics', {}).values():
            spec.pop('baseline', None)
        validate_config(config, corpus_preflight=True)
        configs[name] = config
    if factors != {(f, g) for f in (.1, 1.) for g in (0, 1)}:
        raise ValueError('All four predefined factor combinations are required')
    return configs


def prepare(plan_path, tests_path, output, execution_root):
    plan_ref, tests_ref = file_ref(plan_path), file_ref(tests_path)
    plan, tests = checked_json(plan_ref), checked_json(tests_ref)
    reference = checked_json(plan['reference_runtime_config'])
    accepted = checked_json(plan['accepted_data_and_camera_semantics'])
    accounting = checked_json(plan['accounting'])
    hashes = implementation_hashes()
    changed = {name for name, digest in hashes.items()
               if accepted['implementation_sha256'].get(name) != digest}
    if (accepted['stage'] != 'accepted_for_training' or accepted['training_admission'] is not True
            or accepted['accounting_report'] != plan['accounting']
            or accepted['index'] != plan['index'] or accepted['selected_images'] != 598298
            or reference['data']['geometry_corpus'] != plan['accepted_data_and_camera_semantics']
            or tests['stage'] != 'targeted_tests_passed' or tests['passed'] is not True
            or tests['implementation_sha256'] != hashes or tests['local_tests'] < 1 or tests['remote_tests'] < 1
            or not changed <= {'corpus_control.py', 'corpus_admission.py', 'corpus_model_probe.py', 'train.py'}
            or set(accepted['implementation_sha256']) - set(hashes)):
        raise ValueError('New passed implementation tests and unchanged data/loss semantics required')
    if execution_root != Path(__file__).resolve().parents[1]:
        raise ValueError('Prepare from the actual immutable execution tree')
    checked_path(plan['common_initialization']['file'])
    configs = configurations(plan, reference, output)
    output.mkdir(exist_ok=False)
    binding = dict(stage='unchanged_control_inputs_bound', plan=plan_ref,
        prior_acceptance=plan['accepted_data_and_camera_semantics'], accounting=plan['accounting'],
        index=plan['index'], geometry_images=598298, replay_images=796477,
        data_config=plan['data_config'], feature_replay=plan['feature_replay'],
        loader_sampler_source_and_target_bytes_unchanged=True,
        changed_dependencies=sorted(changed), implementation_sha256=hashes,
        training_started=False, optimizer_updates=0, created_at=time.time())
    publish_json(output / 'input-binding.json', binding)
    base = dict(stage='complete', passed=True, accounting_report_sha256=plan['accounting']['sha256'],
        index_sha256=accounting['index']['sha256'], manifest_sha256=accounting['manifest_sha256'],
        training_admission=False, optimizer_updates=0)
    publish_json(output / 'loss-tests.json', dict(base, gate='loss_tests',
        implementation_sha256=hashes, actual_model_forward_backward=False,
        evidence=[tests_ref, accepted['checks']['loss_tests'], file_ref(output / 'input-binding.json')],
        review='New binary geometry-control, exact-zero gradient, actual-input audit, fresh optimizer '
               'and paired RNG tests; old source/target losses and loaders are unchanged.'))
    old_coverage = checked_json(accepted['checks']['coverage'])
    publish_json(output / 'coverage.json', dict(base, gate='coverage',
        reviewed_counts=old_coverage['reviewed_counts'],
        evidence=[accepted['checks']['coverage'], file_ref(output / 'input-binding.json')],
        review='Same accepted index, exact original source/scene sampler and augmentation; no new inputs. '
               'Actual controls additionally hash every consumed RGB/native target/replay view.'))
    checks = {k: accepted['checks'][k] for k in ('source_quality', 'heldout_isolation')}
    checks.update(coverage=file_ref(output / 'coverage.json'), loss_tests=file_ref(output / 'loss-tests.json'))
    validate_pre_model_evidence(plan['accounting'], checks)
    arms = {}
    for name, config in configs.items():
        directory = output / name
        directory.mkdir()
        publish_json(directory / 'unsealed-config.json', config)
        readiness = dict(accounting_report=plan['accounting'], checks=checks,
            training_config=file_ref(directory / 'unsealed-config.json'),
            initial_checkpoint=plan['common_initialization']['file'], source_step=232288, start_mode='init')
        publish_json(directory / 'readiness.json', readiness)
        arms[name] = dict(readiness=file_ref(directory / 'readiness.json'),
                         config=readiness['training_config'], directory=str(directory))
    result = dict(stage='four_control_nonmodel_gates_ready', plan=plan_ref, tests=tests_ref,
        input_binding=file_ref(output / 'input-binding.json'), arms=arms,
        execution_root=str(execution_root), implementation_sha256=hashes,
        code=file_ref(Path(__file__)), all_model_gates_before_any_training=True,
        actual_model_probes_complete=False, training_started=False, optimizer_updates=0)
    publish_json(output / 'experiment.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--tests', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.plan.resolve(), args.tests.resolve(), args.output.resolve(), Path(__file__).resolve().parents[1])
