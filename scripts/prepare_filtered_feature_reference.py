"""Publish the approved filtered pair's pure-feature arm for gated execution."""
import argparse
from copy import deepcopy
from pathlib import Path

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.projection_fork import COMPARISON_SHA, projection_plan
from run_feature_reference import validate_request


def run(root, output, revision):
    pair_ref = dict(path=str(root / 'projection-filtered-comparison-plan-01/plan.json'),
                    sha256=COMPARISON_SHA)
    pair = checked_json(pair_ref)
    old = checked_json(pair['prior_reference_plan'])
    original_config_ref = file_ref(root / 'feature-original-reference-02/training/config.json')
    config = checked_json(original_config_ref)
    config['data'].update(train_manifest=pair['manifest']['path'], val_manifest=pair['validation']['path'])
    config['output_dir'] = str(output / 'training')
    config['train'].update(eval_at_start=False, log_every=1)
    config['tracking'].update(group='projection-filtered-matched-pair',
        name='projection-filtered-feature1-geometry0-250',
        tags=['projection-filtered-pair', 'feature1-geometry0', 'preserved-adam-rng',
              'effective32', 'three-resolutions', 'fixed250', 'coco-held'])
    perf = checked_json(old['candidate_performance'])
    perf.update(authorized=False, original_started_at=None,
        data_policy='Filtered1597911 RGB permutation, cursor0, fixed250; COCO and white210812 held.')
    plan = {k: deepcopy(old[k]) for k in ('old_execution_root', 'old_execution_data')}
    plan.update(stage='filtered_feature_reference_prepared_requires_actual_gates',
        projection_comparison=pair_ref, reference_images=pair['common']['population'],
        manifest=pair['manifest'], planned_inputs=pair['planned_inputs'],
        source_checkpoint=pair['source_checkpoint'], original_config=original_config_ref,
        geometry_branch=False, geometry_sidecar_required=False,
        geometry_arm_requires_separate_data_and_actual_model_gates=True,
        user_authorized_filtered_paired_training=True, optimizer_updates=0)
    projection_plan(plan)
    output.mkdir(exist_ok=False)
    publish_json(output / 'config.candidate.json', config)
    publish_json(output / 'performance.candidate.json', perf)
    plan.update(candidate_config=file_ref(output / 'config.candidate.json'),
                candidate_performance=file_ref(output / 'performance.candidate.json'))
    publish_json(output / 'plan.json', plan)
    publish_json(output / 'training-config.json', config)
    request = dict(plan=file_ref(output / 'plan.json'), config=file_ref(output / 'training-config.json'),
        execution_root=str(Path(__file__).resolve().parents[1]), execution_revision=revision,
        user_requested_root_cause_investigation=True,
        user_authorized_start=True, automatic_second_arm=False)
    publish_json(output / 'request.json', request)
    validate_request(output / 'request.json')
    return file_ref(output / 'request.json')


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--revision', required=True)
    a = p.parse_args()
    print(json.dumps(run(a.root, a.output, a.revision)), flush=True)
