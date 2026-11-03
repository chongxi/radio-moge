"""Prepare a concrete old-feature-pipeline reference, without launching it.

Stream-compare the existing RGB pool with the original manifest, allowing only
the reviewed white-image exclusion. Reproduce planned sampler keys from the old
execution's actual class. No image, checkpoint, model or optimizer is loaded.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json
from prepare_corpus_rgb_replay import RGB_FIELDS


def run(root, output):
    diagnosis_ref = file_ref(root / 'feature-continuation-contract-review-01/report.json')
    diagnosis = checked_json(diagnosis_ref)
    evidence = {Path(e['path']).name: e for e in diagnosis['evidence']}
    archive = root.parent / 'data-expansion-attention-20261022/model-archive-feature-final232288-20261101'
    config_ref = next(e for e in diagnosis['evidence'] if e['path'] == str(archive / 'config.json'))
    original = checked_json(config_ref)
    archive_plan = checked_json(evidence['plan.json'])
    old_code = Path(archive_plan['remote_code'])
    old_performance = checked_json(evidence['performance.json'])
    pool_ref = file_ref(root / 'rgb-replay-candidates-01/report.json')
    pool = checked_json(pool_ref)
    exclusion_ref = file_ref(root / 'tartanair-rejection-review-10/known-geometry-exclusions.jsonl')
    exclusions = [json.loads(line) for line in checked_path(exclusion_ref).read_text().splitlines()]
    excluded_rgb = [r for r in exclusions if r['rgb_replay_excluded']]
    if (len(excluded_rgb) != 1 or excluded_rgb[0]['corpus_ordinal'] != 210812
            or pool['rgb_excluded_ordinals'] != [210812] or pool['candidate_images'] != 1637908
            or original['data']['train_manifest'] != pool['source_manifest']['path']
            or old_performance['effective_batch'] != 32
            or diagnosis['stage'] != 'feature_continuation_contracts_compared_not_causal'):
        raise ValueError('Exact completed diagnosis, source manifest and only reviewed RGB exclusion required')
    white = excluded_rgb[0]
    white_review = checked_json(white['rgb_input_review'])
    if white_review['excluded_corpus_ordinals'] != [210812] or white_review['passed'] is not True:
        raise ValueError('White exclusion differs from its completed review')
    isolation_ref = file_ref(root / 'heldout-rgb-full-comparison-01/report.json')
    isolation = checked_json(isolation_ref)
    if (isolation['training_rows'] != 1637909 or isolation['heldout_rows'] != 190954
            or isolation['exact_pixel_overlap_rows'] or isolation['forbidden_shared_scene_rows']):
        raise ValueError('Existing full-scope canonical isolation evidence required')

    # Execute only the saved sampler definition, not the module or a model.
    data_path = old_code / 'src/radio_moge/data.py'
    node = next(n for n in ast.parse(data_path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'DeterministicBatches')
    namespace = dict(Sampler=object, random=random, math=math)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(data_path), 'exec'), namespace)
    sampler = namespace['DeterministicBatches'](1637908, 4, original['data']['buckets'], 43, 0, 2000)
    keys = [dict(cursor=i, index=index, augmentation_seed=seed, height=h, width=w)
            for i, batch in enumerate(sampler) for index, seed, h, w in batch]
    needed = {v['index'] for v in keys}
    if len(keys) != 8000 or len(needed) != 8000:
        raise ValueError('Expected first 250 effective32 updates without repeated input indices')
    identities, counts, retained = {}, Counter(), set()
    by_ordinal = {x['corpus_ordinal']: x for x in exclusions if not x['rgb_replay_excluded']}
    source_hash, pool_hash = hashlib.sha256(), hashlib.sha256()
    source_path = Path(pool['source_manifest']['path'])
    included = removed = original_count = 0
    with source_path.open('rb') as source, Path(pool['manifest']['path']).open('rb') as candidate:
        for line in source:
            source_hash.update(line)
            if not line.strip():
                continue
            ordinal = original_count
            original_count += 1
            row = json.loads(line)
            image = os.path.abspath(source_path.parent / row['image'])
            if row['split'] != 'train':
                raise ValueError('Original manifest contains a nontraining input')
            if ordinal == 210812:
                if image != white['image'] or row['sha256'] != white['sha256']:
                    raise ValueError('White exclusion differs from original input')
                removed += 1
                continue
            encoded = candidate.readline()
            pool_hash.update(encoded)
            actual = json.loads(encoded)
            expected = {k: row[k] for k in RGB_FIELDS if k in row}
            expected.update(image=image, corpus_ordinal=ordinal)
            if actual != expected:
                raise ValueError('Candidate pool changed original RGB identity, order, orientation or split')
            if ordinal in by_ordinal:
                decision = by_ordinal[ordinal]
                if any(actual[k] != decision[k] for k in ('image', 'sha256', 'dataset')):
                    raise ValueError('Geometry-only disposition changed its retained RGB identity')
                retained.add(ordinal)
            if included in needed:
                identities[included] = actual
            included += 1
            counts[row['dataset']] += 1
        if candidate.read(1):
            raise ValueError('Candidate pool has unexpected additional rows')
    if (original_count != 1637909 or included != 1637908 or removed != 1
            or retained != set(by_ordinal) or dict(counts) != pool['candidate_counts']
            or source_hash.hexdigest() != pool['source_manifest']['sha256']
            or pool_hash.hexdigest() != pool['manifest']['sha256']):
        raise ValueError('Original/pool scope, ordering, byte hashes or retained decisions differ')

    output.mkdir(exist_ok=False)
    planned = [dict(key, image=identities[key['index']]) for key in keys]
    publish_json(output / 'planned-first8000-inputs.json', planned)
    cfg = deepcopy(original)
    cfg['output_dir'] = str(output / 'proposed-reference-training')
    cfg['data']['train_manifest'] = pool['manifest']['path']
    # This key enforces an entire epoch budget in the old trainer. Removing it
    # allows a bounded diagnostic while retaining the same permutation sampler.
    cfg['data'].pop('coverage_epochs', None)
    cfg['train'].update(steps=232538, max_seconds=2700, save_every=250, eval_every=250)
    cfg['tracking'].update(name='feature-pipeline-reference-candidate', group='feature-contract-diagnosis')
    publish_json(output / 'reference-config.candidate.json', cfg)
    perf = deepcopy(old_performance)
    perf.update(authorized=False, original_started_at=None, anchor_step=232288, anchor_cursor=0,
        anchor_schedule_position=232288., base_effective_batch=32, schedule_ratio=1.,
        data_policy='Planned 250 updates from original ordered full RGB pool minus exact white210812; fresh cursor0.',
        lr_policy='Planned preservation of archived232288 Adam moments and RNG at constant1e-5.')
    publish_json(output / 'performance.candidate.json', perf)
    plan = dict(stage='reference_plan_prepared_not_launched', created_at=time.time(),
        diagnosis=diagnosis_ref, archive_config=config_ref,
        archive_verification=next(e for e in diagnosis['evidence'] if e['path'] == str(archive / 'archive-verification.json')),
        source_checkpoint=dict(path=str(archive / 'final.pt'),
            sha256='3d3920bcaf89b294ba8e3d5016797d8121a9dbd4a1d938f050661ad4e23c66ef', payload_reread=False),
        old_execution_root=str(old_code), old_execution_data=file_ref(data_path),
        prior_controls=diagnosis['paired_result_review'], pool_report=pool_ref, manifest=pool['manifest'],
        original_manifest=pool['source_manifest'], exclusions=exclusion_ref, white_review=white['rgb_input_review'],
        reused_isolation=isolation_ref, original_images=original_count, reference_images=included,
        original_order_and_rgb_fields_verified=True, geometry_only_exclusions_retained=len(retained),
        input_manifest_rewritten=False, rgb_payloads_read=False, model_inference_performed=False,
        checkpoint_payloads_read=False, optimizer_updates=0, training_admission=False, automatic_launch=False,
        candidate_config=file_ref(output / 'reference-config.candidate.json'),
        candidate_performance=file_ref(output / 'performance.candidate.json'),
        planned_inputs=file_ref(output / 'planned-first8000-inputs.json'),
        planned_inputs_are_actual_consumption=False,
        planned_source_counts=dict(Counter(v['image']['dataset'] for v in planned)),
        planned_resolution_counts=dict(Counter(f"{v['height']}x{v['width']}" for v in planned)),
        reference=dict(name='R0 original feature pipeline, clean full RGB data fork', new_updates=250,
            new_image_presentations=8000, new_wall_seconds=2700, source_step=232288, endpoint_step=232538,
            loss=dict(original['loss']), geometry_loss=False, replay_branch=False,
            sampling='Original DeterministicBatches, seed43, logical batch4, full-pool permutation, cursor0',
            view='Original largest aspect crop, random location, horizontal flip, PIL bilinear resize',
            resolutions=original['data']['buckets'], effective_batch=32, actual_accumulation=8,
            compute_batch_cap=32, prepared_pipeline=True, workers=4,
            state='Preserve adapter, fixed RMS, Adam moments and counters, and model RNG; constant LR1e-5',
            intentional_differences=['Remove exactly white RGB210812; start fresh cursor0 because manifest changes.',
                'Short fixed250-update budget instead of full-epoch coverage assertion; sampler remains permutation.',
                'Fresh tracking and budget; fixed endpoint rather than best-checkpoint selection.'],
            is_bitwise_uninterrupted_original_resume=False),
        primary_joint_rule=dict(at='fixed final250 new updates', all_required=True,
            spatial_cosine_gt=.9616920552147653, summary_cosine_gt=.932300927138112,
            scannet_lidar_absrel_lt=.09211844746023416,
            minimum_effect_claim='Directional screen only; any positive needs replication, no significance claim.',
            incomplete_budget='Report actual budget as incomplete; no substitution for fixed250 endpoint.'),
        evaluation=dict(reuse_archive_baseline=True, no_new_baseline_inference_planned=True,
            require_same_four_views=True, require_same_five_heldout_digests=True,
            require_same_teacher_revision_and_per_image_support=True, require_same18_laser_cases=True,
            known_camera_evaluator='Use frozen controls evaluator only after contract parity is verified; source trainer may not provide the later laser probe.',
            report_all_metrics=True, nyu_gain_alone_is_success=False, no_best_selection=True),
        future_isolated_comparisons=[
            dict(name='Adam-state-only', comparator='R0', change='Reset only Adam moments and per-parameter Adam counters',
                fixed='Same parent adapter/RMS/model RNG, sampler keys, pixels, batch, resolution, loss, LR and fixed250 endpoint',
                caution='Existing --init resets model RNG and global step too; it is not a valid Adam-only intervention. New initializer and equality proof needed.'),
            dict(name='Resolution-only', comparator='R0', change='Output400x640 only',
                fixed='Same ordered image identities and per-image augmentation seeds, parent state, effective32, loss and endpoint',
                caution='Freeze independent resolution choice and image/augmentation keys; ordinary RNG list-length changes must not alter crop seeds. Aspect crop changes are part of resolution treatment.'),
            dict(name='Sampling mechanisms', change='Separate replacement, source weights, scene weights, and eligible-pool restriction',
                sequence=['Uniform full-pool permutation vs uniform full-pool draws with replacement.',
                    'Hold replacement and within-source image-uniform draws fixed; change only source probabilities.',
                    'Hold source probabilities fixed; compare image-uniform vs scene-uniform then frame draws.',
                    'Compare full RGB eligibility vs restricted geometry/replay pools separately.'],
                caution='Historical half-corpus/half-replay sampler changes all of these; do not label it one source-weight factor.'),
            dict(name='View-only', change='Ordinary aspect crop vs camera-aware warp',
                fixed='Same common eligible images, source order, output resolution, parent/RNG/loss/batch/budget',
                caution='Needs a new ordinary-crop comparator on exactly the same camera-eligible pool; R0 full-pool vs warped subset is confounded.'),
            dict(name='Effective-batch-only', change='32 vs16 over the same ordered image/augmentation stream',
                comparisons=['Fixed250 updates:8000 vs4000 exposures; report exposure difference explicitly.',
                    'Prespecified equal4000 exposures:R0 checkpoint125 vs batch16 checkpoint250; auxiliary only.'],
                caution='Cannot hold both update count and unique image exposure fixed while changing effective batch. Do not select a better clock after results.' )],
        launch_requirements=[
            'This plan schedules no GPU run. Inspect and concretely implement only the missing reference before deciding further comparisons.',
            'Validate immutable execution vs old90a6821 training/loss/sampling/crop/prepared-pipeline definitions; preserve all semantic fields.',
            'Add actual RGB identity/pixel/augmentation-key audit to plain RGB execution, proving logging leaves values/RNG/state unchanged.',
            'Validate data-fork bookkeeping and250-update/2700-second budget; do not leave full coverage_epochs=1 assertion in the diagnostic config.',
            'Actual separate initial-state proof must compare archived adapter/RMS/Adam counters and RNG; planned metadata is insufficient.',
            'Actual zero-update model gate and new config-specific contract are required; prior corpus gates do not admit this altered pipeline.',
            'Check prepared worker/prefetch memory under current remote load, preserve native Scan/viewer/original GPU lock, no autotune.',
            'Freeze evaluation equality and final endpoint before any updates. No automatic retry, additional arm or production continuation.'],
        generator=file_ref(Path(__file__)))
    publish_json(output / 'plan.json', plan)
    print(json.dumps(dict(plan=file_ref(output / 'plan.json'), images=included,
        retained_geometry_only=len(retained), planned_source_counts=plan['planned_source_counts'],
        optimizer_updates=0)), flush=True)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
