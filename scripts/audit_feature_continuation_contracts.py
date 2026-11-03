"""Compare saved feature-training contracts; no images, model, or optimizer run.

This records differences and hypotheses, not training admission or a causal test.
Replay-source counts are reconstructed from the frozen sampler's source draw;
the old input audit contains actual replay pixel hashes but not replay identity.
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
import time


def file_ref(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_json(path):
    return json.loads(path.read_text())


def checked_json(ref):
    path = Path(ref['path'])
    if file_ref(path) != ref:
        raise ValueError('Changed evidence: ' + str(path))
    return read_json(path)


def definition(path, name):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def run(root, output):
    base = root.parent / 'data-expansion-attention-20261022'
    archive = base / 'model-archive-feature-final232288-20261101'
    archive_config = read_json(archive / 'config.json')
    old_run = Path(archive_config['output_dir'])
    old_provenance = read_json(old_run / 'provenance.json')
    old_performance = read_json(old_run / 'performance.json')
    old_plan = read_json(archive / 'plan.json')
    old_code = Path(old_plan['remote_code'])
    experiment = root / 'corpus-controlled-comparison-01'
    closure = read_json(experiment / 'results-closure-01/report.json')
    review = checked_json(closure['paired_review'])
    completed = checked_json(closure['completion'])
    control_code = Path(checked_json(completed['experiment'])['execution_root'])
    control = experiment / 'feature1p0_geometry0'
    config = read_json(control / 'training-config.json')
    provenance = read_json(control / 'training/provenance.json')
    inventory = read_json(root / 'inventory.json')
    initial_ref = checked_json(closure['arms'][control.name]['completion'])['initial_verification']
    checked_json(initial_ref)
    if (closure['stage'] != 'four_controls_closed_and_online_verified'
            or closure['primary_joint_candidates'] or old_provenance['performance'] != old_performance
            or archive_config != read_json(old_run / 'config.json')
            or old_provenance['revisions'] != provenance['revisions']
            or old_provenance['adapter'] != provenance['adapter']
            or config['models'] != archive_config['models']
            or config['adapter'] != archive_config['adapter']
            or config['corpus_loss_control']['geometry_multiplier'] != 0
            or inventory['images'] != old_provenance['train_images']):
        raise ValueError('Saved archive, runtime, or completed controls differ')

    selected_files = ('losses.py', 'system.py', 'adapters.py', 'architecture_screen.py',
                      'attention_tricks.py', 'feature_refinement.py')
    code_identity = {}
    for name in selected_files:
        old, new = old_code / 'src/radio_moge' / name, control_code / 'src/radio_moge' / name
        before, after = file_ref(old), file_ref(new)
        code_identity[name] = dict(archive=before, control=after,
                                   bytes_identical=before['sha256'] == after['sha256'])
    if not code_identity['losses.py']['bytes_identical']:
        raise ValueError('Loss formula changed; this diagnosis needs revision')
    data_files = [p / 'src/radio_moge/data.py' for p in (old_code, control_code)]
    sampler_same = definition(data_files[0], 'DeterministicBatches') == definition(
        data_files[1], 'DeterministicBatches')

    weights = ('spatial', 'summary', 'cosine', 'summary_cosine')
    if any(archive_config['loss'][k] != 1 or config['loss'][k] != 1 for k in weights):
        raise ValueError('Unexpected feature coefficients')
    # Unlike labels in older narrative summaries, random.choices normalizes
    # relative weights. The seven-source replay weights sum to 0.85, not 1.
    paired_weights = config['data']['mixture_weights']
    replay = config['feature_replay']
    replay_weights = replay['mixture_weights']
    paired_sum, replay_sum = sum(paired_weights.values()), sum(replay_weights.values())
    if replay['fraction'] != .5 or len(config['data']['buckets']) != 1:
        raise ValueError('Unexpected replay branch or resolution schedule')
    audit_ref = read_json(control / 'training/control-input-audit.json')['records']
    audit_path = Path(audit_ref['path'])
    if file_ref(audit_path) != audit_ref or audit_ref['sha256'] != closure['input_sha256']:
        raise ValueError('Actual control inputs changed')
    paired_feature, geometry_all, replay_reconstructed = Counter(), Counter(), Counter()
    sources = sorted(replay_weights)
    replay_slots = 0
    for i, line in enumerate(audit_path.open()):
        row = json.loads(line)
        if row['cursor'] != i or row['replay_slot'] != bool(i % 2) or len(row['identities']) != 1:
            raise ValueError('Control input cursor differs')
        dataset = row['identities'][0]['dataset']
        geometry_all[dataset] += 1
        if row['replay_slot']:
            rng = random.Random(replay['seed'] + replay_slots * 104729)
            rng.choice(config['data']['buckets'])
            replay_reconstructed.update(rng.choices(sources,
                weights=[replay_weights[s] for s in sources], k=1))
            replay_slots += 1
        else:
            paired_feature[dataset] += 1
    if i + 1 != 4000 or replay_slots != 2000 or sum(paired_feature.values()) != 2000:
        raise ValueError('Incomplete fixed 250-step feature input accounting')
    source_counts = inventory['dataset_counts']
    if sum(source_counts.values()) != inventory['images']:
        raise ValueError('Frozen source counts differ')
    source_comparison = {s: dict(
        archive_manifest_images=n, archive_manifest_share=n / inventory['images'],
        expected_control_feature_share=.5 * paired_weights.get(s, 0) / paired_sum
                                      + .5 * replay_weights.get(s, 0) / replay_sum,
        actual_paired_feature_slots=paired_feature[s],
        reconstructed_replay_source_slots=replay_reconstructed[s],
        combined_feature_share=(paired_feature[s] + replay_reconstructed[s]) / 4000)
        for s, n in sorted(source_counts.items())}
    if not math.isclose(sum(v['expected_control_feature_share'] for v in source_comparison.values()), 1):
        raise ValueError('Source probabilities are not normalized')

    inputs = [archive / 'config.json', archive / 'archive-verification.json', archive / 'plan.json',
              old_run / 'config.json', old_run / 'provenance.json', old_run / 'performance.json',
              old_run / 'coverage.json', control / 'training-config.json',
              control / 'training/provenance.json', control / 'training/control-input-audit.json',
              root / 'inventory.json', experiment / 'results-closure-01/report.json']
    code_paths = [old_code / 'src/radio_moge/train.py', control_code / 'src/radio_moge/train.py',
                  *data_files, old_code / 'src/radio_moge/performance.py',
                  old_code / 'src/radio_moge/prefetch.py', control_code / 'src/radio_moge/corpus_data.py',
                  control_code / 'src/radio_moge/geometry_sampling.py',
                  control_code / 'src/radio_moge/feature_replay.py']
    metadata = dict(stage='feature_continuation_contracts_compared_not_causal', created_at=time.time(),
        evidence=[file_ref(p) for p in inputs], code_evidence=[file_ref(p) for p in code_paths],
        generator=file_ref(Path(__file__)), paired_result_review=closure['paired_review'],
        initial_state_verification=initial_ref, actual_input_audit=audit_ref,
        source_comparison=source_comparison,
        replay_source_count_method='Reconstruct only RNG source draw from frozen sampler/config; original input audit hashes actual replay pixels but does not record replay identity. Not a second image decode or identity verification.',
        archive_source_share_method='Frozen full-manifest proportions; archive epoch covered each image plus 11 padding presentations. This is not the source distribution of a particular 250-update window.',
        raw_weight_sums=dict(paired=paired_sum, replay=replay_sum),
        feature_loss_code_and_weights_identical_to_f1_g0=True, code_identity=code_identity,
        original_deterministic_sampler_definition_identical=sampler_same,
        archived_settings=dict(resolutions=archive_config['data']['buckets'],
            effective_batch=old_performance['effective_batch'], compute_batch_cap=old_performance['compute_batch'],
            logical_batch=archive_config['train']['batch_size'],
            actual_accumulation=old_performance['effective_batch'] // archive_config['train']['batch_size'],
            sampling='Full-image permutation, one coverage epoch; not source/scene mixture',
            augmentation='Largest aspect crop, random location, horizontal flip, PIL bilinear resize',
            optimizer='AdamW moments/RNG preserved from prior181103; final archive retains232288 optimizer',
            optimizer_proof=file_ref(archive / 'archive-verification.json')),
        control_settings=dict(resolutions=config['data']['buckets'],
            effective_batch=config['train']['batch_size'] * config['train']['accumulation'],
            compute_batch=config['train']['batch_size'], accumulation=config['train']['accumulation'],
            sampling='Half source/scene/frame mixture of accepted corpus, half seven-source RGB replay mixture; with replacement',
            augmentation='Half camera-aware centered ray rotation/projective RGB warp, center U(.4,.6), zoom U(1,1.2), flip and RGB prefilter/remap; half original RGB aspect crop/flip/resize',
            optimizer='Fresh AdamW; only archived adapter and fixed RMS restored, seed43/cursor0'),
        common=dict(lr=config['train']['lr'], lr_schedule=config['train']['lr_schedule'],
            weight_decay=config['train']['weight_decay'], clip_grad=config['train']['clip_grad'],
            precision=config['train']['precision'], model_revisions=provenance['revisions'],
            adapter_config_equal=True, frozen_rms_equal_from_existing_initial_proof=True),
        saved_f1_g0_validation={name: {key: v['metrics'][key] for key in
            ('macro/spatial/cosine', 'macro/summary/cosine', 'feature/spatial', 'feature/summary',
             'feature/cosine', 'feature/summary_cosine', 'spatial/abs_log_norm_ratio',
             'summary/abs_log_norm_ratio', 'spatial/relative_l2', 'summary/relative_l2')}
            for name, v in review['arms'][control.name]['views'].items()},
        findings=[
            'Geometry-free controls preserve the new corpus training view and sampling changes. They are not original feature-pipeline continuations.',
            'The f1/G0 feature formula and four coefficients match the archive, yet both primary cosines and LiDAR regress; coefficient change alone is insufficient.',
            'The original runtime performance override uses effective batch32, not the16 implied by config batch4/accumulation4. Controls use actual batch1/accumulation16.',
            'Feature-only ScanNet DSLR contribution fell from approximately51.37% of original full manifest to8% expected,8.175% in actual nonreplay feature slots over4000. Replay contains no ScanNet.',
            'All four feature residual components rise on original-resolution validation in f1/G0; this is not evidence of a simple heldout MSE improvement traded for cosine.',
            'No isolated intervention tested sampling, view transform, resolution, effective batch, or preserved-versus-reset Adam; causal attribution remains unresolved.'],
        next_diagnostic='Establish an original-feature-pipeline continuation reference before attributing failure to geometry, then isolate shared changes. Keep confirmed input exclusions, fixed joint criterion and evaluation binding; no automatic optimizer launch.',
        model_inference_performed=False, checkpoint_payloads_read=False, rgb_payloads_read=False,
        optimizer_updates=0, training_admission=False, causal_effect_established=False)
    for key in ('lr', 'lr_schedule', 'weight_decay', 'clip_grad', 'precision'):
        if config['train'][key] != archive_config['train'][key]:
            raise ValueError('Unexpected common optimizer/precision setting: ' + key)
    output.mkdir(exist_ok=False)
    path = output / 'report.json'
    incoming = path.with_suffix('.json.incoming')
    with incoming.open('x') as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    os.rename(incoming, path)
    print(json.dumps(dict(report=file_ref(path), stage=metadata['stage'],
        source_comparison=source_comparison, same_feature_loss_code=True)), flush=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
