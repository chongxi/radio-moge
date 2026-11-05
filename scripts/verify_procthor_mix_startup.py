"""Verify the first committed mixed-feature updates without rereading the model."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.data import DeterministicBatches
from radio_moge.geometry_cache import publish_json


def run(root):
    plan = checked_json(file_ref(root/'plan.json'))
    config = checked_json(plan['config'])
    metadata = checked_json(file_ref(root/'metadata-report.json'))
    initial = checked_json(file_ref(root/'initial-verification.json'))
    model = checked_json(file_ref(root/'mixed-model-zero-update.json'))
    if (initial['step'] != plan['source_step'] or initial['cursor'] != 0
            or initial['real_stream_start_cursor'] != plan['source_cursor']
            or initial['optimizer_states'] != 100 or model['optimizer_updates'] != 0
            or model['state_before'] != model['state_after']
            or model['rng_restored_bitwise'] is not True or model['geometry_branch_executed'] is not False
            or config['loss'] != dict(spatial=1., summary=1., cosine=1., summary_cosine=1.)):
        raise ValueError('Missing actual initial/model proof or changed feature coefficients')
    training = root/'training'
    log = []
    with (training/'train.jsonl').open() as file:
        for line in file:
            log.append(json.loads(line))
            if len(log) == 20:
                break
    if len(log) != 20:
        raise ValueError('Wait for twenty committed optimizer updates')
    for i, row in enumerate(log):
        if (row['step'] != plan['source_step']+i+1 or row['logical_data_cursor'] != (i+1)*8
                or row['effective_batch'] != 32 or row['lr_used'] != 1e-5
                or row['gradient_norm'] <= 0
                or any(not math.isfinite(v) for v in row.values() if isinstance(v, (float, int)))
                or not math.isclose(row['total'], sum(row[k] for k in config['loss']),
                                    rel_tol=1e-5, abs_tol=2e-6)):
            raise ValueError('Training loss/batch/step/cursor is not the declared pure-feature update')
    update_norms = [r['update_norm'] for r in log if 'update_norm' in r]
    if not update_norms or any(v <= 0 for v in update_norms):
        raise ValueError('No measured nonzero adapter update')
    original = [list(k) for batch in DeterministicBatches(metadata['real_images'], 4,
        config['data']['buckets'], config['train']['seed'], plan['source_cursor'], 160) for k in batch]
    planned = checked_json(file_ref(root/'planned-keys.json'))[:640]
    counts, houses, observed = Counter(), set(), []
    with (training/'actual-rgb-inputs.jsonl').open() as file:
        for i, line in enumerate(file):
            if i == 20:
                break
            row = json.loads(line)
            if row['update'] != log[i]['step'] or len(row['logical_images']) != 32:
                raise ValueError('Input audit and successful optimizer updates differ')
            for item in row['logical_images']:
                slot = len(observed)
                if item['key'] != planned[slot] or 'ordinary_geometry' in item:
                    raise ValueError('Planned RGB key or geometry-free input contract differs')
                replaced = (slot+1) % 8 == 0
                if replaced != (item['dataset'] == 'procthor_empty'):
                    raise ValueError('Synthetic source share changed')
                if (item['key'][1:] != original[slot][1:] or
                        (not replaced and item['key'] != original[slot])):
                    raise ValueError('Original real permutation/crop/resolution key changed')
                if replaced:
                    houses.add(item['group'])
                counts[item['dataset']] += 1
                observed.append(item)
    if len(observed) != 640 or counts['procthor_empty'] != 80 or counts.get('coco2017', 0):
        raise ValueError('Committed startup RGB population differs')
    report = dict(stage='actual_mixed_feature_startup_verified', updates=20,
        initial=file_ref(root/'initial-verification.json'), model=file_ref(root/'mixed-model-zero-update.json'),
        plan=file_ref(root/'plan.json'), actual_rgb_inputs=file_ref(training/'actual-rgb-inputs.jsonl'),
        input_file_is_growing=True, verified_prefix_updates=20, presentations=640,
        source_counts=dict(counts), synthetic_houses=len(houses), original_unreplaced_keys_exact=True,
        all_crop_seeds_and_buckets_exact=True, all_feature_weights_one=True, geometry_loss=False,
        finite_loss_and_gradients=True, nonzero_measured_update_norms=update_norms,
        config=plan['config'], verifier=file_ref(Path(__file__)), optimizer_updates_by_verifier=0)
    # Bind the exact inspected prefix separately; the trainer's full audit keeps growing.
    publish_json(root/'startup-input-prefix.json', observed)
    report['fixed_prefix'] = file_ref(root/'startup-input-prefix.json')
    publish_json(root/'startup-verification-01.json', report)
    print(json.dumps(dict(report=file_ref(root/'startup-verification-01.json'),
                          updates=20, source_counts=dict(counts))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
