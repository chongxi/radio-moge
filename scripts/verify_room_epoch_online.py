"""Check current epoch metadata/startup or closed final W&B values without inference."""
import argparse
import json
import math
from pathlib import Path
import time

import wandb

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from review_room_feature_epoch import endpoints, epoch_shape
from wandb_verification import compare_metadata


def metadata_differences(actual_config, actual_provenance, config, provenance):
    differences = compare_metadata(actual_config, config, 'config')
    differences += compare_metadata(actual_provenance, provenance, 'provenance')
    allowed = {f'config/evaluation/checkpoint_metrics/{key}/baseline'
               for key in config['evaluation']['checkpoint_metrics']}
    allowed.add('provenance/performance/original_started_at')
    for difference in differences:
        if difference['absolute_difference'] == 0:
            continue
        if difference['path'] not in allowed:
            raise ValueError('Unexpected online metadata difference')
        value = config if difference['path'].startswith('config/') else provenance
        for key in difference['path'].split('/')[1:]:
            value = value[key]
        if difference['absolute_difference'] != math.ulp(value):
            raise ValueError('Display serialization differs by more than one ULP')
    return differences


def exact_values(actual, expected, label):
    differences = compare_metadata(actual, expected, label)
    if any(d['absolute_difference'] != 0 for d in differences):
        raise ValueError('Online committed values differ numerically')
    return differences


def final_validation_record(rows, step, expected_keys):
    """Ignore null-only history padding, but reject partial or repeated results."""
    keys = tuple(expected_keys)
    if not keys:
        raise ValueError('Final validation metrics are required')
    complete = []
    empty = []
    for row in rows:
        if row.get('optimizer_step') != step:
            continue
        values = [row.get(key) for key in keys]
        if all(value is None for value in values):
            empty.append(dict(internal_step=row.get('_step'), optimizer_step=step,
                              all_expected_validation_values_null_or_absent=True))
            continue
        if any(value is None for value in values):
            raise ValueError('Partially missing final online validation')
        complete.append(row)
    if len(complete) != 1:
        raise ValueError('Missing or repeated final online validation')
    return complete[0], empty


def verify(root, mode):
    plan = checked_json(file_ref(root/'plan.json'))
    batch, updates, _ = epoch_shape(plan)
    config = checked_json(plan['config'])
    provenance_ref = file_ref(root/'training/provenance.json')
    provenance = checked_json(provenance_ref)
    identity = checked_json(file_ref(root/'training/wandb-run.json'))
    run = wandb.Api(timeout=45).run('/'.join(identity[k] for k in ('entity', 'project', 'id')))
    metadata = metadata_differences(run.config['experiment'], run.config['provenance'], config, provenance)
    if mode == 'startup':
        proof_ref = file_ref(root/'startup-verification-01.json')
        proof = checked_json(proof_ref)
        if proof['updates'] != 20 or proof['geometry_loss'] or not proof['all_feature_weights_one']:
            raise ValueError('Missing local startup proof')
        records = []
        with (root/'training/train.jsonl').open() as stream:
            for _ in range(20):
                records.append(json.loads(next(stream)))
        expected = {r['step']:{'train/'+k:v for k,v in r.items() if k != 'step'} for r in records}
        # update_norm is intentionally logged only on profiling updates; fetch non-uniform rows.
        actual = {}
        for row in run.scan_history(page_size=100, max_step=100):
            step = row.get('optimizer_step')
            if step in expected and 'train/total' in row:
                if step in actual:
                    raise ValueError('Repeated online training update')
                actual[step] = {k:row[k] for k in expected[step] if k in row}
                if len(actual) == len(expected):
                    break
        if set(actual) != set(expected):
            raise ValueError('Online first twenty updates have not all arrived')
        values = []
        for step in sorted(expected):
            values += exact_values(actual[step], expected[step], f'training/{step}')
        checked = dict(proof=proof_ref, updates=20, scalar_values=sum(map(len, expected.values())))
    else:
        completion_ref = file_ref(root/'completed.json')
        completion = checked_json(completion_ref)
        if not completion['full_epoch_complete'] or completion['new_updates'] != updates or run.state != 'finished':
            raise ValueError('A finished online run and complete image epoch are required')
        validation_ref = file_ref(root/'training/validation.jsonl')
        records = [json.loads(line) for line in Path(validation_ref['path']).read_text().splitlines()]
        if [r['step'] for r in records] != endpoints(plan['source_step'], batch):
            raise ValueError('Unexpected validation endpoints')
        expected = {'validation/'+k:v for k,v in records[-1].items() if k != 'step'}
        final, empty = final_validation_record(
            run.scan_history(keys=['optimizer_step']+list(expected), page_size=1000),
            plan['source_step']+updates, expected)
        values = exact_values({k:final[k] for k in expected}, expected, 'final_metrics')
        checked = dict(completion=completion_ref, validation=validation_ref,
                       metrics_count=len(expected), empty_history_rows_ignored=empty)
    output = root/f'online-{mode}-01'
    output.mkdir(exist_ok=False)
    report = dict(stage=f'room_epoch_online_{mode}_verified', checked_at=time.time(),
        run_id=identity['id'], state=run.state, config=plan['config'], provenance=provenance_ref,
        all_committed_values_numerically_exact=True, metadata_representation_changes=metadata,
        value_representation_changes=values, evidence=checked, model_inference=False,
        optimizer_updates_by_verifier=0, verifier=file_ref(Path(__file__)))
    publish_json(output/'report.json', report)
    print(json.dumps(dict(report=file_ref(output/'report.json'), run_id=identity['id'], mode=mode,
                         numeric_metadata_differences=[r for r in metadata if r['absolute_difference']])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['startup', 'final'], required=True)
    args = parser.parse_args()
    verify(args.root.resolve(), args.mode)
