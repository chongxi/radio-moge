"""Verify all final online validation scalars of the completed mixed RGB run."""
import argparse
import json
import math
from pathlib import Path
import time

import wandb

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from wandb_verification import compare_metadata


def verify(root):
    completion_ref = file_ref(root / 'completed.json')
    completion = checked_json(completion_ref)
    config_ref = file_ref(root / 'config.json')
    config = checked_json(config_ref)
    training = root / 'training'
    provenance_ref = file_ref(training / 'provenance.json')
    provenance = checked_json(provenance_ref)
    identity = checked_json(file_ref(training / 'wandb-run.json'))
    if not completion['fixed1000_complete'] or completion['new_updates'] != 1000:
        raise ValueError('A closed fixed1000 endpoint is required')
    run = wandb.Api(timeout=45).run('/'.join(identity[k] for k in ('entity', 'project', 'id')))
    if run.state != 'finished':
        raise ValueError('Online run has not finished: ' + run.state)
    metadata = compare_metadata(run.config['experiment'], config, 'config')
    metadata += compare_metadata(run.config['provenance'], provenance, 'provenance')
    allowed = {f'config/evaluation/checkpoint_metrics/{key}/baseline'
               for key in ('nyu-delta1', 'spatial', 'teacher-depth')}
    allowed.add('provenance/performance/original_started_at')
    for difference in metadata:
        if difference['absolute_difference'] == 0:
            continue
        if difference['path'] not in allowed:
            raise ValueError('Unexpected online metadata difference')
        value = config if difference['path'].startswith('config/') else provenance
        for key in difference['path'].split('/')[1:]:
            value = value[key]
        if difference['absolute_difference'] != math.ulp(value):
            raise ValueError('Known display serialization differs by more than one ULP')
    validation_ref = file_ref(training / 'validation.jsonl')
    records = [json.loads(line) for line in Path(validation_ref['path']).read_text().splitlines()]
    if [r['step'] for r in records] != [233788, 234038, 234288, 234538]:
        raise ValueError('Unexpected validation endpoints')
    expected = {'validation/' + k: v for k, v in records[-1].items() if k != 'step'}
    procthor_keys = [k for k in expected if k.startswith('validation/dataset/procthor-empty/')]
    structural_keys = [k for k in expected if k.startswith('validation/dataset/structural-rooms/')]
    if not procthor_keys or not structural_keys:
        raise ValueError('Missing unseen-house validation metrics')
    final = [row for row in run.scan_history(keys=['optimizer_step'] + list(expected), page_size=4)
             if row['optimizer_step'] == 234538]
    if len(final) != 1:
        raise ValueError('Missing or repeated online final validation')
    differences = compare_metadata({k: final[0][k] for k in expected}, expected, 'final_metrics')
    if any(d['absolute_difference'] != 0 for d in differences):
        raise ValueError('Online validation differs numerically from committed local results')
    output = root / 'online-final-01'
    output.mkdir(exist_ok=False)
    report = dict(stage='empty_room_feature1000_online_final_verified', completed_at=time.time(),
        run_id=identity['id'], state=run.state, completion=completion_ref, config=config_ref,
        provenance=provenance_ref, validation=validation_ref, metrics_count=len(expected),
        procthor_metrics_count=len(procthor_keys), structural_metrics_count=len(structural_keys), all_metrics_numerically_exact=True,
        metadata_representation_changes=metadata, metric_representation_changes=differences,
        model_inference=False, optimizer_updates_by_verifier=0, verifier=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(report=file_ref(output / 'report.json'), metrics=len(expected),
                         procthor_metrics=len(procthor_keys), state=run.state)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    verify(args.root.resolve())
