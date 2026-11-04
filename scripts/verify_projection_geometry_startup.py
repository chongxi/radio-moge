"""Read the first committed B updates against completed A; never run a model."""
import argparse
from itertools import islice
import json
import math
from pathlib import Path
import time

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.official_gt import OFFICIAL_GT_WEIGHTS
from review_ordinary_geometry_training import compare_input_rows


def prefix(path, count=20):
    with path.open() as source:
        rows = [json.loads(row) for row in islice(source, count)]
    if len(rows) != count:
        raise ValueError('First20 committed updates are not yet available')
    return rows


def run(root):
    a = root.parent/'projection-filtered-feature1-geometry0-02/training'
    b = root/'training'
    request = checked_json(file_ref(root/'run-request.json'))
    cfg = checked_json(request['config'])
    initial_ref = file_ref(b/'ordinary-initial-state.json')
    initial = checked_json(initial_ref)
    if (initial['stage'] != 'ordinary_actual_initial_state_verified'
            or initial['source_step'] != 232288 or initial['new_cursor'] != 0
            or initial['adapter_rms_adam_rng_exact'] is not True
            or cfg['ordinary_geometry']['multiplier'] != 1
            or set(cfg['loss'].values()) != {1.0}):
        raise ValueError('Actual initial restoration and F1/G1 contract required')
    old, current = prefix(a/'actual-rgb-inputs.jsonl'), prefix(b/'actual-rgb-inputs.jsonl')
    compare_input_rows(old, current, 20)
    log = prefix(b/'train.jsonl')
    if [r['step'] for r in log] != list(range(232289, 232309)):
        raise ValueError('Wrong first committed update steps')
    counts = dict(supervised=0, missing=0, sparse=0, metric=0)
    for i, row in enumerate(log):
        if (any(not math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
                or row['effective_batch'] != 32 or row['logical_data_cursor'] != (i+1)*8
                or row['gradient_norm'] <= 0):
            raise ValueError('Nonfinite or wrongly sized update')
        feature = sum(row[k] for k in cfg['loss'])
        geometry = sum(row[k]*weight for k, weight in OFFICIAL_GT_WEIGHTS.items())
        if not math.isclose(row['total'], feature+geometry, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError('Logged whole-update feature/geometry sum differs')
        for name, key in (('supervised','gt_supervised_images'), ('missing','gt_missing_target_images'),
                          ('sparse','gt_skipped_sparse_images'), ('metric','gt_metric_images')):
            if type(row[key]) is not int:
                raise ValueError('Lost integer coverage')
            counts[name] += row[key]
        if sum(row[k] for k in ('gt_supervised_images','gt_missing_target_images','gt_skipped_sparse_images')) != 32:
            raise ValueError('RGB coverage does not sum to32')
    nonzero = [{k:v for k,v in row.items() if k=='step' or 'update_norm' in k}
               for row in log if any('update_norm' in k and isinstance(v,(float,int)) and v>0 for k,v in row.items())]
    if not nonzero or counts['supervised'] == 0:
        raise ValueError('No observed adapter change or geometry supervision')
    original = prefix(a/'train.jsonl')[0]
    feature_equal = {k: log[0][k] == original[k] for k in cfg['loss']}
    out = root/'startup-proof-01'
    out.mkdir(exist_ok=False)
    publish_json(out/'first20-inputs.json', current)
    publish_json(out/'first20-log.json', log)
    report = dict(stage='filtered_B_first20_actual_updates_verified', actual_initial=initial_ref,
        request=file_ref(root/'run-request.json'), first20_inputs=file_ref(out/'first20-inputs.json'),
        first20_log=file_ref(out/'first20-log.json'), updates=20, actual_rgb_presentations=640,
        A_rgb_pixels_keys_compute_groups_exact=True, all_finite=True, coverage=counts,
        whole_rgb_denominator=32, weighted_sum_checked=True, nonzero_updates=nonzero,
        first_update_feature_components_equal_A=feature_equal,
        no_quality_or_historical_cause_claim=True, model_inference=False,
        checkpoint_payloads_reread=False, optimizer_updates_by_verifier=0,
        verifier=file_ref(Path(__file__)), completed_at=time.time())
    publish_json(out/'report.json', report)
    print(json.dumps(dict(report=file_ref(out/'report.json'), coverage=counts,
                         first_feature_equal_A=feature_equal)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', type=Path, required=True)
    args = parser.parse_args()
    run(args.arm.resolve())
