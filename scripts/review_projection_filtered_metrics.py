"""Reaggregate fixed saved evaluations under a source hold, without inference.

Preserve historical scores and criteria. The new six-source scores are post-hoc
diagnostics, not evidence that removing training contamination fixes a model.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

COSINES = ('spatial/cosine', 'summary/cosine')


def checked(ref):
    path = Path(ref['path'])
    if not path.exists() and str(path).startswith('/home/buildhub/shared/'):
        path = Path(str(path).replace('/home/buildhub/shared/', '/shared/', 1))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('Bound evidence changed: ' + ref['path'])
    return raw


def file_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def identities(rows):
    keys = [(r['image'], r['dataset']) for r in rows]
    if len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Duplicate saved image identity')
    return keys


def cosine_summary(rows):
    identities(rows)
    sources = sorted({r['dataset'] for r in rows})
    if not sources:
        raise ValueError('Empty evaluation scope')
    result, means = {}, {}
    for source in sources:
        subset = [r for r in rows if r['dataset'] == source]
        means[source] = {}
        for key in COSINES:
            if any(type(r.get(key)) not in (int, float) or not math.isfinite(r[key]) for r in subset):
                raise ValueError('Missing or nonfinite per-image cosine')
            means[source][key] = sum(r[key] for r in subset) / len(subset)
    for key in COSINES:
        result[key] = sum(means[s][key] for s in sources) / len(sources)
    return dict(images=len(rows), dataset_counts=dict(Counter(r['dataset'] for r in rows)),
                macro=result, datasets=means)


def reaggregate(rows, metrics, expected_keys, kept_keys):
    if identities(rows) != expected_keys:
        raise ValueError('Saved image identities or ordering differ')
    original = cosine_summary(rows)
    for key in COSINES:
        if original['macro'][key] != metrics['macro/' + key]:
            raise ValueError('Saved historical macro does not reproduce exactly')
    retained = [r for r in rows if r['dataset'] != 'coco2017']
    if identities(retained) != kept_keys:
        raise ValueError('Held-source removal changed other identities or ordering')
    return dict(historical=original, filtered=cosine_summary(retained))


def normalized_manifest(ref):
    rows = [json.loads(line) for line in checked(ref).splitlines() if line.strip()]
    for row in rows:
        for key in ('image', 'depth', 'geometry'):
            if key in row:
                row[key] = os.path.normpath(os.path.join(str(Path(ref['path']).parent), row[key]))
    return rows


def run(index_path, visual_path, paired_path, output):
    index_ref, visual_ref, paired_ref = [file_ref(p) for p in (index_path, visual_path, paired_path)]
    index, visual, paired = [json.loads(checked(r)) for r in (index_ref, visual_ref, paired_ref)]
    if (visual['index']['sha256'] != index_ref['sha256']
            or visual['stage'] != 'heldout_coco_visual_review_complete_source_hold_retained'
            or visual['all_coco_held_pending_individual_admission'] is not True
            or visual['selection_uses_model_metrics'] is not False):
        raise ValueError('A projection-only visual source hold is required')
    old, new = [normalized_manifest(index[k]) for k in ('old_selection', 'new_selection')]
    if (new != [r for r in old if r['dataset'] != 'coco2017']
            or len(old) != 1268 or len(new) != 1012):
        raise ValueError('New validation must retain all and only non-COCO rows unchanged')
    comparisons, evidence, results = paired['comparisons'], {}, {}
    for view in ('original', 'native-640x400', 'dataset-scannetpp-lidar', 'dataset-scannetpp-dslr'):
        current = comparisons['archive']['views'][view]['files']
        refs = {'ordinary_geometry': current}
        for name in ('archive', 'reference02'):
            previous = comparisons[name]['views'][view]
            refs[name] = dict(metrics=previous['baseline'], per_image=previous['baseline_per_image'],
                protocol=dict(path=str(Path(previous['baseline']['path']).with_name('protocol.json')),
                              sha256=current['protocol']['sha256']))
        records = {name: json.loads(checked(r['per_image'])) for name, r in refs.items()}
        protocols = {name: checked(r['protocol']) for name, r in refs.items()}
        if len(set(protocols.values())) != 1:
            raise ValueError('Saved evaluation protocols differ')
        expected = identities(old) if view in ('original', 'native-640x400') else identities(records['archive'])
        kept = identities(new) if view in ('original', 'native-640x400') else expected
        evidence[view], results[view] = refs, {}
        for name, refs_for_run in refs.items():
            results[view][name] = reaggregate(records[name], json.loads(checked(refs_for_run['metrics'])), expected, kept)
            if view == 'dataset-scannetpp-lidar':
                values = [r['gt_student/abs_rel'] for r in records[name]]
                if len(values) != 120 or not all(math.isfinite(v) for v in values):
                    raise ValueError('Independent LiDAR scope differs')
                value = sum(values) / len(values)
                if value != json.loads(checked(refs_for_run['metrics']))['gt_student/abs_rel']:
                    raise ValueError('Independent LiDAR score does not reproduce')
                results[view][name]['unchanged_lidar_abs_rel'] = value
    joint = {}
    for name in ('reference02', 'ordinary_geometry'):
        baseline = results['original']['archive']['filtered']['macro']
        final = results['original'][name]['filtered']['macro']
        lidar_before = results['dataset-scannetpp-lidar']['archive']['unchanged_lidar_abs_rel']
        lidar_after = results['dataset-scannetpp-lidar'][name]['unchanged_lidar_abs_rel']
        joint[name] = dict(cosine_changes={k: final[k] - baseline[k] for k in COSINES},
            lidar_change=lidar_after - lidar_before,
            both_cosines_up_and_lidar_down=all(final[k] > baseline[k] for k in COSINES) and lidar_after < lidar_before)
    output.mkdir(exist_ok=False)
    report = dict(stage='saved_projection_filtered_metrics_reaggregated', index=index_ref,
        visual_review=visual_ref, existing_paired_review=paired_ref, evidence=evidence, views=results,
        comparisons_to_archive=joint, new_scope='Six-source 1012-image macro; equal weight per source.',
        posthoc_diagnostic=True, historical_criteria_or_scores_modified=False,
        filtering_uses_model_performance=False, model_inference=False, optimizer_updates=0,
        training_admission=False, historical_training_exposure_removed=False,
        historical_regression_cause_established=False, implementation=file_ref(__file__))
    incoming = output / 'report.json.incoming'
    incoming.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.rename(incoming, output / 'report.json')
    print(json.dumps(dict(report=file_ref(output / 'report.json'), comparisons=joint)), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'visual', 'paired', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    run(args.index, args.visual, args.paired, args.output)
