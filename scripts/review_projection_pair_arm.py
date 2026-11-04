"""Review a fixed filtered endpoint from saved results, without model inference."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from radio_moge.corpus_admission import checked_json as checked_json_exact, checked_path, file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.metrics import aggregate, dataset_metrics
from review_corpus_controls import change, joint_result
from review_corpus_validation import paired_rows
from review_projection_filtered_metrics import normalized_manifest, identities


VIEWS = ('original', 'native-640x400', 'dataset-scannetpp-lidar', 'dataset-scannetpp-dslr')


def checked_json(ref):
    # Reports can be mirrored between workspace and GPU mount roots. Retain
    # their original identity and require the same content hash at either path.
    resolved = dict(ref)
    if not Path(ref['path']).exists() and ref['path'].startswith('/shared/'):
        resolved['path'] = ref['path'].replace('/shared/', '/home/buildhub/shared/', 1)
    return checked_json_exact(resolved)


def metrics_for(rows):
    value = aggregate(rows)
    value.update(dataset_metrics(rows))
    return value


def compare_view(old_rows, new_rows, old_protocol, new_protocol, new_metrics, view, kept):
    original = view in ('original', 'native-640x400')
    retained = [r for r in old_rows if r['dataset'] != 'coco2017'] if original else old_rows
    expected_protocol = deepcopy(old_protocol)
    if original:
        if (len(old_rows) != 1268 or len(retained) != 1012
                or identities(retained) != kept or old_protocol['images'] != 1268):
            raise ValueError('Only the predeclared COCO validation removal is allowed')
        expected_protocol['images'] = 1012
    if new_protocol != expected_protocol:
        raise ValueError('Evaluation protocol changed beyond the bound image count')
    paired_rows(retained, new_rows)
    if metrics_for(new_rows) != new_metrics:
        raise ValueError('Saved metrics do not exactly reproduce from per-image rows')
    baseline = metrics_for(retained)
    if baseline.keys() != new_metrics.keys():
        raise ValueError('Saved metric coverage changed')
    return dict(images=len(retained), same_order=True, saved_teacher_gt_scalars_unchanged=True,
                protocol_count_only_transition=original,
                metrics={k: change(baseline[k], new_metrics[k]) for k in baseline})


def run(root, output):
    corpus = root.parent
    pair_ref = file_ref(corpus/'projection-filtered-comparison-plan-01/plan.json')
    # The immutable admission helper pins the predeclared pair, not a new rule.
    from radio_moge.projection_fork import projection_comparison
    pair = projection_comparison(pair_ref)
    saved_ref = file_ref(corpus/'projection-filtered-metrics-01/report.json')
    saved = checked_json(saved_ref)
    selection = checked_json(saved['index'])
    kept = identities(normalized_manifest(selection['new_selection']))
    complete_ref = file_ref(root/'completed.json')
    complete = checked_json(complete_ref)
    request = checked_json(complete['request'])
    config = checked_json(request['config'])
    if (complete['optimizer_updates'] != 250 or complete['final_state_finite'] is not True
            or complete['fixed_statistics_unchanged'] is not True
            or complete['stage'] not in ('fixed250_reference_training_complete', 'ordinary_fixed250_training_complete')
            or config['train']['steps'] != 232538 or config['train']['eval_at_start'] is not False
            or config['data']['val_manifest'] != selection['new_selection']['path']):
        raise ValueError('Completed fixed250 filtered endpoint required')
    directory = Path(config['output_dir'])
    provenance_ref = file_ref(directory/'provenance.json')
    provenance = checked_json(provenance_ref)
    prior_ref = file_ref(corpus/'feature-original-reference-02/training/provenance.json')
    prior = checked_json(prior_ref)
    if (provenance['revisions'] != prior['revisions'] or provenance['adapter'] != prior['adapter']
            or provenance['train_images'] != 1597911 or provenance['validation_images'] != 1012
            or provenance['data_digests']['val'] != '06096343da9ab1138a1c8e28b1c016003e73ec16cb4c83501fce6ef101d09f7c'
            or any(provenance['data_digests'][k] != prior['data_digests'][k]
                   for k in ('dataset/scannetpp-dslr', 'dataset/scannetpp-lidar'))):
        raise ValueError('Model or validation identity differs from the bound transition')
    audit = checked_json(complete['actual_inputs'])
    if (audit['presentations'] != 8000 or audit['optimizer_updates'] != 250 or audit['cursor'] != 2000
            or audit['planned_keys_matched'] is not True or audit['actual_compute_groups_verified'] is not True
            or audit['source_counts'].get('coco2017', 0) != 0):
        raise ValueError('Actual full endpoint scope differs')
    records = [json.loads(line) for line in (directory/'validation.jsonl').read_text().splitlines()]
    if len(records) != 1 or records[0]['step'] != 232538:
        raise ValueError('Exactly one fixed final validation record is required')
    views = {}
    for view in VIEWS:
        destination = directory/'eval-0232538'
        if view != 'original':
            destination /= view
        refs = {k: file_ref(destination/(k+'.json')) for k in ('metrics', 'per_image', 'protocol')}
        old = saved['evidence'][view]['archive']
        current = {k: checked_json(ref) for k, ref in refs.items()}
        result = compare_view(checked_json(old['per_image']), current['per_image'],
            checked_json(old['protocol']), current['protocol'], current['metrics'], view, kept)
        prefix = '' if view == 'original' else ('resolution/native-640x400/' if view == 'native-640x400'
                                                else 'dataset/'+view.removeprefix('dataset-')+'/')
        if any(records[0].get(prefix+k) != v for k, v in current['metrics'].items()):
            raise ValueError('Validation log differs from saved metrics')
        views[view] = dict(**result, files=refs, archive=old)
    keys = dict(spatial_cosine=('original', 'macro/spatial/cosine'),
                summary_cosine=('original', 'macro/summary/cosine'),
                lidar_abs_rel=('dataset-scannetpp-lidar', 'gt_student/abs_rel'))
    baseline = {k: views[v]['metrics'][m]['before'] for k, (v, m) in keys.items()}
    final = {k: views[v]['metrics'][m]['final'] for k, (v, m) in keys.items()}
    # Baseline was fixed in the pair plan before either filtered arm ran.
    rule = pair['primary_joint_rule']
    if baseline != dict(spatial_cosine=rule['spatial_cosine_gt'],
                        summary_cosine=rule['summary_cosine_gt'], lidar_abs_rel=rule['lidar_absrel_lt']):
        raise ValueError('Predeclared paired primary baseline differs')
    output.mkdir(exist_ok=False)
    report = dict(stage='filtered_pair_arm_fixed250_reviewed', completion=complete_ref,
        pair_plan=pair_ref, saved_archive_scope=saved_ref, provenance=provenance_ref,
        previous_model_provenance=prior_ref, actual_inputs=complete['actual_inputs'],
        validation_log=file_ref(directory/'validation.jsonl'), views=views,
        primary=joint_result(baseline, final), fixed_final_not_best_selected=True,
        online_final_verification_pending=True, model_inference=False, checkpoint_payload_read=False,
        optimizer_updates_by_reviewer=0, historical_training_exposure_removed=False,
        historical_regression_cause_established=False,
        interpretation='Single-seed fixed250; both six-source original cosines up and unchanged independent LiDAR error down. Saved teacher scalars and revisions checked, not tensor equality. No separate laser evaluation.',
        completed_at=time.time(), reviewer=file_ref(Path(__file__)))
    publish_json(output/'report.json', report)
    print(json.dumps(dict(report=file_ref(output/'report.json'), primary=report['primary'])), flush=True)
    return report


def close_pair(a_path, b_path, output):
    """Compare the predeclared arms only after both fixed endpoints are reviewed."""
    from review_ordinary_geometry_training import compare_input_rows
    refs = {k: file_ref(p) for k, p in (('A', a_path), ('B', b_path))}
    a, b = (checked_json(refs[k]) for k in ('A', 'B'))
    if (a['stage'] != 'filtered_pair_arm_fixed250_reviewed' or a['stage'] != b['stage']
            or a['pair_plan'] != b['pair_plan'] or a['saved_archive_scope'] != b['saved_archive_scope']):
        raise ValueError('Two fixed endpoints from the same predeclared pair are required')
    completions = [checked_json(x['completion']) for x in (a, b)]
    configs = [checked_json(checked_json(x['request'])['config']) for x in completions]
    spec = deepcopy(configs[1]).pop('ordinary_geometry', None)
    if spec is None or spec['multiplier'] != 1 or 'ordinary_geometry' in configs[0]:
        raise ValueError('Expected feature-only A and ordinary geometry B')
    normalized = deepcopy(configs[1])
    normalized.pop('ordinary_geometry')
    normalized['data'].pop('compact_transport')
    for key in ('output_dir', 'tracking'):
        normalized[key] = configs[0][key]
    if normalized != configs[0]:
        raise ValueError('Paired configurations differ outside declared geometry/transport/metadata')
    provenances = [checked_json(x['provenance']) for x in (a, b)]
    if any(provenances[0][k] != provenances[1][k] for k in ('data_digests', 'revisions', 'adapter')):
        raise ValueError('Paired model/data identities differ')
    audits = [checked_json(x['actual_inputs']) for x in (a, b)]
    records = [[json.loads(line) for line in checked_path(x['records']).read_text().splitlines()]
               for x in audits]
    compare_input_rows(records[0], records[1])
    views = {}
    for view in VIEWS:
        old, new = a['views'][view], b['views'][view]
        if old['archive'] != new['archive'] or old['images'] != new['images']:
            raise ValueError('Paired evaluation scope differs')
        paired_rows(checked_json(old['files']['per_image']), checked_json(new['files']['per_image']))
        if checked_json(old['files']['protocol']) != checked_json(new['files']['protocol']):
            raise ValueError('Paired evaluation protocol differs')
        if old['metrics'].keys() != new['metrics'].keys():
            raise ValueError('Paired metric coverage differs')
        views[view] = {k: change(old['metrics'][k]['final'], new['metrics'][k]['final']) for k in old['metrics']}
    before = {k: v['final'] for k, v in a['primary']['metrics'].items()}
    after = {k: v['final'] for k, v in b['primary']['metrics'].items()}
    output.mkdir(exist_ok=False)
    result = dict(stage='filtered_pair_fixed250_comparison_complete', arm_reviews=refs,
        actual_input_records=[x['records'] for x in audits], actual8000_rgb_pixels_keys_groups_exact=True,
        ignored_input_fields=['logical_images[*].ordinary_geometry'], views_B_vs_A=views,
        primary_A_vs_archive=a['primary'], primary_B_vs_archive=b['primary'],
        primary_B_vs_A=joint_result(before, after), model_inference=False,
        checkpoint_payload_read=False, optimizer_updates_by_reviewer=0,
        historical_cause_established=False, replication_required=True,
        completed_at=time.time(), reviewer=file_ref(Path(__file__)))
    publish_json(output/'report.json', result)
    print(json.dumps(dict(report=file_ref(output/'report.json'), primary_B_vs_A=result['primary_B_vs_A'])))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', type=Path)
    parser.add_argument('--A-review', type=Path)
    parser.add_argument('--B-review', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.arm and not (args.A_review or args.B_review):
        run(args.arm.resolve(), args.output.resolve())
    elif args.A_review and args.B_review and not args.arm:
        close_pair(args.A_review.resolve(), args.B_review.resolve(), args.output.resolve())
    else:
        parser.error('Provide one arm, or both completed A/B reviews')
