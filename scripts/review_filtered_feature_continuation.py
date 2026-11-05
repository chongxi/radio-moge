"""Compare a completed continuation validation with saved archive and parent results."""
import argparse
import json
from pathlib import Path
import time

from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from review_corpus_controls import change, joint_result
from review_corpus_validation import paired_rows
from review_projection_pair_arm import VIEWS, checked_json, compare_view
from review_projection_filtered_metrics import identities, normalized_manifest


def run(root, step, output):
    parent_path = root.parent / 'projection-filtered-feature1-geometry0-02/results-review-01/report.json'
    parent = checked_json(file_ref(parent_path))
    saved = checked_json(parent['saved_archive_scope'])
    selection = checked_json(saved['index'])
    kept = identities(normalized_manifest(selection['new_selection']))
    directory = root / 'training'
    provenance_ref = file_ref(directory / 'provenance.json')
    provenance, prior = checked_json(provenance_ref), checked_json(parent['provenance'])
    for key in ('revisions', 'adapter', 'data_digests'):
        if provenance[key] != prior[key]:
            raise ValueError('Model, adapter, or data identity changed: ' + key)
    config = checked_json(file_ref(root / 'config.json'))
    if config['loss'] != dict(spatial=1., summary=1., cosine=1., summary_cosine=1.):
        raise ValueError('Expected the restored feature weights')
    if any(k in config for k in ('geometry_supervision', 'ordinary_geometry', 'feature_replay')):
        raise ValueError('Expected pure feature continuation')
    initial = checked_json(file_ref(root / 'initial-verification.json'))
    if initial['step'] != 232538 or initial['cursor'] != 2000 or initial['optimizer_states'] != 100:
        raise ValueError('Missing actual parent-state continuation proof')
    log = [json.loads(line) for line in (directory / 'validation.jsonl').read_text().splitlines()]
    selected = [row for row in log if row['step'] == step]
    if len(selected) != 1 or not 232538 < step <= 233538:
        raise ValueError('Require a completed validation in the bounded continuation')
    views = {}
    for view in VIEWS:
        destination = directory / f'eval-{step:07d}'
        if view != 'original':
            destination /= view
        refs = {key: file_ref(destination / (key + '.json'))
                for key in ('metrics', 'per_image', 'protocol')}
        now = {key: checked_json(ref) for key, ref in refs.items()}
        old = saved['evidence'][view]['archive']
        result = compare_view(checked_json(old['per_image']), now['per_image'],
            checked_json(old['protocol']), now['protocol'], now['metrics'], view, kept)
        prior_files = parent['views'][view]['files']
        if now['protocol'] != checked_json(prior_files['protocol']):
            raise ValueError('Protocol differs from the completed filtered parent')
        paired_rows(checked_json(prior_files['per_image']), now['per_image'])
        prior_metrics = checked_json(prior_files['metrics'])
        prefix = '' if view == 'original' else ('resolution/native-640x400/'
            if view == 'native-640x400' else 'dataset/' + view.removeprefix('dataset-') + '/')
        for key, value in now['metrics'].items():
            if selected[0].get(prefix + key) != value:
                raise ValueError('Committed validation differs from saved metrics')
        views[view] = dict(files=refs, against_archive=result,
            against_parent={k: change(prior_metrics[k], v) for k, v in now['metrics'].items()})
    names = dict(spatial_cosine=('original', 'macro/spatial/cosine'),
        summary_cosine=('original', 'macro/summary/cosine'),
        lidar_abs_rel=('dataset-scannetpp-lidar', 'gt_student/abs_rel'))
    archive = {k: views[v]['against_archive']['metrics'][m]['before'] for k, (v, m) in names.items()}
    before = {k: views[v]['against_parent'][m]['before'] for k, (v, m) in names.items()}
    current = {k: views[v]['against_parent'][m]['final'] for k, (v, m) in names.items()}
    output.mkdir(exist_ok=False)
    report = dict(stage='saved_feature_continuation_validation_reviewed', step=step,
        new_updates=step - 232538, parent=file_ref(parent_path), provenance=provenance_ref,
        actual_initial=file_ref(root / 'initial-verification.json'),
        views=views, against_archive=joint_result(archive, current),
        against_parent=joint_result(before, current), model_inference=False,
        checkpoint_payload_read=False, optimizer_updates_by_reviewer=0,
        historical_cause_established=False, training_may_still_be_running=True,
        reviewed_at=time.time(), reviewer=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(report=file_ref(output / 'report.json'), step=step,
        against_archive=report['against_archive'], against_parent=report['against_parent'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.step, args.output.resolve())
