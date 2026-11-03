"""Prepare ARKit expansion without widening the existing partial ScanNet scope.

This adds data accounting and binds existing canonical comparisons. It does not
change the current trainer or issue a model gate. Producer payload readbacks
are reused explicitly, with per-read payload hashing still required in training.
"""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from prepare_reviewed_training import publish_db
from prepare_scannetpp_reviewed_training import isolate
from publish_coco_nyu_source_reviews import write_lines
from radio_moge.corpus_admission import build_accounting, checked_json, checked_path, file_ref
from radio_moge.corpus_coverage import metadata_coverage
from radio_moge.geometry_cache import publish_json
from radio_moge.producer_readback import MODE


def extend_inventory(db, previous_expected, added_count):
    """images is the exact prior inventory, not all rows of its source names."""
    if ('arkitscenes' in previous_expected
            or dict(db.execute('SELECT dataset,COUNT(*) FROM images GROUP BY dataset')) != previous_expected):
        raise ValueError('Exact previous source scope required')
    missing = db.execute('SELECT COUNT(*) FROM images i LEFT JOIN original.images o ON i.ordinal=o.ordinal '
        'WHERE o.ordinal IS NULL OR i.dataset IS NOT o.dataset OR i.image IS NOT o.image '
        'OR i.sha256 IS NOT o.sha256 OR i.scene_group IS NOT o.scene_group '
        'OR i.row_json IS NOT o.row_json').fetchone()[0]
    if missing:
        raise ValueError('Previous inventory no longer matches frozen original identities')
    db.execute("INSERT INTO images SELECT * FROM original.images WHERE dataset='arkitscenes'")
    expected = dict(previous_expected, arkitscenes=added_count)
    if dict(db.execute('SELECT dataset,COUNT(*) FROM images GROUP BY dataset')) != expected:
        raise ValueError('Expanded source count differs')
    return expected


def run(root, output):
    parent_ref = file_ref(root / 'inventory.json')
    parent = checked_json(parent_ref)
    original_ref = dict(path=str(root / 'inventory.sqlite'), sha256=parent['inventory_sha256'])
    previous_ref = file_ref(root / 'reviewed-stage3-data-01/accounting/report.json')
    previous = checked_json(previous_ref)
    prior_plan = checked_json(previous['plan'])
    added_ref = file_ref(root / 'arkit-source-review-02/report.json')
    added = checked_json(added_ref)
    if (previous['stage'] != 'accounting_verified_not_accepted' or previous['selected_images'] != 403382
            or previous['total_images'] != 404795 or previous['unresolved'] != 0
            or previous['expected_counts'].get('scannetpp_dslr') != 35783
            or previous['manifest_sha256'] != parent['manifest_sha256']
            or added['stage'] != 'arkit_source_review_and_routes_complete_not_corpus_admission'
            or added['source_disposition_counts'] != dict(native_source_eligible=194916, excluded_geometry=2626)
            or added['original_images'] != 197542 or added['original_source_flags_retained'] is not True
            or added['training_admission'] is not False):
        raise ValueError('Exact previous stage and qualified ARKit source required')
    review = checked_json(added['source_review'])
    if (review['label_kinds'] != ['measured_sensor'] or review['eligible_candidate_images'] != 194916
            or review['quarantined_candidate_images'] != 8 or review['source_flags_cleared'] is not False):
        raise ValueError('Native measured-sensor review differs')
    sources = [*prior_plan['sources'], dict(dataset='arkitscenes', scope_count=197542,
        generation_report=file_ref(root / 'arkit-native-cache-01/report.json'),
        packs_index=file_ref(root / 'arkit-native-cache-01/packs/packs.json'),
        review=added['source_review'], payload_verification=MODE)]
    old_exclusions = [json.loads(line) for line in checked_path(prior_plan['exclusions']).read_text().splitlines()]
    all_exclusions = [json.loads(line) for line in checked_path(added['known_geometry_exclusions']).read_text().splitlines()]
    new_exclusions = [row for row in all_exclusions if row['dataset'] == 'arkitscenes']
    if len(old_exclusions) != 1413 or len(new_exclusions) != 2626:
        raise ValueError('Reviewed exclusion counts differ')
    exclusions = sorted(old_exclusions + new_exclusions, key=lambda x: x['corpus_ordinal'])
    if (len({x['corpus_ordinal'] for x in exclusions}) != 4039
            or len({x['image'] for x in exclusions}) != 4039):
        raise ValueError('Duplicate expanded exclusion identity')
    output.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='reviewed-arkit-stage-') as temporary:
        work = Path(temporary)
        for name, ref in (('original', original_ref), ('inventory', prior_plan['inventory'])):
            shutil.copyfile(checked_path(ref), work / (name + '.sqlite'))
            if file_ref(work / (name + '.sqlite'))['sha256'] != ref['sha256']:
                raise ValueError('Immutable inventory copy differs')
        with sqlite3.connect(work / 'inventory.sqlite') as db:
            db.execute('ATTACH DATABASE ? AS original', (str(work / 'original.sqlite'),))
            expected = extend_inventory(db, previous['expected_counts'], 197542)
        publish_db(work / 'inventory.sqlite', output / 'inventory.sqlite')
        inventory_ref = file_ref(output / 'inventory.sqlite')
        publish_json(output / 'inventory.json', dict(parent, dataset_counts=expected, images=sum(expected.values()),
            inventory_sha256=inventory_ref['sha256'], parent_inventory=original_ref,
            parent_inventory_report=parent_ref, scope_kind='reviewed_source_subset'))
        publish_json(output / 'scope.json', dict(kind='reviewed_source_subset', expected_counts=expected,
            manifest_sha256=parent['manifest_sha256'], parent_inventory_report=parent_ref,
            previous_accounting=previous_ref, added_source_review=added_ref, payload_verification=MODE))
        write_lines(output / 'exclusions.jsonl', exclusions)
        plan = dict(scope=file_ref(output / 'scope.json'), inventory=inventory_ref,
            inventory_report=file_ref(output / 'inventory.json'), sources=sources,
            exclusions=file_ref(output / 'exclusions.jsonl'))
        print(json.dumps(dict(stage='accounting_seven_sources', original_images=602337,
                              native_sensor_candidates_expected=194916)), flush=True)
        accounting = build_accounting(plan, output / 'accounting')
        if (accounting['selected_counts'] != dict(previous['selected_counts'], arkitscenes=194916)
                or accounting['selected_images'] != 598298 or sum(accounting['excluded_counts'].values()) != 4039):
            raise ValueError('Expanded accounting differs from exact reviewed routes')
        shutil.copyfile(checked_path(accounting['index']), work / 'accounting.sqlite')
        with sqlite3.connect(work / 'accounting.sqlite') as db:
            coverage = metadata_coverage(db, accounting=True)
            labels = dict(db.execute("SELECT label_kind,COUNT(*) FROM images WHERE disposition='selected' GROUP BY label_kind"))
        if labels != dict(measured_sensor=194916, synthetic_renderer=189025, teacher_pseudo=214357):
            raise ValueError('Native sensor, synthetic and relative teacher counts differ')
    publish_json(output / 'coverage-summary.json', dict(metadata=coverage, label_counts=labels,
        source_geometry_statistics=accounting['coverage'], crop_support_review_complete=False,
        statistics='Unchanged completed-producer support and quantiles; no new pixel histogram', training_admission=False))
    for ref in (parent_ref, previous_ref, added_ref, prior_plan['exclusions'], added['known_geometry_exclusions']):
        checked_path(ref)
    report = dict(stage='reviewed_expansion_accounted_isolation_pending', completed_at=time.time(),
        original_images=602337, selected_images=598298, excluded_images=4039,
        native_measured_sensor_images=194916, native_synthetic_images=189025, relative_pseudo_images=214357,
        accounting=file_ref(output / 'accounting/report.json'), coverage=file_ref(output / 'coverage-summary.json'),
        previous_accounting=previous_ref, added_source_review=added_ref,
        original_scannet_subset_preserved=True, training_admission=False, optimizer_updates=0,
        actual_model_probe_complete=False, active_training_changed=False, code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(stage=report['stage'], selected=598298)), flush=True)
    isolate(root, output, root / 'heldout-rgb-full-comparison-01/report.json', expected_selected=598298)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
