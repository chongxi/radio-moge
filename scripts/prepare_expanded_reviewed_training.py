"""Prepare a new five-source stage with reviewed Structured3D candidates.

Completed producer readbacks are reused explicitly. The unchanged seven-source
RGB comparison already includes Structured3D; every newly selected identity is
checked against that immutable comparison before its result can be reused.
This creates data evidence, not a training seal or a running experiment.
"""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from prepare_reviewed_training import publish_db
from radio_moge.corpus_admission import (build_accounting, checked_json, checked_path,
    file_ref)
from radio_moge.corpus_coverage import metadata_coverage
from radio_moge.geometry_cache import publish_json
from radio_moge.producer_readback import MODE


def verify_isolated_geometry(db, expected):
    """Both databases are local copies of SHA-bound, immutable reports."""
    counts = dict(db.execute("SELECT dataset,COUNT(*) FROM images WHERE disposition='selected' "
                             "GROUP BY dataset"))
    if counts != expected:
        raise ValueError('Selected expanded source counts differ')
    missing = db.execute("SELECT COUNT(*) FROM images i LEFT JOIN isolated.training t "
        "ON i.ordinal=t.ordinal WHERE i.disposition='selected' AND "
        "(t.ordinal IS NULL OR i.dataset IS NOT t.dataset OR i.scene_group IS NOT t.scene_group "
        "OR i.sha256 IS NOT t.image_sha256 OR t.raw_hash IS NULL OR t.oriented_hash IS NULL)").fetchone()[0]
    if missing:
        raise ValueError('Expanded geometry is outside the verified canonical RGB comparison')
    return sum(counts.values())


def run(root, output):
    parent_ref = file_ref(root / 'inventory.json')
    parent = checked_json(parent_ref)
    parent_index = dict(path=str(root / 'inventory.sqlite'), sha256=parent['inventory_sha256'])
    previous_ref = file_ref(root / 'reviewed-stage1-data-02/accounting/report.json')
    previous = checked_json(previous_ref)
    previous_plan = checked_json(previous['plan'])
    added_ref = file_ref(root / 'structured3d-source-reviews-01/report.json')
    added = checked_json(added_ref)
    isolation_ref = file_ref(root / 'reviewed-stage1-data-02/isolation-report.json')
    isolation = checked_json(isolation_ref)
    replay_ref = file_ref(root / 'reviewed-stage1-replay-02/report.json')
    replay = checked_json(replay_ref)
    if (previous['stage'] != 'accounting_verified_not_accepted'
            or previous['selected_images'] != 234036 or previous['unresolved'] != 0
            or previous['manifest_sha256'] != parent['manifest_sha256']
            or added['stage'] != 'structured3d_source_reviews_and_routes_complete_not_corpus_admission'
            or added['source_disposition_counts'] != dict(native_source_eligible=133563, excluded_geometry=1186)
            or isolation['stage'] != 'selected_sources_comparison_complete'
            or isolation['parent_inventory'] != parent_index
            or isolation['exact_pixel_overlap_pairs'] != 0 or isolation['forbidden_shared_scene_rows'] != 0
            or replay['stage'] != 'rgb_only_manifest_verified' or replay['isolation'] != isolation_ref
            or replay['all_rgb_identities_and_order_unchanged'] is not True
            or replay['images'] != 796477):
        raise ValueError('Completed source, previous accounting and canonical isolation evidence required')
    checked_path(isolation['heldout'])
    checked_path(isolation['rgb_policy'])
    checked_path(replay['replay'])
    expected = dict(previous['expected_counts'], structured3d=parent['dataset_counts']['structured3d'])
    selected_counts = dict(previous['selected_counts'], structured3d=133563)
    sources = list(previous_plan['sources'])
    for name, count in (('structured3d-recovered-01', 27008), ('structured3d-native-cache-02', 107741)):
        review = file_ref(root / 'structured3d-source-reviews-01' / (name + '-source-review.json'))
        if review not in added['source_reviews']:
            raise ValueError('Expanded source review is not in the completed source audit')
        sources.append(dict(dataset='structured3d', scope_count=count,
            generation_report=file_ref(root / name / 'report.json'),
            packs_index=file_ref(root / name / 'packs/packs.json'), review=review,
            payload_verification=MODE))
    output.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='expanded-reviewed-') as temporary:
        work = Path(temporary)
        shutil.copyfile(checked_path(parent_index), work / 'parent.sqlite')
        with sqlite3.connect(work / 'inventory.sqlite') as db:
            db.execute('ATTACH DATABASE ? AS original', (str(work / 'parent.sqlite'),))
            placeholders = ','.join('?' for _ in expected)
            db.execute(f'CREATE TABLE images AS SELECT * FROM original.images WHERE dataset IN ({placeholders})',
                       tuple(expected))
            if dict(db.execute('SELECT dataset,COUNT(*) FROM images GROUP BY dataset')) != expected:
                raise ValueError('Expanded inventory does not cover the reviewed source scope')
        publish_db(work / 'inventory.sqlite', output / 'inventory.sqlite')
        inventory_ref = file_ref(output / 'inventory.sqlite')
        publish_json(output / 'inventory.json', dict(parent, dataset_counts=expected,
            images=sum(expected.values()), inventory_sha256=inventory_ref['sha256'],
            parent_inventory=parent_index, parent_inventory_report=parent_ref,
            scope_kind='reviewed_source_subset'))
        publish_json(output / 'scope.json', dict(kind='reviewed_source_subset',
            manifest_sha256=parent['manifest_sha256'], expected_counts=expected,
            parent_inventory_report=parent_ref, previous_accounting=previous_ref,
            added_source_review=added_ref, payload_verification=MODE))
        exclusions_ref = added['known_geometry_exclusions']
        exclusions = [json.loads(line) for line in checked_path(exclusions_ref).read_text().splitlines()]
        exclusions = [row for row in exclusions if row['dataset'] in expected]
        if len(exclusions) != 1413:
            raise ValueError('Expanded source exclusions differ from the reviewed decisions')
        with (output / 'exclusions.jsonl').open('x') as file:
            for row in exclusions:
                file.write(json.dumps(row, sort_keys=True) + '\n')
        plan = dict(scope=file_ref(output / 'scope.json'), inventory=inventory_ref,
            inventory_report=file_ref(output / 'inventory.json'),
            exclusions=file_ref(output / 'exclusions.jsonl'), sources=sources)
        print(json.dumps(dict(stage='accounting_expanded_sources', sources=len(sources))), flush=True)
        accounting = build_accounting(plan, output / 'accounting')
        if accounting['selected_counts'] != selected_counts or accounting['selected_images'] != 367599:
            raise ValueError('Expanded reviewed selections differ')
        shutil.copyfile(checked_path(accounting['index']), work / 'accounting.sqlite')
        shutil.copyfile(checked_path(isolation['index']), work / 'isolation.sqlite')
        with sqlite3.connect(work / 'accounting.sqlite') as db:
            db.execute('ATTACH DATABASE ? AS isolated', (str(work / 'isolation.sqlite'),))
            isolated_count = verify_isolated_geometry(db, selected_counts)
            coverage = metadata_coverage(db, accounting=True)
            label_counts = dict(db.execute("SELECT label_kind,COUNT(*) FROM images "
                                           "WHERE disposition='selected' GROUP BY label_kind"))
        if label_counts != {'synthetic_renderer': 189025, 'teacher_pseudo': 178574}:
            raise ValueError('Expanded native versus pseudo routing differs')
        for ref in (parent_ref, previous_ref, added_ref, isolation_ref, replay_ref,
                    exclusions_ref, isolation['index'], replay['replay']):
            checked_path(ref)
        publish_json(output / 'isolation-report.json', dict(
            stage='expanded_selection_bound_to_completed_isolation', selected_geometry=isolated_count,
            accounting=file_ref(output / 'accounting/report.json'), previous_comparison=isolation_ref,
            unchanged_replay=replay_ref, replay=replay['replay'],
            exact_pixel_overlap_pairs=0, forbidden_shared_scene_rows=0,
            canonical_comparison_recomputed=False, original_full_census_complete_claimed=False,
            evaluation_depth_read=False, training_admission=False))
        publish_json(output / 'coverage-summary.json', dict(metadata=coverage,
            source_geometry_statistics=accounting['coverage'], label_counts=label_counts,
            statistics='Unchanged completed-producer support and quantiles, not a new pixel histogram',
            crop_support_review_complete=False, training_admission=False))
        report = dict(stage='reviewed_expansion_prepared_not_admitted', completed_at=time.time(),
            original_images=accounting['total_images'], selected_images=accounting['selected_images'],
            excluded_images=sum(accounting['excluded_counts'].values()),
            native_gt_images=label_counts['synthetic_renderer'], relative_pseudo_images=label_counts['teacher_pseudo'],
            accounting=file_ref(output / 'accounting/report.json'),
            isolation=file_ref(output / 'isolation-report.json'),
            coverage=file_ref(output / 'coverage-summary.json'), replay=replay['replay'],
            previous_accounting=previous_ref, added_source_review=added_ref,
            training_admission=False, optimizer_updates=0, actual_model_probe_complete=False,
            code=file_ref(Path(__file__)))
        publish_json(output / 'report.json', report)
        print(json.dumps(dict(stage=report['stage'], selected=report['selected_images'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
