"""Add only the reviewed 25-scene teacher subset to the completed training stage.

Accounting can run while the whole RGB census finishes. A separate isolation
step requires its completed canonical comparison and binds every new stage
identity. Neither step opens the optimizer gate.
"""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from prepare_reviewed_training import publish_db
from radio_moge.corpus_admission import build_accounting, checked_json, checked_path, file_ref
from radio_moge.corpus_coverage import metadata_coverage
from radio_moge.geometry_cache import publish_json
from radio_moge.producer_readback import MODE


def add_exact_scan_scope(db, expected_scenes):
    """Attached original and routed DBs are immutable SHA-verified copies."""
    if (db.execute("SELECT COUNT(*) FROM routed.images").fetchone()[0] != 35783
            or db.execute("SELECT COUNT(*) FROM routed.images WHERE outcome IS NOT 'teacher_candidate'").fetchone()[0]
            or db.execute('SELECT COUNT(DISTINCT scene_group) FROM routed.images').fetchone()[0] != expected_scenes):
        raise ValueError('Exact reviewed ScanNet scope is required')
    missing = db.execute("SELECT COUNT(*) FROM routed.images s LEFT JOIN original.images i ON i.ordinal=s.ordinal "
        "WHERE i.ordinal IS NULL OR i.dataset IS NOT 'scannetpp_dslr' OR i.image IS NOT s.image "
        "OR i.sha256 IS NOT s.sha256 OR i.scene_group IS NOT s.scene_group").fetchone()[0]
    if missing:
        raise ValueError('Routed ScanNet RGB differs from the frozen inventory')
    db.execute('INSERT INTO images SELECT i.* FROM original.images i JOIN routed.images s ON s.ordinal=i.ordinal')


def run(root, output):
    parent_ref = file_ref(root / 'inventory.json')
    parent = checked_json(parent_ref)
    parent_index = dict(path=str(root / 'inventory.sqlite'), sha256=parent['inventory_sha256'])
    previous_ref = file_ref(root / 'reviewed-stage2-data-01/accounting/report.json')
    previous = checked_json(previous_ref)
    previous_plan = checked_json(previous['plan'])
    added_ref = file_ref(root / 'scannetpp-teacher-source-reviews-01/report.json')
    added = checked_json(added_ref)
    audit = checked_json(added['source_outcomes'])
    if (previous['stage'] != 'accounting_verified_not_accepted' or previous['selected_images'] != 367599
            or previous['total_images'] != 369012 or previous['unresolved'] != 0
            or previous['manifest_sha256'] != parent['manifest_sha256']
            or added['stage'] != 'scannetpp_teacher_source_reviews_complete_not_admitted'
            or added['source_eligible_images'] != 35783 or added['source_eligible_scenes'] != 25
            or added['additional_geometry_exclusions'] != 0 or added['native_source_flags_cleared'] is not False
            or 'scannetpp_dslr' in previous['expected_counts']):
        raise ValueError('Completed unchanged previous stage and exact teacher source review required')
    expected = dict(previous['expected_counts'], scannetpp_dslr=35783)
    selected_counts = dict(previous['selected_counts'], scannetpp_dslr=35783)
    sources = list(previous_plan['sources'])
    for source in audit['sources']:
        review = file_ref(root / 'scannetpp-teacher-source-reviews-01' / (source['name'] + '-source-review.json'))
        if review not in added['source_reviews']:
            raise ValueError('Source policy review is outside the completed source publication')
        sources.append(dict(dataset='scannetpp_dslr', scope_count=audit['candidate_policy_counts'][source['policy_sha256']],
            generation_report=source['generation'], packs_index=source['packs'], review=review,
            payload_verification=MODE))
    output.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='reviewed-scan-stage-') as temporary:
        work = Path(temporary)
        for name, ref in (('parent', parent_index), ('routed', audit['index'])):
            shutil.copyfile(checked_path(ref), work / (name + '.sqlite'))
            if file_ref(work / (name + '.sqlite'))['sha256'] != ref['sha256']:
                raise ValueError('Immutable inventory copy differs')
        with sqlite3.connect(work / 'inventory.sqlite') as db:
            db.execute('ATTACH DATABASE ? AS original', (str(work / 'parent.sqlite'),))
            db.execute('ATTACH DATABASE ? AS routed', (str(work / 'routed.sqlite'),))
            names = tuple(previous['expected_counts'])
            db.execute('CREATE TABLE images AS SELECT * FROM original.images WHERE dataset IN ('
                       + ','.join('?' for _ in names) + ')', names)
            add_exact_scan_scope(db, 25)
            if dict(db.execute('SELECT dataset,COUNT(*) FROM images GROUP BY dataset')) != expected:
                raise ValueError('Expanded inventory counts differ')
            db.execute('CREATE UNIQUE INDEX by_ordinal ON images(ordinal)')
            db.execute('CREATE UNIQUE INDEX by_image ON images(image)')
        publish_db(work / 'inventory.sqlite', output / 'inventory.sqlite')
        inventory_ref = file_ref(output / 'inventory.sqlite')
        publish_json(output / 'inventory.json', dict(parent, dataset_counts=expected, images=sum(expected.values()),
            inventory_sha256=inventory_ref['sha256'], parent_inventory=parent_index,
            parent_inventory_report=parent_ref, scope_kind='reviewed_source_subset'))
        publish_json(output / 'scope.json', dict(kind='reviewed_source_subset', expected_counts=expected,
            manifest_sha256=parent['manifest_sha256'], parent_inventory_report=parent_ref,
            previous_accounting=previous_ref, added_source_review=added_ref, exact_teacher_outcomes=added['source_outcomes'],
            payload_verification=MODE))
        exclusions_ref = previous_plan['exclusions']
        exclusions = checked_path(exclusions_ref).read_bytes()
        if len(exclusions.splitlines()) != 1413:
            raise ValueError('Previous stage exclusion scope changed')
        incoming = output / 'exclusions.jsonl.incoming'
        incoming.write_bytes(exclusions)
        incoming.rename(output / 'exclusions.jsonl')
        plan = dict(scope=file_ref(output / 'scope.json'), inventory=inventory_ref,
            inventory_report=file_ref(output / 'inventory.json'),
            exclusions=file_ref(output / 'exclusions.jsonl'), sources=sources)
        print(json.dumps(dict(stage='accounting_six_sources', original_images=404795)), flush=True)
        accounting = build_accounting(plan, output / 'accounting')
        if (accounting['selected_counts'] != selected_counts or accounting['selected_images'] != 403382
                or sum(accounting['excluded_counts'].values()) != 1413):
            raise ValueError('Expanded selected identities differ from reviewed routes')
        shutil.copyfile(checked_path(accounting['index']), work / 'accounting.sqlite')
        with sqlite3.connect(work / 'accounting.sqlite') as db:
            coverage = metadata_coverage(db, accounting=True)
            labels = dict(db.execute("SELECT label_kind,COUNT(*) FROM images WHERE disposition='selected' GROUP BY label_kind"))
        if labels != {'synthetic_renderer': 189025, 'teacher_pseudo': 214357}:
            raise ValueError('Native and relative teacher label counts differ')
    for ref in (parent_ref, previous_ref, added_ref, added['source_outcomes'], exclusions_ref):
        checked_path(ref)
    publish_json(output / 'coverage-summary.json', dict(metadata=coverage,
        source_geometry_statistics=accounting['coverage'], label_counts=labels,
        statistics='Unchanged producer support and quantiles; not a fresh pixel histogram', training_admission=False))
    report = dict(stage='reviewed_expansion_accounted_isolation_pending', completed_at=time.time(),
        original_images=404795, selected_images=403382, excluded_images=1413,
        native_gt_images=189025, relative_pseudo_images=214357,
        accounting=file_ref(output / 'accounting/report.json'), coverage=file_ref(output / 'coverage-summary.json'),
        previous_accounting=previous_ref, added_source_review=added_ref, training_admission=False,
        optimizer_updates=0, actual_model_probe_complete=False, code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(stage=report['stage'], selected=403382)), flush=True)


def verify_canonical_scope(db, expected):
    """Compare all scoped RGB, including excluded geometry, to the full census."""
    if dict(db.execute('SELECT dataset,COUNT(*) FROM images GROUP BY dataset')) != expected:
        raise ValueError('Stage identity counts differ')
    missing = db.execute('SELECT COUNT(*) FROM images i LEFT JOIN rgb.images r ON r.ordinal=i.ordinal '
        'WHERE r.ordinal IS NULL OR r.dataset IS NOT i.dataset OR r.scene_group IS NOT i.scene_group '
        'OR r.image_sha256 IS NOT i.sha256 OR r.raw_canonical_rgb_sha256 IS NULL '
        'OR r.canonical_rgb_sha256 IS NULL OR r.error IS NOT NULL').fetchone()[0]
    if missing:
        raise ValueError('Stage RGB is outside the complete canonical census')
    exact = db.execute('SELECT COUNT(*) FROM images i JOIN compared.overlaps o ON o.train_ordinal=i.ordinal').fetchone()[0]
    groups = db.execute('SELECT COUNT(*) FROM heldout.images h WHERE h.scene_disjoint_required=1 '
        'AND EXISTS(SELECT 1 FROM images i WHERE i.scene_group=h.scene_group)').fetchone()[0]
    if exact or groups:
        raise ValueError(f'Stage train/heldout overlap: {exact} exact pairs, {groups} forbidden scene rows')
    return sum(expected.values())


def isolate(root, data, comparison_path, expected_selected=403382):
    accounting_ref = file_ref(data / 'accounting/report.json')
    accounting = checked_json(accounting_ref)
    comparison_ref = file_ref(comparison_path)
    comparison = checked_json(comparison_ref)
    if (comparison['stage'] != 'heldout_exact_comparison_complete_not_reviewed'
            or comparison['training_rows'] != 1637909 or comparison['heldout_rows'] != 190954
            or accounting['selected_images'] != expected_selected or accounting['unresolved'] != 0):
        raise ValueError('Complete full RGB comparison and new stage accounting required')
    train, heldout = (checked_json(r) for r in comparison['reports'])
    scope = checked_json(accounting['scope'])
    if (train['source_report'] != scope['parent_inventory_report']
            or train['source_report'] != heldout['source_report']
            or train['failed_images'] != 0 or heldout['failed_images'] != 0):
        raise ValueError('Canonical comparison is bound to another frozen corpus')
    replay_ref = file_ref(root / 'reviewed-stage1-replay-02/report.json')
    replay = checked_json(replay_ref)
    previous_isolation = checked_json(replay['isolation'])
    if (replay['stage'] != 'rgb_only_manifest_verified' or replay['images'] != 796477
            or replay['all_rgb_identities_and_order_unchanged'] is not True
            or previous_isolation['stage'] != 'selected_sources_comparison_complete'
            or previous_isolation['exact_pixel_overlap_pairs'] != 0
            or previous_isolation['forbidden_shared_scene_rows'] != 0):
        raise ValueError('Unchanged replay canonical isolation proof required')
    checked_path(replay['replay'])
    with tempfile.TemporaryDirectory(prefix='stage-canonical-isolation-') as temporary:
        work = Path(temporary)
        for name, ref in (('stage', accounting['index']), ('rgb', train['index']),
                          ('heldout', heldout['index']), ('compared', comparison['index'])):
            shutil.copyfile(checked_path(ref), work / (name + '.sqlite'))
            if file_ref(work / (name + '.sqlite'))['sha256'] != ref['sha256']:
                raise ValueError('Canonical comparison copy differs')
        with sqlite3.connect(work / 'stage.sqlite') as db:
            for name in ('rgb', 'heldout', 'compared'):
                db.execute('ATTACH DATABASE ? AS ' + name, (str(work / (name + '.sqlite')),))
            count = verify_canonical_scope(db, accounting['expected_counts'])
    for ref in (accounting_ref, comparison_ref, replay_ref, replay['isolation'], *comparison['reports']):
        checked_path(ref)
    publish_json(data / 'isolation-report.json', dict(stage='expanded_selection_bound_to_full_canonical_comparison',
        original_geometry_scope=count, selected_geometry=accounting['selected_images'], accounting=accounting_ref,
        full_comparison=comparison_ref, exact_pixel_overlap_pairs=0, forbidden_shared_scene_rows=0,
        full_scope_overlap_rows=comparison['exact_pixel_overlap_rows'],
        full_scope_forbidden_scene_rows=comparison['forbidden_shared_scene_rows'],
        unchanged_replay=replay_ref, replay=replay['replay'], replay_original_comparison=replay['isolation'],
        replay_comparison_reused=True, evaluation_depth_read=False, automatic_exclusions=0,
        training_admission=False, optimizer_updates=0, completed_at=time.time(), code=file_ref(Path(__file__))))
    print(json.dumps(dict(stage='expanded_isolation_complete', original_geometry_scope=count)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['account', 'isolate'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--comparison', type=Path)
    args = parser.parse_args()
    if args.mode == 'account':
        run(args.root.resolve(), args.output.resolve())
    else:
        if not args.comparison:
            parser.error('Completed full canonical comparison is required')
        isolate(args.root.resolve(), args.output.resolve(), args.comparison.resolve())
