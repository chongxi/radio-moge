"""Prepare the completed, reviewed four-source stage and verified RGB replay.

Original producer readbacks and quality reviews are reused explicitly. This
does not wait for unfinished sources or claim new pixel-range histograms.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from radio_moge.corpus_admission import (build_accounting, checked_json, checked_path,
    file_ref, readonly_db)
from radio_moge.corpus_coverage import metadata_coverage
from radio_moge.geometry_cache import publish_json
from radio_moge.producer_readback import MODE


SOURCES = (
    ('coco2017', 'coco-teacher-cache-02', 'coco-nyu-source-reviews-01/coco2017-source-review.json', 39997),
    ('hypersim', 'hypersim-native-cache-02', 'hypersim-native-source-review-02/native-source-review.json', 59344),
    ('hypersim', 'hypersim-review-teacher-cache-01', 'teacher-source-reviews-02/hypersim-source-review.json', 3681),
    ('irs', 'irs-teacher-cache-01', 'teacher-source-reviews-02/irs-source-review.json', 133879),
    ('nyuv2', 'nyuv2-teacher-cache-01', 'coco-nyu-source-reviews-01/nyuv2-source-review.json', 1043),
)


def publish_db(local, final):
    incoming = final.with_suffix('.sqlite.incoming')
    shutil.copyfile(local, incoming)
    os.rename(incoming, final)


def run(root, output):
    output.mkdir(exist_ok=False)
    parent_ref = file_ref(root / 'inventory.json')
    parent = checked_json(parent_ref)
    parent_index = dict(path=str(root / 'inventory.sqlite'), sha256=parent['inventory_sha256'])
    expected = {k: parent['dataset_counts'][k] for k in ('coco2017', 'hypersim', 'irs', 'nyuv2')}
    replay_expected = {k: v for k, v in parent['dataset_counts'].items() if k != 'scannetpp_dslr'}
    print(json.dumps(dict(stage='preparing_reviewed_subset', expected=expected)), flush=True)
    with tempfile.TemporaryDirectory(prefix='reviewed-stage-') as temporary:
        work = Path(temporary)
        shutil.copyfile(checked_path(parent_index), work / 'parent.sqlite')
        db = sqlite3.connect(work / 'inventory.sqlite')
        db.execute('ATTACH DATABASE ? AS original', (str(work / 'parent.sqlite'),))
        db.execute('CREATE TABLE images AS SELECT * FROM original.images WHERE dataset IN (?,?,?,?)',
                   tuple(expected))
        actual = dict(db.execute('SELECT dataset,COUNT(*) FROM images GROUP BY dataset'))
        if actual != expected:
            raise ValueError('Reviewed source inventory is incomplete')
        db.commit()
        db.close()
        publish_db(work / 'inventory.sqlite', output / 'inventory.sqlite')
        inventory_ref = file_ref(output / 'inventory.sqlite')
        publish_json(output / 'inventory.json', dict(parent, dataset_counts=expected,
            images=sum(expected.values()), inventory_sha256=inventory_ref['sha256'],
            parent_inventory=parent_index, parent_inventory_report=parent_ref,
            scope_kind='reviewed_source_subset'))
        publish_json(output / 'scope.json', dict(kind='reviewed_source_subset',
            manifest_sha256=parent['manifest_sha256'], expected_counts=expected,
            parent_inventory_report=parent_ref,
            authorization='User requested starting with existing reviewed data while other sources continue',
            payload_verification=MODE))
        exclusions_ref = file_ref(root / 'tartanair-rejection-review-02/known-geometry-exclusions.jsonl')
        exclusions = [json.loads(x) for x in checked_path(exclusions_ref).read_text().splitlines()]
        exclusions = [x for x in exclusions if x['dataset'] in expected]
        with (output / 'exclusions.jsonl').open('x') as file:
            for row in exclusions:
                file.write(json.dumps(row, sort_keys=True) + '\n')
        plan = dict(scope=file_ref(output / 'scope.json'), inventory=inventory_ref,
            inventory_report=file_ref(output / 'inventory.json'),
            exclusions=file_ref(output / 'exclusions.jsonl'), sources=[])
        for dataset, name, review, count in SOURCES:
            plan['sources'].append(dict(dataset=dataset, scope_count=count,
                generation_report=file_ref(root / name / 'report.json'),
                packs_index=file_ref(root / name / 'packs/packs.json'),
                review=file_ref(root / review), payload_verification=MODE))
        publish_json(output / 'accounting-plan.json', plan)
        print(json.dumps(dict(stage='accounting_completed_producers', sources=len(SOURCES))), flush=True)
        accounting = build_accounting(plan, output / 'accounting')
        if accounting['selected_images'] != 234036:
            raise ValueError('Reviewed subset differs from the expected eligible count')
        print(json.dumps(dict(stage='accounting_complete', selected=accounting['selected_images'])), flush=True)

        # Read one SQLite transaction of completed sources from the live census.
        # The unfinished ScanNet portion is neither selected nor reported complete.
        rgb_policy = file_ref(root / 'rgb-identity-census-01/policy.json')
        checked_json(rgb_policy)
        live_rgb = root / 'rgb-identity-census-01/rgb.sqlite.incoming'
        if not live_rgb.exists():
            live_rgb = root / 'rgb-identity-census-01/rgb.sqlite'
        rgb = sqlite3.connect(work / 'isolation.sqlite')
        rgb.execute('PRAGMA cache_size=-32768')
        rgb.execute('PRAGMA temp_store=FILE')
        rgb.execute('CREATE TABLE training(ordinal INTEGER PRIMARY KEY,dataset,scene_group,image_sha256,'
                    'raw_hash TEXT,oriented_hash TEXT)')
        counts = Counter()
        with readonly_db(live_rgb) as incoming:
            incoming.execute('BEGIN')
            for ordinal, dataset, group, checksum, raw, oriented, error in incoming.execute(
                    "SELECT ordinal,dataset,scene_group,image_sha256,raw_canonical_rgb_sha256,"
                    "canonical_rgb_sha256,error FROM images WHERE dataset!='scannetpp_dslr'"):
                if dataset not in replay_expected or error or not raw or not oriented:
                    raise ValueError('Completed source RGB is missing verified canonical identity')
                counts[dataset] += 1
                if ordinal != 210812:
                    rgb.execute('INSERT INTO training VALUES (?,?,?,?,?,?)',
                                (ordinal, dataset, group, checksum, raw, oriented))
        if dict(counts) != replay_expected:
            raise ValueError('Not all selected replay sources have finished RGB verification')
        rgb.execute('ATTACH DATABASE ? AS original', (str(work / 'parent.sqlite'),))
        missing = rgb.execute('SELECT COUNT(*) FROM training t LEFT JOIN original.images o '
            'ON t.ordinal=o.ordinal WHERE o.ordinal IS NULL OR t.dataset!=o.dataset '
            'OR t.scene_group!=o.scene_group OR t.image_sha256!=o.sha256').fetchone()[0]
        if missing:
            raise ValueError('Canonical RGB census differs from frozen source identities')
        replay = output / 'rgb-replay.jsonl'
        with replay.open('x') as file:
            for image, encoded in rgb.execute('SELECT o.image,o.row_json FROM training t '
                    'JOIN original.images o ON t.ordinal=o.ordinal ORDER BY t.ordinal'):
                row = json.loads(encoded)
                row['image'] = image
                file.write(json.dumps(row, sort_keys=True) + '\n')
        heldout_ref = file_ref(root / 'heldout-rgb-census-01/report.json')
        heldout = checked_json(heldout_ref)
        if (heldout['stage'] != 'heldout_rgb_census_complete_not_reviewed' or heldout['failed_images']
                or heldout['processed'] != 190954 or heldout['source_report'] != parent_ref):
            raise ValueError('Complete matching heldout RGB census required')
        shutil.copyfile(checked_path(heldout['index']), work / 'heldout.sqlite')
        rgb.execute('ATTACH DATABASE ? AS heldout', (str(work / 'heldout.sqlite'),))
        rgb.executescript('CREATE INDEX raw ON training(raw_hash); '
            'CREATE INDEX oriented ON training(oriented_hash); '
            'CREATE TABLE groups AS SELECT DISTINCT scene_group FROM training; '
            'CREATE UNIQUE INDEX group_key ON groups(scene_group); '
            'CREATE TABLE overlaps(ordinal INTEGER,heldout_id INTEGER,PRIMARY KEY(ordinal,heldout_id));')
        for own in ('raw_hash', 'oriented_hash'):
            for other in ('raw_canonical_rgb_sha256', 'canonical_rgb_sha256'):
                rgb.execute(f'INSERT OR IGNORE INTO overlaps SELECT t.ordinal,h.id FROM heldout.images h '
                            f'JOIN training t ON t.{own}=h.{other}')
        overlaps = rgb.execute('SELECT COUNT(*) FROM overlaps').fetchone()[0]
        forbidden = rgb.execute('SELECT COUNT(*) FROM heldout.images h JOIN groups g '
            'ON h.scene_group=g.scene_group WHERE h.scene_disjoint_required=1').fetchone()[0]
        rgb.commit()
        rgb.close()
        publish_db(work / 'isolation.sqlite', output / 'isolation.sqlite')
        publish_json(output / 'isolation-report.json', dict(stage='selected_sources_comparison_complete',
            source_counts=dict(counts), rgb_replay_images=sum(counts.values()) - 1,
            geometry_sources=expected, parent_inventory=parent_index, rgb_policy=rgb_policy,
            heldout=heldout_ref, heldout_memberships=190954, exact_pixel_overlap_pairs=overlaps,
            forbidden_shared_scene_rows=forbidden, rgb_excluded_ordinals=[210812],
            index=file_ref(output / 'isolation.sqlite'), replay=file_ref(replay),
            original_full_census_complete_claimed=False, evaluation_depth_read=False,
            near_duplicate_policy='Group-balanced sampling; original group/family holdout rules retained',
            training_admission=False))
        if overlaps or forbidden:
            raise ValueError('Selected-source/replay heldout overlap requires a disposition')
        with readonly_db(accounting['index']['path']) as incoming:
            coverage = metadata_coverage(incoming, accounting=True)
        publish_json(output / 'coverage-summary.json', dict(metadata=coverage,
            source_geometry_statistics=accounting['coverage'],
            statistics='Every retained receipt support and image-depth quantiles from completed producer readback',
            new_pixel_histogram_claimed=False, training_admission=False))
        publish_json(output / 'report.json', dict(stage='reviewed_subset_prepared_not_admitted',
            completed_at=time.time(), selected_images=accounting['selected_images'],
            native_gt_images=55462, relative_pseudo_images=178574,
            accounting=file_ref(output / 'accounting/report.json'),
            isolation=file_ref(output / 'isolation-report.json'),
            coverage=file_ref(output / 'coverage-summary.json'), replay=file_ref(replay),
            training_admission=False, optimizer_updates=0, actual_model_probe_complete=False,
            code=file_ref(Path(__file__))))
        print(json.dumps(dict(stage='reviewed_subset_prepared_not_admitted', selected=234036,
                              replay=sum(counts.values()) - 1, overlaps=overlaps)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
