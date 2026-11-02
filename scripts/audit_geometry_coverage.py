"""Describe full inventory/accounting coverage without issuing an acceptance gate."""
import argparse
import json
import os
from pathlib import Path
import time

from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.corpus_coverage import metadata_coverage, selected_geometry_coverage
from radio_moge.geometry_cache import publish_json


def run(mode, report_path, output):
    report_ref = file_ref(report_path)
    source = checked_json(report_ref)
    if mode == 'inventory':
        if source['stage'] != 'inventory_complete':
            raise ValueError('Complete frozen inventory is required')
        expected = source['dataset_counts']
        index_ref = dict(path=str(report_path.resolve().parent / 'inventory.sqlite'),
                         sha256=source['inventory_sha256'])
        total = source['images']
    else:
        if (source['stage'] != 'accounting_verified_not_accepted'
                or source['unresolved'] != 0 or source['training_admission'] is not False):
            raise ValueError('Complete verified accounting is required')
        expected, index_ref, total = source['expected_counts'], source['index'], source['total_images']
    if total != sum(expected.values()):
        raise ValueError('Scope counts are inconsistent')
    index = checked_path(index_ref)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        with readonly_db(index) as db:
            db.execute('PRAGMA temp_store=FILE')
            counts = dict(db.execute('SELECT dataset,COUNT(*) FROM images GROUP BY dataset'))
            if counts != expected:
                raise ValueError('Database differs from the complete declared scope')
            if mode == 'accounting':
                dispositions = dict(db.execute('SELECT disposition,COUNT(*) FROM images GROUP BY disposition'))
                if (set(dispositions) - {'selected', 'excluded'}
                        or dispositions.get('selected') != source['selected_images']):
                    raise ValueError('Unresolved or inconsistent accounting outcomes')
            temporary = output / 'groups.jsonl.incoming'
            with temporary.open('x') as file:
                def write_group(value):
                    file.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
                metadata = metadata_coverage(db, accounting=mode == 'accounting', write_group=write_group)
            geometry = selected_geometry_coverage(db) if mode == 'accounting' else None
        checked_path(index_ref)
        checked_path(report_ref)
        os.rename(temporary, output / 'groups.jsonl')
        result = dict(stage='coverage_audit_complete_not_reviewed', mode=mode,
            started_at=started, completed_at=time.time(), training_admission=False, optimizer_updates=0,
            source_report=report_ref, index=index_ref, manifest_sha256=source['manifest_sha256'],
            scope_counts=counts, scope_images=total, metadata=metadata, selected_geometry=geometry,
            histogram_intervals='lower-inclusive, upper-exclusive; null last upper edge means unbounded',
            independent_physical_accuracy_checked=False,
            outstanding_review=['source-specific group independence', 'indoor/outdoor content',
                'perceptual near-duplicate sampling', 'post-augmentation support and loss participation',
                'training source/scene balance and diversity after exclusions'],
            evidence=[file_ref(output / 'groups.jsonl'), file_ref(Path(__file__)),
                file_ref(Path(__file__).parents[1] / 'src/radio_moge/corpus_coverage.py')])
        publish_json(output / 'report.json', result)
        print(json.dumps(dict(stage=result['stage'], mode=mode, scope_images=total,
                              training_admission=False)), flush=True)
        return result
    except BaseException as error:
        publish_json(output / 'failure.json', dict(stage='failed', training_admission=False,
            error=type(error).__name__, message=str(error)[:1000]))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inventory', 'accounting'))
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.mode, args.report, args.output)
