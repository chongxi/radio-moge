"""Reconcile completed TartanAir native outcomes without accepting labels.

Receipt metadata is checked after producer payload readback. No payload is
regenerated, no rejection is automatically approved, and stereo consistency
does not establish independent physical accuracy. Indexed updates stay local.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile
import time

import numpy as np

from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import HEX64, canonical_json, publish_json, validate_member_reference


CONVERSION = 'native V1 camera Z float meters, original RGB grid'
CALIBRATION = dict(width=640, height=480, fx=320, fy=320, cx=320, cy=240, baseline_m=.25)
STEREO_SUPPORT = 'both eyes 3x3 stable (.05+.02Z m), x_right=x_left-80/Z, no y shift'


def source_identity(row, proofs):
    parts = PurePosixPath(row['source_member']).parts
    if (row['dataset'] != 'tartanair' or row['split'] != 'train'
            or (row['width'], row['height']) != (640, 480) or row.get('image_rotation_cw', 0)
            or len(parts) != 5 or parts[0] != row['environment'] or parts[1] not in ('Easy', 'Hard')
            or parts[3] != 'image_left' or not parts[4].endswith('_left.png')
            or row['trajectory'] != '/'.join(parts[:3])
            or row['group'] != 'tartanair:environment:' + row['environment']
            or '..' in parts or PurePosixPath(row['source_member']).is_absolute()
            or not row['source_url'].startswith('https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/')
            or not row['source_url'].endswith('/tartanair/' + '/'.join(parts[:2]) + '/image_left.zip')):
        raise ValueError('Frozen TartanAir left-eye source identity differs')
    suffixes = dict(depth_left='_left_depth.npy', depth_right='_right_depth.npy', image_right='_right.png')
    if len(proofs) != 3:
        raise ValueError('Three exact stereo source proofs required')
    for (kind, suffix), proof in zip(suffixes.items(), proofs):
        member = '/'.join((*parts[:3], kind, parts[4].removesuffix('_left.png') + suffix))
        if (proof['source'] != row['source_url'].rsplit('/', 1)[0] + '/' + kind + '.zip'
                or proof['member'] != member or not HEX64.fullmatch(proof['sha256'])
                or type(proof['bytes']) is not int or proof['bytes'] <= 0
                or type(proof['crc32']) is not int or not 0 <= proof['crc32'] < 2**32):
            raise ValueError('Exact corresponding-eye source proof differs')
    return parts[1]


def quality_details(q):
    names = ('native_finite_positive_fraction', 'task_range_fraction', 'stable_left_fraction',
             'stereo_geometric_fraction', 'valid_fraction')
    fractions = [q[n] for n in names]
    if (not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and 0 <= v <= 1 for v in fractions)
            or any(b > a + 1e-12 for a, b in zip(fractions, fractions[1:]))
            or q['task_range_m'] != [.1, 200.]
            or q['task_range_is_not_publisher_invalid_mask'] is not True
            or q['independent_physical_depth_accuracy_checked'] is not False
            or q['source_needs_review'] is not (fractions[-1] < .05)):
        raise ValueError('Native support nesting, range or source-quality flag differs')
    for value, present, maximum in ((q['rgb_stereo_mae_on_geometric'], fractions[-2] > 0, 1.),
                                   (q['stereo_depth_max_error_m'], fractions[-1] > 0, 1.010001)):
        if ((present and (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or not 0 <= value <= maximum))
                or (not present and value is not None)):
            raise ValueError('Stereo diagnostic statistic or empty-support marker differs')
    return dict(support=fractions[-1], task_support=fractions[1],
                rgb_mae=q['rgb_stereo_mae_on_geometric'], max_stereo_error=q['stereo_depth_max_error_m'])


def candidate_details(entry, policy_hash):
    meta, row = entry['geometry_metadata'], entry['image']
    difficulty = source_identity(row, meta['source_proofs'])
    details = quality_details(meta['source_quality'])
    quantiles = np.asarray(entry['depth_quantiles_m'], dtype=float)
    if (meta['dataset'] != 'tartanair' or meta['image_sha256'] != row['sha256']
            or meta['source_id'] != row['source_member'] or meta['policy_sha256'] != policy_hash
            or meta['label_kind'] != 'synthetic_renderer' or meta['camera_kind'] != 'synthetic_camera'
            or meta['metric_calibrated'] is not True or meta['conversion'] != CONVERSION
            or meta['shape'] != [480, 640] or meta['depth_convention'] != 'camera_z'
            or meta['units'] != 'meters' or meta['camera_pixel_convention'] != 'normalized_half_center'
            or meta['source_needs_review'] is not False or not .05 <= details['support'] <= 1
            or type(meta['valid_pixels']) is not int
            or not math.isclose(meta['valid_pixels'] / (480 * 640), details['support'], rel_tol=0, abs_tol=1e-12)
            or not math.isclose(entry['valid_fraction'], details['support'], rel_tol=0, abs_tol=1e-12)
            or hashlib.sha256(canonical_json(meta)).hexdigest() != entry['key']
            or quantiles.shape != (5,) or not np.isfinite(quantiles).all()
            or not (np.diff(quantiles) >= 0).all() or quantiles[0] < .1 - 1e-7 or quantiles[-1] > 200.
            or not HEX64.fullmatch(entry['sha256'])):
        raise ValueError('Native TartanAir candidate provenance, units, grid or support differs')
    validate_member_reference(entry)
    return dict(details, difficulty=difficulty, median_depth=float(quantiles[2]),
                p99_depth=float(quantiles[3]))


def rejection_details(record):
    difficulty = source_identity(record['image'], record['source_proofs'])
    details = quality_details(record['stats'])
    if record['reason'] != 'insufficient_stereo_verified_support' or details['support'] >= .05:
        raise ValueError('Rejection differs from unchanged minimum support')
    return dict(details, difficulty=difficulty)


def bind(db, row, outcome, reason, evidence, details, entry_key=None):
    found = db.execute('SELECT ordinal,row_json,outcome FROM images WHERE image=?', (row['image'],)).fetchone()
    if found is None:
        raise ValueError('Outcome outside frozen TartanAir scope')
    ordinal, original, previous = found
    candidate = dict(row)
    if (candidate.pop('corpus_ordinal', ordinal) != ordinal or candidate != json.loads(original)
            or previous is not None):
        raise ValueError('Changed or duplicate frozen TartanAir outcome')
    db.execute('UPDATE images SET outcome=?,reason=?,evidence_json=?,entry_key=?,details_json=? WHERE ordinal=?',
        (outcome, reason, canonical_json(evidence).decode(), entry_key, canonical_json(details).decode(), ordinal))


def run(config, native_attempt, attempt):
    root, expected = Path(config['root']), config['expected_counts']['tartanair']
    native = root / native_attempt
    refs = {k: file_ref(native / p) for k, p in dict(generation='report.json', packs='packs/packs.json',
        rejections='rejections.json', policy='packs/policy.json').items()}
    generation, packs, rejections, policy = [checked_json(refs[k]) for k in refs]
    source_ref = file_ref(root / 'inventory.json')
    source = checked_json(source_ref)
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if (source['stage'] != 'inventory_complete' or source['manifest_sha256'] != config['manifest_sha256']
            or source['dataset_counts'] != config['expected_counts']
            or generation['stage'] != 'complete' or generation['mode'] != 'generate'
            or generation['training_admission'] is not False or generation['training_started'] is not False
            or generation['optimizer_updates'] != 0 or generation['verified_cache_members'] != generation['completed']
            or generation['completed'] + generation['rejected'] != expected or generation['expected_scope'] != expected
            or len(rejections) != generation['rejected'] or packs['count'] != generation['completed']
            or packs['policy_sha256'] != policy_hash or generation['policy_sha256'] != policy_hash
            or policy['dataset'] != 'tartanair' or policy['manifest_sha256'] != config['manifest_sha256']
            or policy['label_kind'] != 'synthetic_renderer' or policy['conversion'] != CONVERSION
            or policy['calibration'] != CALIBRATION or policy['task_range_m'] != [.1, 200.]
            or policy['task_range_is_not_publisher_invalid_mask'] is not True
            or policy['minimum_image_valid_fraction'] != .05 or policy['native_holes_filled'] is not False
            or policy['independent_physical_depth_accuracy_checked'] is not False
            or policy['stereo_support'] != STEREO_SUPPORT
            or policy['stereo_gates'] != dict(z_difference_m='.01+.005Z', rgb_pixel_mae_max=.08)):
        raise ValueError('Complete producer readback and unchanged native stereo policy required')
    checked_path(inventory_ref)
    output = root / attempt
    output.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='tartanair-outcomes-') as temporary:
        local = Path(temporary) / 'outcomes.sqlite'
        with sqlite3.connect(local) as db:
            db.execute('PRAGMA cache_size=-16384')
            db.execute('CREATE TABLE images(ordinal INTEGER PRIMARY KEY,image TEXT UNIQUE,sha256 TEXT,'
                'scene_group TEXT,environment TEXT,difficulty TEXT,trajectory TEXT,source_url TEXT,row_json TEXT,'
                'outcome TEXT,reason TEXT,evidence_json TEXT,entry_key TEXT UNIQUE,details_json TEXT)')
            with readonly_db(inventory_ref['path']) as inventory:
                for ordinal, image, checksum, group, encoded in inventory.execute(
                        "SELECT ordinal,image,sha256,scene_group,row_json FROM images WHERE dataset='tartanair' ORDER BY ordinal"):
                    row = dict(json.loads(encoded), image=image)
                    parts = PurePosixPath(row['source_member']).parts
                    if (row['dataset'] != 'tartanair' or row['split'] != 'train' or row['sha256'] != checksum
                            or row['group'] != group or len(parts) != 5 or parts[1] not in ('Easy', 'Hard')):
                        raise ValueError('Frozen TartanAir inventory identity differs')
                    db.execute('INSERT INTO images(ordinal,image,sha256,scene_group,environment,difficulty,'
                        'trajectory,source_url,row_json) VALUES(?,?,?,?,?,?,?,?,?)',
                        (ordinal, image, checksum, group, row['environment'], parts[1], row['trajectory'],
                         row['source_url'], canonical_json(row).decode()))
            if db.execute('SELECT COUNT(*) FROM images').fetchone()[0] != expected:
                raise ValueError('Frozen TartanAir count differs')
            for packet in packs['packs']:
                path = native / 'packs' / packet['receipt']
                if path.parent != native / 'packs':
                    raise ValueError('Invalid receipt path')
                ref = dict(path=str(path), sha256=packet['sha256'])
                receipt = checked_json(ref)
                tar = path.parent / receipt['tar']
                if (tar.parent != path.parent or tar.stat().st_size != receipt['bytes']
                        or receipt['policy_sha256'] != policy_hash or len(receipt['entries']) != packet['count']):
                    raise ValueError('Receipt policy, size or count differs')
                for entry in receipt['entries']:
                    details = candidate_details(entry, policy_hash)
                    if entry['offset'] + entry['size'] > receipt['bytes']:
                        raise ValueError('Member exceeds published pack bounds')
                    bind(db, entry['image'], 'native_candidate', 'candidate_pending_source_quality_review',
                         ref, details, entry['key'])
                db.commit()
            rejection_refs = []
            for record in rejections:
                details = rejection_details(record)
                ref = file_ref(native / 'rejected' / (record['image']['sha256'] + '.json'))
                if checked_json(ref) != record:
                    raise ValueError('Individual persisted rejection differs from final aggregate')
                bind(db, record['image'], 'rejected', record['reason'], ref, details)
                rejection_refs.append(ref)
            counts = dict(db.execute('SELECT outcome,COUNT(*) FROM images GROUP BY outcome'))
            bundles, environments, trajectories = db.execute('SELECT COUNT(DISTINCT source_url),'
                'COUNT(DISTINCT environment),COUNT(DISTINCT trajectory) FROM images').fetchone()
            if (None in counts or counts.get('rejected', 0) != len(rejections)
                    or counts.get('native_candidate', 0) != packs['count']
                    or bundles != generation['source_bundles_seen']
                    or environments != generation['environment_names_seen']):
                raise ValueError('Full native outcomes do not reconcile')
            strata = [dict(environment=e, difficulty=d, outcome=k, images=n, trajectories=t)
                for e, d, k, n, t in db.execute('SELECT environment,difficulty,outcome,COUNT(*),'
                    'COUNT(DISTINCT trajectory) FROM images GROUP BY 1,2,3 ORDER BY 1,2,3')]
            selected = {}
            for environment, difficulty in sorted({(s['environment'], s['difficulty']) for s in strata}):
                for metric, order, limit in [('support', 'ASC', 2), ('median_depth', 'ASC', 1),
                                             ('median_depth', 'DESC', 1), ('rgb_mae', 'DESC', 1)]:
                    for ordinal, key, evidence, details in db.execute('SELECT ordinal,entry_key,evidence_json,details_json '
                            "FROM images WHERE outcome='native_candidate' AND environment=? AND difficulty=? ORDER BY "
                            f"json_extract(details_json,'$.{metric}') {order},ordinal LIMIT ?", (environment, difficulty, limit)):
                        item = selected.setdefault(ordinal, dict(corpus_ordinal=ordinal, entry_key=key,
                            receipt=json.loads(evidence), metrics=json.loads(details), environment=environment,
                            difficulty=difficulty, selection_reasons=[]))
                        item['selection_reasons'].append(f'{metric}_{order}')
            db.commit()
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Outcome database integrity failure')
        for ref in [source_ref, *refs.values(), *rejection_refs]:
            checked_path(ref)
        target = output / 'outcomes.sqlite.incoming'
        shutil.copyfile(local, target)
        if file_ref(local)['sha256'] != file_ref(target)['sha256']:
            raise ValueError('Published outcome index differs from local audit')
        os.rename(target, output / 'outcomes.sqlite')
    publish_json(output / 'strata.json', strata)
    publish_json(output / 'review-candidates.json', [selected[k] for k in sorted(selected)])
    report = dict(stage='native_outcomes_accounted_not_accepted', dataset='tartanair', completed_at=time.time(),
        expected_scope=expected, outcome_counts=counts, source_bundles=bundles, environments=environments,
        trajectories=trajectories, trajectory_count_is_not_independent_scene_count=True,
        manifest_sha256=config['manifest_sha256'], policy_sha256=policy_hash, source_report=source_ref,
        inventory=inventory_ref, generation_report=refs['generation'], packs=refs['packs'],
        rejections=refs['rejections'], policy=refs['policy'], index=file_ref(output / 'outcomes.sqlite'),
        strata=file_ref(output / 'strata.json'), review_candidates=file_ref(output / 'review-candidates.json'),
        review_candidates_count=len(selected), selection_is_not_random_prevalence=True,
        native_flags_preserved=True, independent_physical_depth_accuracy_checked=False,
        producer_array_readback_complete=True, independent_array_readback_performed=False,
        pack_payload_hashes_rechecked=False, receipt_metadata_only=True, source_quality_accepted=False,
        automatic_exclusions=0, training_admission=False, optimizer_updates=0,
        local_temporary_index=True, audit_code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--native-attempt', default='tartanair-native-cache-01')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.native_attempt, args.attempt)):
        parser.error('Use attempt basenames')
    result = run(json.loads(args.config.read_text()), args.native_attempt, args.attempt)
    print(json.dumps({k: result[k] for k in ('stage', 'expected_scope', 'outcome_counts', 'review_candidates_count')}))
