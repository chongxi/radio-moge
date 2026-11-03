"""Reconcile completed ARKit native receipts and rejections, without admission.

This is a metadata audit after the producer's full payload readback. It does not
certify physical depth accuracy. Source flags and the existing FARO diagnostic
remain visible. Indexed updates use a local temporary SQLite database to avoid
one network-filesystem lookup per image.
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
from radio_moge.geometry_cache import HEX64, canonical_json, publish_json
from radio_moge.geometry_sources import normalized_opencv_intrinsics


def candidate_details(entry, policy_hash):
    meta, row = entry['geometry_metadata'], entry['image']
    turn = row['image_rotation_cw']
    shape = [256, 192] if turn in (90, 270) else [192, 256]
    q = meta['source_quality']
    quantiles = np.asarray(entry['depth_quantiles_m'], dtype=float)
    support = meta['valid_pixels'] / math.prod(shape)
    transform = np.asarray(meta['depth_to_rgb_uv_transform'], dtype=float)
    kr = normalized_opencv_intrinsics(*row['camera_width_height_fx_fy_cx_cy'])
    if (turn not in (0, 90, 180, 270) or meta['image_rotation_cw'] != turn
            or meta['dataset'] != 'arkitscenes' or meta['image_sha256'] != row['sha256']
            or meta['source_id'] != row['source_member'] or meta['policy_sha256'] != policy_hash
            or meta['label_kind'] != 'measured_sensor' or meta['camera_kind'] != 'calibrated_sensor'
            or meta['metric_calibrated'] is not True or meta['native_depth_not_resized'] is not True
            or meta['depth_convention'] != 'camera_z' or meta['units'] != 'meters'
            or meta['camera_pixel_convention'] != 'normalized_half_center'
            or meta['shape'] != shape or meta['rgb_shape'] != [row['height'], row['width']]
            or type(meta['source_needs_review']) is not bool
            or hashlib.sha256(canonical_json(meta)).hexdigest() != entry['key']
            or type(meta['valid_pixels']) is not int or not .05 <= support <= 1
            or not math.isclose(support, entry['valid_fraction'], rel_tol=0, abs_tol=1e-12)
            or not math.isclose(support, q['valid_fraction'], rel_tol=0, abs_tol=1e-12)
            or type(q['native_confident_stable_pixels']) is not int
            or not meta['valid_pixels'] <= q['native_confident_stable_pixels'] <= math.prod(shape)
            or q['image_registration_pass'] is not True
            or q['independent_physical_depth_accuracy_checked'] is not False
            or not 0 <= q['paired_rgb_mae'] <= .04 or not 0 <= q['paired_rgb_error_p90'] <= .08
            or quantiles.shape != (5,) or not np.isfinite(quantiles).all()
            or not (quantiles > 0).all() or not (np.diff(quantiles) >= 0).all()
            or quantiles[-1] > 65.535
            or transform.shape != (3, 3) or not np.isfinite(transform).all()
            or not np.allclose(transform[2], [0, 0, 1], rtol=0, atol=1e-12)
            or np.linalg.det(transform) <= 0
            or not np.allclose(meta['rgb_intrinsics_normalized'], kr, rtol=0, atol=1e-12)):
        raise ValueError('ARKit candidate provenance, native grid or registration statistics differ')
    stem = PurePosixPath(row['source_member']).stem
    base = row['source_url'].rsplit('/', 1)[0]
    if (not base.startswith('https://docs-assets.developer.apple.com/')
            or '/Training/' not in base or stem.split('_')[0] != row['video_id']
            or float(stem.split('_')[1]) != row['timestamp']):
        raise ValueError('Native source timestamp or publisher identity differs')
    proofs = meta['source_proofs']
    kinds = ('lowres_depth', 'confidence', 'lowres_wide', 'lowres_wide_intrinsics', 'vga_wide_intrinsics')
    if len(proofs) != len(kinds):
        raise ValueError('Five exact native member proofs required')
    for kind, proof in zip(kinds, proofs):
        suffix = '.pincam' if kind.endswith('intrinsics') else '.png'
        if (proof['source'] != base + '/' + kind + '.zip'
                or proof['member'] != f'{kind}/{stem}{suffix}'
                or not HEX64.fullmatch(proof['sha256']) or proof['bytes'] <= 0
                or type(proof['crc32']) is not int or not 0 <= proof['crc32'] < 2**32):
            raise ValueError('Exact native source proof differs')
    return dict(support=support, median_depth=float(quantiles[2]),
                rgb_mae=q['paired_rgb_mae'], rgb_p90=q['paired_rgb_error_p90'])


def rejection_kind(item):
    reason, stats = item['reason'], item['stats']
    if reason in ('missing_exact_timestamp_native_member', 'publisher_asset_not_available_404'):
        if stats.get('reason') != reason:
            raise ValueError('Missing-source rejection evidence differs')
        if reason == 'missing_exact_timestamp_native_member':
            stem = PurePosixPath(item['image']['source_member']).stem
            if PurePosixPath(stats.get('detail', '')).stem != stem:
                raise ValueError('Missing-source timestamp differs')
        return reason
    if reason != 'insufficient_registered_native_support' or not 0 <= stats['valid_fraction'] < .05:
        raise ValueError('Unexpected rejection or support above the unchanged minimum')
    if (type(stats['image_registration_pass']) is not bool
            or stats['independent_physical_depth_accuracy_checked'] is not False
            or not math.isfinite(stats['paired_rgb_mae']) or not math.isfinite(stats['paired_rgb_error_p90'])
            or (stats['image_registration_pass'] and not (
                0 <= stats['paired_rgb_mae'] <= .04 and 0 <= stats['paired_rgb_error_p90'] <= .08))):
        raise ValueError('Rejected registration statistics differ')
    return 'low_registered_support' if stats['image_registration_pass'] else 'image_registration_failed'


def bind(db, row, outcome, reason, evidence, entry_key=None, details=None):
    found = db.execute('SELECT ordinal,row_json,outcome FROM images WHERE image=?', (row['image'],)).fetchone()
    if found is None:
        raise ValueError('Outcome outside frozen ARKit scope')
    ordinal, original, previous = found
    candidate = dict(row)
    if (candidate.pop('corpus_ordinal', ordinal) != ordinal or candidate != json.loads(original)
            or previous is not None):
        raise ValueError('Changed or duplicate frozen ARKit outcome')
    db.execute('UPDATE images SET outcome=?,reason=?,evidence_json=?,entry_key=?,details_json=? WHERE ordinal=?',
        (outcome, reason, canonical_json(evidence).decode(), entry_key,
         canonical_json(details).decode() if details is not None else None, ordinal))


def run(config, native_attempt, attempt):
    root, expected = Path(config['root']), config['expected_counts']['arkitscenes']
    native = root / native_attempt
    refs = {key: file_ref(native / path) for key, path in dict(generation='report.json',
        packs='packs/packs.json', rejections='rejections.json', policy='packs/policy.json').items()}
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
            or policy['dataset'] != 'arkitscenes' or policy['manifest_sha256'] != config['manifest_sha256']
            or policy['minimum_image_valid_fraction'] != .05 or policy['native_holes_filled'] is not False
            or policy['photometric_gate'] != dict(pixel_mae_max=.08, image_mae_max=.04, image_p90_max=.08)):
        raise ValueError('Complete matching producer readback and unchanged native policy required')
    checked_path(inventory_ref)
    output = root / attempt
    output.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='arkit-outcomes-') as temporary:
        local = Path(temporary) / 'outcomes.sqlite'
        with sqlite3.connect(local) as db:
            db.execute('PRAGMA cache_size=-16384')
            db.execute('CREATE TABLE images(ordinal INTEGER PRIMARY KEY,image TEXT UNIQUE,sha256 TEXT,'
                'scene_group TEXT,video TEXT,rotation INTEGER,row_json TEXT,outcome TEXT,reason TEXT,'
                'evidence_json TEXT,entry_key TEXT UNIQUE,details_json TEXT)')
            with readonly_db(inventory_ref['path']) as inventory:
                for ordinal, image, checksum, group, encoded in inventory.execute(
                        "SELECT ordinal,image,sha256,scene_group,row_json FROM images WHERE dataset='arkitscenes' ORDER BY ordinal"):
                    row = dict(json.loads(encoded), image=image)
                    if (row['dataset'] != 'arkitscenes' or row['split'] != 'train'
                            or row['sha256'] != checksum or row['group'] != group
                            or row['image_rotation_cw'] not in (0, 90, 180, 270)):
                        raise ValueError('Frozen ARKit inventory identity differs')
                    db.execute('INSERT INTO images(ordinal,image,sha256,scene_group,video,rotation,row_json) '
                        'VALUES(?,?,?,?,?,?,?)', (ordinal, image, checksum, group, row['video_id'],
                         row['image_rotation_cw'], canonical_json(row).decode()))
            if db.execute('SELECT COUNT(*) FROM images').fetchone()[0] != expected:
                raise ValueError('Frozen ARKit count differs')
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
                    flagged = entry['geometry_metadata']['source_needs_review']
                    bind(db, entry['image'], 'flagged_candidate' if flagged else 'native_candidate',
                        'candidate_pending_source_quality_review', ref, entry['key'], details)
                db.commit()
            for item in rejections:
                bind(db, item['image'], 'rejected', rejection_kind(item), refs['rejections'])
            counts = dict(db.execute('SELECT outcome,COUNT(*) FROM images GROUP BY outcome'))
            videos = db.execute('SELECT COUNT(DISTINCT video) FROM images').fetchone()[0]
            if (None in counts or counts.get('rejected', 0) != len(rejections)
                    or counts.get('native_candidate', 0) + counts.get('flagged_candidate', 0) != packs['count']
                    or videos != generation['videos_seen']):
                raise ValueError('Full native outcomes do not reconcile')
            reasons = dict(db.execute("SELECT reason,COUNT(*) FROM images WHERE outcome='rejected' GROUP BY reason"))
            strata = [dict(group=g, rotation=turn, outcome=kind, images=n) for g, turn, kind, n in db.execute(
                'SELECT scene_group,rotation,outcome,COUNT(*) FROM images GROUP BY 1,2,3 ORDER BY 1,2,3')]
            review_rows = []
            for turn in (0, 90, 180, 270):
                for metric, order in [('support', 'ASC'), ('median_depth', 'DESC'), ('rgb_mae', 'DESC')]:
                    for ordinal, key, evidence, details in db.execute("SELECT ordinal,entry_key,evidence_json,details_json FROM images "
                            "WHERE outcome IN ('native_candidate','flagged_candidate') AND rotation=? ORDER BY "
                            f"json_extract(details_json,'$.{metric}') {order},ordinal LIMIT 3", (turn,)):
                        review_rows.append(dict(corpus_ordinal=ordinal, entry_key=key, receipt=json.loads(evidence),
                            metrics=json.loads(details), rotation=turn, selection_reason=f'{metric}_{order}'))
            db.commit()
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Outcome database integrity failure')
        for ref in [source_ref, *refs.values()]:
            checked_path(ref)
        target = output / 'outcomes.sqlite.incoming'
        shutil.copyfile(local, target)
        if file_ref(local)['sha256'] != file_ref(target)['sha256']:
            raise ValueError('Published outcome index differs from local audit')
        os.rename(target, output / 'outcomes.sqlite')
    publish_json(output / 'strata.json', strata)
    publish_json(output / 'review-candidates.json', review_rows)
    faro = root / 'arkit-faro-visuals-01/visual-review-01.json'
    report = dict(stage='native_outcomes_accounted_not_accepted', dataset='arkitscenes', completed_at=time.time(),
        expected_scope=expected, outcome_counts=counts, rejection_reasons=reasons, videos=videos,
        scene_groups=len({v['group'] for v in strata}), manifest_sha256=config['manifest_sha256'],
        policy_sha256=policy_hash, source_report=source_ref, inventory=inventory_ref,
        generation_report=refs['generation'], packs=refs['packs'], rejections=refs['rejections'], policy=refs['policy'],
        index=file_ref(output / 'outcomes.sqlite'), strata=file_ref(output / 'strata.json'),
        review_candidates=file_ref(output / 'review-candidates.json'),
        existing_faro_review=file_ref(faro) if faro.exists() else None,
        native_flags_preserved=True, faro_visibility_and_calibration_resolved=False,
        producer_array_readback_complete=True, independent_array_readback_performed=False,
        pack_payload_hashes_rechecked=False, receipt_metadata_only=True, source_quality_accepted=False,
        automatic_exclusions=0, training_admission=False, optimizer_updates=0,
        local_temporary_index=True, audit_code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--native-attempt', default='arkit-native-cache-01')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.native_attempt, args.attempt)):
        parser.error('Use new attempt basenames')
    result = run(json.loads(args.config.read_text()), args.native_attempt, args.attempt)
    print(json.dumps({k: result[k] for k in ('stage', 'outcome_counts', 'rejection_reasons')}), flush=True)
