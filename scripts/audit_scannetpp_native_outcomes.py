"""Account native ScanNet++ outcomes once; no source acceptance or new inference.

Completed recovery and producer readback are prerequisites. Receipt metadata,
source-audit identities and every frozen RGB route are checked in a local SQLite
index. Teacher replacements remain separate and never inflate the native count.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

import numpy as np

from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import HEX64, canonical_json, publish_json, validate_member_reference
from radio_moge.geometry_sources import normalized_opencv_intrinsics


CONVERSION = 'CPU nearest-hit camera Z meters on 640-wide pinhole ray grid, no fitting/filling'
VISIBILITY = '5x5 stable (.02+.02Z m); four quarter-pixel rays (.005+.01Z m)'
LARGE_MESH = 'all faces in <=1048576-face passes, nearest-Z merge; full CPU parity unchanged'


def row_shape(row, data):
    scene = Path(data)/'scannetpp-v2-full/data'/row['scene']
    if (row['dataset'] != 'scannetpp_dslr' or row['split'] != 'train'
            or row['official_split'] != 'nvs_sem_train' or row['group'] != 'scannetpp:'+row['scene']
            or Path(row['scene']).name != row['scene'] or Path(row['view']).name != row['view']
            or row['scene'] in ('.', '..') or row['view'] in ('.', '..')
            or row['image'] != str(scene/'dslr/resized_undistorted_images'/row['view'])
            # Publisher calibration can name a mask differently from the RGB stem.
            # Its exact basename is frozen in the inventory and checked by bind().
            or Path(row['mask']).parent != scene/'dslr/resized_undistorted_masks'
            or Path(row['mask']).name in ('', '.', '..')
            or row.get('image_rotation_cw', 0) != 0
            or row['camera_width_height_fx_fy_cx_cy'][:2] != [row['width'], row['height']]
            or min(row['width'], row['height']) <= 0):
        raise ValueError('Frozen undistorted DSLR source/grid identity differs')
    return [round(row['height']*640/row['width']), 640]


def quality(q, shape, large_mesh):
    names = ('verified_pixels', 'cpu_hits', 'cpu_rays_checked', 'retained_pixels', 'initial_valid_pixels')
    counts = [q[k] for k in names]
    pixels = math.prod(shape)
    if (any(type(v) is not int for v in counts) or not 0 <= counts[0]
            or any(a > b for a, b in zip(counts, counts[1:])) or counts[-1] > pixels
            or q['cpu_missing_hits'] != counts[2]-counts[1]
            or q['rejected_inconsistent_hits'] != counts[1]-counts[0]
            or q['depth_filled_or_averaged'] is not False
            or q['independent_physical_depth_accuracy_checked'] is not False
            or q['maximum_allowed_gpu_cpu_error_m'] != .002
            or q['final_depth'] != 'CPU nearest-hit camera Z' or q['visibility_policy'] != VISIBILITY
            or not math.isclose(q['valid_fraction'], counts[0]/pixels, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(q['retained_fraction'], counts[3]/pixels, rel_tol=0, abs_tol=1e-12)):
        raise ValueError('Native visibility/CPU support accounting differs')
    error = q['maximum_accepted_gpu_cpu_error_m']
    if ((counts[0] and (not isinstance(error, (int, float)) or not math.isfinite(error)
                       or not 0 <= error <= .002)) or (not counts[0] and error is not None)):
        raise ValueError('Accepted CPU/GPU error exceeds the unchanged tolerance')
    if large_mesh and (type(q['mesh_faces']) is not int or q['mesh_faces'] <= 0
            or q['max_faces_per_pass'] != 1048576
            or q['raster_passes_per_ray_grid'] != math.ceil(q['mesh_faces']/1048576)):
        raise ValueError('Large-mesh pass accounting differs')
    return counts[0]/pixels


def source_proofs(row, proofs, scene, scene_ref, data, mask=False):
    expected = [dict(source=scene_ref['path'], sha256=scene_ref['sha256'])]
    if scene['stage'] == 'source_checks_passed':
        base = Path(data)/'scannetpp-v2-full/data'/row['scene']
        paths = (base/'scans/mesh_aligned_0.05.ply', base/'dslr/colmap/images.txt',
                 base/'dslr/nerfstudio/transforms_undistorted.json')
        if len(proofs) != 4+int(mask):
            raise ValueError('Native source proof count differs')
        for i, path in enumerate(paths, 1):
            if proofs[i]['source'] != str(path) or not HEX64.fullmatch(proofs[i]['sha256']):
                raise ValueError('Native source proof path/hash differs')
        if (proofs[1]['sha256'] != scene['mesh_sha256']
                or proofs[2]['sha256'] != scene['dslr']['colmap_images_sha256']):
            raise ValueError('Mesh or COLMAP identity differs from bound source audit')
        if mask and (proofs[4]['source'] != row['mask'] or not HEX64.fullmatch(proofs[4]['sha256'])):
            raise ValueError('Candidate anonymity-mask proof differs')
        expected.extend(proofs[1:])
    elif mask:
        raise ValueError('Source-flagged scene cannot become a native candidate')
    if proofs != expected:
        raise ValueError('Bound scene audit proof differs')


def candidate_details(entry, policy_hash, policy, scene, scene_ref, data):
    validate_member_reference(entry)
    row, meta = entry['image'], entry['geometry_metadata']
    shape = row_shape(row, data)
    support = quality(meta['source_quality'], shape, 'large_mesh_handling' in policy)
    quantiles = np.asarray(entry['depth_quantiles_m'], dtype=float)
    if (scene['stage'] != 'source_checks_passed' or meta['dataset'] != 'scannetpp_dslr'
            or meta['schema_version'] != 1 or not HEX64.fullmatch(entry['sha256'])
            or ('large_mesh_handling' in policy and meta['source_quality']['mesh_faces'] != scene['faces'])
            or meta['image_sha256'] != row['sha256'] or meta['source_id'] != row['scene']+'/'+row['view']
            or meta['policy_sha256'] != policy_hash or meta['conversion'] != CONVERSION
            or hashlib.sha256(canonical_json(meta)).hexdigest() != entry['key']
            or meta['label_kind'] != 'laser_mesh_derived' or meta['camera_kind'] != 'calibrated_sensor'
            or meta['metric_calibrated'] is not True or meta['source_needs_review'] is not False
            or meta['not_direct_per_dslr_pixel_measurements'] is not True
            or meta['depth_convention'] != 'camera_z' or meta['units'] != 'meters'
            or meta['camera_pixel_convention'] != 'normalized_half_center' or meta['shape'] != shape
            or meta['rgb_shape'] != [row['height'], row['width']]
            or meta['valid_pixels'] != meta['source_quality']['verified_pixels'] or not .05 <= support <= 1
            or not math.isclose(support, entry['valid_fraction'], rel_tol=0, abs_tol=1e-12)
            or not np.array_equal(meta['depth_to_rgb_uv_transform'], np.eye(3))
            or not np.allclose(meta['rgb_intrinsics_normalized'],
                normalized_opencv_intrinsics(*row['camera_width_height_fx_fy_cx_cy']), rtol=0, atol=1e-12)
            or quantiles.shape != (5,) or not np.isfinite(quantiles).all()
            or (np.diff(quantiles) < 0).any() or quantiles[0] < .01-1e-8 or quantiles[-1] > 100+1e-5):
        raise ValueError('Native candidate identity, camera, units or support differs')
    source_proofs(row, meta['source_proofs'], scene, scene_ref, data, mask=True)
    q = meta['source_quality']
    return dict(valid_fraction=support, median_depth_m=float(quantiles[2]),
        maximum_depth_m=float(quantiles[-1]), maximum_gpu_cpu_error_m=q['maximum_accepted_gpu_cpu_error_m'],
        rejected_hit_fraction=q['rejected_inconsistent_hits']/max(1, q['cpu_hits']))


def rejection_details(item, policy, scene, scene_ref, data):
    row, reason, stats = item['image'], item['reason'], item['stats']
    shape = row_shape(row, data)
    source_proofs(row, item['source_proofs'], scene, scene_ref, data)
    if reason == 'source_scene_pending_review':
        if (scene['stage'] == 'source_checks_passed' or stats != dict(
                source_audit_stage=scene['stage'], review_scans=scene.get('review_scans'))):
            raise ValueError('Native rejection changed retained source flags')
    elif reason == 'insufficient_verified_visible_support':
        if scene['stage'] != 'source_checks_passed' or quality(
                stats, shape, 'large_mesh_handling' in policy) >= .05 or (
                'large_mesh_handling' in policy and stats['mesh_faces'] != scene['faces']):
            raise ValueError('Native rejection violates unchanged support threshold')
    else:
        raise ValueError('Unknown native rejection reason')
    return reason


def bind(db, row, outcome, reason, policy, evidence, key=None, details=None):
    found = db.execute('SELECT ordinal,row_json,outcome FROM images WHERE image=?', (row['image'],)).fetchone()
    if found is None:
        raise ValueError('Outcome outside frozen ScanNet++ scope')
    ordinal, encoded, prior = found
    value = dict(row)
    if value.pop('corpus_ordinal', ordinal) != ordinal or value != json.loads(encoded) or prior is not None:
        raise ValueError('Changed or duplicate ScanNet++ native outcome')
    db.execute('UPDATE images SET outcome=?,reason=?,policy=?,evidence_json=?,entry_key=?,details_json=? '
        'WHERE ordinal=?', (outcome, reason, policy, canonical_json(evidence).decode(), key,
                           canonical_json(details or {}).decode(), ordinal))
    return ordinal


def producer(base, config, source_policy_ref):
    refs = {k: file_ref(base/p) for k, p in [('generation', 'report.json'),
        ('policy', 'packs/policy.json'), ('packs', 'packs/packs.json')]}
    gen, policy, packs = [checked_json(refs[k]) for k in ('generation', 'policy', 'packs')]
    checksum = hashlib.sha256(canonical_json(policy)).hexdigest()
    if (gen['stage'] != 'complete' or gen['mode'] != 'generate'
            or gen['training_started'] is not False or gen['training_admission'] is not False
            or gen['optimizer_updates'] != 0 or gen['completed'] != gen['verified_cache_members']
            or gen['completed'] != packs['count'] or gen['policy_sha256'] != checksum
            or packs['policy_sha256'] != checksum or packs['training_admission'] is not False
            or packs['stage'] != 'cache_written_not_accepted'
            or policy['dataset'] != 'scannetpp_dslr' or policy['manifest_sha256'] != config['manifest_sha256']
            or policy['label_kind'] != 'laser_mesh_derived' or policy['conversion'] != CONVERSION
            or policy['not_direct_per_dslr_pixel_measurements'] is not True
            or policy['render_width'] != 640 or policy['minimum_image_valid_fraction'] != .05
            or policy['gpu_cpu_max_difference_m'] != .002
            or policy['source_audit_policy_sha256'] != source_policy_ref['sha256']
            or policy['anonymization'] != 'native mask==255, float area support, 2-pixel erosion on output grid'
            or policy['visibility'] != 'GPU 5x5/quarter-pixel stability AND EVERY candidate verified by CPU closest hit'
            or policy['source_review_policy'] != 'quarantine entire scene pending explicit source review; no automatic repairs'
            or ('large_mesh_handling' in policy and policy['large_mesh_handling'] != LARGE_MESH)):
        raise ValueError('Complete matching native producer/readback and unchanged source policy required')
    return dict(base=base, refs=refs, gen=gen, policy=policy, packs=packs, hash=checksum)


def receipt_metadata(receipt, item, policy_hash, tar):
    if (receipt['schema_version'] != 1 or receipt['stage'] != 'cache_written_not_accepted'
            or receipt['training_admission'] is not False or Path(receipt['tar']).name != receipt['tar']
            or not HEX64.fullmatch(receipt['sha256']) or tar.stat().st_size != receipt['bytes']
            or receipt['policy_sha256'] != policy_hash or len(receipt['entries']) != item['count']
            or not 1 <= item['count'] <= 1024):
        raise ValueError('Native receipt policy, count, schema or tar size differs')
    offset, keys = 0, set()
    for entry in receipt['entries']:
        validate_member_reference(entry)
        if (entry['key'] in keys or entry['offset'] != offset+512 or not HEX64.fullmatch(entry['sha256'])):
            raise ValueError('Repeated or nonsequential native receipt member')
        keys.add(entry['key'])
        offset = entry['offset']+((entry['size']+511)//512)*512
    if offset+1024 > receipt['bytes']:
        raise ValueError('Native receipt members exceed the closed tar size')


def run(config, recovered_name, native_name, attempt):
    root, expected = Path(config['root']), config['expected_counts']['scannetpp_dslr']
    # Check final producer reports BEFORE inventory hashing or creating output.
    source_policy_ref = file_ref(root/'scannetpp-source-audit-01/policy.json')
    recovered, native = [producer(root/n, config, source_policy_ref) for n in (recovered_name, native_name)]
    scope_ref = file_ref(recovered['base']/'scope.jsonl')
    if (recovered['gen'].get('recovered_verified_subset') is not True
            or recovered['gen']['dataset'] != 'scannetpp_dslr'
            or recovered['gen']['manifest_sha256'] != config['manifest_sha256']
            or recovered['gen']['scope_sha256'] != scope_ref['sha256'] or recovered['gen']['rejected'] != 0
            or recovered['gen']['expected_scope'] != recovered['gen']['completed']
            or recovered['gen']['packs_verified'] != len(recovered['packs']['packs'])
            or native['gen']['recovered_subset'] != recovered['refs']['generation']
            or native['gen']['recovered_skipped'] != recovered['gen']['completed']
            or native['gen']['completed']+native['gen']['rejected'] != native['gen']['expected_scope']
            or native['gen']['expected_scope']+recovered['gen']['completed'] != expected):
        raise ValueError('Recovered subset and native remainder do not partition the frozen scope')
    scope_rows = [json.loads(s) for s in checked_path(scope_ref).read_text().splitlines()]
    scope = {(r['ordinal'], r['sha256']) for r in scope_rows}
    if len(scope) != len(scope_rows) or len(scope) != recovered['gen']['completed']:
        raise ValueError('Recovered scope has duplicate or missing identities')
    source_ref = file_ref(root/'inventory.json')
    source = checked_json(source_ref)
    inventory_ref = dict(path=str(root/'inventory.sqlite'), sha256=source['inventory_sha256'])
    if (source['stage'] != 'inventory_complete' or source['manifest_sha256'] != config['manifest_sha256']
            or source['dataset_counts'] != config['expected_counts']):
        raise ValueError('Matching frozen inventory required')
    checked_path(inventory_ref)
    audit_ref = file_ref(root/'scannetpp-source-audit-01/report.json')
    audit = checked_json(audit_ref)
    if (audit['stage'] != 'complete' or audit['manifest_sha256'] != config['manifest_sha256']
            or audit['scenes_completed'] != audit['expected_scenes']
            or len(audit['scenes']) != audit['scenes_completed'] or audit['training_admission'] is not False):
        raise ValueError('Completed source-audit inventory required')
    scenes, scene_refs = {}, {}
    for item in audit['scenes']:
        name = item['scene']
        if item['report'] != name+'.json' or name in scenes or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Changed or duplicate audited scene')
        ref = dict(path=str(root/'scannetpp-source-audit-01'/item['report']), sha256=item['sha256'])
        scene = checked_json(ref)
        if (scene['scene'] != name or scene['stage'] != item['stage']
                or scene['stage'] not in ('source_checks_passed', 'needs_review')):
            raise ValueError('Source-audit scene identity differs')
        scenes[name], scene_refs[name] = scene, ref
    if sum(s['stage'] == 'needs_review' for s in scenes.values()) != audit['scenes_needing_review']:
        raise ValueError('Source-review scene count differs')
    rejection_ref = file_ref(native['base']/'rejections.json')
    rejections = checked_json(rejection_ref)
    if len(rejections) != native['gen']['rejected']:
        raise ValueError('Native rejection count differs')
    output = root/attempt
    output.mkdir(exist_ok=False)
    policies, counts, native_scenes = Counter(), Counter(), set()
    with tempfile.TemporaryDirectory(prefix='scannet-native-outcomes-') as scratch:
        temporary = Path(scratch)/'outcomes.sqlite'
        with sqlite3.connect(temporary) as db:
            db.execute('PRAGMA cache_size=-32768')
            db.execute('CREATE TABLE images(ordinal INTEGER PRIMARY KEY,image TEXT UNIQUE,sha256 TEXT,'
                'scene_group TEXT,row_json TEXT,outcome TEXT,reason TEXT,policy TEXT,'
                'evidence_json TEXT,entry_key TEXT UNIQUE,details_json TEXT)')
            with readonly_db(inventory_ref['path']) as original:
                for ordinal, image, checksum, group, encoded in original.execute(
                        "SELECT ordinal,image,sha256,scene_group,row_json FROM images WHERE dataset='scannetpp_dslr' ORDER BY ordinal"):
                    row = dict(json.loads(encoded), image=image)
                    row_shape(row, config['dataset_root'])
                    if row['sha256'] != checksum or row['group'] != group or row['scene'] not in scenes:
                        raise ValueError('Frozen native inventory identity differs')
                    db.execute('INSERT INTO images VALUES(?,?,?,?,?,NULL,NULL,NULL,NULL,NULL,NULL)',
                        (ordinal, image, checksum, group, canonical_json(row).decode()))
            if (db.execute('SELECT COUNT(*) FROM images').fetchone()[0] != expected
                    or db.execute('SELECT COUNT(DISTINCT scene_group) FROM images').fetchone()[0] != len(scenes)):
                raise ValueError('Frozen source count or scene scope differs')
            for src in (recovered, native):
                packed, seen = 0, set()
                for item in src['packs']['packs']:
                    name = item['receipt']
                    if Path(name).name != name or name in seen:
                        raise ValueError('Duplicate or invalid receipt path')
                    seen.add(name)
                    path = src['base']/'packs'/name
                    proof = dict(path=str(path), sha256=item['sha256'])
                    receipt = checked_json(proof)
                    receipt_metadata(receipt, item, src['hash'], path.parent/receipt['tar'])
                    for entry in receipt['entries']:
                        scene = entry['image']['scene']
                        details = candidate_details(entry, src['hash'], src['policy'], scenes[scene],
                            scene_refs[scene], config['dataset_root'])
                        ordinal = bind(db, entry['image'], 'native_candidate', 'quality_review_pending',
                            src['hash'], proof, entry['key'], details)
                        if src is recovered:
                            identity = (ordinal, entry['image']['sha256'])
                            if identity not in scope:
                                raise ValueError('Recovered candidate lies outside exact recovery scope')
                            scope.remove(identity)
                        else:
                            native_scenes.add(scene)
                        packed += 1
                    db.commit()
                if packed != src['gen']['completed']:
                    raise ValueError('Not every verified producer member was accounted')
                policies[src['hash']] += packed
            if scope:
                raise ValueError('Recovered scope is incomplete')
            for item in rejections:
                scene = item['image']['scene']
                reason = rejection_details(item, native['policy'], scenes[scene], scene_refs[scene], config['dataset_root'])
                bind(db, item['image'], 'native_rejected', reason, native['hash'], rejection_ref, details=item['stats'])
                native_scenes.add(scene)
            counts.update(dict(db.execute('SELECT outcome,COUNT(*) FROM images GROUP BY outcome')))
            if None in counts or sum(counts.values()) != expected:
                raise ValueError('Full frozen native outcomes are missing')
            if len(native_scenes) != native['gen']['scenes_seen']:
                raise ValueError('Native remainder scene counter differs')
            reasons = dict(db.execute("SELECT reason,COUNT(*) FROM images WHERE outcome='native_rejected' GROUP BY reason"))
            strata = [dict(group=g, policy=p, outcome=o, reason=r, images=n) for g,p,o,r,n in db.execute(
                'SELECT scene_group,policy,outcome,reason,COUNT(*) FROM images GROUP BY 1,2,3,4 ORDER BY 1,2,3,4')]
            choices = {}
            for policy in policies:
                for metric, direction in [('valid_fraction', 'ASC'), ('median_depth_m', 'ASC'),
                        ('median_depth_m', 'DESC'), ('maximum_gpu_cpu_error_m', 'DESC'), ('rejected_hit_fraction', 'DESC')]:
                    for ordinal, row, key, proof, details in db.execute(
                            "SELECT ordinal,row_json,entry_key,evidence_json,details_json FROM images WHERE "
                            "outcome='native_candidate' AND policy=? ORDER BY "
                            f"json_extract(details_json,'$.{metric}') {direction},ordinal LIMIT 3", (policy,)):
                        value = choices.setdefault(ordinal, dict(corpus_ordinal=ordinal, image=json.loads(row),
                            entry_key=key, receipt=json.loads(proof), metrics=json.loads(details),
                            policy_sha256=policy, selection_reasons=[]))
                        value['selection_reasons'].append(metric+':'+direction)
            db.commit()
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Native outcome database integrity failure')
        target = output/'outcomes.sqlite.incoming'
        shutil.copyfile(temporary, target)
        if file_ref(temporary)['sha256'] != file_ref(target)['sha256']:
            raise ValueError('Published native outcome index differs')
        os.rename(target, output/'outcomes.sqlite')
    evidence = [source_ref, inventory_ref, audit_ref, source_policy_ref, scope_ref, rejection_ref,
                *scene_refs.values(), *(v for src in (recovered, native) for v in src['refs'].values())]
    for ref in evidence:
        checked_path(ref)
    publish_json(output/'strata.json', strata)
    publish_json(output/'review-candidates.json', [choices[k] for k in sorted(choices)])
    teacher = root/'scannetpp-review-teacher-outcomes-01/report.json'
    report = dict(stage='native_outcomes_accounted_not_accepted', dataset='scannetpp_dslr',
        completed_at=time.time(), expected_scope=expected, manifest_sha256=config['manifest_sha256'],
        outcome_counts=dict(counts), rejection_reasons=reasons, candidate_policy_counts=dict(policies),
        source_review_scenes=sum(s['stage'] != 'source_checks_passed' for s in scenes.values()),
        source_flags_preserved=True, teacher_replacements_counted_as_additional_native_rows=False,
        existing_teacher_outcomes=file_ref(teacher) if teacher.exists() else None,
        source_report=source_ref, inventory=inventory_ref, evidence=evidence,
        index=file_ref(output/'outcomes.sqlite'), strata=file_ref(output/'strata.json'),
        review_candidates=file_ref(output/'review-candidates.json'), review_candidate_count=len(choices),
        producer_array_readback_complete=True, receipt_metadata_only=True,
        independent_array_readback_performed=False, pack_payload_hashes_rechecked=False,
        physical_accuracy_established=False, source_quality_accepted=False, automatic_exclusions=0,
        training_admission=False, optimizer_updates=0, local_temporary_index=True,
        audit_code=file_ref(Path(__file__)))
    publish_json(output/'report.json', report)
    print(json.dumps({k: report[k] for k in ('stage', 'outcome_counts', 'rejection_reasons')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--recovered', default='scannetpp-recovered-01')
    parser.add_argument('--native', default='scannetpp-native-cache-02')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.recovered, args.native, args.attempt)):
        parser.error('Use attempt basenames')
    run(json.loads(args.config.read_text()), args.recovered, args.native, args.attempt)
