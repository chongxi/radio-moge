"""Sequential full payload readback of a frozen candidate snapshot, CPU only.

Each completed per-pack receipt proves its tar and every declared NPZ payload.
Candidate/source flags remain untouched; this is never a training acceptance.
"""
import argparse
from collections import Counter
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

import numpy as np

from indoor_download import write_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.corpus_coverage import geometry_coverage
from radio_moge.geometry_cache import canonical_json, iter_geometry_pack, publish_json
from radio_moge.geometry_sampling import camera_matrix


def verify_pack(item, candidates, source):
    pack_id, source_name, receipt_encoded, tar_encoded, count = item
    receipt_ref, tar_ref = json.loads(receipt_encoded), json.loads(tar_encoded)
    receipt = checked_json(receipt_ref)
    path = Path(tar_ref['path'])
    if (source_name != source['attempt'] or len(candidates) != count
            or len(receipt['entries']) != count or receipt['bytes'] != tar_ref['bytes']
            or receipt['sha256'] != tar_ref['sha256'] or receipt['tar'] != path.name
            or receipt['policy_sha256'] != source['policy_sha256']
            or receipt['stage'] != 'cache_written_not_accepted'
            or receipt['training_admission'] is not False):
        raise ValueError('Frozen snapshot and published pack receipt disagree')
    expected = {c['entry_key']: c for c in candidates}
    if len(expected) != count:
        raise ValueError('Duplicate snapshot candidate identity')
    records = []
    for entry, (depth, valid, k, meta) in iter_geometry_pack(path, receipt):
        candidate = expected.pop(entry['key'], None)
        row = entry['image']
        if (candidate is None or candidate['policy_sha256'] != receipt['policy_sha256']
                or candidate['member_sha256'] != entry['sha256']
                or candidate['image_sha256'] != row['sha256']
                or candidate['native_flag'] != bool(meta.get('source_needs_review', False))
                or candidate['label_kind'] != meta['label_kind']
                or meta['image_sha256'] != row['sha256'] or row['split'] != 'train'
                or row['dataset'] != source['dataset'] or meta['dataset'] != row['dataset']
                or meta['policy_sha256'] != receipt['policy_sha256']
                or meta['schema_version'] != 1 or meta['depth_convention'] != 'camera_z'
                or meta['units'] != 'meters'
                or meta['camera_pixel_convention'] != 'normalized_half_center'):
            raise ValueError('Payload provenance differs from the frozen candidate')
        camera_matrix(k)
        pseudo = meta['label_kind'] == 'teacher_pseudo'
        if ((pseudo and (meta['metric_calibrated'] is not False
                or meta['camera_kind'] != 'teacher_estimated' or not meta.get('teacher_revision')))
                or (not pseudo and (meta['metric_calibrated'] is not True
                    or meta['camera_kind'] not in ('calibrated_sensor', 'synthetic_camera')))):
            raise ValueError('Payload supervision provenance is invalid')
        quantiles = np.quantile(depth[valid], [0, .01, .5, .99, 1])
        if (not math.isclose(float(valid.mean()), entry['valid_fraction'], rel_tol=0, abs_tol=1e-12)
                or not np.allclose(quantiles, entry['depth_quantiles_m'], rtol=0, atol=1e-12)):
            raise ValueError('Payload support or depth statistics differ from the receipt')
        # Coverage of all candidates is descriptive; flagged arrays stay flagged.
        coverage = geometry_coverage(depth, valid, 'relative_pseudo' if pseudo else 'metric_native')
        records.append(dict(entry_key=entry['key'], member_sha256=entry['sha256'],
            image_sha256=row['sha256'], dataset=row['dataset'], group=row['group'],
            label_kind=meta['label_kind'], native_flag=bool(candidate['native_flag']),
            external_flag=bool(candidate['external_flag']), excluded=bool(candidate['excluded']),
            depth_quantiles=quantiles.tolist(), coverage=coverage))
    if expected:
        raise ValueError('Snapshot members are absent from the complete pack')
    checked_path(receipt_ref)
    return dict(stage='pack_payload_verified_not_accepted', pack_id=pack_id,
        source=source_name, receipt=receipt_ref, tar=dict(tar_ref, payload_rehashed=True),
        verified_members=len(records),
        records=records, whole_tar_hash_verified=True, every_member_decoded=True,
        candidate_flags_preserved=True, training_admission=False, optimizer_updates=0)


def code_refs():
    base = Path(__file__).resolve().parents[1]
    return [file_ref(base / name) for name in ('scripts/verify_snapshot_payloads.py',
        'src/radio_moge/geometry_cache.py', 'src/radio_moge/corpus_coverage.py',
        'src/radio_moge/geometry_sampling.py')]


def run(snapshot_path, output, mode, preflight_path=None):
    if mode not in ('preflight', 'census'):
        raise ValueError('Use preflight or census mode')
    snapshot_ref = file_ref(snapshot_path)
    snapshot = checked_json(snapshot_ref)
    capture = checked_json(snapshot['capture'])
    code = code_refs()
    if (snapshot['stage'] != 'candidate_snapshot_metadata_complete_not_accepted'
            or snapshot['training_admission'] is not False or snapshot['optimizer_updates'] != 0):
        raise ValueError('Complete frozen snapshot metadata required')
    if mode == 'census':
        if preflight_path is None:
            raise ValueError('Completed matching payload preflight required')
        probe = checked_json(file_ref(preflight_path))
        if (probe['stage'] != 'snapshot_payload_preflight_complete'
                or probe['snapshot'] != snapshot_ref or probe['code'] != code
                or probe['failed_packs'] != 0):
            raise ValueError('Completed matching payload preflight required')
    sources = {s['attempt']: s for s in capture['sources']}
    for source in sources.values():
        policy = checked_json(source['policy'])
        if hashlib.sha256(canonical_json(policy)).hexdigest() != source['policy_sha256']:
            raise ValueError('Snapshot source policy changed')
    output.mkdir(exist_ok=False)
    (output / 'packs').mkdir()
    state = dict(stage='initializing_payload_verification', mode=mode, started_at=time.time(),
        verified_packs=0, verified_members=0, verified_bytes=0, failed_packs=0,
        training_admission=False, optimizer_updates=0, gpu_used=False, workers=1)
    write_json(output / 'status.json', state)
    try:
        with tempfile.TemporaryDirectory(prefix='snapshot-payloads-') as temporary:
            local = Path(temporary) / 'snapshot.sqlite'
            shutil.copyfile(checked_path(snapshot['index']), local)
            if file_ref(local)['sha256'] != snapshot['index']['sha256']:
                raise ValueError('Local snapshot copy differs')
            with sqlite3.connect(local) as db:
                db.execute('PRAGMA cache_size=-32768')
                db.execute('PRAGMA temp_store=FILE')
                packs = list(db.execute('SELECT id,source,receipt_json,tar_json,entries FROM packs ORDER BY id'))
                if (len(packs) != snapshot['processed_packs']
                        or sum(p[4] for p in packs) != snapshot['processed_entries']):
                    raise ValueError('Snapshot pack counts differ')
                endpoints = set()
                for source in sources:
                    ids = [p[0] for p in packs if p[1] == source]
                    if not ids:
                        raise ValueError('Source omitted from snapshot')
                    endpoints.update((ids[0], ids[-1]))
                selected = endpoints if mode == 'preflight' else {p[0] for p in packs}
                cols = ['pack_id', 'policy_sha256', 'entry_key', 'image_sha256', 'member_sha256',
                        'native_flag', 'external_flag', 'excluded', 'label_kind']
                cursor = db.execute('SELECT ' + ','.join(cols) + ' FROM candidates ORDER BY pack_id,entry_key')
                grouped = iter(groupby(cursor, key=lambda row: row[0]))
                counts, refs = Counter(), []
                for pack in packs:
                    group_id, values = next(grouped)
                    if group_id != pack[0]:
                        raise ValueError('Candidate pack ordering differs')
                    candidates = [dict(zip(cols, row)) for row in values]
                    if len(candidates) != pack[4]:
                        raise ValueError('Candidate membership count differs')
                    if pack[0] not in selected:
                        continue
                    state.update(stage='verifying_pack_payloads', current_pack=pack[0],
                        current_source=pack[1], expected_packs=len(selected), updated_at=time.time())
                    write_json(output / 'status.json', state)
                    result = verify_pack(pack, candidates, sources[pack[1]])
                    path = output / 'packs' / f'{pack[0]:06d}.json'
                    publish_json(path, result)
                    refs.append(file_ref(path))
                    state['verified_packs'] += 1
                    state['verified_members'] += result['verified_members']
                    state['verified_bytes'] += result['tar']['bytes']
                    counts[pack[1]] += result['verified_members']
                    if state['verified_packs'] % 16 == 0:
                        print(json.dumps(state), flush=True)
                if next(grouped, None) is not None:
                    raise ValueError('Candidate rows reference absent snapshot packs')
            checked_path(snapshot_ref)
            checked_path(snapshot['capture'])
            checked_path(snapshot['index'])
            for source in sources.values():
                checked_path(source['policy'])
            for ref in code:
                checked_path(ref)
            publish_json(output / 'pack-reports.json', refs)
            state.update(stage='snapshot_payload_preflight_complete' if mode == 'preflight'
                else 'snapshot_payload_verified_not_accepted', completed_at=time.time(),
                snapshot=snapshot_ref, source_counts=dict(counts), code=code,
                pack_reports=file_ref(output / 'pack-reports.json'), candidate_flags_preserved=True,
                source_quality_accepted=False, all_snapshot_packs_verified=(mode == 'census'),
                all_snapshot_members_verified=(mode == 'census'))
            publish_json(output / 'report.json', state)
            write_json(output / 'status.json', state)
            return state
    except BaseException as error:
        state.update(stage='failed', failed_packs=1, error=type(error).__name__, message=str(error))
        write_json(output / 'status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['preflight', 'census'], required=True)
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args()
    result = run(args.snapshot, args.output, args.mode, args.preflight)
    print(json.dumps({k: result[k] for k in ('stage', 'verified_packs', 'verified_members')}), flush=True)
