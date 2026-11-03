"""Bind completed snapshot readback to accounting without rereading every tar.

The full CPU census already decoded every member and verified every tar. These
helpers reuse its immutable per-pack results, while retaining current source
reviews, exact membership checks, and training-time per-member checksums.
"""
from pathlib import Path

import numpy as np

from .corpus_admission import checked_json, checked_path


def checked_snapshot_generation(generation, candidate, scope, index):
    snapshot_ref = scope.get('snapshot')
    if (scope.get('kind') != 'frozen_candidate_subset' or snapshot_ref is None
            or generation.get('stage') != 'snapshot_source_payload_verified'
            or generation.get('mode') != 'snapshot' or generation.get('snapshot') != snapshot_ref
            or generation.get('packs_index') != candidate['packs_index']
            or generation.get('training_admission') is not False
            or generation.get('training_started') is not False
            or generation.get('optimizer_updates') != 0 or generation.get('rejected') != 0
            or generation.get('completed') != candidate['scope_count']
            or generation.get('verified_cache_members') != index['count']
            or generation.get('completed') != index['count']
            or generation.get('policy_sha256') != index['policy_sha256']):
        raise ValueError('Snapshot source needs complete scope-bound payload verification')
    snapshot = checked_json(snapshot_ref)
    full = checked_json(generation['payload_report'])
    source = checked_json(generation['source_scope'])
    if (snapshot.get('stage') != 'candidate_snapshot_metadata_complete_not_accepted'
            or snapshot['dataset_counts'] != scope['expected_counts']
            or source.get('stage') != 'snapshot_source_scope_prepared'
            or source.get('snapshot') != snapshot_ref or source.get('dataset') != candidate['dataset']
            or source.get('policy_sha256') != index['policy_sha256']
            or source.get('candidate_images') != index['count']
            or full.get('stage') != 'snapshot_payload_verified_not_accepted'
            or full.get('mode') != 'census' or full.get('snapshot') != snapshot_ref
            or full.get('all_snapshot_packs_verified') is not True
            or full.get('all_snapshot_members_verified') is not True or full.get('failed_packs') != 0
            or full.get('verified_packs') != snapshot['processed_packs']
            or full.get('verified_members') != snapshot['processed_entries']
            or full['source_counts'].get(source['attempt']) != index['count']):
        raise ValueError('Incomplete or mismatched full snapshot payload report')
    # Validate the published verifier code as well as the completed result.
    for ref in full['code']:
        checked_path(ref)
    reports = checked_json(full['pack_reports'])
    if len(reports) != full['verified_packs'] or len({r['path'] for r in reports}) != len(reports):
        raise ValueError('Snapshot payload report membership differs')
    by_id = {int(Path(r['path']).stem): r for r in reports}
    if len(by_id) != len(reports):
        raise ValueError('Repeated snapshot payload report ID')
    pack_dir = Path(candidate['packs_index']['path']).parent
    expected = {str(pack_dir / p['receipt']): (p['sha256'], p['count']) for p in index['packs']}
    captured = {p['receipt']['path']: (p['receipt']['sha256'], p['count']) for p in source['packs']}
    if (len(expected) != len(index['packs']) or len(captured) != len(source['packs'])
            or expected != captured or sum(p['count'] for p in source['packs']) != index['count']):
        raise ValueError('Published stage pack index differs from the captured source')
    result = {}
    for pack in source['packs']:
        if pack['pack_id'] not in by_id:
            raise ValueError('Source pack lacks a completed payload report')
        result[pack['receipt']['path']] = dict(report=by_id[pack['pack_id']], captured=pack,
                                              source=source['attempt'])
    return result


def checked_snapshot_pack(binding, receipt_ref, receipt, tar):
    proof = checked_json(binding['report'])
    original = binding['captured']
    expected_tar = dict(original['tar'], payload_rehashed=True)
    if (proof.get('stage') != 'pack_payload_verified_not_accepted'
            or proof.get('pack_id') != original['pack_id']
            or proof.get('source') != binding['source'] or proof.get('receipt') != receipt_ref
            or proof.get('tar') != expected_tar
            or proof.get('whole_tar_hash_verified') is not True
            or proof.get('every_member_decoded') is not True
            or proof.get('candidate_flags_preserved') is not True
            or proof.get('training_admission') is not False or proof.get('optimizer_updates') != 0
            or proof.get('verified_members') != len(receipt['entries'])
            or receipt['bytes'] != expected_tar['bytes'] or receipt['sha256'] != expected_tar['sha256']
            or str(tar) != expected_tar['path']):
        raise ValueError('Snapshot pack readback does not match the immutable receipt')
    stat = tar.stat()
    if (stat.st_size != receipt['bytes']
            or stat.st_mtime_ns > Path(binding['report']['path']).stat().st_mtime_ns):
        raise ValueError('Snapshot pack changed after its complete readback')
    records = {v['entry_key']: v for v in proof['records']}
    if len(records) != len(proof['records']) or len(records) != len(receipt['entries']):
        raise ValueError('Snapshot pack member coverage is incomplete')
    return records


def checked_snapshot_member(records, entry):
    value = records.pop(entry['key'], None)
    row, meta = entry['image'], entry['geometry_metadata']
    if (value is None or value['member_sha256'] != entry['sha256']
            or value['image_sha256'] != row['sha256'] or value['dataset'] != row['dataset']
            or value['group'] != row['group'] or value['label_kind'] != meta['label_kind']
            or value['native_flag'] != bool(meta.get('source_needs_review', False))
            or not np.allclose(value['depth_quantiles'], entry['depth_quantiles_m'], rtol=0, atol=1e-12)
            or not np.isclose(value['coverage']['valid_fraction'], entry['valid_fraction'], rtol=0, atol=1e-12)):
        raise ValueError('Snapshot member statistics or provenance changed')
    coverage = value['coverage']
    if (coverage['valid_pixels'] < 1 or coverage['valid_pixels'] > coverage['grid_pixels']
            or sum(coverage['histogram_counts']) != coverage['valid_pixels']
            or not np.isclose(coverage['valid_fraction'], coverage['valid_pixels'] / coverage['grid_pixels'],
                              rtol=0, atol=1e-12)):
        raise ValueError('Snapshot member support statistics are invalid')
    return value
