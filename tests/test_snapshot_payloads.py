from copy import deepcopy
import json
import tarfile

import numpy as np
import pytest

from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import (GeometryPackWriter, canonical_json,
    iter_geometry_pack, publish_json, read_geometry_member, sha_file)
from snapshot_geometry_candidates import prepare_db
from verify_snapshot_payloads import run, verify_pack


def fixture(root):
    writer = GeometryPackWriter(root / 'source/packs', dict(dataset='hypersim'))
    for i in range(2):
        row = dict(image=str(root / f'rgb{i}.png'), sha256=str(i) * 64,
            dataset='hypersim', split='train', group='hypersim:scene')
        meta = dict(image_sha256=row['sha256'], dataset='hypersim', source_id=str(i),
            label_kind='synthetic_renderer', camera_kind='synthetic_camera', metric_calibrated=True,
            conversion='fixture', policy_sha256=writer.policy_hash, source_needs_review=bool(i),
            source_proofs=[dict(source='fixture/source', sha256='b' * 64)])
        depth = np.arange(64, dtype=np.float32).reshape(8, 8) + i + 1
        valid = depth % 3 != 0
        writer.add(row, depth, valid, np.eye(3), meta)
    writer.close()
    receipt_path = root / 'source/packs/pack-000000.json'
    receipt = json.loads(receipt_path.read_text())
    pack = receipt_path.parent / receipt['tar']
    source = dict(attempt='source', dataset='hypersim', policy_sha256=writer.policy_hash,
        policy=file_ref(root / 'source/packs/policy.json'))
    candidates = [dict(policy_sha256=writer.policy_hash, entry_key=e['key'],
        image_sha256=e['image']['sha256'], member_sha256=e['sha256'], pack_id=1,
        native_flag=i, external_flag=1-i, excluded=1-i, reviewed_source=0,
        label_kind='synthetic_renderer') for i, e in enumerate(receipt['entries'])]
    item = (1, 'source', json.dumps(file_ref(receipt_path)),
            json.dumps(dict(path=str(pack), sha256=sha_file(pack), bytes=pack.stat().st_size)), 2)
    return pack, receipt_path, receipt, source, candidates, item


def test_single_pass_matches_random_reads_and_preserves_all_flags(tmp_path):
    pack, _, receipt, source, candidates, item = fixture(tmp_path)
    for entry, actual in iter_geometry_pack(pack, receipt):
        expected = read_geometry_member(pack, entry)
        for a, b in zip(actual[:3], expected[:3]):
            np.testing.assert_array_equal(a, b)
        assert actual[3] == expected[3]
    report = verify_pack(item, candidates, source)
    assert report['verified_members'] == 2 and not report['training_admission']
    assert report['records'][0]['external_flag'] and report['records'][0]['excluded']
    assert report['records'][1]['native_flag']
    assert all(sum(r['coverage']['histogram_counts']) == r['coverage']['valid_pixels']
               for r in report['records'])


def test_whole_tar_hash_is_checked_even_after_every_member_decodes(tmp_path):
    pack, _, receipt, _, _, _ = fixture(tmp_path)
    data = bytearray(pack.read_bytes())
    header = tarfile.TarInfo.frombuf(data[:512], 'utf-8', 'strict')
    header.mtime = 1  # Valid archive/member data, but different frozen pack bytes.
    data[:512] = header.tobuf(format=tarfile.USTAR_FORMAT)
    pack.write_bytes(data)
    iterator = iter_geometry_pack(pack, receipt)
    next(iterator)
    next(iterator)
    with pytest.raises(ValueError, match='Whole pack hash'):
        next(iterator)


def test_truncation_extra_members_order_and_member_corruption_fail(tmp_path):
    pack, _, receipt, _, _, _ = fixture(tmp_path)
    original = pack.read_bytes()
    pack.write_bytes(original[:-512])
    with pytest.raises(ValueError, match='byte count'):
        list(iter_geometry_pack(pack, receipt))
    pack.write_bytes(original)
    for entries in (receipt['entries'][::-1], receipt['entries'][:1],
                    [receipt['entries'][0], receipt['entries'][0]]):
        with pytest.raises(ValueError):
            list(iter_geometry_pack(pack, dict(receipt, entries=entries)))
    data = bytearray(original)
    data[receipt['entries'][0]['offset'] + 20] ^= 1
    pack.write_bytes(data)
    with pytest.raises(ValueError, match='hash mismatch'):
        list(iter_geometry_pack(pack, receipt))


def test_payload_statistics_and_snapshot_membership_are_not_trusted(tmp_path):
    _, receipt_path, receipt, source, candidates, item = fixture(tmp_path)
    wrong = deepcopy(candidates)
    wrong[0]['image_sha256'] = 'e' * 64
    with pytest.raises(ValueError, match='provenance'):
        verify_pack(item, wrong, source)
    receipt['entries'][0]['valid_fraction'] = 1.
    receipt_path.write_text(json.dumps(receipt))
    item = (*item[:2], json.dumps(file_ref(receipt_path)), *item[3:])
    with pytest.raises(ValueError, match='statistics differ'):
        verify_pack(item, candidates, source)


def test_full_preflight_then_census_are_snapshot_bound_and_never_admit(tmp_path):
    _, _, _, source, candidates, item = fixture(tmp_path)
    path = tmp_path / 'snapshot.sqlite'
    db = prepare_db(path)
    db.execute('INSERT INTO packs VALUES (?,?,?,?,?)', item)
    for c in candidates:
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)',
            tuple(c[k] for k in ('policy_sha256', 'entry_key', 'image_sha256', 'pack_id',
                'native_flag', 'external_flag', 'excluded', 'reviewed_source', 'label_kind', 'member_sha256')))
    db.commit()
    db.close()
    publish_json(tmp_path / 'capture.json', dict(sources=[source]))
    snapshot = dict(stage='candidate_snapshot_metadata_complete_not_accepted',
        training_admission=False, optimizer_updates=0, processed_packs=1, processed_entries=2,
        index=file_ref(path), capture=file_ref(tmp_path / 'capture.json'))
    publish_json(tmp_path / 'snapshot.json', snapshot)
    with pytest.raises(ValueError, match='preflight required'):
        run(tmp_path / 'snapshot.json', tmp_path / 'missing', 'census')
    probe = run(tmp_path / 'snapshot.json', tmp_path / 'probe', 'preflight')
    assert probe['verified_members'] == 2 and not probe['all_snapshot_packs_verified']
    result = run(tmp_path / 'snapshot.json', tmp_path / 'full', 'census', tmp_path / 'probe/report.json')
    assert result['verified_packs'] == 1 and result['verified_members'] == 2
    assert result['all_snapshot_members_verified'] and not result['source_quality_accepted']
    assert result['optimizer_updates'] == 0 and not result['training_admission']
    snapshot['processed_entries'] = 3
    (tmp_path / 'snapshot.json').write_bytes(canonical_json(snapshot))
    with pytest.raises(ValueError, match='preflight required'):
        run(tmp_path / 'snapshot.json', tmp_path / 'changed', 'census', tmp_path / 'probe/report.json')


def test_trailing_corruption_never_publishes_a_pack_success(tmp_path):
    pack, _, receipt, source, candidates, item = fixture(tmp_path)
    data = bytearray(pack.read_bytes())
    data[-1] = 1
    pack.write_bytes(data)
    with pytest.raises(ValueError, match='Unexpected data'):
        verify_pack(item, candidates, source)
