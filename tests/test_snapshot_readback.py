import json
import os
from pathlib import Path

import pytest

from finalize_snapshot_source_readbacks import run as finalize
from radio_moge.corpus_admission import build_accounting, checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from test_corpus_admission import admission_fixture
from verify_snapshot_payloads import verify_pack


def snapshot_fixture(tmp_path):
    root, plan, _ = admission_fixture(tmp_path)
    scope_path = Path(plan['scope']['path'])
    scope = checked_json(plan['scope'])
    snapshot_path = root / 'snapshot.json'
    publish_json(snapshot_path, dict(stage='candidate_snapshot_metadata_complete_not_accepted',
        dataset_counts=scope['expected_counts'], processed_packs=2, processed_entries=2,
        training_admission=False, optimizer_updates=0, test_fixture_only=True))
    snapshot_ref = file_ref(snapshot_path)
    scope.update(kind='frozen_candidate_subset', snapshot=snapshot_ref)
    scope_path.write_text(json.dumps(scope))
    plan['scope'] = file_ref(scope_path)
    directory = root / 'payloads'
    directory.mkdir()
    reports, source_refs, counts = [], [], {}
    for i, candidate in enumerate(plan['sources'], 1):
        index = checked_json(candidate['packs_index'])
        pack_dir = Path(candidate['packs_index']['path']).parent
        receipt_ref = file_ref(pack_dir / index['packs'][0]['receipt'])
        receipt = checked_json(receipt_ref)
        entry = receipt['entries'][0]
        tar = pack_dir / receipt['tar']
        tar_ref = dict(file_ref(tar), bytes=tar.stat().st_size, payload_rehashed=False)
        name = candidate['dataset']
        source = dict(attempt=name, dataset=name, policy_sha256=index['policy_sha256'])
        c = dict(policy_sha256=index['policy_sha256'], entry_key=entry['key'],
            member_sha256=entry['sha256'], image_sha256=entry['image']['sha256'],
            native_flag=False, external_flag=False, excluded=False, label_kind=entry['geometry_metadata']['label_kind'])
        actual = verify_pack((i, name, json.dumps(receipt_ref), json.dumps(tar_ref), 1), [c], source)
        path = directory / f'{i:06d}.json'
        publish_json(path, actual)
        reports.append(file_ref(path))
        source_path = root / f'scope-{name}.json'
        publish_json(source_path, dict(source, stage='snapshot_source_scope_prepared', snapshot=snapshot_ref,
            candidate_images=1, previous_source_review=candidate['review'],
            packs=[dict(pack_id=i, receipt=receipt_ref, tar=tar_ref, count=1)]))
        source_refs.append(file_ref(source_path))
        counts[name] = 1
    publish_json(root / 'pack-reports.json', reports)
    census = root / 'census.json'
    publish_json(census, dict(stage='snapshot_payload_verified_not_accepted', mode='census',
        snapshot=snapshot_ref, all_snapshot_packs_verified=True, all_snapshot_members_verified=True,
        failed_packs=0, verified_packs=2, verified_members=2, source_counts=counts,
        pack_reports=file_ref(root / 'pack-reports.json'), code=[file_ref(Path(__file__))]))
    for candidate, source_ref in zip(plan['sources'], source_refs):
        previous = Path(candidate['generation_report']['path'])
        gen = checked_json(candidate['generation_report'])
        gen.update(stage='snapshot_source_payload_verified', mode='snapshot', snapshot=snapshot_ref,
            payload_report=file_ref(census), source_scope=source_ref, packs_index=candidate['packs_index'])
        previous.write_text(json.dumps(gen))
        candidate['generation_report'] = file_ref(previous)
    return root, plan, census, source_refs


def test_actual_completed_readback_can_be_reused_without_decoding_tars_again(tmp_path, monkeypatch):
    root, plan, _, _ = snapshot_fixture(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Already verified snapshot payload must not be read a second time')
    monkeypatch.setattr('radio_moge.corpus_admission.read_geometry_member', forbidden)
    import radio_moge.corpus_admission as admission
    original = admission.sha_file
    def hash_once(path):
        if str(path).endswith('.tar'):
            forbidden()
        return original(path)
    monkeypatch.setattr(admission, 'sha_file', hash_once)
    report = build_accounting(plan, root / 'account')
    assert report['selected_images'] == 2 and not report['training_admission']
    assert {s['payload_verification'] for s in report['sources']} == {'completed_snapshot_census'}
    assert {s['label_kind'] for s in report['coverage']} == {'synthetic_renderer', 'teacher_pseudo'}


def test_incomplete_snapshot_census_cannot_be_reused(tmp_path):
    root, plan, census, _ = snapshot_fixture(tmp_path)
    value = json.loads(census.read_text())
    value['all_snapshot_members_verified'] = False
    census.write_text(json.dumps(value))
    path = Path(plan['sources'][0]['generation_report']['path'])
    gen = json.loads(path.read_text())
    gen['payload_report'] = file_ref(census)
    path.write_text(json.dumps(gen))
    plan['sources'][0]['generation_report'] = file_ref(path)
    with pytest.raises(ValueError, match='Incomplete or mismatched'):
        build_accounting(plan, root / 'account')


def test_tar_modified_after_verification_is_rejected_before_training(tmp_path):
    root, plan, _, _ = snapshot_fixture(tmp_path)
    tar = root / 'hypersim/packs/pack-000000.tar'
    stat = tar.stat()
    os.utime(tar, ns=(stat.st_atime_ns, stat.st_mtime_ns + 100_000_000_000))
    with pytest.raises(ValueError, match='changed after'):
        build_accounting(plan, root / 'account')


def test_captured_external_flag_cannot_be_cleared_by_reusing_readback(tmp_path):
    root, plan, census, _ = snapshot_fixture(tmp_path)
    proof = root / 'payloads/000001.json'
    value = json.loads(proof.read_text())
    value['records'][0]['external_flag'] = True
    proof.write_text(json.dumps(value))
    listing = root / 'pack-reports.json'
    refs = json.loads(listing.read_text())
    refs[0] = file_ref(proof)
    listing.write_text(json.dumps(refs))
    full = json.loads(census.read_text())
    full['pack_reports'] = file_ref(listing)
    census.write_text(json.dumps(full))
    path = Path(plan['sources'][0]['generation_report']['path'])
    gen = json.loads(path.read_text())
    gen['payload_report'] = file_ref(census)
    path.write_text(json.dumps(gen))
    plan['sources'][0]['generation_report'] = file_ref(path)
    with pytest.raises(ValueError, match='external quality flag was removed'):
        build_accounting(plan, root / 'account')


def test_finalizer_inherits_completed_reviews_but_never_invents_missing_review(tmp_path):
    root, plan, census, source_refs = snapshot_fixture(tmp_path)
    path = Path(source_refs[1]['path'])
    source = json.loads(path.read_text())
    source['previous_source_review'] = None
    path.write_text(json.dumps(source))
    source_refs[1] = file_ref(path)
    snapshot_ref = checked_json(plan['scope'])['snapshot']
    stage = root / 'stage.json'
    publish_json(stage, dict(stage='snapshot_training_scope_prepared_not_accepted',
        snapshot=snapshot_ref, scope=plan['scope'], inventory=plan['inventory'],
        inventory_report=plan['inventory_report'], exclusions=plan['exclusions'], sources=source_refs))
    result = finalize(stage, census, root / 'finalized')
    assert result['inherited_source_reviews'] == 1
    assert result['pending_source_reviews'] == ['coco2017']
    assert not result['training_admission'] and result['optimizer_updates'] == 0
    prepared = checked_json(result['plan'])
    assert prepared['sources'][0]['review'] is not None and prepared['sources'][1]['review'] is None
