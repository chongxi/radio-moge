from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import pickle
import sqlite3

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from radio_moge.corpus_admission import file_ref
from radio_moge.data import DeterministicBatches, TrainingImages
from radio_moge.ordinary_geometry_data import (GeometrySidecar, GeometryAttachedImages,
                                               collate_geometry_attached)
from radio_moge.performance import grouped_window
from radio_moge.prefetch import FloatLoader, PreparedWindows, normalize_uint8
from radio_moge.rgb_geometry_batch import GeometrySlot, RGBGeometryBatch, concatenate_rgb
from radio_moge.rgb_input_audit import pixels_sha
from radio_moge.losses import FeatureStatistics, FEATURE_LOSS_KEYS, weighted_loss
from test_training import tiny_system
from verify_geometry_sidecar import run as verify_sidecar


class VerifiedRows(list):
    verify_on_read = True


def fixture(tmp_path):
    rows = VerifiedRows()
    for i in range(8):
        path = tmp_path / f'rgb-{i}.png'
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (19, 29, 3), dtype=np.uint8)).save(path)
        rows.append(dict(image=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         split='train', dataset='fixture', group=f'scene-{i//2}', image_rotation_cw=(i % 4) * 90))
    # Eight RGB records; only rows 1/3/4 have geometry. Row 6 is excluded and
    # the others are outside this historical geometry scope. Deliberately give
    # selected samples a different order and ordinals unrelated to RGB indices.
    db_path = tmp_path / 'index.sqlite'
    with sqlite3.connect(db_path) as db:
        db.executescript('''CREATE TABLE images (ordinal INTEGER PRIMARY KEY,image TEXT UNIQUE,
            sha256 TEXT UNIQUE,row_json TEXT,disposition TEXT,pack TEXT,entry_json TEXT);
            CREATE TABLE samples (sample INTEGER PRIMARY KEY,ordinal INTEGER UNIQUE);
            CREATE TABLE packs (path TEXT PRIMARY KEY,sha256 TEXT);''')
        pack = str(tmp_path / 'not-read-by-sidecar.tar')
        db.execute('INSERT INTO packs VALUES (?,?)', (pack, 'a'*64))
        for i in (1, 3, 4, 6):
            excluded = i == 6
            entry = dict(image=rows[i], key=f'{i:064x}', payload_sha256='b'*64)
            db.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?)',
                       (100+i, rows[i]['image'], rows[i]['sha256'], json.dumps(rows[i]),
                        'excluded' if excluded else 'selected', None if excluded else pack,
                        None if excluded else json.dumps(entry)))
        db.executemany('INSERT INTO samples VALUES (?,?)', enumerate((104, 101, 103)))
    return rows, bind_fixture(tmp_path, db_path)


def bind_fixture(tmp_path, db_path):
    accounting = dict(stage='accounting_verified_not_accepted', training_admission=False,
                      unresolved=0, index=file_ref(db_path), manifest_sha256='c'*64,
                      selected_images=3, total_images=4)
    path = tmp_path / 'accounting.json'
    path.write_text(json.dumps(accounting))
    gate = dict(schema_version=1, stage='accepted_for_training', training_admission=True,
                accounting_report=file_ref(path), **{k: accounting[k] for k in
                ('index', 'manifest_sha256', 'selected_images')})
    path = tmp_path / 'historical-seal.json'
    path.write_text(json.dumps(gate))
    return file_ref(path)


def test_exact_lookup_never_uses_subset_sample_number_or_drops_rgb(tmp_path):
    rows, ref = fixture(tmp_path)
    before = deepcopy(rows)
    sidecar = GeometrySidecar(ref)
    original = TrainingImages(rows, compact_transport=True)
    attached = GeometryAttachedImages(original, sidecar)
    assert len(attached) == len(original) == 8 and not sidecar.training_admission
    for i in range(8):
        key = (i, 43, 16, 24)
        image, slot = attached[key]
        assert torch.equal(image, original[key]) and slot.key == key and slot.rgb_row == rows[i]
        if i in (1, 3, 4):
            assert slot.geometry['corpus_ordinal'] == 100+i
            assert slot.geometry['entry']['image'] == rows[i]
        else:
            assert slot.geometry is None
    assert rows == before
    sidecar.close()


@pytest.mark.parametrize('field,value', [('sha256', 'd'*64), ('dataset', 'other'),
                                        ('group', 'other'), ('image_rotation_cw', 90)])
def test_known_path_identity_change_is_not_silently_missing_geometry(tmp_path, field, value):
    rows, ref = fixture(tmp_path)
    for i in (4, 6):  # Selected AND excluded known paths must fail on changed identity.
        with pytest.raises(ValueError, match='Known RGB path'):
            GeometrySidecar(ref).lookup(dict(rows[i], **{field: value}))


def test_absent_path_never_joins_by_hash_alone_and_missing_hash_fails(tmp_path):
    rows, ref = fixture(tmp_path)
    sidecar = GeometrySidecar(ref)
    assert sidecar.lookup(dict(rows[1], image=str(tmp_path / 'same-sha-other-path.png'))) is None
    row = dict(rows[1])
    row.pop('sha256')
    with pytest.raises(ValueError, match='exact training RGB'):
        sidecar.lookup(row)


def test_changed_index_or_disposition_and_stale_acceptance_rejected(tmp_path):
    rows, ref = fixture(tmp_path)
    sidecar = GeometrySidecar(ref)
    db_path = tmp_path / 'index.sqlite'
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE images SET disposition='unresolved' WHERE ordinal=101")
    with pytest.raises(ValueError, match='changed after verification'):
        sidecar.lookup(rows[1])
    with pytest.raises(ValueError, match='Evidence changed'):
        GeometrySidecar(ref)
    ref = bind_fixture(tmp_path, db_path)
    with pytest.raises(ValueError, match='counts or dispositions'):
        GeometrySidecar(ref)


@pytest.mark.parametrize('mutation', ['entry', 'pack', 'membership', 'accounting'])
def test_inconsistent_bound_index_metadata_fails(tmp_path, mutation):
    rows, ref = fixture(tmp_path)
    db_path = tmp_path / 'index.sqlite'
    with sqlite3.connect(db_path) as db:
        if mutation == 'entry':
            value = dict(image=rows[3])
            db.execute('UPDATE images SET entry_json=? WHERE ordinal=101', (json.dumps(value),))
        elif mutation == 'pack':
            db.execute('DELETE FROM packs')
        elif mutation == 'membership':
            db.execute('UPDATE samples SET ordinal=106 WHERE sample=0')
    ref = bind_fixture(tmp_path, db_path)
    if mutation == 'accounting':
        path = tmp_path / 'historical-seal.json'
        gate = json.loads(path.read_text())
        gate['selected_images'] += 1
        path.write_text(json.dumps(gate))
        ref = file_ref(path)
    with pytest.raises(ValueError):
        GeometrySidecar(ref).lookup(rows[1])


def test_connections_are_readonly_thread_local_and_not_pickled(tmp_path):
    rows, ref = fixture(tmp_path)
    sidecar = GeometrySidecar(ref)
    parent_db = sidecar.db
    with pytest.raises(sqlite3.OperationalError):
        parent_db.execute('DELETE FROM images')
    def worker():
        assert sidecar.db is not parent_db
        result = sidecar.lookup(rows[1])
        sidecar.close()
        return result
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(worker).result()['corpus_ordinal'] == 101
    cloned = pickle.loads(pickle.dumps(sidecar))
    assert cloned.db is not parent_db
    assert cloned.lookup(rows[4])['corpus_ordinal'] == 104
    cloned.close()
    sidecar.close()
    assert sidecar.lookup(rows[3])['corpus_ordinal'] == 103


def loader(rows, sidecar, workers=0, start=0, count=16):
    images = TrainingImages(rows, compact_transport=True)
    if sidecar is not None:
        images = GeometryAttachedImages(images, sidecar)
    return FloatLoader(DataLoader(images,
        batch_sampler=DeterministicBatches(len(rows), 4, [(16, 24), (24, 16), (16, 16)], 43, start, count),
        num_workers=workers, generator=torch.Generator().manual_seed(43),
        collate_fn=collate_geometry_attached if sidecar is not None else None))


@pytest.mark.parametrize('workers', [0, 1])
@pytest.mark.parametrize('prepared', [False, True])
def test_full_update_grouping_prefetch_and_resume_preserve_pixels_slots_and_none(tmp_path, workers, prepared):
    rows, ref = fixture(tmp_path)
    sidecar = GeometrySidecar(ref)
    # An already-open parent connection must not be reused by a worker fork.
    assert sidecar.lookup(rows[1]) is not None
    def collect(start, updates):
        a, b = loader(rows, None, workers, start, updates*8), loader(rows, sidecar, workers, start, updates*8)
        plain = PreparedWindows(a, 8, 16, updates) if prepared else iter(a)
        attached = PreparedWindows(b, 8, 16, updates) if prepared else iter(b)
        result = []
        try:
            sampler = iter(DeterministicBatches(8, 4, [(16, 24), (24, 16), (16, 16)], 43, start, updates*8))
            for _ in range(updates):
                aa = next(plain) if prepared else list(grouped_window(plain, 8, 16))
                bb = next(attached) if prepared else list(grouped_window(attached, 8, 16))
                keys = [key for _ in range(8) for key in next(sampler)]
                by_shape = {}
                for key in keys:
                    by_shape.setdefault(tuple(key[2:]), []).append(key)
                expected = [group[i:i+16] for group in by_shape.values() for i in range(0, len(group), 16)]
                assert [len(t) for t in aa] == [len(t) for t in bb] == list(map(len, expected))
                update = []
                for original, batch, key_group in zip(aa, bb, expected, strict=True):
                    assert torch.equal(original, batch.image)
                    assert [s.key for s in batch.slots] == key_group
                    assert [s.geometry is None for s in batch.slots] == [k[0] not in (1, 3, 4) for k in key_group]
                    assert batch.to('cpu').slots is batch.slots
                    update.append((tuple(s.key for s in batch.slots), batch.image.clone()))
                assert sum(len(t) for t in bb) == 32
                result.append(update)
        finally:
            for iterator in (plain, attached):
                if prepared:
                    iterator.close()
                else:
                    iterator._shutdown_workers()
        return result
    full, resumed = collect(0, 2), collect(8, 1)
    assert len(full[1]) == len(resumed[0])
    for (keys_a, image_a), (keys_b, image_b) in zip(full[1], resumed[0], strict=True):
        assert keys_a == keys_b and torch.equal(image_a, image_b)
    sidecar.close()


def test_batch_normalization_exact_and_invalid_transport_rejected():
    image = torch.arange(3*4*8, dtype=torch.uint8).reshape(1, 3, 4, 8)
    slot = GeometrySlot((0, 43, 4, 8), {}, None)
    batch = RGBGeometryBatch(image, (slot,))
    assert torch.equal(normalize_uint8(batch).image, normalize_uint8(image))
    assert normalize_uint8(batch).slots is batch.slots
    assert pixels_sha(concatenate_rgb([batch, batch]).image) == pixels_sha(torch.cat([image, image]))
    with pytest.raises(ValueError, match='mix'):
        concatenate_rgb([image, batch])
    with pytest.raises(ValueError, match='one geometry slot'):
        RGBGeometryBatch(image, ())
    with pytest.raises(ValueError, match='resolution'):
        RGBGeometryBatch(image, (GeometrySlot((0, 43, 8, 4), {}, None),))
    with pytest.raises(ValueError, match='transport'):
        batch.with_image(image[:, :, :2])


def test_unverified_rgb_is_not_silently_wrapped(tmp_path):
    rows, ref = fixture(tmp_path)
    sidecar = GeometrySidecar(ref)
    with pytest.raises(TypeError, match='hash-verified'):
        GeometryAttachedImages(TrainingImages(list(rows), compact_transport=True), sidecar)


def test_tiny_cpu_feature_losses_gradients_and_rng_match_with_sidecar(tmp_path):
    """Actual prepared RGB path and existing feature loss; no geometry loss/model gate."""
    rows, ref = fixture(tmp_path)
    sidecar = GeometrySidecar(ref)
    system, initial_adapter = tiny_system()
    statistics = FeatureStatistics(6, 6)
    statistics.spatial_scale.copy_(torch.linspace(.6, 1.4, 6))
    statistics.summary_scale.copy_(torch.linspace(.7, 1.3, 6))
    frozen = {name: deepcopy(module.state_dict()) for name, module in
              dict(radio=system.radio, encoder=system.encoder, decoder=system.decoder,
                   statistics=statistics).items()}

    def backward(metadata):
        adapter = deepcopy(initial_adapter)
        adapter.train()
        before = torch.get_rng_state().clone()
        prepared = PreparedWindows(loader(rows, metadata, count=8), 8, 16, 1)
        values = {k: torch.tensor(0.) for k in FEATURE_LOSS_KEYS}
        try:
            for batch in next(prepared):
                image = batch.image if isinstance(batch, RGBGeometryBatch) else batch
                with torch.no_grad():
                    features, target = system.encode(image, adapter.indices)
                prediction = adapter(features)
                losses = statistics.loss(prediction, target)
                weight = len(image) / 32
                (weighted_loss(losses, dict.fromkeys(FEATURE_LOSS_KEYS, 1.)) * weight).backward()
                for k in values:
                    values[k] += losses[k].detach() * weight
        finally:
            prepared.close()
        assert torch.equal(torch.get_rng_state(), before)
        gradients = {k: p.grad.clone() for k, p in adapter.named_parameters()}
        assert all(torch.isfinite(x).all() for x in gradients.values())
        assert any(torch.count_nonzero(x) for x in gradients.values())
        return values, gradients

    plain, annotated = backward(None), backward(sidecar)
    for a, b in zip(plain, annotated, strict=True):
        assert set(a) == set(b)
        assert all(torch.equal(a[k], b[k]) for k in a)
    for name, module in dict(radio=system.radio, encoder=system.encoder, decoder=system.decoder,
                             statistics=statistics).items():
        assert all(torch.equal(v, frozen[name][k]) for k, v in module.state_dict().items())
        assert all(p.grad is None for p in module.parameters())
    sidecar.close()


@pytest.mark.parametrize('corrupt_pixels', [False, True])
def test_bounded_cpu_sidecar_verifier(tmp_path, corrupt_pixels):
    rows, ref = fixture(tmp_path)
    manifest = tmp_path / 'rgb.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    images = TrainingImages(rows, compact_transport=True)
    keys = [key for batch in DeterministicBatches(8, 4, [(16, 24), (24, 16)], 43, 0, 8) for key in batch]
    saved = [dict(key=list(key), compact_rgb_sha256=pixels_sha(images[key]),
                  **{k: rows[key[0]][k] for k in ('image', 'sha256', 'dataset')}) for key in keys]
    if corrupt_pixels:
        saved[0]['compact_rgb_sha256'] = 'f'*64
    evidence = tmp_path / 'observed-reference.json'
    evidence.write_text(json.dumps(dict(actual_inputs=saved)))
    job = dict(manifest=file_ref(manifest), reference_pixels=file_ref(evidence), historical_acceptance=ref)
    output = tmp_path / 'sidecar-report.json'
    if corrupt_pixels:
        with pytest.raises(ValueError, match='original RGB slot'):
            verify_sidecar(job, output)
        assert not output.exists()
    else:
        result = verify_sidecar(job, output)
        assert result['presentations'] == 32 and result['prepared_rgb_bitwise_equal']
        assert sum(result['selected_sources'].values()) == 12
        assert sum(result['missing_sources'].values()) == 20
        assert not result['training_admission'] and not result['geometry_payloads_read']
