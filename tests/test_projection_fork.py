from copy import deepcopy
import json

import pytest

from radio_moge.corpus_admission import file_ref
from radio_moge.data import manifest_digest, read_manifest
from radio_moge.projection_fork import validate_retained_validation, normalize_projection_validation


def rows(tmp_path):
    result = []
    for i, source in enumerate(('irs', 'coco2017', 'hypersim')):
        p = tmp_path / f'{i}.rgb'
        p.write_bytes(bytes([i]))
        result.append(dict(image=str(p), dataset=source, group=source))
    return result


def test_only_exact_ordered_source_subsequence_and_actual_file_digests(tmp_path):
    old = rows(tmp_path)
    new = [old[0], old[2]]
    before = deepcopy((old, new))
    validate_retained_validation(old, new, manifest_digest(old), manifest_digest(new))
    assert (old, new) == before
    for bad in (new[::-1], new[:1], old, [dict(new[0], group='changed'), new[1]]):
        with pytest.raises(ValueError, match='every non-COCO'):
            validate_retained_validation(old, bad, manifest_digest(old), manifest_digest(bad))
    with pytest.raises(ValueError, match='runtime identities'):
        validate_retained_validation(old, new, 'wrong', manifest_digest(new))


def test_scoped_normalization_changes_only_val_on_copies(tmp_path, monkeypatch):
    from radio_moge import projection_fork
    base = rows(tmp_path)
    # Exact experiment counts, unique records/files, no image decoding.
    old = []
    for i in range(1268):
        p = tmp_path / f'input-{i}'
        p.write_bytes(b'x')
        old.append(dict(base[1 if i >= 1012 else 0], image=str(p)))
    old_path, new_path = tmp_path / 'old.jsonl', tmp_path / 'new.jsonl'
    old_path.write_text(''.join(json.dumps(r)+'\n' for r in old))
    new_path.write_text(''.join(json.dumps(r)+'\n' for r in old[:1012]))
    old_digest, new_digest = [manifest_digest(read_manifest(p)) for p in (old_path, new_path)]
    monkeypatch.setattr(projection_fork, 'projection_plan', lambda _: dict(validation=file_ref(new_path)))
    config = dict(data=dict(val_manifest=str(new_path), train_manifest='new-train'))
    state = dict(config=dict(data=dict(val_manifest=str(old_path))), data_digests=dict(val=old_digest))
    digests = dict(train='new', val=new_digest, lidar='untouched')
    before = deepcopy((config, state, digests))
    normalized, normalized_digests = normalize_projection_validation(config, state, digests, {})
    assert normalized['data'] == dict(val_manifest=str(old_path), train_manifest='new-train')
    assert normalized_digests == dict(train='new', val=old_digest, lidar='untouched')
    assert (config, state, digests) == before
    with pytest.raises(ValueError, match='runtime identities'):
        normalize_projection_validation(config, state, dict(digests, val='wrong'), {})


def test_unscoped_fork_never_relaxes_validation():
    config, digests = dict(data=dict(val_manifest='unchanged')), dict(val='actual')
    result = normalize_projection_validation(config, {}, digests, {})
    assert result == (config, digests)
    assert result[0] is not config and result[1] is not digests


def test_only_exact_slot_preserving_performance_wrapper_is_permitted(tmp_path):
    from pathlib import Path
    from run_feature_reference import original_component_matches
    current = Path(__file__).resolve().parents[1] / 'src/radio_moge/performance.py'
    old = tmp_path / 'performance.py'
    source = current.read_text().replace('from .rgb_geometry_batch import concatenate_rgb\n', '')
    source = source.replace('concatenate_rgb(selected)', 'torch.cat(selected, dim=0)')
    old.write_text(source)
    assert original_component_matches(old, current, True)
    assert not original_component_matches(old, current, False)
    old.write_text(source.replace('compute_batch // size', 'compute_batch // size + 1'))
    assert not original_component_matches(old, current, True)


def test_ordinary_scope_binds_parent_manifest_sidecar_and_all_paired_keys(tmp_path, monkeypatch):
    from radio_moge import projection_fork as module
    def save(name, value):
        p = tmp_path / name
        p.write_text(json.dumps(value))
        return file_ref(p)
    pair_ref = dict(path='/pinned-pair', sha256='a'*64)
    keys = [[8, 43, 480, 640], [9, 44, 400, 640]]
    common = dict(source_checkpoint=dict(path='/archive', sha256='b'*64, payload_reread=False),
        manifest=dict(path='/rgb', sha256='c'*64), candidate_geometry_index=dict(path='/index', sha256='d'*64),
        planned_inputs=save('inputs.json', [dict(index=i, augmentation_seed=s, height=h, width=w)
                                         for i, s, h, w in keys]), common=dict(population=1597911))
    sidecar = dict(projection_comparison=pair_ref, index=common['candidate_geometry_index'], training_admission=False)
    binding = dict(projection_comparison=pair_ref, manifest=common['manifest'],
        source_checkpoint={k: common['source_checkpoint'][k] for k in ('path', 'sha256')},
        sidecar=save('sidecar.json', sidecar), planned_keys=save('keys.json', keys))
    monkeypatch.setattr(module, 'projection_comparison', lambda _: common)
    monkeypatch.setattr(module, 'projection_plan', lambda _: common)
    _, population = module.ordinary_projection_scope(binding)
    assert population == 1597911
    for field in ('manifest', 'source_checkpoint'):
        with pytest.raises(ValueError, match='paired comparison'):
            module.ordinary_projection_scope(dict(binding, **{field: dict(path='/wrong', sha256='e'*64)}))
    bad = dict(binding, sidecar=save('wrong-sidecar.json', dict(sidecar, training_admission=True)))
    with pytest.raises(ValueError, match='paired comparison'):
        module.ordinary_projection_scope(bad)
    with pytest.raises(ValueError, match='paired feature arm'):
        module.ordinary_projection_scope(dict(binding, planned_keys=save('wrong-keys.json', keys[::-1])))
