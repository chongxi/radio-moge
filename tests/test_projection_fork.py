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
