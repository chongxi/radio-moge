from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from radio_moge import data
from radio_moge.corpus_admission import file_ref
from radio_moge.rgb_input_audit import pixels_sha
from verify_ordinary_rgb_pixels import run, selected_rows


def fixture(tmp_path):
    records, expected = [], []
    for i in range(3):
        image = tmp_path / f'{i}.png'
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (17, 23, 3), dtype=np.uint8)).save(image)
        records.append(dict(image=str(image), sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                            dataset=f'fixture{i}', image_rotation_cw=i * 90))
    images = data.TrainingImages(records, True, compact_transport=True)
    for index, seed, h, w in ((2, 43, 16, 24), (0, 44, 24, 16), (2, 45, 16, 24)):
        key = [index, seed, h, w]
        expected.append(dict(key=key, **{k: records[index][k] for k in ('image', 'sha256', 'dataset')},
                             compact_rgb_sha256=pixels_sha(images[key])))
    manifest, evidence = tmp_path / 'rgb.jsonl', tmp_path / 'observed.json'
    manifest.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    evidence.write_text(json.dumps(dict(actual_inputs=expected)))
    # This unit fixture checks the verifier, not historical code equivalence.
    return dict(manifest=file_ref(manifest), reference_pixels=file_ref(evidence),
                archived_data=file_ref(Path(data.__file__))), expected


def test_sparse_selection_keeps_original_indices_and_repeated_presentations(tmp_path):
    job, expected = fixture(tmp_path)
    rows, count = selected_rows(job['manifest'], expected)
    assert count == 3 and set(rows) == {0, 2}
    report = run(job, tmp_path / 'report.json')
    assert report['presentations'] == 3 and report['frozen_manifest_rows'] == 3
    assert report['sources'] == {'fixture2': 2, 'fixture0': 1}
    assert not report['geometry_branch_executed'] and not report['training_admission']
    assert [r['key'] for r in report['records']] == [r['key'] for r in expected]


@pytest.mark.parametrize('mutation', ['manifest', 'input_bytes', 'saved_pixels', 'row_identity'])
def test_changed_evidence_cannot_publish_success(tmp_path, mutation):
    job, expected = fixture(tmp_path)
    if mutation == 'manifest':
        with Path(job['manifest']['path']).open('a') as file:
            file.write('\n')
    elif mutation == 'input_bytes':
        Image.fromarray(np.zeros((17, 23, 3), np.uint8)).save(expected[0]['image'])
    else:
        changed = deepcopy(expected)
        changed[0]['compact_rgb_sha256' if mutation == 'saved_pixels' else 'dataset'] = 'wrong'
        path = tmp_path / 'changed.json'
        path.write_text(json.dumps(dict(actual_inputs=changed)))
        job['reference_pixels'] = file_ref(path)
    output = tmp_path / 'report.json'
    with pytest.raises(ValueError):
        run(job, output)
    assert not output.exists()
