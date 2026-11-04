"""Check the crop extraction against archived code and saved real RGB hashes.

CPU RGB reads only. Reuses a completed reference's observed pixel identities;
does not rerun its model gate, evaluation, checkpoint reads or optimizer.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.data import TrainingImages, ordinary_crop_parameters, oriented_rgb
from radio_moge.geometry_cache import publish_json
from radio_moge.rgb_input_audit import pixels_sha


class ProbeRecords(dict):
    verify_on_read = True


def selected_rows(manifest_ref, expected):
    """One sequential bounded-memory pass; no stat of the full RGB collection."""
    indices = {p['key'][0] for p in expected}
    if any(type(i) is not int or i < 0 for i in indices):
        raise ValueError('Invalid reference RGB indices')
    path = Path(manifest_ref['path']).resolve()
    digest, selected, count = hashlib.sha256(), ProbeRecords(), 0
    with path.open('rb') as file:
        for line in file:
            digest.update(line)
            if not line.strip():
                continue
            if count in indices:
                row = json.loads(line)
                row['image'] = str((path.parent / row['image']).resolve())
                selected[count] = row
            count += 1
    if digest.hexdigest() != manifest_ref['sha256'] or set(selected) != indices:
        raise ValueError('Frozen manifest identity or requested rows differ')
    return selected, count


def run(job, output):
    evidence = checked_json(job['reference_pixels'])
    expected = evidence['actual_inputs']
    if not 1 <= len(expected) <= 512:
        raise ValueError('Expected bounded completed reference pixel sample')
    rows, scope = selected_rows(job['manifest'], expected)
    archived_path = checked_path(job['archived_data'])
    spec = importlib.util.spec_from_file_location('archived_rgb_pixel_check', archived_path)
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    current = TrainingImages(rows, True, compact_transport=True)
    previous = archived.TrainingImages(rows, True, compact_transport=True)
    records, sources, resolutions = [], Counter(), Counter()
    for item in expected:
        key = item['key']
        row = rows[key[0]]
        if any(row[k] != item[k] for k in ('image', 'sha256', 'dataset')):
            raise ValueError('Actual reference RGB identity differs from frozen row')
        a, b = current[key], previous[key]
        checksum = pixels_sha(a)
        if not torch.equal(a, b) or checksum != item['compact_rgb_sha256']:
            raise ValueError('New RGB pixels differ from archived code or observed reference')
        image = oriented_rgb(row, verify_hash=True)
        box, flip = ordinary_crop_parameters((image.height, image.width), tuple(key[2:]), key[1])
        records.append(dict(key=key, image=row['image'], sha256=row['sha256'],
            dataset=row['dataset'], crop_box=list(box), flipped=flip,
            oriented_rgb_shape=[image.height, image.width], compact_rgb_sha256=checksum))
        sources[row['dataset']] += 1
        resolutions[f'{key[2]}x{key[3]}'] += 1
    # Bind small inputs again after all pixel reads. The manifest was hashed in
    # the single sequential selection pass; no repeated full-corpus inventory.
    checked_path(job['reference_pixels'])
    checked_path(job['archived_data'])
    report = dict(stage='ordinary_rgb_pixel_compatibility_passed', presentations=len(records),
        frozen_manifest_rows=scope, sources=dict(sources), resolutions=dict(resolutions), records=records,
        manifest=job['manifest'], reference_pixels=job['reference_pixels'], archived_data=job['archived_data'],
        current_data=file_ref(Path(__file__).resolve().parents[1] / 'src/radio_moge/data.py'),
        verifier=file_ref(Path(__file__)), current_equals_archived_pixels=True,
        current_equals_saved_reference_hashes=True, encoded_rgb_hashes_checked=True,
        model_inference_performed=False, checkpoint_payloads_read=False,
        geometry_branch_executed=False, optimizer_updates=0, training_admission=False)
    publish_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run(json.loads(args.job.read_text()), args.output)
    print(json.dumps({k: result[k] for k in ('stage', 'presentations', 'sources', 'resolutions')}))
