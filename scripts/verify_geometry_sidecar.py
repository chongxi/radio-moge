"""CPU check of real RGB/optional geometry identities through prepared grouping.

Uses saved reference input keys, not a new training sampler. No geometry payload,
model or checkpoint is read. Historical data acceptance does not admit training.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.data import TrainingImages
from radio_moge.geometry_cache import publish_json
from radio_moge.ordinary_geometry_data import GeometryAttachedImages, GeometrySidecar, collate_geometry_attached
from radio_moge.prefetch import FloatLoader, PreparedWindows
from radio_moge.rgb_input_audit import pixels_sha
from verify_ordinary_rgb_pixels import selected_rows


def run(job, output):
    saved = checked_json(job['reference_pixels'])['actual_inputs']
    if not 1 <= len(saved) <= 512 or len(saved) % 32:
        raise ValueError('Expected bounded reference inputs divisible into full updates of 32')
    rows, manifest_count = selected_rows(job['manifest'], saved)
    sidecar = GeometrySidecar(job['historical_acceptance'])
    keys = [item['key'] for item in saved]
    original = TrainingImages(rows, True, compact_transport=True)
    records, groups, selected_sources, missing_sources = [], [], Counter(), Counter()
    before = torch.get_rng_state().clone()
    windows = []

    def observe(compact_groups):
        offset = len(records)
        expected = saved[offset:offset+32]
        if len(expected) != 32:
            raise ValueError('Prepared beyond the bounded diagnostic input scope')
        by_shape = {}
        for slot, item in enumerate(expected):
            by_shape.setdefault(tuple(item['key'][2:]), []).append(slot)
        mapping = [slots[i:i+32] for slots in by_shape.values() for i in range(0, len(slots), 32)]
        if len(mapping) != len(compact_groups):
            raise ValueError('Prepared shape grouping differs from the original order')
        for batch, positions in zip(compact_groups, mapping, strict=True):
            if len(batch) != len(positions):
                raise ValueError('Prepared compute group size differs')
            for image, slot, position in zip(batch.image, batch.slots, positions, strict=True):
                item = expected[position]
                if (list(slot.key) != item['key'] or pixels_sha(image) != item['compact_rgb_sha256']
                        or any(slot.rgb_row[k] != item[k] for k in ('image', 'sha256', 'dataset'))):
                    raise ValueError('Geometry sidecar changed or mismatched the original RGB slot')
            groups.append(dict(logical_slots=[offset+p for p in positions], shape=list(batch.shape),
                               compact_tensor_sha256=pixels_sha(batch.image)))
        # Restore logical order in the evidence, irrespective of compute grouping.
        slot_by_key = {slot.key: slot for batch in compact_groups for slot in batch.slots}
        if len(slot_by_key) != 32:
            raise ValueError('Diagnostic reference keys must be unique within each update')
        for item in expected:
            slot = slot_by_key[tuple(item['key'])]
            geometry = slot.geometry
            (selected_sources if geometry else missing_sources)[item['dataset']] += 1
            records.append(dict(**item, geometry_available=geometry is not None,
                corpus_ordinal=geometry['corpus_ordinal'] if geometry else None,
                entry_key=geometry['entry']['key'] if geometry else None,
                pack=geometry['pack'] if geometry else None))

    def make(annotated):
        dataset = GeometryAttachedImages(original, sidecar) if annotated else original
        return FloatLoader(DataLoader(dataset, batch_sampler=[keys[i:i+4] for i in range(0, len(keys), 4)],
            num_workers=0, generator=torch.Generator().manual_seed(43),
            collate_fn=collate_geometry_attached if annotated else None))

    try:
        plain = PreparedWindows(make(False), 8, 32, len(keys)//32)
        windows.append(plain)
        annotated = PreparedWindows(make(True), 8, 32, len(keys)//32, observer=observe)
        windows.append(annotated)
        for _ in range(len(keys)//32):
            a, b = next(plain), next(annotated)
            if len(a) != len(b) or any(not torch.equal(x, y.image) for x, y in zip(a, b, strict=True)):
                raise ValueError('Prepared normalized RGB differs with geometry sidecar attached')
    finally:
        for window in windows:
            window.close()
        sidecar.close()
    if len(records) != len(saved) or not torch.equal(before, torch.get_rng_state()):
        raise ValueError('Incomplete diagnostic or global Torch RNG changed')
    root = Path(__file__).resolve().parents[1]
    files = ('ordinary_geometry_data.py', 'rgb_geometry_batch.py', 'data.py', 'performance.py', 'prefetch.py')
    report = dict(stage='ordinary_geometry_sidecar_cpu_verified_not_admitted',
        job=job, manifest_rows=manifest_count, presentations=len(saved), diagnostic_updates=len(saved)//32,
        logical_batch=4, compute_batch=32, effective_batch=32, records=records, groups=groups,
        selected_sources=dict(selected_sources), missing_sources=dict(missing_sources),
        prepared_rgb_bitwise_equal=True, original_observed_pixel_hashes_equal=True,
        torch_rng_unchanged=True, all_rgb_slots_retained=True,
        input_keys='Saved reference diagnostic keys; first 32 are the actual reference first update',
        missing_scope='Missing/excluded targets both retained as feature-only; not full-stream prevalence',
        implementation={p: file_ref(root/'src/radio_moge'/p) for p in files},
        verifier=file_ref(Path(__file__)), model_inference_performed=False, geometry_payloads_read=False,
        camera_loss_contract_admitted=False, training_admission=False, optimizer_updates=0)
    publish_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.job.read_text()), args.output)
    print(json.dumps({k: result[k] for k in ('stage', 'presentations', 'selected_sources', 'missing_sources')}))
