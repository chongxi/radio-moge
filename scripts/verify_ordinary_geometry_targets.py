"""Bounded CPU payload/camera check on saved original RGB presentation keys.

Reuses the completed sidecar pixel proof. Does not rerun RGB comparison, model,
checkpoint or optimizer work. Reads only selected candidate payloads and RGB.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.corpus_losses import target_tensors
from radio_moge.geometry_cache import publish_json
from radio_moge.geometry_loss_camera import loss_focal
from radio_moge.ordinary_geometry_data import GeometrySidecar
from radio_moge.ordinary_geometry_targets import ordinary_target
from radio_moge.rgb_geometry_batch import GeometrySlot
from verify_ordinary_rgb_pixels import selected_rows


def run(proof_ref, output):
    proof = checked_json(proof_ref)
    if (proof['stage'] != 'ordinary_geometry_sidecar_cpu_verified_not_admitted'
            or proof['training_admission'] is not False or proof['optimizer_updates'] != 0
            or not proof['all_rgb_slots_retained'] or not proof['prepared_rgb_bitwise_equal']
            or not proof['original_observed_pixel_hashes_equal'] or not proof['torch_rng_unchanged']
            or not 1 <= proof['presentations'] <= 512 or len(proof['records']) != proof['presentations']):
        raise ValueError('Expected complete bounded original RGB sidecar proof')
    rows, count = selected_rows(proof['job']['manifest'], proof['records'])
    if count != proof['manifest_rows']:
        raise ValueError('Original RGB manifest scope changed')
    sidecar = GeometrySidecar(proof['job']['historical_acceptance'])
    records, counts, sources = [], Counter(), Counter()
    before = torch.get_rng_state().clone()
    try:
        for item in proof['records']:
            key = tuple(item['key'])
            row = rows[key[0]]
            ref = sidecar.lookup(row)
            if ((ref is not None) != item['geometry_available'] or (ref and
                    (ref['entry']['key'] != item['entry_key'] or ref['pack'] != item['pack']
                     or ref['corpus_ordinal'] != item['corpus_ordinal']))):
                raise ValueError('Geometry reference differs from the saved sidecar proof')
            target = ordinary_target(GeometrySlot(key, row, ref), augment=True)
            record = dict(key=key, dataset=row['dataset'], image=row['image'], sha256=row['sha256'])
            if target is None:
                counts['missing'] += 1
                record.update(geometry_available=False)
            else:
                valid = target['valid']
                ids = target['native_index'][valid]
                if (len(set(ids)) != len(ids) or (ids < 0).any()
                        or (ids >= np.prod(target['native_shape'])).any()):
                    raise ValueError('Invalid or repeated native target identities')
                focal = float(loss_focal(target_tensors(target, 'cpu'), ordinary=True))
                h, w = target['output_shape']
                uv = target['query_uv'][valid]
                low = np.array([.5/w, .5/h])
                if not ((uv >= low-1e-7) & (uv <= 1-low+1e-7)).all():
                    raise ValueError('Ordinary target needs prediction extrapolation')
                kind = 'sparse' if len(ids) < 32 else 'supervised'
                counts[kind] += 1
                sources[row['dataset']] += 1
                record.update(geometry_available=True, category=kind, observations=len(ids),
                    supervision=target['supervision'], label_kind=target['label_kind'],
                    cache_key=target['cache_key'], member_sha256=target['member_sha256'],
                    pack=target['pack'], corpus_ordinal=target['corpus_ordinal'],
                    local_focal=focal, intrinsics=target['intrinsics'].tolist(),
                    crop_box=target['ordinary_view']['crop_box'], flipped=target['ordinary_view']['flipped'],
                    camera_contract=target['camera_contract'], local_radius_contract=target['local_radius_contract'],
                    arrays={k: hashlib.sha256(np.ascontiguousarray(target[k]).tobytes()).hexdigest()
                            for k in ('points', 'valid', 'native_index', 'query_uv')})
            records.append(record)
    finally:
        sidecar.close()
    if not torch.equal(before, torch.get_rng_state()):
        raise ValueError('Target preparation changed global Torch RNG')
    root = Path(__file__).resolve().parents[1]
    names = ('ordinary_geometry_targets.py', 'ordinary_geometry.py', 'geometry_sampling.py',
             'geometry_loss_camera.py', 'ordinary_geometry_data.py', 'corpus_losses.py')
    report = dict(stage='ordinary_target_camera_cpu_verified_not_admitted', input_proof=proof_ref,
        presentations=len(records), counts=dict(counts), payload_sources=dict(sources), records=records,
        implementation={name: file_ref(root/'src/radio_moge'/name) for name in names},
        verifier=file_ref(Path(__file__)), geometry_member_payloads_read=sum(sources.values()),
        full_pack_hashes_rechecked=False, geometry_member_hashes_verified=True,
        rgb_hashes_verified_for_geometry=True, camera_rays_verified=True,
        torch_rng_unchanged=True, rgb_pixel_comparison_repeated=False,
        model_inference_performed=False, training_admission=False, optimizer_updates=0)
    publish_json(output, report)
    print(json.dumps({k: report[k] for k in ('stage', 'presentations', 'counts', 'payload_sources')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('proof_reference', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.proof_reference.read_text()), args.output)
