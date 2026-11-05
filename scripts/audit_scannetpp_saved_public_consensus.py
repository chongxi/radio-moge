"""Development-only re-selection from retained renders; no re-render/admission."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from audit_scannetpp_v2_saved_scanner_transfer import read_bound
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.mesh_clip import MeshClipContract, select_public_consensus
import radio_moge.mesh_clip as implementation


def run(root, output):
    def ref(p):
        return dict(path=str(p), sha256=sha_file(p))
    source = root/'scannetpp-native-v2-development-01/report.json'
    d = read_bound(ref(source), root)
    contract = MeshClipContract()
    if (d['completed'] != 35 or d['confirmation_images_read'] != 0
            or d['clip_contract'] != contract.as_dict() or d['training_admission'] is not False):
        raise ValueError('Closed unchanged development source required')
    output.mkdir(exist_ok=False)
    records = []
    for record_ref in d['records']:
        record = read_bound(record_ref, root)
        ordinal = record['corpus_ordinal']
        p = source.parent/f'{ordinal}.npz'
        if sha_file(p) != record['arrays']['sha256']:
            raise ValueError('Changed saved arrays')
        with np.load(p) as arrays:
            cpu, support, public = (arrays[k] for k in ('final_depth', 'final_support', 'raw_pyrender_depth'))
            depth, retained, reason, stats = select_public_consensus(cpu, support, public, contract)
            if (not np.array_equal(depth[retained], cpu[retained]) or (retained & ~support).any()
                    or stats['rejected_pixels'] != record['final_public_missing']+record['final_public_over_2mm']):
                raise ValueError('Consensus altered supported depth or disagreement accounting')
            target = output/f'{ordinal}.npz'
            incoming = target.with_suffix('.npz.incoming')
            with incoming.open('xb') as file:
                np.savez_compressed(file, consensus_support=retained, reason=reason)
            os.rename(incoming, target)
            records.append(dict(corpus_ordinal=ordinal, role=record['role'], original_record=record_ref,
                original_arrays=record['arrays'], selection_arrays=ref(target), statistics=stats,
                original_public_gate_passed=record['public_numerical_parity_passed'],
                minimum_support_after_selection=bool(retained.mean() >= .05),
                rejected_pixel_yx=np.argwhere(support & ~retained).tolist()))
    report = dict(stage='development_consensus_mask_candidate_not_admitted', source=ref(source),
        source_execution_unchanged=d['implementation'], selection_code=ref(Path(__file__)),
        selection_implementation=ref(Path(implementation.__file__)), records=records,
        prior_pixels=sum(r['statistics']['input_verified_pixels'] for r in records),
        retained_pixels=sum(r['statistics']['retained_pixels'] for r in records),
        rejected_pixels=sum(r['statistics']['rejected_pixels'] for r in records),
        original_gate_failure_count=sum(not r['original_public_gate_passed'] for r in records),
        development_selection_after_observing_failures=True, independent_validation_pass=False,
        confirmation_images_read=0, source_quality_accepted=False, quarantines_cleared=False,
        original_labels_changed=False, depth_values_replaced=False, new_source_or_model_reads=False,
        gpu_used=False, training_admission=False, optimizer_updates=0,
        requirement='Future production must compute and bind the public reference for every retained view. No original failed gate is relabeled as a pass. This numerical selection does not remove physically wrong but mutually consistent surfaces.')
    publish_json(output/'report.json', report)
    print(json.dumps({k: report[k] for k in ('prior_pixels', 'retained_pixels', 'rejected_pixels')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.root, args.output)
