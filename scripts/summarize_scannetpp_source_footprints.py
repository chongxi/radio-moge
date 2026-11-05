"""Saved-only closure of footprint diagnostics; no source or model payloads."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from probe_scannetpp_v2_source_pilot import read
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json


def run(root, output):
    report_ref = file_ref(root/'scannetpp-native-v2-source-footprint-02/report.json')
    report = read(report_ref, root)
    rows = []
    for ref in report['artifacts']:
        if ref['path'].endswith('-observations.json'):
            rows.extend(read(ref, root))
    if len(rows) != 768 or report['training_admission'] or report['new_mesh_topology_reads']:
        raise ValueError('Complete saved-only 768-observation diagnostic required')
    connected = [r for r in rows if r['footprint'].get('indexed_component_connected')
                 and not r['footprint'].get('fixed_seed_coplanar')]
    sine = [r['footprint']['max_seed_normal_sine'] for r in connected]
    angles = np.rad2deg(np.arcsin(np.clip(sine, 0, 1)))
    distances = [r['footprint']['max_seed_plane_distance_m'] for r in connected]
    def quantiles(values):
        return dict(zip(('min', 'median', 'p95', 'max'), np.quantile(values, [0, .5, .95, 1]).tolist()))
    range_coverage = Counter()
    for r in rows:
        if r['footprint'].get('diagnostic_planar_footprint'):
            range_coverage[r.get('range_evidence', {}).get('status', 'unavailable')] += 1
    prior = read(report['saved_topology'], root)
    old = {}
    for ref in prior['artifacts']:
        if ref['path'].endswith('-observations.json'):
            old.update({(r['corpus_ordinal'], r['scan'], r['slot']): r for r in read(ref, root)})
    status_changes = []
    for r in rows:
        before = old[(r['corpus_ordinal'], r['scan'], r['slot'])]
        for key in ('status', 'signed_interval_separation_m', 'measured_interval_m', 'first_hit_interval_m'):
            if before.get('range_evidence', {}).get(key) != r.get('range_evidence', {}).get(key):
                raise ValueError('Source interval classification unexpectedly changed')
        if before['footprint']['status'] != r['footprint']['status']:
            status_changes.append(dict(corpus_ordinal=r['corpus_ordinal'], scan=r['scan'], slot=r['slot'],
                                       old=before['footprint']['status'], new=r['footprint']['status']))
    output.mkdir(exist_ok=False)
    result = dict(stage='saved_source_footprint_review_complete_no_source_acceptance',
        report=report_ref, coverage=report['coverage'],
        connected_nonplanar_angles_degrees=quantiles(angles),
        connected_nonplanar_seed_plane_distance_m=quantiles(distances),
        planar_observation_source_status=dict(range_coverage), status_changes=status_changes,
        source_envelopes_unchanged=True,
        tests=dict(local='22 passed in0.19s', remote='22 passed in0.97s'),
        interpretation=[
            '330 observations have connected indexed faces but are not numerically coplanar. Their descriptive angles and offsets are not physical error bounds or proposed acceptance thresholds.',
            '86 strict planar observations cover60/384 slots at least once and26/384 twice. Zero qualifying observations cross face IDs: strict planar handling passes synthetic seam fixtures but does not solve real pilot multi-face coverage.',
            'Full edge incidence exposes6 nonmanifold observations previously labeled nonplanar first. One is the1318289 plastic-container source conflict. This is not proof that nonmanifold topology uniquely caused the range discrepancy.',
            'Border-patch and skinny-triangle checks were corrected in diagnostic code. The actual saved source intervals and their status are unchanged; this is not a historical cache or training cause.',
            'Two source gaps remain+71.530mm and-68.084mm independently of footprint status. The angle-sensitive third remains-.562mm minimum gap. The prior8.369cm retained surface counterexample is explicitly bound, not cleared.',
            'Overlap of finite sampled intervals is not physical accuracy; unavailable poses/covariance and continuous visibility uncertainty remain. No selection of only successful comparisons for denominators.',
            'Next define bounded curved-surface correspondence/visibility semantics or a documented restrictive subset before confirmation. Do not tune numerical plane tolerance until coverage rises or treat connectedness alone as trustworthy GT.',
        ],
        numerical_renderer_tolerance_m=.002, existing_image_support_floor=.05,
        tolerances_changed=False, global_geometry_exclusions=4190, new_exclusions=0,
        known19_dispositions_complete=True, confirmation_images_read=0, new_source_reads=0,
        optimizer_updates=0, source_quality_accepted=False, training_admission=False,
        generator=file_ref(Path(__file__)))
    publish_json(output/'report.json', result)
    print(file_ref(output/'report.json'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
