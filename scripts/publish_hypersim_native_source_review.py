"""Bind completed native Hypersim evidence and reconcile all final source routes.

Source eligibility does not replace full-corpus payload/coverage/heldout/model
gates. Previously flagged native labels are never repaired or selected here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from audit_hypersim_outcomes import outcome_kind
from publish_coco_nyu_source_reviews import MANIFEST, write_lines
from radio_moge.corpus_admission import (SOURCE_CHECKS, checked_json, checked_path,
    file_ref, readonly_db, validate_source_review)
from radio_moge.geometry_cache import canonical_json, publish_json
from radio_moge.geometry_sampling import source_supervision


def run(root, output):
    paths = ('hypersim-outcomes-01/report.json', 'hypersim-combined-routes-01/report.json',
        'hypersim-native-quality-01/report.json', 'hypersim-native-near-depth-01/report.json',
        'hypersim-native-scale-probe-01/report.json', 'hypersim-native-extreme-review-01/report.json',
        'hypersim-camera-audit-01/report.json', 'hypersim-tilt-probe-02/report.json',
        'hypersim-cache-preflight-03/report.json', 'hypersim-source-mismatch-probe-01/visual-review-01.json',
        'teacher-source-reviews-02/hypersim-source-review.json')
    refs = {p: file_ref(root / p) for p in paths}
    evidence = {p: checked_json(ref) for p, ref in refs.items()}
    native, routes, quality, near, scale, extreme, cameras, tilt, preflight, visual, teacher = (
        evidence[p] for p in paths)
    if (refs[paths[2]]['sha256'] != 'e6ee6d323b7cfdddfc6c5b593840222108e0b20d102910f6a54ded75675e6579'
            or refs[paths[5]]['sha256'] != 'a2a50a21d27d736fde7fa3236ed5854c4885831b0aea531c58b35e9adf8060e5'
            or refs[paths[10]]['sha256'] != '436ef7bc6d4d3a0bcbababc6ea116b999a17974739fe537e92df8672f538333a'
            or native['manifest_sha256'] != MANIFEST or native['expected_scope'] != 59344
            or native['outcome_counts'] != {'unflagged_cache': 55663, 'flagged_cache': 3076, 'rejected': 605}
            or native['producer_array_readback_complete'] is not True
            or routes['route_counts'] != {'native_candidate': 55663, 'teacher_candidate': 3680, 'reviewed_input_exclusion': 1}
            or quality['cached_counts'] != {'unflagged_cache': 55663, 'flagged_cache': 3076}
            or preflight['stage'] != 'complete' or preflight['passed'] is not True
            or preflight['policy_sha256'] != native['policy_sha256'] or preflight['verified_cache_members'] != 15
            or cameras['stage'] != 'complete' or cameras['frames'] != 59344
            or cameras['pose_rejected'] != 32 or tilt['passes'] is not True
            or visual['stage'] != 'diagnostic_visual_review_complete'):
        raise ValueError('Complete native review evidence differs')
    teacher_packs = checked_json(teacher['candidate_quarantine'])['packs_index']
    validate_source_review(teacher, 'hypersim', teacher['policy_sha256'], MANIFEST, teacher_packs)
    generation = checked_json(native['generation_report'])
    packs = checked_json(native['packs'])
    if (generation['stage'] != 'complete' or generation['mode'] != 'generate'
            or generation['completed'] != 58739 or generation['verified_cache_members'] != 58739
            or generation['training_started'] or generation['training_admission'] or generation['optimizer_updates']
            or packs['count'] != 58739 or packs['policy_sha256'] != native['policy_sha256']):
        raise ValueError('Native producer did not finish full readback')
    with readonly_db(checked_path(native['index'])) as db:
        originals = {r[0]: r[1:] for r in db.execute(
            'SELECT ordinal,row_json,outcome,evidence_json,entry_key FROM images ORDER BY ordinal')}
    by_key = {v[-1]: (i, *v) for i, v in originals.items() if v[1] != 'rejected'}
    external = checked_json(extreme['candidate_quarantine'])
    external_keys = {r['entry_key'] for r in external['flags']}
    retained, original_flags, groups = {}, set(), Counter()
    for item in packs['packs']:
        receipt_ref = dict(path=str(Path(native['packs']['path']).parent / item['receipt']), sha256=item['sha256'])
        receipt = checked_json(receipt_ref)
        pack = Path(receipt_ref['path']).parent / receipt['tar']
        if (receipt['policy_sha256'] != native['policy_sha256'] or len(receipt['entries']) != item['count']
                or pack.stat().st_size != receipt['bytes']):
            raise ValueError('Native receipt identity or pack size changed')
        for entry in receipt['entries']:
            found = by_key.pop(entry['key'], None)
            if found is None:
                raise ValueError('Duplicate or unknown native member')
            ordinal, encoded, kind, proof, _ = found
            row, meta = dict(entry['image']), entry['geometry_metadata']
            row.pop('corpus_ordinal', None)
            if (row != json.loads(encoded) or json.loads(proof) != receipt_ref
                    or outcome_kind(meta) != kind or meta['policy_sha256'] != native['policy_sha256']
                    or meta['image_sha256'] != row['sha256'] or meta['dataset'] != 'hypersim'
                    or hashlib.sha256(canonical_json(meta)).hexdigest() != entry['key']):
                raise ValueError('Native frozen RGB, metadata or flag differs')
            if kind == 'flagged_cache':
                original_flags.add(entry['key'])
                continue
            if entry['key'] in external_keys:
                continue
            h, w = meta['shape']
            if (source_supervision(meta) != 'metric_native' or meta['units'] != 'meters'
                    or meta['depth_convention'] != 'camera_z' or meta['label_kind'] != 'synthetic_renderer'
                    or meta['valid_pixels'] < 32 or meta['valid_pixels'] / (h * w) != entry['valid_fraction']
                    or entry['valid_fraction'] < .05):
                raise ValueError('Native supervision or original support differs')
            retained[ordinal] = row
            groups[row['group']] += 1
    if by_key or len(retained) != 55462 or len(original_flags) != 3076 or original_flags & external_keys:
        raise ValueError('Native candidate accounting did not reconcile')
    output.mkdir(exist_ok=False)
    documentation = dict(stage='primary_source_review_complete', references=[
        dict(url='https://arxiv.org/pdf/2011.02523', finding='The Hypersim paper describes artist-authored rendered scenes and geometry, including a depth-distribution analysis. Population statistics do not validate a particular extreme scene.'),
        dict(url='https://github.com/apple-aiml-research/ml-hypersim/blob/main/README.md', finding='Official source uses Euclidean ray distance in meters, world positions in scene asset units and camera-specific geometry; frozen RGB here is lossy tonemapped preview, not recommended HDR input.'),
        dict(url='https://raw.githubusercontent.com/apple/ml-hypersim/main/code/python/tools/generate_hdf5_from_vrimg.py', finding='Position and radial distance share renderer provenance before separate FP16 storage. Numerical consistency is not independent physical measurement.')],
        calibrated_real_sensor_accuracy_claimed=False, HDR_equivalence_claimed=False,
        training_admission=False, optimizer_updates=0)
    publish_json(output / 'documentation-review.json', documentation)
    exclusions_ref = extreme['known_geometry_exclusions']
    exclusions = {r['corpus_ordinal']: r for r in (json.loads(line) for line in checked_path(exclusions_ref).read_text().splitlines()) if r['dataset'] == 'hypersim'}
    if len(exclusions) != 204:
        raise ValueError('Hypersim exclusions no longer reconcile')
    final, counts, seen = [], Counter(), set()
    for line in checked_path(routes['routes']).read_text().splitlines():
        row = json.loads(line)
        ordinal = row['corpus_ordinal']
        if ordinal not in originals or ordinal in seen:
            raise ValueError('Combined route is duplicate or outside the source')
        original = json.loads(originals[ordinal][0])
        if any(row[k] != original[k] for k in ('image', 'sha256', 'group')):
            raise ValueError('Combined route changed frozen RGB')
        seen.add(ordinal)
        if ordinal in exclusions:
            decision = exclusions[ordinal]
            if any(decision[k] != original[k] for k in ('image', 'sha256', 'group')):
                raise ValueError('Geometry exclusion changed frozen RGB')
            disposition = 'excluded_geometry'
        elif row['route'] == 'native_candidate' and ordinal in retained:
            disposition = 'native_source_eligible'
        elif row['route'] == 'teacher_candidate':
            disposition = 'relative_teacher_source_eligible'
        else:
            raise ValueError('A native source outcome is unresolved')
        counts[disposition] += 1
        final.append(dict(row, source_disposition=disposition, source_flags_cleared=False,
            decision_evidence=exclusions_ref if disposition == 'excluded_geometry' else refs[paths[10]]
                if disposition == 'relative_teacher_source_eligible' else extreme['candidate_quarantine']))
    if (seen != set(originals) or counts != Counter(native_source_eligible=55462,
            relative_teacher_source_eligible=3678, excluded_geometry=204)):
        raise ValueError('Every original Hypersim input must have exactly one final source route')
    write_lines(output / 'source-routes.jsonl', final)
    review = dict(stage='source_review_complete', decision='eligible_with_flagged_rows_quarantined',
        dataset='hypersim', policy_sha256=native['policy_sha256'], manifest_sha256=MANIFEST,
        checks=dict.fromkeys(SOURCE_CHECKS, True), evidence=[*refs.values(),
            native['generation_report'], native['packs'], native['index'], extreme['visual_review'],
            exclusions_ref, file_ref(output / 'documentation-review.json'), file_ref(output / 'source-routes.jsonl')],
        candidate_quarantine=extreme['candidate_quarantine'], label_kinds=['synthetic_renderer'],
        supervision='metric_native', eligible_candidate_images=55462,
        original_native_flagged_candidates=3076, external_quarantined_candidates=201,
        quarantined_candidate_images=3277, selected_source_groups=len(groups),
        check_details=dict(
            documentation_and_paper='Paper and official formats/exporter reviewed; synthetic geometry and lossy RGB limitations recorded.',
            units_and_camera='Source range-to-Z in exact RQ frame, explicit asset/meter conversion and verified pose normalization. All camera frames audited. The source-scale concern scene is explicitly excluded, never rescaled.',
            valid_support='Full producer arrays reread; all retained original masks exceed frozen support policy. No holes filled, invalid pixels remain invalid. Crop coverage remains a separate final gate.',
            geometric_quality='All native receipt identities/provenance rechecked. Original XYZ inconsistency and pose flags remain quarantined; the scale/precision additions have exact exclusions. Original rejected/flagged images have separately reviewed relative teacher routes or exclusions.',
            visual_quality='Earlier default/extreme-tilt preflight and 29-source-mismatch diagnostics retained; all 12 new extreme/near pages and 8 scale-probe pages inspected (17 distinct new images). Explicit scene-scale and exact near-frame decisions applied.'),
        physical_scale_independently_measured=False, source_flags_cleared=False,
        whole_corpus_coverage_accepted=False, canonical_heldout_isolation_accepted=False,
        actual_archived_model_probe_performed=False, training_admission=False, optimizer_updates=0)
    validate_source_review(review, 'hypersim', native['policy_sha256'], MANIFEST, native['packs'])
    for ref in refs.values():
        checked_path(ref)
    publish_json(output / 'native-source-review.json', review)
    publish_json(output / 'report.json', dict(stage='hypersim_source_reviews_and_routes_complete_not_corpus_admission',
        native_source_review=file_ref(output / 'native-source-review.json'), teacher_source_review=refs[paths[10]],
        source_routes=file_ref(output / 'source-routes.jsonl'), source_disposition_counts=dict(counts),
        original_images=len(seen), native_source_groups=len(groups), candidate_quarantine=extreme['candidate_quarantine'],
        known_geometry_exclusions=exclusions_ref, original_native_flags_retained=True,
        native_receipt_identities_rechecked=58739, full_array_readback_repeated=False,
        producer_full_array_readback_bound=True, training_admission=False, optimizer_updates=0,
        code=file_ref(Path(__file__)), completed_at=time.time()))
    print(json.dumps(dict(source_disposition_counts=dict(counts), native_source_groups=len(groups))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
