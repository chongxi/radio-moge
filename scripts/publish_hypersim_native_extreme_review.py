"""Publish the inspected native scale/precision dispositions, without admission.

This exact-scene/exact-image decision is not a depth cutoff or inferred repair.
RGB is retained. Original native flags and earlier candidate routes stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from publish_coco_nyu_source_reviews import MANIFEST, write_lines
from radio_moge.corpus_admission import (checked_candidate_quarantine, checked_json,
                                        checked_path, file_ref)
from radio_moge.geometry_cache import publish_json


PINNED = {
    'hypersim-native-quality-01/report.json': 'e6ee6d323b7cfdddfc6c5b593840222108e0b20d102910f6a54ded75675e6579',
    'hypersim-native-near-depth-01/report.json': '6448f77361293076df71eb45562f1830883a3be0c695faf432418da42d0b0f05',
    'hypersim-native-scale-probe-01/report.json': '42ee88ea4f1c4211e0cf8c6babab4f80e7f393bf4ab7b381ba592c62995713b9',
    'coco-nyu-source-reviews-01/known-geometry-exclusions.jsonl': '737caac06f33fa2d9c1859eb6579b79b11b47a17f470687e0b31bf4770022b47',
}


def run(root, output):
    refs = {name: dict(path=str(root / name), sha256=sha) for name, sha in PINNED.items()}
    for ref in refs.values():
        checked_path(ref)
    quality, near, scale = [checked_json(refs[name]) for name in list(PINNED)[:3]]
    native = checked_json(quality['source_outcomes'])
    if (quality['source_outcomes'] != near['source_outcomes']
            or quality['source_outcomes'] != scale['source_outcomes']
            or native['manifest_sha256'] != MANIFEST
            or quality['cached_counts'] != {'unflagged_cache': 55663, 'flagged_cache': 3076}
            or scale['scope_images'] != 202 or len(scale['probes']) != 8):
        raise ValueError('Exact completed source review evidence changed')
    scope = [json.loads(line) for line in checked_path(scale['scope']).read_text().splitlines()]
    flags = [r for r in scope if r['group'] == 'hypersim:ai_013_001' or r['corpus_ordinal'] == 251659]
    expected = set(range(211112, 211312)) | {251659}
    if len(flags) != 201 or {r['corpus_ordinal'] for r in flags} != expected:
        raise ValueError('Specific scale/near-surface scope changed')
    pages = [v['page'] for report in (quality, near) for v in report['selected']]
    pages += [v['page'] for v in scale['probes']]
    for ref in pages:
        checked_path(ref)
    for probe in scale['probes']:
        if probe['reproduced_cached_depth_and_mask_exactly'] is not True:
            raise ValueError('Selected native arrays did not reproduce')
        for ref in probe['source_assets']:
            checked_path(ref)
    output.mkdir(exist_ok=False)
    review = dict(stage='native_extreme_visual_review_complete', reviewed_at=time.time(),
        source_outcomes=quality['source_outcomes'], evidence=[refs[n] for n in list(PINNED)[:3]],
        inspected_pages=pages,
        inspected_distinct_ordinals=sorted({v['corpus_ordinal'] for v in quality['selected'] + near['selected'] + scale['probes']}),
        findings=[
            dict(group='hypersim:ai_013_001', scope_images=200, decision='exclude_geometry_retain_rgb',
                reason='Scene-level metric-scale plausibility remains unverified. Both 100-frame cameras use source meters_per_asset_unit=0.009999999776482582, yet cached median Z is 277-315m for the colonnade and 170-174m for downward views of floor tiles and column bases. Six first/middle/last source-array probes reproduce the cached depth and masks. The exclusion is a conservative scene-specific curation decision, not proof of a particular correct scale or a general far-depth cutoff.'),
            dict(corpus_ordinal=251659, scope_images=1, decision='exclude_geometry_retain_rgb',
                reason='Nearly featureless close-surface RGB at cached median Z 0.000917m. Source world-position quantization is too coarse for this specific sample: median raw norm/radial relative discrepancy is 1.4075 while the existing absolute quantization gate still passes all pixels. Geometric evidence is insufficient for reliable metric supervision of this exact image; no claim that the original radial depth is physically wrong.'),
            dict(corpus_ordinal=210303, decision='retain_native_candidate',
                reason='Textured surface viewed at approximately 0.02254m camera Z. Raw world/radial relative discrepancy has p99 about 0.0166; no specific new failure established. Near depth alone does not trigger exclusion.'),
            dict(ordinals=[212357, 212358, 212365, 212366], decision='retain_native_candidates',
                reason='Lowest-support samples depict an exterior view into an interior. Unsupported background/ground stays masked, with 29.7-32.3 percent unchanged valid support and plausible facade/interior structure. No new specific exclusion.')],
        interpretation='Raw world and radial images share renderer provenance and independently quantized storage; they are not independent physical measurements. Numerical agreement alone cannot certify realistic scene scale.',
        primary_sources=[
            dict(url='https://github.com/apple-aiml-research/ml-hypersim/blob/main/README.md', finding='Official documentation defines source distance in meters, world positions in asset units and scene-specific meters_per_asset_unit.'),
            dict(url='https://raw.githubusercontent.com/apple/ml-hypersim/main/code/python/tools/generate_hdf5_from_vrimg.py', finding='Inspected exporter computes radial distance from world position and camera position, multiplies by source scale, then stores both position and distance as float16. This is shared renderer provenance.')],
        all_original_native_flags_retained=True, fitted_scale_applied=False,
        depth_or_mask_changed=False, source_quality_accepted=False,
        training_admission=False, optimizer_updates=0)
    publish_json(output / 'visual-review.json', review)
    review_ref = file_ref(output / 'visual-review.json')
    registered = []
    for row in flags:
        reason = ('reviewed_scene_metric_scale_plausibility_unverified' if row['group'] == 'hypersim:ai_013_001'
                  else 'reviewed_near_surface_storage_precision_and_visual_support_insufficient')
        registered.append(dict(**{k: row[k] for k in ('corpus_ordinal', 'image', 'sha256', 'group',
            'entry_key', 'member_sha256', 'receipt')}, decision='quarantine_candidate', reason=reason,
            evidence=[review_ref, refs['hypersim-native-scale-probe-01/report.json'], scale['scope']]))
    publish_json(output / 'candidate-quarantine.json', dict(schema_version=1,
        stage='candidate_quality_quarantine_recorded', dataset='hypersim',
        policy_sha256=native['policy_sha256'], manifest_sha256=MANIFEST,
        packs_index=native['packs'], flags=registered, training_admission=False, optimizer_updates=0))
    quarantine_ref = file_ref(output / 'candidate-quarantine.json')
    bound = checked_candidate_quarantine(quarantine_ref, 'hypersim', native['policy_sha256'], MANIFEST, native['packs'])
    if len(bound) != 201:
        raise ValueError('Native external quarantine did not bind all reviewed candidates')
    new_exclusions = [dict(**{k: f[k] for k in ('corpus_ordinal', 'image', 'sha256', 'group')},
        dataset='hypersim', decision='exclude', scope='geometry_only', rgb_replay_excluded=False,
        reason=f['reason'], evidence=[quarantine_ref, *f['evidence']],
        candidate_policy_sha256=native['policy_sha256'], candidate_entry_key=f['entry_key'],
        candidate_member_sha256=f['member_sha256'], depth_or_mask_changed=False,
        decision_basis='Specific reviewed scene-scale or image-precision finding; no automatic depth threshold.') for f in registered]
    old_ref = refs['coco-nyu-source-reviews-01/known-geometry-exclusions.jsonl']
    old = [json.loads(line) for line in checked_path(old_ref).read_text().splitlines()]
    combined = sorted(old + new_exclusions, key=lambda r: r['corpus_ordinal'])
    if (len(old) != 37 or len(combined) != 238 or len({r['corpus_ordinal'] for r in combined}) != 238
            or {r['corpus_ordinal'] for r in combined if r['rgb_replay_excluded']} != {210812}):
        raise ValueError('Geometry/RGB exclusion scopes no longer reconcile')
    write_lines(output / 'new-geometry-exclusions.jsonl', new_exclusions)
    write_lines(output / 'known-geometry-exclusions.jsonl', combined)
    pool_ref = file_ref(root / 'rgb-replay-candidates-01/report.json')
    pool = checked_json(pool_ref)
    targets, retained, checksum = {r['image']: r for r in new_exclusions}, set(), hashlib.sha256()
    with Path(pool['manifest']['path']).open('rb') as file:
        for line in file:
            checksum.update(line)
            row = json.loads(line)
            if row['image'] in targets:
                expected_row = targets[row['image']]
                ordinal = expected_row['corpus_ordinal']
                if row['sha256'] != expected_row['sha256'] or ordinal in retained:
                    raise ValueError('Geometry-only RGB replay identity changed or repeated')
                retained.add(ordinal)
    if checksum.hexdigest() != pool['manifest']['sha256'] or retained != expected:
        raise ValueError('Existing RGB pool changed or lost an explicitly retained image')
    publish_json(output / 'report.json', dict(stage='native_quality_dispositions_recorded_not_source_acceptance',
        visual_review=review_ref, candidate_quarantine=quarantine_ref,
        new_geometry_exclusions=file_ref(output / 'new-geometry-exclusions.jsonl'),
        known_geometry_exclusions=file_ref(output / 'known-geometry-exclusions.jsonl'), previous_exclusions=old_ref,
        new_exclusions=201, known_exclusions=238, geometry_only_rgb_retained=sorted(retained),
        pool=pool_ref, pool_manifest=pool['manifest'], rgb_pool_rewritten=False,
        rgb_exclusions=1, original_native_flagged_candidates=3076,
        native_unflagged_candidates_before_external_review=55663,
        native_remaining_candidates=55462, source_quality_accepted=False,
        training_admission=False, optimizer_updates=0, code=file_ref(Path(__file__))))
    print(json.dumps(dict(new_geometry_exclusions=201, known_geometry_exclusions=238,
        rgb_retained=len(retained), source_quality_accepted=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
