"""Publish the two specific source decisions reviewed on 2026-11-03.

This is a curation record, not an automatic quality classifier. The decisions
below apply only to the named immutable evidence and teacher policies. They
authorize source eligibility for later full accounting, never optimization.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from audit_irs_teacher_outcomes import teacher_metrics
from radio_moge.corpus_admission import (SOURCE_CHECKS, checked_json, checked_path,
    file_ref, readonly_db, validate_source_review)
from radio_moge.geometry_cache import canonical_json, publish_json
from radio_moge.geometry_sampling import source_supervision


MANIFEST = 'a9b2b7469c894d7bb760f4de1eea957c4e64e7d7fead7fab0fc196a8b03e8031'
PINNED_EVIDENCE = {
    'candidate-quality-quarantine-01/irs.json':
        '428e5627e25e2633c12b4a7ebf0c3e93193771cefb11c35a93a31ac29239a3ee',
    'candidate-quality-quarantine-01/hypersim.json':
        '747b607cf23cc8d7acd27ca3a24d671ec1e40009bb417be4ca50e00762662dc1',
    'teacher-geometry-quality-decisions-01/geometry-exclusions.jsonl':
        'c2d5f19b38eefe86f5eed4743672d7278c109eea05bd61714e513ac598d82486',
    'rgb-replay-input-decisions-01.jsonl':
        'b1bf61ee595a8d73bdfbd329ed8fd744b2261e078562f2547e2b0acdc55b917b',
    'irs-teacher-outcomes-01/selected-view-qc-01/visual-review-01.json':
        '60ed5817c6f23fa84909f222eb62cf5558e85fdd50ceac15636241e1b850b568',
    'irs-teacher-outcomes-01/shared-palette-flags-02/visual-review-01.json':
        '88858676729bc4ebf6c52b5b7ba89fe5db894d94a84d1969229c653cc9a77355',
    'hypersim-teacher-extremes-02/visual-review-01.json':
        'af636326d80838d8da6eeb20370849b0f92abc94af883d8a959d1ba3b6180812',
    'hypersim-combined-routes-01/report.json':
        '4a37f1ff4ed04a370efd09051531a91725266ea240303f0c6f31dfa3cab00408',
}


def publish(config, attempt):
    root = Path(config['root'])
    if config['manifest_sha256'] != MANIFEST:
        raise ValueError('These source decisions apply only to the reviewed frozen corpus')
    pinned = {name: dict(path=str(root / name), sha256=sha) for name, sha in PINNED_EVIDENCE.items()}
    for ref in pinned.values():
        checked_path(ref)
    exclusions = {}
    for name in ('rgb-replay-input-decisions-01.jsonl',
                 'teacher-geometry-quality-decisions-01/geometry-exclusions.jsonl'):
        for line in checked_path(pinned[name]).read_text().splitlines():
            row = json.loads(line)
            ordinal = row['corpus_ordinal']
            if ordinal in exclusions or row['decision'] != 'exclude' or not row['evidence']:
                raise ValueError('Reviewed exclusions must remain unique and explicit')
            for ref in row['evidence']:
                checked_path(ref)
            exclusions[ordinal] = row
    expected_exclusions = {210812, 238221, 238222, *range(381511, 381522),
        747580, 748046, 748049, 760948, 760951, 763509, 763513, 766073, 790866, 790868}
    if (set(exclusions) != expected_exclusions
            or {i for i, r in exclusions.items() if r['rgb_replay_excluded']} != {210812}):
        raise ValueError('The 24 known exclusion scopes changed')
    source_ref = file_ref(root / 'inventory.json')
    source = checked_json(source_ref)
    inventory = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    checked_path(inventory)
    if (source['stage'] != 'inventory_complete' or source['manifest_sha256'] != MANIFEST
            or source['dataset_counts'] != config['expected_counts']):
        raise ValueError('Frozen corpus identity differs')
    combined = checked_json(pinned['hypersim-combined-routes-01/report.json'])
    hyp_rows = [json.loads(line) for line in checked_path(combined['routes']).read_text().splitlines()]
    hyp_teachers = {r['corpus_ordinal'] for r in hyp_rows if r['route'] == 'teacher_candidate'}
    if len(hyp_teachers) != 3680 or combined['route_counts']['reviewed_input_exclusion'] != 1:
        raise ValueError('Hypersim teacher routing changed')

    output = root / attempt
    output.mkdir(exist_ok=False)
    specs = [
        ('irs', 'irs-teacher-cache-01', 133879, 0,
         'abcdd0054911149c46f7c8cb9ed11251c1dab14cf310ad5b4d9cbdc8f8ab8c8c', [
             'irs-teacher-candidate-routing-20261102.json', 'irs-teacher-outcomes-01/report.json',
             'irs-teacher-preflight-01/all-view-qc-01/visual-review-01.json',
             'irs-teacher-outcomes-01/selected-view-qc-01/visual-review-01.json',
             'irs-teacher-outcomes-01/shared-palette-flags-02/visual-review-01.json']),
        ('hypersim', 'hypersim-review-teacher-cache-01', 3680, 1,
         '9afb7d74dda6899e3d1531ed32e1a85e40b0500de04613250a011c9ea20ec8ff', [
             'hypersim-review-teacher-routing-20261102.json', 'hypersim-combined-routes-01/report.json',
             'hypersim-review-teacher-preflight-02/visual-review.json',
             'hypersim-review-teacher-preflight-02/array-parity-to-reviewed-01.json',
             'hypersim-teacher-extremes-02/report.json',
             'hypersim-teacher-extremes-02/visual-review-01.json',
             'hypersim-review-rgb-degeneracy-01/review.json']),
    ]
    notes = dict(stage='primary_source_and_local_policy_review_complete', reviewed_at_utc='2026-11-03',
        references=[
            dict(url='https://arxiv.org/abs/2507.02546', finding='MoGe-2 predicts monocular geometry and metric scale; its prediction is not independent ground truth for these images.'),
            dict(url='https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/v2.py', finding='Pinned infer estimates intrinsics, thresholds the predicted mask, forces point/depth projection and applies predicted scale. This project discards that scale in relative pseudo supervision.'),
            dict(url='https://arxiv.org/abs/1912.09678', finding='IRS is synthetic indoor stereo imagery with lighting and reflective/transmissive effects. Teacher labels here are separate predictions, not the released disparities.'),
            dict(url='https://github.com/blackjack2015/IRS/issues/3', finding='The author acknowledges incorrect released samples, including an example present in the training list. Existing local native pairing concerns remain recorded.'),
            dict(url='https://arxiv.org/abs/2011.02523', finding='Hypersim supplies synthetic indoor scenes and camera/geometry annotations. Its native geometry and the alternative teacher predictions have distinct provenance.'),
            dict(url='https://github.com/apple-aiml-research/ml-hypersim#coordinate-conventions', finding='Native depths are radial distances; camera/world units and scene-specific intrinsics need conversion. Official documentation prefers lossless HDR images for learning. This frozen corpus uses existing lossy tonemapped RGB previews; no HDR equivalence or native-camera repair is claimed.'),
        ], evidence=[file_ref(Path(__file__).parents[1] / 'src/radio_moge' / f)
                     for f in ('geometry_sampling.py', 'corpus_losses.py')],
        interpretation='Source eligibility below is for relative teacher supervision under the exact reviewed policy, not measured physical accuracy or final corpus acceptance.',
        training_admission=False, optimizer_updates=0)
    publish_json(output / 'documentation-review.json', notes)
    reviews = []
    with readonly_db(inventory['path']) as db:
        for ordinal, row in exclusions.items():
            found = db.execute('SELECT image,sha256,dataset FROM images WHERE ordinal=?', (ordinal,)).fetchone()
            if found != (row['image'], row['sha256'], row['dataset']):
                raise ValueError('Exclusion does not match the frozen inventory')
        for dataset, cache_name, expected, rejected, policy_hash, evidence_names in specs:
            cache = root / cache_name
            gen_ref, packs_ref, policy_ref = [file_ref(cache / p) for p in
                ('report.json', 'packs/packs.json', 'packs/policy.json')]
            gen, packs, policy = map(checked_json, (gen_ref, packs_ref, policy_ref))
            if (gen['stage'] != 'complete' or gen['mode'] != 'generate'
                    or gen['completed'] != expected or gen['verified_cache_members'] != expected
                    or gen['rejected'] != rejected or packs['count'] != expected
                    or gen['training_started'] is not False or gen['training_admission'] is not False
                    or gen['optimizer_updates'] != 0 or gen['teacher_frozen'] is not True
                    or gen['policy_sha256'] != policy_hash or packs['policy_sha256'] != policy_hash
                    or hashlib.sha256(canonical_json(policy)).hexdigest() != policy_hash
                    or policy['teacher'] != config['teacher']):
                raise ValueError('Reviewed generation identity changed')
            quarantine_ref = pinned[f'candidate-quality-quarantine-01/{dataset}.json']
            evidence = [gen_ref, packs_ref, policy_ref, quarantine_ref, source_ref, inventory,
                file_ref(output / 'documentation-review.json'),
                pinned['teacher-geometry-quality-decisions-01/geometry-exclusions.jsonl'],
                pinned['rgb-replay-input-decisions-01.jsonl']]
            evidence.extend(pinned.get(name) or file_ref(root / name) for name in evidence_names)
            review = dict(stage='source_review_complete', decision='eligible_with_flagged_rows_quarantined',
                dataset=dataset, policy_sha256=policy_hash, manifest_sha256=MANIFEST,
                checks=dict.fromkeys(SOURCE_CHECKS, True), evidence=evidence,
                label_kinds=['teacher_pseudo'], candidate_quarantine=quarantine_ref,
                supervision='relative_pseudo', native_geometry_accepted=False,
                native_source_flags_cleared=False, calibrated_metric_accuracy_verified=False,
                mask_interpretation='Predicted support, not calibrated confidence; holes remain invalid.',
                quality_decision='Eligible as relative pseudo supervision after the exact reviewed ambiguous candidates are quarantined and excluded from geometry. Reflective/thin/low-detail predictions retain the documented limitations of monocular teacher estimates.',
                whole_corpus_coverage_accepted=False, canonical_heldout_isolation_accepted=False,
                training_admission=False, optimizer_updates=0, reviewed_at=time.time())
            flags = validate_source_review(review, dataset, policy_hash, MANIFEST, packs_ref)
            expected_flag_count = 10 if dataset == 'irs' else 2
            if len(flags) != expected_flag_count:
                raise ValueError('Reviewed quality flag count changed')
            seen, keys, strata = set(), set(), Counter()
            minimum = 1.
            for packet in packs['packs']:
                receipt_ref = dict(path=str(cache / 'packs' / packet['receipt']), sha256=packet['sha256'])
                receipt = checked_json(receipt_ref)
                tar = cache / 'packs' / receipt['tar']
                if (tar.parent != cache / 'packs' or tar.stat().st_size != receipt['bytes']
                        or receipt['policy_sha256'] != policy_hash or len(receipt['entries']) != packet['count']):
                    raise ValueError('Pack receipt identity changed')
                for entry in receipt['entries']:
                    row, meta = entry['image'], entry['geometry_metadata']
                    ordinal = row['corpus_ordinal']
                    found = db.execute('SELECT image,row_json FROM images WHERE ordinal=?', (ordinal,)).fetchone()
                    original = json.loads(found[1]) if found else {}
                    if found:
                        original['image'] = found[0]
                    actual = {k: v for k, v in row.items() if k != 'corpus_ordinal'}
                    if (actual != original or row['dataset'] != dataset or row['split'] != 'train'
                            or ordinal in seen or entry['key'] in keys
                            or source_supervision(meta) != 'relative_pseudo'):
                        raise ValueError('Changed or duplicated teacher candidate identity or supervision')
                    support, *_ = teacher_metrics(entry, policy_hash, config['teacher']['revision'], dataset)
                    if support < .05 or meta['valid_pixels'] < 32:
                        raise ValueError('Teacher support falls outside the reviewed scope')
                    if dataset == 'hypersim' and meta.get('native_source_review_unresolved') is not True:
                        raise ValueError('Hypersim native flag was cleared')
                    seen.add(ordinal)
                    keys.add(entry['key'])
                    minimum = min(minimum, support)
                    if entry['key'] in flags:
                        decision = exclusions[ordinal]
                        if (decision.get('candidate_entry_key') != entry['key']
                                or decision.get('candidate_member_sha256') != entry['sha256']
                                or decision.get('candidate_policy_sha256') != policy_hash):
                            raise ValueError('Flagged candidate lacks its exact reviewed exclusion')
                    elif ordinal in exclusions:
                        raise ValueError('An excluded geometry candidate was not quarantined')
                    else:
                        strata[(row['group'], row.get('eye', 'single'))] += 1
            if (len(seen) != expected or (dataset == 'hypersim' and seen != hyp_teachers)
                    or not set(flags) <= keys):
                raise ValueError('Teacher source scope differs from the complete outcome audit')
            audit = dict(stage='source_receipts_and_dispositions_verified', dataset=dataset,
                cached_images=expected, eligible_images=expected - len(flags), quarantined_candidates=len(flags),
                reviewed_input_rejections=rejected, eligible_groups=len({g for g, _ in strata}),
                eligible_group_eye_pairs=len(strata), minimum_original_valid_fraction=minimum,
                frozen_rows_compared=expected, receipts_rehashed=True,
                producer_full_array_readback=True, independent_full_array_readback=False,
                pack_payloads_rehashed=False, generation_report=gen_ref, packs=packs_ref,
                candidate_quarantine=quarantine_ref, inventory=inventory,
                code=file_ref(Path(__file__)), training_admission=False, optimizer_updates=0)
            publish_json(output / f'{dataset}-receipt-audit.json', audit)
            review['evidence'].append(file_ref(output / f'{dataset}-receipt-audit.json'))
            review['eligible_candidate_images'] = expected - len(flags)
            review['quarantined_candidate_images'] = len(flags)
            review['check_details'] = dict(
                documentation_and_paper='Primary documentation and prior source studies reviewed; see documentation-review.json and the preserved native routing decisions.',
                units_and_camera='Original RGB grid and teacher-estimated camera; float32 positive camera-Z. Generation checks 3D vector projection consistency; nominal teacher scale is discarded by relative_pseudo losses.',
                valid_support='Every receipt has finite ordered depth statistics, exact RGB shape and at least 5% predicted support. Producer reread every payload; no confidence claim or hole filling.',
                geometric_quality='All cached identities match frozen rows and relative-pseudo provenance. Full normalized-depth distributions and selected payloads were reviewed. Exact later quality flags have geometry exclusions; no ratio cutoff or metric-accuracy inference.',
                visual_quality=('52 stratified views across 13 families and both eyes, 104 full-cache extreme/support views, and 10 common-palette followups reviewed. The old Supermarket broad-color argument remains withdrawn.' if dataset == 'irs' else 'All 116 initial views across 60 unresolved scenes reviewed; white RGB excluded. The 115 retained preflight arrays match the reviewed arrays exactly. Ten full-cache extremes reviewed with an expanded common palette; two enclosure ambiguities excluded.'))
            validate_source_review(review, dataset, policy_hash, MANIFEST, packs_ref)
            publish_json(output / f'{dataset}-source-review.json', review)
            reviews.append(file_ref(output / f'{dataset}-source-review.json'))
    incoming = output / 'known-geometry-exclusions.jsonl.incoming'
    with incoming.open('x') as file:
        for ordinal in sorted(exclusions):
            file.write(canonical_json(exclusions[ordinal]).decode() + '\n')
        file.flush()
        os.fsync(file.fileno())
    os.rename(incoming, output / 'known-geometry-exclusions.jsonl')
    report = dict(stage='two_teacher_source_reviews_complete_not_corpus_acceptance',
        source_reviews=reviews, known_geometry_exclusions=file_ref(output / 'known-geometry-exclusions.jsonl'),
        known_geometry_exclusions_count=len(exclusions), rgb_replay_exclusions=1,
        input_exclusions=[pinned[n] for n in ('rgb-replay-input-decisions-01.jsonl',
            'teacher-geometry-quality-decisions-01/geometry-exclusions.jsonl')],
        source_eligible_images={'irs': 133869, 'hypersim_review_teacher': 3678},
        native_sources_accepted=False, training_admission=False, optimizer_updates=0,
        completed_at=time.time(), code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use an attempt basename')
    publish(json.loads(args.config.read_text()), args.attempt)
