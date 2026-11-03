"""Publish the specific COCO/NYU source decisions inspected on 2026-11-03.

This is an evidence-bound curation recipe, not a classifier or training seal.
Sparse COCO geometry is not rejected by its support fraction. Known bad or
ambiguous candidates receive exact geometry-only exclusions; RGB is retained.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from audit_completed_rgb_teachers import rgb
from audit_irs_teacher_outcomes import teacher_metrics
from radio_moge.corpus_admission import (SOURCE_CHECKS, checked_json, checked_path,
    file_ref, validate_source_review)
from radio_moge.geometry_cache import canonical_json, publish_json
from radio_moge.geometry_sampling import source_supervision


MANIFEST = 'a9b2b7469c894d7bb760f4de1eea957c4e64e7d7fead7fab0fc196a8b03e8031'
PINNED = {
    'coco-teacher-source-audit-01/report.json':
        '91149d67d03432268eb91cb7b8ee7e72fddd6565608b8f69393d22cb97a9efbe',
    'coco-teacher-source-audit-01/visual-review-01.json':
        '897ddabf225b8dd7740394b926364fc52e64063a2922c6864f16e9663ec6f6e6',
    'coco-sparse-support-review-01/report.json':
        '229e8bbddcf2c2cdbd2fbd0bb5ec08b842435ae3115936312284ec4d3a6fa8d4',
    'coco-sparse-support-review-01/visual-review-01.json':
        '0c65d671e8a70ba919704eecb96094b1c07ba0963db2536a594f478d8fcb03d8',
    'nyuv2-teacher-source-audit-01/report.json':
        '8895a53c344805a5e626a24a417ad376d1c0d9c6fefa36e7a34ca23fbfce89c7',
    'nyuv2-teacher-source-audit-01/visual-review-01.json':
        '4939f6bf3136633b6e96e8249454b1282db9acbfc09dc48b4dcc6b47cd5f9869',
    'teacher-source-reviews-02/known-geometry-exclusions.jsonl':
        '4bfdf2b5e64cf7510994fb1926e572f5bcba73737eade36c75ce175c2f50225c',
}
COCO_FLAGS = {260107, 261103, 262238, 267916, 269429, 277511, 277846,
              278651, 280475, 280551, 283941, 287880, 295084}


def write_lines(path, records):
    temporary = path.with_suffix('.jsonl.incoming')
    with temporary.open('x') as file:
        for row in records:
            file.write(canonical_json(row).decode() + '\n')
        file.flush()
        os.fsync(file.fileno())
    os.rename(temporary, path)


def publish(config, attempt):
    root = Path(config['root'])
    if config['manifest_sha256'] != MANIFEST:
        raise ValueError('These decisions apply only to the inspected frozen corpus')
    refs = {name: dict(path=str(root / name), sha256=sha) for name, sha in PINNED.items()}
    for ref in refs.values():
        checked_path(ref)
    output = root / attempt
    output.mkdir(exist_ok=False)
    documentation = dict(stage='primary_source_and_local_policy_review_complete',
        reviewed_at_utc='2026-11-03', references=[
            dict(url='https://arxiv.org/abs/1405.0312', finding='COCO provides diverse everyday object imagery and recognition annotations. The geometry in this project is separately generated teacher supervision, not COCO ground truth.'),
            dict(url='https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html', finding='The official release distinguishes original RGB, projected rawDepths, and hole-filled depths. This source review uses original RGB and separate teacher predictions; it does not authorize hole-filled native targets.'),
            dict(url='https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf', finding='The NYU paper describes diverse indoor RGB-D scenes and frames. Existing project train/validation/test boundaries remain frozen; the paper does not validate this project\'s unresolved native RGB registration.'),
            dict(url='https://arxiv.org/abs/2507.02546', finding='MoGe-2 estimates monocular geometry and scale. These predictions are treated as relative pseudo labels with estimated intrinsics, not independently measured metric geometry.')],
        evidence=[file_ref(Path(__file__).parents[1] / 'src/radio_moge' / n)
                  for n in ('geometry_sampling.py', 'corpus_losses.py')],
        limitations='Predicted masks describe support, not calibrated confidence. Thin structures, smoke trails, blur, reflections and occlusions retain monocular ambiguity. Unsupported pixels stay invalid. No depth filling or clipping is authorized.',
        training_admission=False, optimizer_updates=0)
    publish_json(output / 'documentation-review.json', documentation)
    old_ref = refs['teacher-source-reviews-02/known-geometry-exclusions.jsonl']
    exclusions = [json.loads(line) for line in checked_path(old_ref).read_text().splitlines()]
    if len(exclusions) != 24 or len({r['corpus_ordinal'] for r in exclusions}) != 24:
        raise ValueError('The previously reviewed 24 exclusions changed')
    new_exclusions, reviews, summaries = [], [], []
    for dataset, prefix, count, visual_names in (
            ('coco2017', 'coco', 39997, ['coco-teacher-source-audit-01/visual-review-01.json',
                                       'coco-sparse-support-review-01/visual-review-01.json']),
            ('nyuv2', 'nyuv2', 1043, ['nyuv2-teacher-source-audit-01/visual-review-01.json'])):
        audit_ref = refs[f'{prefix}-teacher-source-audit-01/report.json']
        audit = checked_json(audit_ref)
        gen, policy, packs = [checked_json(audit[n]) for n in ('generation', 'policy', 'packs')]
        policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
        if (audit['stage'] != 'completed_teacher_source_accounted_not_accepted'
                or audit['dataset'] != dataset or audit['candidate_images'] != count
                or config['expected_counts'][dataset] != count
                or audit['manifest_sha256'] != MANIFEST or audit['policy_sha256'] != policy_hash
                or audit['producer_array_readback_complete'] is not True
                or gen['stage'] != 'complete' or gen['mode'] != 'generate'
                or gen['completed'] != count or gen['verified_cache_members'] != count
                or gen['rejected'] or gen['training_started'] or gen['training_admission']
                or gen['optimizer_updates'] != 0 or gen['teacher_frozen'] is not True
                or gen['policy_sha256'] != policy_hash or packs['policy_sha256'] != policy_hash
                or packs['count'] != count or policy['teacher'] != config['teacher']):
            raise ValueError('Completed, immutable teacher audit or policy differs')
        outcomes = {r['corpus_ordinal']: r for r in (
            json.loads(line) for line in checked_path(audit['outcomes']).read_text().splitlines())}
        if len(outcomes) != count:
            raise ValueError('Source outcomes are incomplete or repeated')
        flag_rows, reviewed = {}, set()
        for name in visual_names:
            review = checked_json(refs[name])
            if review['stage'] != 'selected_teacher_visual_review_complete' or review['audit'] != audit_ref:
                raise ValueError('Visual review belongs to another source audit')
            reviewed.update(review['reviewed_ordinals'])
            for item in review['candidate_flags']:
                ordinal = item['corpus_ordinal']
                row = outcomes[ordinal]
                if (ordinal in flag_rows or item['entry_key'] != row['key']
                        or item['member_sha256'] != row['member_sha256'] or item['receipt'] != row['receipt']):
                    raise ValueError('Visual finding identity changed or repeated')
                flag_rows[ordinal] = dict(item, **{k: row[k] for k in ('image', 'sha256', 'group')},
                    decision='quarantine_candidate', evidence=[refs[name], item['page'], audit_ref])
        expected_flags = COCO_FLAGS if dataset == 'coco2017' else set()
        if set(flag_rows) != expected_flags or len(reviewed) != (131 if dataset == 'coco2017' else 10):
            raise ValueError('The explicit inspected source decision changed')
        quarantine = dict(schema_version=1, stage='candidate_quality_quarantine_recorded',
            dataset=dataset, policy_sha256=policy_hash, manifest_sha256=MANIFEST,
            packs_index=audit['packs'], flags=[flag_rows[i] for i in sorted(flag_rows)],
            training_admission=False, optimizer_updates=0)
        publish_json(output / f'{dataset}-candidate-quarantine.json', quarantine)
        quarantine_ref = file_ref(output / f'{dataset}-candidate-quarantine.json')
        minimum_pixels, minimum_support, strata = float('inf'), 1., Counter()
        seen, keys = set(), set()
        for packet in packs['packs']:
            receipt_ref = dict(path=str(Path(audit['packs']['path']).parent / packet['receipt']), sha256=packet['sha256'])
            receipt = checked_json(receipt_ref)
            tar = Path(receipt_ref['path']).parent / receipt['tar']
            if (tar.parent != Path(audit['packs']['path']).parent or tar.stat().st_size != receipt['bytes']
                    or receipt['policy_sha256'] != policy_hash or len(receipt['entries']) != packet['count']):
                raise ValueError('Previously audited receipt or pack size changed')
            for entry in receipt['entries']:
                row, meta = entry['image'], entry['geometry_metadata']
                ordinal = row['corpus_ordinal']
                bound = outcomes.get(ordinal)
                if (bound is None or ordinal in seen or entry['key'] in keys
                        or entry['key'] != bound['key'] or entry['sha256'] != bound['member_sha256']
                        or receipt_ref != bound['receipt'] or row['dataset'] != dataset or row['split'] != 'train'
                        or any(row[k] != bound[k] for k in ('image', 'sha256', 'group'))
                        or source_supervision(meta) != 'relative_pseudo'):
                    raise ValueError('Candidate differs from the complete frozen-row audit')
                shape = None
                if dataset == 'nyuv2':
                    image = rgb(row)
                    shape = image.height, image.width
                support, *_ = teacher_metrics(entry, policy_hash, config['teacher']['revision'], dataset, shape)
                if meta['valid_pixels'] < 32 or support != bound['valid_fraction']:
                    raise ValueError('Insufficient original observations or changed support')
                seen.add(ordinal)
                keys.add(entry['key'])
                minimum_pixels = min(minimum_pixels, meta['valid_pixels'])
                minimum_support = min(minimum_support, support)
                if ordinal not in flag_rows:
                    strata[bound['stratum']] += 1
        if seen != set(outcomes):
            raise ValueError('Receipt audit missed a source outcome')
        for ordinal, flag in sorted(flag_rows.items()):
            new_exclusions.append(dict(corpus_ordinal=ordinal, dataset=dataset,
                **{k: flag[k] for k in ('image', 'sha256', 'group')}, decision='exclude',
                scope='geometry_only', rgb_replay_excluded=False,
                reason=flag['reason'], evidence=[quarantine_ref, *flag['evidence']],
                candidate_policy_sha256=policy_hash, candidate_entry_key=flag['entry_key'],
                candidate_member_sha256=flag['member_sha256'], depth_or_mask_changed=False,
                decision_basis='Explicit visual finding for this candidate; no ratio or support cutoff.'))
        summary = dict(stage='source_receipts_and_dispositions_verified', dataset=dataset,
            cached_images=count, eligible_images=count - len(flag_rows), quarantined_candidates=len(flag_rows),
            minimum_original_valid_pixels=minimum_pixels, minimum_original_valid_fraction=minimum_support,
            eligible_strata=dict(strata), receipts_checked=count, source_audit=audit_ref,
            producer_full_array_readback=True, independent_full_array_readback=False,
            pack_payloads_rehashed=False, candidate_quarantine=quarantine_ref,
            training_admission=False, optimizer_updates=0)
        publish_json(output / f'{dataset}-receipt-audit.json', summary)
        evidence = [audit_ref, audit['generation'], audit['packs'], audit['policy'], audit['outcomes'],
            file_ref(output / 'documentation-review.json'), file_ref(output / f'{dataset}-receipt-audit.json'),
            quarantine_ref, *[refs[n] for n in visual_names]]
        if dataset == 'nyuv2':
            evidence += [file_ref(root / n) for n in ('nyuv2-teacher-candidate-routing-20261102.json',
                'nyuv2-teacher-preflight-01/visual-review.json')]
        else:
            evidence.append(refs['coco-sparse-support-review-01/report.json'])
        review = dict(stage='source_review_complete', decision='eligible_with_flagged_rows_quarantined',
            dataset=dataset, policy_sha256=policy_hash, manifest_sha256=MANIFEST,
            checks=dict.fromkeys(SOURCE_CHECKS, True), evidence=evidence, candidate_quarantine=quarantine_ref,
            label_kinds=['teacher_pseudo'], supervision='relative_pseudo',
            eligible_candidate_images=count - len(flag_rows), quarantined_candidate_images=len(flag_rows),
            native_geometry_accepted=False, calibrated_metric_accuracy_verified=False,
            whole_corpus_coverage_accepted=False, canonical_heldout_isolation_accepted=False,
            training_admission=False, optimizer_updates=0, reviewed_at=time.time(),
            check_details=dict(
                documentation_and_paper='Primary dataset documentation, papers and MoGe-2 context reviewed; teacher supervision remains separate from native labels.',
                units_and_camera='Original RGB grid and teacher-estimated camera-Z geometry; nominal scale is discarded by relative_pseudo losses. Producer verifies point/depth projection consistency.',
                valid_support='All original candidates have at least 32 supported pixels, exact receipt statistics and complete producer payload readback. No support-fraction exclusion or filling. Sparse COCO crop losses are separately bound for final coverage review.',
                geometric_quality='Full source identities and depth/support distributions checked, selected actual arrays reread. The exact later flags are quarantined with geometry-only exclusions; no calibrated confidence or physical accuracy claim.',
                visual_quality=('131 distinct COCO views inspected, including all 13 numeric preflight views, aspect extremes and every one of 101 low-support images. Thirteen exact candidates excluded.' if dataset == 'coco2017' else 'All earlier 45 preflight views covering 24 scene types and ten full-cache extreme/support views inspected. No specific new exclusion. Native RGB/raw-depth registration remains unresolved and is not used.')))
        validate_source_review(review, dataset, policy_hash, MANIFEST, audit['packs'])
        # Publish source decisions only after the exclusion file is complete below.
        reviews.append((dataset, review))
        summaries.append(summary)
    exclusions.extend(new_exclusions)
    if (len(exclusions) != 37 or len({r['corpus_ordinal'] for r in exclusions}) != 37
            or {r['corpus_ordinal'] for r in exclusions if r['rgb_replay_excluded']} != {210812}):
        raise ValueError('Known geometry/RGB exclusion scopes changed')
    write_lines(output / 'new-geometry-exclusions.jsonl', new_exclusions)
    write_lines(output / 'known-geometry-exclusions.jsonl', sorted(exclusions, key=lambda r: r['corpus_ordinal']))
    pool_ref = file_ref(root / 'rgb-replay-candidates-01/report.json')
    pool = checked_json(pool_ref)
    retained = {}
    # Stream a bounded manifest once, checking its hash and the exact 13 retained
    # RGB identities. This neither changes the replay pool nor accepts sampling.
    checksum = hashlib.sha256()
    target_images = {r['image']: r for r in new_exclusions}
    with Path(pool['manifest']['path']).open('rb') as file:
        for line in file:
            checksum.update(line)
            row = json.loads(line)
            if row['image'] in target_images:
                expected = target_images[row['image']]
                if row['sha256'] != expected['sha256'] or expected['corpus_ordinal'] in retained:
                    raise ValueError('Replay identity differs or repeats')
                retained[expected['corpus_ordinal']] = row['sha256']
    if checksum.hexdigest() != pool['manifest']['sha256'] or set(retained) != COCO_FLAGS:
        raise ValueError('Existing RGB pool changed or lost a geometry-only exclusion')
    publish_json(output / 'rgb-retention.json', dict(stage='geometry_only_rgb_retention_verified',
        pool=pool_ref, manifest=pool['manifest'], retained_ordinals=sorted(retained),
        rgb_pool_rewritten=False, sampling_accepted=False, training_admission=False, optimizer_updates=0))
    final_reviews = []
    for dataset, review in reviews:
        review['evidence'].extend([file_ref(output / 'new-geometry-exclusions.jsonl'),
                                  file_ref(output / 'rgb-retention.json')])
        validate_source_review(review, dataset, review['policy_sha256'], MANIFEST,
                               checked_json(review['candidate_quarantine'])['packs_index'])
        publish_json(output / f'{dataset}-source-review.json', review)
        final_reviews.append(file_ref(output / f'{dataset}-source-review.json'))
    report = dict(stage='two_teacher_source_reviews_complete_not_corpus_acceptance',
        source_reviews=final_reviews, summaries=summaries,
        known_geometry_exclusions=file_ref(output / 'known-geometry-exclusions.jsonl'),
        known_geometry_exclusions_count=37, new_geometry_exclusions=13, rgb_replay_exclusions=1,
        previous_exclusions=old_ref, source_eligible_images={'coco2017': 39984, 'nyuv2': 1043},
        training_admission=False, optimizer_updates=0, completed_at=time.time(), code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('stage', 'source_eligible_images', 'known_geometry_exclusions_count')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use an attempt basename')
    publish(json.loads(args.config.read_text()), args.attempt)
