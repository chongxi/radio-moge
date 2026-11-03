"""Record the reviewed 25-scene DSLR teacher source eligibility.

The complete outcome audit and actual visual inspection are immutable inputs.
This reuses producer payload readbacks, preserves native-source flags and
publishes explicit per-policy quarantine registers. It never admits training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from radio_moge.corpus_admission import (SOURCE_CHECKS, checked_json, checked_path,
    file_ref, validate_source_review)
from radio_moge.geometry_cache import canonical_json, publish_json


OUTCOMES_SHA = '172b7996c09609c1fef77d8507293c2caa93ef248476edfeea11aec549151ef2'
VISUAL_SHA = 'ade0c7c57e9379afbfe4f154313404b0309fab86cc2fd88b2d9675ec8e5e4d18'
COUNTS = {'5ede4bef14933feab29e62112a639a752d1c2f3d299a5cdc9a6edeac592f7707': 28160,
          'f5b7f238857b3fcf05201791da6fcc6c4818714d89da5ec85465053418863a5e': 7623}


def validate_review_inputs(audit, visual, rendered):
    if (audit['stage'] != 'teacher_outcomes_accounted_not_accepted'
            or audit['dataset'] != 'scannetpp_dslr' or audit['expected_scope'] != 35783
            or audit['candidate_policy_counts'] != COUNTS
            or audit['outcome_counts'] != {'teacher_candidate': 35783}
            or audit['candidate_images'] != 35783 or audit['rejected_images'] != 0
            or audit['scenes'] != 25 or audit['producer_array_readback_complete'] is not True
            or audit['native_source_flags_cleared'] is not False
            or visual['stage'] != 'visual_review_complete'
            or visual['decision'] != 'eligible_for_relative_pseudo_source_review_with_documented_limits'
            or visual['reviewed_images'] != 100 or len(visual['reviewed_pages']) != 34
            or len(set(visual['reviewed_scenes'])) != 25
            or visual['all_rendered_pages_actually_inspected'] is not True
            or visual['additional_candidate_quarantine_ordinals'] != []
            or visual['additional_geometry_exclusions'] != []
            or visual['native_source_flags_cleared'] is not False
            or rendered['rendered_images'] != 100 or rendered['scenes'] != 25
            or rendered['all_selected_views_included'] is not True
            or rendered['selected_payloads_and_rgb_rechecked'] is not True
            or rendered['original_anonymity_masks_rechecked'] is not True
            or rendered['pages'] != visual['reviewed_pages']
            or {r['scene'] for r in rendered['records']} != set(visual['reviewed_scenes'])
            or len({r['corpus_ordinal'] for r in rendered['records']}) != 100
            or any(r['native_source_review_unresolved'] is not True
                   or r['support_within_eroded_anonymity_mask'] is not True
                   for r in rendered['records'])):
        raise ValueError('Exact completed outcomes and actual all-scene visual review required')


def run(config, attempt):
    root = Path(config['root'])
    audit_ref = dict(path=str(root / 'scannetpp-review-teacher-outcomes-01/report.json'), sha256=OUTCOMES_SHA)
    visual_ref = dict(path=str(root / 'scannetpp-review-teacher-extremes-01/visual-review-01.json'), sha256=VISUAL_SHA)
    audit, visual = checked_json(audit_ref), checked_json(visual_ref)
    rendered = checked_json(visual['rendered_evidence'])
    if (visual['outcomes'] != audit_ref or rendered['audit'] != audit_ref
            or audit['manifest_sha256'] != config['manifest_sha256']):
        raise ValueError('Reviewed source evidence belongs to another scope')
    validate_review_inputs(audit, visual, rendered)
    evidence = [audit_ref, visual_ref, visual['rendered_evidence'], audit['index'], audit['strata'],
                audit['review_candidates'], audit['routing'], *audit['evidence'], *visual['reviewed_pages']]
    evidence = list({canonical_json(r): r for r in evidence}.values())
    for ref in evidence:
        checked_path(ref)
    with tempfile.TemporaryDirectory(prefix='scannet-source-review-') as temporary:
        local = Path(temporary) / 'outcomes.sqlite'
        shutil.copyfile(checked_path(audit['index']), local)
        if file_ref(local)['sha256'] != audit['index']['sha256']:
            raise ValueError('Source outcome copy differs')
        with sqlite3.connect(local) as db:
            counts = dict(db.execute('SELECT policy,COUNT(*) FROM images GROUP BY policy'))
            if (counts != COUNTS or db.execute('SELECT COUNT(DISTINCT scene_group) FROM images').fetchone()[0] != 25
                    or db.execute("SELECT COUNT(*) FROM images WHERE outcome IS NOT 'teacher_candidate' "
                                  "OR reason IS NOT NULL OR support < .05 OR entry_key IS NULL").fetchone()[0]):
                raise ValueError('Frozen source outcome identities or support differ')
    output = root / attempt
    output.mkdir(exist_ok=False)
    publish_json(output / 'documentation-review.json', dict(
        stage='primary_source_and_local_policy_review_complete', reviewed_at_utc='2026-11-03',
        references=[
            dict(url='https://arxiv.org/abs/2308.11417', finding='ScanNet++ records indoor scenes with registered laser, DSLR and iPhone data. This teacher subset does not use the laser labels as targets.'),
            dict(url='https://github.com/scannetpp/scannetpp', finding='Official processing distinguishes undistorted pinhole DSLR images and masks from rendered depth. The frozen original images and anonymity masks are retained here.'),
            dict(url='https://arxiv.org/abs/2507.02546', finding='MoGe-2 predicts geometry from one image; these predictions are not independent measured ground truth.'),
        ], local_policy='Pinned monocular inference with forced projection; estimated pinhole camera, float32 camera-Z and original anonymity mask==255 eroded 5x5. Relative-pseudo losses discard nominal teacher scale.',
        boundaries='No repair or acceptance of the native scene geometry; no physical calibration, confidence guarantee, depth editing or hole filling.',
        training_admission=False, optimizer_updates=0))
    evidence.append(file_ref(output / 'documentation-review.json'))
    reviews = []
    for source in audit['sources']:
        gen, packs, policy = (checked_json(source[k]) for k in ('generation', 'packs', 'policy'))
        checksum, name = source['policy_sha256'], source['name']
        if (checksum not in COUNTS or hashlib.sha256(canonical_json(policy)).hexdigest() != checksum
                or policy['teacher'] != config['teacher'] or policy['native_geometry_read'] is not False
                or gen['stage'] != 'complete' or gen['mode'] != 'generate'
                or gen['completed'] != gen['verified_cache_members'] or gen['completed'] != COUNTS[checksum]
                or gen['rejected'] != 0 or gen['training_started'] is not False
                or gen['training_admission'] is not False or gen['optimizer_updates'] != 0
                or packs['policy_sha256'] != checksum or packs['count'] != COUNTS[checksum]):
            raise ValueError('Completed producer or exact teacher policy changed')
        register = dict(schema_version=1, stage='candidate_quality_quarantine_recorded',
            dataset='scannetpp_dslr', policy_sha256=checksum, manifest_sha256=config['manifest_sha256'],
            packs_index=source['packs'], flags=[], evidence=[visual_ref, audit_ref],
            interpretation='No additional candidate exclusions from this bounded review. Existing native-source flags remain; monocular reflection and fine-structure limits are documented.',
            training_admission=False, optimizer_updates=0)
        publish_json(output / (name + '-quarantine.json'), register)
        review = dict(stage='source_review_complete', decision='eligible_with_flagged_rows_quarantined',
            dataset='scannetpp_dslr', policy_sha256=checksum, manifest_sha256=config['manifest_sha256'],
            checks=dict.fromkeys(SOURCE_CHECKS, True), evidence=evidence,
            label_kinds=['teacher_pseudo'], candidate_quarantine=file_ref(output / (name + '-quarantine.json')),
            eligible_candidate_images=COUNTS[checksum], quarantined_candidate_images=0,
            supervision='relative_pseudo', native_geometry_accepted=False, native_source_flags_cleared=False,
            calibrated_metric_accuracy_verified=False, reviewed_at=time.time(),
            check_details=dict(
                documentation_and_paper='Primary papers and official processing documentation read; exact local policy separately bound.',
                units_and_camera='Teacher-estimated pinhole camera on original RGB grid, positive camera-Z; complete receipt audit checks projection and provenance. Relative losses remove nominal scale.',
                valid_support='Every producer payload reread; full support range 0.9562154--0.9943001. Original anonymity masks independently rechecked on all 100 selected members; no invalid pixels filled.',
                geometric_quality='All 35783 exact frozen RGB identities in 25 routed scenes accounted once. Policy/mask/projection and normalized-depth statistics checked. Relative tails are not quality cutoffs.',
                visual_quality='50 preflight views across all 25 scenes reviewed previously and preserved by exact array parity. All 34 new pages, 100 low-support/tail views across all 25 scenes inspected, using individual and common palettes.'),
            limitations=visual['limitations'], mask_interpretation='Predicted support intersected with anonymity mask; not calibrated confidence.',
            whole_corpus_coverage_accepted=False, canonical_heldout_isolation_accepted=False,
            training_admission=False, optimizer_updates=0)
        if validate_source_review(review, 'scannetpp_dslr', checksum, config['manifest_sha256'], source['packs']):
            raise ValueError('Unexpected candidate flag in the explicitly empty register')
        publish_json(output / (name + '-source-review.json'), review)
        reviews.append(file_ref(output / (name + '-source-review.json')))
    publish_json(output / 'report.json', dict(stage='scannetpp_teacher_source_reviews_complete_not_admitted',
        completed_at=time.time(), dataset='scannetpp_dslr', source_reviews=reviews, source_outcomes=audit_ref,
        source_eligible_images=35783, source_eligible_scenes=25, relative_pseudo_images=35783,
        candidate_policy_counts=COUNTS, additional_geometry_exclusions=0, native_source_flags_cleared=False,
        native_geometry_accepted=False, whole_scannet_source_accepted=False,
        training_admission=False, optimizer_updates=0, code=file_ref(Path(__file__))))
    print(json.dumps(dict(stage='source_reviews_complete', eligible=35783, scenes=25)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a fresh attempt basename')
    run(json.loads(args.config.read_text()), args.attempt)
