"""Bind completed snapshot payloads to source-specific accounting inputs.

Previously completed source reviews are restricted to the frozen pack list and
reused with their original evidence. Sources without a review remain pending.
This command never writes a final gate, training readiness plan, or optimizer.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from radio_moge.corpus_admission import (checked_json, checked_path, file_ref,
                                        validate_source_review)
from radio_moge.geometry_cache import publish_json
from radio_moge.snapshot_readback import checked_snapshot_generation


def run(stage_path, payload_path, output):
    stage_ref, payload_ref = file_ref(stage_path), file_ref(payload_path)
    stage, payload = checked_json(stage_ref), checked_json(payload_ref)
    scope = checked_json(stage['scope'])
    snapshot = checked_json(stage['snapshot'])
    if (stage['stage'] != 'snapshot_training_scope_prepared_not_accepted'
            or payload.get('stage') != 'snapshot_payload_verified_not_accepted'
            or payload.get('snapshot') != stage['snapshot'] or payload.get('mode') != 'census'
            or payload.get('all_snapshot_packs_verified') is not True
            or payload.get('all_snapshot_members_verified') is not True
            or payload.get('failed_packs') != 0 or payload.get('verified_packs') != snapshot['processed_packs']
            or payload.get('verified_members') != snapshot['processed_entries']):
        raise ValueError('Completed whole-snapshot payload verification is required')
    if not output.name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Use a simple unique output basename for immutable sidecar indexes')
    output.mkdir(exist_ok=False)
    sources, pending = [], []
    for source_ref in stage['sources']:
        source = checked_json(source_ref)
        packs = source['packs']
        parents = {Path(p['receipt']['path']).parent for p in packs}
        if len(parents) != 1:
            raise ValueError('One source attempt must retain one original pack directory')
        parent = next(iter(parents))
        index_path = parent / f'packs.{output.name}.json'
        index = dict(schema_version=1, policy_sha256=source['policy_sha256'],
            count=source['candidate_images'], snapshot=stage['snapshot'],
            packs=[dict(receipt=Path(p['receipt']['path']).name,
                sha256=p['receipt']['sha256'], count=p['count']) for p in packs],
            training_admission=False)
        publish_json(index_path, index)
        directory = output / source['attempt']
        directory.mkdir()
        generation = dict(stage='snapshot_source_payload_verified', mode='snapshot',
            training_admission=False, training_started=False, optimizer_updates=0,
            snapshot=stage['snapshot'], payload_report=payload_ref, source_scope=source_ref,
            policy_sha256=source['policy_sha256'], packs_index=file_ref(index_path),
            completed=index['count'], rejected=0, verified_cache_members=index['count'],
            original_producer_may_continue=True, source_quality_accepted=False)
        candidate = dict(dataset=source['dataset'], packs_index=file_ref(index_path),
                         scope_count=index['count'])
        checked_snapshot_generation(generation, candidate, scope, index)
        publish_json(directory / 'readback-report.json', generation)
        candidate['generation_report'] = file_ref(directory / 'readback-report.json')
        prior_ref = source.get('previous_source_review')
        flags = {}
        if prior_ref is not None:
            prior = checked_json(prior_ref)
            register = checked_json(prior['candidate_quarantine'])
            flags = validate_source_review(prior, source['dataset'], source['policy_sha256'],
                scope['manifest_sha256'], register['packs_index'])
        captured_receipts = {p['receipt']['path']: p['receipt']['sha256'] for p in packs}
        if any(captured_receipts.get(f['receipt']['path']) != f['receipt']['sha256'] for f in flags.values()):
            raise ValueError('Frozen source omits an already reviewed candidate flag')
        quarantine = dict(schema_version=1, stage='candidate_quality_quarantine_recorded',
            dataset=source['dataset'], policy_sha256=source['policy_sha256'],
            manifest_sha256=scope['manifest_sha256'], packs_index=candidate['packs_index'],
            flags=list(flags.values()), snapshot=stage['snapshot'],
            training_admission=False, optimizer_updates=0)
        publish_json(directory / 'candidate-quarantine.json', quarantine)
        if prior_ref is not None:
            review = deepcopy(prior)
            for name in ('eligible_candidate_images', 'quarantined_candidate_images'):
                if name in review:
                    review['parent_' + name] = review.pop(name)
            review.update(candidate_quarantine=file_ref(directory / 'candidate-quarantine.json'),
                parent_source_review=prior_ref, scope_snapshot=stage['snapshot'],
                captured_candidate_images=index['count'],
                review_scope='Captured packs from the previously reviewed policy; original checks and flags retained',
                evidence=[*prior['evidence'], prior_ref, source_ref,
                          candidate['generation_report'], candidate['packs_index']])
            validate_source_review(review, source['dataset'], source['policy_sha256'],
                                   scope['manifest_sha256'], candidate['packs_index'])
            publish_json(directory / 'source-review.json', review)
            candidate['review'] = file_ref(directory / 'source-review.json')
        else:
            pending.append(source['attempt'])
            publish_json(directory / 'review-needed.json', dict(dataset=source['dataset'],
                source=source_ref, generation_report=candidate['generation_report'],
                candidate_quarantine=file_ref(directory / 'candidate-quarantine.json'),
                stage='source_review_required', checks_passed=False, training_admission=False))
            candidate['review'] = None
        sources.append(candidate)
    for ref in (stage_ref, payload_ref, stage['scope'], stage['inventory'], stage['exclusions']):
        checked_path(ref)
    plan = dict(scope=stage['scope'], inventory=stage['inventory'],
        inventory_report=stage['inventory_report'], exclusions=stage['exclusions'], sources=sources)
    publish_json(output / 'accounting-plan-candidate.json', plan)
    report = dict(stage='snapshot_readbacks_bound_not_accepted', snapshot=stage['snapshot'],
        stage_scope=stage_ref, payload_report=payload_ref, source_attempts=len(sources),
        inherited_source_reviews=len(sources) - len(pending), pending_source_reviews=pending,
        plan=file_ref(output / 'accounting-plan-candidate.json'),
        training_admission=False, optimizer_updates=0, code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-report', type=Path, required=True)
    parser.add_argument('--payload-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.stage_report, args.payload_report, args.output)), flush=True)
