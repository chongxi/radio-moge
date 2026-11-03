"""Exercise actual RGB/point crops for the reviewed all-scene teacher extremes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_training_crop_support import observation_metrics
from indoor_download import write_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, read_geometry_member
from radio_moge.geometry_sampling import load_candidate


def run(root, attempt):
    torch.set_num_threads(1)
    audit_ref = file_ref(root / 'scannetpp-review-teacher-outcomes-01/report.json')
    audit = checked_json(audit_ref)
    selected = checked_json(audit['review_candidates'])
    if (audit['stage'] != 'teacher_outcomes_accounted_not_accepted'
            or audit['candidate_images'] != 35783 or audit['rejected_images'] != 0
            or audit['scenes'] != 25 or len(selected) != 100
            or len({r['corpus_ordinal'] for r in selected}) != 100
            or len({r['group'] for r in selected}) != 25):
        raise ValueError('Exact completed all-scene teacher outcome selection required')
    output = root / attempt
    output.mkdir(exist_ok=False)
    status = dict(stage='checking_actual_loader', started_at=time.time(), completed_views=0,
        expected_views=300, training_admission=False, optimizer_updates=0)
    write_json(output / 'status.json', status)
    records, receipts = [], {}
    try:
        for chosen in selected:
            proof = chosen['receipt']
            if proof['path'] not in receipts:
                receipts[proof['path']] = checked_json(proof)
            receipt = receipts[proof['path']]
            entries = [e for e in receipt['entries'] if e['key'] == chosen['key']]
            if (len(entries) != 1 or receipt['policy_sha256'] != chosen['policy_sha256']
                    or receipt['policy_sha256'] not in audit['candidate_policy_counts']):
                raise ValueError('Selected receipt differs from the audited policy')
            entry = entries[0]
            if any(entry['image'][k] != chosen[k] for k in ('image', 'sha256', 'group', 'corpus_ordinal')):
                raise ValueError('Selected RGB identity changed')
            pack = Path(proof['path']).parent / receipt['tar']
            depth, valid, kd, metadata = read_geometry_member(pack, entry)
            flips = set()
            for augment, seed in ((False, 0), (True, 43), (True, 0)):
                rgb, target, view = load_candidate(pack, entry, (400, 640), seed, augment)
                if (rgb.shape != (400, 640, 3) or target['supervision'] != 'relative_pseudo'
                        or target['label_kind'] != 'teacher_pseudo'
                        or target['image_sha256'] != entry['image']['sha256']
                        or metadata.get('native_source_review_unresolved') is not True):
                    raise ValueError('Actual loader changed RGB identity or supervision')
                flips.add(view['flipped'])
                records.append(dict(corpus_ordinal=chosen['corpus_ordinal'], group=chosen['group'],
                    policy_sha256=receipt['policy_sha256'], cache_key=entry['key'],
                    image_sha256=entry['image']['sha256'], receipt=proof,
                    augmented=augment, seed=seed, flipped=view['flipped'], supervision=target['supervision'],
                    **observation_metrics(depth, valid, kd, target, view)))
            if flips != {False, True}:
                raise ValueError('Diagnostic must exercise an actual horizontal reflection')
            if len(records) % 30 == 0:
                status.update(completed_views=len(records), updated_at=time.time())
                write_json(output / 'status.json', status)
                print(json.dumps(status), flush=True)
        checked_path(audit_ref)
        publish_json(output / 'views.json', records)
        counts = [r['selected_observations'] for r in records]
        report = dict(stage='candidate_crop_support_observed_not_admitted', dataset='scannetpp_dslr',
            completed_at=time.time(), outcomes=audit_ref, selection=audit['review_candidates'],
            images=100, views=300, groups=25, output_shape=[400, 640],
            policy_image_counts={p: sum(c['policy_sha256'] == p for c in selected)
                                 for p in audit['candidate_policy_counts']},
            minimum_selected_observations=min(counts), selected_observation_quantiles=np.quantile(counts, [0,.01,.5,.99,1]).tolist(),
            sparse_geometry_views=sum(r['sparse_geometry_skipped'] for r in records),
            max_native_to_target_relative_error=max(r['native_to_target_max_relative_error'] for r in records),
            actual_flip_exercised_per_image=True, all_observations_unique_original_samples=True,
            source_prevalence_estimate=False, original_mask_unchanged=True, depth_edits=0,
            views_file=file_ref(output / 'views.json'), model_probe_performed=False,
            training_admission=False, optimizer_updates=0, code=file_ref(Path(__file__)),
            metric_helper=file_ref(Path(__file__).with_name('audit_training_crop_support.py')))
        publish_json(output / 'report.json', report)
        write_json(output / 'status.json', report)
        print(json.dumps({k:report[k] for k in ('stage','images','views','sparse_geometry_views')}), flush=True)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a fresh attempt basename')
    run(args.root.resolve(), args.attempt)
