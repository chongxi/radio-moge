"""Exercise the real geometry crop on all reviewed TartanAir candidate extremes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_training_crop_support import observation_metrics
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, read_geometry_member
from radio_moge.geometry_sampling import load_candidate


def probe_entry(pack, entry):
    depth, valid, camera, meta = read_geometry_member(pack, entry)
    results = []
    for augment, seed in ((False, 0), (True, 43), (True, 0)):
        rgb, target, view = load_candidate(pack, entry, (400, 640), seed, augment)
        if (rgb.shape != (400, 640, 3) or target['supervision'] != 'metric_native'
                or target['label_kind'] != 'synthetic_renderer' or meta['dataset'] != 'tartanair'
                or target['image_sha256'] != entry['image']['sha256']
                or meta['source_needs_review'] is not False):
            raise ValueError('Actual crop changed native identity or label semantics')
        results.append(dict(augmented=augment, seed=seed, flipped=view['flipped'],
            **observation_metrics(depth, valid, camera, target, view)))
    if {r['flipped'] for r in results} != {False, True}:
        raise ValueError('Both ordinary and reflected views must be exercised')
    return results


def run(root, output):
    torch.set_num_threads(1)
    review_ref = file_ref(root / 'tartanair-candidate-visuals-01/visual-review-01.json')
    review = checked_json(review_ref)
    visuals = checked_json(review['evidence'])
    audit = checked_json(visuals['audit'])
    rows = visuals['records']
    if (review['stage'] != 'selected_native_candidates_visually_reviewed'
            or review['all_rendered_images_viewed'] is not True
            or review['new_geometry_exclusions'] != 0
            or len(rows) != review['reviewed_images'] or not 1 <= len(rows) <= 128
            or len({r['corpus_ordinal'] for r in rows}) != len(rows)
            or audit['dataset'] != 'tartanair' or audit['producer_array_readback_complete'] is not True
            or visuals['arrays_camera_mask_and_statistics_reproduced_exactly'] is not True):
        raise ValueError('Bounded completed candidate visual/source review required')
    output.mkdir(exist_ok=False)
    records, receipts = [], {}
    for row in rows:
        ref = row['receipt']
        if ref['path'] not in receipts:
            receipts[ref['path']] = checked_json(ref)
        receipt = receipts[ref['path']]
        found = [e for e in receipt['entries'] if e['key'] == row['entry_key']]
        if (len(found) != 1 or receipt['policy_sha256'] != audit['policy_sha256']
                or found[0]['image'] != row['image'] or found[0]['sha256'] != row['member_sha256']):
            raise ValueError('Selected crop does not match reviewed candidate')
        pack = Path(ref['path']).parent / receipt['tar']
        if pack.parent != Path(ref['path']).parent or pack.stat().st_size != receipt['bytes']:
            raise ValueError('Selected pack path or size differs')
        for result in probe_entry(pack, found[0]):
            records.append(dict(corpus_ordinal=row['corpus_ordinal'], environment=row['environment'],
                difficulty=row['difficulty'], group=row['image']['group'], image=row['image']['image'],
                sha256=row['image']['sha256'], receipt=ref, entry_key=row['entry_key'], **result))
        if len(records) % 30 == 0:
            print(json.dumps(dict(views=len(records), expected=3 * len(rows))), flush=True)
    for ref in (review_ref, review['evidence'], visuals['audit']):
        checked_path(ref)
    publish_json(output / 'views.json', records)
    counts = [r['selected_observations'] for r in records]
    report = dict(stage='candidate_crop_support_observed_not_admitted', dataset='tartanair',
        completed_at=time.time(), images=len(rows), views=len(records),
        groups=len({r['group'] for r in records}),
        source_bundles=len({(r['environment'], r['difficulty']) for r in records}),
        source_candidates=audit['outcome_counts']['native_candidate'],
        output_shape=[400, 640], minimum_selected_observations=min(counts),
        selected_observation_quantiles=np.quantile(counts, [0, .01, .5, .99, 1]).tolist(),
        sparse_views=sum(r['sparse_geometry_skipped'] for r in records),
        sparse_ordinals=sorted({r['corpus_ordinal'] for r in records if r['sparse_geometry_skipped']}),
        no_stable_edge_views=sum(r['stable_edge_pairs'] == 0 for r in records),
        max_native_to_target_relative_error=max(r['native_to_target_max_relative_error'] for r in records),
        actual_production_loader_used=True, actual_flip_exercised_per_image=True,
        all_observations_unique_original_samples=True, original_mask_unchanged=True,
        geometry_skip_rule='Fewer than32 selected observations: zero geometry loss; retain feature supervision',
        source_prevalence_estimate=False, new_exclusions=0, native_holes_filled=False,
        model_probe_performed=False, training_admission=False, source_quality_accepted=False, optimizer_updates=0,
        visual_review=review_ref, outcomes=visuals['audit'], selection=visuals['selection'],
        views_file=file_ref(output / 'views.json'), generator=file_ref(Path(__file__)),
        helper=file_ref(Path(__file__).with_name('audit_training_crop_support.py')),
        implementation={name: file_ref(Path(__file__).parents[1] / 'src/radio_moge' / name)
                        for name in ('geometry_sampling.py', 'geometry_sources.py', 'corpus_losses.py')})
    publish_json(output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('images', 'views', 'sparse_views', 'minimum_selected_observations')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
