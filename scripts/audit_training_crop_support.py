"""Exercise actual training crops on reviewed source extremes and scene samples.

The selection deliberately oversamples low support and depth extremes. Its
sparse-view fraction is a diagnostic, not a corpus prevalence estimate. No
geometry is edited, no model is loaded, and no admission gate is issued.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

import numpy as np
import torch

from indoor_download import write_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.corpus_losses import stable_edge_support
from radio_moge.geometry_cache import publish_json, read_geometry_member
from radio_moge.geometry_sampling import load_candidate
from radio_moge.geometry_sources import pinhole_rays


def choose_samples(rows, groups_per_policy=64, support_extremes=32, depth_extremes=8):
    """Rows contain ordinal, group, support, median and immutable cache policy."""
    if len({r[0] for r in rows}) != len(rows) or not rows:
        raise ValueError('Unique nonempty source identities required')
    by_policy, reasons = defaultdict(list), defaultdict(set)
    for row in rows:
        by_policy[row[4]].append(row)
    for policy, values in sorted(by_policy.items()):
        for reason, field, reverse, count in (('low_support', 2, False, support_extremes),
                ('near_depth', 3, False, depth_extremes), ('far_depth', 3, True, depth_extremes)):
            for row in sorted(values, key=lambda x: ((-1 if reverse else 1) * x[field], x[0]))[:count]:
                reasons[row[0]].add(reason)
        groups = defaultdict(list)
        for row in values:
            groups[row[1]].append(row)
        def key(text):
            return hashlib.sha256(('crop-support-v1:' + text).encode()).hexdigest()
        for group in sorted(groups, key=key)[:groups_per_policy]:
            row = min(groups[group], key=lambda x: key(policy + ':' + str(x[0])))
            reasons[row[0]].add('deterministic_scene_sample')
    return {ordinal: sorted(values) for ordinal, values in sorted(reasons.items())}


def observation_metrics(depth, valid, intrinsics, target, view):
    mask = target['valid']
    ids = target['native_index'][mask]
    if (ids.size != np.unique(ids).size or (ids < 0).any() or (ids >= depth.size).any()
            or not valid.ravel()[ids].all()):
        raise ValueError('Crop duplicated or invented a native observation')
    points = target['points'][mask]
    exact = (pinhole_rays(intrinsics, *depth.shape).reshape(-1, 3)[ids]
             * depth.ravel()[ids, None]) @ view['camera_transform'].T
    errors = np.linalg.norm(exact - points, axis=1) / np.maximum(np.linalg.norm(exact, axis=1), 1)
    maximum_error = float(errors.max(initial=0))
    if (not np.isfinite(points).all() or not (points[:, 2] > 0).all()
            or not np.isfinite(maximum_error) or maximum_error > 4 * np.finfo(np.float32).eps):
        raise ValueError('Crop changed its exact native point geometry')
    projected = points.astype(np.float64) @ target['intrinsics'].T
    if len(points) and not np.allclose(projected[:, :2] / projected[:, 2:],
                                      target['query_uv'][mask], atol=2e-7, rtol=0):
        raise ValueError('Crop point and prediction query are misaligned')
    truth = torch.from_numpy(target['points'])
    if target['supervision'] == 'relative_pseudo' and mask.any():
        truth = truth / truth[..., 2][torch.from_numpy(mask)].median()
    edge_mask = stable_edge_support(truth, torch.from_numpy(mask))
    edge_pairs = int((edge_mask[1:] & edge_mask[:-1]).sum()
                     + (edge_mask[:, 1:] & edge_mask[:, :-1]).sum())
    return dict(selected_observations=len(ids), native_valid_pixels=int(valid.sum()),
        supported_loss_fraction=float(mask.mean()), loss_shape=list(mask.shape),
        visible_native_count=target['visible_native_count'], unique_native_observations=True,
        native_to_target_max_relative_error=maximum_error, stable_edge_pairs=edge_pairs,
        sparse_geometry_skipped=len(ids) < 32)


def run(accounting_path, output, dataset, output_shape=(400, 640)):
    torch.set_num_threads(1)
    accounting_ref = file_ref(accounting_path)
    accounting = checked_json(accounting_ref)
    if (accounting['stage'] != 'accounting_verified_not_accepted' or accounting['unresolved'] != 0
            or dataset not in accounting['selected_counts']):
        raise ValueError('Completed reviewed accounting with the requested source is required')
    output.mkdir(exist_ok=False)
    status = dict(stage='selecting_training_crops', started_at=time.time(), completed_views=0,
                  training_admission=False, optimizer_updates=0, dataset=dataset)
    write_json(output / 'status.json', status)
    try:
        with tempfile.TemporaryDirectory(prefix='crop-support-') as temporary:
            local = Path(temporary) / 'accounting.sqlite'
            shutil.copyfile(checked_path(accounting['index']), local)
            with sqlite3.connect(local) as db:
                rows = db.execute("SELECT ordinal,scene_group,valid_fraction,median_depth,"
                    "json_extract(entry_json,'$.geometry_metadata.policy_sha256') FROM images "
                    "WHERE disposition='selected' AND dataset=? ORDER BY ordinal", (dataset,)).fetchall()
                if len(rows) != accounting['selected_counts'][dataset]:
                    raise ValueError('Source selections differ from completed accounting')
                selection = choose_samples(rows)
                entries = []
                for ordinal, reasons in selection.items():
                    pack, encoded, group = db.execute('SELECT pack,entry_json,scene_group FROM images '
                                                      'WHERE ordinal=?', (ordinal,)).fetchone()
                    entries.append(dict(ordinal=ordinal, pack=pack, entry=json.loads(encoded),
                                        group=group, selection_reasons=reasons))
            publish_json(output / 'selection.json', dict(accounting=accounting_ref,
                source_images=len(rows), selected_images=len(entries), entries=entries,
                selection='Per policy: 32 lowest support, 8 nearest/farthest depths, 64 hash-selected groups',
                source_prevalence_estimate=False))
            records = []
            cases = ((False, 0), (True, 43), (True, 13145))
            for chosen in entries:
                entry, pack = chosen['entry'], chosen['pack']
                depth, valid, intrinsics, metadata = read_geometry_member(pack, entry)
                for augment, seed in cases:
                    rgb, target, view = load_candidate(pack, entry, output_shape, seed, augment)
                    if (rgb.shape != (*output_shape, 3) or target['image_sha256'] != entry['image']['sha256']
                            or metadata['policy_sha256'] != entry['geometry_metadata']['policy_sha256']):
                        raise ValueError('Actual loader returned different RGB or policy')
                    metrics = observation_metrics(depth, valid, intrinsics, target, view)
                    records.append(dict(corpus_ordinal=chosen['ordinal'], group=chosen['group'],
                        selection_reasons=chosen['selection_reasons'], cache_key=entry['key'],
                        image_sha256=entry['image']['sha256'], policy_sha256=metadata['policy_sha256'],
                        augmented=augment, seed=seed, flipped=view['flipped'],
                        output_shape=list(output_shape), supervision=target['supervision'], **metrics))
                status.update(stage='checking_actual_loader', completed_views=len(records),
                              expected_views=3 * len(entries), updated_at=time.time())
                if len(records) % 24 == 0:
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        checked_path(accounting_ref)
        checked_path(accounting['index'])
        publish_json(output / 'views.json', records)
        selected_observations = [r['selected_observations'] for r in records]
        report = dict(stage='training_crop_support_observed_not_admitted', dataset=dataset,
            completed_at=time.time(), source_images=len(rows), images=len(entries), views=len(records),
            groups=len({r['group'] for r in records}), output_shape=list(output_shape),
            policy_images=dict(Counter(e['entry']['geometry_metadata']['policy_sha256'] for e in entries)),
            minimum_selected_observations=min(selected_observations),
            selected_observation_quantiles=np.quantile(selected_observations, [0, .01, .5, .99, 1]).tolist(),
            sparse_views=sum(r['sparse_geometry_skipped'] for r in records),
            no_stable_edge_views=sum(r['stable_edge_pairs'] == 0 for r in records),
            sparse_ordinals=sorted({r['corpus_ordinal'] for r in records if r['sparse_geometry_skipped']}),
            max_native_to_target_relative_error=max(r['native_to_target_max_relative_error'] for r in records),
            original_support_unchanged=True, actual_production_loader_used=True,
            corpus_sparse_prevalence_estimated=False, source_quality_repeated=False,
            model_forward_backward_tested=False, training_admission=False, optimizer_updates=0,
            accounting=accounting_ref, index=accounting['index'],
            selection=file_ref(output / 'selection.json'), observations=file_ref(output / 'views.json'),
            implementation={name: file_ref(Path(__file__).parents[1] / 'src/radio_moge' / name)
                for name in ('geometry_sampling.py', 'geometry_sources.py', 'geometry_cache.py', 'corpus_losses.py')},
            code=file_ref(Path(__file__)))
        publish_json(output / 'report.json', report)
        status.update(stage='complete', completed_at=time.time(), completed_views=len(records))
        write_json(output / 'status.json', status)
        print(json.dumps({k: report[k] for k in ('images', 'views', 'sparse_views',
                                                'minimum_selected_observations')}), flush=True)
        return report
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounting', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    run(args.accounting.resolve(), args.output.resolve(), args.dataset)
