"""Test native ARKit candidates with the real crop loader, without admission.

Source quality remains pending. The input is the completed outcome audit and
explicit exclusions, not an invented accepted accounting report. Only display
RGB is warped; every geometry target must remain one exact native observation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

import numpy as np
import torch

from audit_training_crop_support import choose_samples, observation_metrics
from indoor_download import write_json
from radio_moge.corpus_admission import checked_candidate_quarantine, checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, read_geometry_member
from radio_moge.geometry_sampling import load_candidate


def select_rotations(rows, excluded, policy):
    """Keep all four orientation strata; never sample an excluded candidate."""
    ids = {r[0] for r in rows}
    if (len(ids) != len(rows) or not excluded <= ids
            or {r[4] for r in rows} != {0, 90, 180, 270}):
        raise ValueError('Unique four-orientation scope and exact exclusions required')
    selection = {}
    for turn in (0, 90, 180, 270):
        values = [(i, g, support, median, policy) for i, g, support, median, rotation in rows
                  if rotation == turn and i not in excluded]
        chosen = choose_samples(values, groups_per_policy=16, support_extremes=8, depth_extremes=4)
        selection.update({i: [f'rotation_{turn}', *reasons] for i, reasons in chosen.items()})
    return dict(sorted(selection.items()))


def probe_entry(pack, entry, output_shape=(400, 640)):
    depth, valid, kd, metadata = read_geometry_member(pack, entry)
    records = []
    # Seed 43 exercises an unreflected crop; seed 0 exercises a reflection in
    # the unchanged production RNG sequence. Assert that coverage below.
    for augment, seed in ((False, 0), (True, 43), (True, 0)):
        rgb, target, view = load_candidate(pack, entry, output_shape, seed, augment)
        if (rgb.shape != (*output_shape, 3) or target['image_sha256'] != entry['image']['sha256']
                or target['supervision'] != 'metric_native' or target['label_kind'] != 'measured_sensor'
                or metadata['policy_sha256'] != entry['geometry_metadata']['policy_sha256']):
            raise ValueError('Actual native loader changed source identity or supervision')
        records.append(dict(augmented=augment, seed=seed, flipped=view['flipped'],
            output_shape=list(output_shape), supervision=target['supervision'],
            **observation_metrics(depth, valid, kd, target, view)))
    if {x['flipped'] for x in records} != {False, True}:
        raise ValueError('Crop probe must include an actual horizontal reflection')
    return records


def run(root, attempt):
    torch.set_num_threads(1)
    audit_ref = file_ref(root / 'arkit-outcomes-01/report.json')
    decisions_ref = file_ref(root / 'arkit-rejection-decisions-01/report.json')
    external_ref = file_ref(root / 'arkit-candidate-decisions-01/report.json')
    audit, decisions, external = (checked_json(r) for r in (audit_ref, decisions_ref, external_ref))
    if (audit['stage'] != 'native_outcomes_accounted_not_accepted'
            or audit['outcome_counts'] != dict(native_candidate=194924, rejected=2618)
            or decisions['source_scope'] != 197542 or decisions['source_geometry_exclusions'] != 2621
            or decisions['remaining_native_candidates'] != 194921
            or decisions['remaining_candidates_accepted'] is not False
            or external['excluded_images'] != 3):
        raise ValueError('Complete native outcomes and exact geometry dispositions required')
    checked_path(decisions['known_geometry_exclusions'])
    flags = checked_candidate_quarantine(external['quarantine'], 'arkitscenes',
        audit['policy_sha256'], audit['manifest_sha256'], audit['packs'])
    excluded = {f['corpus_ordinal'] for f in flags.values()}
    if excluded != {153819, 167294, 182561}:
        raise ValueError('Mandatory exact candidate flags differ')
    output = root / attempt
    output.mkdir(exist_ok=False)
    status = dict(stage='selecting_candidate_crops', started_at=time.time(), completed_views=0,
        dataset='arkitscenes', training_admission=False, source_quality_accepted=False, optimizer_updates=0)
    write_json(output / 'status.json', status)
    try:
        with tempfile.TemporaryDirectory(prefix='arkit-crop-probe-') as temporary:
            local = Path(temporary) / 'outcomes.sqlite'
            shutil.copyfile(checked_path(audit['index']), local)
            if file_ref(local)['sha256'] != audit['index']['sha256']:
                raise ValueError('Temporary outcome index differs')
            with sqlite3.connect(local) as db:
                rows = db.execute("SELECT ordinal,scene_group,json_extract(details_json,'$.support'),"
                    "json_extract(details_json,'$.median_depth'),rotation FROM images "
                    "WHERE outcome='native_candidate' ORDER BY ordinal").fetchall()
                if len(rows) != 194924:
                    raise ValueError('Native candidate count differs')
                selection = select_rotations(rows, excluded, audit['policy_sha256'])
                entries, receipts = [], {}
                for ordinal, reasons in selection.items():
                    encoded, key, proof, turn = db.execute('SELECT row_json,entry_key,evidence_json,rotation '
                        'FROM images WHERE ordinal=?', (ordinal,)).fetchone()
                    row, receipt_ref = json.loads(encoded), json.loads(proof)
                    path = Path(receipt_ref['path'])
                    if str(path) not in receipts:
                        receipts[str(path)] = checked_json(receipt_ref)
                    receipt = receipts[str(path)]
                    found = [e for e in receipt['entries'] if e['key'] == key]
                    if len(found) != 1 or key in flags:
                        raise ValueError('Missing, repeated or quarantined selected candidate')
                    entry = found[0]
                    original = dict(entry['image'])
                    if (original.pop('corpus_ordinal', ordinal) != ordinal or original != row
                            or receipt['policy_sha256'] != audit['policy_sha256']
                            or row['image_rotation_cw'] != turn):
                        raise ValueError('Selected receipt differs from the frozen outcome')
                    pack = path.parent / receipt['tar']
                    if pack.parent != path.parent or pack.stat().st_size != receipt['bytes']:
                        raise ValueError('Selected candidate pack size or path changed')
                    entries.append(dict(ordinal=ordinal, rotation=turn, group=row['group'],
                        selection_reasons=reasons, entry=entry, receipt=receipt_ref, pack=str(pack)))
            publish_json(output / 'selection.json', dict(outcomes=audit_ref, decisions=decisions_ref,
                quarantine=external['quarantine'], source_candidates=194924, excluded_candidates=3,
                remaining_candidates=194921, images=len(entries), entries=entries,
                selection='Per rotation: 16 deterministic visit groups, 8 lowest support, 4 near/far medians',
                candidate_diagnostic_only=True, source_prevalence_estimate=False))
            records = []
            for chosen in entries:
                for value in probe_entry(chosen['pack'], chosen['entry']):
                    records.append(dict(corpus_ordinal=chosen['ordinal'], group=chosen['group'],
                        rotation_cw=chosen['rotation'], selection_reasons=chosen['selection_reasons'],
                        cache_key=chosen['entry']['key'], image_sha256=chosen['entry']['image']['sha256'],
                        policy_sha256=audit['policy_sha256'], **value))
                status.update(stage='checking_actual_loader', completed_views=len(records),
                              expected_views=3 * len(entries), updated_at=time.time())
                if len(records) % 24 == 0:
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        for ref in (audit_ref, decisions_ref, external_ref, audit['index']):
            checked_path(ref)
        publish_json(output / 'views.json', records)
        counts = [x['selected_observations'] for x in records]
        report = dict(stage='candidate_crop_support_observed_not_admitted', dataset='arkitscenes',
            completed_at=time.time(), source_candidates=194924, excluded_candidates=3,
            remaining_candidates=194921, images=len(entries), views=len(records),
            groups=len({x['group'] for x in records}), output_shape=[400, 640],
            rotation_images=dict(Counter(e['rotation'] for e in entries)),
            minimum_selected_observations=min(counts),
            selected_observation_quantiles=np.quantile(counts, [0, .01, .5, .99, 1]).tolist(),
            sparse_views=sum(x['sparse_geometry_skipped'] for x in records),
            sparse_ordinals=sorted({x['corpus_ordinal'] for x in records if x['sparse_geometry_skipped']}),
            no_stable_edge_views=sum(x['stable_edge_pairs'] == 0 for x in records),
            max_native_to_target_relative_error=max(x['native_to_target_max_relative_error'] for x in records),
            actual_production_loader_used=True, original_support_unchanged=True,
            corpus_sparse_prevalence_estimated=False, source_quality_accepted=False,
            faro_visibility_and_calibration_resolved=False, model_forward_backward_tested=False,
            training_admission=False, optimizer_updates=0, outcomes=audit_ref,
            exclusions=decisions_ref, quarantine=external['quarantine'],
            existing_faro_review=audit['existing_faro_review'],
            selection=file_ref(output / 'selection.json'), observations=file_ref(output / 'views.json'),
            code=file_ref(Path(__file__)), crop_metrics_code=file_ref(Path(__file__).with_name('audit_training_crop_support.py')),
            implementation={name: file_ref(Path(__file__).parents[1] / 'src/radio_moge' / name)
                for name in ('geometry_sampling.py', 'geometry_sources.py', 'geometry_cache.py', 'corpus_losses.py')})
        publish_json(output / 'report.json', report)
        status.update(stage='complete', completed_at=time.time(), completed_views=len(records))
        write_json(output / 'status.json', status)
        print(json.dumps({k:report[k] for k in ('images','views','sparse_views','minimum_selected_observations')}), flush=True)
        return report
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
        parser.error('Use a new attempt basename')
    run(args.root.resolve(), args.attempt)
