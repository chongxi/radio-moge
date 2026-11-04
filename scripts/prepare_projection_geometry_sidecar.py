"""Verify a filtered data-only sidecar; never publish training admission."""
import argparse
import json
from pathlib import Path

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.ordinary_geometry_data import GeometrySidecar
from radio_moge.projection_fork import COMPARISON_SHA, projection_comparison
from radio_moge.projection_sidecar import STAGE, checked_derivation


def run(root, output):
    pair_ref = dict(path=str(root/'projection-filtered-comparison-plan-01/plan.json'), sha256=COMPARISON_SHA)
    pair = projection_comparison(pair_ref)
    parent_ref = dict(path=str(root/'reviewed-stage4-launch-01/acceptance.json'),
        sha256='5d238e12ab80c2979f63496850b051aa66d72d409dba616ece195d65ad654be9')
    parent = checked_json(parent_ref)
    accounting = checked_json(parent['accounting_report'])
    hold = checked_json(pair['hold'])
    gate = dict(schema_version=1, stage=STAGE, training_admission=False,
        projection_comparison=pair_ref, parent_acceptance=parent_ref, materialization=pair['hold'],
        index=pair['candidate_geometry_index'], selected_images=hold['geometry']['selected_images'],
        total_images=accounting['total_images'], manifest_sha256=parent['manifest_sha256'],
        no_sampling_or_loss_authorization=True)
    counts = checked_derivation(gate)
    output.mkdir(exist_ok=False)
    candidate = output/'sidecar.candidate.json'
    publish_json(candidate, gate)
    sidecar = GeometrySidecar(file_ref(candidate))
    sidecar.close()
    candidate.rename(output/'sidecar.json')
    report = dict(stage='projection_sidecar_data_verified_not_training_admitted',
        sidecar=file_ref(output/'sidecar.json'), exact_disposition_counts=counts,
        index_hash_and_closed_database_verified=True, geometry_payloads_reread=False,
        prior_source_flags_and_exclusions_preserved=True, training_admission=False, optimizer_updates=0,
        publisher=file_ref(Path(__file__)))
    publish_json(output/'report.json', report)
    return file_ref(output/'report.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.root, a.output)), flush=True)
