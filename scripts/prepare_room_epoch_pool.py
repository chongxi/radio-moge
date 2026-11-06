"""Extend the closed 46,844 RGB pool with the last two reviewed structural batches."""
import argparse
from collections import Counter
import json
from pathlib import Path

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json
from prepare_room_rgb_pool import check_rows, overlap


def check_review(closure, frozen, visual):
    if (frozen['stage'] != 'frozen_verified_structural_feature_rgb'
            or frozen['split'] != 'train' or not frozen['feature_training_eligible']
            or not frozen['every_selected_rgb_decoded_and_hashed']
            or closure['manifest'] != frozen['manifest']
            or closure.get('images', closure.get('qualified_images')) != frozen['images']
            or visual['source_report'] != closure['freeze']
            or not visual['all64_images_actually_viewed']
            or not visual['all16_pages_actually_viewed']
            or visual['new_exclusions'] or not visual['feature_rgb_data_eligible']
            or visual.get('rejected_image_admitted', False)
            or visual.get('replacement_images', 0)):
        raise ValueError('New batch lacks its exact freeze and actual sampled review')
    if visual.get('rejected_diagnostic_images', 0):
        if (visual['rejected_diagnostic_images'] != 1 or visual['admitted_sample_images'] != 63
                or visual.get('rejected_image_admitted') is not False):
            raise ValueError('Rejected review-only images must remain outside the training population')


def prepare(request, output):
    parent = checked_json(request['parent_pool'])
    if parent['stage'] != 'frozen_room_full_epoch_pool_ready' or parent['images'] != 46844:
        raise ValueError('Expected the already qualified parent room pool')
    proofs = [parent]
    for reference in request['new_closures']:
        closure = checked_json(reference)
        frozen = checked_json(closure['freeze'])
        visual = checked_json(closure['visual_review'])
        check_review(closure, frozen, visual)
        proofs.append(frozen)
    paths = [checked_path(p['manifest']) for p in proofs]
    contents = [p.read_bytes() for p in paths]
    if any(not c.endswith(b'\n') for c in contents):
        raise ValueError('Manifest is not closed')
    populations = [[json.loads(line) for line in c.splitlines()] for c in contents]
    if [len(p) for p in populations] != [46844, 1981, 1962]:
        raise ValueError('The authorized 50,787-image scope changed')
    validations = []
    for reference in request['validation_manifests']:
        rows = [json.loads(line) for line in checked_path(reference).read_text().splitlines()]
        check_rows(rows, 'val')
        validations.append(rows)
    if list(map(len, validations)) != [64, 317]:
        raise ValueError('Fixed validation populations changed')
    isolation = []
    for i, rows in enumerate(populations):
        check_rows(rows, 'train')
        for key in ('image', 'decoded_rgb_sha256'):
            if len({r[key] for r in rows}) != len(rows):
                raise ValueError('Duplicate RGB within a frozen input')
        for j, other in enumerate(populations[:i] + validations):
            counts = overlap(rows, other)
            if any(counts.values()):
                raise ValueError('Cross-batch or validation overlap')
            isolation.append(dict(population=i, comparison=j, overlaps=counts))
    rows = [r for population in populations for r in population]
    counts = dict(Counter(r['dataset'] for r in rows))
    projections = dict(Counter(r['projection'] for r in rows))
    if counts != dict(procthor_empty=40000, structural_rooms=10787):
        raise ValueError('Source counts changed')
    if projections != dict(perspective=49699, orthographic=1088):
        raise ValueError('Projection counts changed')
    output.mkdir(exist_ok=False)
    incoming = output / 'rgb.jsonl.incoming'
    incoming.write_bytes(b''.join(contents))
    incoming.replace(output / 'rgb.jsonl')
    report = dict(stage='frozen_room_full_epoch_pool_ready', request=request,
        manifest=file_ref(output/'rgb.jsonl'), images=len(rows), source_counts=counts,
        projections=projections, groups=len({r['group'] for r in rows}),
        input_manifests=[file_ref(p) for p in paths], isolation=isolation,
        input_lines_preserved_byte_for_byte=True, prior_payload_checks_and_visual_reviews_reused=True,
        all_images_manually_reviewed=False, geometry_loss=False, geometry_labels=False,
        growing_generation_included=False)
    publish_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(checked_json(file_ref(args.request)), args.output.resolve())
    print(json.dumps(dict(report=file_ref(args.output/'report.json'), images=result['images'])))
