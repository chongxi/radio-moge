"""Stream the frozen RGB inventory into a candidate pool with scoped exclusions.

This prepares metadata, not a training pass or a sampling policy. Geometry-only
exclusions stay in the RGB pool. RGB exclusions need an explicit, byte-verified
constant-image review; no pixel-quality decision is inferred from depth support.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import publish_json
from radio_moge.teacher_rgb_review import verify_constant_rgb


RGB_FIELDS = ('image', 'sha256', 'sha256_kind', 'decoded_rgb_sha256', 'dataset',
              'group', 'split', 'image_rotation_cw', 'width', 'height', 'official_split')


def load_decisions(ref):
    decisions, evidence = {}, {}
    with checked_path(ref).open() as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            ordinal = item.get('corpus_ordinal')
            scope = item.get('scope')
            if (type(ordinal) is not int or ordinal < 0 or ordinal in decisions
                    or scope not in ('geometry_supervision', 'geometry_and_rgb_replay')
                    or item.get('decision') != 'exclude' or not item.get('reason')
                    or item.get('rgb_replay_excluded') is not (scope == 'geometry_and_rgb_replay')
                    or not item.get('evidence')):
                raise ValueError('Unique, explicitly scoped and evidenced input decisions required')
            for proof in item['evidence']:
                key = json.dumps(proof, sort_keys=True)
                if key not in evidence:
                    checked_path(proof)
                    evidence[key] = proof
            if item['rgb_replay_excluded']:
                review_ref = item.get('rgb_input_review')
                if review_ref not in item['evidence']:
                    raise ValueError('RGB exclusion requires its own reviewed input evidence')
                review = checked_json(review_ref)
                if (review.get('stage') != 'rgb_input_review_complete' or review.get('passed') is not True
                        or review.get('decision') != 'exclude_exactly_spatially_constant_rgb'
                        or review.get('training_admission') is not False):
                    raise ValueError('Only explicitly reviewed constant RGB exclusions are supported')
                matches = [r for r in review['excluded_images'] if r['corpus_ordinal'] == ordinal]
                if (len(matches) != 1 or ordinal not in review['excluded_corpus_ordinals']
                        or any(matches[0][k] != item[k] for k in ('image', 'sha256'))
                        or matches[0].get('spatially_constant') is not True
                        or matches[0]['channel_min'] != matches[0]['channel_max']):
                    raise ValueError('RGB exclusion identity differs from its constant-image review')
                item = dict(item, reviewed_constant=matches[0])
            decisions[ordinal] = item
    return decisions, list(evidence.values())


def project_rgb(ordinal, dataset, group, image, checksum, encoded, manifest):
    row = json.loads(encoded)
    if (row.get('split') != 'train' or row.get('dataset') != dataset
            or not group or row.get('group') != group or row.get('sha256') != checksum
            or os.path.abspath(manifest.parent / row['image']) != image
            or (dataset == 'irs' and row.get('official_split') != 'train')):
        raise ValueError('RGB replay row differs from the frozen training identity or split')
    rotation = row.get('image_rotation_cw', 0)
    if type(rotation) is not int or rotation not in (0, 90, 180, 270):
        raise ValueError('Invalid original RGB orientation')
    rgb = {key: row[key] for key in RGB_FIELDS if key in row}
    rgb.update(image=image, corpus_ordinal=ordinal)
    return rgb


def run(config, decisions_ref, attempt):
    root = Path(config['root'])
    source_ref = file_ref(root / 'inventory.json')
    source = checked_json(source_ref)
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    manifest_ref = dict(path=source['manifest'], sha256=source['manifest_sha256'])
    if (source['stage'] != 'inventory_complete'
            or source['manifest_sha256'] != config['manifest_sha256']
            or source['dataset_counts'] != config['expected_counts']
            or source['images'] != sum(config['expected_counts'].values())
            or source['duplicate_encoded_image_rows'] != 0):
        raise ValueError('Complete matching frozen inventory required')
    inventory, manifest = checked_path(inventory_ref), checked_path(manifest_ref)
    decisions, evidence = load_decisions(decisions_ref)
    output = root / attempt
    output.mkdir(exist_ok=False)
    incoming = output / 'candidates-rgb.jsonl.incoming'
    counts, excluded, retained, groups, seen = Counter(), Counter(), Counter(), Counter(), set()
    total = 0
    try:
        with readonly_db(inventory) as db, incoming.open('x') as file:
            for fields in db.execute('SELECT ordinal,dataset,scene_group,image,sha256,row_json '
                                     'FROM images ORDER BY ordinal'):
                ordinal, dataset, group, image, checksum, _ = fields
                if ordinal != total or dataset not in config['expected_counts']:
                    raise ValueError('Frozen inventory ordinals or source scope differ')
                total += 1
                rgb = project_rgb(*fields, manifest)
                decision = decisions.get(ordinal)
                if decision:
                    if any(decision[k] != rgb[k] for k in ('image', 'sha256', 'dataset')):
                        raise ValueError('Exclusion changed a frozen RGB identity')
                    seen.add(ordinal)
                    if decision['rgb_replay_excluded']:
                        reviewed = decision['reviewed_constant']
                        if reviewed['group'] != group:
                            raise ValueError('Reviewed blank image scene differs')
                        verify_constant_rgb(rgb, reviewed['channel_min'])
                        excluded[dataset] += 1
                        continue
                    retained[dataset] += 1
                file.write(json.dumps(rgb, separators=(',', ':'), allow_nan=False) + '\n')
                counts[dataset] += 1
                groups[(dataset, group)] += 1
            file.flush()
            os.fsync(file.fileno())
        if (seen != set(decisions) or total != source['images']
                or dict(counts + excluded) != source['dataset_counts']
                or set(counts) != set(config['expected_counts'])):
            raise ValueError('Every frozen input and scoped decision must be accounted exactly once')
        # Recheck the exact inputs before publishing any candidate manifest.
        for ref in (source_ref, inventory_ref, manifest_ref, decisions_ref, *evidence):
            checked_path(ref)
        summary = []
        for dataset in sorted(counts):
            sizes = [n for (d, _), n in groups.items() if d == dataset]
            summary.append(dict(dataset=dataset, candidate_images=counts[dataset], groups=len(sizes),
                largest_group_images=max(sizes), largest_group_fraction=max(sizes) / counts[dataset],
                uniform_image_sampling_share=counts[dataset] / sum(counts.values()),
                physical_independence_of_groups_verified=False))
        publish_json(output / 'groups.json', [dict(dataset=d, group=g, images=n)
                                             for (d, g), n in sorted(groups.items())])
        os.rename(incoming, output / 'candidates-rgb.jsonl')
        report = dict(stage='rgb_replay_candidates_prepared_not_accepted', completed_at=time.time(),
            frozen_images=total, candidate_images=sum(counts.values()), candidate_counts=dict(counts),
            rgb_excluded_counts=dict(excluded), geometry_only_retained_counts=dict(retained),
            rgb_excluded_ordinals=sorted(k for k, v in decisions.items() if v['rgb_replay_excluded']),
            geometry_only_retained_ordinals=sorted(k for k, v in decisions.items()
                                                   if not v['rgb_replay_excluded']),
            sources=summary, source_report=source_ref, inventory=inventory_ref,
            source_manifest=manifest_ref, decisions=decisions_ref, decision_evidence=evidence,
            manifest=file_ref(output / 'candidates-rgb.jsonl'), groups=file_ref(output / 'groups.json'),
            generator=file_ref(Path(__file__)), inventory_rows_streamed_once=True,
            all_rgb_pixels_reread=False, constant_rgb_exclusions_reverified=sum(excluded.values()),
            geometry_labels_read=False, canonical_heldout_comparison_complete=False,
            sampling_policy_applied=False, source_quality_accepted=False,
            training_admission=False, optimizer_updates=0)
        publish_json(output / 'report.json', report)
        print(json.dumps({k: report[k] for k in ('stage', 'candidate_counts', 'rgb_excluded_counts',
                                               'geometry_only_retained_counts')}), flush=True)
        return report
    except BaseException as error:
        publish_json(output / 'failure.json', dict(stage='failed', training_admission=False,
            error=type(error).__name__, message=str(error)[:1000]))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use an attempt basename')
    run(json.loads(args.config.read_text()), file_ref(args.decisions), args.attempt)
