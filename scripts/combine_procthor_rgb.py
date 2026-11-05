"""Combine disjoint verified RGB batches without changing their frozen records."""
import argparse
from collections import Counter
import json
from pathlib import Path

from render_procthor_empty_batch import TRAIN_SHA, VAL_SHA
from render_procthor_empty_pilot import publish, sha


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def load_frozen(report_path, split):
    report = json.loads(report_path.read_text())
    expected_sha = TRAIN_SHA if split == 'train' else VAL_SHA
    if (report['stage'] != 'frozen_verified_feature_rgb' or report['split'] != split
            or report['source']['sha256'] != expected_sha
            or report['every_selected_rgb_decoded_and_hashed'] is not True
            or report['original_houses_preserved_except_empty_objects'] is not True
            or report['perspective_only'] is not True or report['agent_visible'] is not False
            or report['geometry_labels'] is not False or report['exact_pixel_duplicates'] != 0
            or report['training_eligible'] is not (split == 'train')):
        raise ValueError('Frozen report contract changed')
    manifest = Path(report['manifest']['path'])
    raw = manifest.read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != report['manifest']['sha256']:
        raise ValueError('Frozen manifest hash changed')
    lines = raw.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    first, end = report['house_range']
    counts = Counter(r['group'] for r in rows)
    expected_groups = {f'procthor10k/{split}/{i:05d}' for i in range(first, end)}
    if (first < 0 or end <= first or end > (10000 if split == 'train' else 1000)
            or set(counts) != expected_groups or len(counts) != report['houses']
            or len(rows) != report['images']
            or report['views_per_house'] != (20 if split == 'train' else 4)
            or set(counts.values()) != {report['views_per_house']}
            or any(not line.endswith(b'\n') for line in lines)):
        raise ValueError('Frozen house range or complete view counts changed')
    for row in rows:
        if (row['dataset'] != 'procthor_empty' or row['split'] != split
                or row['source_archive_sha256'] != expected_sha
                or row['projection'] != 'perspective' or row['agent_visible'] is not False
                or row['geometry_label'] is not None or row['label_role'] != 'feature_only_rgb'
                or row['training_admission'] is not (split == 'train')):
            raise ValueError('Frozen RGB contract changed')
    return rows, lines, dict(report=ref(report_path), manifest=ref(manifest))


def identities(rows):
    result = {key: set() for key in ('image', 'group', 'decoded_rgb_sha256',
                                    'original_house_sha256', 'empty_house_sha256')}
    group_values, owners = {}, {'original_house_sha256': {}, 'empty_house_sha256': {}}
    for row in rows:
        group = row['group']
        house = (row['original_house_sha256'], row['house']['sha256'])
        if group in group_values and group_values[group] != house:
            raise ValueError('House identity changed within group')
        group_values[group] = house
        for key, digest in zip(owners, house):
            if digest in owners[key] and owners[key][digest] != group:
                raise ValueError('Repeated house identity across groups: ' + key)
            owners[key][digest] = group
        for key in result:
            value = row['house']['sha256'] if key == 'empty_house_sha256' else row[key]
            if key in ('image', 'decoded_rgb_sha256') and value in result[key]:
                raise ValueError('Duplicate image identity: ' + key)
            result[key].add(value)
    return result


def combine(batch_reports, validation_report, output):
    if len(batch_reports) < 2:
        raise ValueError('At least two disjoint frozen training batches required')
    rows, lines, inputs = [], [], []
    seen = None
    for path in batch_reports:
        current, current_lines, source = load_frozen(path, 'train')
        current_ids = identities(current)
        if seen is not None:
            for key in seen:
                if seen[key] & current_ids[key]:
                    raise ValueError('Cross-batch overlap: ' + key)
                seen[key].update(current_ids[key])
        else:
            seen = current_ids
        rows.extend(current)
        lines.extend(current_lines)
        inputs.append(source)
    heldout, _, validation = load_frozen(validation_report, 'val')
    heldout_ids = identities(heldout)
    for key in seen:
        if seen[key] & heldout_ids[key]:
            raise ValueError('Train/validation overlap: ' + key)
    # All inputs are hash-bound frozen manifests. Reuse their image readbacks;
    # only new visual-review samples need their PNGs opened again.
    for source in inputs + [validation]:
        for bound in source.values():
            if sha(Path(bound['path'])) != bound['sha256']:
                raise ValueError('Input changed during combination')
    output.mkdir(exist_ok=False)
    incoming, manifest = output / 'rgb.jsonl.incoming', output / 'rgb.jsonl'
    with incoming.open('xb') as stream:
        stream.writelines(lines)
    incoming.rename(manifest)
    report = dict(stage='combined_verified_rgb_sample_review_pending', inputs=inputs,
        validation=validation, manifest=ref(manifest), images=len(rows),
        houses=len(seen['group']), validation_images=len(heldout),
        cross_batch_exact_overlaps={key: 0 for key in seen},
        train_validation_exact_overlaps={key: 0 for key in seen},
        input_lines_preserved_byte_for_byte=True, previous_image_readbacks_reused=True,
        perspective_only=True, agent_visible=False, geometry_labels=False,
        training_launched=False, current_training_manifest_changed=False,
        visual_review='New batch sampled review still required; no all-image certification',
        isolation_scope='Exact identities against declared frozen validation selection; not near-duplicate detection',
        generator=ref(Path(__file__).resolve()))
    publish(output / 'report.json', report)
    print(json.dumps(dict(report=ref(output / 'report.json'), images=len(rows),
                         houses=report['houses'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-report', type=Path, action='append', required=True)
    parser.add_argument('--validation-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    combine([p.resolve() for p in args.batch_report], args.validation_report.resolve(),
            args.output.resolve())
