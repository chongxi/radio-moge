"""Freeze only closed ProcTHOR houses; verify every selected RGB before training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from render_procthor_empty_batch import houses_from, source_spec
from render_procthor_empty_pilot import publish, sha


def freeze(root, batch, output, split, count, views):
    if split not in ('train', 'val') or count < 1 or views not in (4, 20):
        raise ValueError('Explicit official split, complete houses and four/twenty views required')
    archive, archive_sha, size = source_spec(root, split)
    if count > size or sha(archive) != archive_sha:
        raise ValueError('Pinned official house source differs')
    selected, summaries, originals = {}, [], {}
    for ordinal, original, line_sha in houses_from(archive, 0, count):
        name = f'{split}-{ordinal:05d}'
        house_path = batch / 'houses' / (name + '.json')
        receipt = batch / 'house-reports' / (name + '.json')
        report = json.loads(receipt.read_text())
        house = json.loads(house_path.read_text())
        group = f'procthor10k/{split}/{ordinal:05d}'
        if (house != dict(original, objects=[]) or report['house_id'] != group
                or report['archive_sha256'] != archive_sha
                or report['source_line_sha256'] != line_sha
                or report['empty_house_sha256'] != sha(house_path)
                or len(report['images']) != 20 or report.get('create_house_error')):
            raise ValueError('House is not a closed unchanged empty structural scene: '+name)
        images = report['images'] if views == 20 else report['images'][::5]
        for row in images:
            if row['image'] in selected:
                raise ValueError('Duplicate house image')
            selected[row['image']] = (group, row['sha256'], house_path, line_sha)
        summaries.append(dict(path=str(receipt), sha256=sha(receipt), group=group,
                              selected_images=len(images)))
        originals[group] = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    if len(summaries) != count:
        raise ValueError('Incomplete source house range')
    records, pixels, groups = [], set(), set()
    with (batch / 'rgb-candidates.jsonl').open() as file:
        for line in file:
            row = json.loads(line)
            if row['image'] not in selected:
                continue
            group, image_sha, house_path, line_sha = selected.pop(row['image'])
            path = Path(row['image'])
            if (row['dataset'] != 'procthor_empty' or row['split'] != split or row['group'] != group
                    or row['source_archive_sha256'] != archive_sha
                    or row['source_line_sha256'] != line_sha
                    or row['projection'] != 'perspective' or row['agent_visible'] is not False
                    or row['geometry_label'] is not None or row['training_admission'] is not False
                    or row['house'] != dict(path=str(house_path), sha256=sha(house_path))
                    or path.parent.resolve() != (batch / 'rgb').resolve()
                    or sha(path) != image_sha or row['sha256'] != image_sha):
                raise ValueError('RGB identity or rendering contract changed')
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (640, 480) or image.getexif().get(274, 1) != 1:
                    raise ValueError('Unexpected rendered perspective RGB format')
                array = np.array(image)
            pixel_sha = hashlib.sha256(array.tobytes()).hexdigest()
            if pixel_sha != row['decoded_rgb_sha256'] or pixel_sha in pixels:
                raise ValueError('Changed or duplicate decoded RGB')
            if float(array.astype(np.float32).std(axis=(0, 1)).mean()) < .3:
                raise ValueError('Uniform rendered RGB')
            pixels.add(pixel_sha)
            groups.add(group)
            records.append(dict(row, training_admission=split == 'train',
                label_role='feature_only_rgb', original_house_sha256=originals[group],
                candidate_record_sha256=hashlib.sha256(line.encode()).hexdigest()))
            if not selected:
                break  # Never inspect the active renderer's unfinished last house.
    if selected or len(records) != count * views or len(groups) != count:
        raise ValueError('Frozen selection has missing images or house groups')
    output.mkdir(exist_ok=False)
    manifest = output / 'rgb.jsonl'
    incoming = output / 'rgb.jsonl.incoming'
    with incoming.open('x') as file:
        for row in records:
            file.write(json.dumps(row, allow_nan=False) + '\n')
    incoming.rename(manifest)
    report = dict(stage='frozen_verified_feature_rgb', split=split, images=len(records),
        houses=len(groups), views_per_house=views, house_range=[0, count],
        manifest=dict(path=str(manifest), sha256=sha(manifest)), house_receipts=summaries,
        source=dict(path=str(archive), sha256=archive_sha),
        source_batch=str(batch), every_selected_rgb_decoded_and_hashed=True,
        original_houses_preserved_except_empty_objects=True, agent_visible=False,
        exact_pixel_duplicates=0, perspective_only=True, geometry_labels=False,
        training_eligible=split == 'train', visual_review_scope='Separate sampled review; not every image inspected',
        generator=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__))))
    publish(output / 'report.json', report)
    print(json.dumps(dict(images=len(records), houses=len(groups), split=split,
                          manifest=report['manifest'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'batch', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--houses', type=int, required=True)
    parser.add_argument('--views-per-house', type=int, choices=(4, 20), required=True)
    args = parser.parse_args()
    freeze(args.root.resolve(), args.batch.resolve(), args.output.resolve(),
           args.split, args.houses, args.views_per_house)
