"""Freeze receipt-complete DSLR views without modifying the live download.

Only public receipt/inventory metadata is read from control; no official downloader
source, configuration, credentials or URLs. Output is a new sibling dataset.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image


def atomic(path, value):
    tmp = path.with_name(path.name + '.incoming')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def receipt_ready(control, dataset, relative, cutoff, archive=False):
    file = control / 'receipts' / (sha(relative.encode()) + '.json')
    if not file.is_file():
        return None
    receipt = json.loads(file.read_text())
    if receipt['path'] != relative or receipt['completed_at'] > cutoff:
        return None
    if archive:
        file = control / 'extractions' / (sha(str(dataset / relative).encode()) + '.json')
        if not file.is_file():
            return None
        extraction = json.loads(file.read_text())
        if (extraction.get('stage') != 'passed' or not extraction.get('crc_checked')
                or extraction['completed_at'] > cutoff):
            return None
    else:
        contents = (dataset / relative).read_bytes()
        if sha(contents) != receipt['sha256'] or len(contents) != receipt['bytes']:
            raise ValueError('Completed metadata differs from its receipt')
    return receipt


def select_scenes(dataset, control, cutoff):
    inventory = json.loads((control / 'inventory.json').read_text())
    split_names = ['nvs_sem_train', 'nvs_sem_val', 'nvs_test', 'sem_test', 'nvs_test_iphone']
    splits = {}
    for name in split_names:
        relative = f'splits/{name}.txt'
        if not receipt_ready(control, dataset, relative, cutoff):
            raise ValueError('Official split metadata is not complete')
        raw = (dataset / relative).read_bytes()
        if sha(raw) != inventory['metadata_sha256'][relative]:
            raise ValueError('Official split differs from frozen inventory')
        splits[name] = set(raw.decode().split())
    for i, name in enumerate(split_names):
        for other in split_names[i + 1:]:
            if splits[name] & splits[other]:
                raise ValueError('Official scene splits overlap')
    eligible = []
    for split in ['nvs_sem_train', 'nvs_sem_val']:
        for scene in sorted(splits[split]):
            prefix = f'data/{scene}/dslr/'
            required = ['train_test_lists.json', 'nerfstudio/transforms_undistorted.zip',
                        'resized_undistorted_images.zip', 'resized_undistorted_masks.zip']
            if all(receipt_ready(control, dataset, prefix + r, cutoff, r.endswith('.zip'))
                   for r in required):
                eligible.append(dict(scene=scene, split=split))
    return eligible, {k: len(v) for k, v in splits.items()}


def candidate_views(dataset, scene, split):
    directory = dataset / 'data' / scene / 'dslr'
    views = json.loads((directory / 'train_test_lists.json').read_text())
    camera = json.loads((directory / 'nerfstudio/transforms_undistorted.json').read_text())
    if camera['camera_model'] != 'PINHOLE' or any(camera.get(k, 0) != 0
                                               for k in ['k1', 'k2', 'k3', 'k4']):
        raise ValueError('Expected undistorted pinhole cameras')
    train, test = set(views['train']), set(views['test'])
    if train & test:
        raise ValueError('Official train/test views overlap')
    frames = {r['file_path']: r for r in camera['frames']}
    if len(frames) != len(camera['frames']):
        raise ValueError('Duplicate camera frames')
    result = []
    for name in sorted(train):
        frame = frames.get(name)
        if frame is None or frame.get('is_bad', False):
            continue
        mask = frame.get('mask_path', Path(name).with_suffix('.png').name)
        if Path(name).name != name or Path(mask).name != mask:
            raise ValueError('Unexpected nested/unsafe camera path')
        result.append(dict(image=str(directory / 'resized_undistorted_images' / name),
                           mask=str(directory / 'resized_undistorted_masks' / mask),
                           dataset='scannetpp_dslr', group='scannetpp:' + scene,
                           split='train' if split == 'nvs_sem_train' else 'val',
                           scene=scene, view=name, official_split=split,
                           official_view_split='train', width=camera['w'], height=camera['h'],
                           camera_width_height_fx_fy_cx_cy=[camera[k] for k in
                                                          ['w', 'h', 'fl_x', 'fl_y', 'cx', 'cy']],
                           transform_matrix=frame['transform_matrix']))
    return result


def fixed_validation(rows, count=128):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group']].append(row)
    pools = [sorted(groups[g], key=lambda r: sha(r['image'].encode())) for g in sorted(groups)]
    result = []
    for i in range(max(map(len, pools), default=0)):
        for pool in pools:
            if i < len(pool):
                result.append(pool[i])
                if count is not None and len(result) == count:
                    return result
    if count is None:
        return result
    raise ValueError('Insufficient completed held-out validation views')


def internal_holdout(scene):
    # Independent of completion order and snapshot cutoff; retained in later phases.
    return int(sha(('scannetpp-internal-val-v1:' + scene).encode())[:8], 16) % 20 == 0


def bounded_inspection(rows, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(rows), workers * 2):
            batch = rows[start:start + workers * 2]
            yield from zip(batch, executor.map(inspect_image, batch), strict=True)


def inspect_image(row):
    raw = Path(row['image']).read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()  # Fully decode; a valid header alone is insufficient.
        if image.size != (row['width'], row['height']) or image.getexif().get(274, 1) != 1:
            raise ValueError('Camera/RGB dimensions or orientation differ')
        if min(image.size) < 480:
            return None, 'too_small'
    with Image.open(row['mask']) as image:
        image.load()
        if image.size != (row['width'], row['height']):
            raise ValueError('Camera/mask dimensions differ')
        mask = np.asarray(image)
        if mask.ndim != 2:
            raise ValueError('Expected single-channel validity mask')
        fraction = float((mask == 255).mean())
    # Avoid exposing masked borders/large anonymous regions to an unmasked feature loss.
    if fraction < .98:
        return None, 'valid_fraction_below_0.98'
    return dict(row, sha256=sha(raw), sha256_kind='encoded_file_bytes',
                valid_fraction=fraction, encoded_bytes=len(raw)), None


def prepare(dataset, control, original, output, cutoff, workers=1):
    if workers not in range(1, 5):
        raise ValueError('Use one to four bounded CPU decoding workers')
    if output.exists():
        raise FileExistsError('Use a new snapshot; do not modify a published or partial snapshot')
    if output.parent != original.parent:
        raise ValueError('Sibling destination is required to retain original relative image paths')
    output.mkdir(mode=0o700)
    counts = Counter()
    started = time.time()

    def status(stage, **extra):
        atomic(output / 'status.json', dict(stage=stage, pid=os.getpid(), started_at=started,
               updated_at=time.time(), cutoff=cutoff, counts=dict(counts), **extra))

    status('freezing_receipts')
    try:
        scenes, splits = select_scenes(dataset, control, cutoff)
        atomic(output / 'snapshot.json', dict(cutoff=cutoff, scenes=scenes, splits=splits,
               source=str(dataset), original=str(original), modality='undistorted DSLR',
               validation_limit=128, mask_valid_fraction_min=.98, all_train_views=True,
               internal_validation_rule='sha256(scannetpp-internal-val-v1:scene) first8 hex %20 ==0',
               internal_validation_scenes=[r['scene'] for r in scenes
                                           if r['split']=='nvs_sem_train' and internal_holdout(r['scene'])],
               decoding_workers=workers))
        heldout_hashes = set()
        for file in sorted(original.glob('*.jsonl')):
            if not (file.name.startswith('val') or file.name.startswith('test')):
                continue
            for line in file.open():
                row = json.loads(line)
                if row.get('sha256'):
                    heldout_hashes.add(row['sha256'])
        old_hashes = set()
        with (original / 'train.jsonl').open('rb') as source, (output / 'train.jsonl.incoming').open('wb') as target:
            original_hash = hashlib.sha256()
            for line in source:
                row = json.loads(line)
                if row.get('sha256'):
                    old_hashes.add(row['sha256'])
                counts['original_train'] += 1
                original_hash.update(line)
                target.write(line)
        candidates = []
        for scene in scenes:
            if scene['split'] != 'nvs_sem_train':
                continue  # Official validation/test scenes remain untouched for future evaluation.
            rows = candidate_views(dataset, **scene)
            if internal_holdout(scene['scene']):
                rows = [dict(r, split='val', internal_scene_holdout=True) for r in rows]
            candidates.extend(rows)
        validation = fixed_validation([r for r in candidates if r['split'] == 'val'], None)
        if len(validation) < 128:
            raise ValueError('Not enough independent internal-validation scenes/views')
        selected = validation + [r for r in candidates if r['split'] == 'train']
        status('decoding', candidate_train=len(selected) - len(validation), candidate_val=len(validation))
        seen = set(old_hashes) | heldout_hashes
        val_rows, new_scenes = [], set()
        last = time.monotonic()
        with (output / 'train.jsonl.incoming').open('a') as merged, (output / 'scannetpp-train.jsonl.incoming').open('w') as new:
            # Validation first, round-robin across permanently held-out scenes.
            # Train decoding is bounded independently; no million-future eager queue.
            def inspected():
                for candidate in validation:
                    if len(val_rows) == 128:
                        break
                    yield candidate, inspect_image(candidate)
                training = [r for r in selected if r['split'] == 'train']
                yield from bounded_inspection(training, workers)

            for candidate, (row, reason) in inspected():
                counts['inspected'] += 1
                if row is None:
                    counts[reason] += 1
                elif row['sha256'] in seen:
                    if row['split'] == 'val':
                        raise ValueError('Validation duplicates another source/view')
                    counts['duplicate_filtered'] += 1
                else:
                    seen.add(row['sha256'])
                    if row['split'] == 'val':
                        val_rows.append(row)
                    else:
                        text = json.dumps(row, sort_keys=True, allow_nan=False) + '\n'
                        merged.write(text)
                        new.write(text)
                        counts['scannetpp_train'] += 1
                        new_scenes.add(row['scene'])
                if time.monotonic() - last > 10:
                    status('decoding', current_scene=candidate['scene'])
                    last = time.monotonic()
        if counts['scannetpp_train'] < 1 or len(val_rows) != 128:
            raise ValueError('No valid training data or incomplete validation')
        for name in ['train', 'scannetpp-train']:
            (output / (name + '.jsonl.incoming')).replace(output / (name + '.jsonl'))
        (output / 'val-scannetpp.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in val_rows))
        hashes = {}
        for name in ['train.jsonl', 'scannetpp-train.jsonl', 'val-scannetpp.jsonl']:
            digest = hashlib.sha256()
            with (output / name).open('rb') as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b''):
                    digest.update(chunk)
            hashes[name] = digest.hexdigest()
        atomic(output / 'verification.json', dict(stage='passed', completed_at=time.time(),
               counts=dict(counts), training_scenes=len(new_scenes), validation_images=128,
               validation_scenes=len({r['scene'] for r in val_rows}), cutoff=cutoff,
               manifest_sha256=hashes, original_train_sha256=original_hash.hexdigest(),
               original_prefix_unchanged=True, official_test_scenes_excluded=True,
               official_test_views_excluded=True, decoded_all_accepted_images=True,
               duplicate_hashes_filtered=True, old_heldouts_excluded=True,
               internal_validation_scenes_permanently_excluded=True,
               source_files_modified=False, training_ready_modality='undistorted DSLR RGB'))
        status('passed')
    except Exception as error:
        status('failed', error=str(error), error_type=type(error).__name__)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['dataset', 'control', 'original', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--cutoff', type=float, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    prepare(**vars(args))
