"""Append newly completed DSLR scenes while preserving a verified snapshot byte-for-byte.

Only download receipt, extraction and inventory metadata is consulted. Existing
images are not decoded again; new images pass the same camera/mask/decode checks.
The live download and the parent snapshot are never modified.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from prepare_scannetpp_snapshot import (
    atomic, bounded_inspection, candidate_views, internal_holdout, select_scenes,
)


def expand(dataset, control, previous, heldout_root, output, cutoff, workers=2):
    if workers not in range(1, 5):
        raise ValueError('Use one to four bounded CPU decoding workers')
    if output.exists() or output.parent != previous.parent:
        raise ValueError('Use a new sibling snapshot destination')
    parent = json.loads((previous / 'verification.json').read_text())
    frozen = json.loads((previous / 'snapshot.json').read_text())
    if parent['stage'] != 'passed' or not parent['decoded_all_accepted_images']:
        raise ValueError('Parent snapshot must already be verified')
    if cutoff <= frozen['cutoff']:
        raise ValueError('Expansion cutoff must follow the parent cutoff')
    output.mkdir(mode=0o700)
    counts, started = Counter(), time.time()

    def status(stage, **extra):
        atomic(output / 'status.json', dict(stage=stage, pid=os.getpid(),
               started_at=started, updated_at=time.time(), cutoff=cutoff,
               counts=dict(counts), **extra))

    status('freezing_receipts')
    try:
        scenes, splits = select_scenes(dataset, control, cutoff)
        previous_scenes = {(r['scene'], r['split']) for r in frozen['scenes']}
        current_scenes = {(r['scene'], r['split']) for r in scenes}
        if not previous_scenes <= current_scenes:
            raise ValueError('Previously verified scenes are missing from completed receipts')
        new_scenes = [r for r in scenes if (r['scene'], r['split']) not in previous_scenes]
        admitted = [r for r in new_scenes if r['split'] == 'nvs_sem_train'
                    and not internal_holdout(r['scene'])]
        if not admitted:
            raise ValueError('No newly completed eligible training scenes')
        snapshot = dict(cutoff=cutoff, scenes=scenes, splits=splits, source=str(dataset),
                        original=str(previous), heldout_root=str(heldout_root),
                        parent_verification=str(previous / 'verification.json'),
                        new_scenes=new_scenes, new_training_scenes=admitted,
                        modality='undistorted DSLR', validation_limit=128,
                        mask_valid_fraction_min=.98, all_train_views=True,
                        internal_validation_rule=frozen['internal_validation_rule'],
                        internal_validation_scenes=[r['scene'] for r in scenes
                            if r['split'] == 'nvs_sem_train' and internal_holdout(r['scene'])],
                        decoding_workers=workers, existing_images_redecoded=False,
                        fixed_validation_source=str(previous / 'val-scannetpp.jsonl'))
        atomic(output / 'snapshot.json', snapshot)
        validation_bytes = (previous / 'val-scannetpp.jsonl').read_bytes()
        validation_sha = hashlib.sha256(validation_bytes).hexdigest()
        if validation_sha != parent['manifest_sha256']['val-scannetpp.jsonl']:
            raise ValueError('Fixed validation differs from parent proof')
        validation = [json.loads(line) for line in validation_bytes.splitlines()]
        if len(validation) != 128 or any(not internal_holdout(r['scene']) for r in validation):
            raise ValueError('Fixed validation must retain permanent held-out scenes')
        seen = {r['sha256'] for r in validation}
        for file in sorted(heldout_root.glob('*.jsonl')):
            if file.name.startswith(('val', 'test')):
                with file.open() as source:
                    for line in source:
                        value = json.loads(line).get('sha256')
                        if value:
                            seen.add(value)
        expected_count = parent['counts']['original_train'] + parent['counts']['scannetpp_train']
        previous_hash = hashlib.sha256()
        merged_hash = hashlib.sha256()
        status('copying_verified_parent')
        with (previous / 'train.jsonl').open('rb') as source, \
                (output / 'train.jsonl.incoming').open('wb') as merged:
            for line in source:
                row = json.loads(line)
                if row.get('sha256'):
                    seen.add(row['sha256'])
                previous_hash.update(line)
                merged_hash.update(line)
                merged.write(line)
                counts['original_train'] += 1
        if (previous_hash.hexdigest() != parent['manifest_sha256']['train.jsonl']
                or counts['original_train'] != expected_count):
            raise ValueError('Parent manifest differs from its verified bytes/count')
        delta_hash, accepted_scenes = hashlib.sha256(), set()
        status('decoding_new_scenes', new_training_scenes=len(admitted))
        with (output / 'train.jsonl.incoming').open('ab') as merged, \
                (output / 'scannetpp-train.jsonl.incoming').open('wb') as delta:
            for scene in admitted:
                candidates = candidate_views(dataset, **scene)
                for candidate, (row, reason) in bounded_inspection(candidates, workers):
                    counts['inspected'] += 1
                    if row is None:
                        counts[reason] += 1
                    elif row['sha256'] in seen:
                        counts['duplicate_filtered'] += 1
                    else:
                        seen.add(row['sha256'])
                        encoded = (json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode()
                        merged.write(encoded)
                        delta.write(encoded)
                        merged_hash.update(encoded)
                        delta_hash.update(encoded)
                        counts['scannetpp_train'] += 1
                        accepted_scenes.add(row['scene'])
                status('decoding_new_scenes', current_scene=scene['scene'],
                       accepted_new_training_scenes=len(accepted_scenes))
        if not counts['scannetpp_train']:
            raise ValueError('No new images pass the training filters')
        for name in ['train', 'scannetpp-train']:
            (output / (name + '.jsonl.incoming')).replace(output / (name + '.jsonl'))
        shutil.copyfile(previous / 'val-scannetpp.jsonl', output / 'val-scannetpp.jsonl')
        if (output / 'val-scannetpp.jsonl').read_bytes() != validation_bytes:
            raise ValueError('Fixed validation copy changed')
        hashes = {'train.jsonl': merged_hash.hexdigest(),
                  'scannetpp-train.jsonl': delta_hash.hexdigest(),
                  'val-scannetpp.jsonl': validation_sha}
        atomic(output / 'verification.json', dict(stage='passed', completed_at=time.time(),
               counts=dict(counts), new_training_scenes=len(accepted_scenes),
               validation_images=128, validation_scenes=len({r['scene'] for r in validation}),
               cutoff=cutoff, manifest_sha256=hashes,
               original_train_sha256=previous_hash.hexdigest(), original_prefix_unchanged=True,
               fixed_validation_byte_identical=True, previous_snapshot=str(previous),
               parent_preparation_proof=str(previous / 'verification.json'),
               parent_accepted_images_redecoded=False, decoded_all_new_accepted_images=True,
               decoded_all_accepted_images=True,
               inherited_decode_proof=parent['manifest_sha256']['train.jsonl'],
               official_test_scenes_excluded=True, official_test_views_excluded=True,
               duplicate_hashes_filtered=True, old_heldouts_excluded=True,
               internal_validation_scenes_permanently_excluded=True,
               source_files_modified=False, training_ready_modality='undistorted DSLR RGB',
               delta_manifest='scannetpp-train.jsonl'))
        status('passed')
    except Exception as error:
        status('failed', error=str(error), error_type=type(error).__name__)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['dataset', 'control', 'previous', 'heldout-root', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cutoff', type=float, required=True)
    parser.add_argument('--workers', type=int, default=2)
    expand(**vars(parser.parse_args()))
