"""Read-only laser-GT acceptance audit; evidence goes to a NEW output directory.

Separates transfer/conversion integrity, coverage diagnostics and physical
correctness. No model inference, GT production, source edits or automatic
training admission. Contact sheets are diagnostic plots, not generated labels.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from radio_moge.checkpoint import atomic_json
from radio_moge.data import read_manifest
if __package__:
    from .audit_scannetpp_laser_gt import audit, digest
else:
    from audit_scannetpp_laser_gt import audit, digest


SPLITS = ('nvs_sem_train', 'nvs_sem_val', 'nvs_test', 'sem_test', 'nvs_test_iphone')
MODALITIES = {'rgb': ('images', '.jpg'), 'range': ('depth', '.png'),
              'azimuth': ('azim', '.png'), 'polar': ('elev', '.png'),
              'anonymous_mask': ('anon_mask', '.png')}


def heldout(scene):
    return int(hashlib.sha256(('scannetpp-internal-val-v1:' + scene).encode()).hexdigest()[:8], 16) % 20 == 0


def source_inventory(dataset, prepared):
    splits = {name: set((dataset / 'splits' / (name + '.txt')).read_text().split()) for name in SPLITS}
    for i, name in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            if splits[name] & splits[other]:
                raise ValueError('Official scene split overlap')
    prepared_scenes = {}
    for split, rows in prepared.items():
        scenes = {row['scene'] for row in rows}
        if not scenes <= splits['nvs_sem_train']:
            raise ValueError('Prepared GT contains a forbidden official scene')
        if any(heldout(scene) != (split == 'val') for scene in scenes):
            raise ValueError('Prepared GT violates the permanent internal holdout')
        prepared_scenes[split] = scenes
    if prepared_scenes['train'] & prepared_scenes['val']:
        raise ValueError('Prepared train/validation scenes overlap')
    if {r['sha256'] for r in prepared['train']} & {r['sha256'] for r in prepared['val']}:
        raise ValueError('Prepared train/validation RGB hashes overlap')
    records = []
    for scene in sorted(splits['nvs_sem_train']):
        directory = dataset / 'data' / scene
        stems = {key: {p.stem for p in (directory / 'panocam' / folder).glob('*' + suffix)}
                 for key, (folder, suffix) in MODALITIES.items()}
        complete = set.intersection(*stems.values())
        union = set.union(*stems.values())
        records.append(dict(scene=scene, internal_split='val' if heldout(scene) else 'train',
                            complete_scan_pairs=len(complete),
                            unmatched_scan_ids=sorted(union - complete),
                            modality_counts={key: len(value) for key, value in stems.items()},
                            scanner_poses_available=(directory / 'scans/scanner_poses.json').is_file(),
                            prepared_scene=any(scene in x for x in prepared_scenes.values())))
    eligible = [r for r in records if r['internal_split'] == 'train']
    return dict(scope='File inventory only, NOT quality approval of unprocessed scans',
                official_split_counts={key: len(value) for key, value in splits.items()},
                split_sha256={name: digest(dataset / 'splits' / (name + '.txt')) for name in SPLITS},
                prepared_scene_counts={key: len(value) for key, value in prepared_scenes.items()},
                scene_and_rgb_hash_disjoint=True,
                eligible_train_scenes=len(eligible),
                eligible_train_complete_scan_pairs=sum(r['complete_scan_pairs'] for r in eligible),
                eligible_train_scenes_with_scanner_poses=sum(r['scanner_poses_available'] for r in eligible),
                scenes=records)


def quality_stats(depth, valid, rgb):
    if depth.ndim != 2 or valid.shape != depth.shape or rgb.shape != (*depth.shape, 3):
        raise ValueError('GT dimensions differ')
    if not valid.any() or not np.isfinite(depth[valid]).all() or (depth[valid] <= 0).any():
        raise ValueError('Accepted GT depth must be finite and positive')
    if np.any(depth[~valid] != 0):
        raise ValueError('Invalid GT pixels must stay zero, not filled')
    values = depth[valid]
    quantiles = np.quantile(values, [.01, .05, .5, .95, .99])
    grid = [[float(cell.mean()) for cell in np.array_split(part, 4, axis=1)]
            for part in np.array_split(valid, 4, axis=0)]
    flags = []
    if valid.mean() < .5:
        flags.append('less_than_half_measured_support')
    if quantiles[3] - quantiles[1] < .05:
        flags.append('very_small_depth_spread_not_necessarily_wrong')
    if np.mean(values < .15) > .01:
        flags.append('near_range_pixels_need_review')
    return dict(valid_fraction=float(valid.mean()), valid_pixels=int(valid.sum()),
                depth_z_m_p01_p05_p50_p95_p99=quantiles.tolist(),
                fraction_below_015m=float(np.mean(values < .15)),
                fraction_above_20m=float(np.mean(values > 20)),
                support_4x4=grid, rgb_channel_std=np.std(rgb.reshape(-1, 3), axis=0).tolist(),
                invalid_depth_unfilled=True, review_flags=flags)


def depth_plot(depth, valid):
    # Fixed 0.1--20 m log scale across every view; gray = no measurement.
    x = np.clip(np.log(np.maximum(depth, .1) / .1) / np.log(200), 0, 1)
    palette = np.array([[32, 20, 100], [20, 130, 180], [40, 180, 100],
                        [230, 210, 50], [220, 60, 30]])
    rgb = np.stack([np.interp(x, np.linspace(0, 1, len(palette)), palette[:, c]) for c in range(3)], -1)
    rgb[~valid] = 110
    return Image.fromarray(rgb.astype(np.uint8))


def inspect_pairs(rows, split, output):
    results, pages = [], []
    # 18 pairs per sheet, each with unchanged RGB beside GT depth/missing mask.
    for start in range(0, len(rows), 18):
        page = Image.new('RGB', (960, 30 + 6 * 136), 'white')
        draw = ImageDraw.Draw(page)
        draw.text((6, 7), f'{split}: RGB | laser Z, fixed log 0.1--20 m; gray=invalid; page {start // 18 + 1}', fill='black')
        for index, row in enumerate(rows[start:start + 18]):
            with Image.open(row['image']) as file:
                rgb = np.array(file.convert('RGB'))
            with np.load(row['geometry'], allow_pickle=False) as packed:
                depth, valid = packed['depth'], packed['valid'].astype(bool)
            stats = quality_stats(depth, valid, rgb)
            results.append(dict(id=row['id'], scene=row['scene'], scan=row['source_scan'], **stats))
            x, y = (index % 3) * 320, 30 + (index // 3) * 136
            draw.text((x + 3, y), row['id'], fill='black')
            draw.text((x + 3, y + 14), f"support={stats['valid_fraction']:.1%}; median Z={np.median(depth[valid]):.2f}m", fill='black')
            page.paste(Image.fromarray(rgb).resize((160, 100)), (x, y + 32))
            page.paste(depth_plot(depth, valid).resize((160, 100), Image.Resampling.NEAREST), (x + 160, y + 32))
        name = f'{split}-contact-{start // 18 + 1:02d}.jpg'
        temp = output / (name + '.incoming')
        page.save(temp, format='JPEG', quality=92)
        temp.replace(output / name)
        pages.append(name)
    return dict(rows=results, contact_sheets=pages,
                views=len(results), scenes=len({r['scene'] for r in rows}),
                original_scans=len({(r['scene'], r['source_scan']) for r in rows}),
                mean_measured_fraction=float(np.mean([r['valid_fraction'] for r in results])),
                min_measured_fraction=min(r['valid_fraction'] for r in results),
                flag_counts=dict(Counter(flag for r in results for flag in r['review_flags'])),
                flags_are_diagnostics_not_automatic_truth_labels=True)


def run(dataset, prepared_root, output, completion):
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    records = {split: read_manifest(prepared_root / split / (split + '.jsonl')) for split in ('train', 'val')}
    try:
        proof = json.loads(completion.read_text())
        if proof['stage'] != 'passed' or not proof['all_checks_passed']:
            raise ValueError('Official download completion evidence has not passed')
        atomic_json(dict(stage='running', started_at=started, source_read_only=True), output / 'status.json')
        inventory = source_inventory(dataset, records)
        atomic_json(inventory, output / 'source-inventory.json')
        summaries = {}
        for split, rows in records.items():
            print(json.dumps(dict(stage='original_source_recheck', split=split)), flush=True)
            source_check = audit(prepared_root / split, output / (split + '-source-recheck.json'))
            summaries[split] = inspect_pairs(rows, split, output)
            summaries[split]['original_source_checks_passed'] = source_check['stage'] == 'passed'
            atomic_json(summaries[split], output / (split + '-quality.json'))
        prepared_scenes = [r for r in inventory['scenes'] if r['prepared_scene']]
        missing_poses = [r['scene'] for r in prepared_scenes if not r['scanner_poses_available']]
        result = dict(stage='audit_complete', acceptance='HOLD_FOR_PHYSICAL_QUALITY_REVIEW',
                      generated_at=time.time(), started_at=started, source_read_only=True,
                      training_started=False, pseudo_labels_generated=False,
                      integrity_and_conversion='passed',
                      quality_summary={k: {key: value for key, value in v.items() if key != 'rows'}
                                       for k, v in summaries.items()},
                      independent_cross_scan_check='not_run', missing_scanner_pose_scenes=missing_poses,
                      visual_review='pending',
                      download_completion_evidence=dict(path=str(completion), sha256=digest(completion),
                          publisher_checksums=False, note='Local SHA receipts and ZIP CRC; not physical accuracy'),
                      limits=['No sensor-level guarantee from source/transform consistency',
                              'No use of teacher/student agreement as evidence that GT is true',
                              'Unprocessed scan inventory is not quality approval',
                              'No GT holes filled and no new labels generated',
                              'No fitted registration used to hide missing official scanner poses'])
        atomic_json(result, output / 'report.json')
        atomic_json(dict(stage='complete', report=str(output / 'report.json')), output / 'status.json')
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        atomic_json(dict(stage='failed', error=type(error).__name__ + ': ' + str(error)), output / 'status.json')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'prepared', 'output', 'download-completion'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.prepared, args.output, args.download_completion)
