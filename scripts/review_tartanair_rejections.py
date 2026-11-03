"""Reproduce a bounded snapshot of native rejections from retained sources only.

This diagnostic cannot change producer thresholds, download sources, infer a
teacher label, or admit training data. Display copies leave all arrays intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json, sha_file
from radio_moge.tartanair_depth import convert_tartanair


def retained_source(source_root, row, kind, proofs):
    suffix = dict(depth_left='_left_depth.npy', depth_right='_right_depth.npy',
                  image_right='_right.png')
    source = row['source_url']
    if (not source.startswith('https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/')
            or not source.endswith('/image_left.zip') or kind not in suffix
            or '/image_left/' not in row['source_member']
            or not row['source_member'].endswith('_left.png')):
        raise ValueError('Expected original publisher left-eye source identity')
    url = source.rsplit('/', 1)[0] + '/' + kind + '.zip'
    member = row['source_member'].replace('/image_left/', '/' + kind + '/')
    member = member.removesuffix('_left.png') + suffix[kind]
    matched = [v for v in proofs if v['source'] == url and v['member'] == member]
    if len(matched) != 1 or Path(member).is_absolute() or '..' in Path(member).parts:
        raise ValueError('Exact unique corresponding-eye proof required')
    proof = matched[0]
    path = source_root / hashlib.sha256(url.encode()).hexdigest()[:16] / member
    receipt = path.with_name(path.name + '.receipt.json')
    if (json.loads(receipt.read_text()) != proof or sha_file(path) != proof['sha256']
            or path.stat().st_size != proof['bytes']):
        raise ValueError('Retained source or receipt changed')
    return path, file_ref(receipt)


def read_rgb(path):
    with Image.open(path) as source:
        if source.mode != 'RGB' or source.getexif().get(274, 1) != 1:
            raise ValueError('Unexpected RGB mode or orientation')
        return np.array(source)


def reproduce(record, source_root):
    row = record['image']
    if (row['dataset'] != 'tartanair' or row['split'] != 'train'
            or (row['width'], row['height']) != (640, 480)
            or sha_file(row['image']) != row['sha256']
            or record['reason'] != 'insufficient_stereo_verified_support'):
        raise ValueError('Unchanged frozen training RGB and native rejection required')
    paths, refs = {}, []
    for kind in ('depth_left', 'depth_right', 'image_right'):
        paths[kind], ref = retained_source(source_root, row, kind, record['source_proofs'])
        refs.append(ref)
    left = np.load(paths['depth_left'], allow_pickle=False)
    right = np.load(paths['depth_right'], allow_pickle=False)
    left_rgb, right_rgb = read_rgb(row['image']), read_rgb(paths['image_right'])
    _, valid, _, quality = convert_tartanair(left, right, left_rgb, right_rgb)
    if canonical_json(quality) != canonical_json(record['stats']) or valid.mean() >= .05:
        raise ValueError('Recomputed support differs from the published rejection')
    return left, left_rgb, right_rgb, valid, quality, refs


def draw_page(row, left, left_rgb, right_rgb, valid, quality):
    finite = np.isfinite(left) & (left > 0)
    task = finite & (left >= .1) & (left <= 200.)
    colors = np.zeros((*left.shape, 3), np.uint8)
    value = np.zeros(left.shape)
    value[task] = np.log(left[task] / .1) / np.log(2000.)
    colors[task] = (255 * np.stack((value, 1 - np.abs(2 * value - 1), 1 - value), -1)[task]).astype(np.uint8)
    colors[finite & (left > 200)] = 160
    overlay = (left_rgb.astype(float) * .25).astype(np.uint8)
    overlay[valid] = left_rgb[valid]
    panels = [(left_rgb, 'Original left RGB'), (right_rgb, 'Original right RGB'),
              (colors, 'Native Z: log 0.1-200m; gray >200m; black other'),
              (task.astype(np.uint8) * 255, 'Task range only (white): not publisher validity'),
              (valid.astype(np.uint8) * 255, 'Unchanged final stereo support (white)'),
              (overlay, 'Left RGB: only retained pixels at full brightness')]
    canvas = Image.new('RGB', (1440, 848), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), row['source_member'], fill='black')
    draw.text((6, 24), f"Task {quality['task_range_fraction']:.4%}; stable {quality['stable_left_fraction']:.4%}; "
              f"stereo {quality['stereo_geometric_fraction']:.4%}; final {quality['valid_fraction']:.4%}", fill='black')
    draw.text((6, 42), 'Native source/rejection reproduced. Not label acceptance or independent physical GT.', fill='black')
    for i, (data, title) in enumerate(panels):
        x, y = (i % 3) * 480, 70 + (i // 3) * 390
        draw.text((x + 4, y), title, fill='black')
        resample = Image.Resampling.LANCZOS if i in (0, 1, 5) else Image.Resampling.NEAREST
        canvas.paste(Image.fromarray(data).convert('RGB').resize((480, 360), resample), (x, y + 22))
    return canvas


def run(config, cache_attempt, attempt):
    root = Path(config['root'])
    cache = root / cache_attempt
    source_ref, policy_ref = file_ref(root / 'inventory.json'), file_ref(cache / 'packs/policy.json')
    source, policy = checked_json(source_ref), checked_json(policy_ref)
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    checked_path(inventory_ref)
    repo = Path(__file__).resolve().parents[1]
    if (source['manifest_sha256'] != config['manifest_sha256']
            or policy['manifest_sha256'] != config['manifest_sha256']
            or policy['dataset'] != 'tartanair' or policy['task_range_m'] != [.1, 200.]
            or policy['minimum_image_valid_fraction'] != .05):
        raise ValueError('Frozen scope and unchanged producer selection policy required')
    for name in ('src/radio_moge/tartanair_depth.py', 'src/radio_moge/geometry_sources.py'):
        if sha_file(repo / name) != policy['code_sha256'][name]:
            raise ValueError('Reproduction converter differs from producer policy')
    selection = [file_ref(p) for p in sorted((cache / 'rejected').glob('*.json'))]
    if not 1 <= len(selection) <= 64:
        raise ValueError('Bounded rejection snapshot of 1-64 records required')
    output = root / attempt
    output.mkdir(exist_ok=False)
    publish_json(output / 'selection.json', selection)
    records, seen = [], set()
    with readonly_db(inventory_ref['path']) as db:
        for i, ref in enumerate(selection):
            record = checked_json(ref)
            row = record['image']
            match = db.execute('SELECT ordinal,row_json,image FROM images WHERE image=?', (row['image'],)).fetchone()
            if match is None or match[0] in seen or row != dict(json.loads(match[1]), image=match[2]):
                raise ValueError('Rejected row differs from unique frozen inventory input')
            seen.add(match[0])
            left, rgb, right_rgb, valid, quality, refs = reproduce(record, root / 'native-source/tartanair')
            page = output / f'{i:03d}.png'
            temporary = page.with_suffix('.png.incoming')
            draw_page(row, left, rgb, right_rgb, valid, quality).save(temporary, format='PNG')
            os.rename(temporary, page)
            records.append(dict(corpus_ordinal=match[0], image=row, rejection=ref, source_receipts=refs,
                page=file_ref(page), quality=quality, reproduced_statistics_exact=True))
    for ref in (source_ref, policy_ref, *selection):
        checked_path(ref)
    result = dict(stage='rejection_evidence_rendered_not_reviewed', dataset='tartanair', records=records,
        reviewed_subset_size=len(records), full_source_scope=config['expected_counts']['tartanair'],
        inventory=inventory_ref, source_report=source_ref, producer_policy=policy_ref,
        selection=file_ref(output / 'selection.json'), audit_code=file_ref(Path(__file__)),
        selection_is_snapshot_of_running_producer=True, full_source_accounting_complete=False,
        independent_physical_depth_accuracy_checked=False, thresholds_changed=False,
        native_holes_filled=False, network_requests=0, teacher_inference_performed=False,
        automatic_exclusions=0, source_quality_accepted=False, training_admission=False, optimizer_updates=0)
    publish_json(output / 'report.json', result)
    print(json.dumps(dict(stage=result['stage'], images=len(records), output=str(output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache-attempt', default='tartanair-native-cache-01')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.cache_attempt, args.attempt)):
        parser.error('Use new attempt basenames')
    run(json.loads(args.config.read_text()), args.cache_attempt, args.attempt)
