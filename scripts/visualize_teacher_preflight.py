"""Checksum-bound pages of EVERY cached teacher preflight view, for human review.

Only display copies are resized. Depth colors use per-image relative values;
neither rendered pages nor this script constitute a visual or training pass.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from radio_moge.geometry_cache import canonical_json, publish_json, read_geometry_member, sha_file


def file_ref(path):
    return dict(path=str(path.resolve()), sha256=sha_file(path))


def entries_for_preflight(directory):
    report_path = directory / 'report.json'
    report_ref = file_ref(report_path)
    report = json.loads(report_path.read_text())
    packs = directory / 'packs'
    index = json.loads((packs / 'packs.json').read_text())
    policy = json.loads((packs / 'policy.json').read_text())
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if (report['stage'] != 'complete' or report['mode'] != 'preflight'
            or not 1 <= report['completed'] <= 256
            or report['verified_cache_members'] != report['completed']
            or index['count'] != report['completed']
            or index['policy_sha256'] != report['policy_sha256']
            or policy_hash != report['policy_sha256'] or policy['label_kind'] != 'teacher_pseudo'):
        raise ValueError('Complete bounded teacher preflight and matching pack policy required')
    result, keys = [], set()
    for item in index['packs']:
        receipt_path = packs / item['receipt']
        receipt_ref = file_ref(receipt_path)
        if receipt_ref['sha256'] != item['sha256']:
            raise ValueError('Preflight receipt changed')
        receipt = json.loads(receipt_path.read_text())
        pack = packs / receipt['tar']
        pack_ref = file_ref(pack)
        if (pack_ref['sha256'] != receipt['sha256'] or len(receipt['entries']) != item['count']
                or receipt['policy_sha256'] != policy_hash):
            raise ValueError('Preflight pack changed')
        for entry in receipt['entries']:
            if entry['key'] in keys:
                raise ValueError('Duplicate preflight geometry identity')
            keys.add(entry['key'])
            result.append((pack, entry, receipt_ref, pack_ref))
    if len(result) != report['completed']:
        raise ValueError('Incomplete preflight page scope')
    return result, report, report_ref


def view_row(pack, entry, ordinal, width=360):
    depth, valid, _, metadata = read_geometry_member(pack, entry)
    row = entry['image']
    data = Path(row['image']).read_bytes()
    if (hashlib.sha256(data).hexdigest() != row['sha256']
            or row['sha256'] != metadata['image_sha256'] or row['split'] != 'train'
            or metadata['label_kind'] != 'teacher_pseudo' or metadata['metric_calibrated']
            or metadata['camera_kind'] != 'teacher_estimated' or row.get('image_rotation_cw', 0)):
        raise ValueError('Only unchanged original training RGB and nonmetric teacher labels are displayed')
    with Image.open(io.BytesIO(data)) as source:
        if source.getexif().get(274, 1) != 1:
            raise ValueError('Unexpected teacher input orientation')
        rgb = source.convert('RGB')
    if depth.shape != (rgb.height, rgb.width):
        raise ValueError('Teacher geometry and original RGB grids differ')
    quantiles = np.quantile(depth[valid], [.01, .5, .99])
    lo, median, hi = quantiles
    relative = quantiles / median
    value = np.zeros_like(depth)
    value[valid] = np.clip((np.log(depth[valid]) - np.log(lo)) / max(1e-6, np.log(hi / lo)), 0, 1)
    colors = np.stack((value, 1 - np.abs(2 * value - 1), 1 - value), -1)
    colors[~valid] = 0
    panels = (rgb, Image.fromarray((colors * 255).astype(np.uint8)),
              Image.fromarray((valid * 255).astype(np.uint8)).convert('RGB'))
    header = [f"#{ordinal:03d} {row['group']} | {metadata['source_id']}",
              f"RGB | relative depth (log p01-p99; blue near/red far) | valid mask {valid.mean():.3f}",
              f"Depth / image median: p01={relative[0]:.3f}, p50=1, p99={relative[2]:.3f}; NOT metric GT"]
    if metadata.get('native_reason'):
        header.append('Unresolved native reason: ' + metadata['native_reason'])
    lines = [part for line in header for part in textwrap.wrap(line, width=142)]
    header_height = 8 + 15 * len(lines)
    tile_height = min(360, round(width * rgb.height / rgb.width))
    result = Image.new('RGB', (3 * width, header_height + tile_height + 8), 'white')
    draw = ImageDraw.Draw(result)
    for i, line in enumerate(lines):
        draw.text((4, 4 + 15 * i), line, fill='black')
    for i, panel in enumerate(panels):
        resized = ImageOps.contain(panel, (width, tile_height),
            Image.Resampling.LANCZOS if i == 0 else Image.Resampling.NEAREST)
        result.paste(resized, (i * width + (width - resized.width) // 2, header_height))
    identity = dict(ordinal=ordinal, group=row['group'], dataset=row['dataset'],
        source_id=metadata['source_id'], key=entry['key'], image=dict(path=row['image'], sha256=row['sha256']),
        valid_fraction=float(valid.mean()), grid=list(depth.shape),
        relative_depth_quantiles=relative.tolist(), native_reason=metadata.get('native_reason'))
    return result, identity


def render(directory, output):
    entries, report, report_ref = entries_for_preflight(directory)
    output.mkdir(parents=True, exist_ok=False)
    pages, records = [], []
    for start in range(0, len(entries), 4):
        tiles, identities = [], []
        for ordinal in range(start, min(start + 4, len(entries))):
            pack, entry, receipt_ref, pack_ref = entries[ordinal]
            tile, identity = view_row(pack, entry, ordinal)
            identity.update(receipt=receipt_ref, pack=pack_ref)
            tiles.append(tile)
            identities.append(identity)
        page = Image.new('RGB', (1080, sum(tile.height for tile in tiles)), '#cccccc')
        offset = 0
        for tile, identity in zip(tiles, identities):
            page.paste(tile, (0, offset))
            identity.update(page=f'{len(pages):03d}.jpg', page_y=offset, page_height=tile.height)
            offset += tile.height
        target = output / f'{len(pages):03d}.jpg'
        incoming = target.with_suffix('.jpg.incoming')
        with incoming.open('xb') as file:
            page.save(file, format='JPEG', quality=94)
            file.flush()
            os.fsync(file.fileno())
        os.rename(incoming, target)
        pages.append(file_ref(target))
        records.extend(identities)
    if sha_file(Path(report_ref['path'])) != report_ref['sha256']:
        raise ValueError('Preflight report changed while rendering')
    publish_json(output / 'index.json', dict(stage='visual_evidence_rendered_not_reviewed',
        preflight_report=report_ref, policy_sha256=report['policy_sha256'],
        numeric_preflight_passed=report['passed'], rejected_images=report['rejected'],
        rendered_images=len(records), all_cached_views_included=True,
        groups=sorted({r['group'] for r in records}), pages=pages, records=records,
        display='Original RGB; per-image log-depth percentiles; unchanged valid support. Display-only resizing.',
        generator=file_ref(Path(__file__)), training_admission=False, optimizer_updates=0))
    print(json.dumps(dict(rendered_images=len(records), pages=len(pages), output=str(output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preflight', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.preflight.resolve(), args.output.resolve())
