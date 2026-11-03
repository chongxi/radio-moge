"""Render selected native outcome extremes; images do not confer acceptance."""
import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from indoor_download import atomic_write
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, read_geometry_member, sha_file


def depth_panel(depth, valid, lo, hi):
    value = np.zeros_like(depth)
    value[valid] = np.clip((np.log(depth[valid]) - np.log(lo)) / max(1e-6, np.log(hi / lo)), 0, 1)
    colors = np.stack((value, 1 - np.abs(2 * value - 1), 1 - value), -1)
    colors[~valid] = 0
    return Image.fromarray((colors * 255).astype(np.uint8))


def render(audit_path, output):
    audit_ref = file_ref(audit_path)
    audit = checked_json(audit_ref)
    selected = checked_json(audit['review_candidates'])
    if (audit['dataset'] != 'structured3d' or audit['stage'] != 'native_outcomes_accounted_not_accepted'
            or audit['producer_array_readback_complete'] is not True
            or not 1 <= len(selected) == audit['review_candidate_count'] <= 64):
        raise ValueError('Complete source outcome audit and bounded selection required')
    output.mkdir(exist_ok=False)
    pages, records, tiles = [], [], []
    for i, chosen in enumerate(selected):
        receipt = checked_json(chosen['receipt'])
        entries = [e for e in receipt['entries'] if e['key'] == chosen['entry_key']]
        if len(entries) != 1 or receipt['policy_sha256'] != chosen['policy_sha256']:
            raise ValueError('Selected candidate receipt differs')
        entry = entries[0]
        if entry['image'] != chosen['image']:
            raise ValueError('Selected RGB identity differs')
        pack = Path(chosen['receipt']['path']).parent / receipt['tar']
        if pack.stat().st_size != receipt['bytes']:
            raise ValueError('Selected pack size changed')
        depth, valid, _, meta = read_geometry_member(pack, entry)
        row = entry['image']
        checked_path(dict(path=row['image'], sha256=row['sha256']))
        with Image.open(row['image']) as rgb_file:
            if rgb_file.getexif().get(274, 1) != 1 or row.get('image_rotation_cw', 0):
                raise ValueError('Unexpected RGB orientation')
            rgb = rgb_file.convert('RGB')
        if (depth.shape != (rgb.height, rgb.width) or meta['source_needs_review']
                or meta['policy_sha256'] != chosen['policy_sha256']):
            raise ValueError('Selected geometry grid or policy differs')
        q = np.quantile(depth[valid], [.01, .5, .99])
        if (not math.isclose(valid.mean(), chosen['metrics']['valid_fraction'], abs_tol=1e-12)
                or not math.isclose(q[1], chosen['metrics']['median_depth_m'], abs_tol=1e-7)):
            raise ValueError('Payload differs from audited support/depth statistics')
        width, height = 512, round(512 * rgb.height / rgb.width)
        tile = Image.new('RGB', (width * 4, height + 75), 'white')
        draw = ImageDraw.Draw(tile)
        lines = [f"#{chosen['corpus_ordinal']} {row['group']} room={row['room']} {row['configuration']} | {', '.join(chosen['selection_reasons'])}",
                 'RGB | fixed log camera-Z 0.025--64 m (blue near/red far) | per-image log p01--p99 | unchanged mask',
                 f"p01/median/p99(m)={q[0]:.5f}/{q[1]:.5f}/{q[2]:.5f}; support={valid.mean():.5f}; "
                 f"layout verified/conflicting pixels={chosen['metrics']['layout_verified_pixels']}/{chosen['metrics']['layout_conflicting_pixels']}"]
        for line, text in enumerate(lines):
            draw.text((4, 5 + 20 * line), text, fill='black')
        panels = [rgb, depth_panel(depth, valid, .025, 64), depth_panel(depth, valid, q[0], q[2]),
                  Image.fromarray((valid * 255).astype(np.uint8)).convert('RGB')]
        for j, panel in enumerate(panels):
            tile.paste(panel.resize((width, height), Image.Resampling.LANCZOS if j == 0
                                   else Image.Resampling.NEAREST), (j * width, 75))
        tiles.append(tile)
        records.append(dict(chosen, member_sha256=entry['sha256'], grid=list(depth.shape),
            depth_quantiles_m=q.tolist(), selected_payload_rechecked=True, rgb_rechecked=True,
            page=f'{i // 2:03d}.jpg', page_row=i % 2))
        if len(tiles) == 2 or i == len(selected) - 1:
            page = Image.new('RGB', (4 * width, sum(t.height for t in tiles)), 'white')
            y = 0
            for tile in tiles:
                page.paste(tile, (0, y))
                y += tile.height
            data = io.BytesIO()
            page.save(data, format='JPEG', quality=95)
            path = output / f'{len(pages):03d}.jpg'
            atomic_write(path, data.getvalue())
            pages.append(file_ref(path))
            tiles.clear()
    if sha_file(audit_path) != audit_ref['sha256']:
        raise ValueError('Audit changed during rendering')
    publish_json(output / 'index.json', dict(stage='native_visual_evidence_rendered_not_reviewed',
        audit=audit_ref, selection=audit['review_candidates'], pages=pages, records=records,
        rendered_images=len(records), fixed_depth_limits_m=[.025, 64], display_only_resizing=True,
        selected_payloads_rechecked=True, pack_payload_hashes_rechecked=False,
        source_quality_accepted=False, training_admission=False, optimizer_updates=0,
        generator=file_ref(Path(__file__))))
    print(json.dumps(dict(rendered_images=len(records), pages=len(pages))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    render(args.audit.resolve(), args.output.resolve())
