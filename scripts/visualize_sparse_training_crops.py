"""Render exact sparse crop evidence from a completed production-loader probe.

Display copies alone are resized. The source grid, selected observations and
training masks remain unchanged; this renderer does not approve a source.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from audit_training_crop_support import observation_metrics
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, read_geometry_member
from radio_moge.geometry_sampling import load_candidate


def render(probe_path, output):
    torch.set_num_threads(1)
    probe_ref = file_ref(probe_path)
    probe = checked_json(probe_ref)
    selection = checked_json(probe['selection'])
    observations = checked_json(probe['observations'])
    sparse = set(probe['sparse_ordinals'])
    if (probe['actual_production_loader_used'] is not True
            or probe['original_support_unchanged'] is not True or not 0 < len(sparse) <= 16
            or sparse != {r['corpus_ordinal'] for r in observations if r['sparse_geometry_skipped']}):
        raise ValueError('A completed bounded sparse crop probe is required')
    cases = defaultdict(list)
    for row in observations:
        if row['corpus_ordinal'] in sparse:
            cases[row['corpus_ordinal']].append(row)
    output.mkdir(exist_ok=False)
    pages = []
    for chosen in selection['entries']:
        ordinal = chosen['ordinal']
        if ordinal not in sparse:
            continue
        entry, pack = chosen['entry'], chosen['pack']
        depth, valid, kd, _ = read_geometry_member(pack, entry)
        row = entry['image']
        with Image.open(row['image']) as source:
            original = source.convert('RGB')
        rotation = row.get('image_rotation_cw', 0)
        if rotation:
            original = original.transpose({90: Image.Transpose.ROTATE_270,
                180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[rotation])
        outline = ImageDraw.Draw(original)
        outputs = []
        colors = ('#ffff00', '#00ffff', '#ff50ff')
        for record, color in zip(cases[ordinal], colors, strict=True):
            rgb, target, view = load_candidate(pack, entry, tuple(probe['output_shape']),
                                               record['seed'], record['augmented'])
            metrics = observation_metrics(depth, valid, kd, target, view)
            if any(metrics[k] != record[k] for k in metrics):
                raise ValueError('Reproduced crop metrics differ from the bound probe')
            if record['cache_key'] != entry['key'] or record['image_sha256'] != row['sha256']:
                raise ValueError('Reproduced crop identity differs')
            corners = np.array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
            corners = corners @ np.linalg.inv(view['rgb_to_view_uv']).T
            uv = corners[:, :2] / corners[:, 2:] * original.size
            outline.line([tuple(p) for p in np.vstack((uv, uv[0]))], fill=color, width=3)
            outputs.append((rgb, target, record, color))
        page = Image.new('RGB', (1260, 1130), '#eeeeee')
        draw = ImageDraw.Draw(page)
        draw.text((8, 6), f"#{ordinal} {row['group']} | original valid {valid.sum()} / {valid.size}"
                  f" ({valid.mean():.4%}) | rotation {rotation}", fill='black')
        draw.text((8, 24), 'Source RGB + viewport outlines | native Z, valid only | native support', fill='black')
        lo, hi = np.quantile(depth[valid], [.01, .99])
        z = np.zeros_like(depth)
        z[valid] = np.clip((np.log(depth[valid]) - np.log(lo)) / max(np.log(hi / lo), 1e-8), 0, 1)
        color_depth = np.stack((z, 1 - np.abs(2 * z - 1), 1 - z), -1)
        color_depth[~valid] = 0
        panels = [original, Image.fromarray((255 * color_depth).astype(np.uint8)),
                  Image.fromarray((255 * valid).astype(np.uint8)).convert('RGB')]
        for col, panel in enumerate(panels):
            panel = ImageOps.contain(panel, (410, 430),
                Image.Resampling.LANCZOS if col == 0 else Image.Resampling.NEAREST)
            page.paste(panel, (420 * col + (420 - panel.width) // 2, 50))
        for col, (rgb, target, record, color) in enumerate(outputs):
            x = col * 420
            draw.rectangle((x + 5, 494, x + 413, 523), fill=color)
            draw.text((x + 8, 499), f"augment={record['augmented']} seed={record['seed']} "
                      f"flip={record['flipped']}", fill='black')
            crop = Image.fromarray(rgb).resize((410, 256), Image.Resampling.LANCZOS)
            points = ImageDraw.Draw(crop)
            for u, v in target['query_uv'][target['valid']]:
                px, py = u * 410, v * 256
                points.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill='#ff00ff')
            page.paste(crop, (x + 5, 530))
            draw.text((x + 8, 795), f"Exact selected points: {record['selected_observations']}; "
                      f"skip geometry: {record['sparse_geometry_skipped']}", fill='black')
            mask = Image.fromarray((target['valid'] * 255).astype(np.uint8)).convert('RGB')
            page.paste(mask.resize((410, 256), Image.Resampling.NEAREST), (x + 5, 820))
        draw.text((8, 1090), 'Magenta dots are display markers at exact query UVs. Bottom: unchanged loss masks (nearest display resize).', fill='black')
        draw.text((8, 1107), 'No hole filling, new depth values, model inference or source admission. Native depth colors use per-image p01-p99.', fill='black')
        target_path = output / f'{ordinal}.jpg'
        temporary = target_path.with_suffix('.jpg.incoming')
        page.save(temporary, format='JPEG', quality=95)
        temporary.rename(target_path)
        pages.append(dict(corpus_ordinal=ordinal, file=file_ref(target_path),
            image=dict(path=row['image'], sha256=row['sha256']), cache_key=entry['key'],
            views=cases[ordinal], exact_metrics_reproduced=True))
    if {p['corpus_ordinal'] for p in pages} != sparse:
        raise ValueError('Sparse visual selection is incomplete')
    checked_path(probe_ref)
    publish_json(output / 'index.json', dict(stage='sparse_crop_evidence_rendered_not_reviewed',
        probe=probe_ref, selection=probe['selection'], observations=probe['observations'],
        pages=pages, original_support_unchanged=True, display_resizing_only=True,
        selected_payloads_rechecked=True, model_inference=False, training_admission=False,
        generator=file_ref(Path(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    render(args.probe.resolve(), args.output.resolve())
