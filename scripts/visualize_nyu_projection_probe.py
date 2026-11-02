"""Inspect raw RGB, literal toolbox RGB, and published training RGB side by side."""
import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from indoor_download import atomic_write
from radio_moge.geometry_cache import publish_json
from radio_moge.nyu_projection import reproduce_undistorted_rgb


def visualize(root, probe, output):
    destination = root / output
    destination.mkdir(exist_ok=False)
    report = json.loads((root / probe / 'report.json').read_text())
    records = []
    with h5py.File(root / 'native-source/nyuv2/nyu_depth_v2_labeled.mat', 'r') as mat:
        for record in report['records']:
            name = record['id']
            index = int(name.split('_')[-1]) - 1
            truth = np.asarray(mat['images'][index]).transpose(2, 1, 0)
            path = root / probe / name / Path(record['pair_identity']['raw_names']['rgb']).name
            with Image.open(path) as image:
                raw = np.array(image)
            projected = reproduce_undistorted_rgb(raw)
            differences = np.abs(projected.astype(float) - truth)
            item = dict(id=name, raw_exact_fraction=float((raw == truth).mean()),
                raw_mae_u8=float(np.abs(raw.astype(float) - truth).mean()),
                reproduced_mae_u8=float(differences.mean()),
                reproduced_interior_mae_u8=float(differences[40:-40, 40:-40].mean()),
                calibration_acceptance=False)
            records.append(item)
            panels = [raw, projected, truth, (np.clip(differences * 3, 0, 255)).astype(np.uint8)]
            sheet = Image.new('RGB', (640 * 2, 500 * 2), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, (panel, title) in enumerate(zip(panels,
                    ('raw PPM', 'reproduced toolbox undistort', 'published MAT RGB', 'absolute diff x3'), strict=True)):
                x, y = i % 2 * 640, i // 2 * 500
                draw.text((x + 4, y + 3), name + ': ' + title, fill='black')
                sheet.paste(Image.fromarray(panel), (x, y + 20))
            buffer = io.BytesIO()
            sheet.save(buffer, format='JPEG', quality=95)
            atomic_write(destination / (name + '.jpg'), buffer.getvalue())
    publish_json(destination / 'report.json', dict(stage='diagnostic_complete', records=records,
                                                training_admission=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if any(Path(x).name != x for x in (args.probe, args.output)):
        parser.error('Use evidence directory basenames')
    visualize(args.root, args.probe, args.output)
