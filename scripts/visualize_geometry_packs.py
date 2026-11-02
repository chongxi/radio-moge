"""Write bounded cache QC contact sheets; no model calls or training acceptance."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from indoor_download import atomic_write
from radio_moge.geometry_cache import read_geometry_member


def sheets(directory, limit):
    output = directory.parent / 'visual-qc'
    if output.exists():
        raise FileExistsError('Keep prior visual evidence')
    output.mkdir()
    count = 0
    for path in sorted(directory.glob('pack-*.json')):
        pack = json.loads(path.read_text())
        for entry in pack['entries']:
            depth, valid, _, metadata = read_geometry_member(directory / pack['tar'], entry)
            with Image.open(entry['image']['image']) as file:
                rgb = file.convert('RGB')
            w, h = 384, round(384 * rgb.height / rgb.width)
            lo, hi = np.quantile(depth[valid], [.01, .99])
            value = np.clip((np.log1p(np.where(valid, depth, 0)) - np.log1p(lo)) /
                            max(1e-6, np.log1p(hi) - np.log1p(lo)), 0, 1)
            colored = np.stack((value, 1 - np.abs(value * 2 - 1), 1 - value), -1)
            colored[~valid] = 0
            image = Image.new('RGB', (w * 3, h + 45), 'white')
            draw = ImageDraw.Draw(image)
            draw.text((4, 3), f"{metadata['label_kind']} | {metadata['source_id']}", fill='black')
            draw.text((4, 18), f"RGB | depth p01={lo:.2f}, p99={hi:.2f} nominal m | mask={valid.mean():.3f}",
                      fill='black')
            panels = [rgb, Image.fromarray((colored * 255).astype(np.uint8)),
                      Image.fromarray((valid * 255).astype(np.uint8)).convert('RGB')]
            for i, panel in enumerate(panels):
                image.paste(panel.resize((w, h), Image.Resampling.NEAREST), (i * w, 45))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=90)
            atomic_write(output / f'{count:03d}.jpg', buffer.getvalue())
            count += 1
            if count >= limit:
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packs', type=Path)
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.limit <= 32:
        parser.error('Visual evidence is deliberately bounded')
    sheets(args.packs, args.limit)
