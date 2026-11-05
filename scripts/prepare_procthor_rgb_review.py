"""Check frozen RGB split identities and render a declared new-house review sample."""
import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw

from render_procthor_empty_pilot import publish, sha


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def read_manifest(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare(training, validation, previous, output):
    current, heldout, old = map(read_manifest, (training, validation, previous))
    by_image = {r['image']: r for r in current}
    if len(by_image) != len(current) or any(by_image.get(r['image']) != r for r in old):
        raise ValueError('Previous frozen training rows changed or duplicate image identities')
    if (any(r['split'] != 'train' for r in current)
            or any(r['split'] != 'val' for r in heldout)
            or any(r['projection'] != 'perspective' or r['agent_visible'] is not False
                   or r['geometry_label'] is not None or r['label_role'] != 'feature_only_rgb'
                   for r in current + heldout)):
        raise ValueError('Official split, projection or feature-only contract changed')
    for key in ('image', 'group', 'decoded_rgb_sha256', 'original_house_sha256'):
        if {r[key] for r in current} & {r[key] for r in heldout}:
            raise ValueError('Train/validation overlap: ' + key)
    if ({r['house']['sha256'] for r in current}
            & {r['house']['sha256'] for r in heldout}):
        raise ValueError('Empty structural house overlaps validation')
    if len({r['decoded_rgb_sha256'] for r in current}) != len(current):
        raise ValueError('Duplicate training pixels')
    old_groups = {r['group'] for r in old}
    new_rows = [r for r in current if r['group'] not in old_groups]
    groups = sorted({r['group'] for r in new_rows})
    if len(groups) < 24:
        raise ValueError('This review requires at least24 newly added houses')
    selected_groups = [groups[i * (len(groups) - 1) // 23] for i in range(24)]
    selected, roles = [], {}
    for group in selected_groups:
        rows = [r for r in new_rows if r['group'] == group]
        if len(rows) != 20:
            raise ValueError('Every selected house must contain20 closed views')
        for row in (rows[0], rows[10]):
            selected.append(row)
            roles[row['image']] = 'Fixed evenly spaced new house; retained view0 or10'
    for row in sorted(new_rows, key=lambda r: (r['spatial_rgb_std'], r['image']))[:4]:
        if row['image'] not in roles:
            selected.append(row)
            roles[row['image']] = 'One of four lowest spatial RGB standard deviations in new houses'
    output.mkdir(exist_ok=False)
    records, pages = [], []
    for start in range(0, len(selected), 12):
        rows = selected[start:start + 12]
        page = Image.new('RGB', (960, 280 * ((len(rows) + 2) // 3)), 'white')
        draw = ImageDraw.Draw(page)
        page_name = f'page-{len(pages):02d}.jpg'
        for index, row in enumerate(rows):
            path = Path(row['image'])
            if sha(path) != row['sha256']:
                raise ValueError('Selected RGB changed before visual review')
            with Image.open(path) as image:
                tile = image.convert('RGB').resize((320, 240), Image.Resampling.LANCZOS)
            x, y = (index % 3) * 320, (index // 3) * 280
            page.paste(tile, (x, y))
            label = (f"{Path(row['image']).stem} | FOV {row['vertical_fov_degrees']}\n"
                     f"{row['room']} | std {row['spatial_rgb_std']:.3f}")
            draw.text((x + 4, y + 243), label, fill='black')
            records.append(dict(image=ref(path), group=row['group'], room=row['room'],
                role=roles[row['image']], page=page_name, page_slot=index,
                decoded_rgb_sha256=row['decoded_rgb_sha256']))
        target = output / page_name
        incoming = target.with_suffix('.incoming')
        page.save(incoming, format='JPEG', quality=94)
        incoming.rename(target)
        pages.append(ref(target))
    report = dict(stage='split_identities_checked_visual_sample_rendered_not_reviewed',
        training=ref(training), validation=ref(validation), previous_training=ref(previous),
        images=len(current), houses=len({r['group'] for r in current}),
        new_images=len(new_rows), new_houses=len(groups), previous_rows_exact=True,
        train_validation_house_and_pixel_overlap=0, exact_training_pixel_duplicates=0,
        perspective_only=True, agent_visible=False, geometry_labels=False,
        sampled_images=len(selected), sampled_houses=len({r['group'] for r in selected}),
        fov_counts=dict(Counter(str(r['vertical_fov_degrees']) for r in current)),
        sampled_fov_counts=dict(Counter(str(r['vertical_fov_degrees']) for r in selected)),
        pages=pages, records=records, training_launched=False,
        sampling='24 evenly spaced new house IDs; views0,10; plus four lowest contrast new frames',
        every_image_visually_reviewed=False, generator=ref(Path(__file__)))
    publish(output / 'index.json', report)
    print(json.dumps(dict(report=ref(output / 'index.json'), images=len(current),
                         sampled_images=len(selected), pages=len(pages))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('training', 'validation', 'previous', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    prepare(*(getattr(args, key).resolve() for key in ('training', 'validation', 'previous', 'output')))
