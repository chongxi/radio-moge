"""Inspect every low-support COCO candidate without a numerical rejection rule."""
import argparse
import json
import os
from pathlib import Path

import cv2
from PIL import Image, ImageOps

from audit_completed_rgb_teachers import render
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.geometry_sampling import load_candidate


def run(audit_path, output):
    audit_ref = file_ref(audit_path)
    audit = checked_json(audit_ref)
    if (audit['stage'] != 'completed_teacher_source_accounted_not_accepted'
            or audit['dataset'] != 'coco2017' or not audit['producer_array_readback_complete']):
        raise ValueError('Complete COCO source audit required')
    selected = []
    with checked_path(audit['outcomes']).open() as source:
        for line in source:
            row = json.loads(line)
            if row['valid_fraction'] < .05:
                selected.append(dict(row, selection_reasons=['below_five_percent_diagnostic_only']))
    if len(selected) != audit['below_five_percent_support'] or not 0 < len(selected) <= 256:
        raise ValueError('Bounded complete low-support scope required')
    output.mkdir(exist_ok=False)
    render(selected, output / 'full-resolution', audit['policy_sha256'])
    views = checked_json(file_ref(output / 'full-resolution/report.json'))
    cv2.setNumThreads(1)
    samples = []
    for row in selected:
        receipt = checked_json(row['receipt'])
        entry, = [e for e in receipt['entries'] if e['key'] == row['key']]
        pack = Path(row['receipt']['path']).parent / receipt['tar']
        for shape in ((256, 256), (400, 640)):
            for seed, augment in ((0, False), (0, True), (43, True)):
                _, target, _ = load_candidate(pack, entry, shape, seed, augment)
                samples.append(dict(corpus_ordinal=row['corpus_ordinal'], shape=list(shape), seed=seed,
                    augment=augment, native_valid=target['native_valid_count'],
                    visible_native=target['visible_native_count'], loss_observations=int(target['valid'].sum())))
    pages = []
    for start in range(0, len(views['records']), 4):
        tiles = []
        for row in views['records'][start:start + 4]:
            with Image.open(checked_path(row['page'])) as image:
                tiles.append(ImageOps.contain(image.convert('RGB'), (1600, 600), Image.Resampling.LANCZOS))
        page = Image.new('RGB', (1600, sum(tile.height for tile in tiles)), '#dddddd')
        y = 0
        for tile in tiles:
            page.paste(tile, (0, y))
            y += tile.height
        path = output / f'contact-{start // 4:03d}.png'
        temporary = path.with_suffix('.png.incoming')
        page.save(temporary, format='PNG')
        os.rename(temporary, path)
        pages.append(file_ref(path))
    checked_path(audit_ref)
    publish_json(output / 'report.json', dict(stage='sparse_coco_evidence_rendered_not_reviewed',
        audit=audit_ref, selection=selected, full_resolution=file_ref(output / 'full-resolution/report.json'),
        pages=pages, image_count=len(selected), sampler_diagnostics=samples,
        diagnostics_not_final_training_config=True,
        views_below_loss_minimum=sum(r['loss_observations'] < 32 for r in samples),
        images_with_some_view_below_minimum=len({r['corpus_ordinal'] for r in samples if r['loss_observations'] < 32}),
        original_depth_and_masks_unchanged=True, automatic_exclusions=0,
        source_quality_accepted=False, training_admission=False, optimizer_updates=0,
        code=file_ref(Path(__file__)), sampler=file_ref(Path(__file__).parents[1] / 'src/radio_moge/geometry_sampling.py')))
    print(json.dumps(dict(images=len(selected), diagnostic_views=len(samples), pages=len(pages))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.audit.resolve(), args.output.resolve())
