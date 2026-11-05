"""Display three saved pilot residuals using retained RGB, never training panoramas."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from probe_scannetpp_v2_source_pilot import bound_path, read
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json


def panel(image, point, width=580, height=360):
    view = ImageOps.contain(image, (width, height))
    draw = ImageDraw.Draw(view)
    x, y = point[0]*view.width/image.width, point[1]*view.height/image.height
    draw.line((x-12, y, x+12, y), fill='red', width=3)
    draw.line((x, y-12, x, y+12), fill='red', width=3)
    return view


def cropped(image, point, radius=180):
    x, y = point
    bounds = (max(0, int(x)-radius), max(0, int(y)-radius),
              min(image.width, int(x)+radius), min(image.height, int(y)+radius))
    return image.crop(bounds), (x-bounds[0], y-bounds[1]), list(bounds)


def run(root, analysis_path, output):
    analysis_ref = file_ref(analysis_path)
    analysis = read(analysis_ref, root)
    cases = analysis['center_over_5cm_cases']
    if {(x['corpus_ordinal'], x['slot']) for x in cases} != {(1318289, 68), (1412812, 35), (1412812, 71)}:
        raise ValueError('Exact three existing fixed-pilot residuals required')
    output.mkdir(exist_ok=False)
    records = []
    for selected in cases:
        ordinal, slot = selected['corpus_ordinal'], selected['slot']
        case_ref = file_ref(root/'scannetpp-native-v2-development-01'/f'{ordinal}.json')
        case = read(case_ref, root)
        row = dict(selected=selected, development=case_ref, DSLR=case['image'], scanners=[])
        y, x = selected['pixel_yx']
        h, w = case['grid']
        image_path = bound_path(case['image'], root)
        with Image.open(image_path) as source:
            rgb = source.convert('RGB')
        point = ((x+.5)*rgb.width/w, (y+.5)*rgb.height/h)
        zoom, zoom_point, bounds = cropped(rgb, point)
        panels = [(panel(rgb, point), 'Original perspective DSLR; marked frozen sample'),
                  (panel(zoom, zoom_point), 'DSLR detail; display only')]
        row['DSLR_point_xy'] = list(point)
        row['DSLR_crop_bounds'] = bounds
        del rgb, zoom
        for scan in ['0', '1']:
            station_ref = file_ref(root/'scannetpp-native-v2-source-pilot-01'/f'{ordinal}-scan{scan}.json')
            station = read(station_ref, root)
            measure = station['measurements'][slot]
            sy, sx = measure['scanner_pixel_yx']
            range_path = bound_path(station['inputs']['range'], root)
            photo = range_path.parent.parent/'images'/f'{scan}.jpg'
            photo_ref = file_ref(photo)
            with Image.open(range_path) as depth, Image.open(photo) as source:
                if source.size != depth.size:
                    raise ValueError('Published RGB/range grids differ; no implicit resampling')
                crop, center, bounds = cropped(source, (sx+.5, sy+.5))
            panels.append((panel(crop.convert('RGB'), center),
                f'Raw scanner {scan} panorama detail; DIAGNOSTIC ONLY; RGB never used for training'))
            row['scanners'].append(dict(station=station_ref, scan=scan, image=photo_ref,
                scanner_pixel_yx=[sy, sx], crop_bounds=bounds,
                range=station['inputs']['range'], numerical_grid_sizes_match=True,
                physical_RGB_range_coregistration_independently_verified=False,
                status=measure['status'], residual_m=measure.get('center_signed_residual_m')))
            del crop
        page = Image.new('RGB', (1200, 850), 'white')
        draw = ImageDraw.Draw(page)
        draw.text((8, 6), f"Ordinal {ordinal} / slot {slot} / DSLR (y,x)=({y},{x}); retained source RGB", fill='black')
        draw.text((8, 23), 'Shared acquisition; views and ranges are NOT an independent calibration or a motion/registration diagnosis.', fill='black')
        draw.text((8, 40), f"Selected scan {selected['scan']} center measured-minus-mesh: {selected['signed_residual_m']:.6f} m", fill='black')
        for i, (view, title) in enumerate(panels):
            px, py = (i % 2)*600+8, 65+(i//2)*390
            draw.text((px, py), title, fill='black')
            page.paste(view, (px, py+20))
        path = output/f'{ordinal}-{slot}.png'
        temporary = path.with_suffix('.png.incoming')
        page.save(temporary, format='PNG')
        temporary.rename(path)
        row['page'] = file_ref(path)
        records.append(row)
    publish_json(output/'report.json', dict(stage='source_RGB_context_rendered_not_reviewed',
        analysis=analysis_ref, records=records, image_pages=3, new_geometry_renders=0,
        source_images_edited=False, training_panorama_images_added=0,
        labels_changed=False, independent_physical_calibration=False,
        code=file_ref(Path(__file__)), training_admission=False, optimizer_updates=0))
    print(json.dumps(file_ref(output/'report.json')), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.root.resolve(), args.analysis.resolve(), args.output.resolve())
