"""Reproduce the bounded ARKit outcome selection from retained native bytes.

RGB is projected onto the unchanged native depth grid using the two cameras.
All four quarter turns are checked against sampling before rotation. This is
visual evidence, not a physical calibration or source-quality acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import map_coordinates

from audit_arkit_outcomes import candidate_details
from cache_arkit_geometry import convert, load_image
from radio_moge.arkit_depth import registration_uv, rotate_camera
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json, read_geometry_member, sha_file
from visualize_arkit_faro_probe import RetainedBundle


def sample_rgb(rgb, low_camera, rgb_camera):
    uv, kd, kr = registration_uv(low_camera, rgb_camera)
    if rgb.dtype != np.uint8 or rgb.shape != (int(rgb_camera[1]), int(rgb_camera[0]), 3):
        raise ValueError('Original RGB grid differs from camera')
    xy = uv * [rgb.shape[1], rgb.shape[0]] - .5
    sampled = np.stack([map_coordinates(rgb[..., c].astype(float) / 255,
        [xy[..., 1], xy[..., 0]], order=1, mode='nearest') for c in range(3)], -1)
    inside = ((uv[..., 0] >= 0) & (uv[..., 0] <= 1)
              & (uv[..., 1] >= 0) & (uv[..., 1] <= 1))
    return sampled, inside, kd, kr


def aligned_display(rgb, low_rgb, low_camera, rgb_camera, turn, quality):
    if turn not in (0, 90, 180, 270) or low_rgb.shape != (int(low_camera[1]), int(low_camera[0]), 3):
        raise ValueError('Native RGB grid or quarter turn differs')
    original, inside, _, _ = sample_rgb(rgb, low_camera, rgb_camera)
    upright = np.rot90(rgb, -turn // 90).copy()
    sampled, oriented_inside, kd, kr = sample_rgb(upright,
        rotate_camera(low_camera, turn), rotate_camera(rgb_camera, turn))
    expected = np.rot90(original, -turn // 90)
    if (not np.allclose(sampled, expected, rtol=0, atol=1e-12)
            or not np.array_equal(oriented_inside, np.rot90(inside, -turn // 90))):
        raise ValueError('Oriented camera does not preserve native RGB correspondence')
    error = np.abs(original - low_rgb.astype(float) / 255).mean(-1)
    if (not np.isclose(error.mean(), quality['paired_rgb_mae'], rtol=0, atol=1e-12)
            or not np.isclose(np.quantile(error, .9), quality['paired_rgb_error_p90'], rtol=0, atol=1e-12)):
        raise ValueError('Display RGB differences do not reproduce producer statistics')
    return upright, np.rot90(low_rgb, -turn // 90).copy(), sampled, np.rot90(error, -turn // 90), kd, kr


def require_array_parity(cached, reproduced):
    depth, valid, kd, meta = cached
    actual_depth, actual_valid, actual_kd, kr, proofs, quality = reproduced
    if (any(not np.array_equal(a, b) for a, b in
            ((depth, actual_depth), (valid, actual_valid), (kd, actual_kd)))
            or not np.array_equal(kr, np.asarray(meta['rgb_intrinsics_normalized']))
            or canonical_json(quality) != canonical_json(meta['source_quality'])
            or proofs != meta['source_proofs']
            or not np.array_equal(kr @ np.linalg.inv(kd), np.asarray(meta['depth_to_rgb_uv_transform']))):
        raise ValueError('Retained native sources do not reproduce cached arrays, cameras or provenance')


def depth_colors(depth, valid):
    value = np.zeros(depth.shape)
    value[valid] = np.clip(np.log(depth[valid] / .125) / np.log(64), 0, 1)
    colors = (255 * np.stack((value, 1 - np.abs(2 * value - 1), 1 - value), -1)).astype(np.uint8)
    colors[~valid] = 0
    return colors


def render(config, audit_name, attempt):
    root, code_root = Path(config['root']), Path(__file__).parents[1]
    audit_ref = file_ref(root / audit_name / 'report.json')
    audit = checked_json(audit_ref)
    policy, selected = checked_json(audit['policy']), checked_json(audit['review_candidates'])
    generation = checked_json(audit['generation_report'])
    conversion_file = Path(__file__).with_name('cache_arkit_geometry.py')
    if (audit['stage'] != 'native_outcomes_accounted_not_accepted' or audit['dataset'] != 'arkitscenes'
            or audit['manifest_sha256'] != config['manifest_sha256']
            or audit['producer_array_readback_complete'] is not True
            or generation['stage'] != 'complete' or generation['verified_cache_members'] != generation['completed']
            or hashlib.sha256(canonical_json(policy)).hexdigest() != audit['policy_sha256']
            or sha_file(conversion_file) != policy['script_sha256']
            or not 1 <= len(selected) <= 64
            or len({s['corpus_ordinal'] for s in selected}) != len(selected)):
        raise ValueError('Complete matching audit, bounded unique selection and original converter required')
    # Source math must match the producer; the current payload reader is bound
    # separately and verifies each selected member's original SHA256.
    for name in ('src/radio_moge/arkit_depth.py', 'src/radio_moge/geometry_sources.py'):
        if sha_file(code_root / name) != policy['code_sha256'][name]:
            raise ValueError('Native conversion implementation changed')
    index = checked_path(audit['index'])
    output = root / attempt
    output.mkdir(exist_ok=False)
    records = []
    with readonly_db(index) as db:
        for number, chosen in enumerate(selected):
            receipt = checked_json(chosen['receipt'])
            entries = [e for e in receipt['entries'] if e['key'] == chosen['entry_key']]
            if len(entries) != 1 or receipt['policy_sha256'] != audit['policy_sha256']:
                raise ValueError('Selected receipt identity differs')
            entry = entries[0]
            row, meta = entry['image'], entry['geometry_metadata']
            original = db.execute('SELECT row_json,outcome,entry_key,evidence_json FROM images WHERE ordinal=?',
                                  (chosen['corpus_ordinal'],)).fetchone()
            compare_row = dict(row)
            if (compare_row.pop('corpus_ordinal', chosen['corpus_ordinal']) != chosen['corpus_ordinal']
                    or original is None or compare_row != json.loads(original[0])
                    or original[1] != 'native_candidate' or original[2] != entry['key']
                    or json.loads(original[3]) != chosen['receipt']
                    or candidate_details(entry, audit['policy_sha256']) != chosen['metrics']
                    or row['image_rotation_cw'] != chosen['rotation']):
                raise ValueError('Selection differs from the frozen native outcome')
            pack = Path(chosen['receipt']['path']).parent / receipt['tar']
            if pack.parent != Path(chosen['receipt']['path']).parent or pack.stat().st_size != receipt['bytes']:
                raise ValueError('Pack path or size changed')
            cached = read_geometry_member(pack, entry)
            bundle = RetainedBundle(root / 'native-source/arkitscenes', meta['source_proofs'])
            reproduced = convert(row, bundle)
            require_array_parity(cached, reproduced)
            depth, valid, kd, _ = cached
            base, stem = row['source_url'].rsplit('/', 1)[0], PurePosixPath(row['source_member']).stem
            paths = {kind: bundle.get(base, kind, stem)[0] for kind in
                ('lowres_wide', 'lowres_wide_intrinsics', 'vga_wide_intrinsics')}
            low_camera, rgb_camera = [np.loadtxt(paths[k]).tolist() for k in
                                     ('lowres_wide_intrinsics', 'vga_wide_intrinsics')]
            upright, low_rgb, sampled, error, display_kd, kr = aligned_display(load_image(row['image']),
                load_image(paths['lowres_wide']), low_camera, rgb_camera,
                row['image_rotation_cw'], meta['source_quality'])
            if not np.array_equal(kd, display_kd) or not np.array_equal(kr, reproduced[3]):
                raise ValueError('Display camera differs from cache')
            colors = depth_colors(depth, valid)
            overlay = low_rgb.copy()
            overlay[valid] = (.55 * low_rgb[valid] + .45 * colors[valid]).astype(np.uint8)
            error_color = (np.clip(error / .08, 0, 1)[..., None] * np.array([255, 110, 0])).astype(np.uint8)
            panels = [(upright, 'Original VGA RGB, upright'), (low_rgb, 'Native lowres RGB, upright'),
                ((sampled * 255).clip(0, 255).astype(np.uint8), 'VGA RGB via K onto native depth grid'),
                (colors, 'Native Z: fixed log 0.125-8m; black invalid'),
                (overlay, 'Unchanged valid depth over native RGB'),
                (error_color, 'RGB MAE: black=0, orange>=0.08')]
            page = Image.new('RGB', (1200, 1200), 'white')
            draw = ImageDraw.Draw(page)
            lines = [f"#{chosen['corpus_ordinal']} {row['group']} | {stem} | CW {row['image_rotation_cw']}",
                f"{chosen['selection_reason']}; support={valid.mean():.3%}; median Z={np.median(depth[valid]):.3f}m; "
                f"RGB MAE={error.mean():.5f}, p90={np.quantile(error, .9):.5f}",
                'Selected member and retained source bytes verified; depth, mask, cameras reproduced exactly.',
                'RGB correspondence is not independent physical depth accuracy. No calibration fit or hole filling.']
            for i, line in enumerate(lines):
                draw.text((8, 8 + 20 * i), line, fill='black')
            for i, (array, label) in enumerate(panels):
                x, y = i % 3 * 400, 100 + i // 3 * 550
                draw.text((x + 5, y), label, fill='black')
                thumb = ImageOps.contain(Image.fromarray(array), (392, 520),
                    Image.Resampling.LANCZOS if i < 3 else Image.Resampling.NEAREST)
                page.paste(thumb, (x + (400 - thumb.width) // 2, y + 22))
            target = output / f'{number:02d}-{chosen["corpus_ordinal"]}.png'
            incoming = target.with_suffix('.png.incoming')
            with incoming.open('xb') as file:
                page.save(file, format='PNG')
            incoming.rename(target)
            record = dict(**chosen, image=file_ref(row['image']), group=row['group'], video=row['video_id'],
                source_proofs=meta['source_proofs'], member_sha256=entry['sha256'], page=file_ref(target),
                native_grid=list(depth.shape), source_array_and_camera_parity=True,
                independent_physical_accuracy_checked=False)
            records.append(record)
            print(json.dumps(dict(rendered=len(records), corpus_ordinal=chosen['corpus_ordinal'])), flush=True)
    checked_path(audit_ref)
    checked_path(audit['review_candidates'])
    publish_json(output / 'report.json', dict(stage='native_visual_evidence_rendered_not_reviewed',
        audit=audit_ref, selection=audit['review_candidates'], rendered_images=len(records), records=records,
        rotations=sorted({r['rotation'] for r in records}), renderer=file_ref(Path(__file__)),
        source_converter=file_ref(conversion_file), retained_reader=file_ref(Path(__file__).with_name(
            'visualize_arkit_faro_probe.py')), payload_reader=file_ref(code_root / 'src/radio_moge/geometry_cache.py'),
        source_array_and_camera_parity=True, selected_member_payloads_rechecked=True,
        pack_payload_hashes_rechecked=False, network_used=False, gpu_used=False,
        depth_or_mask_resized=False, display_only_resizing=True, source_quality_accepted=False,
        source_flags_cleared=False, training_admission=False, optimizer_updates=0))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--audit', default='arkit-outcomes-01')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.audit, args.attempt)):
        parser.error('Use new attempt basenames')
    render(json.loads(args.config.read_text()), args.audit, args.attempt)
