"""Reproduce selected native candidates from retained stereo sources, CPU only.

The fixed-range display is evidence for review, not a source-quality pass.
No downloads, source edits, threshold changes, exclusions, or model inference.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import zlib

import numpy as np
from PIL import Image, ImageDraw

from audit_tartanair_outcomes import candidate_details
from indoor_download import atomic_write
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json, read_geometry_member, sha_file
from radio_moge.tartanair_depth import convert_tartanair
from review_tartanair_rejections import read_rgb, retained_source


def reproduce(entry, pack, source_root, policy_hash, metrics):
    if candidate_details(entry, policy_hash) != metrics:
        raise ValueError('Candidate receipt differs from audited metrics')
    row, meta = entry['image'], entry['geometry_metadata']
    checked_path(dict(path=row['image'], sha256=row['sha256']))
    paths, refs = {}, []
    for kind, proof in zip(('depth_left', 'depth_right', 'image_right'), meta['source_proofs']):
        path, ref = retained_source(source_root, row, kind, meta['source_proofs'])
        if zlib.crc32(path.read_bytes()) & 0xffffffff != proof['crc32']:
            raise ValueError('Retained source CRC differs')
        paths[kind] = path
        refs.append(ref)
    left, right = [np.load(paths[k], allow_pickle=False) for k in ('depth_left', 'depth_right')]
    rgb, right_rgb = read_rgb(row['image']), read_rgb(paths['image_right'])
    depth, valid, camera, quality = convert_tartanair(left, right, rgb, right_rgb)
    cached_depth, cached_valid, cached_camera, _ = read_geometry_member(pack, entry)
    if (canonical_json(quality) != canonical_json(meta['source_quality'])
            or not np.array_equal(cached_depth, depth)
            or not np.array_equal(cached_valid, valid)
            or not np.array_equal(cached_camera, camera)
            or not np.array_equal(np.quantile(depth[valid], [0, .01, .5, .99, 1]),
                                  entry['depth_quantiles_m'])):
        raise ValueError('Reproduced native depth, mask, camera or statistics differ from cached member')
    return left, rgb, right_rgb, valid, quality, refs


def tile(chosen, row, native, rgb, right_rgb, valid):
    task = np.isfinite(native) & (native >= .1) & (native <= 200.)
    value = np.zeros(native.shape, dtype=float)
    value[task] = np.log(native[task] / .1) / np.log(2000.)
    colors = np.zeros((*native.shape, 3), np.uint8)
    colors[task] = (255 * np.stack((value, 1 - np.abs(2 * value - 1), 1 - value), -1)[task]).astype(np.uint8)
    colors[np.isfinite(native) & (native > 200.)] = 160
    m = chosen['metrics']
    result = Image.new('RGB', (1280, 310), 'white')
    draw = ImageDraw.Draw(result)
    draw.text((4, 3), f"#{chosen['corpus_ordinal']} {row['source_member']} | {','.join(chosen['selection_reasons'])}", fill='black')
    draw.text((4, 19), f"support {m['support']:.5f}; median/p99 Z {m['median_depth']:.4f}/{m['p99_depth']:.4f} m; "
              f"RGB MAE on geometric support {m['rgb_mae']:.4f}", fill='black')
    panels = [(rgb, 'Original left RGB'), (right_rgb, 'Original right RGB'),
              (colors, 'Native Z: log .1-200m; gray >200m'),
              (valid.astype(np.uint8) * 255, 'Unchanged stereo-verified support')]
    for i, (data, label) in enumerate(panels):
        draw.text((i * 320 + 3, 39), label, fill='black')
        result.paste(Image.fromarray(data).convert('RGB').resize((320, 240),
            Image.Resampling.LANCZOS if i < 2 else Image.Resampling.NEAREST), (i * 320, 60))
    return result


def render(audit_path, source_root, output):
    audit_ref = file_ref(audit_path)
    audit = checked_json(audit_ref)
    selected = checked_json(audit['review_candidates'])
    policy = checked_json(audit['policy'])
    if (audit['stage'] != 'native_outcomes_accounted_not_accepted' or audit['dataset'] != 'tartanair'
            or audit['producer_array_readback_complete'] is not True
            or not 1 <= len(selected) == audit['review_candidates_count'] <= 128
            or len({s['corpus_ordinal'] for s in selected}) != len(selected)
            or hashlib.sha256(canonical_json(policy)).hexdigest() != audit['policy_sha256']):
        raise ValueError('Complete native audit and distinct bounded selection required')
    repo = Path(__file__).resolve().parents[1]
    for name in ('src/radio_moge/tartanair_depth.py', 'src/radio_moge/geometry_sources.py'):
        if sha_file(repo / name) != policy['code_sha256'][name]:
            raise ValueError('Reproduction converter differs from producer policy')
    output.mkdir(exist_ok=False)
    pages, records, tiles, receipts = [], [], [], {}
    with tempfile.TemporaryDirectory(prefix='tartanair-selected-') as temporary:
        local = Path(temporary) / 'outcomes.sqlite'
        shutil.copyfile(audit['index']['path'], local)
        if sha_file(local) != audit['index']['sha256']:
            raise ValueError('Complete native outcome index changed')
        with readonly_db(local) as db:
            for i, chosen in enumerate(selected):
                ref = chosen['receipt']
                if ref['path'] not in receipts:
                    receipts[ref['path']] = checked_json(ref)
                receipt = receipts[ref['path']]
                entries = [e for e in receipt['entries'] if e['key'] == chosen['entry_key']]
                if len(entries) != 1 or receipt['policy_sha256'] != audit['policy_sha256']:
                    raise ValueError('Selected member or policy differs')
                entry = entries[0]
                found = db.execute('SELECT row_json,outcome,entry_key,evidence_json,details_json '
                                   'FROM images WHERE ordinal=?', (chosen['corpus_ordinal'],)).fetchone()
                if (found is None or json.loads(found[0]) != entry['image'] or found[1] != 'native_candidate'
                        or found[2] != entry['key'] or json.loads(found[3]) != ref
                        or json.loads(found[4]) != chosen['metrics']
                        or chosen['environment'] != entry['image']['environment']
                        or chosen['difficulty'] != chosen['metrics']['difficulty']):
                    raise ValueError('Selected identity differs from complete native outcome')
                pack = Path(ref['path']).parent / receipt['tar']
                if pack.parent != Path(ref['path']).parent or pack.stat().st_size != receipt['bytes']:
                    raise ValueError('Selected pack path or size changed')
                left, rgb, right_rgb, valid, quality, refs = reproduce(
                    entry, pack, source_root, audit['policy_sha256'], chosen['metrics'])
                tiles.append(tile(chosen, entry['image'], left, rgb, right_rgb, valid))
                records.append(dict(chosen, image=entry['image'], source_receipts=refs,
                    member_sha256=entry['sha256'], source_quality=quality,
                    page=f'{i // 4:03d}.jpg', page_row=i % 4,
                    arrays_camera_mask_and_statistics_reproduced_exactly=True))
                if len(tiles) == 4 or i == len(selected) - 1:
                    page = Image.new('RGB', (1280, 310 * len(tiles)), 'white')
                    for j, view in enumerate(tiles):
                        page.paste(view, (0, j * 310))
                    buffer = io.BytesIO()
                    page.save(buffer, format='JPEG', quality=96)
                    path = output / f'{len(pages):03d}.jpg'
                    atomic_write(path, buffer.getvalue())
                    pages.append(file_ref(path))
                    tiles.clear()
                    print(json.dumps(dict(processed=len(records), expected=len(selected))), flush=True)
    for ref in (audit_ref, audit['review_candidates'], audit['policy']):
        checked_path(ref)
    result = dict(stage='native_visual_evidence_rendered_not_reviewed', audit=audit_ref,
        selection=audit['review_candidates'], producer_policy=audit['policy'], pages=pages, records=records,
        rendered_images=len(records), all_selected_views_included=True,
        source_member_sha_crc_verified=True, selected_payloads_rechecked=True,
        arrays_camera_mask_and_statistics_reproduced_exactly=True,
        fixed_depth_limits_m=[.1, 200.], gray_pixels='Native finite depth above task limit, not supervised',
        display_only_resizing=True, selection_is_not_random_prevalence=True,
        full_pack_payload_hashes_rechecked=False, independent_physical_depth_accuracy_checked=False,
        source_quality_accepted=False, native_holes_filled=False, automatic_exclusions=0,
        network_requests=0, model_inference_performed=False, training_admission=False, optimizer_updates=0,
        generator=file_ref(Path(__file__)), helpers=[file_ref(Path(__file__).with_name(n)) for n in
            ('audit_tartanair_outcomes.py', 'review_tartanair_rejections.py')])
    publish_json(output / 'index.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    render(args.audit.resolve(), args.source_root.resolve(), args.output.resolve())
