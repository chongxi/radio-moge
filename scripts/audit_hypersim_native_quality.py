"""Inspect the complete native Hypersim receipt distributions and bounded extremes.

Uses the completed frozen-row outcome audit. Does not rerun full payload reads,
clear native flags, alter geometry, or issue source/training acceptance.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_completed_rgb_teachers import rgb
from audit_hypersim_outcomes import outcome_kind
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json, read_geometry_member
from radio_moge.geometry_sampling import source_supervision


def run(root, output):
    audit_ref = file_ref(root / 'hypersim-outcomes-01/report.json')
    audit = checked_json(audit_ref)
    generation = checked_json(audit['generation_report'])
    packs = checked_json(audit['packs'])
    if (audit['stage'] != 'native_outcomes_accounted_not_accepted'
            or audit['expected_scope'] != 59344 or audit['producer_array_readback_complete'] is not True
            or generation['stage'] != 'complete' or generation['completed'] != 58739
            or generation['verified_cache_members'] != 58739 or packs['count'] != 58739
            or generation['rejected'] != 605 or generation['optimizer_updates'] != 0
            or generation['training_started'] or generation['training_admission']):
        raise ValueError('Complete native generation and outcome reconciliation required')
    with readonly_db(checked_path(audit['index'])) as db:
        originals = {row[0]: row[1:] for row in db.execute(
            "SELECT entry_key,ordinal,row_json,outcome,evidence_json FROM images WHERE outcome!='rejected'")}
    if len(originals) != 58739:
        raise ValueError('Original native cache scope differs')
    counts, records = Counter(), []
    pack_directory = Path(audit['packs']['path']).parent
    for packet in packs['packs']:
        receipt_ref = dict(path=str(pack_directory / packet['receipt']), sha256=packet['sha256'])
        receipt = checked_json(receipt_ref)
        tar = pack_directory / receipt['tar']
        if (tar.parent != pack_directory or tar.stat().st_size != receipt['bytes']
                or receipt['policy_sha256'] != audit['policy_sha256']
                or len(receipt['entries']) != packet['count']):
            raise ValueError('Receipt policy, count or pack size changed')
        for entry in receipt['entries']:
            meta, row = entry['geometry_metadata'], dict(entry['image'])
            bound = originals.pop(entry['key'], None)
            if bound is None:
                raise ValueError('Duplicate or unknown native cache member')
            ordinal, encoded, kind, proof = bound
            row.pop('corpus_ordinal', None)
            if (row != json.loads(encoded) or receipt_ref != json.loads(proof)
                    or outcome_kind(meta) != kind or meta['policy_sha256'] != audit['policy_sha256']
                    or meta['dataset'] != 'hypersim' or meta['image_sha256'] != row['sha256']
                    or hashlib.sha256(canonical_json(meta)).hexdigest() != entry['key']
                    or meta['label_kind'] != 'synthetic_renderer' or meta['units'] != 'meters'
                    or meta['depth_convention'] != 'camera_z' or meta['camera_kind'] != 'synthetic_camera'
                    or meta['metric_calibrated'] is not True):
                raise ValueError('Native provenance, identity or flag changed')
            counts[kind] += 1
            if kind == 'flagged_cache':
                continue
            if source_supervision(meta) != 'metric_native':
                raise ValueError('Native label loss route differs')
            h, w = meta['shape']
            q = np.asarray(entry['depth_quantiles_m'])
            support = meta['valid_pixels'] / (h * w)
            quality = meta['source_quality']
            if (support < .05 or support != entry['valid_fraction']
                    or support != quality['valid_fraction'] or meta['valid_pixels'] < 32
                    or (h, w) != (row['height'], row['width']) or q.shape != (5,)
                    or not np.isfinite(q).all() or q[0] <= 0 or np.any(np.diff(q) < 0)):
                raise ValueError('Native valid support or depth statistics differ')
            records.append(dict(corpus_ordinal=ordinal, image=row['image'], sha256=row['sha256'],
                group=row['group'], key=entry['key'], member_sha256=entry['sha256'], receipt=receipt_ref,
                valid_fraction=support, depth_quantiles_m=q.tolist(), p99_m=float(q[3]),
                median_depth_m=float(q[2]),
                quantization_bound_p99_m=quality['quantization_bound_quantiles_m'][2],
                source_inconsistency_fraction=quality['inconsistent_supported_pixels'] / quality['supported_pixels']))
    if originals or counts != Counter(unflagged_cache=55663, flagged_cache=3076):
        raise ValueError('Native counts no longer match the completed outcome audit')
    selected = {}
    for key, count, descending in [('valid_fraction', 4, False), ('p99_m', 4, True),
                                   ('quantization_bound_p99_m', 2, True)]:
        for row in sorted(records, key=lambda r: ((-1 if descending else 1) * r[key], r['corpus_ordinal']))[:count]:
            selected.setdefault(row['corpus_ordinal'], dict(row, reasons=[]))['reasons'].append(key)
    output.mkdir(exist_ok=False)
    visuals = []
    for i, chosen in enumerate(sorted(selected.values(), key=lambda r: r['corpus_ordinal'])):
        receipt = checked_json(chosen['receipt'])
        entry, = [e for e in receipt['entries'] if e['key'] == chosen['key']]
        depth, valid, _, meta = read_geometry_member(pack_directory / receipt['tar'], entry)
        image = rgb(entry['image'])
        if depth.shape != (image.height, image.width) or source_supervision(meta) != 'metric_native':
            raise ValueError('Selected original RGB and geometry grids differ')
        q = np.quantile(depth[valid], [0, .01, .5, .99, 1])
        np.testing.assert_allclose(q, chosen['depth_quantiles_m'], rtol=1e-6, atol=1e-12)
        if valid.mean() != chosen['valid_fraction']:
            raise ValueError('Selected actual support differs from receipt')
        panels = [image]
        for lo, hi in ((q[1], q[3]), (.1, 100.)):
            value = np.zeros(depth.shape, np.float32)
            value[valid] = np.clip((np.log(depth[valid]) - np.log(lo)) / max(1e-6, np.log(hi / lo)), 0, 1)
            colors = (np.stack((value, 1 - np.abs(2 * value - 1), 1 - value), -1) * 255).astype(np.uint8)
            colors[~valid] = 0
            panels.append(Image.fromarray(colors))
        panels.append(Image.fromarray((valid * 255).astype(np.uint8)).convert('RGB'))
        page = Image.new('RGB', (4 * image.width, image.height + 64), 'white')
        draw = ImageDraw.Draw(page)
        for y, line in enumerate((f"#{i} corpus {chosen['corpus_ordinal']} {meta['source_id']}",
                'RGB | per-image log p01-p99 | common log camera-Z [0.1,100] meters | unchanged support',
                f"support={valid.mean():.4f}, q01/50/99(m)={q[[1,2,3]].round(3).tolist()}, reasons={chosen['reasons']}")):
            draw.text((4, 3 + 18 * y), line, fill='black')
        for column, panel in enumerate(panels):
            page.paste(panel, (column * image.width, 64))
        path = output / f'{i:03d}.png'
        temporary = path.with_suffix('.png.incoming')
        page.save(temporary, format='PNG')
        os.rename(temporary, path)
        visuals.append(dict(chosen, page=file_ref(path), selected_payload_and_rgb_rechecked=True))
    probabilities = [0, .01, .5, .9, .99, 1]
    summary = {key: np.quantile([r[key] for r in records], probabilities).tolist()
               for key in ('valid_fraction', 'p99_m', 'median_depth_m', 'quantization_bound_p99_m',
                           'source_inconsistency_fraction')}
    for ref in (audit_ref, audit['index'], audit['generation_report'], audit['packs']):
        checked_path(ref)
    publish_json(output / 'report.json', dict(stage='native_source_quality_evidence_not_reviewed',
        dataset='hypersim', source_outcomes=audit_ref, policy_sha256=audit['policy_sha256'],
        cached_counts=dict(counts), eligible_before_visual_count=len(records),
        eligible_groups=len({r['group'] for r in records}), probabilities=probabilities,
        statistics=summary, selected=visuals, source_flags_cleared=False,
        selected_payloads_rechecked=True, full_pack_payloads_rehashed=False,
        producer_full_readback_bound=True, source_quality_accepted=False,
        training_admission=False, optimizer_updates=0, code=file_ref(Path(__file__))))
    print(json.dumps(dict(counts=dict(counts), selected=len(visuals), statistics=summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
