"""Display the exact full-audit IRS extremes, without inventing a preflight pass.

Only selected members and RGB are read back. Entire tar payloads are not rehashed;
the producer report supplies the earlier full readback. No acceptance is issued.
"""
import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from visualize_teacher_preflight import view_row


def render(audit_path, output):
    audit_ref = file_ref(audit_path)
    audit = checked_json(audit_ref)
    generation = checked_json(audit['generation_report'])
    selected = checked_json(audit['review_candidates'])
    if (audit['stage'] != 'teacher_outcomes_accounted_not_accepted' or audit['dataset'] != 'irs'
            or audit['producer_array_readback_complete'] is not True
            or generation['stage'] != 'complete'
            or generation['verified_cache_members'] != audit['candidate_images']
            or not 1 <= len(selected) == audit['review_candidates_count'] <= 256
            or len({r['corpus_ordinal'] for r in selected}) != len(selected)):
        raise ValueError('Complete full IRS audit and bounded distinct review selection required')
    output.mkdir(exist_ok=False)
    records, pages = [], []
    for start in range(0, len(selected), 4):
        tiles, identities = [], []
        for ordinal in range(start, min(start + 4, len(selected))):
            chosen = selected[ordinal]
            receipt = checked_json(chosen['receipt'])
            entries = [e for e in receipt['entries'] if e['key'] == chosen['key']]
            if len(entries) != 1 or receipt['policy_sha256'] != audit['policy_sha256']:
                raise ValueError('Selected IRS member does not belong to the audited policy')
            entry = entries[0]
            row = entry['image']
            if (row['dataset'] != 'irs' or row['group'] != chosen['group']
                    or row['image'] != chosen['image'] or row['sha256'] != chosen['sha256']
                    or row['eye'] != chosen['eye'] or row['corpus_ordinal'] != chosen['corpus_ordinal']):
                raise ValueError('Selected IRS image, eye or family differs from the frozen receipt')
            pack = Path(chosen['receipt']['path']).parent / receipt['tar']
            if pack.stat().st_size != receipt['bytes']:
                raise ValueError('Selected IRS pack size changed')
            tile, identity = view_row(pack, entry, ordinal)
            for actual, expected in ((identity['valid_fraction'], chosen['valid_fraction']),
                    (identity['relative_depth_quantiles'][0], chosen['p01_over_median']),
                    (identity['relative_depth_quantiles'][2], chosen['p99_over_median'])):
                if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-12):
                    raise ValueError('Selected payload statistics differ from the full receipt audit')
            identity.update(corpus_ordinal=chosen['corpus_ordinal'], eye=chosen['eye'],
                selection_reasons=chosen['selection_reasons'], receipt=chosen['receipt'],
                pack=dict(path=str(pack), producer_sha256=receipt['sha256'], rehashed=False),
                member_sha256=entry['sha256'], member_payload_rechecked=True,
                max_over_median=chosen['max_over_median'])
            tiles.append(tile)
            identities.append(identity)
        page = Image.new('RGB', (1080, sum(t.height for t in tiles)), '#cccccc')
        offset = 0
        for tile, identity in zip(tiles, identities):
            page.paste(tile, (0, offset))
            identity.update(page=f'{len(pages):03d}.jpg', page_y=offset, page_height=tile.height)
            offset += tile.height
        target = output / f'{len(pages):03d}.jpg'
        incoming = target.with_suffix('.jpg.incoming')
        with incoming.open('xb') as file:
            page.save(file, format='JPEG', quality=94)
            file.flush()
            os.fsync(file.fileno())
        os.rename(incoming, target)
        pages.append(file_ref(target))
        records.extend(identities)
    checked_json(audit_ref)
    publish_json(output / 'index.json', dict(stage='visual_evidence_rendered_not_reviewed',
        audit=audit_ref, selection=audit['review_candidates'], rendered_images=len(records),
        records=records, pages=pages, all_selected_views_included=True,
        selection_method='Top two relative p99 tails and least-support two per family/eye; not random prevalence',
        display='Original RGB, per-image relative log depth, unchanged support; display-only resizing',
        pack_payload_hashes_rechecked=False, selected_member_payloads_rechecked=True,
        generator=file_ref(Path(__file__)), display_generator=file_ref(Path(__file__).with_name(
            'visualize_teacher_preflight.py')), training_admission=False, optimizer_updates=0))
    print(json.dumps(dict(rendered_images=len(records), pages=len(pages), output=str(output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.audit.resolve(), args.output.resolve())
