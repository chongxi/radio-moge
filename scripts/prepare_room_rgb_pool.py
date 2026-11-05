"""Join already verified room RGB manifests without rereading rendered payloads."""
import argparse
from collections import Counter
import json
from pathlib import Path

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json


def check_rows(rows, split):
    for row in rows:
        if (row.get('split') != split or row.get('agent_visible') is not False
                or row.get('geometry_label') is not None
                or row.get('label_role') != 'feature_only_rgb'
                or row.get('training_admission') != (split == 'train')
                or any(row.get(k) is not None for k in ('depth', 'geometry'))):
            raise ValueError('Room inputs must be explicitly feature-only RGB in their declared split')
        source = row.get('dataset')
        if source == 'procthor_empty':
            if (row.get('projection') != 'perspective'
                    or not row.get('group', '').startswith(f'procthor10k/{split}/')):
                raise ValueError('Official ProcTHOR projection or group changed')
        elif source == 'structural_rooms':
            projection = row.get('projection')
            # The frozen validation manifest includes seven earlier, already
            # pixel-checked scenes. Only those validation rows lack this tag.
            legacy_val = (split == 'val' and row.get('frame_policy') is None
                and type(row.get('scene_ordinal')) is int and 0 <= row['scene_ordinal'] <= 6)
            if (projection not in ('perspective', 'orthographic')
                    or row.get('camera_model') != ('pinhole' if projection == 'perspective' else 'parallel')
                    or row.get('user_approved_projection') is not True
                    or row.get('geometry_admission') is not False
                    or (row.get('frame_policy') != 'strict-rgb-frame-filter-v2' and not legacy_val)
                    or any(row.get(k) != 0 for k in ('window_count', 'door_count', 'exterior_background_pixels'))
                    or not row.get('group', '').startswith(f'structural-room-rgb-multiview-v1/{split}/')):
                raise ValueError('Structural source, camera or closed-room contract changed')
        else:
            raise ValueError('Undeclared synthetic source')


def overlap(a, b):
    result = {}
    for key in ('image', 'group', 'decoded_rgb_sha256', 'original_house_sha256',
                'empty_house_sha256', 'scene_layout_sha256'):
        result[key] = len({r[key] for r in a if r.get(key)} & {r[key] for r in b if r.get(key)})
    return result


def prepare(request, output):
    old = checked_json(request['procthor_report'])
    old_closure = checked_json(request['procthor_closure'])
    new = checked_json(request['structural_report'])
    visual = checked_json(request['structural_visual'])
    val = checked_json(request['structural_val_report'])
    val_visual = checked_json(request['structural_val_visual'])
    official_val = checked_json(request['procthor_val_report'])
    if (old['images'] != 40000 or old['houses'] != 2000
            or old_closure['combined'] != request['procthor_report']
            or old_closure['combined_manifest'] != old['manifest']
            or old_closure['stage'] != 'completed_fourth_batch_frozen_combined_and_sample_reviewed'
            or new['stage'] != 'frozen_verified_structural_feature_rgb'
            or new['images'] != 890 or new['split'] != 'train'
            or not new['feature_training_eligible'] or not new['every_selected_rgb_decoded_and_hashed']
            or visual['source_report'] != request['structural_report']
            or visual['stage'] != 'sampled_rgb_visual_review_complete'
            or not visual['feature_rgb_data_eligible']
            or val['images'] != 317 or val['split'] != 'val'
            or val['feature_training_eligible'] or not val['every_selected_rgb_decoded_and_hashed']
            or val_visual['source_report'] != request['structural_val_report']
            or val_visual['stage'] != 'sampled_rgb_visual_review_complete'
            or official_val['images'] != 64 or official_val['split'] != 'val'):
        raise ValueError('Frozen pool or sampled-review receipt does not match the authorized scope')
    checked_json(old_closure['visual_review'])
    manifests = [checked_path(r['manifest']) for r in (old, new, official_val, val)]
    contents = [p.read_bytes() for p in manifests]
    if any(not content.endswith(b'\n') for content in contents):
        raise ValueError('Frozen manifests must have closed newline-terminated records')
    rows = [[json.loads(line) for line in content.splitlines() if line] for content in contents]
    if list(map(len, rows)) != [40000, 890, 64, 317]:
        raise ValueError('Frozen manifest counts changed')
    for records, split in zip(rows, ('train', 'train', 'val', 'val')):
        check_rows(records, split)
        if any(len({r[k] for r in records}) != len(records) for k in ('image', 'decoded_rgb_sha256')):
            raise ValueError('Duplicate RGB identity within a frozen population')
    isolated = []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3)):
        counts = overlap(rows[i], rows[j])
        if any(counts.values()):
            raise ValueError('Training/validation or cross-source identity overlap')
        isolated.append(dict(left=file_ref(manifests[i]), right=file_ref(manifests[j]), overlaps=counts))
    all_train = rows[0] + rows[1]
    output.mkdir(exist_ok=False)
    incoming = output / 'rgb.jsonl.incoming'
    incoming.write_bytes(contents[0] + contents[1])
    incoming.replace(output / 'rgb.jsonl')
    report = dict(stage='frozen_room_feature_pool_ready', request=request,
        manifest=file_ref(output/'rgb.jsonl'), images=len(all_train),
        source_counts=dict(Counter(r['dataset'] for r in all_train)),
        projections=dict(Counter(r['projection'] for r in all_train)),
        groups=len({r['group'] for r in all_train}), isolation=isolated,
        input_lines_preserved_byte_for_byte=True, original_image_readbacks_reused=True,
        original_sampled_visual_reviews_reused=True, all_images_manually_reviewed=False,
        geometry_loss=False, geometry_labels=False, growing_generation_included=False)
    publish_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(checked_json(file_ref(args.request)), args.output.resolve())
    print(json.dumps(dict(report=file_ref(args.output/'report.json'), images=report['images'])))
