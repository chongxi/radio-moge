"""Transfer existing scanner contradictions only to unchanged saved v2 surfaces.

No source meshes/scans/images or models are read; no rays are rerun. Published
scan/pose limitations remain attached to the prior conditional evidence.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from radio_moge.geometry_cache import publish_json, sha_file


def read_bound(ref, root):
    # Local mirror has a different root; bytes, not a rewritten reference,
    # establish the identity of copied evidence.
    path = Path(ref['path'])
    remote = '/home/buildhub/shared/radio-moge/chongxi/geometry-corpus-20261102/'
    if str(path).startswith(remote):
        path = root/str(path)[len(remote):]
    if sha_file(path) != ref['sha256']:
        raise ValueError('Changed saved evidence')
    return json.loads(path.read_text())


def same_camera_and_source(old, new):
    if (old['corpus_ordinal'] != new['corpus_ordinal'] or old['grid'] != new['grid']
            or old['image'] != new['image'] or old['receipt'] != new['old_receipt']
            or old['entry_key'] != new['old_entry_key']
            or not np.array_equal(old['intrinsics'], new['intrinsics'])
            or not np.array_equal(old['c2w'], new['c2w'])):
        raise ValueError('Original camera, image or cache identity differs')
    previous = {r['path']: r['sha256'] for r in old['source_refs']}
    if not new['source_refs'] or any(previous.get(r['path']) != r['sha256'] for r in new['source_refs']):
        raise ValueError('Source geometry or camera bytes differ')


def run(root, output):
    def ref(p):
        return dict(path=str(p), sha256=sha_file(p))
    reports = [root/'scannetpp-scanner-pose-correction-audit-01/report.json',
               root/'scannetpp-triangle-angle-sensitivity-04/report.json']
    development = root/'scannetpp-native-v2-development-01/report.json'
    dev = read_bound(ref(development), root)
    bound = {int(Path(r['path']).stem): r for r in dev['records']}
    observations = {}
    for p in reports:
        d = read_bound(ref(p), root)
        for i, row in enumerate(d['records']):
            if row['all_nine_tested_angles_retain_range_disagreement']:
                target = row['target']
                key = (target['corpus_ordinal'], *target['pixel_yx'])
                observations.setdefault(key, []).append((ref(p), i, row))
    if len(observations) != 7 or sum(map(len, observations.values())) != 10:
        raise ValueError('Exact seven DSLR targets and ten existing observations required')
    records = []
    for (ordinal, y, x), scans in sorted(observations.items()):
        first = scans[0][2]['target']
        if any(s[2]['target'] != first for s in scans):
            raise ValueError('Scanner observations disagree on DSLR target identity')
        old, new = read_bound(first['case'], root), read_bound(bound[ordinal], root)
        same_camera_and_source(old, new)
        rays = [q for q in old['rays'] if q['pixel_yx'] == [y, x]]
        if len(rays) != 1:
            raise ValueError('One exact prior DSLR ray required')
        ray = rays[0]
        hit = ray['analytical']['first_hits'][0]
        if (hit['face_id'] != first['face_id'] or ray['cached_z_m'] != first['cached_z_m']
                or not np.array_equal(hit['vertices_world'], first['vertices_world'])):
            raise ValueError('Prior ray does not bind scanner target triangle')
        path = root/'scannetpp-native-v2-development-01'/f'{ordinal}.npz'
        if sha_file(path) != new['arrays']['sha256']:
            raise ValueError('Changed v2 arrays')
        with np.load(path) as a:
            if (float(a['old_depth'][y, x]) != ray['cached_z_m']
                    or bool(a['old_valid'][y, x]) != ray['cached_valid']
                    or not np.array_equal(a['intrinsics'], old['intrinsics'])
                    or not np.array_equal(a['c2w'], old['c2w'])):
                raise ValueError('Saved old candidate or camera differs')
            supported = bool(a['final_support'][y, x])
            retained = supported and float(a['final_depth'][y, x]) == ray['cached_z_m']
            records.append(dict(corpus_ordinal=ordinal, pixel_yx=[y, x], scene=scans[0][2]['scene'],
                old_case=first['case'], new_case=bound[ordinal], arrays=new['arrays'],
                face_id=hit['face_id'], old_z_m=ray['cached_z_m'], old_supported=ray['cached_valid'],
                new_z_m=float(a['final_depth'][y, x]), new_supported=supported,
                new_gpu_z_m=float(a['raw_cuda_depth'][y, x]),
                new_public_z_m=float(a['raw_pyrender_depth'][y, x]),
                unchanged_original_surface_retained=retained,
                existing_conditional_source_contradiction_transfers=retained,
                observations=[dict(report=p, index=i, scan=s['scan'], scanner_pixel_yx=s['pixel_yx'],
                    minimum_tested_patch_clearance_m=s['sensitivity'][
                        'minimum_measured_patch_minus_maximum_tested_triangle_m']) for p, i, s in scans],
                replacement_surface_physically_accepted=False))
    output.mkdir(exist_ok=False)
    result = dict(stage='saved_v2_source_contradiction_transfer_complete', code=ref(Path(__file__)),
        development=ref(development), source_reports=[ref(p) for p in reports], records=records,
        unchanged_contradicted_surfaces_retained=sum(r['unchanged_original_surface_retained'] for r in records),
        prior_observations=10, unique_prior_DSLR_pixels=7,
        limitation='Existing nine-angle, shared-acquisition evidence; scanner pose uncertainty and independent physical calibration remain untested. A changed exposed surface does not inherit the old contradiction or gain acceptance.',
        new_source_reads=False, new_rays_or_renders=False, gpu_used=False,
        labels_changed=False, quarantines_cleared=False, training_admission=False, optimizer_updates=0)
    publish_json(output/'report.json', result)
    print(json.dumps(dict(records=len(records), retained=result['unchanged_contradicted_surfaces_retained'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
