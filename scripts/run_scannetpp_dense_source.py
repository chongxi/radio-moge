"""Bounded CPU full-grid pilot on one existing development view, never training.

Benchmark/parity must finish before a separate bounded full-grid run. Reuse saved
numerical arrays; do not open confirmation images or rerender any depth/RGB.
"""
import argparse
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

from indoor_download import write_json
from probe_scannetpp_v2_source_pilot import bound_path, read, target_slots
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.mesh_clip import MeshClipContract
from radio_moge.scanner_support import ScannerSupportContract, dense_image_support, paired_support
from run_scannetpp_source_confirmation import implementation as scalar_implementation
from scannetpp_dense_source import StationSession, TopologyIndex, merge_masks

ORDINAL = 851268  # first prospectively selected development control, not selected by success
PILOT_SHA = 'a9efd96a83ca590a382d27d86b8f8867ac13a3920eb15dc522fbe21f92bd57e7'
POLICY_SHA = '3216cac67ac2a03854858b174a6362f8bddbb73940c8018b610719de0ac23d75'
EXCLUSIONS_SHA = '58a1b02acfe087db2c28e4ae334c762359d0efc6ab16a6d99dd3deba7ab3d86b'
BINDINGS_SHA = 'e29cd1490678bd368b59977551049cb00aa868509430eff2a51cdbefdbfc62df'


def implementation():
    root = Path(__file__).resolve().parents[1]
    return dict(scalar_implementation(), **{n: sha_file(root/n) for n in
        ['scripts/run_scannetpp_dense_source.py', 'scripts/scannetpp_dense_source.py']})


def freeze(root, output, mode, benchmark=None):
    refs = {k: file_ref(root/p) for k, p in dict(
        pilot='scannetpp-native-v2-source-pilot-plan-01/report.json',
        policy='scannetpp-native-v2-conditional-policy-plan-01/report.json',
        bindings='scannetpp-native-v2-source-pilot-01/source-bindings.json',
        exclusions='scannetpp-native-candidate-decisions-01/known-geometry-exclusions.jsonl',
        old_decisions=f'scannetpp-native-v2-conditional-support-01/{ORDINAL}-decisions.json',
        old_targets=f'scannetpp-native-v2-source-pilot-01/{ORDINAL}-targets.json').items()}
    for key, expected in [('pilot', PILOT_SHA), ('policy', POLICY_SHA),
                          ('bindings', BINDINGS_SHA), ('exclusions', EXCLUSIONS_SHA),
                          ('old_decisions', 'd64e0f5158badca1f9ec70f1d755a024da0d7150cfa711e2f88c794503ec58e4'),
                          ('old_targets', '591dca21b536562d57c0fc81e2c890966022f038ee0c6e65d45ac448c87e5154')]:
        if refs[key]['sha256'] != expected:
            raise ValueError('Frozen source/policy evidence changed')
    pilot, policy = read(refs['pilot'], root), read(refs['policy'], root)
    if policy['policy'] != asdict(ScannerSupportContract()):
        raise ValueError('Frozen source policy changed')
    rows = read(pilot['selection'], root)
    if [r['corpus_ordinal'] for r in rows] != [851268, 1318289, 1415712, 1412812]:
        raise ValueError('Original development selection required')
    row = rows[0]
    binding = read(refs['bindings'], root)[0]
    if (row['role'] != 'ordinary_control' or binding['corpus_ordinal'] != ORDINAL
            or [s['scan'] for s in binding['scanners']] != ['0', '1']):
        raise ValueError('Only first existing development view and fixed scans0/1 allowed')
    h, w = row['grid']
    old_pixels = row['pixel_yx']
    # Independent of depth/support: fixed full-grid row-major strata for timing.
    extra = [[int((i+.5)*h*w/256)//w, int((i+.5)*h*w/256)%w] for i in range(256)]
    pixels = old_pixels + [p for p in extra if p not in old_pixels]
    plan = dict(stage='bounded_dense_source_plan_frozen', mode=mode, corpus_ordinal=ORDINAL,
        row=row, binding=binding, evidence=refs, implementation=implementation(),
        policy=asdict(ScannerSupportContract()), clip_contract=MeshClipContract().as_dict(),
        parity_original_slots=96, benchmark_pixels=pixels, grid=[h, w],
        maximum_wall_seconds=900 if mode == 'benchmark' else 7200,
        maximum_process_RSS_GiB=24, maximum_output_GiB=8, maximum_mesh_faces=20_000_000,
        chunk_slots=128, scanner_order=['0', '1'], one_raw_scanner_in_memory=True,
        new_RGB_reads=0, new_numerical_renders=0, confirmation_images_read=0,
        source_quality_accepted=False, training_admission=False, optimizer_updates=0,
        physical_accuracy_accepted=False)
    if mode == 'full':
        if benchmark is None:
            raise ValueError('Completed exact-parity/timing benchmark required before dense run')
        b = read(file_ref(benchmark), root)
        if (b['stage'] != 'dense_source_benchmark_complete' or b['implementation'] != implementation()
                or b['parity_slots'] != 96 or b['parity_passed'] is not True
                or b['corpus_ordinal'] != ORDINAL or b['estimated_full_seconds'] > 5400
                or b['maximum_RSS_GiB'] > 16 or b['plan_evidence'] != refs):
            raise ValueError('Exact current parity and runtime/memory headroom required')
        plan['benchmark'] = file_ref(benchmark)
        plan['estimated_full_seconds'] = b['estimated_full_seconds']
    output.mkdir(exist_ok=False)
    publish_json(output/'plan.json', plan)
    return plan


def write_chunk(path, rows):
    incoming = path.with_suffix(path.suffix+'.incoming')
    with incoming.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as packed:
            for row in rows:
                packed.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n')
        raw.flush()
        os.fsync(raw.fileno())
    os.rename(incoming, path)
    return file_ref(path)


def footprint_equivalent(a, b):
    # Ignore no metadata: these are the full core decision payloads, not only pass bits.
    return all(a[k] == b[k] for k in
               ('projected_footprint', 'angle_matches', 'exact_target_visibility'))


def run(root, output, mode, benchmark=None):
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    # Real assets only on the64GiB remote allocation, never local8GiB workspace.
    memory_root = Path('/sys/fs/cgroup')
    limit, used = ((memory_root/name).read_text().strip() for name in ('memory.max', 'memory.current'))
    if not limit.isdigit() or int(limit) < 40*1024**3 or int(limit)-int(used) < 32*1024**3:
        raise RuntimeError('Explicit remote memory allocation and32GiB free cgroup headroom required')
    started = time.monotonic()
    plan = freeze(root, output, mode, benchmark)
    status = dict(stage='loading_bound_saved_arrays_and_mesh', mode=mode, started_at=time.time(),
        corpus_ordinal=ORDINAL, completed_station_slots=0, expected_station_slots=0,
        training_admission=False, optimizer_updates=0)
    write_json(output/'status.json', status)
    def check_budget():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        if rss > plan['maximum_process_RSS_GiB']:
            raise RuntimeError('Declared24GiB resident memory bound exceeded')
        return time.monotonic()-started < plan['maximum_wall_seconds']
    def startup_budget():
        if not check_budget():
            raise RuntimeError('Declared wall budget exhausted during source preparation')
    try:
        row, binding = plan['row'], plan['binding']
        record = read(row['saved_v2_record'], root)
        if (record['corpus_ordinal'] != ORDINAL or record['role'] != 'ordinary_control'
                or record['grid'] != row['grid']):
            raise ValueError('Bound existing numerical view changed')
        with np.load(bound_path(record['arrays'], root)) as a, np.load(bound_path(row['saved_numerical_mask'], root)) as m:
            depth, numeric = a['final_depth'].copy(), m['consensus_support'].copy()
            if (numeric.dtype != bool or depth.shape != numeric.shape
                    or (numeric & ~a['final_support']).any()
                    or not np.array_equal(a['intrinsics'], record['intrinsics'])
                    or not np.array_equal(a['c2w'], record['c2w'])):
                raise ValueError('Retained numerical camera/support changed')
        startup_budget()
        poses = np.asarray(read(binding['scanner_poses'], root))
        if np.array_equal(poses[0], poses[1]):
            raise ValueError('Both fixed distinct published stations required')
        mesh_refs = [p for p in record['source_refs'] if p['path'].endswith('.ply')]
        if len(mesh_refs) != 1:
            raise ValueError('One bound original mesh required')
        audit = read(binding['audit'], root)
        if mesh_refs[0]['sha256'] != audit['mesh_sha256']:
            raise ValueError('Numerical/source mesh binding differs')
        mesh = trimesh.load_mesh(bound_path(mesh_refs[0], root), process=False)
        startup_budget()
        topology = TopologyIndex(mesh.vertices, mesh.faces, maximum_faces=plan['maximum_mesh_faces'])
        startup_budget()
        engine = RayMeshIntersector(mesh)
        h, w = numeric.shape
        pixels = plan['benchmark_pixels'] if mode == 'benchmark' else [[i//w, i%w] for i in range(h*w)]
        target_started = time.monotonic()
        targets = target_slots(mesh, engine, record, pixels, depth, numeric, MeshClipContract())
        startup_budget()
        target_ref = write_chunk(output/'targets.jsonl.gz', targets)
        target_seconds = time.monotonic()-target_started
        old = read(plan['evidence']['old_decisions'], root)
        prior_targets = read(plan['evidence']['old_targets'], root)['slots']
        reference_index = {tuple(t['pixel_yx']): t['slot'] for t in prior_targets}
        for target in targets:
            if tuple(target['pixel_yx']) in reference_index:
                slot = reference_index[tuple(target['pixel_yx'])]
                if dict(target, slot=slot) != prior_targets[slot]:
                    raise ValueError('Batch target geometry differs from saved scalar96 targets')
        excluded = {json.loads(s)['corpus_ordinal'] for s in bound_path(plan['evidence']['exclusions'], root).read_text().splitlines()}
        if len(excluded) != 4190:
            raise ValueError('All exact prior exclusions required')
        masks, done = [np.zeros_like(numeric) for _ in range(2)], [np.zeros_like(numeric) for _ in range(2)]
        references, parity_rows, times = [], [[], []], []
        status['expected_station_slots'] = 2*len(targets)
        bytes_written = Path(target_ref['path']).stat().st_size
        stopped = False
        for i, scanner in enumerate(binding['scanners']):
            raw = {}
            for key, proof in scanner['inputs'].items():
                with Image.open(bound_path(proof, root)) as im:
                    raw[key] = np.asarray(im).copy()
                startup_budget()
            session = StationSession(raw, poses[int(scanner['scan'])], engine, topology)
            station_started = time.monotonic()
            for start in range(0, len(targets), plan['chunk_slots']):
                if not check_budget():
                    stopped = True
                    break
                chunk = session.evaluate(targets[start:start+plan['chunk_slots']], ORDINAL, binding['scene'], scanner['scan'])
                for observation in chunk:
                    y, x = observation['pixel_yx']
                    masks[i][y, x] = observation['decision']['conditional_station_support']
                    done[i][y, x] = True
                    if tuple(observation['pixel_yx']) in reference_index:
                        slot = reference_index[tuple(observation['pixel_yx'])]
                        before = next(r for r in old[slot]['station_evidence'] if r['scan'] == scanner['scan'])
                        if not footprint_equivalent(observation, before):
                            publish_json(output/'parity-failure.json', dict(slot=slot, scan=scanner['scan'],
                                actual=observation, saved=before))
                            raise ValueError('Optimized original96 source evidence differs')
                        parity_rows[i].append(dict(observation, slot=slot))
                p = output/f'scan{scanner["scan"]}-chunk{start:06d}.jsonl.gz'
                ref = write_chunk(p, chunk)
                references.append(ref)
                bytes_written += p.stat().st_size
                if bytes_written > plan['maximum_output_GiB']*1024**3:
                    raise RuntimeError('Declared8GiB output bound exceeded')
                status.update(stage='checking_conditional_source_full_grid' if mode == 'full' else 'benchmarking_conditional_source',
                    completed_station_slots=int(sum(a.sum() for a in done)), scan=scanner['scan'],
                    elapsed_seconds=time.monotonic()-started, updated_at=time.time())
                write_json(output/'status.json', status)
            times.append(time.monotonic()-station_started)
            del session, raw
            if stopped:
                break
        mask, evaluated = merge_masks(numeric, masks, done, ORDINAL in excluded)
        path = output/'conditional-mask.npz'
        incoming = output/'conditional-mask.npz.incoming'
        with incoming.open('xb') as f:
            np.savez_compressed(f, conditional_support=mask, numerical_support=numeric,
                both_stations_evaluated=evaluated, scan0_support=masks[0], scan1_support=masks[1],
                scan0_evaluated=done[0], scan1_evaluated=done[1])
        os.rename(incoming, path)
        parity = False
        if not stopped:
            if any(len(x) != 96 for x in parity_rows):
                raise ValueError('All original96 paired parity slots required')
            parity_rows = [sorted(x, key=lambda r: r['slot']) for x in parity_rows]
            for slot in range(96):
                decision = paired_support([x[slot] for x in parity_rows], ['0', '1'],
                    prior_targets[slot]['numerical_support'], ORDINAL in excluded)
                if decision != old[slot]['decision']:
                    raise ValueError('Final paired source decision differs from scalar baseline')
            parity = True
        complete = bool(evaluated.all()) and mode == 'full' and not stopped
        result = dict(stage=('dense_source_benchmark_complete' if mode == 'benchmark'
                            else 'full_grid_conditional_source_complete') if not stopped else 'bounded_source_run_incomplete',
            mode=mode, corpus_ordinal=ORDINAL, plan=file_ref(output/'plan.json'),
            plan_evidence=plan['evidence'], implementation=implementation(), targets=target_ref,
            chunks=references, mask=file_ref(path), evaluated_station_slots=int(sum(a.sum() for a in done)),
            requested_station_slots=2*len(targets), full_grid_pixels=h*w,
            dense_coverage=dense_image_support(numeric, mask, complete), full_grid_evaluated=complete,
            parity_slots=96 if parity else 0, parity_passed=parity,
            elapsed_seconds=time.monotonic()-started, station_processing_seconds=times,
            target_processing_seconds=target_seconds,
            estimated_full_seconds=((sum(times)+target_seconds)/len(targets)*h*w*1.25
                                    +(time.monotonic()-started-sum(times)-target_seconds))
                if mode == 'benchmark' and not stopped else None,
            maximum_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            output_bytes=bytes_written, source_quality_accepted=False, training_admission=False,
            optimizer_updates=0, confirmation_images_read=0, new_RGB_reads=0, new_numerical_renders=0,
            label_role='laser_mesh_derived conditional same-acquisition support; not physical accuracy certification')
        publish_json(output/'report.json', result)
        status.update(stage=result['stage'], elapsed_seconds=result['elapsed_seconds'])
        write_json(output/'status.json', status)
        print(json.dumps({k: result[k] for k in ('stage', 'elapsed_seconds', 'estimated_full_seconds', 'maximum_RSS_GiB', 'dense_coverage')}), flush=True)
        return result
    except Exception as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error), elapsed_seconds=time.monotonic()-started)
        write_json(output/'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=['benchmark', 'full'], required=True)
    parser.add_argument('--benchmark', type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.mode, args.benchmark)
