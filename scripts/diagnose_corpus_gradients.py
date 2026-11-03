"""Bounded zero-update loss diagnosis on accepted training views and two checkpoints.

Reuse an existing source-stratified model-probe selection; select two views per
source/label by a fixed hash, independently of model results. Both checkpoints
see identical augmented RGB, native targets and RGB replay. No seal is issued.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from indoor_download import write_json
from probe_corpus_model import probe_batches
from radio_moge.config import load_config
from radio_moge.corpus_admission import (checked_json, checked_path, file_ref,
    implementation_hashes, readonly_db, training_contract_hash)
from radio_moge.corpus_gradient_diagnostic import observe_components
from radio_moge.geometry_cache import canonical_json, publish_json
from radio_moge.losses import FeatureStatistics
from radio_moge.system import build_system


class SelectedReplayRecords(list):
    verify_on_read = True


def choose_entries(entries):
    groups = defaultdict(list)
    for item in entries:
        row = item['entry']['image']
        kind = item['entry']['geometry_metadata']['label_kind']
        if row['split'] != 'train' or not isinstance(item['ordinal'], int):
            raise ValueError('Only exact training entries may be diagnosed')
        groups[(row['dataset'], kind)].append(item)
    selected = []
    for values in groups.values():
        ordered = sorted(values, key=lambda x: hashlib.sha256(
            ('component-gradient-v1:' + str(x['ordinal'])).encode()).hexdigest())
        first = ordered[0]
        second = next((x for x in ordered[1:] if x['group'] != first['group']),
                      ordered[1] if len(ordered) > 1 else None)
        selected.extend([first] + ([second] if second else []))
    selected.sort(key=lambda x: x['ordinal'])
    if not 1 <= len(selected) <= 32 or len({r['ordinal'] for r in selected}) != len(selected):
        raise ValueError('Bounded distinct diagnostic selection required')
    return selected


def bind_entries(selected, db):
    for item in selected:
        actual = db.execute('SELECT pack,entry_json,disposition FROM images WHERE ordinal=?',
                            (item['ordinal'],)).fetchone()
        if (actual is None or actual[2] != 'selected' or actual[0] != item['pack']
                or json.loads(actual[1]) != item['entry']):
            raise ValueError('Diagnostic selection differs from accepted accounting')


def replay_representatives(ref, datasets):
    chosen, digest, lines = {}, hashlib.sha256(), 0
    with Path(ref['path']).open('rb') as file:
        for line in file:
            digest.update(line)
            if not line.strip():
                continue
            lines += 1
            if set(chosen) != set(datasets):
                row = json.loads(line)
                if row['dataset'] not in chosen:
                    if (row['dataset'] not in datasets or row.get('split') != 'train'
                            or 'depth' in row or 'geometry' in row or not row.get('group')):
                        raise ValueError('Only original training RGB replay is allowed')
                    chosen[row['dataset']] = row
    if digest.hexdigest() != ref['sha256'] or set(chosen) != set(datasets):
        raise ValueError('Replay identity or source coverage differs')
    return SelectedReplayRecords(chosen[k] for k in sorted(chosen)), lines


def batch_identity(batch):
    digest = hashlib.sha256(canonical_json(batch['identities']))
    arrays = [('image', batch['image'])]
    if 'replay_image' in batch:
        arrays.append(('replay_image', batch['replay_image']))
    for i, target in enumerate(batch['geometry_targets']):
        for key in ('points', 'query_uv', 'valid', 'intrinsics', 'native_index'):
            arrays.append((f'target{i}/{key}', target[key]))
        digest.update(canonical_json({k: target[k] for k in ('label_kind', 'supervision')}))
    for name, array in arrays:
        array = np.ascontiguousarray(array.detach().cpu().numpy() if torch.is_tensor(array) else array)
        digest.update(canonical_json(dict(name=name, dtype=str(array.dtype), shape=list(array.shape))))
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def run(plan_path, output):
    plan_ref = file_ref(plan_path)
    plan = checked_json(plan_ref)
    config = load_config(checked_path(plan['training_config']))
    gate = checked_json(plan['acceptance'])
    model = checked_json(plan['prior_model_probe'])
    selection = checked_json(plan['prior_selection'])
    hashes = implementation_hashes()
    if (gate['stage'] != 'accepted_for_training' or gate['training_admission'] is not True
            or config['data']['geometry_corpus'] != plan['acceptance']
            or gate['training_contract_sha256'] != training_contract_hash(config)
            or hashes != gate['implementation_sha256'] or hashes != model['implementation_sha256']
            or gate['checks']['model_zero_update'] != plan['prior_model_probe']
            or plan['prior_selection'] not in model['evidence']
            or gate['index']['sha256'] != model['index_sha256']
            or set(plan['checkpoints']) != {'feature_baseline', 'geometry_final'}
            or config['train']['batch_size'] != 1 or not 1 <= plan['max_seconds'] <= 1800):
        raise ValueError('Unchanged accepted configuration, implementation and prior probe required')
    if str(torch.__version__) != '2.8.0+cu128' or torch.version.cuda != '12.8':
        raise ValueError('Use established actual-model runtime')
    checked_path(gate['index'])
    selected = choose_entries(selection['entries'])
    with readonly_db(gate['index']['path']) as db:
        bind_entries(selected, db)
        strata = set(db.execute("SELECT DISTINCT dataset,label_kind FROM images WHERE disposition='selected'"))
    if {(x['entry']['image']['dataset'], x['entry']['geometry_metadata']['label_kind'])
            for x in selected} != strata:
        raise ValueError('Every accepted source/label stratum must be diagnosed')
    replay_ref = dict(path=config['feature_replay']['manifest'],
                      sha256=config['feature_replay']['manifest_sha256'])
    replay, replay_count = replay_representatives(replay_ref, config['feature_replay']['mixture_weights'])
    for value in plan['checkpoints'].values():
        checked_path(value['file'])
    for ref in model['backbone_checkpoints'].values():
        checked_path(ref)
    for ref in model['source_revisions'].values():
        revision = subprocess.check_output(['git', '-C', ref['path'], 'rev-parse', 'HEAD'], text=True).strip()
        changes = subprocess.check_output(['git', '-C', ref['path'], 'status', '--porcelain'], text=True)
        if revision != ref['commit'] or changes.strip():
            raise ValueError('Pinned upstream model source differs')
    output.mkdir(parents=True, exist_ok=False)
    publish_json(output / 'plan.json', plan)
    publish_json(output / 'selection.json', dict(entries=selected,
        method='Two fixed-hash views per stratum from prior probe, preferring distinct groups; not prevalence.',
        source=plan['prior_selection'], expected_cases_per_checkpoint=2 * len(selected),
        replay_manifest=replay_ref, replay_manifest_images=replay_count,
        replay_representatives=list(replay), replay_representatives_only=True,
        training_admission=False))
    started = time.time()
    status = dict(stage='loading_models', started_at=started, optimizer_updates=0,
                  completed_cases=0, expected_cases=4 * len(selected))
    write_json(output / 'status.json', status)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(config['train']['seed'])
        system, adapter = build_system(config, 'cuda')
        statistics = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim']).cuda()
        input_hashes, reports = {}, {}
        for arm in ('feature_baseline', 'geometry_final'):
            checkpoint = plan['checkpoints'][arm]
            state = torch.load(checkpoint['file']['path'], weights_only=True, map_location='cpu', mmap=True)
            if (state['step'] != checkpoint['step'] or state['adapter_config'] != adapter.config
                    or state['model_config'] != system.model_config
                    or state['config']['models'] != config['models']):
                raise ValueError('Checkpoint architecture, source step or frozen models differ')
            adapter.load_state_dict(state['adapter'], strict=True)
            statistics.load_state_dict(state['statistics'], strict=True)
            del state
            if any(not torch.isfinite(t).all() or not (t > 0).all() for t in statistics.buffers()):
                raise ValueError('Invalid frozen feature statistics')
            def batches():
                for cursor, batch in enumerate(probe_batches(selected, replay, config)):
                    if time.time() - started > plan['max_seconds']:
                        raise TimeoutError('Diagnostic wall-clock bound reached')
                    checksum = batch_identity(batch)
                    if arm == 'feature_baseline':
                        input_hashes[cursor] = checksum
                    elif input_hashes[cursor] != checksum:
                        raise ValueError('Checkpoint comparisons used different input views')
                    yield batch
            with (output / f'{arm}-cases.jsonl').open('x') as log:
                def on_case(row):
                    row['input_sha256'] = input_hashes[row['cursor']]
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                    status.update(stage='observing_gradients', arm=arm, updated_at=time.time(),
                                  completed_cases=status['completed_cases'] + 1)
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
                report = observe_components(system, adapter, statistics, batches(), config, on_case)
            if len(report.pop('rows')) != 2 * len(selected):
                raise ValueError('Incomplete paired diagnostic scope')
            report.update(checkpoint=checkpoint, cases=2 * len(selected),
                          case_records=file_ref(output / f'{arm}-cases.jsonl'))
            reports[arm] = report
            publish_json(output / f'{arm}-state-report.json', report)
        for name in ('radio', 'encoder', 'decoder', 'statistics'):
            if reports['feature_baseline']['state_before'][name] != reports['geometry_final']['state_before'][name]:
                raise ValueError('Backbones or feature normalization differ between checkpoints')
        for ref in (plan_ref, plan['training_config'], plan['acceptance'], plan['prior_model_probe'],
                    plan['prior_selection'], replay_ref, *[v['file'] for v in plan['checkpoints'].values()]):
            checked_path(ref)
        if hashes != implementation_hashes():
            raise ValueError('Production dependency bytes changed during diagnosis')
        result = dict(stage='component_gradients_observed', reports=reports,
            optimizer_updates=0, optimizer_constructed=False, training_admission=False,
            actual_model_gradients=True, paired_augmented_inputs_identical=True,
            original_configuration_modified=False, implementation_sha256=hashes,
            source_models=model['source_revisions'], backbone_checkpoints=model['backbone_checkpoints'],
            selection=file_ref(output / 'selection.json'), plan=plan_ref,
            elapsed_seconds=time.time() - started, torch=str(torch.__version__), cuda=torch.version.cuda,
            diagnostic_code=file_ref(Path(__file__)), helper_code=file_ref(Path(__file__).parents[1] /
                'src/radio_moge/corpus_gradient_diagnostic.py'),
            limitations=reports['feature_baseline']['limitations'])
        publish_json(output / 'report.json', result)
        status.update(stage='complete', completed_at=time.time())
        write_json(output / 'status.json', status)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-lock', type=Path, required=True)
    args = parser.parse_args()
    if not args.gpu_lock.is_absolute() or not args.gpu_lock.is_file():
        parser.error('Use the existing absolute GPU training lock')
    with args.gpu_lock.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.plan.resolve(), args.output.resolve())
