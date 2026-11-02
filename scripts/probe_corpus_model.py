"""Actual zero-update corpus preflight, after all four nonmodel gates pass.

Plan JSON: accounting_report, checks (the four nonmodel gates), training_config,
initial_checkpoint are absolute SHA256 file references; source_step is an exact
integer. This script never calls train(), creates an optimizer, or emits a seal.
Run once with a new output directory while holding the existing GPU lock.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

import torch

from indoor_download import write_json
from radio_moge.config import load_config, MOGE_COMMIT, RADIO_COMMIT
from radio_moge.corpus_admission import (checked_path, file_ref, readonly_db,
    training_contract_hash, validate_pre_model_evidence)
from radio_moge.corpus_data import collate_corpus
from radio_moge.corpus_model_probe import observe_model, select_probe_entries
from radio_moge.data import read_manifest, TrainingImages
from radio_moge.feature_replay import load_replay_records
from radio_moge.geometry_cache import publish_json
from radio_moge.geometry_sampling import load_candidate
from radio_moge.losses import FeatureStatistics
from radio_moge.system import build_system
from radio_moge.train import validate_config
from radio_moge.upstream import source_path


def probe_batches(selected, replay_records, config):
    h, w = config['data']['buckets'][0]
    if config['train']['batch_size'] != 1:
        raise ValueError('This bounded preflight supports the configured one-image microbatch only')
    # Stable representatives from every replay source. Use the production RGB
    # transform and replay seed; this stratified probe is not a sampler epoch.
    replay_choices = {}
    for index, row in enumerate(replay_records):
        replay_choices.setdefault(row['dataset'], index)
    choices = [replay_choices[k] for k in sorted(replay_choices)]
    if not choices:
        raise ValueError('Diverse RGB replay is missing')
    replay = TrainingImages(replay_records, config['data']['augment'])
    if len(selected) < len(choices):
        raise ValueError('Model probe cannot cover every replay source')
    for index, item in enumerate(selected):
        for slot in range(2):
            cursor = 2 * index + slot
            rgb, target, _ = load_candidate(item['pack'], item['entry'], (h, w),
                config['train']['seed'] + cursor, config['data']['augment'])
            batch = collate_corpus([dict(image=torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
                                        geometry_target=target)])
            identity = dict(ordinal=item['ordinal'], image_sha256=target['image_sha256'],
                dataset=target['dataset'], label_kind=target['label_kind'], group=item['group'],
                observations=int(target['valid'].sum()), native_grid=list(target['valid'].shape))
            if slot:
                replay_index = choices[index % len(choices)]
                record = replay_records[replay_index]
                if record.get('sha256'):
                    checked_path(dict(path=record['image'], sha256=record['sha256']))
                batch['replay_image'] = replay[(replay_index,
                    config['feature_replay']['seed'] + index, h, w)][None]
                identity['replay'] = dict(dataset=record['dataset'], group=record['group'],
                                          image=file_ref(record['image']))
            batch['identities'] = [identity]
            yield batch


def run(plan_path, output):
    plan_ref = file_ref(plan_path)
    plan = json.loads(checked_path(plan_ref).read_text())
    config = load_config(checked_path(plan['training_config']))
    validate_config(config, corpus_preflight=True)
    if config['train']['batch_size'] != 1 or type(plan['source_step']) is not int:
        raise ValueError('Exact initialization step and one-image microbatch are required')
    accounting, hashes = validate_pre_model_evidence(plan['accounting_report'], plan['checks'])
    if file_ref(config['data']['train_manifest'])['sha256'] != accounting['manifest_sha256']:
        raise ValueError('Training manifest is not the fully accounted corpus')
    checkpoint_path = checked_path(plan['initial_checkpoint'])
    if str(torch.__version__) != '2.8.0+cu128' or torch.version.cuda != '12.8':
        raise ValueError('Use the established Torch2.8cu128 model environment')
    heldout_paths = [config['data'].get(k) for k in ('val_manifest', 'geometry_val_manifest')]
    heldout_paths += [value['manifest'] for value in config.get('evaluation', {}).get('dataset_variants', {}).values()]
    heldout_refs = [file_ref(p) for p in heldout_paths if p]
    replay_ref = file_ref(config['feature_replay']['manifest'])
    replay_records = load_replay_records(config, [read_manifest(r['path']) for r in heldout_refs])
    with readonly_db(accounting['index']['path']) as db:
        if db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] != accounting['selected_images']:
            raise ValueError('Sample index differs from full accounting')
        selected = select_probe_entries(db)
    if {v['entry']['image']['dataset'] for v in selected} != set(accounting['expected_counts']):
        raise ValueError('Actual model probe must cover every corpus source')
    revisions = {}
    for name, revision in (('MoGe', MOGE_COMMIT), ('RADIO', RADIO_COMMIT)):
        override = config['models'].get('moge_source' if name == 'MoGe' else 'radio_source')
        path = source_path(name, revision, override)
        if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Pinned upstream source has uncommitted modifications')
        revisions[name] = dict(path=str(path), commit=revision)
    output.mkdir(parents=True, exist_ok=False)
    publish_json(output / 'plan.json', plan)
    selection_path = output / 'selection.json'
    publish_json(selection_path, dict(entries=selected, cases=2 * len(selected),
        selection='per-source/label support and depth extremes plus distinct scene representatives',
        training_admission=False))
    status = dict(stage='loading_models', started_at=time.time(), training_admission=False,
                  optimizer_updates=0, completed_cases=0, expected_cases=2 * len(selected))
    write_json(output / 'status.json', status)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(config['train']['seed'])
        torch.cuda.reset_peak_memory_stats()
        system, adapter = build_system(config, 'cuda')
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        if (state['step'] != plan['source_step'] or state['adapter_config'] != adapter.config
                or state['model_config'] != system.model_config or state['config']['models'] != config['models']):
            raise ValueError('Initialization does not match the exact configured model and source step')
        adapter.load_state_dict(state['adapter'], strict=True)
        statistics = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim']).cuda()
        statistics.load_state_dict(state['statistics'], strict=True)
        if any(not torch.isfinite(value).all() or not (value > 0).all()
               for value in statistics.buffers()):
            raise ValueError('Checkpoint feature scales are invalid')
        del state
        rows_path = output / 'cases.jsonl'
        with rows_path.open('x') as log:
            def on_case(row):
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                status.update(stage='observing_forward_backward', updated_at=time.time(),
                              completed_cases=status['completed_cases'] + 1)
                write_json(output / 'status.json', status)
                print(json.dumps(status), flush=True)
            result = observe_model(system, adapter, statistics,
                probe_batches(selected, replay_records, config), config, on_case)
        # Evidence and implementation must still have the original identities
        # after the potentially long model run, not only at its beginning.
        _, final_hashes = validate_pre_model_evidence(plan['accounting_report'], plan['checks'])
        if hashes != final_hashes:
            raise ValueError('Implementation changed during the model probe')
        for ref in [plan_ref, plan['training_config'], plan['initial_checkpoint'], replay_ref, *heldout_refs]:
            checked_path(ref)
        del result['rows']
        report = dict(result, gate='model_zero_update', stage='complete', passed=True,
            actual_model_forward_backward=True, training_admission=False,
            accounting_report_sha256=plan['accounting_report']['sha256'],
            index_sha256=accounting['index']['sha256'], manifest_sha256=accounting['manifest_sha256'],
            training_contract_sha256=training_contract_hash(config), implementation_sha256=hashes,
            initial_checkpoint=plan['initial_checkpoint'], initial_source_step=plan['source_step'],
            cases=status['completed_cases'], source_revisions=revisions,
            torch=str(torch.__version__), cuda=torch.version.cuda,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            evidence=[plan_ref, plan['training_config'], file_ref(selection_path), file_ref(rows_path),
                      file_ref(Path(__file__)), file_ref(Path(__file__).parents[1] /
                          'src/radio_moge/corpus_model_probe.py'), replay_ref, *heldout_refs])
        publish_json(output / 'report.json', report)
        status.update(stage='complete', completed_at=time.time())
        write_json(output / 'status.json', status)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-lock', required=True, type=Path)
    args = parser.parse_args()
    if not args.gpu_lock.is_absolute() or not args.gpu_lock.is_file():
        parser.error('Use the existing absolute GPU lock path; never create a replacement')
    with args.gpu_lock.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.plan, args.output)
