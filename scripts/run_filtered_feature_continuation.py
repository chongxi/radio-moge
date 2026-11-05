"""Bounded pure-feature continuation; reuse the completed filtered RGB pipeline."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.rgb_input_audit import load_reference_records
import radio_moge.train as trainer
from verify_corpus_stage_startup import equal_finite


def validate_config(config, parent, plan):
    trainer.check_lr_fork(config, parent, parent['data_digests'])
    if (plan.get('authorized') is not True or parent['step'] != 232538
            or parent['cursor'] != 2000 or plan['additional_updates'] != 1000
            or config['train']['steps'] != 233538 or config['train']['max_seconds'] != 3600
            or config['loss'] != dict(spatial=1., summary=1., cosine=1., summary_cosine=1.)
            or config['data']['buckets'] != [[480, 640], [640, 480], [400, 640]]
            or config['data'].get('mixture_weights') or config['data'].get('coverage_epochs')
            or any(k in config for k in ('geometry_supervision', 'ordinary_geometry', 'feature_replay'))
            or parent['performance']['effective_batch'] != 32
            or parent['performance']['workers'] != 4):
        raise ValueError('Only authorized filtered pure-feature continuation is supported')
    if config['train']['lr'] != 1e-5 or parent['config']['train']['lr'] != 1e-5:
        raise ValueError('Preserve constant learning rate')


def verified_loader(original, manifest, digest, on_progress, on_complete):
    def read(path):
        if Path(path).resolve() != manifest.resolve():
            return original(path)
        started = time.monotonic()
        records = load_reference_records(path, workers=4, on_progress=on_progress)
        if len(records) != 1597911 or records.digest() != digest:
            raise ValueError('Filtered population or original metadata digest changed')
        on_complete(dict(rows=len(records), seconds=time.monotonic()-started,
                         digest=records.digest(), workers=4, verify_on_read=True))
        return records
    return read


def initial_checks(value, parent):
    def cpu(item):
        if torch.is_tensor(item):
            return item.detach().cpu()
        if isinstance(item, dict):
            return type(item)((k, cpu(v)) for k, v in item.items())
        if isinstance(item, (tuple, list)):
            return type(item)(cpu(v) for v in item)
        return item
    if (value['step'], value['cursor']) != (parent['step'], parent['cursor']):
        raise ValueError('Continuation reset step or cursor')
    names = ('adapter', 'statistics', 'optimizer', 'rng', 'adapter_config',
             'model_config', 'revisions', 'data_digests', 'performance')
    for name in names:
        if not equal_finite(cpu(value[name]), parent[name]):
            raise ValueError('Actual initial state changed: '+name)
    return names


def run(plan_path):
    plan = checked_json(file_ref(plan_path))
    root = plan_path.parent
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'status.json').exists() or (root/'training').exists():
            raise FileExistsError('Continuation already launched; no automatic retry')
        status = dict(stage='checking_source_and_configuration', started_at=time.time(),
                      optimizer_updates=0, automatic_retry=False)
        def record(**changes):
            status.update(changes, updated_at=time.time())
            atomic_json(status, root/'status.json')
        record()
        try:
            config = checked_json(plan['config'])
            source_path = checked_path(plan['source_checkpoint'])
            parent = torch.load(source_path, map_location='cpu', weights_only=True, mmap=True)
            validate_config(config, parent, plan)
            code = Path(__file__).resolve().parents[1]
            old = Path(plan['source_execution'])
            for p in sorted((code/'src').rglob('*.py')):
                if sha_file(p) != sha_file(old/p.relative_to(code)):
                    raise ValueError('Previously exercised trainer implementation changed')
            checked_path(plan['manifest'])
            if str(checked_path(plan['manifest'])) != config['data']['train_manifest']:
                raise ValueError('Unbound training manifest')
            if config['output_dir'] != str(root/'training'):
                raise ValueError('Isolated continuation output required')
            last = [0.]
            def progress(rows):
                if time.monotonic()-last[0] >= 5:
                    record(stage='loading_filtered_metadata', metadata_rows=rows)
                    last[0] = time.monotonic()
            def completed(report):
                publish_json(root/'metadata-report.json', report)
                record(stage='loading_model_and_initial_state', metadata_rows=report['rows'])
            reader = verified_loader(trainer.read_manifest, Path(config['data']['train_manifest']),
                                     parent['data_digests']['train'], progress, completed)
            save = trainer.atomic_save
            def checked_save(value, path):
                if Path(path).name == 'initial.pt':
                    names = initial_checks(value, parent)
                    save(value, path)
                    publish_json(root/'initial-verification.json', dict(
                        stage='actual_initial_state_verified_before_updates', source=plan['source_checkpoint'],
                        step=value['step'], cursor=value['cursor'], equal_sections=list(names),
                        optimizer_states=len(value['optimizer']['state']), optimizer_updates=0,
                        scheduler='New constant wrapper bookkeeping; same constant LR, Adam untouched',
                        config=plan['config'], execution_revision=plan['execution_revision']))
                    record(stage='training', source_step=value['step'], source_cursor=value['cursor'])
                else:
                    save(value, path)
            with patch.object(trainer, 'read_manifest', reader), patch.object(trainer, 'atomic_save', checked_save):
                final = trainer.train(config, fork_from=source_path,
                    execution_revision=plan['execution_revision'])
            stop = json.loads((root/'training/stop-report.json').read_text())
            publish_json(root/'completed.json', dict(stage='bounded_feature_continuation_finished',
                source=plan['source_checkpoint'], final_checkpoint=file_ref(final), stop_report=stop,
                new_updates=stop['step']-parent['step'], fixed1000_complete=stop['step']==233538,
                deployment_selection_pending=True))
            record(stage='finished_review_pending', final_step=stop['step'],
                   optimizer_updates=stop['step']-parent['step'])
        except Exception as error:
            record(stage='failed_inspect_no_retry', error=type(error).__name__, message=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan.resolve())
