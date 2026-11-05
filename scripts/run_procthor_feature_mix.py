"""Bounded pure-feature training with frozen empty-room RGB and original real keys."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.data import check_split, read_manifest
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.rgb_input_audit import RGBInputAudit, load_reference_records
from radio_moge.rgb_replacement import ReplacementRecords, ReplacementBatches, validate_replacement
from radio_moge.tracking import preserve_rng
import radio_moge.train as trainer
from run_feature_reference import observe_reference_model
from verify_corpus_stage_startup import equal_finite


def check_mix_contract(config, parent, plan):
    validate_replacement(config)
    spec = config['data']['rgb_replacement']
    normalized = deepcopy(config)
    normalized['data'].pop('rgb_replacement')
    view = normalized['evaluation']['dataset_variants'].pop('procthor-empty')
    trainer.check_lr_fork(normalized, parent, parent['data_digests'])
    if (plan.get('authorized') is not True or plan['additional_updates'] != 1000
            or config['train']['steps'] != parent['step'] + 1000
            or config['train']['max_seconds'] != 3600
            or config['train']['eval_at_start'] is not False
            or config['train']['lr'] != parent['config']['train']['lr']
            or config['train']['lr'] != 1e-5
            or spec['real_start_cursor'] != parent['cursor']
            or spec['manifest'] != plan['synthetic_manifest'] or spec['seed'] != 43
            or view != dict(manifest=plan['validation_manifest']['path'], max_side=640, num_tokens=1200)
            or parent['performance']['effective_batch'] != 32
            or parent['performance']['compute_batch'] != 32
            or parent['performance']['workers'] != 4
            or config['data']['buckets'] != [[480, 640], [640, 480], [400, 640]]):
        raise ValueError('Only the declared bounded empty-room RGB addition is permitted')


def initial_state_check(value, parent, config, performance):
    def cpu(item):
        if torch.is_tensor(item):
            return item.detach().cpu()
        if isinstance(item, dict):
            return type(item)((k, cpu(v)) for k, v in item.items())
        if isinstance(item, (tuple, list)):
            return type(item)(cpu(v) for v in item)
        return item
    if (value['step'] != parent['step'] or value['cursor'] != 0
            or value['config'] != config or value['performance'] != performance):
        raise ValueError('Actual mixed-stream initial position/configuration changed')
    names = ('adapter', 'statistics', 'optimizer', 'rng', 'adapter_config', 'model_config', 'revisions')
    for name in names:
        if not equal_finite(cpu(value[name]), parent[name]):
            raise ValueError('Actual initial state changed: '+name)
    return names


class MixAudit(RGBInputAudit):
    def __init__(self, root, plan, config, parent, progress):
        self.root, self.plan, self.config, self.parent = root, plan, config, parent
        self.progress = progress
        super().__init__(config['output_dir'], parent['step'], 4, 8, 32)

    def load_records(self, path):
        if str(checked_path(self.plan['real_manifest'])) != path:
            raise ValueError('Real manifest changed')
        start = time.monotonic()
        real = load_reference_records(path, self.progress, workers=4)
        if len(real) != 1597911 or real.digest() != self.parent['data_digests']['train']:
            raise ValueError('Original real RGB population or metadata changed')
        synthetic = load_reference_records(checked_path(self.plan['synthetic_manifest']), workers=1)
        heldout = read_manifest(checked_path(self.plan['validation_manifest']))
        for rows, split in ((synthetic, 'train'), (heldout, 'val')):
            if (not rows or any(r.get('dataset') != 'procthor_empty' or r.get('split') != split
                    or r.get('projection') != 'perspective' or r.get('agent_visible') is not False
                    or r.get('geometry_label') is not None or r.get('label_role') != 'feature_only_rgb'
                    or not r.get('group', '').startswith(f'procthor10k/{split}/') for r in rows)):
                raise ValueError('Synthetic projection, house split or feature-only contract changed')
        check_split(synthetic, heldout)
        if ({r['decoded_rgb_sha256'] for r in synthetic} & {r['decoded_rgb_sha256'] for r in heldout}
                or {r['original_house_sha256'] for r in synthetic} & {r['original_house_sha256'] for r in heldout}):
            raise ValueError('Rendered train/validation content or house duplicates')
        check_split(real, synthetic)
        records = ReplacementRecords(real, synthetic)
        self.records = records
        self.planned_keys = [list(key) for batch in ReplacementBatches(records,
            self.config['data']['rgb_replacement'], 4, self.config['data']['buckets'],
            self.config['train']['seed'], 0, self.plan['additional_updates'] * 8) for key in batch]
        publish_json(self.root / 'planned-keys.json', self.planned_keys)
        self.planned = [dict(index=k[0], augmentation_seed=k[1], height=k[2], width=k[3])
                        for k in self.planned_keys]
        counts = {}
        for key in self.planned_keys:
            source = records[key[0]]['dataset']
            counts[source] = counts.get(source, 0) + 1
        if counts['procthor_empty'] != self.plan['additional_updates'] * 4:
            raise ValueError('Synthetic share differs from four images per full32 update')
        publish_json(self.root / 'metadata-report.json', dict(stage='frozen_rgb_mix_verified',
            real_images=len(real), synthetic_images=len(synthetic), heldout_images=len(heldout),
            real_digest=real.digest(), synthetic_digest=synthetic.digest(), mixed_digest=records.digest(),
            seconds=time.monotonic()-start, planned_source_counts=counts,
            planned_keys=file_ref(self.root / 'planned-keys.json'),
            real_start_cursor=self.parent['cursor'], real_unreplaced_keys_preserved=True,
            geometry_loss=False, full_batch_images=32, synthetic_images_per_update=4,
            hashes_checked_on_each_consumed_rgb=True))
        return records

    def check_fork(self, config, state, digests):
        check_mix_contract(config, state, self.plan)
        expected = dict(state['data_digests'])
        expected['train'] = self.records.digest()
        # The fixed new validation set adds a distinct metric namespace only.
        expected['dataset/procthor-empty'] = trainer.manifest_digest(
            read_manifest(self.plan['validation_manifest']['path']))
        if digests != expected:
            raise ValueError('Existing heldouts or declared new data identities changed')

    def before_updates(self, system, adapter, statistics, records, config, output):
        self.progress(len(records), stage='checking_new_mixed_inputs_before_updates')
        proof = observe_reference_model(system, adapter, statistics, records, config, self.planned,
            Path(self.plan['source_execution']) / 'src/radio_moge/data.py',
            expected_sources={r['dataset'] for r in records})
        publish_json(self.root / 'mixed-model-zero-update.json', proof)
        heldout = read_manifest(self.plan['validation_manifest']['path'])
        variant = deepcopy(config)
        variant['evaluation'].update(max_side=640, num_tokens=1200)
        with preserve_rng():
            metrics = trainer.evaluate(system, adapter, statistics, heldout, variant,
                self.root / 'procthor-baseline', None)
        publish_json(self.root / 'procthor-baseline.json', dict(
            stage='new_house_heldout_baseline_complete', source_step=self.parent['step'],
            validation=self.plan['validation_manifest'], metrics=metrics,
            previous_real_baselines_reused=True, geometry_loss=False))
        self.progress(len(records), stage='training')


def run(plan_path):
    root = plan_path.parent
    plan = checked_json(file_ref(plan_path))
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'status.json').exists() or (root / 'training').exists():
            raise FileExistsError('This bounded mixed training was already launched')
        status = dict(stage='checking_mixed_feature_configuration', started_at=time.time(), optimizer_updates=0)
        def progress(rows=0, stage='reading_verified_real_metadata'):
            status.update(stage=stage, metadata_rows=rows, updated_at=time.time())
            atomic_json(status, root / 'status.json')
        progress(stage=status['stage'])
        try:
            config = checked_json(plan['config'])
            source = checked_path(plan['source_checkpoint'])
            parent = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
            check_mix_contract(config, parent, plan)
            for role, split, count, houses in (('synthetic', 'train', 2000, 100), ('validation', 'val', 64, 16)):
                frozen = checked_json(plan[role+'_report'])
                if (frozen.get('stage') != 'frozen_verified_feature_rgb'
                        or frozen['manifest'] != plan[role+'_manifest']
                        or frozen['split'] != split or frozen['images'] != count or frozen['houses'] != houses
                        or frozen['training_eligible'] != (split == 'train')
                        or frozen['agent_visible'] is not False or not frozen['perspective_only']
                        or not frozen['every_selected_rgb_decoded_and_hashed']):
                    raise ValueError('Frozen RGB receipt differs from the declared first batch')
            visual = checked_json(plan['sample_visual_review'])
            if (visual.get('stage') != 'actual_rgb_samples_visually_reviewed'
                    or visual['train_manifest'] != plan['synthetic_manifest']
                    or visual['val_manifest'] != plan['validation_manifest']):
                raise ValueError('Missing visual review for these frozen RGB batches')
            if config['output_dir'] != str(root / 'training'):
                raise ValueError('Separate training output required')
            # The sampler changes; model, loss, ordinary pixels and prepared math do not.
            code = Path(__file__).resolve().parents[1]
            previous = Path(plan['source_execution'])
            for name in ('data.py', 'losses.py', 'system.py', 'performance.py', 'prefetch.py',
                         'evaluate.py', 'decoder.py', 'adapters.py', 'architecture_screen.py',
                         'attention_tricks.py', 'feature_refinement.py'):
                old = previous / 'src/radio_moge' / name
                new = code / 'src/radio_moge' / name
                if old.exists() and (not new.exists() or sha_file(old) != sha_file(new)):
                    raise ValueError('Unplanned feature implementation change: '+name)
            performance = deepcopy(parent['performance'])
            performance.update(anchor_step=parent['step'], anchor_cursor=0,
                anchor_schedule_position=float(parent['step']), original_started_at=time.time(),
                data_policy='Continue original real keys; replace exactly4/32 slots by frozen ProcTHOR RGB',
                lr_policy='Preserve full parent Adam/RNG/RMS; unchanged constant1e-5')
            audit = MixAudit(root, plan, config, parent, progress)
            save = trainer.atomic_save
            def checked_save(value, path):
                if Path(path).name == 'initial.pt':
                    names = initial_state_check(value, parent, config, performance)
                    save(value, path)
                    publish_json(root / 'initial-verification.json', dict(
                        stage='actual_parent_state_preserved_before_updates', equal_sections=list(names),
                        source=plan['source_checkpoint'], step=value['step'], cursor=0,
                        real_stream_start_cursor=parent['cursor'],
                        mixed_cursor_reset_explicit=True, optimizer_states=len(value['optimizer']['state']),
                        optimizer_updates=0, data_digests=value['data_digests'], performance=performance,
                        scheduler='Constant LR wrapper reset; Adam moments and update counters preserved'))
                else:
                    save(value, path)
            with patch.object(trainer, 'atomic_save', checked_save):
                final = trainer.train(config, fork_from=source, data_fork=True,
                    performance=performance, reference=audit, execution_revision=plan['execution_revision'])
            stop = json.loads((root / 'training/stop-report.json').read_text())
            updates = stop['step'] - parent['step']
            publish_json(root / 'completed.json', dict(stage='bounded_rgb_mix_finished',
                new_updates=updates, fixed1000_complete=updates == 1000,
                final_checkpoint=file_ref(final), stop_report=stop, source=plan['source_checkpoint'],
                actual_inputs=file_ref(root / 'training/actual-rgb-input-audit.json'),
                automatic_next_segment=False, quality_review_pending=True))
            status.update(optimizer_updates=updates, final_step=stop['step'])
            progress(stage='finished_review_pending')
        except Exception as error:
            status.update(error=type(error).__name__, message=str(error))
            progress(stage='failed_inspect_no_retry')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan.resolve())
