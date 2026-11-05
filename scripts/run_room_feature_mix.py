"""Bounded feature-only training on the qualified 40,890-image room pool."""
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
from prepare_room_rgb_pool import check_rows
from run_procthor_feature_mix import check_mix_contract as check_original_mix_contract
from verify_corpus_stage_startup import equal_finite


def check_mix_contract(config, parent, plan):
    normalized = deepcopy(config)
    view = normalized['evaluation']['dataset_variants'].pop('structural-rooms')
    if view != dict(manifest=plan['structural_validation_manifest']['path'], max_side=640, num_tokens=1200):
        raise ValueError('Structural validation must retain the fixed 317-image population')
    check_original_mix_contract(normalized, parent, plan)


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
        check_rows(synthetic, 'train')
        check_rows(heldout, 'val')
        if len(synthetic) != 40890 or len(heldout) != 64:
            raise ValueError('Wrong fixed room pool')
        check_split(synthetic, heldout)
        structural_val = read_manifest(checked_path(self.plan['structural_validation_manifest']))
        check_rows(structural_val, 'val')
        check_split(synthetic, structural_val)
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
        if (sum(counts.get(k, 0) for k in ('procthor_empty', 'structural_rooms')) != self.plan['additional_updates'] * 4
                or counts.get('structural_rooms', 0) < 1):
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
        expected['dataset/structural-rooms'] = trainer.manifest_digest(
            read_manifest(self.plan['structural_validation_manifest']['path']))
        if digests != expected:
            raise ValueError('Existing heldouts or declared new data identities changed')

    def before_updates(self, system, adapter, statistics, records, config, output):
        self.progress(len(records), stage='checking_new_mixed_inputs_before_updates')
        proof = observe_reference_model(system, adapter, statistics, records, config, self.planned,
            Path(self.plan['source_execution']) / 'src/radio_moge/data.py',
            expected_sources={r['dataset'] for r in records},
            extra_probe_keys=[(index, 43+j, h, w)
                for h, w in config['data']['buckets']
                for j, index in enumerate([i for i, r in enumerate(records)
                    if r.get('projection') == 'orthographic'][:4])])
        publish_json(self.root / 'mixed-model-zero-update.json', proof)
        # Identical parent and official holdout: bind the completed saved baseline.
        previous = checked_json(self.plan['procthor_baseline'])
        if (previous['source_step'] != self.parent['step']
                or previous['validation'] != self.plan['validation_manifest']):
            raise ValueError('Saved official-house baseline has a different parent or population')
        for ref in self.plan['procthor_baseline_files'].values():
            checked_path(ref)
        heldout = read_manifest(self.plan['structural_validation_manifest']['path'])
        variant = deepcopy(config)
        variant['evaluation'].update(max_side=640, num_tokens=1200)
        with preserve_rng():
            metrics = trainer.evaluate(system, adapter, statistics, heldout, variant,
                self.root / 'structural-baseline', None)
        publish_json(self.root / 'structural-baseline.json', dict(
            stage='new_structural_heldout_baseline_complete', source_step=self.parent['step'],
            validation=self.plan['structural_validation_manifest'], metrics=metrics,
            original_and_official_house_baselines_reused=True, geometry_loss=False))
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
            pool = checked_json(plan['synthetic_report'])
            if (pool['stage'] != 'frozen_room_feature_pool_ready'
                    or pool['manifest'] != plan['synthetic_manifest'] or pool['images'] != 40890
                    or pool['source_counts'] != dict(procthor_empty=40000, structural_rooms=890)
                    or pool['projections'] != dict(perspective=40800, orthographic=90)
                    or pool['geometry_loss'] or pool['geometry_labels']):
                raise ValueError('Missing qualified 40,890-image pool proof')
            for reference in pool['request'].values():
                checked_path(reference)
            checked_path(plan['synthetic_manifest'])
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
                data_policy='Continue original real keys; replace exactly4/32 slots by frozen 40,890 room RGB',
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
