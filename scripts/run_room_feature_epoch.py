"""One complete, audited permutation of the frozen 50,787 room RGB images."""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.data import DeterministicBatches, check_split, read_manifest
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.rgb_input_audit import RGBInputAudit, load_reference_records
from radio_moge.rgb_replacement import ReplacementRecords, ReplacementBatches, validate_replacement
import radio_moge.train as trainer
from prepare_room_rgb_pool import check_rows
from run_feature_reference import observe_reference_model
from run_room_feature_mix import initial_state_check

IMAGES = 50787
UPDATES = (IMAGES + 3) // 4
SECONDS = 28800
SOURCES = dict(procthor_empty=40000, structural_rooms=10787)
TAIL_POLICY = 'remaining_unique_synthetic_then_original_slots'


class SingleEpochBatches(ReplacementBatches):
    """Retain original slots after the one synthetic permutation is exhausted.

    The final update can have fewer than four synthetic images. It still has
    32 images, with unchanged original keys filling the remaining slots. Any
    unused prefetch beyond this endpoint also stays original, never wrapping.
    """
    def __iter__(self):
        real = DeterministicBatches(self.records.real_size, self.batch_size, self.buckets,
            self.seed, self.spec['real_start_cursor'] + self.start, self.count)
        limit = self.records.synthetic_size * self.spec['every_images']
        for absolute, (mixed, original) in enumerate(zip(super().__iter__(), real, strict=True), self.start):
            yield [key if absolute*self.batch_size+offset < limit else old
                   for offset, (key, old) in enumerate(zip(mixed, original, strict=True))]


def real_cursor(parent):
    return parent['config']['data']['rgb_replacement']['real_start_cursor'] + parent['cursor']


def check_epoch_contract(config, parent, plan):
    validate_replacement(config)
    validate_replacement(parent['config'])
    normalized = deepcopy(config)
    normalized['data']['rgb_replacement'] = deepcopy(parent['config']['data']['rgb_replacement'])
    trainer.check_lr_fork(normalized, parent, parent['data_digests'])
    spec = config['data']['rgb_replacement']
    perf = parent['performance']
    if (plan.get('authorized') is not True or plan.get('coverage_goal') != 'every_frozen_synthetic_rgb_once'
            or plan.get('tail_policy') != TAIL_POLICY
            or plan['synthetic_images'] != IMAGES or plan['additional_updates'] != UPDATES
            or plan['maximum_training_seconds'] != SECONDS
            or plan['source_step'] != parent['step'] or plan['source_parent_cursor'] != parent['cursor']
            or plan['source_cursor'] != real_cursor(parent)
            or config['train']['steps'] != parent['step'] + UPDATES
            or config['train']['max_seconds'] != SECONDS
            or config['train']['eval_at_start'] is not False
            or config['train']['eval_every'] != 1000 or config['train']['save_every'] != 250
            or config['train']['lr'] != 1e-5 or parent['config']['train']['lr'] != 1e-5
            or spec['real_start_cursor'] != real_cursor(parent)
            or spec['manifest'] != plan['synthetic_manifest'] or spec['seed'] != 43
            or config['data']['buckets'] != [[480, 640], [640, 480], [400, 640]]
            or any(perf[k] != v for k, v in dict(effective_batch=32, compute_batch=32, workers=4).items())):
        raise ValueError('Only the declared complete room-RGB epoch is permitted')
    for name, field in [('procthor-empty', 'validation_manifest'),
                        ('structural-rooms', 'structural_validation_manifest')]:
        if config['evaluation']['dataset_variants'][name] != dict(
                manifest=plan[field]['path'], max_side=640, num_tokens=1200):
            raise ValueError('Validation population changed')


class Coverage:
    """Count actual committed image identities, excluding unused prefetched data."""
    def __init__(self, real_size, synthetic):
        self.real_size = real_size
        self.identities = [(r['image'], r['sha256'], r['dataset']) for r in synthetic]
        self.counts = [0] * len(synthetic)
        self.used = 0

    def validate(self, rows):
        indices = []
        for row in rows:
            if row['key'][0] < self.real_size:
                if row['dataset'] in SOURCES:
                    raise ValueError('Synthetic source presented in an original-data slot')
                continue
            index = row['key'][0] - self.real_size
            if not 0 <= index < len(self.counts):
                raise ValueError('Undeclared synthetic identity')
            if (row['image'], row['sha256'], row['dataset']) != self.identities[index]:
                raise ValueError('Actual synthetic identity differs from the frozen manifest')
            if self.counts[index] or index in indices:
                raise ValueError('A synthetic image repeated before the complete epoch ended')
            indices.append(index)
        expected = min(4, len(self.counts)-self.used)
        if expected <= 0 or len(indices) != expected:
            raise ValueError('Each update must contain the remaining unique synthetic slots, up to four')
        return indices

    def commit(self, indices):
        for index in indices:
            self.counts[index] += 1
        self.used += len(indices)

    def report(self):
        return dict(expected=len(self.counts), committed=sum(self.counts),
            unique=sum(n > 0 for n in self.counts), missing=sum(n == 0 for n in self.counts),
            repeated=sum(n > 1 for n in self.counts),
            each_image_exactly_once=all(n == 1 for n in self.counts))


class EpochAudit(RGBInputAudit):
    def __init__(self, root, plan, config, parent, progress):
        self.root, self.plan, self.config, self.parent, self.progress = root, plan, config, parent, progress
        super().__init__(config['output_dir'], parent['step'], 4, 8, 32)

    def load_records(self, path):
        if str(checked_path(self.plan['real_manifest'])) != path:
            raise ValueError('Original manifest changed')
        started = time.monotonic()
        previous = checked_json(self.plan['source_metadata'])
        if previous['mixed_digest'] != self.parent['data_digests']['train']:
            raise ValueError('Parent stream metadata does not bind this checkpoint')
        real = load_reference_records(path, self.progress, workers=4)
        if len(real) != 1597911 or real.digest() != previous['real_digest']:
            raise ValueError('Original RGB population or metadata changed')
        synthetic = load_reference_records(checked_path(self.plan['synthetic_manifest']), workers=4)
        check_rows(synthetic, 'train')
        if len(synthetic) != IMAGES or dict(Counter(r['dataset'] for r in synthetic)) != SOURCES:
            raise ValueError('Frozen epoch population changed')
        for field, expected in [('validation_manifest', 64), ('structural_validation_manifest', 317)]:
            heldout = read_manifest(checked_path(self.plan[field]))
            check_rows(heldout, 'val')
            if len(heldout) != expected:
                raise ValueError('Validation population changed')
            check_split(synthetic, heldout)
        check_split(real, synthetic)
        self.records = ReplacementRecords(real, synthetic)
        self.coverage = Coverage(len(real), synthetic)
        self.planned_keys = [list(k) for batch in SingleEpochBatches(self.records,
            self.config['data']['rgb_replacement'], 4, self.config['data']['buckets'],
            self.config['train']['seed'], 0, UPDATES*8) for k in batch]
        indices = [k[0]-len(real) for k in self.planned_keys if k[0] >= len(real)]
        if len(indices) != IMAGES or sorted(indices) != list(range(IMAGES)):
            raise ValueError('The exact plan does not cover each synthetic image once')
        publish_json(self.root/'planned-keys.json', self.planned_keys)
        self.planned = [dict(index=k[0], augmentation_seed=k[1], height=k[2], width=k[3])
                        for k in self.planned_keys]
        counts = Counter(self.records[k[0]]['dataset'] for k in self.planned_keys)
        publish_json(self.root/'metadata-report.json', dict(stage='full_epoch_rgb_plan_verified',
            real_images=len(real), synthetic_images=IMAGES, real_digest=real.digest(),
            synthetic_digest=synthetic.digest(), mixed_digest=self.records.digest(),
            planned_source_counts=dict(counts), planned_keys=file_ref(self.root/'planned-keys.json'),
            planned_synthetic_unique=IMAGES, planned_synthetic_repeats=0,
            real_start_cursor=real_cursor(self.parent), parent_mixed_cursor=self.parent['cursor'],
            real_unreplaced_keys_preserved=True, hashes_checked_on_each_consumed_rgb=True,
            full_batch_images=32, synthetic_images_per_update=4, geometry_loss=False,
            last_update_synthetic_images=IMAGES-4*(UPDATES-1),
            last_update_original_images=32-(IMAGES-4*(UPDATES-1)), tail_policy=TAIL_POLICY,
            seconds=time.monotonic()-started))
        return self.records

    def check_fork(self, config, state, digests):
        check_epoch_contract(config, state, self.plan)
        expected = dict(state['data_digests'], train=self.records.digest())
        if digests != expected:
            raise ValueError('Only the declared synthetic training population may change')

    def before_updates(self, system, adapter, statistics, records, config, output):
        self.progress(len(records), stage='checking_actual_epoch_model_before_updates')
        proof = observe_reference_model(system, adapter, statistics, records, config, self.planned,
            Path(self.plan['source_execution'])/'src/radio_moge/data.py',
            expected_sources={r['dataset'] for r in records},
            extra_probe_keys=[(i, 43+j, h, w) for h, w in config['data']['buckets']
                for j, i in enumerate([i for i, r in enumerate(records)
                                      if r.get('projection') == 'orthographic'][:4])]
                + self.planned_keys[-32:])
        publish_json(self.root/'mixed-model-zero-update.json', proof)
        baseline = checked_json(self.plan['source_results'])
        if baseline['step'] != self.parent['step'] or len(baseline['views']) != 6:
            raise ValueError('Saved six-view baseline has the wrong parent')
        for view in baseline['views'].values():
            for reference in view['files'].values():
                checked_path(reference)
        publish_json(self.root/'baseline-reuse.json', dict(stage='six_saved_parent_views_bound',
            source=self.plan['source_results'], step=self.parent['step'],
            views={k: v['files'] for k, v in baseline['views'].items()},
            model_inference=False, geometry_loss=False))
        self.progress(len(records), stage='training_full_synthetic_epoch')

    def commit(self, step, cursor):
        with self.lock:
            rows = self.pending_windows[0]['logical_images']
        indices = self.coverage.validate(rows)
        super().commit(step, cursor)
        self.coverage.commit(indices)
        if self.committed % 250 == 0 or self.committed == UPDATES:
            atomic_json(dict(step=step, updates=self.committed, **self.coverage.report()),
                        self.root/'coverage-progress.json')

    def finish(self, step, cursor):
        result = super().finish(step, cursor)
        coverage = self.coverage.report()
        if self.committed == UPDATES and not coverage['each_image_exactly_once']:
            raise ValueError('Full step budget did not give exact image coverage')
        publish_json(self.root/'coverage-counts.json', self.coverage.counts)
        publish_json(self.root/'coverage-final.json', dict(stage='committed_synthetic_coverage_verified',
            step=step, updates=self.committed, **coverage, counts=file_ref(self.root/'coverage-counts.json'),
            actual_inputs=file_ref(self.path), unused_prefetch_excluded=True,
            matched_encoded_image_identity=True, full_epoch_completed=self.committed == UPDATES
                and coverage['each_image_exactly_once']))
        return result


def run(plan_path):
    root = plan_path.parent
    plan = checked_json(file_ref(plan_path))
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'status.json').exists() or (root/'training').exists():
            raise FileExistsError('Epoch already launched; inspect the existing process')
        status = dict(stage='checking_full_epoch_configuration', started_at=time.time(), optimizer_updates=0)
        def progress(rows=0, stage='reading_verified_real_metadata'):
            status.update(stage=stage, metadata_rows=rows, updated_at=time.time())
            atomic_json(status, root/'status.json')
        progress(stage=status['stage'])
        try:
            config = checked_json(plan['config'])
            source = checked_path(plan['source_checkpoint'])
            parent = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
            check_epoch_contract(config, parent, plan)
            finish = checked_json(plan['source_finish'])
            if finish['checkpoint']['sha256'] != plan['source_checkpoint']['sha256'] or finish['step'] != parent['step']:
                raise ValueError('Parent checkpoint is not the verified final state')
            pool = checked_json(plan['synthetic_report'])
            if (pool['stage'] != 'frozen_room_full_epoch_pool_ready'
                    or pool['manifest'] != plan['synthetic_manifest'] or pool['images'] != IMAGES
                    or pool['source_counts'] != SOURCES or pool['geometry_loss'] or pool['geometry_labels']):
                raise ValueError('Missing qualified full-epoch pool proof')
            checked_path(plan['synthetic_manifest'])
            if config['output_dir'] != str(root/'training'):
                raise ValueError('Separate training output required')
            code, previous = Path(__file__).resolve().parents[1], Path(plan['source_execution'])
            if checked_path(plan['runner']) != Path(__file__).resolve():
                raise ValueError('The frozen single-epoch sampler/runner changed')
            names = ('train.py', 'data.py', 'losses.py', 'system.py', 'performance.py', 'prefetch.py',
                'evaluate.py', 'decoder.py', 'adapters.py', 'architecture_screen.py',
                'attention_tricks.py', 'feature_refinement.py', 'rgb_replacement.py', 'rgb_input_audit.py')
            hashes = {}
            for name in names:
                old, new = previous/'src/radio_moge'/name, code/'src/radio_moge'/name
                if sha_file(old) != sha_file(new):
                    raise ValueError('Unplanned training implementation change: '+name)
                hashes[name] = sha_file(new)
            publish_json(root/'source-module-identity.json', dict(hashes=hashes, unchanged=True))
            performance = deepcopy(parent['performance'])
            performance.update(anchor_step=parent['step'], anchor_cursor=0,
                anchor_schedule_position=float(parent['step']), original_started_at=time.time(),
                data_policy='Continue original real cursor; up to4/32 synthetic slots; all50787 once; final3+29',
                lr_policy='Preserve full parent Adam/RNG/RMS; unchanged constant1e-5')
            audit = EpochAudit(root, plan, config, parent, progress)
            save = trainer.atomic_save
            def checked_save(value, path):
                if Path(path).name == 'initial.pt':
                    equal = initial_state_check(value, parent, config, performance)
                    save(value, path)
                    publish_json(root/'initial-verification.json', dict(
                        stage='actual_parent_state_preserved_before_updates', equal_sections=list(equal),
                        source=plan['source_checkpoint'], step=value['step'], cursor=0,
                        parent_mixed_cursor=parent['cursor'], real_stream_start_cursor=real_cursor(parent),
                        mixed_cursor_reset_explicit=True, optimizer_states=len(value['optimizer']['state']),
                        optimizer_updates=0, data_digests=value['data_digests'], performance=performance))
                else:
                    save(value, path)
            with patch.object(trainer, 'atomic_save', checked_save), \
                    patch.object(trainer, 'ReplacementBatches', SingleEpochBatches):
                final = trainer.train(config, fork_from=source, data_fork=True, performance=performance,
                    reference=audit, execution_revision=plan['execution_revision'])
            stop = json.loads((root/'training/stop-report.json').read_text())
            coverage = checked_json(file_ref(root/'coverage-final.json'))
            complete = stop['step'] == parent['step'] + UPDATES and coverage['full_epoch_completed']
            publish_json(root/'completed.json', dict(stage='full_room_epoch_finished' if complete
                else 'incomplete_room_epoch_stopped', full_epoch_complete=complete,
                new_updates=stop['step']-parent['step'], final_checkpoint=file_ref(final),
                stop_report=stop, coverage=file_ref(root/'coverage-final.json'),
                source=plan['source_checkpoint'], automatic_next_segment=False, quality_review_pending=True))
            status.update(optimizer_updates=stop['step']-parent['step'], final_step=stop['step'])
            progress(stage='finished_review_pending' if complete else 'incomplete_inspect_no_retry')
        except Exception as error:
            status.update(error=type(error).__name__, message=str(error))
            progress(stage='failed_inspect_no_retry')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan.resolve())
