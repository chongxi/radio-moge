"""One frozen synthetic-image pass, with one actual 64/128-image model call per update."""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
from functools import partial
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import torch

from radio_moge.checkpoint import atomic_json, rng_state, restore_rng
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.corpus_model_probe import assert_frozen, gradient_summary, module_digest
from radio_moge.data import TrainingImages, check_split, read_manifest
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.losses import weighted_loss
from radio_moge.prefetch import normalize_uint8
from radio_moge.rgb_input_audit import RGBInputAudit, pixels_sha, load_reference_records
from radio_moge.rgb_replacement import ReplacementRecords, validate_replacement
import radio_moge.train as trainer
from prepare_room_rgb_pool import check_rows
from run_room_feature_epoch import SingleEpochBatches, real_cursor, IMAGES, SOURCES
from run_room_feature_mix import initial_state_check

SECONDS = 14400
BUCKET_POLICY = 'uniform_one_bucket_per_actual_forward_preserve_identity_and_crop_seed'


def updates_for(batch):
    if type(batch) is not int or batch not in (64, 128):
        raise ValueError('An actual batch of 64 or 128 is required')
    return (IMAGES + batch//8 - 1) // (batch//8)


class TrueBatchSampler(SingleEpochBatches):
    """Keep image identities/seed stream, assigning one bucket per complete GPU batch.

    Logical groups of four exist only for bounded shared-memory transport. They
    are concatenated before the only forward/backward call for the update.
    """
    def __init__(self, *args, actual_batch, **kwargs):
        super().__init__(*args, **kwargs)
        updates_for(actual_batch)
        self.actual_batch = actual_batch

    def __iter__(self):
        width = self.actual_batch//4
        for absolute, batch in enumerate(super().__iter__(), self.start):
            first = self.spec['real_start_cursor'] + (absolute//width)*width
            height, width_px = random.Random(self.seed+first*104729).choice(self.buckets)
            yield [(i, seed, height, width_px) for i, seed, _, _ in batch]


class BatchCoverage:
    def __init__(self, real_size, synthetic, batch):
        self.real_size, self.batch = real_size, batch
        self.identities = [(r['image'], r['sha256'], r['dataset']) for r in synthetic]
        self.counts = [0]*len(synthetic)
        self.used = 0

    def validate(self, rows):
        if len(rows) != self.batch or len({tuple(r['key'][2:]) for r in rows}) != 1:
            raise ValueError('Each full update must have one actual same-resolution batch')
        indices = []
        for row in rows:
            i = row['key'][0]-self.real_size
            if i < 0:
                if row['dataset'] in SOURCES:
                    raise ValueError('Synthetic image in an original slot')
                continue
            if (not 0 <= i < len(self.counts) or self.counts[i] or i in indices
                    or (row['image'], row['sha256'], row['dataset']) != self.identities[i]):
                raise ValueError('Unexpected or repeated synthetic identity')
            indices.append(i)
        if len(indices) != min(self.batch//8, len(self.counts)-self.used) or not indices:
            raise ValueError('Wrong synthetic share or exhausted epoch')
        return indices

    def commit(self, indices):
        for i in indices:
            self.counts[i] += 1
        self.used += len(indices)

    def report(self):
        return dict(expected=len(self.counts), committed=sum(self.counts),
            unique=sum(n>0 for n in self.counts), missing=sum(n==0 for n in self.counts),
            repeated=sum(n>1 for n in self.counts), each_image_exactly_once=all(n==1 for n in self.counts))


def check_contract(config, parent, plan):
    validate_replacement(config)
    batch = plan['actual_batch']
    updates = updates_for(batch)
    normalized = deepcopy(config)
    normalized['data']['rgb_replacement'] = deepcopy(parent['config']['data']['rgb_replacement'])
    trainer.check_lr_fork(normalized, parent, parent['data_digests'])
    if (plan.get('authorized') is not True or plan['coverage_goal'] != 'every_frozen_synthetic_rgb_once'
            or plan['bucket_policy'] != BUCKET_POLICY or plan['synthetic_images'] != IMAGES
            or plan['additional_updates'] != updates or plan['source_step'] != parent['step']
            or plan['source_parent_cursor'] != parent['cursor']
            or plan['source_cursor'] != real_cursor(parent)
            or plan['maximum_training_seconds'] != SECONDS
            or config['train']['steps'] != parent['step']+updates
            or config['train']['max_seconds'] != SECONDS or config['train']['eval_at_start'] is not False
            or config['train']['eval_every'] != 32000//batch or config['train']['save_every'] != 128
            or config['train']['lr'] != parent['config']['train']['lr'] or config['train']['lr'] != 1e-5
            or config['train']['batch_size'] != 4 or config['train']['precision'] != 'bf16'
            or config['data']['buckets'] != [[480,640],[640,480],[400,640]]
            or config['data']['rgb_replacement']['real_start_cursor'] != real_cursor(parent)
            or config['data']['rgb_replacement']['manifest'] != plan['synthetic_manifest']
            or config['data']['rgb_replacement']['manifest'] != parent['config']['data']['rgb_replacement']['manifest']
            or config['data']['rgb_replacement']['seed'] != 43):
        raise ValueError('Unexpected true-batch epoch/configuration change')
    for name, field in [('procthor-empty','validation_manifest'), ('structural-rooms','structural_validation_manifest')]:
        if config['evaluation']['dataset_variants'][name] != dict(
                manifest=plan[field]['path'], max_side=640, num_tokens=1200):
            raise ValueError('Validation population changed')


def check_performance(config, parent, plan, performance):
    check_contract(config, parent, plan)
    batch = plan['actual_batch']
    expected = dict(authorized=True, effective_batch=batch, compute_batch=batch, workers=4,
        data_pipeline='prepared', compact_transport=True, gpu_prefetch=True, anchor_cursor=0,
        anchor_step=parent['step'], anchor_schedule_position=float(parent['step']))
    if (any(performance.get(k) != v for k,v in expected.items())
            or performance.get('autotune') or performance.get('data_autotune')):
        raise ValueError('True-batch fork must use one prepared/prefetched full forward')


class TrueBatchAudit(RGBInputAudit):
    def __init__(self, root, plan, config, parent, progress):
        self.root, self.plan, self.config, self.parent, self.progress = root, plan, config, parent, progress
        self.batch = plan['actual_batch']
        self.updates = updates_for(self.batch)
        super().__init__(config['output_dir'], parent['step'], 4, self.batch//4, self.batch)

    def check_batch_transition(self, config, parent, performance):
        check_performance(config, parent, self.plan, performance)

    def check_fork(self, config, state, digests):
        check_contract(config, state, self.plan)
        if digests != state['data_digests']:
            raise ValueError('Batch change must preserve every training and heldout identity')

    def load_records(self, path):
        if str(checked_path(self.plan['real_manifest'])) != path:
            raise ValueError('Original manifest changed')
        previous = checked_json(self.plan['source_metadata'])
        if previous['mixed_digest'] != self.parent['data_digests']['train']:
            raise ValueError('Parent stream metadata mismatch')
        real = load_reference_records(path, self.progress, workers=4)
        synthetic = load_reference_records(checked_path(self.plan['synthetic_manifest']), workers=4)
        check_rows(synthetic, 'train')
        if (len(real) != 1597911 or real.digest() != previous['real_digest'] or len(synthetic) != IMAGES
                or dict(Counter(r['dataset'] for r in synthetic)) != SOURCES):
            raise ValueError('Frozen population or metadata changed')
        for field, count in [('validation_manifest',64),('structural_validation_manifest',317)]:
            heldout = read_manifest(checked_path(self.plan[field]))
            check_rows(heldout, 'val')
            if len(heldout) != count:
                raise ValueError('Validation population changed')
            check_split(synthetic, heldout)
        check_split(real, synthetic)
        self.records = ReplacementRecords(real, synthetic)
        if self.records.digest() != self.parent['data_digests']['train']:
            raise ValueError('Batch-only fork changed the dataset digest')
        self.coverage = BatchCoverage(len(real), synthetic, self.batch)
        self.planned_keys = [list(k) for b in TrueBatchSampler(self.records,
            self.config['data']['rgb_replacement'], 4, self.config['data']['buckets'],
            self.config['train']['seed'], 0, self.updates*self.batch//4,
            actual_batch=self.batch) for k in b]
        indices = [k[0]-len(real) for k in self.planned_keys if k[0] >= len(real)]
        if len(indices) != IMAGES or sorted(indices) != list(range(IMAGES)):
            raise ValueError('Synthetic epoch omitted/repeated an identity')
        publish_json(self.root/'planned-keys.json', self.planned_keys)
        counts = Counter(self.records[k[0]]['dataset'] for k in self.planned_keys)
        publish_json(self.root/'metadata-report.json', dict(stage='true_batch_epoch_plan_verified',
            real_images=len(real), synthetic_images=IMAGES, real_digest=real.digest(),
            synthetic_digest=synthetic.digest(), mixed_digest=self.records.digest(),
            planned_source_counts=dict(counts), planned_keys=file_ref(self.root/'planned-keys.json'),
            planned_synthetic_unique=IMAGES, planned_synthetic_repeats=0,
            real_start_cursor=real_cursor(self.parent), parent_mixed_cursor=self.parent['cursor'],
            image_identities_and_crop_seeds_preserved=True, resolution_assignment_changed=True,
            bucket_policy=BUCKET_POLICY, full_batch_images=self.batch,
            synthetic_images_per_update=self.batch//8,
            last_update_synthetic_images=IMAGES%(self.batch//8), geometry_loss=False))
        return self.records

    def before_updates(self, system, adapter, statistics, records, config, output):
        self.progress(len(records), stage='checking_actual_full_batch_model_before_updates')
        modules = dict(radio=system.radio, encoder=system.encoder, decoder=system.decoder,
                       adapter=adapter, statistics=statistics)
        before = {k:module_digest(v) for k,v in modules.items()}
        rng = rng_state()
        data = TrainingImages(records, config['data']['augment'], compact_transport=True)
        rows = []
        try:
            # Actual first and final full training batches, including the three-image synthetic tail.
            for label, keys in [('first',self.planned_keys[:self.batch]), ('tail',self.planned_keys[-self.batch:])]:
                adapter.zero_grad(set_to_none=True)
                compact = torch.stack([data[k] for k in keys])
                image = normalize_uint8(compact).to(next(adapter.parameters()).device)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    features, target = system.encode(image, adapter.indices)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    predicted = adapter(features)
                losses = statistics.loss(predicted, target)
                loss = weighted_loss(losses, config['loss'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite actual full-batch feature loss')
                loss.backward()
                gradients = gradient_summary(adapter.parameters())
                assert_frozen(system)
                rows.append(dict(label=label, shape=list(image.shape), pixel_sha256=pixels_sha(compact),
                    keys=keys, total=float(loss.detach()), gradient=gradients))
                del image, compact, features, target, predicted, losses, loss
            after = {k:module_digest(v) for k,v in modules.items()}
            if before != after:
                raise ValueError('Zero-update true batch gate modified model state')
        finally:
            adapter.zero_grad(set_to_none=True)
            restore_rng(rng)
        publish_json(self.root/'mixed-model-zero-update.json', dict(stage='actual_full_batch_model_passed',
            rows=rows, actual_batch=self.batch, cases=len(rows), images=self.batch*len(rows),
            state_before=before, state_after=after, optimizer_updates=0, gradients_discarded=True,
            rng_restored_bitwise=True, geometry_branch_executed=False,
            prior_all_three_resolution_probe=self.plan['batch_probe']))
        baseline = checked_json(self.plan['source_results'])
        if baseline['step'] != self.parent['step'] or len(baseline['views']) != 6:
            raise ValueError('Wrong saved six-view baseline')
        for view in baseline['views'].values():
            for ref in view['files'].values():
                checked_path(ref)
        publish_json(self.root/'baseline-reuse.json', dict(stage='six_saved_parent_views_bound',
            source=self.plan['source_results'], step=self.parent['step'],
            views={k:v['files'] for k,v in baseline['views'].items()}, model_inference=False))
        self.progress(len(records), stage='training_true_batch_epoch')

    def observe_window(self, tensors):
        if len(tensors) != 1 or len(tensors[0]) != self.batch:
            raise ValueError('Prepared window was split into multiple forward batches')
        super().observe_window(tensors)

    def commit(self, step, cursor):
        with self.lock:
            rows = self.pending_windows[0]['logical_images']
        indices = self.coverage.validate(rows)
        super().commit(step, cursor)
        self.coverage.commit(indices)
        if self.committed % 100 == 0 or self.committed == self.updates:
            atomic_json(dict(step=step, updates=self.committed, **self.coverage.report()),
                        self.root/'coverage-progress.json')

    def finish(self, step, cursor):
        result = super().finish(step, cursor)
        coverage = self.coverage.report()
        if self.committed == self.updates and not coverage['each_image_exactly_once']:
            raise ValueError('Incomplete synthetic coverage')
        publish_json(self.root/'coverage-counts.json', self.coverage.counts)
        publish_json(self.root/'coverage-final.json', dict(stage='committed_synthetic_coverage_verified',
            step=step, updates=self.committed, **coverage, counts=file_ref(self.root/'coverage-counts.json'),
            actual_inputs=file_ref(self.path), unused_prefetch_excluded=True,
            matched_encoded_image_identity=True, full_epoch_completed=self.committed == self.updates
                and coverage['each_image_exactly_once']))
        return result


def run(plan_path):
    root = plan_path.parent
    plan = checked_json(file_ref(plan_path))
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'status.json').exists() or (root/'training').exists():
            raise FileExistsError('Already launched; inspect existing process')
        status = dict(stage='checking_true_batch_configuration', started_at=time.time(), optimizer_updates=0)
        def progress(rows=0, stage='reading_verified_real_metadata'):
            status.update(stage=stage, metadata_rows=rows, updated_at=time.time())
            atomic_json(status, root/'status.json')
        progress(stage=status['stage'])
        try:
            config = checked_json(plan['config'])
            source = checked_path(plan['source_checkpoint'])
            parent = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
            check_contract(config, parent, plan)
            finish = checked_json(plan['source_finish'])
            if finish['checkpoint']['sha256'] != plan['source_checkpoint']['sha256'] or finish['step'] != parent['step']:
                raise ValueError('Parent is not the verified final checkpoint')
            probe = checked_json(plan['batch_probe'])
            probe_plan = checked_json(probe['plan'])
            if (probe['stage'] != 'complete' or probe['selected_actual_batch'] != plan['actual_batch']
                    or probe['model_states_unchanged'] is not True
                    or probe_plan['checkpoint'] != plan['source_checkpoint']):
                raise ValueError('Missing current-checkpoint actual-batch memory/throughput proof')
            if config['output_dir'] != str(root/'training'):
                raise ValueError('A separate training destination is required')
            code, prior = Path(__file__).resolve().parents[1], Path(plan['source_execution'])
            if checked_path(plan['runner']) != Path(__file__).resolve():
                raise ValueError('Frozen batch runner changed')
            hashes = {}
            for name in ('data.py','losses.py','system.py','performance.py','prefetch.py','evaluate.py',
                         'decoder.py','adapters.py','architecture_screen.py','attention_tricks.py',
                         'feature_refinement.py','rgb_replacement.py','rgb_input_audit.py'):
                if sha_file(code/'src/radio_moge'/name) != sha_file(prior/'src/radio_moge'/name):
                    raise ValueError('Unexpected core change: '+name)
                hashes[name] = sha_file(code/'src/radio_moge'/name)
            checked_path(plan['trainer'])
            if Path(plan['trainer']['path']) != code/'src/radio_moge/train.py':
                raise ValueError('The batch-fork trainer implementation differs')
            publish_json(root/'source-module-identity.json', dict(hashes=hashes,
                unchanged_except_explicit_batch_fork=True, trainer=plan['trainer']))
            performance = deepcopy(parent['performance'])
            batch = plan['actual_batch']
            performance.update(effective_batch=batch, compute_batch=batch, workers=4, gpu_prefetch=True,
                anchor_step=parent['step'], anchor_cursor=0, anchor_schedule_position=float(parent['step']),
                original_started_at=time.time(), schedule_ratio=batch/16,
                data_policy='Full original identities/crop seeds; one bucket per actual forward; all50787 once',
                lr_policy='Preserve full Adam/RNG/RMS; unchanged constant1e-5')
            check_performance(config, parent, plan, performance)
            audit = TrueBatchAudit(root, plan, config, parent, progress)
            save = trainer.atomic_save
            def checked_save(value, path):
                if Path(path).name == 'initial.pt':
                    equal = initial_state_check(value, parent, config, performance)
                    save(value, path)
                    publish_json(root/'initial-verification.json', dict(
                        stage='actual_parent_state_preserved_before_updates', equal_sections=list(equal),
                        source=plan['source_checkpoint'], step=value['step'], cursor=0,
                        parent_mixed_cursor=parent['cursor'], real_stream_start_cursor=real_cursor(parent),
                        optimizer_states=len(value['optimizer']['state']), optimizer_updates=0,
                        data_digests=value['data_digests'], performance=performance))
                else:
                    save(value, path)
            torch.cuda.set_per_process_memory_fraction(.88)
            with patch.object(trainer, 'atomic_save', checked_save), patch.object(trainer,
                    'ReplacementBatches', partial(TrueBatchSampler, actual_batch=batch)):
                final = trainer.train(config, fork_from=source, data_fork=True, batch_fork=True,
                    performance=performance, reference=audit, execution_revision=plan['execution_revision'])
            stop = json.loads((root/'training/stop-report.json').read_text())
            coverage = checked_json(file_ref(root/'coverage-final.json'))
            complete = stop['step'] == parent['step']+updates_for(batch) and coverage['full_epoch_completed']
            publish_json(root/'completed.json', dict(stage='true_batch_epoch_finished' if complete else 'incomplete',
                full_epoch_complete=complete, new_updates=stop['step']-parent['step'],
                final_checkpoint=file_ref(final), stop_report=stop, coverage=file_ref(root/'coverage-final.json'),
                source=plan['source_checkpoint'], automatic_next_segment=False, quality_review_pending=True))
            status.update(optimizer_updates=stop['step']-parent['step'], final_step=stop['step'])
            progress(stage='finished_review_pending' if complete else 'incomplete_inspect_no_retry')
        except Exception as error:
            status.update(error=type(error).__name__, message=str(error))
            progress(stage='failed_inspect_no_retry')
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    torch.set_num_threads(1)
    run(p.parse_args().plan.resolve())
