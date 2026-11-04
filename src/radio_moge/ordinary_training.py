"""Separate admission and observer for geometry on the original feature stream.

No publisher or successful model evidence is provided by this module. Historical
corpus/reference seals cannot pass this gate. CPU fixtures replace the gate in
memory; they must never publish a training admission.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import torch

from .corpus_admission import checked_json, checked_path, file_ref
from .data import DeterministicBatches
from .data_fork import check_data_fork
from .geometry_cache import canonical_json
from .losses import FEATURE_LOSS_KEYS
from .ordinary_geometry import ORDINARY_CAMERA_CONTRACT, ORDINARY_LOCAL_RADIUS
from .ordinary_geometry_data import GeometrySidecar
from .rgb_input_audit import RGBInputAudit, load_reference_records
from .checkpoint import atomic_json


COUNT_KEYS = frozenset(('gt_supervised_images', 'gt_skipped_sparse_images',
    'gt_missing_target_images', 'gt_metric_images', 'gt_observations', 'gt_edge_pairs'))
CPU_REQUIREMENTS = ('production_loop_g0_exact', 'geometry_off_no_calls',
    'slot_audit_and_commit_only', 'whole_rgb_denominator', 'integer_coverage',
    'ordinary_camera_and_payload_checks')
MODEL_REQUIREMENTS = ('cuda_transport_exact', 'initial_adapter_rms_adam_rng_exact',
    'feature_pixels_losses_gradients_exact', 'geometry_gradients_finite_nonzero',
    'frozen_state_unchanged', 'rng_restored', 'gradients_discarded',
    'optimizer_not_constructed')


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def config_digest(config):
    value = deepcopy(config)
    value['ordinary_geometry'].pop('admission')
    return digest(value)


def implementation_digest():
    # Bind every local runtime helper, including indirect camera/cache imports.
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_ref(p)['sha256']
                   for p in sorted(root.rglob('*.py'))})


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and torch.equal(a, b))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def verify_initial_state(old, new, config, source_step, digests):
    """Compare the checkpoint actually saved by this trainer before any update."""
    if (old['step'] != source_step or new['step'] != source_step or new['cursor'] != 0
            or new['config'] != config or new['data_digests'] != digests):
        raise ValueError('Ordinary actual initial checkpoint/config/cursor differs')
    for key in ('adapter', 'statistics', 'optimizer', 'rng', 'adapter_config', 'model_config', 'revisions'):
        if not equal_state(old[key], new[key]):
            raise ValueError('Ordinary actual initial state differs: ' + key)
    if not old['optimizer']['state'] or {int(v['step']) for v in new['optimizer']['state'].values()} != {source_step}:
        raise ValueError('Ordinary Adam counters were reset or missing')


def validate_ordinary_config(config):
    spec = config.get('ordinary_geometry')
    if spec is None:
        return
    if (not isinstance(spec, dict) or set(spec) != {'sidecar', 'multiplier', 'camera_contract',
            'local_radius_contract', 'pixel_convention', 'admission'}
            or type(spec['multiplier']) not in (int, float) or spec['multiplier'] not in (0, 1)
            or spec['camera_contract'] != ORDINARY_CAMERA_CONTRACT
            or spec['local_radius_contract'] != ORDINARY_LOCAL_RADIUS
            or spec['pixel_convention'] != 'normalized_half_center'):
        raise ValueError('Explicit ordinary geometry contracts and multiplier 0/1 required')
    for key in ('sidecar', 'admission'):
        ref = spec[key]
        if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}
                or not isinstance(ref['path'], str) or not Path(ref['path']).is_absolute()
                or not isinstance(ref['sha256'], str) or len(ref['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in ref['sha256'])):
            raise ValueError('Ordinary geometry needs separate exact evidence references')
    data, settings = config['data'], config['train']
    if (config.get('geometry_supervision') is not None or config.get('feature_replay') is not None
            or config.get('corpus_loss_control') is not None or data.get('geometry_corpus') is not None
            or data.get('mixture_weights') is not None or data.get('coverage_epochs') is not None
            or config['loss'] != {key: 1. for key in FEATURE_LOSS_KEYS}
            or data.get('augment') is not True or data.get('compact_transport') is not True
            or len(data['buckets']) != 3 or len(set(map(tuple, data['buckets']))) != 3
            or settings.get('eval_at_start') is not False
            or settings.get('lr_schedule') != 'constant'):
        raise ValueError('Ordinary geometry preserves F1, permutation, three RGB buckets and compact preparation')


def validate_ordinary_performance(config, performance):
    from .performance import validate_performance
    if (not performance or performance.get('autotune') or performance.get('data_autotune')
            or performance.get('effective_batch') != 32 or performance.get('base_effective_batch') != 32
            or performance.get('schedule_ratio') != 1
            or performance.get('data_pipeline') != 'prepared'
            or performance.get('compact_transport') is not True):
        raise ValueError('Ordinary geometry requires the unchanged prepared effective batch32 profile')
    validate_performance(performance, config, constant_fork=True)


def check_ordinary_data_fork(config, state, digests):
    # Parent must be original feature training, never a geometry checkpoint silently
    # normalized into one. General data-fork checks still bind all old eval IDs.
    previous = state['config']
    if (previous.get('ordinary_geometry') is not None or previous.get('geometry_supervision')
            or previous.get('feature_replay') or previous['data'].get('mixture_weights')
            or previous['loss'] != {key: 1. for key in FEATURE_LOSS_KEYS}
            or previous['data']['buckets'] != config['data']['buckets']):
        raise ValueError('Ordinary fork requires the original F1 permutation and resolution contract')
    value = deepcopy(config)
    value.pop('ordinary_geometry')
    for key in ('eval_at_start', 'log_every'):
        value['train'][key] = previous['train'][key]
    # Transport is observational; its pixels are separately bound by the new gate.
    value['data'].pop('compact_transport', None)
    if 'compact_transport' in previous['data']:
        value['data']['compact_transport'] = previous['data']['compact_transport']
    check_data_fork(value, state, digests)


class OrdinaryTrainingAudit(RGBInputAudit):
    def __init__(self, config, performance, seal, planned):
        super().__init__(config['output_dir'], seal['source_step'], config['train']['batch_size'],
            32 // config['train']['batch_size'], performance['compute_batch'], planned)
        self.config, self.seal = deepcopy(config), seal
        self.sidecar = GeometrySidecar(config['ordinary_geometry']['sidecar'])

    def load_records(self, path):
        if path != self.seal['manifest']['path']:
            raise ValueError('Ordinary RGB manifest path changed')
        records = load_reference_records(path, workers=1)
        checked_path(self.seal['manifest'])
        if len(records) != self.seal['rgb_images'] or records.digest() != self.seal['rgb_digest']:
            raise ValueError('Ordinary full RGB scope, order or identity changed')
        return records

    def check_fork(self, config, state, digests):
        if (config != self.config or state['step'] != self.seal['source_step']
                or digests != self.seal['data_digests']):
            raise ValueError('Ordinary runtime config, source step or heldout identities changed')
        check_ordinary_data_fork(config, state, digests)

    def before_updates(self, system, adapter, statistics, records, config, output):
        # Actual-model evidence is checked BEFORE model/optimizer construction by
        # the opener. This runtime hook checks the objects used by the trainer.
        if (config != self.config or len(records) != self.seal['rgb_images']
                or records.digest() != self.seal['rgb_digest']):
            raise ValueError('Ordinary runtime records/config differ from sealed evidence')
        for module in (system.radio, system.encoder, system.decoder):
            if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise ValueError('Ordinary frozen modules became trainable')
        source = checked_path(self.seal['source_checkpoint'])
        initial_path = Path(output) / 'initial.pt'
        old = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        new = torch.load(initial_path, map_location='cpu', weights_only=True, mmap=True)
        verify_initial_state(old, new, config, self.seal['source_step'], self.seal['data_digests'])
        if (not equal_state(adapter.state_dict(), new['adapter'])
                or not equal_state(statistics.state_dict(), new['statistics'])):
            raise ValueError('Ordinary live adapter/RMS differs from actual initial checkpoint')
        atomic_json(dict(stage='ordinary_actual_initial_state_verified', optimizer_updates=0,
            source=self.seal['source_checkpoint'], initial=file_ref(initial_path),
            source_step=self.seal['source_step'], new_cursor=0, adapter_rms_adam_rng_exact=True,
            independent_process=False, cpu_checkpoint_readback=True,
            scheduler='Constant LR wrapper intentionally reset; Adam and its counters preserved'),
            Path(output) / 'ordinary-initial-state.json')


def open_ordinary_training_gate(config, source, performance):
    """Reject absent/stale CPU AND actual CUDA evidence before building any model."""
    validate_ordinary_config(config)
    validate_ordinary_performance(config, performance)
    spec = config['ordinary_geometry']
    seal = checked_json(spec['admission'])
    if (seal.get('stage') != 'ordinary_rgb_geometry_training_accepted'
            or seal.get('schema_version') != 1 or seal.get('training_admission') is not True):
        raise ValueError('New ordinary geometry admission required; historical seals are not valid')
    expected = dict(config_sha256=config_digest(config), performance_sha256=digest(performance),
                    implementation_sha256=implementation_digest())
    if any(seal.get(k) != v for k, v in expected.items()):
        raise ValueError('Ordinary config/profile/code differs from its gate')
    if str(Path(source).resolve()) != seal['source_checkpoint']['path']:
        raise ValueError('Ordinary initial checkpoint path differs')
    checked_path(seal['source_checkpoint'])
    if (seal['manifest']['path'] != config['data']['train_manifest']
            or seal['sidecar'] != spec['sidecar'] or seal['rgb_images'] != 1637908
            or seal['source_step'] != 232288
            or seal['optimizer_updates'] != 250
            or config['train']['steps'] - seal['source_step'] != 250
            or performance['anchor_step'] != seal['source_step'] or performance['anchor_cursor'] != 0):
        raise ValueError('Ordinary full RGB, source anchor or bounded 250-update scope differs')
    checked_path(seal['manifest'])
    checked_path(seal['sidecar'])
    planned = checked_json(seal['planned_keys'])
    keys = [list(k) for batch in DeterministicBatches(seal['rgb_images'], config['train']['batch_size'],
            config['data']['buckets'], config['train']['seed'], 0,
            250 * (32 // config['train']['batch_size'])) for k in batch]
    if planned != keys:
        raise ValueError('Ordinary complete permutation/augmentation plan differs')
    binding = dict(**expected, source_checkpoint=seal['source_checkpoint'], manifest=seal['manifest'],
                   sidecar=seal['sidecar'], planned_keys=seal['planned_keys'],
                   data_digests=seal['data_digests'])
    for field, stage, required in (
        ('cpu_evidence', 'ordinary_production_loop_cpu_verified', CPU_REQUIREMENTS),
        ('model_evidence', 'ordinary_actual_model_zero_update_verified', MODEL_REQUIREMENTS)):
        report = checked_json(seal[field])
        if (report.get('stage') != stage or report.get('binding') != binding
                or report.get('production_optimizer_updates' if field == 'cpu_evidence'
                              else 'optimizer_updates') != 0
                or report.get('training_admission') is not False
                or any(report.get(k) is not True for k in required)):
            raise ValueError('Ordinary CPU/actual-model gate is incomplete or differs from runtime')
        if field == 'model_evidence' and report.get('device') != 'cuda':
            raise ValueError('Ordinary actual-model gate must verify CUDA transport')
    return OrdinaryTrainingAudit(config, performance, seal, planned)
