"""Optimizer-free checks for the ordinary RGB/geometry execution path.

This observer never updates a parameter or constructs an optimizer. Full batches
are retained for memory diagnostics; exceptions do not shrink them or retry.
"""
from contextlib import nullcontext
import hashlib

import numpy as np
import torch

from .checkpoint import rng_state, restore_rng
from .corpus_model_probe import assert_frozen, module_digest, gradient_summary
from .corpus_losses import ordinary_geometry_losses
from .geometry_cache import canonical_json
from .losses import weighted_loss
from .official_gt import OFFICIAL_GT_WEIGHTS
from .ordinary_geometry_targets import ordinary_target
from .ordinary_training import equal_state, COUNT_KEYS
from .rgb_geometry_batch import RGBGeometryBatch


def state_digest(value):
    """Bounded device-independent identity for nested checkpoint state."""
    result = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            result.update(canonical_json(dict(dtype=str(x.dtype), shape=list(x.shape))))
            flat = x.detach().reshape(-1)
            chunk = max(1, 4*1024*1024 // x.element_size())
            for start in range(0, flat.numel(), chunk):
                t = flat[start:start+chunk].cpu()
                if not torch.isfinite(t).all():
                    raise ValueError('Nonfinite checkpoint/probe state')
                result.update(t.contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                result.update(canonical_json([type(k).__name__, k]))
                visit(x[k])
        elif isinstance(x, (tuple, list)):
            result.update(canonical_json([type(x).__name__, len(x)]))
            for part in x:
                visit(part)
        else:
            result.update(canonical_json([type(x).__name__, x]))
    visit(value)
    return result.hexdigest()


def autocast(config, device):
    return (lambda: torch.autocast(device.type, dtype=torch.bfloat16)
            ) if config['train']['precision'] == 'bf16' else nullcontext


def feature_window(system, adapter, statistics, groups, config, denominator):
    """Same encode/raw-adapter/F1/whole-update weighting as the production loop."""
    adapter.zero_grad(set_to_none=True)
    amp = autocast(config, next(adapter.parameters()).device)
    rows, images = [], 0
    for batch in groups:
        image = batch.image if isinstance(batch, RGBGeometryBatch) else batch
        images += len(image)
        with amp(), torch.no_grad():
            features, target = system.encode(image, adapter.indices)
        with amp():
            prediction = adapter(features)
        losses = statistics.loss(prediction, target)
        total = weighted_loss(losses, config['loss'])
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite G=0 model loss')
        (total * (len(image) / denominator)).backward()
        rows.append(dict(shape=list(image.shape), losses={k: float(v.detach()) for k, v in losses.items()},
                         total=float(total.detach())))
    if images != denominator:
        raise ValueError('Probe feature denominator differs from its complete window')
    result = dict(rows=rows, gradient=gradient_summary(adapter.parameters()))
    assert_frozen(system)
    return result


def capture_gradients(adapter):
    return [None if p.grad is None else p.grad.detach().cpu().clone() for p in adapter.parameters()]


def compare_feature_window(system, adapter, statistics, plain, attached, config, *, same_path_check=False):
    """Pair exact G=0 losses/gradients/RNG; diagnose A/A before any relaxation."""
    if (len(plain) != len(attached) or any(not isinstance(x, RGBGeometryBatch) for x in attached)
            or any(not equal_state(a, b.image) for a, b in zip(plain, attached, strict=True))):
        raise ValueError('Ordinary CUDA groups/pixels differ from the original feature path')
    before = rng_state()
    denominator = sum(len(x) for x in plain)
    a = feature_window(system, adapter, statistics, plain, config, denominator)
    gradients, after = capture_gradients(adapter), rng_state()
    restore_rng(before)
    b = feature_window(system, adapter, statistics, attached, config, denominator)
    exact = equal_state(a, b) and equal_state(gradients, capture_gradients(adapter)) and equal_state(after, rng_state())
    aa = None
    if same_path_check or not exact:
        restore_rng(before)
        repeated = feature_window(system, adapter, statistics, plain, config, denominator)
        aa = (equal_state(a, repeated) and equal_state(gradients, capture_gradients(adapter))
              and equal_state(after, rng_state()))
    if not exact or aa is False:
        raise ValueError(f'Exact G=0 comparison failed: original_vs_attached={exact}, original_vs_original={aa}')
    adapter.zero_grad(set_to_none=True)
    restore_rng(after)
    return dict(feature_pixels_losses_gradients_exact=True, rng_after_equal=True,
                original_vs_original_exact=aa, images=denominator, groups=a['rows'], gradient=a['gradient'])


def verify_target(target, expected):
    if (target is not None) != expected['geometry_available']:
        raise ValueError('Actual target availability differs from bound camera proof')
    if target is None:
        return
    keys = ('corpus_ordinal', 'label_kind', 'supervision', 'cache_key', 'member_sha256',
            'pack', 'camera_contract', 'local_radius_contract')
    if (any(target[k] != expected[k] for k in keys)
            or int(target['valid'].sum()) != expected['observations']
            or target['intrinsics'].tolist() != expected['intrinsics']
            or any(hashlib.sha256(np.ascontiguousarray(target[k]).tobytes()).hexdigest() != value
                   for k, value in expected['arrays'].items())):
        raise ValueError('Actual target identity/rays/support differ from bound camera proof')


def verify_scale_routing(geometry, scale, targets):
    gradient = torch.autograd.grad(geometry, scale, allow_unused=True, retain_graph=True)[0]
    metric_slots = [t is not None and t['supervision'] == 'metric_native'
                    and int(t['valid'].sum()) >= 32 for t in targets]
    if gradient is not None:
        if not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite metric-scale gradient')
        for index, metric in enumerate(metric_slots):
            if not metric and torch.count_nonzero(gradient[index]):
                raise ValueError('Relative/missing/sparse slot acquired a metric-scale gradient')
    if not any(metric_slots) and gradient is not None:
        raise ValueError('Relative/missing/sparse batch acquired a metric-scale graph')


def geometry_window(system, adapter, statistics, groups, config, denominator, start_cursor=0,
                    expected_targets=None):
    """Check geometry connectivity separately, then the real combined backward.

    Geometry-only autograd.grad temporarily costs an additional adapter-gradient
    buffer. This makes memory observations conservative, not an optimizer step.
    """
    adapter.zero_grad(set_to_none=True)
    amp = autocast(config, next(adapter.parameters()).device)
    counts, rows, images, cursor = {k: 0 for k in COUNT_KEYS}, [], 0, start_cursor
    for batch in groups:
        if not isinstance(batch, RGBGeometryBatch) or any(type(s.augment) is not bool for s in batch.slots):
            raise ValueError('Geometry observer needs original annotated RGB groups')
        image = batch.image
        images += len(image)
        targets = [ordinary_target(s, augment=s.augment) for s in batch.slots]
        for slot, target in zip(batch.slots, targets, strict=True):
            if expected_targets is not None and slot.key in expected_targets:
                verify_target(target, expected_targets[slot.key])
        with amp(), torch.no_grad():
            features, target_features = system.encode(image, adapter.indices)
        with amp():
            prediction = adapter(features)
        feature = weighted_loss(statistics.loss(prediction, target_features), config['loss'])
        raw = system.decoder(prediction, image.shape[-2:])
        values = ordinary_geometry_losses(raw, targets, config['models'].get('moge_source'),
                                           seed=config['train']['seed']+cursor)
        geometry = sum(values[k]*w for k, w in OFFICIAL_GT_WEIGHTS.items())
        if not torch.isfinite(feature+geometry):
            raise FloatingPointError('Nonfinite ordinary model geometry/feature objective')
        grads = torch.autograd.grad(geometry, tuple(adapter.parameters()), retain_graph=True, allow_unused=True)
        squared = 0.
        for g in grads:
            if g is not None:
                if not torch.isfinite(g).all():
                    raise FloatingPointError('Nonfinite geometry-only adapter gradient')
                squared += float(g.detach().double().square().sum())
        del grads
        supervised = int(values['gt_supervised_images'])
        if supervised and not squared > 0:
            raise ValueError('Geometry has no adapter gradient despite valid targets')
        if not supervised and squared != 0:
            raise ValueError('Missing/sparse geometry produced a gradient')
        verify_scale_routing(geometry, raw['metric_scale'], targets)
        ((feature+geometry) * (len(image)/denominator)).backward()
        for k in COUNT_KEYS:
            if values[k].dtype != torch.int64:
                raise ValueError('Geometry probe lost integer coverage')
            counts[k] += int(values[k])
        rows.append(dict(shape=list(image.shape), feature=float(feature.detach()), geometry=float(geometry.detach()),
            geometry_gradient_norm=squared**.5, counts={k: int(values[k]) for k in COUNT_KEYS},
            targets=[None if t is None else dict(corpus_ordinal=t['corpus_ordinal'],
                label_kind=t['label_kind'], observations=int(t['valid'].sum()),
                camera_contract=t['camera_contract'], local_radius_contract=t['local_radius_contract']) for t in targets]))
        cursor += len(image) // config['train']['batch_size']
    if images != denominator or sum(counts[k] for k in
            ('gt_supervised_images', 'gt_skipped_sparse_images', 'gt_missing_target_images')) != images:
        raise ValueError('Geometry probe full-image coverage/denominator differs')
    gradients = gradient_summary(adapter.parameters())
    assert_frozen(system)
    return dict(images=images, counts=counts, groups=rows, combined_gradient=gradients,
                raw_adapter_decoder_input=True, whole_rgb_denominator=denominator)


def module_states(system, adapter, statistics):
    return {name: module_digest(module) for name, module in dict(
        radio=system.radio, encoder=system.encoder, decoder=system.decoder,
        adapter=adapter, statistics=statistics).items()}
