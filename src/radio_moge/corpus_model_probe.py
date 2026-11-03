"""Observational student/teacher backward checks; never constructs an optimizer."""
from contextlib import nullcontext
import hashlib
import json
import math

import torch

from .corpus_losses import corpus_geometry_losses
from .feature_replay import replay_feature_losses
from .geometry_cache import canonical_json
from .losses import weighted_loss
from .official_gt import OFFICIAL_GT_KEYS
from .corpus_control import geometry_multiplier


def module_digest(module):
    """Hash all parameters AND buffers with bounded host memory, including BF16."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if tensor.layout != torch.strided:
            raise ValueError('Only dense model state can be audited')
        digest.update(canonical_json(dict(name=name, dtype=str(tensor.dtype), shape=list(tensor.shape))))
        # Current model states are contiguous. Reject unknown layouts rather than
        # allocate an unbounded contiguous copy of a large strided parameter.
        if not tensor.is_contiguous():
            raise ValueError(f'Noncontiguous model state: {name}')
        flat = tensor.detach().reshape(-1)
        count = max(1, 8 * 1024 * 1024 // tensor.element_size())
        for start in range(0, flat.numel(), count):
            chunk = flat[start:start + count].cpu().view(torch.uint8).numpy()
            digest.update(memoryview(chunk))
    return digest.hexdigest()


def gradient_summary(parameters):
    squared, present, missing = 0., 0, 0
    for value in parameters:
        if value.grad is None:
            missing += 1
            continue
        grad = value.grad.detach().float()
        if not torch.isfinite(grad).all():
            raise FloatingPointError('Nonfinite adapter gradient')
        squared += float(grad.square().sum(dtype=torch.float64))
        present += 1
    norm = math.sqrt(squared)
    if not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError('No finite nonzero adapter gradient')
    return dict(norm=norm, tensors_with_gradient=present, tensors_without_gradient=missing)


def assert_frozen(system):
    for module in (system.radio, system.encoder, system.decoder):
        if any(child.training for child in module.modules()):
            raise ValueError('Frozen backbone is in training mode')
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError('Frozen backbone has trainable parameters or gradients')


def select_probe_entries(db):
    """Deterministic support/depth extremes and different scenes in every stratum."""
    strata = db.execute("SELECT DISTINCT dataset,label_kind FROM images "
                        "WHERE disposition='selected' ORDER BY dataset,label_kind").fetchall()
    selected = {}
    for dataset, kind in strata:
        prefix = ("SELECT ordinal,pack,entry_json,scene_group FROM images "
                  "WHERE disposition='selected' AND dataset=? AND label_kind=? ")
        for order in ('valid_fraction ASC', 'valid_fraction DESC',
                      'median_depth ASC', 'median_depth DESC'):
            value = db.execute(prefix + 'ORDER BY ' + order + ',ordinal LIMIT 1', (dataset, kind)).fetchone()
            selected[value[0]] = value
        # Do not let all extremes come from one scene when others are present.
        for value in db.execute(prefix + 'GROUP BY scene_group HAVING ordinal=MIN(ordinal) '
                                'ORDER BY ordinal LIMIT 2', (dataset, kind)):
            selected[value[0]] = value
    if not selected:
        raise ValueError('No selected geometry for model preflight')
    return [dict(ordinal=i, pack=pack, entry=json.loads(encoded), group=group)
            for i, pack, encoded, group in sorted(selected.values())]


def backward_case(system, adapter, statistics, batch, config, cursor):
    """Use the production loss and RGB replay branch, with initial weights only."""
    device = next(adapter.parameters()).device
    image = batch['image'].to(device)
    if len(image) != config['train']['batch_size']:
        raise ValueError('Probe must use the configured microbatch size')
    amp = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)
           ) if config['train']['precision'] == 'bf16' else nullcontext
    adapter.zero_grad(set_to_none=True)
    assert_frozen(system)
    with amp(), torch.no_grad():
        features, target = system.encode(image, adapter.indices)
    if (not torch.isfinite(target.spatial).all() or not torch.isfinite(target.summary).all()
            or target.spatial.requires_grad or target.summary.requires_grad):
        raise FloatingPointError('Teacher features must be finite and detached')
    with amp():
        prediction = adapter(features)
    feature = statistics.loss(prediction, target)
    if cursor % 2:
        replay = batch['replay_image'].to(device)
        if replay.shape != image.shape:
            raise ValueError('Replay must preserve the configured batch shape')
        feature = replay_feature_losses(system, adapter, statistics, replay, adapter.indices, amp,
                                         config['feature_replay']['seed'] + cursor // 2)
    raw = system.decoder(prediction, image.shape[-2:])
    geometry = corpus_geometry_losses(raw, batch['geometry_targets'],
        config['models'].get('moge_source'), seed=config['train']['seed'] + cursor)
    skipped = int(geometry['gt_skipped_sparse_images'])
    enabled = geometry_multiplier(config, corpus_preflight=True)
    # Include explicit zero multipliers so the disabled branch is differentiable
    # and its exact-zero decoder-input gradient can be measured independently.
    geometry_total = sum(geometry[k] * config['loss'][k] for k in OFFICIAL_GT_KEYS)
    if not geometry_total.requires_grad:
        raise ValueError('Geometry loss is detached from adapter outputs')
    # A finite combined gradient alone could conceal a detached geometry branch.
    # Independently differentiate geometry through the frozen decoder first.
    decoder_grads = torch.autograd.grad(geometry_total, (prediction.spatial, prediction.summary),
                                       retain_graph=True, allow_unused=True)
    geometry_norm = 0.
    for grad in decoder_grads:
        if grad is not None:
            if not torch.isfinite(grad).all():
                raise FloatingPointError('Nonfinite gradient through frozen decoder')
            geometry_norm += float(grad.detach().float().square().sum(dtype=torch.float64))
    if not math.isfinite(geometry_norm) or (enabled and geometry_norm <= 0 and not skipped):
        raise FloatingPointError('Geometry loss did not reach adapter outputs')
    if not enabled and (geometry_norm != 0 or float(geometry_total.detach()) != 0):
        raise ValueError('Disabled geometry must have exactly zero loss and gradient')
    if skipped and (skipped != len(batch['geometry_targets']) or float(geometry_total.detach()) != 0):
        raise ValueError('A sparse-only view must have an explicit zero geometry loss')
    del decoder_grads
    feature_total = weighted_loss(feature, config['loss'])
    total = feature_total + geometry_total
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite actual-model objective')
    (total / config['train']['accumulation']).backward()
    gradients = gradient_summary(adapter.parameters())
    assert_frozen(system)
    return dict(cursor=cursor, feature_branch='replay_rgb' if cursor % 2 else 'paired_rgb',
        identities=batch['identities'], feature_weighted=float(feature_total.detach()),
        geometry_weighted=float(geometry_total.detach()), total=float(total.detach()),
        losses={k: float(v.detach()) for k, v in {**feature, **geometry}.items()},
        gradient=gradients, skipped_sparse_images=skipped,
        geometry_control_multiplier=enabled,
        geometry_gradient_at_adapter_outputs=math.sqrt(geometry_norm))


def observe_model(system, adapter, statistics, batches, config, on_case=None):
    """Reusable tiny-model tests are separate from the CLI's actual-model proof."""
    assert_frozen(system)
    adapter.train()
    modules = dict(radio=system.radio, encoder=system.encoder, decoder=system.decoder,
                   adapter=adapter, statistics=statistics)
    before = {name: module_digest(module) for name, module in modules.items()}
    rows = []
    try:
        for cursor, batch in enumerate(batches):
            row = backward_case(system, adapter, statistics, batch, config, cursor)
            rows.append(row)
            if on_case is not None:
                on_case(row)
        if not rows or len(rows) % 2 or {r['feature_branch'] for r in rows} != {'paired_rgb', 'replay_rgb'}:
            raise ValueError('Complete paired and replay feature slots are required')
        strata = {(i['dataset'], i['label_kind']) for row in rows for i in row['identities']}
        active = {(i['dataset'], i['label_kind']) for row in rows for i in row['identities']
                  if row['geometry_gradient_at_adapter_outputs'] > 0}
        enabled = geometry_multiplier(config, corpus_preflight=True)
        if (enabled and strata != active) or (not enabled and active):
            raise ValueError('No geometry gradient for a probed source/label stratum')
        after = {name: module_digest(module) for name, module in modules.items()}
        if before != after:
            raise ValueError('Model parameters or buffers changed during zero-update preflight')
        return dict(rows=rows, state_before=before, state_after=after, optimizer_updates=0,
                    optimizer_constructed=False, frozen_backbones_unchanged=True,
                    adapter_unchanged=True, statistics_unchanged=True,
                    geometry_control_multiplier=enabled,
                    disabled_geometry_exact_zero=not enabled,
                    finite_nonzero_adapter_gradient=True)
    finally:
        adapter.zero_grad(set_to_none=True)
