"""Observe component gradients without constructing an optimizer or changing weights.

The Gram matrix describes single-microbatch Euclidean gradients. It does not
reconstruct accumulated AdamW steps or predict heldout improvements.
"""
from contextlib import nullcontext
import math

import torch

from .corpus_losses import corpus_geometry_losses
from .corpus_model_probe import assert_frozen, module_digest
from .feature_replay import replay_feature_losses
from .losses import weighted_loss
from .official_gt import OFFICIAL_GT_KEYS


FEATURE_KEYS = ('spatial', 'summary', 'cosine', 'summary_cosine')


def component_gram(losses, parameters):
    """Differentiate each scalar independently; unused parameters have zero entries."""
    parameters = tuple(p for p in parameters if p.requires_grad)
    if not parameters or not losses or any(p.grad is not None for p in parameters):
        raise ValueError('Trainable parameters, component losses and empty .grad fields required')
    names = list(losses)
    vectors, presence = {}, {}
    size = sum(p.numel() for p in parameters)
    for index, (name, loss) in enumerate(losses.items()):
        if loss.numel() != 1 or not loss.requires_grad or not torch.isfinite(loss):
            raise ValueError('Finite differentiable scalar components required')
        grads = torch.autograd.grad(loss, parameters, allow_unused=True,
                                    retain_graph=index + 1 < len(losses))
        vector = torch.zeros(size, dtype=torch.float32, device=parameters[0].device)
        offset, count = 0, 0
        for parameter, grad in zip(parameters, grads, strict=True):
            if grad is not None:
                if not torch.isfinite(grad).all():
                    raise FloatingPointError('Nonfinite component gradient')
                vector[offset:offset + parameter.numel()].copy_(grad.detach().reshape(-1))
                count += 1
            offset += parameter.numel()
        vectors[name], presence[name] = vector, count
        del grads
    gram = {name: {} for name in names}
    # Bounded FP64 products avoid large extra copies of the 222M-parameter head.
    for i, a in enumerate(names):
        for b in names[i:]:
            value = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
            for start in range(0, size, 1024 * 1024):
                x = vectors[a][start:start + 1024 * 1024].double()
                y = vectors[b][start:start + 1024 * 1024].double()
                value += torch.dot(x, y)
            number = float(value)
            if not math.isfinite(number):
                raise FloatingPointError('Nonfinite gradient inner product')
            gram[a][b] = gram[b][a] = number
    norms = {name: math.sqrt(max(0., gram[name][name])) for name in names}
    cosines = {a: {b: gram[a][b] / (norms[a] * norms[b])
                  if norms[a] * norms[b] > 0 else None for b in names} for a in names}
    if any(p.grad is not None for p in parameters):
        raise ValueError('Observational gradients unexpectedly populated .grad')
    return dict(gram=gram, norms=norms, pairwise_cosine=cosines,
                parameters_with_gradient=presence, parameter_elements=size)


def combined_direction(gram, coefficients):
    """Positive dot means gradient descent on this combination descends that term."""
    if (not coefficients or set(coefficients) - set(gram)
            or any(not math.isfinite(v) for v in coefficients.values())):
        raise ValueError('Finite coefficients naming measured components required')
    norm2 = sum(a * b * gram[x][y] for x, a in coefficients.items()
                for y, b in coefficients.items())
    norm = math.sqrt(max(0., norm2))
    dots = {name: sum(weight * gram[name][other] for other, weight in coefficients.items())
            for name in gram}
    return dict(coefficients=coefficients, total_gradient_l2=norm,
        component_dot_total=dots,
        predicted_loss_change_per_unit_small_sgd_step={k: -v for k, v in dots.items()},
        component_cosine_to_total={name: dots[name] / (math.sqrt(max(0., gram[name][name])) * norm)
            if gram[name][name] > 0 and norm > 0 else None for name in gram})


def diagnostic_case(system, adapter, statistics, batch, config, cursor):
    device = next(adapter.parameters()).device
    image = batch['image'].to(device)
    if len(image) != 1 or config['train']['batch_size'] != 1:
        raise ValueError('Diagnostic supports the actual one-image microbatch')
    weights = [config['loss'][k] for k in FEATURE_KEYS]
    if len(set(weights)) != 1 or not math.isfinite(weights[0]) or weights[0] <= 0:
        raise ValueError('Expected one positive coefficient for all four feature terms')
    amp = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)
           ) if config['train']['precision'] == 'bf16' else nullcontext
    assert_frozen(system)
    with amp(), torch.no_grad():
        features, target = system.encode(image, adapter.indices)
    with amp():
        prediction = adapter(features)
    feature = statistics.loss(prediction, target)
    if cursor % 2:
        replay = batch['replay_image'].to(device)
        if replay.shape != image.shape:
            raise ValueError('Replay shape differs')
        feature = replay_feature_losses(system, adapter, statistics, replay, adapter.indices, amp,
                                        config['feature_replay']['seed'] + cursor // 2)
    raw = system.decoder(prediction, image.shape[-2:])
    geometry = corpus_geometry_losses(raw, batch['geometry_targets'],
        config['models'].get('moge_source'), seed=config['train']['seed'] + cursor)
    g = weighted_loss({k: geometry[k] for k in OFFICIAL_GT_KEYS}, config['loss'])
    components = dict(feature_unit=sum(feature[k] for k in FEATURE_KEYS),
        spatial_cosine_loss=feature['cosine'], summary_cosine_loss=feature['summary_cosine'],
        geometry_weighted=g)
    measured = component_gram(components, adapter.parameters())
    if measured['norms']['feature_unit'] <= 0:
        raise ValueError('Feature diagnostic needs a nonzero gradient')
    skipped = int(geometry['gt_skipped_sparse_images'])
    if measured['norms']['geometry_weighted'] <= 0 and not skipped:
        raise ValueError('Nonsparse geometry has no adapter gradient')
    if skipped and (float(g.detach()) != 0 or measured['norms']['geometry_weighted'] != 0):
        raise ValueError('Skipped geometry must contribute exact zero')
    scenarios = {
        'current': dict(feature_unit=weights[0], geometry_weighted=1.),
        'feature_weight_one': dict(feature_unit=1., geometry_weighted=1.),
        'feature_only': dict(feature_unit=1.),
        'geometry_only': dict(geometry_weighted=1.),
    }
    assert_frozen(system)
    return dict(cursor=cursor, feature_branch='replay_rgb' if cursor % 2 else 'paired_rgb',
        identities=batch['identities'], feature_coefficient=weights[0],
        losses={k: float(v.detach()) for k, v in {**feature, **geometry}.items()},
        components={k: float(v.detach()) for k, v in components.items()},
        skipped_sparse_images=skipped, gradients=measured,
        scenarios={name: combined_direction(measured['gram'], values)
                   for name, values in scenarios.items()})


def observe_components(system, adapter, statistics, batches, config, on_case=None):
    """State hashes verify observation only; parameter gradients never accumulate."""
    assert_frozen(system)
    adapter.train()
    adapter.zero_grad(set_to_none=True)
    modules = dict(radio=system.radio, encoder=system.encoder, decoder=system.decoder,
                   adapter=adapter, statistics=statistics)
    before = {name: module_digest(module) for name, module in modules.items()}
    rows = []
    try:
        for cursor, batch in enumerate(batches):
            # Matched across checkpoints; these are diagnostic, not resumed training RNG states.
            torch.manual_seed(config['train']['seed'] + cursor)
            row = diagnostic_case(system, adapter, statistics, batch, config, cursor)
            rows.append(row)
            if on_case is not None:
                on_case(row)
        if not rows or len(rows) % 2:
            raise ValueError('Complete paired/replay diagnostic cases required')
        after = {name: module_digest(module) for name, module in modules.items()}
        if before != after:
            raise ValueError('Model parameters or buffers changed during diagnosis')
        return dict(rows=rows, state_before=before, state_after=after,
            optimizer_constructed=False, optimizer_updates=0, model_state_unchanged=True,
            actual_adam_update_reconstructed=False,
            limitations=['Single-microbatch Euclidean gradients, not accumulated AdamW updates.',
                         'Selected training views, not a population or heldout-gain estimate.',
                         'Linearized weight scenarios are not training experiments.'])
    finally:
        adapter.zero_grad(set_to_none=True)
