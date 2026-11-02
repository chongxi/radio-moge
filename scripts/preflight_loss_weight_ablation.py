"""Zero-update real-model loss/backward checks and feature-output gradient audit."""
import argparse
import json
from pathlib import Path
import time

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.data import read_manifest
from radio_moge.gt_geometry import GT_LOSS_KEYS, ground_truth_losses
from radio_moge.losses import FeatureStatistics
from radio_moge.official_gt import OFFICIAL_GT_WEIGHTS, official_ground_truth_losses
from radio_moge.system import build_system
from radio_moge.train import make_loader


def run(root):
    plan = json.loads((root / 'plan.json').read_text())
    config = json.loads(Path(plan['arms']['custom-f01']['path']).read_text())
    torch.set_num_threads(1)
    state = torch.load(plan['source'], map_location='cpu', weights_only=True, mmap=True)
    assert state['step'] == 232288
    system, adapter = build_system(config, torch.device('cuda'))
    adapter.load_state_dict(state['adapter'])
    stats = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim']).cuda()
    stats.load_state_dict(state['statistics'])
    del state
    records = read_manifest(config['data']['train_manifest'])
    loader = make_loader(records, config, 0, 4)
    results = []
    for index, batch in enumerate(loader):
        image = batch.pop('image').cuda()
        target = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            features, teacher = system.encode(image, adapter.indices)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = adapter(features)
        raw = system.decoder(prediction, image.shape[-2:])
        feature_loss = sum(stats.loss(prediction, teacher).values())
        inputs = (prediction.spatial, prediction.summary)
        feature_grad = torch.autograd.grad(feature_loss, inputs, retain_graph=True)
        for objective in ('metric_custom', 'moge2_core'):
            started = time.monotonic()
            if objective == 'metric_custom':
                losses = ground_truth_losses(raw, target)
                weights = {key: config['loss'][key] for key in GT_LOSS_KEYS}
            else:
                losses = official_ground_truth_losses(raw, target, seed=43 + index)
                weights = OFFICIAL_GT_WEIGHTS
            geometry_loss = sum(losses[key] * weight for key, weight in weights.items())
            gradient = torch.autograd.grad(geometry_loss, inputs, retain_graph=True)
            row = {'microbatch': index, 'objective': objective,
                   'losses': {k: v.item() for k, v in losses.items()},
                   'unweighted_feature_loss': feature_loss.item(),
                   'weighted_geometry_loss': geometry_loss.item()}
            for branch, f, g in zip(('spatial', 'summary'), feature_grad, gradient):
                f, g = f.float().flatten(), g.float().flatten()
                assert torch.isfinite(f).all() and torch.isfinite(g).all()
                fn, gn = f.norm(), g.norm()
                row[branch] = dict(feature_norm_at_weight1=fn.item(), geometry_norm=gn.item(),
                                   gradient_cosine=(f.dot(g) / (fn * gn).clamp_min(1e-20)).item(),
                                   geometry_to_feature_ratio={str(w): (gn / (fn * w).clamp_min(1e-20)).item()
                                                              for w in (.1, 1., 3.)})
            if objective == 'moge2_core' and index == 0:
                (geometry_loss + .1 * feature_loss).backward(retain_graph=True)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters())
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
                for name in ('radio', 'encoder', 'decoder'):
                    assert all(p.grad is None for p in getattr(system, name).parameters())
                adapter.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            row['loss_and_gradient_seconds'] = time.monotonic() - started
            results.append(row)
            print(json.dumps(row), flush=True)
    atomic_json(dict(stage='passed', optimizer_updates=0, microbatches=4, results=results,
                     interpretation='Gradients measured at adapter spatial/summary outputs, not parameter gradients; '
                                    'four training microbatches only, not a generalization result.'),
                root / 'preflight.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
