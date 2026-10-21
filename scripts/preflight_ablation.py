"""Measure real-model forward/backward paths on training images before long runs."""
import argparse
import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.config import load_config
from radio_moge.data import read_manifest
from radio_moge.losses import FeatureStatistics, geometry_losses, weighted_loss
from radio_moge.system import build_system
from radio_moge.train import make_loader


def check(config, checkpoint=None, geometry=False):
    torch.manual_seed(config['train']['seed'])
    torch.cuda.reset_peak_memory_stats()
    system, adapter = build_system(config, 'cuda')
    stats = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim']).cuda()
    if checkpoint:
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        assert state['adapter_config'] == adapter.config
        adapter.load_state_dict(state['adapter'])
        stats.load_state_dict(state['statistics'])
        del state
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config['train']['lr'])
    records = read_manifest(config['data']['train_manifest'])
    loader = make_loader(records, config, 0, 3, calibration=True)
    def amp():
        return (torch.autocast('cuda', dtype=torch.bfloat16)
                if config['train']['precision'] == 'bf16' else nullcontext())
    calls = [0]

    def count(*args):
        calls[0] += 1

    handle = system.radio.model.blocks[0].register_forward_hook(count)
    rows = []
    for i, image in enumerate(loader):
        image = image.cuda()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with amp(), torch.no_grad():
            features, target = system.encode(image, adapter.indices)
        if not checkpoint and i == 0:
            stats.fit([target])
        with amp():
            prediction = adapter(features)
        losses = stats.loss(prediction, target)
        if geometry:
            with torch.no_grad():
                teacher = system.decoder(target, image.shape[-2:])
            student = system.decoder(prediction, image.shape[-2:])
            terms = geometry_losses(student, teacher)
            if i == 0:
                grads = torch.autograd.grad(weighted_loss(terms, config['loss']),
                                            tuple(adapter.parameters()), retain_graph=True)
                assert all(torch.isfinite(g).all() for g in grads)
                assert sum(g.abs().sum().item() for g in grads) > 0
            losses.update(terms)
        total = weighted_loss(losses, config['loss'])
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
        for module in (system.radio, system.encoder, system.decoder):
            assert all(not p.requires_grad and p.grad is None for p in module.parameters())
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        rows.append(dict(batch=i, seconds=time.perf_counter()-start, total=float(total.detach()),
                         losses={k:float(v.detach()) for k,v in losses.items()},
                         grad_norm=float(norm), peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))
        assert calls[0] == i+1, 'RADIO backbone must execute exactly once per microbatch'
    handle.remove()
    return dict(kind=adapter.kind, geometry=geometry, batch_size=config['train']['batch_size'],
                parameter_count=sum(p.numel() for p in adapter.parameters()),
                intermediate_shapes=[list(v.shape) for v in features.intermediates],
                radio_forwards=calls[0], rows=rows, frozen_models_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fusion-config', required=True)
    parser.add_argument('--mlp-refine-config', required=True)
    parser.add_argument('--mlp-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results=[]
    for config,checkpoint,geometry in [
        (args.fusion_config,None,False),
        (args.mlp_refine_config,args.mlp_checkpoint,True),
    ]:
        result=check(load_config(config),checkpoint,geometry)
        results.append(result)
        atomic_json(results, Path(args.output))
        print(json.dumps(result),flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
