"""Three disposable updates; verify all RADIO input paths and frozen targets."""

import argparse
import json
from pathlib import Path
import signal
import time

import torch

from radio_moge.adapters import zero_dominant_channels, zero_dominant_summary_channels
from radio_moge.checkpoint import atomic_json
from radio_moge.data import read_manifest
from radio_moge.losses import FeatureStatistics, weighted_loss
from radio_moge.system import build_system
from radio_moge.train import make_loader


def diagnostics(before, after, summary=False):
    if summary:
        before, after = before.reshape(-1, 2, 1280), after.reshape(-1, 2, 1280)
        hot = (before != after)
        energy = before.float().square().sum(-1)
        removed = ((before.float() - after.float()).square().sum(-1) / energy.clamp_min(1e-12))
        return dict(removed_channels=hot.sum(-1).tolist(), removed_energy_fraction=removed.tolist())
    hot = (before != after).flatten(2).any(-1)
    return dict(removed_channels=hot.sum(-1).tolist(), total_channels=before.shape[1])


def check(root):
    started = time.time()
    config = json.loads((root / 'config.json').read_text())
    torch.set_num_threads(2)
    torch.manual_seed(43)
    torch.cuda.reset_peak_memory_stats()
    system, adapter = build_system(config, 'cuda')
    assert adapter.summary_token_dim == 1280 and system.radio.summary_dim == 2560
    assert sum(p.numel() for p in adapter.parameters()) == 221740868
    stats = FeatureStatistics(1024, 1024).cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config['train']['lr'], weight_decay=.01)
    records = read_manifest(config['data']['train_manifest'])
    loader = make_loader(records, config, 0, 3, calibration=True)
    rows, cleaning = [], []
    calls = [0]
    handle = system.radio.model.blocks[0].register_forward_hook(
        lambda *_: calls.__setitem__(0, calls[0] + 1))
    for i, image in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        image = image.cuda()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            features, target = system.encode(image, adapter.indices)
        original = [x.clone() for x in (*features.intermediates, features.summary,
                                        target.spatial, target.summary)]
        expected = [zero_dominant_channels(f) for f in features.intermediates]
        expected_summary = zero_dominant_summary_channels(features.summary, 1280)
        report = dict(batch=i, taps=[diagnostics(x, y) for x, y in
                                    zip(features.intermediates, expected)],
                      summary=diagnostics(features.summary, expected_summary, True))
        cleaning.append(report)
        hooks = []
        for layer, correct in zip(adapter.input, expected):
            def verify(_module, inputs, correct=correct):
                assert torch.equal(inputs[0], correct)
            hooks.append(layer.register_forward_pre_hook(verify))
        def verify_summary(_module, inputs):
            assert torch.equal(inputs[0], expected_summary)
        hooks.append(adapter.summary_input.register_forward_pre_hook(verify_summary))
        if i == 0:
            stats.fit([target])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = adapter(features)
        for hook in hooks:
            hook.remove()
        for before, after in zip(original, (*features.intermediates, features.summary,
                                           target.spatial, target.summary)):
            assert torch.equal(before, after), 'Raw features/teacher targets were modified'
        losses = stats.loss(prediction, target)
        total = weighted_loss(losses, config['loss'])
        assert torch.isfinite(total)
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
        assert torch.isfinite(norm) and norm > 0
        before = adapter.summary_output[-1].weight.detach().clone()
        optimizer.step()
        update = (adapter.summary_output[-1].weight.detach() - before).float().norm().item()
        assert update > 0
        for module in (system.radio, system.encoder, system.decoder):
            assert all(not p.requires_grad and p.grad is None for p in module.parameters())
        torch.cuda.synchronize()
        assert calls[0] == i + 1
        rows.append(dict(batch=i, loss=float(total.detach()), grad_norm=float(norm),
                         update_l2_summary_head=update,
                         seconds=time.perf_counter()-begin,
                         losses={k:float(v.detach()) for k,v in losses.items()}))
    handle.remove()
    proof = dict(stage='passed', checked_at=time.time(), elapsed_seconds=time.time()-started,
                 disposable_optimizer_updates=3, weights_saved=False, main_run_initialization='fresh',
                 all_consumed_radio_inputs_cleaned_before_projection=True,
                 raw_features_and_teacher_targets_unchanged=True,
                 cleaning=cleaning, rows=rows, radio_forwards=calls[0], frozen_backbones=True,
                 parameter_count=221740868, peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    atomic_json(proof, root / 'preflight.json')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('cleaning','rows')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    def timeout(*_):
        raise TimeoutError('Disposable preflight exceeded 1800 seconds; no training')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
    check(args.root)
