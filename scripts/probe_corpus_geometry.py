"""Bounded cache-to-camera-to-loss probe; no model weights or optimizer calls."""
import argparse
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from indoor_download import atomic_write, write_json
from radio_moge.corpus_losses import corpus_geometry_losses
from radio_moge.decoder import camera_points
from radio_moge.geometry_cache import publish_json, read_geometry_member, sha_file
from radio_moge.geometry_sampling import load_candidate
from radio_moge.geometry_sources import pinhole_rays
from radio_moge.official_gt import OFFICIAL_GT_KEYS


PREFLIGHTS = ('arkit-cache-preflight-03', 'hypersim-cache-preflight-03',
    'structured3d-cache-preflight-01', 'tartanair-cache-preflight-01',
    'scannetpp-cache-preflight-01', 'coco-teacher-preflight-02', 'irs-teacher-preflight-01')


def contact_sheet(path, rgb, target, title):
    mask, depth = target['valid'], target['points'][..., 2]
    lo, hi = np.quantile(depth[mask], [.01, .99])
    value = np.clip((np.log1p(np.where(mask, depth, 0)) - np.log1p(lo)) /
                     max(1e-6, np.log1p(hi) - np.log1p(lo)), 0, 1)
    color = np.stack((value, 1 - np.abs(value * 2 - 1), 1 - value), -1)
    color[~mask] = 0
    panels = [Image.fromarray(rgb), Image.fromarray((color * 255).astype(np.uint8)),
              Image.fromarray((mask * 255).astype(np.uint8)).convert('RGB')]
    sheet = Image.new('RGB', (384 * 3, 240 + 40), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 2), title, fill='black')
    draw.text((4, 17), 'rectified RGB | exact selected Z (display-only nearest resize) | support', fill='black')
    for index, panel in enumerate(panels):
        sheet.paste(panel.resize((384, 240), Image.Resampling.NEAREST), (index * 384, 40))
    buffer = io.BytesIO()
    sheet.save(buffer, format='JPEG', quality=92)
    atomic_write(path, buffer.getvalue())


def probe_loss(target, device, seed):
    """Full pinned parameters, real cached labels, analytic differentiable input.

    This proves loss wiring/gradients, NOT student/teacher forward compatibility.
    The latter remains a separate locked, zero-update model preflight gate.
    """
    h, w = target['output_shape']
    k = torch.as_tensor(target['intrinsics'], dtype=torch.float32, device=device)
    u = (torch.arange(w, device=device) + .5) / w
    depth = (2. + .2 * u + .15 * u.square())[None].expand(h, -1)
    raw = dict(points=camera_points(depth[None], k[None]).detach().requires_grad_(),
               metric_scale=torch.tensor([1.2], device=device, requires_grad=True),
               mask=torch.zeros((1, h, w), device=device))
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    # CUDA scatter reductions in the pinned local loss can differ by one ULP
    # even for UNCHANGED inputs. Compare exact masks with deterministic forward
    # reductions; do not weaken the equality criterion or mistake repeat noise
    # for an invalid-label contribution. grid_sample backward is separately
    # checked for finite gradients (not claimed bitwise deterministic).
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        values = corpus_geometry_losses(raw, [target], seed=seed)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)
    sum(values[key] for key in OFFICIAL_GT_KEYS).backward()
    gradient = raw['points'].grad
    if not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
        raise ValueError('Loss did not produce finite nonzero geometry gradients')
    scale_grad = raw['metric_scale'].grad
    if target['supervision'] == 'relative_pseudo' and scale_grad is not None:
        raise ValueError('Pseudo labels leaked into metric-scale supervision')
    if not torch.equal(cpu_before, torch.get_rng_state()):
        raise ValueError('Loss perturbed caller CPU RNG')
    if cuda_before is not None and not torch.equal(cuda_before, torch.cuda.get_rng_state()):
        raise ValueError('Loss perturbed caller CUDA RNG')
    # Invalid *label* values and query positions must have literally no effect.
    changed = dict(target, points=target['points'].copy(), query_uv=target['query_uv'].copy())
    changed['points'][~target['valid']] = np.nan
    changed['query_uv'][~target['valid']] = np.inf
    raw['mask'].fill_(1)
    try:
        torch.use_deterministic_algorithms(True)
        repeat = corpus_geometry_losses(raw, [changed], seed=seed)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)
    maximum_difference = max(float((values[key] - repeat[key]).detach().abs()) for key in OFFICIAL_GT_KEYS)
    if maximum_difference != 0:
        raise ValueError('Invalid labels or predicted mask changed the geometry objective')
    return dict(losses={key: float(value.detach()) for key, value in values.items()},
                point_gradient_l1=float(gradient.abs().sum()),
                scale_gradient=None if scale_grad is None else float(scale_grad),
                masked_label_max_abs_loss_difference=maximum_difference,
                caller_rng_preserved=True, deterministic_forward_for_exact_mask_comparison=True,
                backward_finite_tested_not_bitwise_determinism=True, optimizer_updates=0)


def probe(root, attempt, device):
    output = root / attempt
    output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.set_num_threads(1)
    status = dict(stage='probing', completed=0, loss_cases=0, optimizer_updates=0,
                  training_admission=False, started_at=time.time())
    write_json(output / 'status.json', status)
    records, evidence = [], []
    try:
        for source in PREFLIGHTS:
            directory = root / source
            report = json.loads((directory / 'report.json').read_text())
            if not report.get('passed'):
                raise ValueError('Source preflight not passed: ' + source)
            pack_index = json.loads((directory / 'packs/packs.json').read_text())
            entries = []
            for pack in pack_index['packs']:
                receipt = directory / 'packs' / pack['receipt']
                if sha_file(receipt) != pack['sha256']:
                    raise ValueError('Source receipt changed')
                body = json.loads(receipt.read_text())
                path = receipt.parent / body['tar']
                if sha_file(path) != body['sha256']:
                    raise ValueError('Source pack changed')
                entries.extend((path, entry) for entry in body['entries'])
            evidence.append(dict(source=source, report_sha256=sha_file(directory / 'report.json'),
                packs_index_sha256=sha_file(directory / 'packs/packs.json'), count=len(entries)))
            rotation_seen = set()
            offcenter_seen = False
            for index, (pack, entry) in enumerate(entries):
                depth, valid, kd, _ = read_geometry_member(pack, entry)
                turn = entry['image'].get('image_rotation_cw', 0)
                loss_case = (index in (0, len(entries) - 1)
                             or turn not in rotation_seen
                             or (kd[1, 2] > 1 and not offcenter_seen))
                rotation_seen.add(turn)
                offcenter_seen |= kd[1, 2] > 1
                for augmented in (False, True):
                    rgb, target, view = load_candidate(pack, entry, (400, 640),
                                                       seed=13145, augment=augmented)
                    selected = target['valid']
                    ids = target['native_index'][selected]
                    exact = (pinhole_rays(kd, *depth.shape).reshape(-1, 3)[ids]
                             * depth.ravel()[ids, None]) @ view['camera_transform'].T
                    error = np.linalg.norm(exact - target['points'][selected], axis=1) / np.maximum(
                        np.linalg.norm(exact, axis=1), 1)
                    if (len(ids) < 32 or len(ids) != len(np.unique(ids))
                            or not valid.ravel()[ids].all() or error.max() > 4 * np.finfo(np.float32).eps):
                        raise ValueError('Observation identity, geometry, or support check failed')
                    item = dict(source=source, cache_key=entry['key'], image_sha256=entry['image']['sha256'],
                        dataset=target['dataset'], supervision=target['supervision'], augmented=augmented,
                        image_rotation_cw=turn, source_k=kd.tolist(),
                        target_k=view['intrinsics'].tolist(), camera_transform=view['camera_transform'].tolist(),
                        native_shape=list(depth.shape), loss_shape=list(selected.shape),
                        native_valid_pixels=int(valid.sum()), selected_observations=len(ids),
                        supported_loss_fraction=float(selected.mean()),
                        native_to_target_max_relative_error=float(error.max()))
                    if loss_case:
                        item['loss_probe'] = probe_loss(target, device, 43)
                        status['loss_cases'] += 1
                        visual_name = f'{source}-{index:03d}-{int(augmented)}.jpg'
                        contact_sheet(output / visual_name, rgb, target,
                                      f'{target["dataset"]} {target["supervision"]} augmented={augmented} {entry["key"][:12]}')
                        item['visual'] = visual_name
                    records.append(item)
                    status['completed'] += 1
                    status['updated_at'] = time.time()
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        publish_json(output / 'report.json', dict(stage='diagnostic_complete', passed=True,
            training_admission=False, optimizer_updates=0, device=device, torch_version=torch.__version__,
            full_pinned_loss_parameters=True, model_forward_tested=False,
            source_evidence=evidence, records=records,
            code_sha256={name: sha_file(Path(name)) for name in (
                'scripts/probe_corpus_geometry.py', 'src/radio_moge/geometry_sampling.py',
                'src/radio_moge/corpus_losses.py')}))
        status.update(stage='complete', passed=True)
        write_json(output / 'status.json', status)
    except BaseException as error:
        status.update(stage='failed', error=repr(error))
        write_json(output / 'status.json', status)
        publish_json(output / 'failure.json', dict(status=status, source_evidence=evidence, records=records))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a new attempt directory basename')
    probe(args.root, args.attempt, args.device)
