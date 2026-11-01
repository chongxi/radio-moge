"""Heldout measured-geometry evaluation with known cameras and fixed GT masks."""
from pathlib import Path

import torch

from .checkpoint import atomic_json
from .gt_data import GroundTruthImages, validate_gt_records
from .gt_geometry import calibrated_geometry, ground_truth_losses
from .metrics import depth_metrics


@torch.no_grad()
def evaluate_ground_truth(system, adapter, records, config, output):
    validate_gt_records(records, split='val')
    device = next(adapter.parameters()).device
    previous_mode = adapter.training
    adapter.eval()
    data = GroundTruthImages(records, augment=False)
    rows = []
    try:
        for index, record in enumerate(records):
            height, width = record['height'], record['width']
            batch = {k: v[None].to(device) for k, v in data[index, 0, height, width].items()}
            image = batch.pop('image')
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda' and config['train']['precision'] == 'bf16'):
                features, _ = system.encode(image, adapter.indices)
                predicted = adapter(features)
            raw = system.decoder(predicted, (height, width))
            decoded = calibrated_geometry(raw, batch['intrinsics'], batch['valid'])
            metrics = depth_metrics(decoded['depth'], batch['depth'], batch['valid'])
            metrics.update({k: v.item() for k, v in ground_truth_losses(raw, batch).items()})
            labels = batch['plane_labels'].long()
            equations = torch.gather(batch['planes'], 1,
                                     (labels.flatten(1).clamp_min(1) - 1)[..., None].expand(-1, -1, 4))
            equations = equations.reshape(*labels.shape, 4)
            mask = batch['valid'] & (labels > 0)
            if mask.any():
                residual = (decoded['projected_points'] * equations[..., :3]).sum(-1) + equations[..., 3]
                metrics['plane_rmse_m'] = residual[mask].square().mean().sqrt().item()
            rows.append({'id': record['id'], 'scene': record['scene'], **metrics})
        keys = set().union(*(set(row) - {'id', 'scene'} for row in rows))
        means = {key: sum(row[key] for row in rows if key in row) / sum(key in row for row in rows)
                 for key in keys}
        means['images'] = len(rows)
        means['plane_images'] = sum('plane_rmse_m' in row for row in rows)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(rows, output / 'per_image.json')
        atomic_json(means, output / 'metrics.json')
        atomic_json({'camera': 'known measured intrinsics', 'scale_alignment': False,
                     'support': 'fixed measured GT; predicted mask ignored',
                     'split': 'permanent heldout scenes', 'aggregation': 'equal image weighting',
                     'plane_metric': 'distance to fixed GT plane, including pose/scale/flatness error'},
                    output / 'protocol.json')
        return means
    finally:
        adapter.train(previous_mode)
