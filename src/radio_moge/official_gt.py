"""Pinned MoGe-2 type-B geometry/scale losses on our measured RGB/GT pairs.

This deliberately compares the geometry core, not the entire upstream recipe:
normal-head/mask losses and upstream label refinement are NOT reproduced here.
The global/local/edge/scale functions and their parameters are used unchanged.
"""
from functools import lru_cache
import json

import torch

from .config import MOGE_COMMIT
from .decoder import camera_points
from .upstream import source_path


OFFICIAL_GT_KEYS = ('gt_global', 'gt_local4', 'gt_local16', 'gt_edge', 'gt_scale')
OFFICIAL_GT_WEIGHTS = dict(zip(OFFICIAL_GT_KEYS, (1., 1., 1., 1., .1)))


@lru_cache(maxsize=4)
def official_functions(moge_source=None):
    import sys
    source = source_path('MoGe', MOGE_COMMIT, moge_source)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from moge.train import losses
    spec = json.loads((source / 'configs/train/v2.json').read_text())['loss']['B']
    mapping = {
        'gt_global': spec['points']['global'],
        'gt_local4': spec['points']['patch_4'],
        'gt_local16': spec['points']['patch_16'],
        'gt_edge': spec['points']['normal'],
        'gt_scale': spec['metric_scale']['metric_scale'],
    }
    if any(value['weight'] != OFFICIAL_GT_WEIGHTS[key] for key, value in mapping.items()):
        raise ValueError('Pinned upstream geometry weights changed')
    return {key: (getattr(losses, value['function']), value.get('params', {}))
            for key, value in mapping.items()}


def official_ground_truth_losses(raw, target, moge_source=None, seed=0):
    """Affine shape and detached-target log-scale supervision in FP32.

    Fixed measured support, never the predicted mask. Invalid holes are ignored,
    not labelled as infinity. Local sampling has a private, reproducible RNG so
    changing the loss cannot alter the model/data random stream in the A/B test.
    """
    functions = official_functions(moge_source)
    points = raw['points']
    devices = [points.device.index] if points.is_cuda else []
    with torch.autocast(points.device.type, enabled=False), torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        depth = target['depth'].detach().float()
        valid = target['valid'].bool() & torch.isfinite(depth) & (depth > 0)
        if not valid.flatten(1).any(1).all():
            raise ValueError('Every GT image requires valid positive metric depth')
        intrinsics = target['intrinsics'].detach().float()
        truth = camera_points(torch.where(valid, depth, 1), intrinsics)
        # Match the upstream training loop: invalid XYZ becomes 1, with mask false.
        truth = torch.where(valid[..., None], truth, 1)
        focal = (intrinsics[:, 0, 0].reciprocal().square()
                 + intrinsics[:, 1, 1].reciprocal().square()).rsqrt()
        values = {key: [] for key in OFFICIAL_GT_KEYS}
        for index in range(len(points)):
            p, gt, mask = points[index].float(), truth[index], valid[index]
            function, params = functions['gt_global']
            global_loss, _, optimal_scale, _ = function(p, gt, mask, **params)
            values['gt_global'].append(global_loss)
            optimal_scale = optimal_scale.detach()
            for key in ('gt_local4', 'gt_local16'):
                function, params = functions[key]
                value, _ = function(p, gt, mask, focal[index], optimal_scale, **params)
                values[key].append(value)
            function, params = functions['gt_edge']
            values['gt_edge'].append(function(p, gt, mask, **params)[0])
            function, params = functions['gt_scale']
            values['gt_scale'].append(function(raw['metric_scale'][index].float(),
                                                 optimal_scale, **params)[0])
        result = {key: torch.stack(value).mean() for key, value in values.items()}
        result['gt_valid_fraction'] = valid.float().mean()
        return result
