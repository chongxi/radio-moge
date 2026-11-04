"""Explicit camera dispatch; legacy centered and ordinary crop stay separate."""
import torch

from .ordinary_geometry import ORDINARY_CAMERA_CONTRACT, ORDINARY_LOCAL_RADIUS


def loss_focal(target, ordinary=False):
    k = target['intrinsics'].detach().float()
    if not ordinary:
        if (target.get('camera_contract') is not None or k.shape != (3, 3)
                or not torch.isfinite(k).all()
                or not torch.allclose(k[:2, 2], k.new_tensor([.5, .5]), atol=1e-6, rtol=0)
                or not (torch.diag(k) > 0).all()):
            raise ValueError('Loss requires the verified centered camera')
    else:
        if (target.get('camera_contract') != ORDINARY_CAMERA_CONTRACT
                or target.get('camera_pixel_convention') != 'normalized_half_center'
                or target.get('local_radius_contract') != ORDINARY_LOCAL_RADIUS
                or k.shape != (3, 3) or not torch.isfinite(k).all()
                or not torch.equal(k[2], k.new_tensor([0., 0., 1.]))
                or not (torch.diag(k) > 0).all() or torch.linalg.det(k[:2, :2]) <= 1e-12):
            raise ValueError('Ordinary loss requires its explicit pinhole/radius contract')
        # Fail if target XYZ and query UV silently refer to different cameras.
        mask = target['valid']
        xyz = target['points'].detach().float()[mask]
        q = xyz @ k.T
        if not torch.allclose(q[:, :2] / q[:, 2:], target['query_uv'][mask].float(),
                              rtol=0, atol=2e-6):
            raise ValueError('Ordinary target rays differ from the declared camera')
    if not ordinary or (k[0, 1] == 0 and k[1, 0] == 0):
        # Preserve the exact legacy expression for the zero-skew special case.
        focal = (k[0, 0].reciprocal().square() + k[1, 1].reciprocal().square()).rsqrt()
    else:
        inverse = torch.linalg.inv(k[:2, :2])
        focal = inverse.square().sum().rsqrt()
    if ordinary and (not torch.isfinite(focal) or focal <= 0):
        raise ValueError('Invalid ordinary local-radius focal')
    return focal
