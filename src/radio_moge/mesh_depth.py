"""Point-sampled CUDA mesh depth, with explicit OpenCV/GL conventions.

This creates laser-*derived* labels, not direct per-pixel sensor measurements.
Never antialias depth or fill missing triangles. Importing does not touch a GPU.
"""
import numpy as np
import torch


def conservative_visibility(depth, valid, shifted):
    """Remove unsupported/unstable neighborhoods and subpixel visibility changes.

    This is a *selection mask*, not repaired depth. It uses only rasterizer
    outputs so independent CPU intersections remain a genuine parity check.
    Triangle-ID erosion is deliberately avoided: tiny coplanar triangles are
    not occlusions. Four +/-quarter-pixel diagonal samples are required.
    """
    from scipy.ndimage import binary_erosion, maximum_filter, minimum_filter

    depth, valid = np.asarray(depth), np.asarray(valid)
    if (depth.ndim != 2 or valid.shape != depth.shape or valid.dtype != np.bool_
            or len(shifted) != 4 or not np.isfinite(depth[valid]).all()):
        raise ValueError('Finite center depth/support and four offset renders required')
    safe = np.where(valid, depth, 0)
    support = binary_erosion(valid, iterations=2)
    support &= maximum_filter(safe, 5) - minimum_filter(safe, 5) <= .02 + .02 * safe
    for other, other_valid in shifted:
        other, other_valid = np.asarray(other), np.asarray(other_valid)
        if other.shape != depth.shape or other_valid.shape != depth.shape or other_valid.dtype != np.bool_:
            raise ValueError('Offset renders must retain the original pixel grid')
        support &= other_valid & np.isfinite(other) & (np.abs(other - safe) <= .005 + .01 * safe)
    return support


class CudaMeshDepth:
    def __init__(self, vertices, faces, device='cuda', near=.01, far=100.):
        import nvdiffrast.torch as dr
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu'
                or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)
                or not 0 < near < far):
            raise ValueError('Invalid finite mesh or clip bounds')
        self.dr, self.device, self.near, self.far = dr, device, near, far
        self.context = dr.RasterizeCudaContext(device=device)
        self.vertices = torch.as_tensor(np.array(vertices, np.float32), device=device)
        self.faces = torch.as_tensor(np.array(faces, np.int32), device=device)

    @torch.inference_mode()
    def render(self, intrinsics, c2w, height, width):
        k, pose = np.asarray(intrinsics, np.float64), np.asarray(c2w, np.float64)
        if (k.shape != (3, 3) or not np.isfinite(k).all()
                or not np.allclose(k[2], [0, 0, 1]) or k[0, 0] <= 0 or k[1, 1] <= 0
                or pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or np.linalg.det(pose[:3, :3]) < .9999 or min(height, width) < 1):
            raise ValueError('Finite normalized pinhole calibration and rigid pose required')
        k = torch.tensor(k, device=self.device, dtype=torch.float32)
        pose = torch.tensor(pose, device=self.device, dtype=torch.float32)
        cam = (self.vertices - pose[:3, 3]) @ pose[:3, :3]
        uvz = cam @ k.T
        z = cam[:, 2]
        a, b = ((self.far + self.near) / (self.far - self.near),
                -2 * self.far * self.near / (self.far - self.near))
        clip = torch.stack((2 * uvz[:, 0] - z, z - 2 * uvz[:, 1], a * z + b, z), -1)
        rast, _ = self.dr.rasterize(self.context, clip[None].contiguous(), self.faces,
                                    resolution=(height, width), grad_db=False)
        depth, _ = self.dr.interpolate(cam[:, 2:3][None].contiguous(), rast, self.faces)
        # Nvdiffrast's first array row is the bottom of the GL image.
        depth = depth[0, :, :, 0].flip(0)
        triangles = rast[0, :, :, 3].flip(0).to(torch.int64) - 1
        valid = ((triangles >= 0) & torch.isfinite(depth)
                 & (depth > self.near) & (depth < self.far))
        depth = torch.where(valid, depth, 0)
        return depth.cpu().numpy(), valid.cpu().numpy(), triangles.cpu().numpy()

    def render_conservative(self, intrinsics, c2w, height, width):
        depth, valid, triangles = self.render(intrinsics, c2w, height, width)
        shifted = []
        for dx, dy in ((-.25, -.25), (-.25, .25), (.25, -.25), (.25, .25)):
            k = np.array(intrinsics, dtype=np.float64, copy=True)
            # Moving K's center opposite the sample offset evaluates that ray.
            k[0, 2] -= dx / width
            k[1, 2] -= dy / height
            other, other_valid, _ = self.render(k, c2w, height, width)
            shifted.append((other, other_valid))
        support = conservative_visibility(depth, valid, shifted)
        quality = dict(initial_valid_pixels=int(valid.sum()), retained_pixels=int(support.sum()),
            retained_fraction=float(support.mean()), depth_filled_or_averaged=False,
            visibility_policy='5x5 stable (.02+.02Z m); four quarter-pixel rays (.005+.01Z m)')
        return np.where(support, depth, 0), support, triangles, quality
