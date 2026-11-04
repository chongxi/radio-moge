"""Point-sampled CUDA mesh depth, with explicit OpenCV/GL conventions.

This creates laser-*derived* labels, not direct per-pixel sensor measurements.
Never antialias depth or fill missing triangles. Importing does not touch a GPU.
"""
import numpy as np
import torch

from radio_moge.geometry_sources import pinhole_rays


def verify_cpu_intersections(intersector, intrinsics, c2w, candidate_depth, candidate_valid):
    """Verify EVERY candidate ray, using nearest CPU mesh hits, not a sample.

    GPU visibility alone did not meet the strict 2mm parity requirement. Only
    mutually supported hits within that unchanged tolerance survive. Cache the
    CPU intersection's Z, not interpolated/averaged or filled depth. This checks
    software visibility, not independent physical accuracy of the source mesh.
    Reuse one intersector/BVH per scene.
    """
    depth, valid = np.asarray(candidate_depth), np.asarray(candidate_valid)
    pose = np.asarray(c2w, np.float64)
    if (depth.ndim != 2 or valid.shape != depth.shape or valid.dtype != np.bool_
            or not np.isfinite(depth[valid]).all() or not (depth[valid] > 0).all()
            or pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(pose[:3, :3]) < .9999):
        raise ValueError('Valid candidate depth/support and calibrated rigid pose required')
    indices = np.flatnonzero(valid)
    result, support = np.zeros(depth.size, np.float32), np.zeros(depth.size, bool)
    checked, mismatched, max_error = 0, 0, None
    if len(indices):
        rays = pinhole_rays(intrinsics, *depth.shape).reshape(-1, 3)[indices] @ pose[:3, :3].T
        origins = np.broadcast_to(pose[:3, 3], rays.shape).copy()
        points, hits, _ = intersector.intersects_location(origins, rays, multiple_hits=False)
        if len(np.unique(hits)) != len(hits):
            raise ValueError('Nearest-hit query returned duplicate pixel support')
        z = ((points - pose[:3, 3]) @ pose[:3, :3])[:, 2]
        error = np.abs(z - depth.ravel()[indices[hits]])
        good = np.isfinite(z) & (z > .01) & (z < 100) & (error <= .002)
        support[indices[hits[good]]] = True
        result[indices[hits[good]]] = z[good]
        checked, mismatched = len(hits), int((~good).sum())
        max_error = float(error[good].max()) if good.any() else None
    quality = dict(cpu_rays_checked=len(indices), cpu_hits=checked,
        cpu_missing_hits=len(indices) - checked, rejected_inconsistent_hits=mismatched,
        verified_pixels=int(support.sum()), maximum_accepted_gpu_cpu_error_m=max_error,
        maximum_allowed_gpu_cpu_error_m=.002, final_depth='CPU nearest-hit camera Z',
        independent_physical_depth_accuracy_checked=False)
    return result.reshape(depth.shape), support.reshape(depth.shape), quality


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
    def __init__(self, vertices, faces, device='cuda', near=None, far=None,
                 max_faces_per_pass=1048576, *, clip_contract=None):
        if clip_contract is not None:
            from radio_moge.mesh_clip import MeshClipContract
            if not isinstance(clip_contract, MeshClipContract) or near is not None or far is not None:
                raise ValueError('Use one explicit clip contract without separate bounds')
            near, far = clip_contract.near_m, clip_contract.far_m
        else:
            near, far = .01 if near is None else near, 100. if far is None else far
        import nvdiffrast.torch as dr
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu'
                or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)
                or not np.isfinite([near, far]).all() or not 0 < near < far
                or type(max_faces_per_pass) is not int
                or not 1 <= max_faces_per_pass <= 1048576):
            raise ValueError('Invalid finite mesh or clip bounds')
        self.dr, self.device, self.near, self.far = dr, device, near, far
        self.clip_contract = clip_contract
        self.max_faces_per_pass = max_faces_per_pass
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
        # Bound rasterizer work/triangle IDs for very large source meshes.
        # Every triangle is rendered. Merge by nearest camera Z, preserving
        # cross-chunk occlusion; never average depths or discard a mesh subset.
        depth = torch.full((height, width), float('inf'), device=self.device)
        triangles = torch.full((height, width), -1, dtype=torch.int64, device=self.device)
        positions, attributes = clip[None].contiguous(), cam[:, 2:3][None].contiguous()
        for start in range(0, len(self.faces), self.max_faces_per_pass):
            faces = self.faces[start:start + self.max_faces_per_pass]
            rast, _ = self.dr.rasterize(self.context, positions, faces,
                                        resolution=(height, width), grad_db=False)
            value, _ = self.dr.interpolate(attributes, rast, faces)
            # Nvdiffrast's first array row is the bottom of the GL image.
            value = value[0, :, :, 0].flip(0)
            face_ids = rast[0, :, :, 3].flip(0).to(torch.int64) - 1
            good = ((face_ids >= 0) & torch.isfinite(value)
                    & (value > self.near) & (value < self.far) & (value < depth))
            depth = torch.where(good, value, depth)
            triangles = torch.where(good, face_ids + start, triangles)
        valid = triangles >= 0
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
            mesh_faces=len(self.faces), max_faces_per_pass=self.max_faces_per_pass,
            raster_passes_per_ray_grid=(len(self.faces) + self.max_faces_per_pass - 1) // self.max_faces_per_pass,
            visibility_policy='5x5 stable (.02+.02Z m); four quarter-pixel rays (.005+.01Z m)')
        if self.clip_contract is not None:
            quality['clip_contract'] = self.clip_contract.as_dict()
        return np.where(support, depth, 0), support, triangles, quality
