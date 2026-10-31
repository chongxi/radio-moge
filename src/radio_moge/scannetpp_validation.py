"""CPU-only camera/frame alignment for a separate ScanNet++ LiDAR validation set."""
import hashlib
from pathlib import Path
import re

import cv2
import numpy as np


def internal_holdout(scene):
    if not re.fullmatch(r'[0-9a-f]{10}', scene):
        raise ValueError('Invalid scene identifier')
    return int(hashlib.sha256(('scannetpp-internal-val-v1:' + scene).encode())
               .hexdigest()[:8], 16) % 20 == 0


def registered_frames(camera_path, image_path):
    cameras, frames = {}, {}
    for line in Path(camera_path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 12 or fields[1] != 'OPENCV':
            raise ValueError('Expected OPENCV camera with fx fy cx cy k1 k2 p1 p2')
        key = int(fields[0])
        width, height = map(int, fields[2:4])
        values = np.asarray(fields[4:], dtype=float)
        if (key in cameras or (width, height) != (1920, 1440)
                or not np.isfinite(values).all() or np.any(values[:2] <= 0)):
            raise ValueError('Invalid or duplicate iPhone camera')
        cameras[key] = dict(width=width, height=height, parameters=values.tolist())
    with Path(image_path).open() as file:
        for line in file:
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) != 10:
                raise ValueError('Invalid COLMAP image header')
            name = Path(fields[9]).name
            match = re.fullmatch(r'frame_(\d{6})\.jpg', name)
            if match is None:
                raise ValueError('Unexpected registered frame name')
            index, camera = int(match[1]), int(fields[8])
            pose = np.asarray(fields[1:8], dtype=float)
            if (index in frames or camera not in cameras or not np.isfinite(pose).all()
                    or not np.isclose(np.linalg.norm(pose[:4]), 1, atol=1e-4)):
                raise ValueError('Invalid frame identity or camera pose')
            frames[index] = dict(frame_index=index, camera=cameras[camera],
                                 camera_id=camera, colmap_qvec_tvec=pose.tolist())
            if next(file, None) is None:
                raise ValueError('Truncated COLMAP image points line')
    if not frames:
        raise ValueError('No registered iPhone frames')
    return frames


def select_spaced_frames(frames, count=4):
    indices = sorted(frames)
    if len(indices) < count or count < 1:
        raise ValueError('Insufficient registered frames')
    # Fixed temporal strata, independent of model errors or image content.
    positions = [min(len(indices)-1, int((k+.5)*len(indices)/count)) for k in range(count)]
    return [indices[k] for k in positions]


def read_video_frame(capture, index):
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
        raise ValueError('Video backend refused exact frame seek')
    if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - index) > .01:
        raise ValueError('Video frame position differs before read')
    ok, frame = capture.read()
    if not ok or frame is None or abs(capture.get(cv2.CAP_PROP_POS_FRAMES)-(index+1)) > .01:
        raise ValueError('Video frame position differs after read')
    return frame


def aligned_view(rgb_bgr, mask_bgr, depth, camera, width=768, height=576):
    """Use one distortion map for RGB, mask and native aligned LiDAR coordinates.

    RGB-to-depth coordinates use the official scaled-intrinsic convention.
    Depth is nearest-neighbor sampled in metres, never interpolated or fitted to
    predictions. All out-of-image, anonymous and invalid sensor pixels become 0.
    """
    if rgb_bgr.shape != (1440, 1920, 3) or mask_bgr.shape != rgb_bgr.shape:
        raise ValueError('RGB/mask dimensions or channels differ from camera')
    if depth.shape != (192, 256) or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError('Invalid native LiDAR depth')
    if (camera['width'], camera['height']) != (1920, 1440):
        raise ValueError('Camera dimensions differ')
    if (width, height) != (768, 576):
        raise ValueError('Versioned validation output must be 768x576')
    if not (np.array_equal(mask_bgr[..., 0], mask_bgr[..., 1])
            and np.array_equal(mask_bgr[..., 0], mask_bgr[..., 2])):
        raise ValueError('Anonymization mask is not grayscale')
    mask = mask_bgr[..., 0]
    if not np.isin(mask, [0, 255]).all():
        raise ValueError('Expected lossless binary anonymization mask')
    fx, fy, cx, cy, *distortion = camera['parameters']
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    output_k = k.copy()
    output_k[:2] *= width / 1920
    mx, my = cv2.initUndistortRectifyMap(k, np.array(distortion), None, output_k,
                                        (width, height), cv2.CV_32FC1)
    rgb = cv2.remap(rgb_bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # Require the interpolation neighborhood to be valid, including image bounds.
    safe = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT,
                     borderValue=0)
    valid_rgb = cv2.remap(safe, mx, my, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT) == 255
    sampled = cv2.remap(depth.astype(np.float32), mx * (256/1920), my * (192/1440),
                        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    sampled[~valid_rgb] = 0
    rgb[~valid_rgb] = 0
    valid = sampled > 0
    proof = dict(output_intrinsic=output_k.tolist(), output_width=width,
                 output_height=height, native_rgb_hw=[1440, 1920], native_depth_hw=[192, 256],
                 rgb_valid_fraction=float(valid_rgb.mean()), depth_valid_fraction=float(valid.mean()),
                 depth_units='metres', depth_interpolation='nearest', distortion_removed=True,
                 rgb_depth_coordinate_scale=[256/1920, 192/1440],
                 depth_invalid_value=0, metric_scale_fitting=False)
    return rgb, sampled, proof
