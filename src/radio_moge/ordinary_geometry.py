"""Camera mathematics for the unchanged ordinary feature RGB crop.

This is a CPU preparation helper, not a new training or loss-admission path.
It never warps RGB, interpolates depth, fits calibration, or centers a camera.
The legacy centered-camera entry point does not admit this contract. The new
ordinary loss core and actual feature pixel/loss/gradient equivalence still
require a separate training gate before this view enters a trainer.
"""

import numpy as np

from .geometry_sampling import camera_matrix


ORDINARY_CAMERA_CONTRACT = 'ordinary_rgb_crop_v1'
ORDINARY_LOCAL_RADIUS = 'inverse_ray_jacobian_frobenius_v1'


def ordinary_view(k_rgb, rgb_shape, output_shape, crop_box, flip=False):
    """Describe the exact crop/mirror/resize performed by TrainingImages.

    Inputs describe the already-oriented RGB grid; do not rotate calibration a
    second time. ``crop_box`` uses PIL's integer boundary coordinates (l,t,r,b).
    Cameras use normalized pixel centers ((x+.5)/width, (y+.5)/height).

    Resizing changes pixel spacing, not normalized coordinates. Integer crop
    rounding may produce anisotropic pixels: preserve that, as well as skew and
    off-image principal points. A mirror reflects camera X and leaves Z/length
    unchanged; it is not a physical camera pose or a translation.
    """
    k_rgb = camera_matrix(k_rgb)
    if (len(rgb_shape) != 2 or len(output_shape) != 2 or len(crop_box) != 4
            or any(type(x) is not int or x < 1 for x in (*rgb_shape, *output_shape))
            or any(type(x) is not int for x in crop_box) or type(flip) is not bool):
        raise ValueError('Expected integer RGB/output shapes, integer crop box and Boolean flip')
    ih, iw = rgb_shape
    oh, ow = output_shape
    left, top, right, bottom = crop_box
    if not (0 <= left < right <= iw and 0 <= top < bottom <= ih):
        raise ValueError('Ordinary crop must be nonempty and inside the oriented RGB grid')
    cw, ch = right - left, bottom - top
    crop = np.array([[iw / cw, 0., -left / cw],
                     [0., ih / ch, -top / ch], [0., 0., 1.]])
    mirror = np.array([[-1., 0., 1.], [0., 1., 0.], [0., 0., 1.]]) if flip else np.eye(3)
    reflection = np.diag([-1., 1., 1.]) if flip else np.eye(3)
    homography = mirror @ crop
    # R^-1 == R for this coordinate reflection. Thus K' R K^-1 == H,
    # with positive focal lengths and correctly signed skew after mirroring.
    intrinsics = camera_matrix(homography @ k_rgb @ reflection)
    return dict(camera_contract=ORDINARY_CAMERA_CONTRACT,
                camera_pixel_convention='normalized_half_center',
                rgb_shape=(ih, iw), crop_box=tuple(crop_box), output_shape=(oh, ow),
                flipped=flip, intrinsics=intrinsics, camera_transform=reflection,
                rgb_to_view_uv=homography)
