"""Build exact observed-ray targets beside, never instead of, ordinary RGB.

Historical sidecar evidence binds the candidate. This CPU builder verifies its
member and image bytes; it is not a new training acceptance or calibration fit.
"""
from copy import deepcopy
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

from .data import image_rotation, ordinary_crop_parameters
from .geometry_cache import HEX64, read_geometry_member
from .geometry_sampling import camera_matrix, select_observations, source_supervision
from .ordinary_geometry import ORDINARY_LOCAL_RADIUS, ordinary_view
from .ordinary_geometry_data import image_identity
from .rgb_geometry_batch import GeometrySlot


def ordinary_target(slot, *, augment):
    """Return None or a target on the exact crop described by this RGB key.

    No depth interpolation, camera recentering, RGB warp or sampling decision.
    A local Random instance reproduces the original crop's draws. Geometry
    verification never consumes the model's RNG or clears source flags. Caller
    must pass the actual TrainingImages.augment setting; it is not inferred.
    """
    if (not isinstance(slot, GeometrySlot) or len(slot.key) != 4
            or any(type(v) is not int for v in slot.key) or min(slot.key[2:]) < 1
            or type(augment) is not bool):
        raise ValueError('Expected an original RGB slot and Boolean augmentation')
    if slot.geometry is None:
        return None
    ref, row = slot.geometry, slot.rgb_row
    entry = ref['entry']
    if (image_identity(row) != image_identity(entry['image'])
            or not HEX64.fullmatch(ref['pack']['sha256'])
            or not HEX64.fullmatch(ref['source_index']['sha256'])):
        raise ValueError('Ordinary target differs from its bound RGB identity')
    depth, valid, kd, metadata = read_geometry_member(ref['pack']['path'], entry)
    if (metadata['image_sha256'] != row['sha256'] or metadata['dataset'] != row['dataset']
            or image_rotation(row) != metadata.get('image_rotation_cw', 0)
            or metadata.get('depth_convention') != 'camera_z' or metadata.get('units') != 'meters'
            or metadata.get('camera_pixel_convention') != 'normalized_half_center'):
        raise ValueError('Ordinary target RGB, orientation or camera convention differs')
    mode = source_supervision(metadata)
    data = Path(row['image']).read_bytes()
    if hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('Ordinary target RGB bytes changed')
    with Image.open(io.BytesIO(data)) as image:
        # TrainingImages supports EXIF, but these source cameras were admitted
        # only for the explicit row rotation. Do not guess a new calibration.
        if image.getexif().get(274, 1) != 1:
            raise ValueError('Unbound EXIF orientation in geometry RGB')
        iw, ih = image.size
    if image_rotation(row) in (90, 270):
        ih, iw = iw, ih
    shape = (ih, iw)
    if ('height' in row or 'width' in row) and shape != (row.get('height'), row.get('width')):
        raise ValueError('Oriented geometry RGB dimensions differ')
    kd = camera_matrix(kd)
    if 'rgb_intrinsics_normalized' in metadata:
        kr = camera_matrix(metadata['rgb_intrinsics_normalized'])
        mapping = np.asarray(metadata.get('depth_to_rgb_uv_transform'), dtype=np.float64)
        if (list(shape) != metadata.get('rgb_shape') or mapping.shape != (3, 3)
                or not np.allclose(mapping, kr @ np.linalg.inv(kd), rtol=0, atol=1e-8)):
            raise ValueError('Unverified native-to-RGB camera mapping')
    else:
        if depth.shape != shape:
            raise ValueError('Unequal RGB/depth grids need explicit calibrated mapping')
        kr = kd
    _, seed, h, w = slot.key
    box, flip = ordinary_crop_parameters(shape, (h, w), seed, augment)
    view = ordinary_view(kr, shape, (h, w), box, flip)
    target = select_observations(depth, valid, kd, view)
    target.update(local_radius_contract=ORDINARY_LOCAL_RADIUS, supervision=mode,
                  label_kind=metadata['label_kind'], image_sha256=row['sha256'],
                  dataset=row['dataset'], cache_key=entry['key'], rgb_key=slot.key,
                  corpus_ordinal=ref['corpus_ordinal'], source_index=deepcopy(ref['source_index']),
                  pack=deepcopy(ref['pack']), member_sha256=entry['sha256'],
                  source_metadata=deepcopy(metadata), ordinary_view=view)
    return target
