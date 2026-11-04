"""Optional CPU geometry references carried alongside unchanged RGB tensors.

No native arrays are copied to CUDA here. These containers preserve image slots,
including missing targets; they do not authorize a trainer or define a loss.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GeometrySlot:
    key: tuple
    rgb_row: dict
    geometry: dict | None
    augment: bool | None = None  # Bound by the actual TrainingImages wrapper, never inferred from K.


@dataclass(frozen=True)
class RGBGeometryBatch:
    image: torch.Tensor
    slots: tuple[GeometrySlot, ...]

    def __post_init__(self):
        if (not isinstance(self.image, torch.Tensor) or self.image.ndim != 4
                or not isinstance(self.slots, tuple) or len(self.image) != len(self.slots)
                or any(not isinstance(s, GeometrySlot) for s in self.slots)):
            raise ValueError('Expected one geometry slot for every batched RGB image')
        if any(tuple(s.key[2:]) != tuple(self.image.shape[-2:]) for s in self.slots):
            raise ValueError('RGB tensor resolution differs from its original sample keys')

    def __len__(self):
        return len(self.image)

    @property
    def shape(self):
        return self.image.shape

    @property
    def dtype(self):
        return self.image.dtype

    def with_image(self, image):
        if image.shape != self.image.shape:
            raise ValueError('RGB transport cannot change sample count or image grid')
        return RGBGeometryBatch(image, self.slots)

    def is_pinned(self):
        return self.image.is_pinned()

    def pin_memory(self):
        return self.with_image(self.image.pin_memory())

    def to(self, *args, **kwargs):
        return self.with_image(self.image.to(*args, **kwargs))

    def record_stream(self, stream):
        self.image.record_stream(stream)


def concatenate_rgb(images):
    """Use the original torch.cat operation, carrying the same slot permutation."""
    annotated = [isinstance(x, RGBGeometryBatch) for x in images]
    if any(annotated):
        if not all(annotated):
            raise ValueError('Cannot mix annotated and unannotated RGB batches')
        return RGBGeometryBatch(torch.cat([x.image for x in images], dim=0),
                                tuple(s for x in images for s in x.slots))
    return torch.cat(images, dim=0)
