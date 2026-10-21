"""MoGe-2 decoder boundary, retaining autograd to its feature inputs.

The forward layout follows microsoft/MoGe at the revision in config.py.
Postprocessing delegates to that exact upstream implementation.
"""

import torch
import torch.nn.functional as F
from torch import nn


def view_plane_uv(height, width, aspect_ratio, device, dtype):
    span_y = (1 + aspect_ratio**2) ** -0.5
    span_x = aspect_ratio * span_y
    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        device=device,
        dtype=dtype,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        device=device,
        dtype=dtype,
    )
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    return torch.stack((uu, vv), dim=0)


class MoGeDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        for name in ("neck", "points_head", "mask_head", "normal_head", "scale_head"):
            if hasattr(model, name):
                setattr(self, name, getattr(model, name))
        self.remap_output = model.remap_output
        self.eval().requires_grad_(False)

    @classmethod
    def from_config(cls, config, moge_source=None):
        # Allocate only decoder modules when loading a deployment artifact.
        from .upstream import import_moge

        import_moge(moge_source)
        from moge.model.modules.conv_stack import ConvStack
        from moge.model.modules.mlp import MLP

        model = nn.Module()
        model.remap_output = config.get("remap_output", "linear")
        model.neck = ConvStack(**config["neck"])
        for name in ("points_head", "mask_head", "normal_head"):
            if config.get(name) is not None:
                head = ConvStack(**config[name])
                if name != "mask_head":
                    head.fp32_output_projection = True
                setattr(model, name, head)
        if config.get("scale_head") is not None:
            model.scale_head = MLP(**config["scale_head"])
        return cls(model)

    def forward(self, encoded, output_size):
        # MoGe's official mixed-precision path keeps its neck and heads in FP32.
        with torch.autocast(encoded.spatial.device.type, enabled=False):
            spatial, summary = encoded.spatial.float(), encoded.summary.float()
            batch, _, h, w = spatial.shape
            out_h, out_w = output_size
            features = []
            for level in range(5):
                uv = view_plane_uv(
                    h * 2**level, w * 2**level, out_w / out_h, spatial.device, spatial.dtype
                )
                uv = uv.unsqueeze(0).expand(batch, -1, -1, -1)
                features.append(torch.cat((spatial, uv), 1) if level == 0 else uv)
            features = self.neck(features)
            result = {}
            for key in ("points", "normal", "mask"):
                head = getattr(self, f"{key}_head", None)
                if head is None:
                    continue
                value = F.interpolate(
                    head(features)[-1],
                    output_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
                if key == "mask":
                    value = value.squeeze(1).sigmoid()
                else:
                    value = value.permute(0, 2, 3, 1)
                    if key == "normal":
                        value = F.normalize(value, dim=-1)
                    elif self.remap_output == "sinh":
                        value = value.sinh()
                    elif self.remap_output == "exp":
                        xy, z = value.split([2, 1], -1)
                        value = torch.cat((xy * z.exp(), z.exp()), -1)
                    elif self.remap_output == "sinh_exp":
                        xy, z = value.split([2, 1], -1)
                        value = torch.cat((xy.sinh(), z.exp()), -1)
                    elif self.remap_output != "linear":
                        raise ValueError(f"Unknown point remapping: {self.remap_output}")
                result[key] = value
            if hasattr(self, "scale_head"):
                result["metric_scale"] = self.scale_head(summary).squeeze(1).exp()
            return result


@torch.no_grad()
def postprocess(raw, image, moge_source=None):
    """Run the official focal recovery, projection and metric scaling with no encoder.

    The tiny forward facade supplies already decoded predictions to upstream infer().
    Clones are necessary because infer() modifies its point map in-place.
    """
    from .upstream import import_moge

    cls = import_moge(moge_source)

    class DecodedPredictions:
        device = image.device
        dtype = torch.float32

        def forward(self, image, num_tokens):
            return {key: value.clone() for key, value in raw.items()}

    return cls.infer(
        DecodedPredictions(),
        image,
        num_tokens=1,
        use_fp16=False,
        force_projection=True,
        apply_mask=False,
    )
