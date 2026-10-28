from dataclasses import dataclass

import torch

from .adapters import EncoderFeatures, build_adapter
from .decoder import MoGeDecoder
from .upstream import extract_radio, load_moge, load_radio, radio_input


@dataclass
class FrozenSystem:
    radio: torch.nn.Module
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    model_config: dict

    @torch.no_grad()
    def encode(self, image, indices=(), num_tokens=None):
        radio_image = radio_input(image, self.radio, num_tokens)
        features = extract_radio(self.radio, radio_image, indices)
        h, w = features.spatial.shape[-2:]
        spatial, summary = self.encoder(image, h, w, return_class_token=True)
        return features, EncoderFeatures(spatial.float(), summary.float())


def build_system(config, device):
    model, model_config = load_moge(config["models"], device)
    system = FrozenSystem(
        load_radio(config["models"], device), model.encoder, MoGeDecoder(model), model_config
    )
    if not all(
        hasattr(system.decoder, name) for name in ("points_head", "mask_head", "scale_head")
    ):
        raise ValueError(
            "This trainer requires a MoGe-2 teacher with points, mask and metric scale"
        )
    adapter = build_adapter(
        spatial_dim=system.radio.embed_dim,
        summary_dim=system.radio.summary_dim,
        target_dim=model.encoder.output_projections[0].out_channels,
        target_summary_dim=model.encoder.dim_features,
        **config["adapter"],
    ).to(device)
    for h, w in config["data"]["buckets"]:
        if h % system.radio.min_resolution_step or w % system.radio.min_resolution_step:
            raise ValueError("Training buckets must respect RADIO patch/window resolution step")
    return system, adapter
