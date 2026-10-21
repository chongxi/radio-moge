import argparse

import torch
from torch import nn

from .adapters import GeometryAdapter
from .checkpoint import atomic_save
from .decoder import MoGeDecoder, postprocess
from .upstream import load_moge


class GeometryHead(nn.Module):
    """Deployment module: adapter + decoder. No RADIO or MoGe encoder is owned here."""

    def __init__(self, adapter, decoder, moge_source=None):
        super().__init__()
        self.adapter, self.decoder = adapter, decoder
        self.moge_source = moge_source

    def forward(self, radio_features, output_size):
        return self.decoder(self.adapter(radio_features), output_size)

    @torch.no_grad()
    def infer(self, radio_features, image):
        return postprocess(self(radio_features, image.shape[-2:]), image, self.moge_source)

    @classmethod
    def load(cls, path, device="cuda", moge_source=None):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        adapter = GeometryAdapter(**artifact["adapter_config"])
        decoder = MoGeDecoder.from_config(artifact["model_config"], moge_source)
        model = cls(adapter, decoder, moge_source)
        model.load_state_dict(artifact["state_dict"], strict=True)
        return model.eval().requires_grad_(False).to(device)


def export(checkpoint, output):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model, config = load_moge(state["config"]["models"], "cpu")
    if config != state["model_config"]:
        raise ValueError("MoGe configuration differs from the training checkpoint")
    adapter = GeometryAdapter(**state["adapter_config"])
    adapter.load_state_dict(state["adapter"])
    head = GeometryHead(adapter, MoGeDecoder(model))
    assert not any("encoder" in key or "backbone" in key for key in head.state_dict())
    atomic_save(
        {
            "format_version": 1,
            "adapter_config": adapter.config,
            "model_config": config,
            "state_dict": head.state_dict(),
            "revisions": state["revisions"],
            "models": state["config"]["models"],
            "training_step": state["step"],
        },
        output,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export adapter + decoder, excluding both backbones"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
