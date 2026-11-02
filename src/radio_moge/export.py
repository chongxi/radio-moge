import argparse

import torch
from torch import nn

from .adapters import build_adapter
from .checkpoint import atomic_save
from .decoder import MoGeDecoder, postprocess
from .upstream import load_moge


class GeometryHead(nn.Module):
    """Deployment module: adapter + decoder. No RADIO or MoGe encoder is owned here.

    ``indices`` lists the RADIO blocks whose intermediate maps the adapter consumes;
    the caller must request them in the same backbone pass. ``buckets`` holds the
    training (height, width) resolutions; inputs should use one of them.
    """

    def __init__(self, adapter, decoder, moge_source=None, buckets=None, precision=None):
        super().__init__()
        self.adapter, self.decoder = adapter, decoder
        self.moge_source = moge_source
        self.indices = tuple(getattr(adapter, "indices", ()))
        self.buckets = [tuple(b) for b in buckets] if buckets else None
        self.precision = precision

    def forward(self, radio_features, output_size):
        return self.decoder(self.adapter(radio_features), output_size)

    @torch.no_grad()
    def infer(self, radio_features, image):
        return postprocess(self(radio_features, image.shape[-2:]), image, self.moge_source)

    @classmethod
    def load(cls, path, device="cuda", moge_source=None):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        adapter = build_adapter(**artifact["adapter_config"])
        decoder = MoGeDecoder.from_config(artifact["model_config"], moge_source)
        model = cls(adapter, decoder, moge_source,
                    artifact.get("buckets"), artifact.get("precision"))
        model.load_state_dict(artifact["state_dict"], strict=True)
        model.artifact_info = {k: v for k, v in artifact.items() if k != "state_dict"}
        return model.eval().requires_grad_(False).to(device)


def export(checkpoint, output):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model, config = load_moge(state["config"]["models"], "cpu")
    if config != state["model_config"]:
        raise ValueError("MoGe configuration differs from the training checkpoint")
    adapter = build_adapter(**state["adapter_config"])
    adapter.load_state_dict(state["adapter"], strict=True)
    head = GeometryHead(adapter, MoGeDecoder(model))
    assert not any("encoder" in key or "backbone" in key for key in head.state_dict())
    training = state["config"]
    atomic_save(
        {
            "format_version": 2,
            "adapter_config": adapter.config,
            "model_config": config,
            "state_dict": head.state_dict(),
            "revisions": state["revisions"],
            "models": training["models"],
            "training_step": state["step"],
            "indices": list(head.indices),
            "buckets": training.get("data", {}).get("buckets"),
            "precision": training.get("train", {}).get("precision"),
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
