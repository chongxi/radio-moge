"""Prespecified width/depth grid for the combined joint-attention adapter."""

from .architecture_screen import candidates


def capacity_candidates(shapes=None):
    shapes = shapes or ((4, 512), (4, 768), (4, 1024), (8, 512), (8, 768), (8, 1024))
    assert len(shapes) == len(set(shapes)) and all(
        b > 0 and w > 0 and w % 64 == 0 for b, w in shapes
    )
    base = next(c for c in candidates() if c["name"] == "joint_combined")
    return [
        dict(
            base,
            name=f"combined_{blocks}x{width}_s{seed}",
            blocks=blocks,
            width=width,
            heads=width // 64,
            seed=seed,
        )
        for seed in (42, 43)
        for blocks, width in shapes
    ]


def initialization_seed(spec, plan):
    return spec.get("seed", plan["seed"])
