"""Construct controls that separate objective, data-mixture and architecture effects."""

from copy import deepcopy


GEOMETRY_TERMS = ("points", "mask", "scale", "gradient", "normal")


def feature_continuation(config, output):
    """Keep the geometry run's schedule and sampling; remove geometric supervision."""
    result = deepcopy(config)
    result["output_dir"] = str(output)
    for term in GEOMETRY_TERMS:
        result["loss"][term] = 0.0
    result["tracking"].update(
        name="indoor-feature-only-matched-continuation",
        group="geometry-objective-control-20261021",
        tags=["objective-control", "feature-only", "matched-10000"],
    )
    return result


def add_indoor_control(configs, indoor_manifest, root):
    """Hold the head, schedules and common validation fixed; change training mixture."""
    for stage in ("features", "geometry"):
        config = deepcopy(configs[("fusion", stage)])
        weights = config["data"]["mixture_weights"]
        indoor = {key: weights[key] for key in ("arkitscenes", "hypersim")}
        total = sum(indoor.values())
        config["data"].update(
            train_manifest=str(indoor_manifest),
            mixture_weights={key: value / total for key, value in indoor.items()},
        )
        config["output_dir"] = str(root / "fusion_indoor" / stage / "train")
        config["tracking"].update(
            name=f"indoor-control-fusion-{stage}",
            tags=["data-control", "indoor-only", "fusion", stage, "screen-6000-6000"],
        )
        configs[("fusion_indoor", stage)] = config
    return configs
