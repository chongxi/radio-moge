"""A geometry head consuming shared RADIO features, with no second backbone."""

__version__ = "0.2.0"

__all__ = ["RadioMoGe", "PointCloudPipeline", "PointCloudResult", "RadioFeatures"]


def __getattr__(name):
    # Keep a plain import lightweight; model/source loading is always explicit.
    if name == "RadioMoGe":
        from .deployment import RadioMoGe
        return RadioMoGe
    if name in ("PointCloudPipeline", "PointCloudResult"):
        from .pipeline import PointCloudPipeline, PointCloudResult
        return {"PointCloudPipeline": PointCloudPipeline, "PointCloudResult": PointCloudResult}[name]
    if name == "RadioFeatures":
        from .adapters import RadioFeatures
        return RadioFeatures
    raise AttributeError(name)
