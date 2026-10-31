"""Optional W&B logging. Authentication is owned by the SDK, never the config."""

from contextlib import contextmanager
import json
from pathlib import Path
import secrets

from .checkpoint import atomic_json, restore_rng, rng_state


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


class ExperimentTracker:
    def __init__(self, config, output, resume=False, provenance=None):
        self.run = None
        settings = config.get("tracking", {})
        if not settings.get("enabled", False):
            return
        import wandb

        self.sdk = wandb
        self.image_count = settings.get("images", 4)
        mode = settings.get("mode", "online")
        if mode not in {"online", "offline"}:
            raise ValueError("Tracking mode must be online or offline")
        output = Path(output)
        identity_path = output / "wandb-run.json"
        previous = json.loads(identity_path.read_text()) if resume and identity_path.exists() else None
        if resume and previous is None:
            raise ValueError("W&B resume needs wandb-run.json from the original run directory")
        if previous and previous["mode"] != mode:
            raise ValueError("Cannot change W&B mode during resume")
        with preserve_rng():
            self.run = wandb.init(
                project=previous["project"] if previous else settings.get("project", "radio-moge"),
                entity=previous["entity"] if previous else settings.get("entity"),
                id=previous["id"] if previous else secrets.token_hex(4),
                resume="must" if previous and mode == "online" else None,
                name=settings.get("name", output.name),
                group=settings.get("group"),
                tags=settings.get("tags", []),
                mode=mode,
                dir=str(output),
                config={"experiment": config, "provenance": provenance or {}},
                settings=wandb.Settings(init_timeout=60, save_code=False, disable_git=True),
            )
        self.run.define_metric("optimizer_step")
        for namespace in ("train/*", "validation/*", "train_probe/*"):
            self.run.define_metric(namespace, step_metric="optimizer_step")
        atomic_json(
            {"id": self.run.id, "entity": self.run.entity, "project": self.run.project,
             "url": self.run.url, "mode": mode},
            identity_path,
        )
        print(f"W&B run: {self.run.url or 'offline (sync later)'}", flush=True)

    def log(self, namespace, values, step, image_dir=None):
        if self.run is None:
            return
        payload = {"optimizer_step": step}
        payload.update({f"{namespace}/{k}": v for k, v in values.items() if k != "step"})
        with preserve_rng():
            if image_dir is not None:
                files = sorted(Path(image_dir).glob("*.png"))[:self.image_count]
                if files:
                    payload[f"{namespace}/decoded_depth"] = [
                        self.sdk.Image(str(path), caption=(
                            f"Step {step}, validation sample {path.stem}: "
                            "RGB | MoGe-2 depth | RADIO adapter depth | relative error. "
                            "Teacher/student use the same depth range."
                        )) for path in files
                    ]
            self.run.log(payload)

    def record_lr_adjustments(self, history):
        if self.run is not None:
            with preserve_rng():
                self.run.config.update({'learning_rate_adjustments': history}, allow_val_change=True)

    def finish(self, success=True):
        if self.run is not None:
            self.run.finish(exit_code=0 if success else 1)
