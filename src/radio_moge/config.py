from copy import deepcopy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RADIO_COMMIT = "4ffaf15d0ff48197c194e306772bdb5fb92db230"
MOGE_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"
RADIO_REVISION = "0057b339059c0b9e1b4ba996f975410ebbfdfcc8"
MOGE_REPO = "Ruicheng/moge-2-vitl-normal"
MOGE_REVISION = "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"


def load_config(path):
    path = Path(path).resolve()
    # JSON scientific notation such as 3e-05 is a string under PyYAML's YAML 1.1
    # resolver. Preserve JSON number types in generated experiment configs.
    config = (json.loads(path.read_text()) if path.suffix == '.json'
              else yaml.safe_load(path.read_text()))
    if "extends" in config:
        base = load_config(path.parent / config.pop("extends"))
        config = merge(base, config)
    # Save the chosen teacher and immutable weight revision with every run config.
    # Evaluation/export then use the same release even if defaults change later.
    if "models" in config:
        config["models"].setdefault("moge_repo", MOGE_REPO)
        config["models"].setdefault("moge_revision", MOGE_REVISION)
    return config


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
