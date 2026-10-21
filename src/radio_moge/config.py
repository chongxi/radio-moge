from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RADIO_COMMIT = "4ffaf15d0ff48197c194e306772bdb5fb92db230"
MOGE_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"
RADIO_REVISION = "0057b339059c0b9e1b4ba996f975410ebbfdfcc8"
MOGE_REVISION = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if "extends" in config:
        base = load_config(path.parent / config.pop("extends"))
        config = merge(base, config)
    return config


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
