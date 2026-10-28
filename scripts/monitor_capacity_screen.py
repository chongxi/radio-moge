"""Reuse the read-only evidence monitor with the separate capacity experiment handoff."""

import argparse
from pathlib import Path

from monitor_architecture_screen import monitor
from render_capacity_screen import render


NAME = "adapter-combined-capacity-screen-20261028"

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--remote-base", type=Path, required=True)
    a = p.parse_args()
    monitor(
        a.base,
        a.remote_base,
        name=NAME,
        handoff_key="active_adapter_capacity_screen",
        render_fn=render,
    )
