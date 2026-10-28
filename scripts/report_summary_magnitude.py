"""Plot a verified frozen summary-magnitude sweep report."""

import argparse
from pathlib import Path
import json
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
r = parser.parse_args().root
d = json.loads((r / "analysis.json").read_text())
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
names = {"day2_selected_156000": "Selected day 2 (156k)", "lr1e5_88000": "LR 1e-5 pilot (88k)"}
for name, label in names.items():
    rows = [x for x in d["rows"] if x["checkpoint"] == name]
    base = next(x for x in rows if x["multiplier"] == 1)
    xs = [x["multiplier"] for x in rows]
    axes[0].plot(xs, [x["gt_relative_change_pct"] for x in rows], marker="o", label=label)
    axes[1].plot(
        xs, [100 * (x["teacher_abs_rel"] / base["teacher_abs_rel"] - 1) for x in rows], marker="o"
    )
    axes[2].plot(xs, [x["metric_scale_ratio"] for x in rows], marker="o")
for ax, title, y, ref in zip(
    axes,
    ["NYU ground-truth error", "Teacher-agreement error", "Predicted metric scale"],
    ["AbsRel change (%)", "AbsRel change (%)", "Mean ratio to unmodified model"],
    [0, 0, 1],
):
    ax.set(title=title, xlabel="Summary magnitude multiplier", ylabel=y)
    ax.axhline(ref, color="grey", linestyle="--", linewidth=1)
    ax.axvline(1, color="grey", linestyle=":", linewidth=1)
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=8)
fig.suptitle(
    "Frozen summary magnitude sweep: direction and spatial features unchanged", fontsize=12
)
fig.text(
    0.5,
    0.01,
    "Exploratory reused NYU116 validation panel; no training, no deployed calibration, no independent test claim.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
tmp = r / "magnitude-sweep.tmp.png"
fig.savefig(tmp, dpi=170)
os.replace(tmp, r / "magnitude-sweep.png")
