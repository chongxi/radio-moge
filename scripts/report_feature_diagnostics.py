"""Export paired frozen feature diagnostics as CSV, JSON and a scientific figure."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write(path, data):
    temporary = path.with_name(path.name + f".new-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    initial = json.loads((args.initial / "summary.json").read_text())
    factor = json.loads((args.factor / "summary.json").read_text())
    assert initial["stage"] == factor["stage"] == "passed"
    assert initial["images"] == factor["images"] == 116
    for name, result in factor["checkpoints"].items():
        for metric, value in initial["checkpoints"][name]["means"].items():
            assert abs(result["means"][metric] - value) < 1e-10, (name, metric)
    merged = {**initial["checkpoints"], **factor["checkpoints"]}
    comparisons = []
    features = []
    for name, result in merged.items():
        means = result["means"]
        features.append({"checkpoint": name, **{k: means[k] for k in (
            "spatial/cosine", "summary/cosine", "spatial/norm_ratio", "summary/norm_ratio",
            "spatial/abs_log_norm_ratio", "summary/abs_log_norm_ratio", "scale_log_error")}})
        for variant in ["student", *result["paired_interventions"]]:
            row = {"checkpoint": name, "variant": variant,
                   "NYU_AbsRel": means[f"{variant}/gt/abs_rel"],
                   "teacher_AbsRel": means[f"{variant}/teacher/abs_rel"]}
            for target in ("gt", "teacher"):
                if variant == "student":
                    delta, ci = 0, [0, 0]
                else:
                    intervention = result["paired_interventions"][variant][target]
                    delta = intervention["difference"]
                    ci = intervention["scene_bootstrap_ci95"]
                row.update({f"{target}_delta": delta, f"{target}_delta_ci_low": ci[0],
                            f"{target}_delta_ci_high": ci[1]})
            comparisons.append(row)
    for name, rows in (("feature-summary.csv", features), ("interventions.csv", comparisons)):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        write(args.output / name, buffer.getvalue().encode())

    checkpoint = "day2_selected_156000"
    result = factor["checkpoints"][checkpoint]
    variants = ["student", "oracle_summary_norm", "oracle_summary_direction", "oracle_summary",
                "oracle_spatial_norm", "oracle_spatial_direction", "oracle_spatial", "teacher"]
    labels = ["Student unchanged", "Teacher summary norm", "Teacher summary direction",
              "Teacher whole summary", "Teacher spatial norm", "Teacher spatial direction",
              "Teacher whole spatial", "Teacher both branches"]
    colors = ["#555555", "#e69f00", "#e69f00", "#e69f00", "#0072b2", "#0072b2", "#0072b2",
              "#009e73"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, target, title in zip(axes[0], ("teacher", "gt"),
                                ("Error relative to teacher", "Error relative to sensor depth")):
        values = [result["means"][f"{v}/{target}/abs_rel"] for v in variants]
        ax.barh(np.arange(len(values)), values, color=colors, alpha=0.85)
        ax.set_yticks(np.arange(len(values)), labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Mean AbsRel (lower is better)")
        ax.axvline(values[0], linestyle="--", color="#555555", linewidth=1)
        ax.set_xlim(0, max(values) * 1.25)
        for i, value in enumerate(values):
            ax.text(value + max(values) * 0.015, i, f"{value:.4f}", va="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    rows = [r for r in json.loads((args.factor / "per_image.json").read_text())
            if r["checkpoint"] == checkpoint]
    for ax, key, label in zip(axes[1], ("spatial/cosine", "spatial/abs_log_norm_ratio"),
                              ("Spatial cosine similarity", "Spatial absolute log norm ratio")):
        ax.scatter([r[key] for r in rows], [r["student/teacher/abs_rel"] for r in rows],
                   s=20, alpha=0.65, color="#0072b2")
        rho = result["spearman"][f"{key} -> student/teacher/abs_rel"]
        ax.set_title(f"Per-image association (Spearman rho = {rho:.3f})")
        ax.set_xlabel(label)
        ax.set_ylabel("Student-teacher depth AbsRel")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("What feature errors matter? Frozen checkpoint 156,000", fontsize=15)
    fig.text(0.5, 0.012,
             "NYU validation: 116 images / 39 scenes. Teacher replacements are diagnostic oracles. "
             "Associations are descriptive; no training was performed.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    destination = args.output / "diagnostic-overview.png"
    temporary = destination.with_name(destination.name + ".new")
    fig.savefig(temporary, format="png", dpi=170)
    os.replace(temporary, destination)
    plt.close(fig)
    sources = {str(root / "summary.json"): hashlib.sha256((root / "summary.json").read_bytes())
               .hexdigest() for root in (args.initial, args.factor)}
    report = {"stage": "complete", "images": 116, "scenes": 39, "features": features,
              "comparisons": comparisons, "source_sha256": sources,
              "repeated_conditions_exactly_reproduced": True,
              "limits": initial["limits"], "figure_checkpoint": checkpoint}
    write(args.output / "analysis.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps({"output": str(args.output), "comparisons": len(comparisons),
                      "checkpoints": len(merged), "repeat_match": True}))


if __name__ == "__main__":
    main()
