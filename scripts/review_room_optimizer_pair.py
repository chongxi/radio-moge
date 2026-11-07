"""Compare all fixed-final metrics of the optimizer switch and reset control."""

import argparse
import json
from pathlib import Path
import time

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.optimizer_trial import MODES, check_counts
from review_corpus_controls import joint_result
from review_room_optimizer_arm import VIEWS
from run_room_optimizer_arm import UPDATES, compare_arm_configs


def contrast(left, right):
    if (
        left["new_updates"] != UPDATES
        or right["new_updates"] != UPDATES
        or not left["budget_complete"]
        or not right["budget_complete"]
        or left["step"] != right["step"]
        or left["parent"] != right["parent"]
        or set(left["views"]) != VIEWS
        or set(right["views"]) != VIEWS
        or left["actual_inputs"]["actual_input_prefix"]["sha256"]
        != right["actual_inputs"]["actual_input_prefix"]["sha256"]
    ):
        raise ValueError("Fixed budget, parent, populations or actual input bytes differ")
    values = {}
    for view in sorted(VIEWS):
        a, b = left["views"][view]["metrics"], right["views"][view]["metrics"]
        if set(a) != set(b):
            raise ValueError("Metric keys differ")
        values[view] = {}
        for metric in a:
            if a[metric]["before"] != b[metric]["before"]:
                raise ValueError("Baseline metric differs")
            x, y = a[metric]["final"], b[metric]["final"]
            values[view][metric] = dict(
                reference=x,
                candidate=y,
                difference=y - x,
                relative_percent=100 * (y - x) / abs(x) if x else None,
            )
    primary = {k: v["final"] for k, v in left["against_parent"]["metrics"].items()}
    candidate = {k: v["final"] for k, v in right["against_parent"]["metrics"].items()}
    return dict(all_six_view_metrics=values, primary_joint=joint_result(primary, candidate))


def run(root):
    plan = checked_json(file_ref(root / "plan.json"))
    if checked_json(file_ref(root / "report.json"))["stage"] != "training_complete_review_pending":
        raise ValueError("Both new arms must finish before the fixed-final comparison")
    baseline_root = Path(plan["baseline_arm"])
    baseline_ref = file_ref(baseline_root / "validation-263121-review-01/report.json")
    reports = {"continued-adamw": checked_json(baseline_ref)}
    evidence = {
        "continued-adamw": dict(
            review=baseline_ref, finish=plan["baseline_finish"], online=plan["baseline_online"]
        )
    }
    configs = [checked_json(file_ref(baseline_root / "config.json"))]
    for ref in plan["arms"]:
        arm = checked_json(ref)
        path = Path(ref["path"]).parent
        mode = arm["arm"]
        if mode not in MODES or mode in reports:
            raise ValueError("Unexpected optimizer arm")
        review = file_ref(path / "validation-263121-review-01/report.json")
        finish = file_ref(path / "finish-verification-01/report.json")
        online = file_ref(path / "online-final-01/report.json")
        final, live = checked_json(finish), checked_json(online)
        check_counts(final["optimizer_counts"], mode, arm["source_step"], UPDATES)
        if (
            final["result_review"] != review
            or not final["fixed_prefix_exactly_once"]
            or final["step"] != 263121
            or final["new_updates"] != UPDATES
            or not live["all_committed_values_numerically_exact"]
            or live["state"] != "finished"
        ):
            raise ValueError("Missing final optimizer state, data or online verification")
        reports[mode] = checked_json(review)
        evidence[mode] = dict(plan=ref, review=review, finish=finish, online=online)
        configs.append(checked_json(arm["config"]))
    for cfg in configs[1:]:
        compare_arm_configs(configs[0], cfg)
    pairs = [
        ("continued-adamw", "core-reset-adamw"),
        ("continued-adamw", "core-muon"),
        ("core-reset-adamw", "core-muon"),
    ]
    result = dict(
        stage="matched_optimizer_fixed_final_comparison_complete",
        evidence=evidence,
        contrasts={f"{b}_vs_{a}": contrast(reports[a], reports[b]) for a, b in pairs},
        each_arm_vs_parent={k: v["against_parent"] for k, v in reports.items()},
        each_arm_vs_archive={k: v["against_archive"] for k, v in reports.items()},
        fixed_final_no_best_selection=True,
        historical_reference_reused=True,
        actual_rgb_input_bytes_identical=True,
        model_inference=False,
        new_optimizer_updates=0,
        automatic_deployment=False,
        replicated=False,
        significance_or_real_planarity_established=False,
        created_at=time.time(),
        reviewer=file_ref(Path(__file__)),
    )
    output = root / "results-review-01"
    output.mkdir(exist_ok=False)
    publish_json(output / "report.json", result)
    print(json.dumps(file_ref(output / "report.json")))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args().root.resolve())
