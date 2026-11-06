"""Close the fixed-final LR comparison using saved evidence only."""

import argparse
import json
from pathlib import Path
import time

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from review_corpus_controls import joint_result
from review_room_lr_arm import VIEWS
from run_room_lr_arm import UPDATES, compare_arm_configs


def compare_final(a, b):
    if (
        a["new_updates"] != UPDATES
        or b["new_updates"] != UPDATES
        or not a["budget_complete"]
        or not b["budget_complete"]
        or a["step"] != b["step"]
        or a["parent"] != b["parent"]
        or set(a["views"]) != VIEWS
        or set(b["views"]) != VIEWS
        or a["actual_inputs"]["actual_input_prefix"]["sha256"]
        != b["actual_inputs"]["actual_input_prefix"]["sha256"]
    ):
        raise ValueError("Fixed step, baseline, views or actual RGB bytes differ")
    contrasts = {}
    for view in sorted(VIEWS):
        left, right = a["views"][view]["metrics"], b["views"][view]["metrics"]
        if set(left) != set(right):
            raise ValueError("Metric population differs")
        contrasts[view] = {}
        for key in left:
            if left[key]["before"] != right[key]["before"]:
                raise ValueError("Saved baseline metric differs")
            x, y = left[key]["final"], right[key]["final"]
            contrasts[view][key] = dict(
                lr1e5=x,
                lr2e5=y,
                difference=y - x,
                relative_percent=100 * (y - x) / abs(x) if x else None,
            )
    primary = {
        k: a["against_parent"]["metrics"][k]["final"] for k in a["against_parent"]["metrics"]
    }
    current = {k: b["against_parent"]["metrics"][k]["final"] for k in primary}
    return contrasts, joint_result(primary, current)


def run(root):
    plan = checked_json(file_ref(root / "plan.json"))
    controller = checked_json(file_ref(root / "report.json"))
    if controller["stage"] != "training_complete_review_pending":
        raise ValueError("Both controller arms must finish first")
    reviews, evidence, configs = [], [], []
    for ref in plan["arms"]:
        arm = checked_json(ref)
        path = Path(ref["path"]).parent
        final_step = arm["source_step"] + UPDATES
        review_ref = file_ref(path / f"validation-{final_step}-review-01/report.json")
        finish_ref = file_ref(path / "finish-verification-01/report.json")
        online_ref = file_ref(path / "online-final-01/report.json")
        finish, online = checked_json(finish_ref), checked_json(online_ref)
        if (
            finish["step"] != final_step
            or finish["new_updates"] != UPDATES
            or finish["learning_rate"] != arm["learning_rate"]
            or not finish["fixed_prefix_exactly_once"]
            or finish["result_review"] != review_ref
            or not online["all_committed_values_numerically_exact"]
            or online["state"] != "finished"
        ):
            raise ValueError("Missing fixed-final checkpoint or online closure")
        configs.append(checked_json(arm["config"]))
        reviews.append(checked_json(review_ref))
        evidence.append(dict(plan=ref, review=review_ref, finish=finish_ref, online=online_ref))
    compare_arm_configs(*configs)
    contrasts, joint = compare_final(*reviews)
    output = root / "results-review-01"
    output.mkdir(exist_ok=False)
    report = dict(
        stage="matched_lr_fixed_final_comparison_complete",
        arms=evidence,
        all_six_view_metrics=contrasts,
        lr2e5_vs_lr1e5_primary=joint,
        each_arm_vs_parent=[r["against_parent"] for r in reviews],
        each_arm_vs_archive=[r["against_archive"] for r in reviews],
        actual_rgb_input_bytes_identical=True,
        fixed_final_no_best_selection=True,
        model_inference=False,
        new_optimizer_updates=0,
        automatic_deployment=False,
        replicated=False,
        significance_or_real_planarity_established=False,
        created_at=time.time(),
        reviewer=file_ref(Path(__file__)),
    )
    publish_json(output / "report.json", report)
    print(json.dumps(dict(report=file_ref(output / "report.json"), comparison=joint)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve())
