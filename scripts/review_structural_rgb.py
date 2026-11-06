"""Create hash-bound contact pages for selected frozen structural RGB views."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from render_procthor_empty_pilot import publish, sha


def rejected_view(report, group, slot):
    """Retrieve a rejected image only from the frozen report's bound receipts."""
    matches = []
    for reference in report.get("rejection_proofs", []):
        path = Path(reference["path"])
        if sha(path) != reference["sha256"]:
            raise ValueError("Bound rejection receipt differs")
        receipt = json.loads(path.read_text())
        if (receipt.get("scene"), receipt.get("slot")) == (group, slot):
            matches.append((reference, receipt))
    if len(matches) != 1:
        raise ValueError("Missing review slot requires exactly one bound rejection")
    reference, receipt = matches[0]
    if receipt.get("training_admission") is not False or receipt.get("retained_for_diagnosis_only") is not True:
        raise ValueError("Rejected review image must remain diagnosis-only")
    return dict(image=receipt["image"]["path"], sha256=receipt["image"]["sha256"],
                decoded_rgb_sha256=receipt["decoded_rgb_sha256"], group=group, slot=slot,
                projection="orthographic" if receipt["requested_camera"]["orthographic"] else "perspective",
                rejection_proof=reference, rejection_reason=receipt["reason"],
                review_role="rejected_diagnosis_only", training_admission=False)


def prepare(report_path, output, maximum_scenes=None, include_rejected_for_review=False):
    report = json.loads(report_path.read_text())
    if report["stage"] != "frozen_verified_structural_feature_rgb":
        raise ValueError("Require verified custom-source RGB")
    manifest = Path(report["manifest"]["path"])
    if sha(manifest) != report["manifest"]["sha256"]:
        raise ValueError("Frozen manifest differs")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    groups = sorted({r["group"] for r in rows})
    source_scenes = len(groups)
    if maximum_scenes is not None:
        if maximum_scenes < 2:
            raise ValueError("Select at least two evenly spaced scenes")
        if len(groups) > maximum_scenes:
            groups = [groups[i * (len(groups) - 1) // (maximum_scenes - 1)]
                      for i in range(maximum_scenes)]
    grouped_views = []
    for group in groups:
        views = []
        for slot in (0, 10, 11, 18):
            found = [r for r in rows if r["group"] == group and r["slot"] == slot]
            if len(found) == 1:
                views.append(found[0])
            elif not found and include_rejected_for_review:
                views.append(rejected_view(report, group, slot))
            else:
                raise ValueError("A fixed review slot was rejected; require an explicit new selection")
        grouped_views.append(views)
    output.mkdir(exist_ok=False)
    pages, selected = [], []
    for index, (group, views) in enumerate(zip(groups, grouped_views)):
        page = Image.new("RGB", (1280, 1016), "#eef0ed")
        draw = ImageDraw.Draw(page)
        for cell, row in enumerate(views):
            path = Path(row["image"])
            if sha(path) != row["sha256"]:
                raise ValueError("Review image differs")
            with Image.open(path) as image:
                if hashlib.sha256(image.tobytes()).hexdigest() != row["decoded_rgb_sha256"]:
                    raise ValueError("Review pixels differ")
                x, y = cell % 2 * 640, cell // 2 * 508
                page.paste(image, (x, y + 28))
                rejected = row.get("review_role") == "rejected_diagnosis_only"
                if rejected:
                    draw.rectangle((x, y, x + 639, y + 27), fill="#820c12")
                draw.text(
                    (x + 8, y + 6),
                    f"{group.split('/')[-1]} slot {row['slot']:02d} | "
                    + ("REJECTED / NOT TRAINING: " + row["rejection_reason"] if rejected else row["projection"]),
                    fill="white" if rejected else "#26342b",
                )
            selected.append(
                dict(
                    image=row["image"],
                    sha256=row["sha256"],
                    group=group,
                    slot=row["slot"],
                    projection=row["projection"],
                    review_role=row.get("review_role", "frozen_rgb_sample"),
                    **({"rejection_proof": row["rejection_proof"], "training_admission": False,
                        "rejection_reason": row["rejection_reason"]} if rejected else {}),
                )
            )
        target = output / f"page-{index:02d}.png"
        incoming = target.with_suffix(".png.incoming")
        page.save(incoming, format="PNG")
        incoming.replace(target)
        pages.append(dict(path=str(target), sha256=sha(target), group=group, images=4))
    result = dict(
        stage="contact_pages_ready_not_reviewed",
        source_report=dict(path=str(report_path), sha256=sha(report_path)),
        selected=selected,
        pages=pages,
        images=len(selected),
        scenes=len(groups),
        source_scenes=source_scenes,
        maximum_scenes=maximum_scenes,
        scene_selection="all" if source_scenes == len(groups) else "fixed_evenly_spaced_sorted_groups",
        fixed_slots=[0, 10, 11, 18],
        training_admission=False,
        include_rejected_for_review=include_rejected_for_review,
        admitted_sample_images=sum(r["review_role"] == "frozen_rgb_sample" for r in selected),
        rejected_diagnostic_images=sum(r["review_role"] == "rejected_diagnosis_only" for r in selected),
        replacement_images=0,
        generator=dict(path=str(Path(__file__)), sha256=sha(Path(__file__))),
    )
    publish(output / "index.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--maximum-scenes", type=int)
    p.add_argument("--include-rejected-for-review", action="store_true",
                   help="Keep fixed missing slots as clearly marked, receipt-bound rejected images; never admit them")
    a = p.parse_args()
    r = prepare(a.report, a.output, a.maximum_scenes, a.include_rejected_for_review)
    print(json.dumps(dict(images=r["images"], scenes=r["scenes"])))
