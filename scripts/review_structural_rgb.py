"""Create hash-bound contact pages for selected frozen structural RGB views."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from render_procthor_empty_pilot import publish, sha


def prepare(report_path, output, maximum_scenes=None):
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
    output.mkdir(exist_ok=False)
    pages, selected = [], []
    for index, group in enumerate(groups):
        views = [r for r in rows if r["group"] == group and r["slot"] in (0, 10, 11, 18)]
        if [r["slot"] for r in views] != [0, 10, 11, 18]:
            raise ValueError("A fixed review slot was rejected; require an explicit new selection")
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
                draw.text(
                    (x + 8, y + 6),
                    f"{group.split('/')[-1]} slot {row['slot']:02d} | {row['projection']}",
                    fill="#26342b",
                )
            selected.append(
                dict(
                    image=row["image"],
                    sha256=row["sha256"],
                    group=group,
                    slot=row["slot"],
                    projection=row["projection"],
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
        generator=dict(path=str(Path(__file__)), sha256=sha(Path(__file__))),
    )
    publish(output / "index.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--maximum-scenes", type=int)
    a = p.parse_args()
    r = prepare(a.report, a.output, a.maximum_scenes)
    print(json.dumps(dict(images=r["images"], scenes=r["scenes"])))
