"""Retain a bounded RGB diagnostic for a rejected structural-room camera."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from render_structural_room_batch import camera_slots, scene_spec, spawn_structures
from render_procthor_empty_pilot import create_controller, publish, sha
from render_structural_room_samples import save_image


def run(root, output, source):
    output.mkdir(exist_ok=False)
    (output / "assets").mkdir()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    spec = json.loads(source.read_text())
    if spec != scene_spec(spec["split"], spec["scene_ordinal"]):
        raise ValueError("Retained scene differs from the pinned generator")
    started = time.monotonic()
    controller = create_controller(root, output, spec["house"], field_of_view=32)
    records = []
    try:
        assets = spawn_structures(controller, spec, output, "diagnostic")
        for index, slot_index in enumerate((12, 13)):
            slot = camera_slots(spec, 20)[slot_index]
            for variant, color in enumerate(("#ff00ff", "#00ffff", "#000000")):
                request = dict(slot["request"], skyboxColor=color)
                event = controller.step(
                    action="AddThirdPartyCamera" if not records else "UpdateThirdPartyCamera",
                    **({} if not records else dict(thirdPartyCameraId=0)),
                    **request,
                    raise_for_failure=True,
                )
                rgb = event.third_party_camera_frames[0]
                target = output / f"slot-{slot_index:02d}-background-{variant}.png"
                save_image(target, rgb)
                mask = (rgb[..., 0] > 240) & (rgb[..., 1] < 15) & (rgb[..., 2] > 240)
                coordinates = np.argwhere(mask)
                records.append(dict(
                    slot=slot_index, request=request, actual=event.metadata["thirdPartyCameras"],
                    image=dict(path=str(target), sha256=sha(target)),
                    decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                    magenta_pixels=int(mask.sum()), magenta_yx=coordinates.tolist(),
                    spatial_std=float(rgb.astype(np.float32).std(axis=(0, 1)).mean()),
                ))
        report = dict(
            stage="diagnostic_rgb_rendered_not_training", source=dict(path=str(source), sha256=sha(source)),
            source_exact=True, assets=assets, records=records, elapsed_seconds=time.monotonic()-started,
            geometry_labels=False, optimizer_started=False, training_admission=False,
            code={p.name:sha(p) for p in (Path(__file__), Path(__file__).with_name("render_structural_room_batch.py"))},
        )
        publish(output / "report.json", report)
        return report
    finally:
        controller.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "source"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output, args.source)
