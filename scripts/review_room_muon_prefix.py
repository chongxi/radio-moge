"""Independent committed-prefix review of per-epoch identity and full-forward grouping."""

from collections import Counter
import hashlib
import json
from run_room_muon_continuation import MultiEpochCoverage


def committed_prefix(path, planned, synthetic, real_images, source_step, updates, batch=32):
    tracker = MultiEpochCoverage(real_images, synthetic, batch)
    sources, digest, byte_count = Counter(), hashlib.sha256(), 0
    with path.open("rb") as stream:
        for number in range(updates):
            raw = stream.readline()
            if not raw:
                raise ValueError("Missing committed RGB input audit")
            row = json.loads(raw)
            images = row["logical_images"]
            if (
                row["update"] != source_step + number + 1
                or len(images) != batch
                or [r["key"] for r in images] != planned[number * batch : (number + 1) * batch]
                or any(r["dataset"] == "coco2017" or "ordinary_geometry" in r for r in images)
            ):
                raise ValueError("Committed update, source or RGB keys differ from the frozen plan")
            if batch != 32 and (
                len(row["compute_groups"]) != 1
                or row["compute_groups"][0]["shape"][0] != batch
                or row["compute_groups"][0]["logical_image_slots"] != list(range(batch))
            ):
                raise ValueError("Actual true batch was split/reordered")
            tracker.commit(tracker.validate(images))
            sources.update(r["dataset"] for r in images)
            digest.update(raw)
            byte_count += len(raw)
    return dict(
        updates=updates,
        presentations=updates * batch,
        source_counts=dict(sources),
        actual_planned_keys_matched=True,
        synthetic_identity_and_no_within_epoch_repeats_verified=True,
        synthetic_coverage=tracker.report(),
        actual_input_prefix=dict(
            path=str(path),
            bytes=byte_count,
            sha256=digest.hexdigest(),
            lines=updates,
            file_may_still_grow=True,
        ),
        newly_decoded_images=0,
        unused_prefetch_excluded=True,
    )
