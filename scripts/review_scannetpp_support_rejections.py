"""Review retained rejection counts and current RGB/masks; never fabricate depth.

Rejected depth/support arrays and original mask hashes were not saved. Current
mask observations are explicitly separate from historical producer statistics.
No renderer, mesh intersection, model, GPU, source admission or data repair runs.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from audit_scannetpp_native_outcomes import quality, rejection_details, row_shape
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json
from summarize_scannetpp_flagged_scene_scope import verified_temporary_copy
from visualize_scannetpp_native_outcomes import anonymity_support


REASON = "insufficient_verified_visible_support"


def attrition(stats, shape):
    support = quality(stats, shape, large_mesh=True)
    if support >= 0.05:
        raise ValueError("Only unchanged below-five-percent rejection support is eligible")
    values = [int(np.prod(shape))] + [
        stats[k]
        for k in (
            "initial_valid_pixels",
            "retained_pixels",
            "cpu_rays_checked",
            "cpu_hits",
            "verified_pixels",
        )
    ]
    names = (
        "outside_initial_raster_support",
        "combined_conservative_visibility_loss",
        "producer_anonymity_intersection_loss",
        "cpu_no_hit",
        "cpu_range_or_parity_rejection",
    )
    losses = dict(zip(names, (a - b for a, b in zip(values, values[1:]))))
    losses["verified_pixels"] = values[-1]
    assert sum(losses.values()) == values[0]
    return losses


def bind_rejection(item, indexed, evidence, policy_hash):
    ordinal, encoded, outcome, reason, policy, proof, details = indexed
    row = dict(item["image"])
    if (
        row.pop("corpus_ordinal", ordinal) != ordinal
        or row != json.loads(encoded)
        or outcome != "native_rejected"
        or reason != REASON
        or item["reason"] != REASON
        or policy != policy_hash
        or json.loads(proof) != evidence
        or json.loads(details) != item["stats"]
    ):
        raise ValueError("Exact frozen rejection row, policy, evidence and counts required")
    return ordinal


def current_inputs(row, shape):
    rgb_ref = dict(path=row["image"], sha256=row["sha256"])
    with Image.open(checked_path(rgb_ref)) as source:
        if source.size != (row["width"], row["height"]) or source.getexif().get(274, 1) != 1:
            raise ValueError("Frozen RGB dimensions or orientation differ")
        rgb = source.convert("RGB").copy()
    # This is a NEW evidence hash, not a claim that the producer saved this hash.
    mask_ref = file_ref(row["mask"])
    with Image.open(checked_path(mask_ref)) as source:
        mask = np.asarray(source)
    allowed = anonymity_support(mask, (row["height"], row["width"]), shape)
    return (
        rgb,
        mask,
        allowed,
        dict(
            rgb=rgb_ref,
            current_mask=mask_ref,
            original_mask_hash_recorded=False,
            mask_identity_to_generation_established=False,
            current_anonymity_pixels=int(allowed.sum()),
            current_anonymity_fraction=float(allowed.mean()),
        ),
    )


def tile(ordinal, row, stats, rgb, mask, allowed):
    page = Image.new("RGB", (1080, 338), "white")
    draw = ImageDraw.Draw(page)
    draw.text((5, 4), f"#{ordinal} {row['scene']} / {row['view']}", fill="black")
    draw.text(
        (5, 22),
        f"Producer counts: initial {stats['initial_valid_pixels']} -> visibility "
        f"{stats['retained_pixels']} -> masked rays {stats['cpu_rays_checked']} -> hits "
        f"{stats['cpu_hits']} -> verified {stats['verified_pixels']} ({stats['valid_fraction']:.4%})",
        fill="black",
    )
    draw.text(
        (5, 40),
        "Rejected final depth/support NOT retained. Below are CURRENT anonymity masks, NOT final geometry support.",
        fill="black",
    )
    panels = [
        rgb,
        Image.fromarray((mask == 255).astype(np.uint8) * 255),
        Image.fromarray(allowed.astype(np.uint8) * 255),
    ]
    for i, (panel, title) in enumerate(
        zip(
            panels,
            (
                "Original RGB (frozen SHA verified)",
                "Current native mask == 255",
                "Current float-BOX + 2 cross erosions",
            ),
        )
    ):
        draw.text((i * 360 + 5, 62), title, fill="black")
        scaled = ImageOps.contain(
            panel.convert("RGB"),
            (356, 250),
            Image.Resampling.LANCZOS if i == 0 else Image.Resampling.NEAREST,
        )
        page.paste(scaled, (i * 360 + (360 - scaled.width) // 2, 82))
    return page


def run(root, output):
    # Importing these helpers does not invoke CudaMeshDepth or touch a GPU.
    from cache_scannetpp_geometry import camera_for_row, colmap_poses

    audit_ref = file_ref(root / "scannetpp-native-outcomes-01/report.json")
    audit = checked_json(audit_ref)
    if (
        audit["stage"] != "native_outcomes_accounted_not_accepted"
        or audit["producer_array_readback_complete"] is not True
        or audit["expected_scope"] != 841431
        or audit["rejection_reasons"].get(REASON) != 70
    ):
        raise ValueError("Completed frozen native audit with all seventy support rejects required")
    cache = root / "scannetpp-native-cache-02"
    rejection_refs = [r for r in audit["evidence"] if r["path"] == str(cache / "rejections.json")]
    policy_ref = next(r for r in audit["evidence"] if r["path"] == str(cache / "packs/policy.json"))
    if len(rejection_refs) != 1:
        raise ValueError("Unique frozen rejection evidence required")
    rejection_ref = rejection_refs[0]
    policy = checked_json(policy_ref)
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if (
        policy_hash not in audit["candidate_policy_counts"]
        or policy["minimum_image_valid_fraction"] != 0.05
        or policy["manifest_sha256"] != audit["manifest_sha256"]
    ):
        raise ValueError("Unchanged source policy required")
    selected = [x for x in checked_json(rejection_ref) if x["reason"] == REASON]
    if len(selected) != 70 or len({x["image"]["image"] for x in selected}) != 70:
        raise ValueError("Seventy unique support rejection records required")
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/geometry-corpus-20261102.json").read_text()
    )
    output.mkdir(exist_ok=False)
    records, pages, tiles, verified_sources, scenes = [], [], [], {}, {}
    with tempfile.TemporaryDirectory(prefix="scannet-rejection-review-") as scratch:
        db_path = Path(scratch) / "index.sqlite"
        verified_temporary_copy(audit["index"], db_path)
        with readonly_db(db_path) as db:
            count = db.execute(
                "SELECT COUNT(*) FROM images WHERE outcome=? AND reason=?",
                ("native_rejected", REASON),
            ).fetchone()[0]
            if count != 70:
                raise ValueError("Indexed rejection scope differs")
            for item in selected:
                row = item["image"]
                indexed = db.execute(
                    "SELECT ordinal,row_json,outcome,reason,policy,evidence_json,details_json "
                    "FROM images WHERE image=?",
                    (row["image"],),
                ).fetchone()
                if indexed is None:
                    raise ValueError("Rejected RGB absent from frozen index")
                ordinal = bind_rejection(item, indexed, rejection_ref, policy_hash)
                individual_ref = file_ref(cache / "rejected" / (row["sha256"] + ".json"))
                if checked_json(individual_ref) != item:
                    raise ValueError("Original per-image rejection receipt differs")
                for p in item["source_proofs"]:
                    ref = dict(path=p["source"], sha256=p["sha256"])
                    if ref["path"] in verified_sources and verified_sources[ref["path"]] != ref:
                        raise ValueError("Conflicting retained source identity")
                    if ref["path"] not in verified_sources:
                        checked_path(ref)
                        verified_sources[ref["path"]] = ref
                scene_ref = dict(
                    path=item["source_proofs"][0]["source"],
                    sha256=item["source_proofs"][0]["sha256"],
                )
                scene = checked_json(scene_ref)
                rejection_details(item, policy, scene, scene_ref, config["dataset_root"])
                shape = row_shape(row, config["dataset_root"])
                losses = attrition(item["stats"], shape)
                if row["scene"] not in scenes:
                    calibration = json.loads(Path(item["source_proofs"][3]["source"]).read_text())
                    frames = {Path(f["file_path"]).name: f for f in calibration["frames"]}
                    poses = colmap_poses(item["source_proofs"][2]["source"])
                    scenes[row["scene"]] = (calibration, frames, poses)
                k, _ = camera_for_row(row, *scenes[row["scene"]])
                rgb, mask, allowed, inputs = current_inputs(row, shape)
                tiles.append(tile(ordinal, row, item["stats"], rgb, mask, allowed))
                records.append(
                    dict(
                        corpus_ordinal=ordinal,
                        image=row,
                        reason=REASON,
                        original_rejection=individual_ref,
                        producer_statistics=item["stats"],
                        attrition_counts=losses,
                        **inputs,
                        policy_sha256=policy_hash,
                        normalized_intrinsics=k.tolist(),
                        colmap_pose_verified=True,
                        final_depth_and_support_retained=False,
                        producer_statistics_reproduced=False,
                        source_proofs=item["source_proofs"],
                        page_name=f"{len(pages):02d}.png",
                    )
                )
                if len(tiles) == 4 or len(records) == len(selected):
                    canvas = Image.new("RGB", (1080, sum(t.height for t in tiles)), "#cccccc")
                    y = 0
                    for t in tiles:
                        canvas.paste(t, (0, y))
                        y += t.height
                    path = output / f"{len(pages):02d}.png"
                    temp = path.with_suffix(".png.incoming")
                    canvas.save(temp, format="PNG")
                    temp.rename(path)
                    pages.append(file_ref(path))
                    tiles.clear()
                print(
                    json.dumps(dict(processed=len(records), expected=70, ordinal=ordinal)),
                    flush=True,
                )
    for rec in records:
        name = rec.pop("page_name")
        rec["page"] = next(p for p in pages if Path(p["path"]).name == name)
    records_path = output / "records.json"
    publish_json(records_path, records)
    report = dict(
        stage="support_rejection_evidence_rendered_not_reviewed",
        audit=audit_ref,
        index=audit["index"],
        rejections=rejection_ref,
        policy=policy_ref,
        policy_sha256=policy_hash,
        manifest_sha256=audit["manifest_sha256"],
        processed=70,
        all_rejection_identities_bound=True,
        groups=dict(Counter(x["image"]["group"] for x in records)),
        records=file_ref(records_path),
        pages=pages,
        source_proofs=list(verified_sources.values()),
        original_statistics_reproduced=False,
        original_mask_hashes_recorded=False,
        final_depth_and_support_arrays_retained=False,
        interpretation="Original count accounting is verified; current RGB/mask/camera observed. "
        "No rejected geometry rerender or physical-error claim. Current mask hash cannot bind generation mask.",
        generator=file_ref(Path(__file__).resolve()),
        optimizer_updates=0,
        gpu_used=False,
        source_quality_accepted=False,
        training_admission=False,
    )
    publish_json(output / "report.json", report)
    print(
        json.dumps(dict(report=file_ref(output / "report.json"), pages=len(pages), processed=70)),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
