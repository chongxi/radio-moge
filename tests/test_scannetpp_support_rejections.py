from copy import deepcopy
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from review_scannetpp_support_rejections import REASON, attrition, bind_rejection, current_inputs
from publish_scannetpp_support_decisions import checked_decision_scope
from test_scannetpp_native_outcomes import support


def test_count_waterfall_covers_each_pixel_once_without_cause_claims():
    stats = support(100)
    stats.update(
        retained_pixels=400,
        retained_fraction=400 / 5120,
        cpu_rays_checked=300,
        cpu_hits=200,
        cpu_missing_hits=100,
        rejected_inconsistent_hits=100,
    )
    values = attrition(stats, [8, 640])
    assert sum(values.values()) == 5120
    assert values["combined_conservative_visibility_loss"] == 4720
    assert values["producer_anonymity_intersection_loss"] == 100
    assert values["cpu_range_or_parity_rejection"] == 100
    assert values["verified_pixels"] == 100


def test_zero_support_and_exact_five_percent_boundary():
    assert attrition(support(0), [8, 640])["verified_pixels"] == 0
    with pytest.raises(ValueError, match="below-five-percent"):
        attrition(support(256), [8, 640])


@pytest.mark.parametrize(
    "key,value",
    [
        ("cpu_missing_hits", 3),
        ("valid_fraction", 0.01),
        ("verified_pixels", 600),
        ("depth_filled_or_averaged", True),
    ],
)
def test_invalid_historical_counts_do_not_get_repaired(key, value):
    stats = support(100)
    stats[key] = value
    with pytest.raises(ValueError):
        attrition(stats, [8, 640])


@pytest.mark.parametrize("mutation", ["image", "stats", "reason", "policy", "evidence"])
def test_exact_rejection_binding_rejects_changed_scope(mutation):
    item = dict(image=dict(image="/frozen/a", sha256="a" * 64), reason=REASON, stats=support(100))
    proof = dict(path="/frozen/rejections.json", sha256="b" * 64)
    indexed = [
        42,
        json.dumps(item["image"]),
        "native_rejected",
        REASON,
        "c" * 64,
        json.dumps(proof),
        json.dumps(item["stats"]),
    ]
    assert bind_rejection(item, indexed, proof, "c" * 64) == 42
    item = deepcopy(item)
    if mutation == "image":
        item["image"]["image"] = "/changed/a"
    elif mutation == "stats":
        item["stats"]["verified_pixels"] += 1
    elif mutation == "reason":
        indexed[3] = "source_scene_pending_review"
    elif mutation == "policy":
        indexed[4] = "d" * 64
    else:
        indexed[5] = json.dumps(dict(proof, sha256="e" * 64))
    with pytest.raises(ValueError, match="Exact frozen"):
        bind_rejection(item, indexed, proof, "c" * 64)


def test_current_mask_is_not_labeled_original_geometry_support(tmp_path):
    rgb = tmp_path / "a.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (24, 16), "white").save(rgb)
    m = np.full((16, 24), 255, dtype=np.uint8)
    m[:, :6] = 0
    Image.fromarray(m).save(mask)
    row = dict(
        image=str(rgb),
        sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(),
        width=24,
        height=16,
        mask=str(mask),
    )
    _, _, allowed, record = current_inputs(row, [8, 12])
    assert allowed.dtype == bool and 0 < allowed.sum() < allowed.size
    assert record["original_mask_hash_recorded"] is False
    assert record["mask_identity_to_generation_established"] is False
    row["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        current_inputs(row, [8, 12])


@pytest.mark.parametrize(
    "change", ["duplicate", "not_viewed", "different_image", "claimed_reproduction"]
)
def test_publication_requires_exact_review_and_preserves_evidence_limits(change):
    row = dict(
        image="/frozen/a",
        sha256="a" * 64,
        dataset="scannetpp_dslr",
        split="train",
        width=640,
        height=8,
    )
    rec = dict(
        corpus_ordinal=42,
        image=row,
        reason=REASON,
        page=dict(path="/frozen/page", sha256="b" * 64),
        original_rejection=dict(path="/frozen/rejection", sha256="c" * 64),
        producer_statistics=support(100),
        attrition_counts=attrition(support(100), [8, 640]),
        producer_statistics_reproduced=False,
        final_depth_and_support_retained=False,
        original_mask_hash_recorded=False,
    )
    choice = {k: rec[k] for k in ("corpus_ordinal", "image", "page", "original_rejection")}
    choice.update(decision="exclude_geometry_supervision_retain_rgb", actual_page_viewed=True)
    review = dict(records=[choice])
    checked_decision_scope([rec], review)
    if change == "duplicate":
        review["records"].append(deepcopy(choice))
    elif change == "not_viewed":
        choice["actual_page_viewed"] = False
    elif change == "different_image":
        choice["image"] = dict(row, image="/changed/a")
    else:
        rec["producer_statistics_reproduced"] = True
    with pytest.raises(ValueError):
        checked_decision_scope([rec], review)
