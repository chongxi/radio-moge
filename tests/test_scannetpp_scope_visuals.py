from copy import deepcopy
import json

import pytest

from visualize_scannetpp_scene_scope import checked_choice


def fixture():
    image = dict(image="a.jpg", sha256="rgb", group="scannetpp:scene", view="a.jpg", mask="a.png")
    proof = dict(path="receipt", sha256="receipt-sha")
    metrics = dict(median_depth_m=0.02)
    selected = dict(
        corpus_ordinal=7,
        image=dict(path="a.jpg", sha256="rgb"),
        group=image["group"],
        view="a.jpg",
        policy_sha256="policy",
        entry_key="key",
        evidence=proof,
        metrics=metrics,
        existing_quarantine=False,
        selection_reason="lowest_supported_depth_median",
    )
    indexed = (
        json.dumps(image),
        "native_candidate",
        "policy",
        "key",
        json.dumps(proof),
        json.dumps(metrics),
    )
    return selected, indexed, {"policy": 123}


def test_frozen_full_image_metadata_preserved_for_existing_pixel_renderer():
    selected, indexed, policies = fixture()
    chosen = checked_choice(selected, indexed, policies)
    assert chosen["image"] == json.loads(indexed[0])
    assert chosen["selection_reasons"] == [selected["selection_reason"]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", dict(path="b.jpg", sha256="rgb")),
        ("entry_key", "wrong"),
        ("metrics", dict(median_depth_m=0.03)),
        ("existing_quarantine", True),
    ],
)
def test_changed_identity_or_known_quarantine_cannot_become_unreviewed_selection(field, value):
    selected, indexed, policies = deepcopy(fixture())
    selected[field] = value
    with pytest.raises(ValueError, match="frozen native identity"):
        checked_choice(selected, indexed, policies)
