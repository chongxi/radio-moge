from copy import deepcopy
import hashlib

import pytest

from summarize_scannetpp_flagged_scene_scope import (
    bin_index,
    bind_existing_quarantine,
    choose_followup,
    summarize,
    verified_temporary_copy,
)


def candidate(ordinal, depth, support=0.8, group="scannetpp:test"):
    return dict(
        corpus_ordinal=ordinal,
        group=group,
        outcome="native_candidate",
        policy_sha256="p",
        entry_key=f"key{ordinal}",
        evidence=dict(path="receipt", sha256="r"),
        image=dict(path=f"{ordinal}.jpg", sha256=f"hash{ordinal}"),
        existing_quarantine=False,
        metrics=dict(
            median_depth_m=depth,
            maximum_depth_m=depth + 1,
            valid_fraction=support,
            rejected_hit_fraction=0.01,
        ),
    )


def test_half_open_bins_use_image_medians_and_exclude_rejection_from_denominator():
    values = [0.01, 0.02, 0.05, 0.1, 0.25, 1.0]
    assert [bin_index(x) for x in values] == list(range(6))
    rows = [candidate(i, x) for i, x in enumerate(values)]
    rows.append(dict(corpus_ordinal=9, outcome="native_rejected", metrics={}))
    report = summarize(rows)
    assert report["images"] == 7 and report["candidates"] == 6 and report["rejected"] == 1
    assert [b["ordinals"] for b in report["supported_depth_median_bins"]] == [[i] for i in range(6)]
    assert report["per_image_quantiles"]["median_depth_m"][3] == pytest.approx(0.075)
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            bin_index(value)


def test_bounded_sampling_excludes_previous_reviews_and_keeps_control_without_duplicates():
    rows = [
        candidate(1, 0.01),
        candidate(2, 0.02, 0.6),
        candidate(3, 0.03, 0.95),
        candidate(4, 1),
        candidate(5, 2),
    ]
    selected = choose_followup(rows, {1})
    assert [r["corpus_ordinal"] for r in selected] == [2, 3, 4]
    assert "not_assumed_good" in selected[-1]["selection_reason"]
    # If no <0.1m image exists, do not manufacture a signature sample.
    assert len(choose_followup([candidate(1, 2), candidate(2, 3)], set())) == 2
    assert choose_followup(rows, {1, 2, 3, 4, 5}) == []


def quarantine_fixture():
    row = candidate(7, 0.02)
    flag = dict(
        corpus_ordinal=7,
        decision="quarantine_candidate",
        entry_key=row["entry_key"],
        receipt=row["evidence"],
        group=row["group"],
        image=row["image"]["path"],
        sha256=row["image"]["sha256"],
        member_sha256="member",
    )
    visual = dict(
        corpus_ordinal=7,
        decision="quarantine_candidate",
        entry_key=row["entry_key"],
        receipt=row["evidence"],
        group=row["group"],
        image=row["image"],
        member_sha256="member",
        policy_sha256="p",
    )
    review = dict(records=[visual], quarantined_candidate_ordinals=[7])
    register = dict(
        policy_sha256="p",
        packs_index={},
        stage="candidate_quality_quarantine_recorded",
        training_admission=False,
        optimizer_updates=0,
        flags=[flag],
    )
    binding = dict(policy_sha256="p", packs_index={}, count=1, ordinals=[7])
    return [row], review, dict(quarantined_candidates=1), [(binding, register)]


def test_quarantine_overlay_preserves_exact_identities():
    args = quarantine_fixture()
    bind_existing_quarantine(*args)
    assert args[0][0]["existing_quarantine"] is True


@pytest.mark.parametrize("field", ["policy_sha256", "entry_key", "group", "image", "evidence"])
def test_same_ordinal_with_changed_identity_cannot_inherit_quarantine(field):
    args = deepcopy(quarantine_fixture())
    args[0][0][field] = "changed"
    with pytest.raises(ValueError, match="identity"):
        bind_existing_quarantine(*args)


def test_copy_hash_and_open_journal_are_checked(tmp_path):
    path = tmp_path / "index.sqlite"
    path.write_bytes(b"fixture bytes")
    ref = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    verified_temporary_copy(ref, tmp_path / "copy")
    assert (tmp_path / "copy").read_bytes() == path.read_bytes()
    with pytest.raises(ValueError, match="hash changed"):
        verified_temporary_copy(dict(ref, sha256="f" * 64), tmp_path / "bad")
    (tmp_path / "index.sqlite-wal").touch()
    with pytest.raises(ValueError, match="immutable"):
        verified_temporary_copy(ref, tmp_path / "open")
