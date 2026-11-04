"""Render only the bounded, previously unviewed selection from a scene-scope audit."""

import argparse
import json
from pathlib import Path
import tempfile

from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import publish_json
from summarize_scannetpp_flagged_scene_scope import verified_temporary_copy
from visualize_scannetpp_native_outcomes import render_entry


def checked_choice(selected, indexed, source_policies):
    encoded, outcome, policy, key, evidence, details = indexed
    row, proof, metrics = json.loads(encoded), json.loads(evidence), json.loads(details)
    if (
        outcome != "native_candidate"
        or policy not in source_policies
        or selected["existing_quarantine"]
        or policy != selected["policy_sha256"]
        or row["image"] != selected["image"]["path"]
        or row["sha256"] != selected["image"]["sha256"]
        or row["group"] != selected["group"]
        or row["view"] != selected["view"]
        or key != selected["entry_key"]
        or proof != selected["evidence"]
        or metrics != selected["metrics"]
    ):
        raise ValueError("Follow-up selection differs from exact frozen native identity")
    return dict(
        corpus_ordinal=selected["corpus_ordinal"],
        image=row,
        policy_sha256=policy,
        entry_key=key,
        receipt=proof,
        metrics=metrics,
        selection_reasons=[selected["selection_reason"]],
    )


def render(scope_path, output):
    scope_ref = file_ref(scope_path)
    scope = checked_json(scope_ref)
    audit = checked_json(scope["native_audit"])
    selected = checked_json(scope["followup_candidates"])
    if (
        scope["stage"] != "flagged_scene_metadata_scope_complete"
        or scope["training_admission"] is not False
        or scope["source_quality_accepted"] is not False
        or scope["index"] != audit["index"]
        or scope["manifest_sha256"] != audit["manifest_sha256"]
        or audit["stage"] != "native_outcomes_accounted_not_accepted"
        or audit["producer_array_readback_complete"] is not True
        or audit["source_flags_preserved"] is not True
        or not 1 <= len(selected) == scope["followup_images"] <= 15
        or len({r["corpus_ordinal"] for r in selected}) != len(selected)
        or {r["corpus_ordinal"] for r in selected} & set(scope["reviewed_ordinal_set"])
    ):
        raise ValueError(
            "Completed bounded metadata selection outside previous visual scope required"
        )
    output.mkdir(exist_ok=False)
    records = []
    with tempfile.TemporaryDirectory(prefix="scannet-scope-visual-") as scratch:
        database = Path(scratch) / "index.sqlite"
        verified_temporary_copy(audit["index"], database)
        with readonly_db(database) as db:
            for selection in selected:
                row = db.execute(
                    "SELECT row_json,outcome,policy,entry_key,evidence_json,details_json FROM images WHERE ordinal=?",
                    (selection["corpus_ordinal"],),
                ).fetchone()
                if row is None:
                    raise ValueError("Follow-up candidate absent from frozen index")
                chosen = checked_choice(selection, row, audit["candidate_policy_counts"])
                receipt = checked_json(chosen["receipt"])
                entries = [e for e in receipt["entries"] if e["key"] == chosen["entry_key"]]
                if len(entries) != 1 or receipt["policy_sha256"] != chosen["policy_sha256"]:
                    raise ValueError("Follow-up candidate receipt differs")
                pack = Path(chosen["receipt"]["path"]).parent / receipt["tar"]
                if (
                    pack.parent != Path(chosen["receipt"]["path"]).parent
                    or pack.stat().st_size != receipt["bytes"]
                ):
                    raise ValueError("Follow-up pack path/size differs")
                page, identity = render_entry(pack, entries[0], chosen)
                path = output / f"{len(records):02d}-{chosen['corpus_ordinal']}.png"
                temporary = path.with_suffix(".png.incoming")
                with temporary.open("xb") as stream:
                    page.save(stream, format="PNG")
                temporary.rename(path)
                records.append(dict(**identity, receipt=chosen["receipt"], page=file_ref(path)))
                print(
                    json.dumps(
                        dict(rendered=len(records), corpus_ordinal=chosen["corpus_ordinal"])
                    ),
                    flush=True,
                )
    checked_path(scope_ref)
    checked_path(scope["followup_candidates"])
    report = dict(
        stage="native_scope_visual_evidence_rendered_not_reviewed",
        scope=scope_ref,
        native_audit=scope["native_audit"],
        selection=scope["followup_candidates"],
        records=records,
        rendered_images=len(records),
        selected_payloads_rechecked=True,
        original_masks_rechecked=True,
        normalized_camera_and_support_checked=True,
        fixed_depth_limits_m=[0.025, 64],
        depth_or_mask_edited=False,
        mesh_rerendered=False,
        source_flags_cleared=False,
        source_quality_accepted=False,
        training_admission=False,
        optimizer_updates=0,
        gpu_used=False,
        renderer=file_ref(Path(__file__)),
        pixel_renderer=file_ref(Path(__file__).with_name("visualize_scannetpp_native_outcomes.py")),
    )
    publish_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.scope.resolve(), args.output.resolve())
