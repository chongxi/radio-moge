import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import review_structural_rgb as review


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def ref(path):
    return dict(path=str(path), sha256=review.sha(path))


@pytest.fixture
def source(tmp_path):
    group = "structural/test/000001"
    rows = []
    for slot in (0, 10, 11, 18):
        image = Image.new("RGB", (640, 480), (slot + 10, 20, 30))
        path = tmp_path / f"{slot}.png"
        image.save(path)
        row = dict(image=str(path), sha256=review.sha(path),
                   decoded_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                   group=group, slot=slot, projection="perspective")
        if slot == 11:
            rejection = dict(scene=group, slot=slot, image=ref(path),
                             decoded_rgb_sha256=row["decoded_rgb_sha256"],
                             requested_camera=dict(orthographic=False),
                             reason="Exterior/background is visible",
                             training_admission=False, retained_for_diagnosis_only=True)
        else:
            rows.append(row)
    manifest = tmp_path / "rgb.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    proof = tmp_path / "rejection.json"
    write(proof, rejection)
    report = tmp_path / "report.json"
    write(report, dict(stage="frozen_verified_structural_feature_rgb",
                       manifest=ref(manifest), rejection_proofs=[ref(proof)]))
    return report, manifest, proof


def test_default_refuses_missing_slot_before_output(source, tmp_path):
    report, _, _ = source
    with pytest.raises(ValueError, match="fixed review slot"):
        review.prepare(report, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_diagnostic_retains_slot_without_promoting_or_replacing(source, tmp_path):
    report, manifest, _ = source
    original = manifest.read_bytes()
    result = review.prepare(report, tmp_path / "review", include_rejected_for_review=True)
    assert [r["slot"] for r in result["selected"]] == [0, 10, 11, 18]
    assert result["admitted_sample_images"] == 3
    assert result["rejected_diagnostic_images"] == 1
    assert result["replacement_images"] == 0
    assert result["selected"][2]["training_admission"] is False
    assert manifest.read_bytes() == original
    with Image.open(result["pages"][0]["path"]) as image:
        assert image.getpixel((0, 508)) == (130, 12, 18)
        assert image.getpixel((0, 540)) == (21, 20, 30)


@pytest.mark.parametrize("mutation", ["unbound_receipt", "admitted_receipt", "wrong_slot", "image_tamper"])
def test_rejected_diagnostic_requires_bound_unadmitted_exact_identity(source, tmp_path, mutation):
    report, _, proof = source
    receipt = json.loads(proof.read_text())
    if mutation == "image_tamper":
        Path(receipt["image"]["path"]).write_bytes(b"wrong image")
    else:
        if mutation == "admitted_receipt":
            receipt["training_admission"] = True
        elif mutation == "wrong_slot":
            receipt["slot"] = 12
        else:
            receipt["reason"] = "changed after report"
        write(proof, receipt)
        if mutation != "unbound_receipt":
            obj = json.loads(report.read_text())
            obj["rejection_proofs"] = [ref(proof)]
            write(report, obj)
    with pytest.raises(ValueError):
        review.prepare(report, tmp_path / "review", include_rejected_for_review=True)
