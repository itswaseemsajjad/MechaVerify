"""Tests for the synthetic-to-real manifest builder."""

import csv

import pytest

from mechaverify.dataset import read_manifest
from mechaverify.realset import build_real_manifest, write_template_csv
from mechaverify.verify import verify


def _img(path):
    # a tiny valid PNG so existence + image_path checks pass
    from PIL import Image
    Image.new("RGB", (8, 8), "white").save(path)


def test_template_has_required_columns(tmp_path):
    p = tmp_path / "labels.csv"
    write_template_csv(str(p))
    with open(p, encoding="utf-8") as f:
        header = next(csv.reader(f))
    for col in ("image", "family", "task", "question", "answer_value", "answer_kind"):
        assert col in header


def test_build_real_manifest_and_verify(tmp_path):
    images = tmp_path / "imgs"
    images.mkdir()
    _img(images / "a.png")
    _img(images / "b.png")
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "family", "task", "question", "answer_value",
                    "answer_kind", "choices", "tolerance"])
        w.writerow(["a.png", "kinematics", "gear_sense", "Which way?",
                    "clockwise", "categorical", "clockwise|counterclockwise", ""])
        w.writerow(["b.png", "kinematics", "gear_ratio", "Ratio?",
                    "0.5", "numeric", "", "0.02"])

    out = tmp_path / "manifest.jsonl"
    n = build_real_manifest(str(images), str(csv_path), str(out))
    assert n == 2
    insts = read_manifest(str(out))
    assert [i.split for i in insts] == ["real", "real"]
    # answers flow through the same verifier
    assert verify(insts[0], "<answer>cw</answer>").correct
    assert verify(insts[1], "<answer>0.5</answer>").correct
    assert not verify(insts[1], "<answer>2.0</answer>").correct


def test_worklist_and_build_drawn(tmp_path):
    from mechaverify.realset import build_drawn_manifest, write_worklist
    out = tmp_path / "wl"
    n = write_worklist(str(out), per_task=2)
    assert n > 0
    assert (out / "worklist.md").exists() and (out / "labels.jsonl").exists()
    labels = read_manifest(str(out / "labels.jsonl"))
    assert labels and all(i.split == "real" for i in labels)
    # simulate drawing the first two items (photos named <id>.png)
    photos = tmp_path / "photos"
    photos.mkdir()
    for inst in labels[:2]:
        _img(photos / f"{inst.id}.png")
    m = build_drawn_manifest(str(photos), str(out / "labels.jsonl"), str(tmp_path / "m.jsonl"))
    assert m == 2
    built = read_manifest(str(tmp_path / "m.jsonl"))
    # known answer flows through the verifier
    a = built[0].answer.value
    payload = f"{a:g}" if built[0].answer.kind == "numeric" else a
    assert verify(built[0], f"<answer>{payload}</answer>").correct


def test_missing_image_raises(tmp_path):
    images = tmp_path / "imgs"
    images.mkdir()
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "family", "task", "question", "answer_value", "answer_kind"])
        w.writerow(["missing.png", "dfma", "dfma_fit", "Machinable?", "no", "categorical"])
    with pytest.raises(FileNotFoundError):
        build_real_manifest(str(images), str(csv_path), str(tmp_path / "m.jsonl"))
