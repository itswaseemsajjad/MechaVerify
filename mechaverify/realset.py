"""Build a MechaVerify manifest for the synthetic-to-real transfer test set.

Hand-drawn and textbook mechanism diagrams are labeled by a domain expert (the
answer is computable from the depicted mechanism) and described in a CSV; this
turns that CSV + an images folder into a standard ``manifest.jsonl`` so the real
set flows through the *same* verifier and evaluation harness as synthetic data.

CSV columns (header required):
    image            file name (relative to the images dir)        [required]
    family           kinematics | dfma                              [required]
    task             gear_sense | gear_ratio | fourbar_grashof |
                     fourbar_rotatability | mobility | dfma_fit |
                     dfma_deflection                                [required]
    question         the prompt shown to the model                 [required]
    answer_value     the closed-form answer (string or number)     [required]
    answer_kind      categorical | numeric                         [required]
    choices          '|'-separated options (categorical, optional)
    tolerance        relative tolerance (numeric, optional)
    difficulty       integer (optional, default 0)
    source           citation / provenance of the image (optional)
    symbolic_description  optional text for diagnostic condition S
"""

from __future__ import annotations

import csv
import json
import os

from mechaverify.specs import Answer, Instance

REQUIRED = ("image", "family", "task", "question", "answer_value", "answer_kind")

TEMPLATE_HEADER = [
    "image", "family", "task", "question", "answer_value", "answer_kind",
    "choices", "tolerance", "difficulty", "source", "symbolic_description",
]


def write_template_csv(path: str) -> None:
    """Write an empty labeling template (header + one example row)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(TEMPLATE_HEADER)
        w.writerow([
            "gear_train_textbook_01.png", "kinematics", "gear_sense",
            "The input gear (leftmost) rotates clockwise. Which way does the output "
            "gear rotate? Answer 'clockwise' or 'counterclockwise'.",
            "counterclockwise", "categorical", "clockwise|counterclockwise",
            "", "2", "Norton, Design of Machinery, Fig. X", "",
        ])


def _row_to_instance(i: int, row: dict, images_dir: str) -> Instance:
    row = {k: (v or "") for k, v in row.items() if k}
    missing = [c for c in REQUIRED if not (row.get(c) or "").strip()]
    if missing:
        raise ValueError(f"row {i}: missing required columns {missing}")
    kind = row["answer_kind"].strip().lower()
    choices = [c.strip() for c in row.get("choices", "").split("|") if c.strip()] or None
    tol = row.get("tolerance", "").strip()
    answer = Answer(
        value=row["answer_value"].strip(), kind=kind,
        tolerance=float(tol) if (kind == "numeric" and tol) else None,
        choices=choices,
    )
    return Instance(
        id=f"real-{i:04d}", family=row["family"].strip(), task=row["task"].strip(),
        split="real", difficulty=int(row.get("difficulty", "0") or 0), seed=0,
        spec={"type": "real", "source": row.get("source", "").strip()},
        question=row["question"].strip(), answer=answer, rationale="",
        symbolic_description=row.get("symbolic_description", "").strip(),
        image_path=row["image"].strip(),
        meta={"source": row.get("source", "").strip()},
    )


def write_worklist(out_dir: str, per_task: int = 6, seed0: int = 2_000_000) -> int:
    """Generate a hand-drawing worklist: per item, a full spec to *draw* and the
    question, plus a hidden labels.jsonl with the known answers. Because the user
    draws from the given spec, the labels are correct by construction (no eyeballing
    of real drawings, so ground truth stays trustworthy). Returns item count."""
    from mechaverify.tasks import generate, list_tasks, FAMILY_OF
    os.makedirs(out_dir, exist_ok=True)
    from mechaverify.rng import SeededRNG
    items: list[Instance] = []
    seed = seed0
    for task in list_tasks():
        for k in range(per_task):
            gr = generate(task, SeededRNG(seed), difficulty=2 + (k % 3))
            iid = f"draw-{task}-{k:02d}"
            items.append(Instance(
                id=iid, family=FAMILY_OF[task], task=task, split="real",
                difficulty=2 + (k % 3), seed=seed, spec={"type": "real_drawn"},
                question=gr.question, answer=gr.answer, rationale=gr.rationale,
                symbolic_description=gr.symbolic_description, image_path=None,
                meta={"draw_from": gr.symbolic_description}))
            seed += 1
    # labels.jsonl (full ground truth, keyed by id)
    with open(os.path.join(out_dir, "labels.jsonl"), "w", encoding="utf-8") as f:
        for inst in items:
            f.write(json.dumps(inst.to_dict(), ensure_ascii=False) + "\n")
    # worklist.md (drawing instructions; no answers, to keep it clean)
    lines = ["# MechaVerify hand-drawing worklist",
             "", f"{len(items)} items. For each: draw the mechanism described, by hand, on paper; "
             "photograph/scan it; save the file as `<id>.png` (or .jpg) in your photos folder. "
             "Then run `build-drawn`. Labels are already known from the spec. Do not re-label.", ""]
    cur = None
    for inst in items:
        if inst.task != cur:
            cur = inst.task
            lines.append(f"\n## {cur}\n")
        lines.append(f"### {inst.id}")
        lines.append(f"**Draw:**\n```\n{inst.symbolic_description}\n```")
        lines.append(f"**Question (for reference):** {inst.question}")
        lines.append(f"**Save photo as:** `{inst.id}.png`\n")
    with open(os.path.join(out_dir, "worklist.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return len(items)


def _rel_to(path: str, start: str) -> str:
    """Image path as stored in a manifest: relative to the manifest's folder, which
    is where the evaluation harness resolves it."""
    try:
        return os.path.relpath(os.path.abspath(path), start).replace(os.sep, "/")
    except ValueError:  # different drives on Windows
        return os.path.abspath(path)


def build_drawn_manifest(images_dir: str, labels_jsonl: str, out_manifest: str) -> int:
    """Match hand-drawn photos (named <id>.*) to the known labels and write a
    manifest. Returns the number matched."""
    from mechaverify.dataset import read_manifest
    labels = read_manifest(labels_jsonl)
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    parent = os.path.dirname(os.path.abspath(out_manifest))
    matched = []
    for inst in labels:
        found = next((f"{inst.id}{e}" for e in exts
                      if os.path.exists(os.path.join(images_dir, f"{inst.id}{e}"))), None)
        if found:
            inst.image_path = _rel_to(os.path.join(images_dir, found), parent)
            matched.append(inst)
    os.makedirs(parent, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        for inst in matched:
            f.write(json.dumps(inst.to_dict(), ensure_ascii=False) + "\n")
    return len(matched)


def build_real_manifest(images_dir: str, labels_csv: str, out_manifest: str) -> int:
    """Read the labeling CSV, validate images exist, write manifest.jsonl.
    Returns the number of instances written."""
    with open(labels_csv, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    parent = os.path.dirname(os.path.abspath(out_manifest))
    instances = []
    for i, row in enumerate(rows):
        inst = _row_to_instance(i, row, images_dir)
        img = os.path.join(images_dir, inst.image_path)
        if not os.path.exists(img):
            raise FileNotFoundError(f"row {i}: image not found: {img}")
        inst.image_path = _rel_to(img, parent)
        instances.append(inst)
    os.makedirs(parent, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst.to_dict(), ensure_ascii=False) + "\n")
    return len(instances)
