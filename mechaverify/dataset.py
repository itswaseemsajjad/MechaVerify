"""Dataset assembly and manifest I/O.

A MechaVerify dataset is a directory containing:
    images/                 rendered PNG schematics
    manifest.jsonl          one JSON record per instance (the full Instance)
    dataset_card.json       counts, tasks, splits, seeds and generation params

Everything is regenerable from ``dataset_card.json`` (seeds + params), so the
heavy PNGs need never be committed to version control.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from mechaverify.render import RenderConfig, render_to_file
from mechaverify.rng import SeededRNG
from mechaverify.specs import Instance
from mechaverify.tasks import COUNTERFACTUAL, FAMILY_OF, generate


@dataclass
class SplitPlan:
    name: str            # split id (train / id_test / ood_test / adversarial)
    variant: str         # generator variant
    difficulties: list[int]
    n_per_task: int      # instances per task in this split


def default_plans(n: int = 50) -> list[SplitPlan]:
    """A small, balanced default. OOD uses longer trains than train/id."""
    return [
        SplitPlan("train", "mixed", [1, 2, 3, 4], n),
        SplitPlan("id_test", "mixed", [1, 2, 3, 4], max(10, n // 4)),
        SplitPlan("ood_test", "mixed", [5, 6], max(10, n // 4)),
        SplitPlan("adversarial", "adv_idler", [2, 3, 4], max(10, n // 4)),
    ]


def render_config_for(difficulty: int, base: RenderConfig | None = None) -> RenderConfig:
    """Scale visual difficulty (distractor density, perturbation) with difficulty."""
    base = base or RenderConfig()
    return RenderConfig(
        dpi=base.dpi,
        show_title_block=base.show_title_block,
        show_dimensions=base.show_dimensions,
        distractor_density=min(1.0, 0.15 * difficulty),
        perturb_deg=base.perturb_deg,
        title=base.title,
        fontsize=base.fontsize,
        sketch=base.sketch,
    )


def build_instance(task: str, seed: int, difficulty: int, split: str, variant: str,
                   out_dir: str, render: bool = True,
                   base_cfg: RenderConfig | None = None) -> Instance:
    rng = SeededRNG(seed)
    gr = generate(task, rng, difficulty, variant)
    iid = f"{task}-{split}-{seed:08d}"
    inst = Instance(
        id=iid, family=FAMILY_OF[task], task=task, split=split,
        difficulty=difficulty, seed=seed, spec=gr.spec, question=gr.question,
        answer=gr.answer, rationale=gr.rationale,
        symbolic_description=gr.symbolic_description,
    )
    if render:
        rel = os.path.join("images", f"{iid}.png")
        render_to_file(gr.spec, os.path.join(out_dir, rel),
                       render_config_for(difficulty, base_cfg), seed)
        inst.image_path = rel.replace(os.sep, "/")
    return inst


def generate_dataset(out_dir: str, tasks: list[str], plans: list[SplitPlan] | None = None,
                     seed0: int = 1_000_000, render: bool = True,
                     counterfactual: bool = True,
                     base_cfg: RenderConfig | None = None) -> dict:
    """Generate a full dataset to ``out_dir``; returns the dataset card dict."""
    from mechaverify.tasks import dfma as _dfma
    plans = plans or default_plans()
    if tasks is None:
        from mechaverify.tasks import list_tasks
        tasks = list_tasks()
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    instances: list[Instance] = []
    seed = seed0

    for plan in plans:
        for task in tasks:
            for i in range(plan.n_per_task):
                difficulty = plan.difficulties[i % len(plan.difficulties)]
                instances.append(build_instance(task, seed, difficulty, plan.name,
                                                 plan.variant, out_dir, render, base_cfg))
                seed += 1

    # counterfactual split: minimal-edit pairs for tasks that support it
    if counterfactual:
        for task in tasks:
            if task not in COUNTERFACTUAL:
                continue
            for i in range(max(10, plans[0].n_per_task // 4)):
                rng = SeededRNG(seed)
                base_gr, edit_gr, edit_desc = COUNTERFACTUAL[task](rng, 3)
                for tag, gr in (("a", base_gr), ("b", edit_gr)):
                    iid = f"{task}-counterfactual-{seed:08d}{tag}"
                    inst = Instance(
                        id=iid, family=FAMILY_OF[task], task=task,
                        split="counterfactual", difficulty=3, seed=seed,
                        spec=gr.spec, question=gr.question, answer=gr.answer,
                        rationale=gr.rationale,
                        symbolic_description=gr.symbolic_description,
                        meta={"pair": tag, "edit": edit_desc},
                    )
                    if render:
                        rel = os.path.join("images", f"{iid}.png")
                        render_to_file(gr.spec, os.path.join(out_dir, rel),
                                       render_config_for(3, base_cfg), seed)
                        inst.image_path = rel.replace(os.sep, "/")
                    instances.append(inst)
                seed += 1

    write_manifest(out_dir, instances)
    card = {
        "name": os.path.basename(os.path.abspath(out_dir)),
        "version": "0.1.0",
        "n_instances": len(instances),
        "tasks": tasks,
        "splits": {p.name: p.n_per_task for p in plans},
        "counterfactual": counterfactual,
        "seed0": seed0,
        "seed_end": seed,
        "render": asdict(base_cfg or RenderConfig()),
        "dfma_tau": _dfma.DEFAULT_TAU,
        "plans": [asdict(p) for p in plans],
        "counts_by_split": _counts(instances, "split"),
        "counts_by_task": _counts(instances, "task"),
        "counts_by_difficulty": _counts(instances, "difficulty"),
    }
    with open(os.path.join(out_dir, "dataset_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    return card


def _counts(instances: list[Instance], attr: str) -> dict:
    out: dict = {}
    for inst in instances:
        k = str(getattr(inst, attr))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def write_manifest(out_dir: str, instances: list[Instance]) -> str:
    path = os.path.join(out_dir, "manifest.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for inst in instances:
            f.write(json.dumps(inst.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_manifest(path: str) -> list[Instance]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Instance.from_dict(json.loads(line)))
    return out
