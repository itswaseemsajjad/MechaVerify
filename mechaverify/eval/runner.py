"""Run a provider over a manifest under one or more conditions, score with the
deterministic verifier, and aggregate accuracy / perception gap."""

from __future__ import annotations

import datetime
import json
import os
import time

from mechaverify.eval.prompts import DESCRIBE, build_prompt, uses_image
from mechaverify.eval.providers import Provider
from mechaverify.specs import Instance
from mechaverify.verify import verify


def _resolve_image(dataset_dir: str, inst: Instance) -> str | None:
    if not inst.image_path:
        return None
    return os.path.join(dataset_dir, inst.image_path)


def evaluate(instances: list[Instance], dataset_dir: str, provider: Provider,
             condition: str, limit: int | None = None, delay: float = 0.0,
             progress: bool = True) -> list[dict]:
    """Evaluate one condition; returns a per-instance result row list."""
    items = instances[:limit] if limit else instances
    rows: list[dict] = []
    for i, inst in enumerate(items):
        img = _resolve_image(dataset_dir, inst) if uses_image(condition) else None
        if condition == "DS":
            desc = provider.generate_with_retry(DESCRIBE, img)
            out = provider.generate_with_retry(build_prompt(inst, "DS", desc), img)
        else:
            out = provider.generate_with_retry(build_prompt(inst, condition), img)
        vr = verify(inst, out)
        rows.append({
            "id": inst.id, "task": inst.task, "family": inst.family,
            "split": inst.split, "difficulty": inst.difficulty,
            "condition": condition, "correct": bool(vr.correct),
            "format_ok": bool(vr.format_ok), "expected": vr.expected,
            "parsed": vr.parsed_answer, "raw": (out or "")[:2000],
        })
        if progress and (i + 1) % 10 == 0:
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"  [{condition}] {i + 1}/{len(items)}  running acc={acc:.3f}")
        if delay:
            time.sleep(delay)
    return rows


def _acc(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


def _group_acc(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(str(r[key]), []).append(r["correct"])
    return {k: {"acc": sum(v) / len(v), "n": len(v)} for k, v in sorted(out.items())}


def aggregate(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "overall_acc": _acc(rows),
        "format_rate": sum(r["format_ok"] for r in rows) / len(rows) if rows else 0.0,
        "by_task": _group_acc(rows, "task"),
        "by_split": _group_acc(rows, "split"),
        "by_difficulty": _group_acc(rows, "difficulty"),
    }


def perception_gap(summaries: dict[str, dict]) -> dict | None:
    """Perception gap = acc(S) - acc(D), overall and per task (needs both runs)."""
    if "D" not in summaries or "S" not in summaries:
        return None
    d, s = summaries["D"], summaries["S"]
    per_task = {}
    for task in d["by_task"]:
        if task in s["by_task"]:
            per_task[task] = round(s["by_task"][task]["acc"] - d["by_task"][task]["acc"], 4)
    return {"overall": round(s["overall_acc"] - d["overall_acc"], 4), "by_task": per_task}


def select_instances(instances: list[Instance], split: str | None = None,
                     per_task: int | None = None, limit: int | None = None) -> list[Instance]:
    """Filter to a split and/or take a balanced ``per_task`` subset (per-task cap)."""
    if split:
        instances = [i for i in instances if i.split == split]
    if per_task:
        by_task: dict[str, list[Instance]] = {}
        for i in instances:
            by_task.setdefault(i.task, [])
            if len(by_task[i.task]) < per_task:
                by_task[i.task].append(i)
        instances = [i for t in sorted(by_task) for i in by_task[t]]
    if limit:
        instances = instances[:limit]
    return instances


def run_suite(manifest_path: str, provider: Provider, conditions: list[str],
              limit: int | None = None, delay: float = 0.0,
              out_path: str | None = None, model: str = "",
              split: str | None = None, per_task: int | None = None) -> dict:
    from mechaverify.dataset import read_manifest
    dataset_dir = os.path.dirname(os.path.abspath(manifest_path))
    instances = select_instances(read_manifest(manifest_path), split, per_task, limit)
    report: dict = {"provider": provider.name, "model": model,
                    "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "temperature": getattr(provider, "temperature", None),
                    "max_tokens": getattr(provider, "max_tokens", None),
                    "manifest": manifest_path, "split": split, "per_task": per_task,
                    "n_eval": len(instances), "conditions": {}}
    summaries = {}
    for cond in conditions:
        print(f"== condition {cond} ==")
        rows = evaluate(instances, dataset_dir, provider, cond, None, delay)
        summ = aggregate(rows)
        summaries[cond] = summ
        report["conditions"][cond] = {"summary": summ, "rows": rows}
        print(f"   overall acc={summ['overall_acc']:.3f}  format={summ['format_rate']:.3f}")
    gap = perception_gap(summaries)
    if gap is not None:
        report["perception_gap"] = gap
        print(f"== perception gap (S - D) = {gap['overall']:+.3f} ==")
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"   wrote {out_path}")
    return report
