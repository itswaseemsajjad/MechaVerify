"""MechaVerify command line. Generate datasets, render samples, evaluate models and
build real-drawing manifests."""

from __future__ import annotations

import argparse
import json
import os

from mechaverify.dataset import build_instance, default_plans, generate_dataset
from mechaverify.tasks import list_tasks


def _cmd_tasks(args: argparse.Namespace) -> None:
    for t in list_tasks():
        print(t)


def _cmd_generate(args: argparse.Namespace) -> None:
    from mechaverify.render import RenderConfig
    tasks = args.tasks or list_tasks()
    plans = default_plans(args.n)
    from mechaverify.tasks import dfma
    dfma.DEFAULT_TAU = args.tau
    base_cfg = RenderConfig(sketch=args.sketch)
    card = generate_dataset(args.out, tasks=tasks, plans=plans,
                            seed0=args.seed0, render=not args.no_render,
                            base_cfg=base_cfg)
    print(json.dumps(card, indent=2))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from mechaverify.eval.providers import get_provider
    from mechaverify.eval.runner import run_suite
    provider = get_provider(args.provider, args.model, temperature=args.temperature)
    conditions = [c.strip().upper() for c in args.conditions.split(",")]
    run_suite(args.manifest, provider, conditions, limit=args.limit,
              delay=args.delay, out_path=args.out, model=args.model or args.provider,
              split=args.split, per_task=args.per_task)


def _cmd_real_template(args: argparse.Namespace) -> None:
    from mechaverify.realset import write_template_csv
    write_template_csv(args.out)
    print(f"wrote labeling template -> {args.out}")


def _cmd_build_real(args: argparse.Namespace) -> None:
    from mechaverify.realset import build_real_manifest
    n = build_real_manifest(args.images, args.labels, args.out)
    print(f"wrote {n} real instances -> {args.out}")


def _cmd_draw_worklist(args: argparse.Namespace) -> None:
    from mechaverify.realset import write_worklist
    n = write_worklist(args.out, per_task=args.per_task)
    print(f"wrote {n}-item worklist -> {args.out}/worklist.md (+ labels.jsonl)")


def _cmd_build_drawn(args: argparse.Namespace) -> None:
    from mechaverify.realset import build_drawn_manifest
    n = build_drawn_manifest(args.images, args.labels, args.out)
    print(f"matched {n} hand-drawn images -> {args.out}")


def _cmd_sample(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    inst = build_instance(args.task, args.seed, args.difficulty, "id_test",
                          args.variant, args.out_dir, render=True)
    print(f"id:        {inst.id}")
    print(f"question:  {inst.question}")
    print(f"answer:    {inst.answer.value}  ({inst.answer.kind})")
    print(f"rationale: {inst.rationale}")
    print(f"image:     {os.path.join(args.out_dir, inst.image_path)}")
    print("--- symbolic description (condition S) ---")
    print(inst.symbolic_description)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mechaverify", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("tasks", help="list registered tasks")
    sp.set_defaults(func=_cmd_tasks)

    sp = sub.add_parser("generate", help="generate a dataset")
    sp.add_argument("--out", required=True, help="output directory")
    sp.add_argument("--tasks", nargs="*", default=None, help="tasks (default: all)")
    sp.add_argument("--n", type=int, default=50, help="instances per task in train split (60 gives the 825-instance v0.1 release)")
    sp.add_argument("--seed0", type=int, default=1_000_000)
    sp.add_argument("--no-render", action="store_true", help="skip PNG rendering")
    sp.add_argument("--sketch", action="store_true",
                    help="hand-drawn style rendering (exact labels; for the style-shift test)")
    sp.add_argument("--tau", type=float, default=4.0,
                    help="depth-to-diameter deflection threshold for dfma_deflection (4 in the paper)")
    sp.set_defaults(func=_cmd_generate)

    sp = sub.add_parser("evaluate", help="evaluate a model over a manifest (conditions D/S/DS)")
    sp.add_argument("--manifest", required=True, help="path to manifest.jsonl")
    sp.add_argument("--provider", required=True, help="gemini | openrouter | dummy")
    sp.add_argument("--model", default=None, help="model id (e.g. gemini-2.5-pro, openai/gpt-5)")
    sp.add_argument("--conditions", default="D,S", help="comma-separated: D,S,DS (DS is the D+S condition)")
    sp.add_argument("--split", default=None, help="restrict to a split (e.g. id_test)")
    sp.add_argument("--per-task", type=int, default=None, dest="per_task",
                    help="balanced subset: at most N instances per task")
    sp.add_argument("--limit", type=int, default=None, help="evaluate only the first N instances")
    sp.add_argument("--delay", type=float, default=0.0, help="seconds between calls (rate limits)")
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--out", default="runs/eval.json", help="output report path")
    sp.set_defaults(func=_cmd_evaluate)

    sp = sub.add_parser("real-template", help="write a CSV labeling template for the real test set")
    sp.add_argument("--out", default="real/labels.csv")
    sp.set_defaults(func=_cmd_real_template)

    sp = sub.add_parser("build-real", help="build a manifest from labeled real images")
    sp.add_argument("--images", required=True, help="folder of real diagram images")
    sp.add_argument("--labels", required=True, help="filled-in labels CSV")
    sp.add_argument("--out", default="real/manifest.jsonl")
    sp.set_defaults(func=_cmd_build_real)

    sp = sub.add_parser("draw-worklist", help="generate a hand-drawing worklist (+ known labels)")
    sp.add_argument("--out", default="real_drawn")
    sp.add_argument("--per-task", type=int, default=6, dest="per_task")
    sp.set_defaults(func=_cmd_draw_worklist)

    sp = sub.add_parser("build-drawn", help="match hand-drawn photos (<id>.png) to known labels")
    sp.add_argument("--images", required=True, help="folder of your drawn photos named <id>.png")
    sp.add_argument("--labels", default="real_drawn/labels.jsonl")
    sp.add_argument("--out", default="real_drawn/manifest.jsonl")
    sp.set_defaults(func=_cmd_build_drawn)

    sp = sub.add_parser("sample", help="render a single sample and print Q/A")
    sp.add_argument("--task", required=True)
    sp.add_argument("--difficulty", type=int, default=3)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--variant", default="mixed")
    sp.add_argument("--out-dir", default="outputs/samples")
    sp.set_defaults(func=_cmd_sample)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
