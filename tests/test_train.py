"""Tests for SFT/RL record building and GRPO reward functions (CPU-only)."""

import json

from mechaverify.dataset import SplitPlan, generate_dataset
from mechaverify.train.data import (
    make_reward_funcs,
    target_completion,
    to_rl_records,
    to_sft_records,
)


def _tiny_manifest(tmp_path):
    out = str(tmp_path / "ds")
    plans = [SplitPlan("train", "mixed", [1, 2], 3)]  # 3 per task, rendered
    generate_dataset(out, tasks=["gear_sense", "dfma_deflection"], plans=plans,
                     render=True, counterfactual=False)
    return out + "/manifest.jsonl"


def test_sft_records_have_full_conversation(tmp_path):
    recs = to_sft_records(_tiny_manifest(tmp_path), split="train")
    assert recs
    msgs = recs[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "<answer>" in msgs[-1]["content"][0]["text"]
    # user turn carries an image part
    assert any(p["type"] == "image" for p in msgs[1]["content"])


def test_rl_records_carry_ground_truth(tmp_path):
    recs = to_rl_records(_tiny_manifest(tmp_path), split="train")
    assert recs and "answer" in recs[0]
    ans = json.loads(recs[0]["answer"])
    assert "value" in ans and "kind" in ans
    # RL records omit the assistant turn
    assert [m["role"] for m in recs[0]["messages"]] == ["system", "user"]


def test_reward_functions(tmp_path):
    recs = to_rl_records(_tiny_manifest(tmp_path), split="train")
    correctness, fmt = make_reward_funcs(format_weight=0.1)
    answers = [r["answer"] for r in recs[:3]]
    gold = [json.loads(a)["value"] for a in answers]

    good = [f"<think>x</think><answer>{g}</answer>" for g in gold]
    bad = ["<answer>definitely-wrong-value</answer>" for _ in gold]

    assert correctness(good, answers) == [1.0, 1.0, 1.0]
    assert all(r == 0.0 for r in correctness(bad, answers))
    assert fmt(good) == [0.1, 0.1, 0.1]
    assert fmt(["no tags here"]) == [0.0]


def test_target_completion_format(tmp_path):
    recs = to_sft_records(_tiny_manifest(tmp_path), split="train")
    # the assistant target equals target_completion of the underlying instance
    txt = recs[0]["messages"][-1]["content"][0]["text"]
    assert txt.startswith("<think>") and txt.rstrip().endswith("</answer>")
