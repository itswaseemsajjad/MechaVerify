"""Turn a MechaVerify manifest into training records, plus GRPO rewards.

All functions here are CPU-only and dependency-light (no torch/transformers), so
they are covered by the unit tests in ``tests/``; the notebooks apply the model
processor on top of the chat records returned here.

Record format (Qwen2.5-VL chat style)::

    {"messages": [{"role": "system",   "content": [{"type": "text",  "text": ...}]},
                  {"role": "user",      "content": [{"type": "image", "image": path},
                                                    {"type": "text",  "text": question}]},
                  {"role": "assistant", "content": [{"type": "text",  "text": completion}]}],
     "image": path}

For RL records the assistant turn is omitted and the ground-truth answer is
carried in a JSON column so the reward function can score generations.
"""

from __future__ import annotations

import json
import os

from mechaverify.dataset import read_manifest
from mechaverify.eval.prompts import SYSTEM
from mechaverify.specs import Answer, Instance
from mechaverify.verify import check_answer, parse_response


def _answer_text(inst: Instance) -> str:
    a = inst.answer
    return f"{float(a.value):g}" if a.kind == "numeric" else str(a.value)


def target_completion(inst: Instance) -> str:
    """Gold completion in the think/answer format used for SFT supervision."""
    return f"<think>{inst.rationale}</think>\n<answer>{_answer_text(inst)}</answer>"


def to_chat(inst: Instance, dataset_dir: str, include_target: bool = True) -> dict:
    image_path = os.path.join(dataset_dir, inst.image_path) if inst.image_path else None
    user_content: list = []
    if image_path:
        user_content.append({"type": "image", "image": image_path})
    user_content.append({"type": "text", "text": inst.question})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": user_content},
    ]
    if include_target:
        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": target_completion(inst)}]})
    return {"messages": messages, "image": image_path}


def _load(manifest_path: str, split: str | None, tasks: list[str] | None):
    dataset_dir = os.path.dirname(os.path.abspath(manifest_path))
    insts = read_manifest(manifest_path)
    if split:
        insts = [i for i in insts if i.split == split]
    if tasks:
        insts = [i for i in insts if i.task in tasks]
    return insts, dataset_dir


def to_sft_records(manifest_path: str, split: str | None = "train",
                   tasks: list[str] | None = None) -> list[dict]:
    insts, d = _load(manifest_path, split, tasks)
    return [to_chat(i, d, include_target=True) for i in insts]


def to_rl_records(manifest_path: str, split: str | None = "train",
                  tasks: list[str] | None = None) -> list[dict]:
    insts, d = _load(manifest_path, split, tasks)
    out = []
    for i in insts:
        rec = to_chat(i, d, include_target=False)
        rec["answer"] = json.dumps(i.answer.to_dict())  # ground truth for the reward
        rec["task"] = i.task
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# GRPO reward functions (signature matches TRL: receive lists, return list[float])
# --------------------------------------------------------------------------- #
def _completion_text(completion) -> str:
    """Accept either a raw string or a chat-style [{'role','content'}] completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            c = last.get("content", "")
            if isinstance(c, list):  # content may be a list of parts
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c
    return str(completion)


def make_reward_funcs(format_weight: float = 0.1):
    """Return [correctness_reward, format_reward] for trl.GRPOTrainer.

    The trainer passes generated ``completions`` and any extra dataset columns
    (here ``answer``) as parallel lists. Correctness is the binary verifier
    signal; format rewards a response with a well-formed <answer> tag.
    """

    def correctness_reward(completions, answer, **kwargs):
        rewards = []
        for comp, ans_json in zip(completions, answer):
            a = Answer.from_dict(json.loads(ans_json))
            parsed = parse_response(_completion_text(comp))
            ok, _, _ = check_answer(a, parsed.answer_text)
            rewards.append(1.0 if ok else 0.0)
        return rewards

    def format_reward(completions, **kwargs):
        return [format_weight if parse_response(_completion_text(c)).format_ok else 0.0
                for c in completions]

    correctness_reward.__name__ = "correctness_reward"
    format_reward.__name__ = "format_reward"
    return [correctness_reward, format_reward]
