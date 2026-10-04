"""Prompt construction for the three diagnostic input conditions.

    D   : the diagram alone (image + question)
    S   : a matched gold symbolic description (text only, no image)
    D+S : the diagram together with the model's own generated description

The perception gap is acc(S) - acc(D): how much the model improves when given a
faithful symbolic description instead of having to read the picture.
"""

from __future__ import annotations

from mechaverify.specs import Instance

SYSTEM = (
    "You are an expert mechanical engineer analyzing an engineering drawing. "
    "Reason step by step inside <think> </think> tags, then give ONLY the final "
    "answer inside <answer> </answer> tags. For a numeric answer, put a single "
    "number inside <answer>. For a categorical answer, put exactly one of the "
    "allowed option words inside <answer>."
)

DESCRIBE = (
    "Describe this engineering schematic as precisely and completely as you can. "
    "List every component, label, tooth count or dimension, joint, and connection "
    "you can read from the drawing. Do NOT attempt to answer any question yet; "
    "only transcribe what is visually present."
)

CONDITIONS = ("D", "S", "DS")


def build_prompt(instance: Instance, condition: str, description: str | None = None) -> str:
    """Return the user prompt text for a given condition (image is attached
    separately by the provider for D and DS)."""
    if condition == "D":
        return instance.question
    if condition == "S":
        return (f"Symbolic description of the mechanism:\n{instance.symbolic_description}"
                f"\n\n{instance.question}")
    if condition == "DS":
        return (f"Here is a description of the drawing:\n{description}\n\n"
                f"Using the drawing and the description, answer:\n{instance.question}")
    raise ValueError(f"unknown condition {condition!r} (use one of {CONDITIONS})")


def uses_image(condition: str) -> bool:
    return condition in ("D", "DS")
