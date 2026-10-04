"""The MechaVerify data model.

An ``Instance`` is one benchmark item: a symbolic specification, a rendered
schematic (referenced by path), a templated question, the closed-form answer,
a gold structured rationale, and a gold *symbolic description* used for the
perception-vs-reasoning diagnostic (condition S). Everything is JSON-serializable
so a dataset is a directory of PNGs plus a JSONL manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Canonical task / family / split identifiers ---------------------------------
FAMILIES = ("kinematics", "dfma")

SPLITS = ("train", "id_test", "ood_test", "adversarial", "counterfactual", "real")

ANSWER_KINDS = ("categorical", "numeric")


@dataclass
class Answer:
    """The ground-truth answer payload.

    For ``categorical`` answers, ``value`` is a canonical string (the verifier
    normalizes synonyms before comparison). For ``numeric`` answers, ``value`` is
    a float and ``tolerance`` defines the acceptance band (relative unless
    ``abs_tolerance`` is set).
    """

    value: Any
    kind: str  # one of ANSWER_KINDS
    unit: str | None = None
    tolerance: float | None = None       # relative tolerance for numeric answers
    abs_tolerance: float | None = None    # absolute tolerance (overrides relative if set)
    choices: list[str] | None = None      # optional closed answer set (categorical)

    def __post_init__(self) -> None:
        if self.kind not in ANSWER_KINDS:
            raise ValueError(f"unknown answer kind: {self.kind!r}")
        if self.kind == "numeric":
            self.value = float(self.value)
            if self.tolerance is None and self.abs_tolerance is None:
                self.tolerance = 0.01  # default 1% relative band

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Answer":
        return cls(**d)


@dataclass
class Instance:
    """A single, fully-specified benchmark item."""

    id: str
    family: str          # one of FAMILIES
    task: str            # e.g. "gear_ratio", "gear_sense", "fourbar_grashof", ...
    split: str           # one of SPLITS
    difficulty: int
    seed: int
    spec: dict[str, Any]            # the symbolic specification (regenerates the item)
    question: str                   # templated, image-facing prompt
    answer: Answer
    rationale: str                  # gold step-by-step solution from the solver
    symbolic_description: str       # gold text description for diagnostic condition S
    image_path: str | None = None   # relative path to the rendered PNG
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family: {self.family!r}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split: {self.split!r}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["answer"] = self.answer.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Instance":
        d = dict(d)
        d["answer"] = Answer.from_dict(d["answer"])
        return cls(**d)
