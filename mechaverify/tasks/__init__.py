"""Task-family registry.

Each task family contributes one or more *tasks* (question types). A task
provides a generator that, from a seeded RNG, produces a symbolic spec, a
templated question, the closed-form ``Answer``, a gold rationale, and a gold
symbolic description (for diagnostic condition S). Rendering is dispatched
separately on ``spec["type"]`` (see :mod:`mechaverify.render`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mechaverify.rng import SeededRNG
from mechaverify.specs import Answer


@dataclass
class GenResult:
    spec: dict
    question: str
    answer: Answer
    rationale: str
    symbolic_description: str


# task -> generator(rng, difficulty: int, variant: str) -> GenResult
GENERATORS: dict[str, Callable[[SeededRNG, int, str], GenResult]] = {}
# task -> family name
FAMILY_OF: dict[str, str] = {}
# family -> [tasks]
TASKS_BY_FAMILY: dict[str, list[str]] = {}
# task -> optional counterfactual maker(rng, difficulty) -> (base, edited, edit_desc)
COUNTERFACTUAL: dict[str, Callable[[SeededRNG, int], tuple[GenResult, GenResult, str]]] = {}


def register(
    task: str,
    family: str,
    gen_fn: Callable[[SeededRNG, int, str], GenResult],
    counterfactual_fn: Callable[[SeededRNG, int], tuple[GenResult, GenResult, str]] | None = None,
) -> None:
    GENERATORS[task] = gen_fn
    FAMILY_OF[task] = family
    TASKS_BY_FAMILY.setdefault(family, []).append(task)
    if counterfactual_fn is not None:
        COUNTERFACTUAL[task] = counterfactual_fn


def generate(task: str, rng: SeededRNG, difficulty: int, variant: str = "mixed") -> GenResult:
    if task not in GENERATORS:
        raise KeyError(f"unknown task {task!r}; registered: {sorted(GENERATORS)}")
    return GENERATORS[task](rng, difficulty, variant)


def list_tasks() -> list[str]:
    return sorted(GENERATORS)


# Importing a family module registers its tasks as a side effect.
from mechaverify.tasks import gears  # noqa: E402,F401
from mechaverify.tasks import fourbar  # noqa: E402,F401
from mechaverify.tasks import mobility  # noqa: E402,F401
from mechaverify.tasks import dfma  # noqa: E402,F401
