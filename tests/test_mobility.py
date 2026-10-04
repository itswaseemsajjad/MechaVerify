"""Tests for the Kutzbach-Grubler mobility family."""

import pytest

from mechaverify.rng import SeededRNG
from mechaverify.tasks import generate
from mechaverify.tasks import mobility as mob


def _count_joints(spec):
    j1 = sum(1 for j in spec["joints"] if j["kind"] in ("revolute", "prismatic"))
    j2 = len(spec.get("higher_pairs", []))
    return j1, j2


@pytest.mark.parametrize("fn,M", [
    (lambda r: mob._loop(r, 4), 1),
    (lambda r: mob._loop(r, 5), 2),
    (lambda r: mob._loop(r, 6), 3),
    (lambda r: mob._loop(r, 7), 4),
    (mob._slider_crank, 1),
    (mob._cam_follower, 1),
    (mob._stephenson_six, 1),
])
def test_template_mobility(fn, M):
    spec = fn(SeededRNG(5))
    assert spec["true_M"] == M


def test_formula_consistency_and_drawn_counts():
    for fn in (lambda r: mob._loop(r, 5), mob._slider_crank, mob._cam_follower,
               mob._stephenson_six, mob._parallelogram_paradox):
        spec = fn(SeededRNG(9))
        n = spec["n_links"]
        j1, j2 = _count_joints(spec)
        # stored counts match the drawable structure
        assert n == len(spec["links"])
        assert (j1, j2) == (spec["j1"], spec["j2"])
        # Kutzbach formula holds
        assert spec["kutzbach_M"] == 3 * (n - 1) - 2 * j1 - j2


def test_paradox_diverges():
    spec = mob._parallelogram_paradox(SeededRNG(2))
    assert spec["paradox"] is True
    assert spec["kutzbach_M"] == 0 and spec["true_M"] == 1


def test_generate_answer_is_true_mobility():
    gr = generate("mobility", SeededRNG(3), 4, "mixed")
    assert float(gr.answer.value) == float(gr.spec["true_M"])
    # adversarial split returns the paradox, answered with the true mobility
    adv = generate("mobility", SeededRNG(3), 3, "adv_idler")
    assert adv.spec["paradox"] and float(adv.answer.value) == 1.0
