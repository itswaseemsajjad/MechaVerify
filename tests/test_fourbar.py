"""Tests for the four-bar Grashof family."""

import pytest

from mechaverify.rng import SeededRNG
from mechaverify.tasks import generate
from mechaverify.tasks.fourbar import (
    assemble,
    can_link_fully_rotate,
    classify_fourbar,
    sample_fourbar,
)


@pytest.mark.parametrize("L,expect", [
    ([8, 3, 7, 6], "crank-rocker"),    # shortest = input
    ([3, 8, 7, 6], "double-crank"),    # shortest = ground
    ([8, 7, 3, 6], "double-rocker"),   # shortest = coupler
    ([8, 3, 4, 6], "triple-rocker"),   # non-Grashof
    ([5, 7, 5, 7], "change-point"),    # s + l == p + q
])
def test_classification(L, expect):
    assert classify_fourbar([float(x) for x in L])["type"] == expect


def test_rotatability_rules():
    assert can_link_fully_rotate([8, 3, 7, 6], 1) is True    # input is the crank
    assert can_link_fully_rotate([8, 7, 6, 3], 1) is False   # output is the crank
    assert can_link_fully_rotate([3, 8, 7, 6], 1) is True    # ground shortest
    assert can_link_fully_rotate([8, 3, 4, 6], 1) is False   # non-Grashof


def test_generated_specs_are_valid_quadrilaterals():
    for seed in range(40):
        spec = sample_fourbar(SeededRNG(seed), difficulty=4, variant="mixed")
        L = spec["lengths"]
        assert max(L) < sum(L) - max(L)               # quadrilateral inequality
        assert len(set(L)) == 4                        # distinct (unambiguous shortest)
        assert assemble([float(x) for x in L], spec["input_angle_deg"]) is not None


def test_generate_answer_matches_solver():
    gr = generate("fourbar_grashof", SeededRNG(11), 4, "mixed")
    assert gr.answer.value == classify_fourbar([float(x) for x in gr.spec["lengths"]])["type"]


def test_change_point_variant():
    spec = sample_fourbar(SeededRNG(1), 3, variant="change_point")
    assert classify_fourbar([float(x) for x in spec["lengths"]])["type"] == "change-point"
