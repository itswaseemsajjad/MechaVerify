"""Closed-form correctness tests for the gear-train family."""

from fractions import Fraction

import pytest

from mechaverify.rng import SeededRNG
from mechaverify.tasks import generate
from mechaverify.tasks.gears import sample_gear_train, solve_gear_train


def test_simple_train_telescopes():
    # A simple train (all idlers) reduces to N_in / N_out regardless of idlers.
    spec = {"type": "gear_train", "shaft_gears": [[20], [40], [60]],
            "input_dir": "clockwise", "all_external": True}
    sol = solve_gear_train(spec)
    assert Fraction(*sol["ratio_fraction"]) == Fraction(20, 60)
    assert sol["ratio_magnitude"] == pytest.approx(1 / 3)


def test_compound_does_not_telescope():
    spec = {"type": "gear_train", "shaft_gears": [[20], [40, 15], [60]],
            "input_dir": "clockwise", "all_external": True}
    sol = solve_gear_train(spec)
    assert Fraction(*sol["ratio_fraction"]) == Fraction(1, 8)
    assert sol["ratio_magnitude"] == pytest.approx(0.125)


def test_sense_parity():
    even = {"type": "gear_train", "shaft_gears": [[20], [40], [60]],
            "input_dir": "clockwise", "all_external": True}
    odd = {"type": "gear_train", "shaft_gears": [[20], [40], [30], [60]],
           "input_dir": "clockwise", "all_external": True}
    assert solve_gear_train(even)["output_dir"] == "clockwise"      # 2 meshes
    assert solve_gear_train(odd)["output_dir"] == "counterclockwise"  # 3 meshes


def test_idler_changes_sense_not_magnitude():
    base = {"type": "gear_train", "shaft_gears": [[20], [40], [60]],
            "input_dir": "clockwise", "all_external": True}
    with_idler = {"type": "gear_train", "shaft_gears": [[20], [40], [30], [60]],
                  "input_dir": "clockwise", "all_external": True}
    a, b = solve_gear_train(base), solve_gear_train(with_idler)
    assert a["ratio_magnitude"] == pytest.approx(b["ratio_magnitude"])
    assert a["output_dir"] != b["output_dir"]


def test_adversarial_idler_invariant():
    # adv_idler forces a simple train, so a large idler must not affect magnitude.
    rng = SeededRNG(42)
    spec = sample_gear_train(rng, 4, variant="adv_idler")
    sol = solve_gear_train(spec)
    expected = spec["shaft_gears"][0][0] / spec["shaft_gears"][-1][0]
    assert sol["ratio_magnitude"] == pytest.approx(expected)


def test_determinism():
    a = sample_gear_train(SeededRNG(7), 3, "mixed")
    b = sample_gear_train(SeededRNG(7), 3, "mixed")
    assert a == b


def test_generate_produces_consistent_answer():
    gr = generate("gear_sense", SeededRNG(123), 3, "mixed")
    sol = solve_gear_train(gr.spec)
    assert gr.answer.value == sol["output_dir"]
    assert gr.answer.value in ("clockwise", "counterclockwise")


def test_difficulty_bounds():
    for d in range(1, 9):
        spec = sample_gear_train(SeededRNG(d), d, "mixed")
        n_mesh = len(spec["shaft_gears"]) - 1
        assert 1 <= n_mesh <= 6
