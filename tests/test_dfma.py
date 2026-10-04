"""Tests for the DFMA (tool-fit + deflection) family."""

from mechaverify.rng import SeededRNG
from mechaverify.tasks import generate
from mechaverify.tasks.dfma import sample_deflection, solve_deflection, solve_fit


def test_fit_corner_rule():
    # tool radius 3 (Ø6) vs corner radius 2 -> corner too sharp
    s = solve_fit(W=20, R_c=2, D=6)
    assert s["machinable"] is False and s["binding"] == "corner radius"
    # tool fits both corner and width
    s = solve_fit(W=20, R_c=5, D=8)
    assert s["machinable"] is True


def test_fit_width_rule():
    # corner is fine (R_c >= r) but tool wider than the pocket
    s = solve_fit(W=10, R_c=8, D=12)
    assert s["machinable"] is False and s["binding"] == "pocket width"


def test_deflection_threshold():
    assert solve_deflection(h=9, D=2, tau=4)["risk"] is True       # 4.5 > 4
    assert solve_deflection(h=8, D=4, tau=4)["risk"] is False      # 2.0 <= 4
    # exactly at the threshold is NOT a risk (the rule is strict: h/D > tau)
    assert solve_deflection(h=16, D=4, tau=4)["ratio"] == 4.0
    assert solve_deflection(h=16, D=4, tau=4)["risk"] is False
    assert solve_deflection(h=20, D=4, tau=4)["risk"] is True      # 5.0 > 4


def test_adversarial_deflection_is_a_real_trap():
    # the depth/width heuristic should disagree with the true depth/diameter rule
    p = sample_deflection(SeededRNG(8), 3, "adv_idler")
    by_diameter = solve_deflection(p["depth"], p["tool_diameter"], p["tau"])["risk"]
    by_width = (p["depth"] / p["width"]) > p["tau"]
    assert by_diameter is True and by_width is False


def test_generate_answer_matches_solver():
    gr = generate("dfma_fit", SeededRNG(4), 2, "mixed")
    s = solve_fit(gr.spec["width"], gr.spec["corner_radius"], gr.spec["tool_diameter"])
    assert gr.answer.value == ("yes" if s["machinable"] else "no")

    gd = generate("dfma_deflection", SeededRNG(6), 3, "mixed")
    sd = solve_deflection(gd.spec["depth"], gd.spec["tool_diameter"], gd.spec["tau"])
    assert gd.answer.value == ("yes" if sd["risk"] else "no")
