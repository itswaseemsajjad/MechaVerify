"""Task Family I (kinematics): planar four-bar linkages.

A four-bar is a closed loop of four links connected by revolute joints, in loop
order ``[ground, input, coupler, output]`` (index 0 is the fixed frame). With the
four lengths ordered s <= p <= q <= l, the Grashof condition for at least one
link to fully rotate is

    s + l <= p + q

The classification depends on which link is shortest relative to the ground:

    * shortest is the ground link            -> double-crank (drag-link)
    * shortest is a side link (input/output) -> crank-rocker
    * shortest is the coupler                 -> double-rocker (Grashof)
    * non-Grashof (s + l > p + q)            -> triple-rocker
    * s + l == p + q                         -> change-point (e.g. parallelogram)

A side link X (input or output) can make a full revolution iff the linkage is
Grashof and the shortest link is either the ground or X itself.
"""

from __future__ import annotations

import math

from mechaverify.rng import SeededRNG
from mechaverify.specs import Answer
from mechaverify.tasks import GenResult, register

_LABELS = ("ground", "input", "coupler", "output")
_TYPES = ["crank-rocker", "double-crank", "double-rocker", "triple-rocker", "change-point"]


# --------------------------------------------------------------------------- #
# Solving
# --------------------------------------------------------------------------- #
def classify_fourbar(lengths: list[float]) -> dict:
    L = list(lengths)
    s = sorted(L)
    S, P, Q, Lg = s
    margin = (S + Lg) - (P + Q)
    shortest_index = min(range(4), key=lambda i: L[i])
    if abs(margin) < 1e-9:
        gtype = "change-point"
    elif margin > 0:
        gtype = "triple-rocker"
    elif shortest_index == 0:
        gtype = "double-crank"
    elif shortest_index == 2:
        gtype = "double-rocker"
    else:
        gtype = "crank-rocker"
    grashof = margin <= 1e-9   # s + l <= p + q (change-point included)
    return {
        "type": gtype,
        "grashof": grashof,
        "margin": margin,
        "shortest_index": shortest_index,
        "s_plus_l": S + Lg,
        "p_plus_q": P + Q,
    }


def can_link_fully_rotate(lengths: list[float], link_index: int) -> bool:
    """True iff side link ``link_index`` (1=input or 3=output) fully rotates."""
    sol = classify_fourbar(lengths)
    if not sol["grashof"]:
        return False
    si = sol["shortest_index"]
    return si == 0 or si == link_index


# --------------------------------------------------------------------------- #
# Geometry for rendering (four-bar position solution)
# --------------------------------------------------------------------------- #
def assemble(lengths: list[float], input_deg: float) -> dict | None:
    """Return joint coordinates O2, A, B, O4 for a given input angle, or None
    if the linkage cannot be assembled at that angle."""
    g, a, b, c = lengths  # ground, input, coupler, output
    O2 = (0.0, 0.0)
    O4 = (g, 0.0)
    th = math.radians(input_deg)
    A = (a * math.cos(th), a * math.sin(th))
    # B is intersection of circle(A, b) and circle(O4, c)
    dx, dy = O4[0] - A[0], O4[1] - A[1]
    d = math.hypot(dx, dy)
    if d > b + c or d < abs(b - c) or d == 0:
        return None
    aa = (b * b - c * c + d * d) / (2 * d)
    h2 = b * b - aa * aa
    if h2 < 0:
        return None
    h = math.sqrt(h2)
    px, py = A[0] + aa * dx / d, A[1] + aa * dy / d
    B = (px - h * dy / d, py + h * dx / d)  # upper intersection
    return {"O2": O2, "A": A, "B": B, "O4": O4}


def _good_input_angle(lengths: list[float]) -> float:
    """Pick an input angle that yields a clean, assembled (non-degenerate) pose."""
    for deg in (60, 75, 45, 90, 110, 30, 120, 135, 150):
        pose = assemble(lengths, deg)
        if pose is not None:
            ax, ay = pose["A"]
            bx, by = pose["B"]
            if ay > 0.1 and by > -0.1:  # keep coupler above the ground line
                return float(deg)
    return 60.0


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _valid_quad(L: list[int]) -> bool:
    return max(L) < (sum(L) - max(L))  # quadrilateral inequality


def sample_fourbar(rng: SeededRNG, difficulty: int, variant: str = "mixed") -> dict:
    span = 8 + 3 * max(1, min(difficulty, 6))  # range widens with difficulty
    if variant == "change_point":
        # parallelogram-type: two equal pairs -> s + l == p + q exactly
        x = rng.randint(4, span)
        y = rng.randint(4, span)
        while y == x:
            y = rng.randint(4, span)
        lengths = [x, y, x, y]  # ground, input, coupler, output
        return {"type": "fourbar", "lengths": lengths,
                "input_angle_deg": _good_input_angle([float(v) for v in lengths]),
                "variant": variant}
    lo = 5
    for _ in range(4000):
        L = rng.sample(range(lo, span + 1), 4)
        if not _valid_quad(L):
            continue
        if max(L) > 2.6 * min(L):  # keep shapes compact / drawable
            continue
        sol = classify_fourbar([float(v) for v in L])
        if abs(sol["margin"]) < 1.0:  # avoid near-change-point ambiguity in normal splits
            continue
        if variant == "grashof" and not sol["grashof"]:
            continue
        if variant == "non_grashof" and sol["grashof"]:
            continue
        return {"type": "fourbar", "lengths": L,
                "input_angle_deg": _good_input_angle([float(v) for v in L]),
                "variant": variant}
    # fallback: a known crank-rocker
    L = [8, 3, 7, 6]
    return {"type": "fourbar", "lengths": L,
            "input_angle_deg": _good_input_angle([float(v) for v in L]),
            "variant": variant}


# --------------------------------------------------------------------------- #
# Descriptions
# --------------------------------------------------------------------------- #
def _symbolic_description(spec: dict, sol: dict) -> str:
    L = spec["lengths"]
    lines = ["Planar four-bar linkage (revolute joints). Link lengths:"]
    for lab, v in zip(_LABELS, L):
        tag = " [FIXED FRAME]" if lab == "ground" else ""
        lines.append(f"  {lab}{tag}: {v}")
    lines.append(f"Shortest + longest = {sol['s_plus_l']}; other two = {sol['p_plus_q']}.")
    return "\n".join(lines)


def _rationale_class(spec: dict, sol: dict) -> str:
    rel = "<=" if sol["s_plus_l"] <= sol["p_plus_q"] else ">"
    base = (f"Order the lengths: s+l = {sol['s_plus_l']}, p+q = {sol['p_plus_q']}, "
            f"so s+l {rel} p+q.")
    if sol["type"] == "change-point":
        return base + " Equality holds, so this is a change-point linkage."
    if not sol["grashof"]:
        return base + " The Grashof condition fails, so every inversion is a triple-rocker."
    where = {0: "the ground link", 1: "a side (input) link", 2: "the coupler",
             3: "a side (output) link"}[sol["shortest_index"]]
    return base + f" Grashof is satisfied and the shortest link is {where}, giving a {sol['type']}."


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def _gen_grashof(rng: SeededRNG, difficulty: int, variant: str = "mixed") -> GenResult:
    v = "change_point" if variant == "adv_idler" else variant
    spec = sample_fourbar(rng, difficulty, "mixed" if v == "mixed" else v)
    sol = classify_fourbar([float(x) for x in spec["lengths"]])
    q = ("Classify this four-bar linkage (the hatched link is the fixed frame). "
         "Answer one of: crank-rocker, double-crank, double-rocker, triple-rocker, "
         "change-point.")
    ans = Answer(value=sol["type"], kind="categorical", choices=_TYPES)
    return GenResult(spec, q, ans, _rationale_class(spec, sol),
                     _symbolic_description(spec, sol))


def _gen_rotatability(rng: SeededRNG, difficulty: int, variant: str = "mixed") -> GenResult:
    spec = sample_fourbar(rng, difficulty, "grashof" if variant == "mixed" else variant)
    L = [float(x) for x in spec["lengths"]]
    sol = classify_fourbar(L)
    can = can_link_fully_rotate(L, 1)  # the input (link index 1)
    q = ("Can the INPUT link (the driven link joined to the frame at the left "
         "pivot) complete a full 360-degree revolution? Answer yes or no.")
    ans = Answer(value="yes" if can else "no", kind="categorical", choices=["yes", "no"])
    rationale = _rationale_class(spec, sol) + (
        " The input link fully rotates iff the linkage is Grashof and the shortest "
        "link is the ground or the input itself; here the answer is "
        f"{'yes' if can else 'no'}.")
    return GenResult(spec, q, ans, rationale, _symbolic_description(spec, sol))


def _counterfactual_class(rng: SeededRNG, difficulty: int) -> tuple[GenResult, GenResult, str]:
    """Minimal edit: nudge one length across the Grashof boundary to flip the class."""
    base = _gen_grashof(rng, max(2, difficulty), "mixed")
    L = list(base.spec["lengths"])
    sol = classify_fourbar([float(x) for x in L])
    # change the longest link to flip Grashof status
    li = max(range(4), key=lambda i: L[i])
    delta = int(abs(sol["margin"]) + 2) if sol["grashof"] else -int(abs(sol["margin"]) + 2)
    L2 = list(L)
    L2[li] = max(3, L2[li] + delta)
    while not _valid_quad(L2):  # step back toward the valid original
        L2[li] += 1 if delta < 0 else -1
    spec2 = {"type": "fourbar", "lengths": L2,
             "input_angle_deg": _good_input_angle([float(x) for x in L2]),
             "variant": "counterfactual"}
    sol2 = classify_fourbar([float(x) for x in L2])
    ans2 = Answer(value=sol2["type"], kind="categorical", choices=_TYPES)
    edited = GenResult(spec2, base.question, ans2, _rationale_class(spec2, sol2),
                       _symbolic_description(spec2, sol2))
    return base, edited, f"changed the longest link from {L[li]} to {L2[li]} (crosses Grashof boundary)"


register("fourbar_grashof", "kinematics", _gen_grashof, counterfactual_fn=_counterfactual_class)
register("fourbar_rotatability", "kinematics", _gen_rotatability)
