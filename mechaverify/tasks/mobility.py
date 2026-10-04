"""Task Family I (kinematics): planar mobility (Kutzbach-Grubler criterion).

    M = 3 (n - 1) - 2 j1 - j2

n = number of links (including the frame), j1 = number of lower (1-DOF) pairs
(revolute, prismatic), j2 = number of higher (2-DOF) pairs (cam/gear contact).

Mechanisms are emitted as labeled *skeletons* (links as bars/plates, joints as
pins) with all pins kept binary so the link/joint counts are unambiguous from
the drawing. The criterion is known to fail for certain overconstrained /
paradoxical mechanisms with special geometry (e.g. a redundant parallel link);
those populate the adversarial split, where the ground-truth answer is the
*true* mobility rather than the Kutzbach count.
"""

from __future__ import annotations

from mechaverify.rng import SeededRNG
from mechaverify.specs import Answer
from mechaverify.tasks import GenResult, register


def _finalize(spec: dict, paradox_true_M: int | None = None) -> dict:
    n = len(spec["links"])
    j1 = sum(1 for j in spec["joints"] if j["kind"] in ("revolute", "prismatic"))
    j2 = len(spec.get("higher_pairs", []))
    kutzbach = 3 * (n - 1) - 2 * j1 - j2
    spec.update({
        "type": "linkage_skeleton",
        "n_links": n, "j1": j1, "j2": j2,
        "kutzbach_M": kutzbach,
        "true_M": kutzbach if paradox_true_M is None else paradox_true_M,
        "paradox": paradox_true_M is not None,
    })
    return spec


def _rev(pos, fixed=False):
    return {"pos": [round(pos[0], 3), round(pos[1], 3)], "kind": "revolute", "fixed": fixed}


def _pri(pos, angle=0.0):
    return {"pos": [round(pos[0], 3), round(pos[1], 3)], "kind": "prismatic",
            "fixed": True, "angle": angle}


# --------------------------------------------------------------------------- #
# Templates  (geometry is schematic; only the topology determines M)
# --------------------------------------------------------------------------- #
def _loop(rng: SeededRNG, k: int) -> dict:
    """A k-bar single-loop linkage: k links, k revolute joints, M = k - 3."""
    import math
    g = 5 + rng.randint(0, 3)
    O2, O4 = (0.0, 0.0), (float(g), 0.0)
    inner = k - 2  # free vertices across the top
    pts = []
    for i in range(inner):
        t = (i + 1) / (inner + 1)
        x = g * t
        y = 3.0 + 1.2 * math.sin(math.pi * t) + rng.uniform(-0.3, 0.3)
        pts.append((x, y))
    chain = [O2] + pts + [O4]
    links = [{"label": "GROUND", "points": [list(O2), list(O4)], "ground": True}]
    for i in range(len(chain) - 1):
        links.append({"label": f"L{i+1}", "points": [list(chain[i]), list(chain[i+1])],
                      "ground": False})
    joints = [_rev(O2, True), _rev(O4, True)] + [_rev(p) for p in pts]
    return _finalize({"name": f"{k}-bar loop", "links": links, "joints": joints,
                      "higher_pairs": []})


def _slider_crank(rng: SeededRNG) -> dict:
    g = 5 + rng.randint(0, 2)
    O2 = (0.0, 0.0)
    A = (1.2 + rng.uniform(-0.2, 0.3), 2.4 + rng.uniform(-0.2, 0.3))
    B = (float(g), 0.8)
    links = [
        {"label": "GROUND", "points": [list(O2), [float(g) + 1.5, 0.8]], "ground": True},
        {"label": "L1 (crank)", "points": [list(O2), list(A)], "ground": False},
        {"label": "L2 (rod)", "points": [list(A), list(B)], "ground": False},
        {"label": "L3 (slider)", "points": [[B[0] - 0.6, B[1]], [B[0] + 0.6, B[1]]],
         "ground": False, "slider": True},
    ]
    joints = [_rev(O2, True), _rev(A), _rev(B), _pri((B[0], B[1]))]
    return _finalize({"name": "slider-crank", "links": links, "joints": joints,
                      "higher_pairs": []})


def _cam_follower(rng: SeededRNG) -> dict:
    O = (0.0, 0.0)
    r = 1.4 + rng.uniform(-0.2, 0.3)
    contact = (0.0, r)
    fol_x = 0.0
    links = [
        {"label": "GROUND", "points": [[-2.0, -1.2], [2.0, -1.2]], "ground": True},
        {"label": "L1 (cam)", "points": [list(O)], "ground": False, "cam_r": r},
        {"label": "L2 (follower)", "points": [[fol_x, r + 0.3], [fol_x, r + 2.6]],
         "ground": False, "slider": True},
    ]
    joints = [_rev(O, True), _pri((fol_x, r + 1.4), angle=90.0)]
    higher = [{"center": list(O), "r": r, "contact": list(contact)}]
    return _finalize({"name": "cam-follower", "links": links, "joints": joints,
                      "higher_pairs": higher})


def _stephenson_six(rng: SeededRNG) -> dict:
    """Six-bar with two ternary links: n=6, j1=7, M=1 (breaks side-counting)."""
    g = 6 + rng.randint(0, 2)
    O2, O4 = (0.0, 0.0), (float(g), 0.0)
    A = (1.2, 3.2 + rng.uniform(-0.2, 0.2))
    B = (4.2 + rng.uniform(-0.3, 0.3), 3.6)
    C = (2.5, 4.5)        # third pin on ternary L2 (A-B-C), lifted off the A-B line
    E = (5.2, 1.7)        # third pin on ternary L3 (B-O4-E)
    D = (3.3, 1.6)
    links = [
        {"label": "GROUND", "points": [list(O2), list(O4)], "ground": True},
        {"label": "L1", "points": [list(O2), list(A)], "ground": False},
        {"label": "L2", "points": [list(A), list(B), list(C)], "ground": False},
        {"label": "L3", "points": [list(B), list(O4), list(E)], "ground": False},
        {"label": "L4", "points": [list(C), list(D)], "ground": False},
        {"label": "L5", "points": [list(D), list(E)], "ground": False},
    ]
    joints = [_rev(O2, True), _rev(O4, True), _rev(A), _rev(B), _rev(C),
              _rev(D), _rev(E)]
    return _finalize({"name": "Stephenson six-bar", "links": links, "joints": joints,
                      "higher_pairs": []})


def _parallelogram_paradox(rng: SeededRNG) -> dict:
    """Overconstrained parallelogram with a redundant parallel link.
    Kutzbach gives M=0 but the true mobility is 1 (special geometry)."""
    g = 4 + rng.randint(0, 2)
    h = 3.0 + rng.uniform(-0.3, 0.3)
    O2, O3, O4 = (0.0, 0.0), (g / 2.0, 0.0), (float(g), 0.0)
    A, C, B = (0.0, h), (g / 2.0, h), (float(g), h)
    links = [
        {"label": "GROUND", "points": [list(O2), list(O4)], "ground": True},
        {"label": "L1", "points": [list(O2), list(A)], "ground": False},
        {"label": "L2", "points": [list(O4), list(B)], "ground": False},
        {"label": "L3 (coupler)", "points": [list(A), list(C), list(B)], "ground": False},
        {"label": "L4 (redundant)", "points": [list(O3), list(C)], "ground": False},
    ]
    joints = [_rev(O2, True), _rev(O3, True), _rev(O4, True),
              _rev(A), _rev(B), _rev(C)]
    return _finalize({"name": "redundant parallelogram", "links": links,
                      "joints": joints, "higher_pairs": []}, paradox_true_M=1)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _pick_template(rng: SeededRNG, difficulty: int):
    if difficulty <= 2:
        pool = [lambda r: _loop(r, 4), _slider_crank, _cam_follower]
    elif difficulty <= 4:
        pool = [lambda r: _loop(r, 5), _slider_crank, _cam_follower, _stephenson_six]
    else:
        pool = [lambda r: _loop(r, 6), lambda r: _loop(r, 7), _stephenson_six]
    return rng.choice(pool)


def _symbolic_description(spec: dict) -> str:
    kinds = {}
    for j in spec["joints"]:
        kinds[j["kind"]] = kinds.get(j["kind"], 0) + 1
    parts = [f"Planar mechanism '{spec['name']}'.",
             f"Links (including the frame): n = {spec['n_links']}.",
             f"Lower (1-DOF) pairs: j1 = {spec['j1']} "
             f"({', '.join(f'{v} {k}' for k, v in kinds.items()) or 'none'}).",
             f"Higher (2-DOF) pairs: j2 = {spec['j2']}."]
    if spec["paradox"]:
        parts.append("Note: special (parallel) geometry makes this overconstrained.")
    return " ".join(parts)


def _rationale(spec: dict) -> str:
    base = (f"Kutzbach-Grubler: M = 3(n-1) - 2 j1 - j2 = "
            f"3({spec['n_links']}-1) - 2({spec['j1']}) - {spec['j2']} = {spec['kutzbach_M']}.")
    if spec["paradox"]:
        return (base + f" However, the redundant link is geometrically parallel and "
                f"imposes no independent constraint, so the true mobility is "
                f"{spec['true_M']}.")
    return base


def _gen_mobility(rng: SeededRNG, difficulty: int, variant: str = "mixed") -> GenResult:
    if variant == "adv_idler":
        spec = _parallelogram_paradox(rng)
    else:
        spec = _pick_template(rng, difficulty)(rng)
    q = ("What is the mobility (number of degrees of freedom) M of this planar "
         "mechanism? The hatched link is the fixed frame. Give a single integer.")
    ans = Answer(value=float(spec["true_M"]), kind="numeric", abs_tolerance=0.5)
    return GenResult(spec, q, ans, _rationale(spec), _symbolic_description(spec))


register("mobility", "kinematics", _gen_mobility)
