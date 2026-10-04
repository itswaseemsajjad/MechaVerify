"""Task Family I (kinematics): spur gear trains.

Representation
--------------
A gear train is a chain of shafts. ``spec["shaft_gears"]`` lists, per shaft, the
tooth counts of the gear(s) mounted on it, ordered input -> output. By
convention ``shaft_gears[j][0]`` is the gear that meshes the previous shaft and
``shaft_gears[j][-1]`` the gear that meshes the next shaft; a single-gear shaft
is an *idler* (both roles are the same physical gear) and a two-gear shaft is a
*compound* stage (two gears keyed to one shaft, sharing angular velocity).

Closed-form ground truth
------------------------
For mesh ``k`` between shaft ``k`` (driver) and shaft ``k+1`` (driven),

    omega_{k+1} / omega_k = -(N_driver / N_driven)        (external mesh)

so the overall magnitude is the product of mesh tooth ratios

    |omega_out / omega_in| = prod_k ( N_driver_k / N_driven_k )

and the output rotation sense relative to the input is (-1)^m, where m is the
number of external meshes. An idler reverses sense without changing magnitude;
in a *simple* train the intermediate tooth counts telescope away and the
magnitude reduces to N_input / N_output.
"""

from __future__ import annotations

from fractions import Fraction

from mechaverify.rng import SeededRNG
from mechaverify.specs import Answer
from mechaverify.tasks import GenResult, register

_TEETH_MIN, _TEETH_MAX = 8, 96
_DIR_FLIP = {"clockwise": "counterclockwise", "counterclockwise": "clockwise"}
_DIR_CANON = {"cw": "clockwise", "ccw": "counterclockwise",
              "clockwise": "clockwise", "counterclockwise": "counterclockwise"}


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _teeth(rng: SeededRNG) -> int:
    return rng.randint(_TEETH_MIN, _TEETH_MAX)


def sample_gear_train(rng: SeededRNG, difficulty: int, variant: str = "mixed") -> dict:
    """Sample a gear-train spec. ``difficulty`` ~ number of meshes (clamped 1..6)."""
    n_mesh = max(1, min(int(difficulty), 6))
    n_shaft = n_mesh + 1
    n_inter = n_shaft - 2  # intermediate shafts that may be compound

    if variant in ("simple", "adv_idler"):
        compound = [False] * n_inter
    elif variant == "compound":
        compound = [rng.chance(0.6) for _ in range(n_inter)]
        if n_inter > 0 and not any(compound):
            compound[rng.randint(0, n_inter - 1)] = True
    else:  # mixed
        compound = [rng.chance(0.4) for _ in range(n_inter)]

    shaft_gears: list[list[int]] = [[_teeth(rng)]]  # shaft 0 (input)
    for i in range(1, n_shaft):
        in_g = _teeth(rng)
        if i == n_shaft - 1:  # output shaft: in-gear only
            shaft_gears.append([in_g])
        elif compound[i - 1]:  # compound stage: distinct out-gear
            out_g = _teeth(rng)
            shaft_gears.append([in_g, out_g])
        else:  # idler: single gear serves both roles
            shaft_gears.append([in_g])

    # Adversarial idler: make a mid-train idler conspicuously large to bait an
    # incorrect ratio (a salient gear that does not affect the magnitude).
    if variant == "adv_idler" and n_inter >= 1:
        idx = 1 + rng.randint(0, n_inter - 1)
        shaft_gears[idx] = [_TEETH_MAX]

    return {
        "type": "gear_train",
        "shaft_gears": shaft_gears,
        "input_dir": _DIR_CANON[rng.choice(["cw", "ccw"])],
        "all_external": True,
    }


# --------------------------------------------------------------------------- #
# Solving
# --------------------------------------------------------------------------- #
def _meshes(spec: dict) -> list[tuple[int, int]]:
    sg = spec["shaft_gears"]
    return [(sg[k][-1], sg[k + 1][0]) for k in range(len(sg) - 1)]


def solve_gear_train(spec: dict) -> dict:
    """Return closed-form ground truth: ratio magnitude and output sense."""
    meshes = _meshes(spec)
    ratio = Fraction(1)
    for driver, driven in meshes:
        ratio *= Fraction(driver, driven)
    m = len(meshes)
    in_dir = _DIR_CANON[spec["input_dir"]]
    out_dir = in_dir if m % 2 == 0 else _DIR_FLIP[in_dir]
    return {
        "meshes": meshes,
        "n_external": m,
        "ratio_fraction": (ratio.numerator, ratio.denominator),
        "ratio_magnitude": float(ratio),
        "input_dir": in_dir,
        "output_dir": out_dir,
    }


# --------------------------------------------------------------------------- #
# Descriptions
# --------------------------------------------------------------------------- #
def _symbolic_description(spec: dict, sol: dict) -> str:
    sg = spec["shaft_gears"]
    lines = ["Spur gear train (all meshes external)."]
    for j, gears in enumerate(sg):
        role = "INPUT" if j == 0 else ("OUTPUT" if j == len(sg) - 1 else "")
        if len(gears) == 2:
            kind = "compound stage"
            desc = f"in-gear {gears[0]}T keyed to out-gear {gears[1]}T"
        else:
            kind = "idler" if 0 < j < len(sg) - 1 else "gear"
            desc = f"{gears[0]}T"
        tag = f" [{role}]" if role else ""
        lines.append(f"  Shaft {j + 1}{tag}: {desc} ({kind}).")
    chain = ", ".join(f"{d}->{n}" for d, n in sol["meshes"])
    lines.append(f"Meshes (driver->driven teeth): {chain}.")
    lines.append(f"Input shaft rotates {sol['input_dir']}.")
    return "\n".join(lines)


def _rationale(spec: dict, sol: dict, task: str) -> str:
    terms = " x ".join(f"({d}/{n})" for d, n in sol["meshes"])
    num, den = sol["ratio_fraction"]
    parts = [
        f"There are {sol['n_external']} external meshes.",
        f"Speed-ratio magnitude = product of (N_driver/N_driven) = {terms} "
        f"= {num}/{den} = {sol['ratio_magnitude']:.4g}.",
    ]
    if task == "gear_sense":
        parity = "even" if sol["n_external"] % 2 == 0 else "odd"
        same = "the same direction as" if sol["n_external"] % 2 == 0 else "the opposite direction to"
        parts.append(
            f"The number of external meshes is {parity}, so the output turns "
            f"{same} the input. Input is {sol['input_dir']}, hence the output is "
            f"{sol['output_dir']}."
        )
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Generators (registered tasks)
# --------------------------------------------------------------------------- #
def _gen_sense(rng: SeededRNG, difficulty: int, variant: str = "mixed") -> GenResult:
    spec = sample_gear_train(rng, difficulty, variant)
    sol = solve_gear_train(spec)
    q = (
        f"The input gear (leftmost, shaft 1) rotates {sol['input_dir']}. "
        "Which way does the output gear (rightmost shaft) rotate? "
        "Answer 'clockwise' or 'counterclockwise'."
    )
    ans = Answer(value=sol["output_dir"], kind="categorical",
                 choices=["clockwise", "counterclockwise"])
    return GenResult(spec, q, ans, _rationale(spec, sol, "gear_sense"),
                     _symbolic_description(spec, sol))


def _gen_ratio(rng: SeededRNG, difficulty: int, variant: str = "mixed") -> GenResult:
    spec = sample_gear_train(rng, difficulty, variant)
    sol = solve_gear_train(spec)
    q = (
        "What is the speed-ratio magnitude |omega_output / omega_input| of this "
        "gear train? Give a single number (output speed divided by input speed)."
    )
    ans = Answer(value=sol["ratio_magnitude"], kind="numeric", tolerance=0.02)
    return GenResult(spec, q, ans, _rationale(spec, sol, "gear_ratio"),
                     _symbolic_description(spec, sol))


def _counterfactual_sense(rng: SeededRNG, difficulty: int) -> tuple[GenResult, GenResult, str]:
    """Minimal visual edit that flips the answer: insert one idler, reversing the
    output sense while leaving the textual framing unchanged."""
    base = _gen_sense(rng, max(2, difficulty), variant="mixed")
    spec2 = {**base.spec, "shaft_gears": [list(g) for g in base.spec["shaft_gears"]]}
    # Insert an idler just before the output shaft (adds one external mesh).
    idler = [_teeth(rng.derive("cf-idler"))]
    spec2["shaft_gears"].insert(len(spec2["shaft_gears"]) - 1, idler)
    sol2 = solve_gear_train(spec2)
    q2 = (
        f"The input gear (leftmost, shaft 1) rotates {sol2['input_dir']}. "
        "Which way does the output gear (rightmost shaft) rotate? "
        "Answer 'clockwise' or 'counterclockwise'."
    )
    ans2 = Answer(value=sol2["output_dir"], kind="categorical",
                  choices=["clockwise", "counterclockwise"])
    edited = GenResult(spec2, q2, ans2, _rationale(spec2, sol2, "gear_sense"),
                       _symbolic_description(spec2, sol2))
    return base, edited, "inserted one idler gear before the output (adds a mesh, flips sense)"


register("gear_sense", "kinematics", _gen_sense, counterfactual_fn=_counterfactual_sense)
register("gear_ratio", "kinematics", _gen_ratio)
