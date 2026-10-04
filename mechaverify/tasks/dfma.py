"""Task Family II: design for manufacturing (DFMA).

Two machinability judgments on a milled internal pocket:

1. Tool fit: a flat end mill of radius r = D/2 cannot cut an internal
   corner sharper than itself, so the corner is machinable only if

       R_c >= r_tool = D/2

   and the tool must also fit the pocket width (D <= W).

2. Deflection / chatter: a tool of diameter D reaching depth h has a
   length-to-diameter ratio h/D; flag a risk when

       h / D > tau

The relevant ratio is depth-to-*diameter*, not depth-to-*width*. The adversarial
split exploits this with a wide pocket whose tight corner forces a small tool that
is then slender at depth.
"""

from __future__ import annotations

from mechaverify.rng import SeededRNG
from mechaverify.specs import Answer
from mechaverify.tasks import GenResult, register

TOOLS = [3, 4, 5, 6, 8, 10, 12, 16, 20]  # standard end-mill diameters (mm)
DEFAULT_TAU = 4.0


# --------------------------------------------------------------------------- #
# Solving
# --------------------------------------------------------------------------- #
def solve_fit(W: float, R_c: float, D: float) -> dict:
    r = D / 2.0
    corner_ok = r <= R_c + 1e-9
    width_ok = D <= W + 1e-9
    machinable = corner_ok and width_ok
    binding = None if machinable else ("corner radius" if not corner_ok else "pocket width")
    return {"r_tool": r, "corner_ok": corner_ok, "width_ok": width_ok,
            "machinable": machinable, "binding": binding}


def solve_deflection(h: float, D: float, tau: float) -> dict:
    ratio = h / D
    return {"ratio": ratio, "risk": ratio > tau + 1e-9, "tau": tau}


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _base_pocket(rng: SeededRNG) -> dict:
    return {
        "type": "dfma_pocket",
        "width": float(rng.randint(10, 38)),
        "depth": float(rng.randint(6, 36)),
        "corner_radius": float(rng.choice([1, 1.5, 2, 3, 4, 5, 6, 8])),
        "tau": DEFAULT_TAU,
    }


def sample_fit(rng: SeededRNG, difficulty: int, variant: str) -> dict:
    p = _base_pocket(rng)
    if variant == "adv_idler":
        # corner looks fine but the tool is too wide for the (narrow) pocket
        p["corner_radius"] = float(rng.choice([5, 6, 8]))
        D = rng.choice([12, 16, 20])
        p["width"] = float(D - rng.choice([1, 2, 3]))  # width is the binding constraint
        p["tool_diameter"] = float(D)
    else:
        p["tool_diameter"] = float(rng.choice(TOOLS))
    return p


def sample_deflection(rng: SeededRNG, difficulty: int, variant: str) -> dict:
    p = _base_pocket(rng)
    if variant == "adv_idler":
        # wide pocket (h/W looks safe) but a tight corner forces a small tool
        p["corner_radius"] = float(rng.choice([1, 1.5, 2]))
        D = 2 * p["corner_radius"]                      # largest tool that fits the corner
        p["width"] = float(rng.randint(28, 40))         # wide -> h/W small
        p["depth"] = float(rng.randint(int(DEFAULT_TAU * D) + 1, int(DEFAULT_TAU * D) + 8))
        p["tool_diameter"] = float(D)
    else:
        D = float(rng.choice(TOOLS))
        p["tool_diameter"] = D
        p["depth"] = float(rng.randint(4, int(D * 7) + 1))  # span both sides of tau
    return p


# --------------------------------------------------------------------------- #
# Descriptions
# --------------------------------------------------------------------------- #
def _describe(p: dict) -> str:
    return (f"Milled internal pocket (section view). Width W = {p['width']:g} mm, "
            f"depth h = {p['depth']:g} mm, internal corner radius R_c = "
            f"{p['corner_radius']:g} mm. Offered tool: flat end mill of diameter "
            f"D = {p['tool_diameter']:g} mm (radius {p['tool_diameter']/2:g} mm). "
            f"Deflection threshold tau = {p['tau']:g}.")


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def _gen_fit(rng: SeededRNG, difficulty: int, variant: str = "mixed") -> GenResult:
    p = sample_fit(rng, difficulty, variant)
    s = solve_fit(p["width"], p["corner_radius"], p["tool_diameter"])
    q = (f"Can the internal corner of this pocket be machined with the shown "
         f"Ø{p['tool_diameter']:g} mm flat end mill? Answer yes or no.")
    ans = Answer(value="yes" if s["machinable"] else "no", kind="categorical",
                 choices=["yes", "no"])
    rationale = (
        f"Tool radius r = D/2 = {s['r_tool']:g} mm. Corner rule (R_c >= r): "
        f"R_c = {p['corner_radius']:g} vs r = {s['r_tool']:g} -> "
        f"{'OK' if s['corner_ok'] else 'FAILS'}. Width (D <= W): "
        f"D = {p['tool_diameter']:g} vs W = {p['width']:g} -> "
        f"{'OK' if s['width_ok'] else 'FAILS'}. "
        f"Therefore {'machinable' if s['machinable'] else 'NOT machinable'}"
        + (f" (binding: {s['binding']})." if s['binding'] else "."))
    return GenResult(p, q, ans, rationale, _describe(p))


def _gen_deflection(rng: SeededRNG, difficulty: int, variant: str = "mixed") -> GenResult:
    p = sample_deflection(rng, difficulty, variant)
    s = solve_deflection(p["depth"], p["tool_diameter"], p["tau"])
    q = (f"A Ø{p['tool_diameter']:g} mm flat end mill must reach the full pocket "
         f"depth. Using the rule 'flag a tool-deflection/chatter risk when "
         f"depth/diameter > {p['tau']:g}', is there a deflection risk? Answer yes or no.")
    ans = Answer(value="yes" if s["risk"] else "no", kind="categorical",
                 choices=["yes", "no"])
    rationale = (
        f"Length-to-diameter ratio = h/D = {p['depth']:g}/{p['tool_diameter']:g} = "
        f"{s['ratio']:.2f}; threshold tau = {p['tau']:g}; since {s['ratio']:.2f} "
        f"{'>' if s['risk'] else '<='} {p['tau']:g}, deflection risk = "
        f"{'yes' if s['risk'] else 'no'}. The governing ratio is depth-to-diameter, "
        f"not depth-to-width (W = {p['width']:g}).")
    return GenResult(p, q, ans, rationale, _describe(p))


def _counterfactual_fit(rng: SeededRNG, difficulty: int) -> tuple[GenResult, GenResult, str]:
    """Minimal edit: nudge the corner radius across r_tool to flip machinability."""
    base = _gen_fit(rng, difficulty, "mixed")
    if base.spec["width"] < base.spec["tool_diameter"] + 2:
        # the width would need editing too, so this base cannot give a single edit
        return _counterfactual_fit(rng.derive("cf-retry"), difficulty)
    p = dict(base.spec)
    r = p["tool_diameter"] / 2.0
    s = solve_fit(p["width"], p["corner_radius"], p["tool_diameter"])
    if s["corner_ok"]:
        p["corner_radius"] = float(max(0.5, r - 1.0))   # make corner too sharp
    else:
        p["corner_radius"] = float(r + 1.0)             # open the corner up
    p["width"] = float(max(p["width"], p["tool_diameter"] + 2))  # remove width as binding
    s2 = solve_fit(p["width"], p["corner_radius"], p["tool_diameter"])
    ans2 = Answer(value="yes" if s2["machinable"] else "no", kind="categorical",
                  choices=["yes", "no"])
    rationale2 = (f"With R_c = {p['corner_radius']:g} vs r = {r:g}, the corner is "
                  f"{'machinable' if s2['machinable'] else 'NOT machinable'}.")
    edited = GenResult(p, base.question, ans2, rationale2, _describe(p))
    return base, edited, f"changed corner radius to {p['corner_radius']:g} mm (crosses r_tool)"


register("dfma_fit", "dfma", _gen_fit, counterfactual_fn=_counterfactual_fit)
register("dfma_deflection", "dfma", _gen_deflection)
