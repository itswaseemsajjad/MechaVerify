"""Renderer for spur gear trains.

Gears are drawn as tangent gears (a finely toothed addendum circle, a pitch
circle in thin chain-line style, a hub, and a shaft centre-mark). All gears
share one module (a meshing requirement), so tooth height is constant while
pitch radius scales with tooth count. Tooth counts are annotated as callouts
(as on real gear drawings, where the count lives in a gear table rather than
being counted off the geometry); the model must still read the *structure*:
chain order, compound vs. idler shafts, the number of meshes, and the input
arrow.
"""

from __future__ import annotations

import math

from mechaverify.render import register_renderer
from mechaverify.render.style import LINE, MED, THIN, RenderConfig, add_distractors

_SCALE = 0.06          # pitch radius (data units) per tooth
_MODULE = 2 * _SCALE   # constant module -> constant tooth height across the train
_TOOTH = _MODULE       # addendum height


def __circle(cx, cy, r, ls="-", lw=1.0, fill=False, fc="none", ec=LINE):
    from matplotlib.patches import Circle
    return Circle((cx, cy), r, fill=fill, facecolor=fc, edgecolor=ec, lw=lw, ls=ls)


def _draw_gear(ax, cx, cy, r, teeth):
    addendum = r + _TOOTH
    ax.add_patch(__circle(cx, cy, r, ls=(0, (7, 3)), lw=THIN))      # pitch circle
    ax.add_patch(__circle(cx, cy, addendum, ls="-", lw=MED))         # addendum circle
    n = int(teeth)
    for i in range(n):                                               # tooth ticks
        a = 2 * math.pi * i / n
        ax.plot([cx + r * math.cos(a), cx + addendum * math.cos(a)],
                [cy + r * math.sin(a), cy + addendum * math.sin(a)],
                color=LINE, lw=THIN)
    ax.add_patch(__circle(cx, cy, max(0.07, r * 0.16), lw=THIN, fill=True, fc="white"))
    cs = max(0.09, r * 0.22)                                          # shaft centre-mark
    ax.plot([cx - cs, cx + cs], [cy, cy], color=LINE, lw=THIN)
    ax.plot([cx, cx], [cy - cs, cy + cs], color=LINE, lw=THIN)


def _rotation_arrow(ax, cx, cy, r, direction):
    """Curved arrow on the input gear indicating its given rotation sense."""
    rr = max(0.3, r * 0.6)
    angs = [math.radians(a) for a in range(60, 300, 8)]
    pts = [(cx + rr * math.cos(a), cy + rr * math.sin(a)) for a in angs]
    if direction == "clockwise":
        pts = pts[::-1]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="#b00000", lw=MED)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color="#b00000", lw=MED))


def draw_gear_train(ax, spec: dict, cfg: RenderConfig, seed: int) -> None:
    sg = spec["shaft_gears"]
    radii = [[t * _SCALE for t in gears] for gears in sg]

    # tangency layout along a horizontal line
    xs = [0.0]
    for k in range(len(sg) - 1):
        xs.append(xs[-1] + radii[k][-1] + radii[k + 1][0])
    cy = 0.0
    max_r = max(max(r) for r in radii)

    # label offsets sit just outside each gear's rim (independent of radius)
    Z_OFF, ROLE_OFF, SUB_OFF = 0.40, 1.05, 0.50

    for j, gears in enumerate(sg):
        cx = xs[j]
        rmax = max(radii[j])
        for gi, t in enumerate(gears):
            _draw_gear(ax, cx, cy, radii[j][gi], t)
        if len(gears) == 2:
            ax.text(cx, cy + rmax + Z_OFF, f"z={gears[0]} | {gears[1]}",
                    ha="center", fontsize=8, color=LINE)
            ax.text(cx, cy - rmax - SUB_OFF, "COMPOUND", ha="center",
                    fontsize=6.5, color="#555")
        else:
            ax.text(cx, cy + rmax + Z_OFF, f"z={gears[0]}", ha="center",
                    fontsize=8, color=LINE)
            if 0 < j < len(sg) - 1:
                ax.text(cx, cy - rmax - SUB_OFF, "idler", ha="center",
                        fontsize=6.5, color="#555")
        if j == 0:
            ax.text(cx, cy + rmax + ROLE_OFF, "INPUT", ha="center", fontsize=8,
                    fontweight="bold", color="#b00000")
            _rotation_arrow(ax, cx, cy, radii[j][-1], spec["input_dir"])
        elif j == len(sg) - 1:
            ax.text(cx, cy + rmax + ROLE_OFF, "OUTPUT", ha="center", fontsize=8,
                    fontweight="bold", color="#00557f")

    # axis limits: symmetric top room for role labels, larger bottom strip so the
    # title block (bottom ~15% of the axes) clears the largest gear.
    xmin = xs[0] - radii[0][0] - 0.3
    xmax = xs[-1] + radii[-1][-1] + 0.3
    ymax = max_r + 1.5
    bottom = max(2.4, 0.5 * max_r + 1.3)
    ymin = -max_r - bottom
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    add_distractors(ax, spec, cfg, seed, (xmin + 0.2, xmax - 0.2), -max_r - 1.0)


register_renderer("gear_train", draw_gear_train)
