"""Renderer for DFMA milled-pocket section views.

Draws a sectioned (hatched) block with an internal pocket: width W, depth h, and
filleted bottom corners of radius R_c, dimensioned in mm. A flat end mill of the
offered diameter D is drawn nestled in the bottom corner (so an oversized tool
visibly interferes with the fillet). The model must read W, h, R_c and D and
identify the binding constraint.
"""

from __future__ import annotations

import math

from mechaverify.render import register_renderer
from mechaverify.render.style import LINE, MED, THIN, RenderConfig, add_distractors, draw_dimension

TOOL_COLOR = "#1f6feb"


def _arc(cx, cy, r, a0_deg, a1_deg, n=24):
    pts = []
    for i in range(n + 1):
        a = math.radians(a0_deg + (a1_deg - a0_deg) * i / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def draw_pocket(ax, spec: dict, cfg: RenderConfig, seed: int) -> None:
    W = spec["width"]
    h = spec["depth"]
    R = min(spec["corner_radius"], W / 2 - 0.5, h - 0.5)  # clamp for drawable geometry
    R = max(R, 0.3)
    D = spec["tool_diameter"]

    margin = max(8.0, 0.5 * W)
    Wb = W + 2 * margin
    base = max(6.0, 0.45 * h)
    Hb = h + base
    x0 = margin
    x1 = margin + W
    yb = Hb - h  # pocket floor

    # material section polygon (outer block minus the filleted pocket)
    poly = [(0, 0), (Wb, 0), (Wb, Hb), (x1, Hb), (x1, yb + R)]
    poly += _arc(x1 - R, yb + R, R, 0, -90)       # right fillet
    poly += [(x0 + R, yb)]
    poly += _arc(x0 + R, yb + R, R, -90, -180)    # left fillet
    poly += [(x0, Hb), (0, Hb)]

    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(poly, closed=True, facecolor="#f2f2f2", edgecolor=LINE,
                         lw=MED, hatch="////", zorder=2))

    # tool (flat end mill cross-section), nestled tangent to the left wall + floor
    from matplotlib.patches import Circle
    r = D / 2.0
    tcx, tcy = x0 + r, yb + r
    ax.add_patch(Circle((tcx, tcy), r, fill=True, facecolor="#dCE9fb",
                        edgecolor=TOOL_COLOR, lw=MED, zorder=5, alpha=0.85))
    ax.plot(tcx, tcy, "+", color=TOOL_COLOR, zorder=6)
    ax.text(tcx + r + 0.5, tcy, f"Ø{D:g}", color=TOOL_COLOR, fontsize=8,
            va="center", ha="left")
    ax.text((x0 + x1) / 2, yb + h * 0.55, "POCKET", ha="center", color="#888",
            fontsize=7)

    # dimensions: width (top), depth (right), corner radius (leader)
    draw_dimension(ax, x0, Hb + 1.2, x1, Hb + 1.2, f"W={W:g}")
    draw_dimension(ax, x1 + margin * 0.45, yb, x1 + margin * 0.45, Hb, f"h={h:g}")
    ax.plot([x0, x0], [Hb, Hb + 1.4], color=LINE, lw=THIN)
    ax.plot([x1, x1], [Hb, Hb + 1.4], color=LINE, lw=THIN)
    # corner radius leader
    fcx, fcy = x0 + R, yb + R
    lx, ly = fcx + R * math.cos(math.radians(225)), fcy + R * math.sin(math.radians(225))
    ax.annotate(f"R{spec['corner_radius']:g}", xy=(lx, ly),
                xytext=(x0 - margin * 0.5, yb - 1.5), fontsize=7.5, color=LINE,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=THIN))

    xmin, xmax = -margin * 0.6, Wb + margin * 0.7
    ymax = Hb + 3.0
    span = xmax - xmin
    ymin = -max(2.6, 0.32 * span)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    add_distractors(ax, spec, cfg, seed, (xmin + 0.5, xmin + 0.5 * span), ymin + 1.0)


register_renderer("dfma_pocket", draw_pocket)
