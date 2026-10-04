"""Renderer for planar four-bar linkages.

The mechanism is drawn in an assembled pose: the fixed frame as a hatched
ground link between two fixed pivots, the input/coupler/output as bars between
revolute joints, with each link's length dimensioned. The model must read the
four lengths and identify the frame (hatched) to classify the linkage.
"""

from __future__ import annotations

import math

from mechaverify.render import register_renderer
from mechaverify.render.style import LINE, MED, THIN, RenderConfig, add_distractors
from mechaverify.tasks.fourbar import assemble


def _joint(ax, x, y, r=0.12):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, fill=True, facecolor="white", edgecolor=LINE, lw=MED, zorder=5))


def _fixed_pivot(ax, x, y, size=0.45):
    """Ground pivot: triangle with hatching beneath the joint."""
    tri_x = [x, x - size * 0.6, x + size * 0.6, x]
    tri_y = [y, y - size, y - size, y]
    ax.plot(tri_x, tri_y, color=LINE, lw=THIN)
    base_y = y - size
    n = 6
    for i in range(n + 1):
        hx = x - size * 0.6 + i * (1.2 * size) / n
        ax.plot([hx, hx - 0.18], [base_y, base_y - 0.18], color=LINE, lw=THIN)
    ax.plot([x - size * 0.8, x + size * 0.8], [base_y, base_y], color=LINE, lw=THIN)


def _bar(ax, p, q, color=LINE, lw=MED):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, solid_capstyle="round", zorder=3)


def _dim_label(ax, p, q, text, off=0.28):
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / n, dx / n  # unit normal
    ax.text(mx + nx * off, my + ny * off, text, ha="center", va="center",
            fontsize=8, color=LINE,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))


def draw_fourbar(ax, spec: dict, cfg: RenderConfig, seed: int) -> None:
    L = [float(x) for x in spec["lengths"]]
    pose = assemble(L, spec.get("input_angle_deg", 60.0)) or assemble(L, 60.0)
    if pose is None:  # degenerate fallback: draw a generic crank-rocker pose
        pose = assemble([8, 3, 7, 6], 60.0)
        L = [8, 3, 7, 6]
    O2, A, B, O4 = pose["O2"], pose["A"], pose["B"], pose["O4"]

    # ground (frame) link: hatched bar between the fixed pivots
    _bar(ax, O2, O4, lw=MED)
    _fixed_pivot(ax, *O2)
    _fixed_pivot(ax, *O4)
    _dim_label(ax, O2, O4, f"{L[0]:g}", off=-0.65)
    ax.text((O2[0] + O4[0]) / 2, -1.5, "GROUND (frame)", ha="center", fontsize=7,
            color="#555")

    # moving links
    _bar(ax, O2, A)                      # input
    _bar(ax, A, B)                       # coupler
    _bar(ax, O4, B)                      # output
    _dim_label(ax, O2, A, f"{L[1]:g}")
    _dim_label(ax, A, B, f"{L[2]:g}")
    _dim_label(ax, O4, B, f"{L[3]:g}")

    # joints
    for p in (A, B):
        _joint(ax, *p)

    # role labels + input rotation arrow
    ax.text(A[0], A[1] + 0.3, "INPUT", ha="center", fontsize=8, fontweight="bold",
            color="#b00000")
    ax.text(B[0], B[1] + 0.3, "OUTPUT", ha="center", fontsize=8, fontweight="bold",
            color="#00557f")
    mid = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
    ax.text(mid[0], mid[1] + 0.35, "COUPLER", ha="center", fontsize=7, color="#555")
    _input_arrow(ax, O2, A)

    xs = [O2[0], O4[0], A[0], B[0]]
    ys = [O2[1], O4[1], A[1], B[1]]
    xmin, xmax = min(xs) - 1.2, max(xs) + 1.2
    ymax = max(ys) + 1.2
    span = xmax - xmin
    ymin = min(ys) - max(2.6, 0.35 * span)
    # enforce a minimum width:height so the title block keeps a sane aspect
    w, h = xmax - xmin, ymax - ymin
    if h > 1.5 * w:
        pad = (h / 1.5 - w) / 2
        xmin, xmax = xmin - pad, xmax + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    add_distractors(ax, spec, cfg, seed, (xmin + 0.2, xmin + 0.5 * (xmax - xmin)), ymin + 1.2)


def _input_arrow(ax, O2, A):
    """Small curved arrow at the input pivot indicating it is the driver."""
    r = 0.6
    ang0 = math.degrees(math.atan2(A[1] - O2[1], A[0] - O2[0]))
    angs = [math.radians(ang0 - 40 + i * 8) for i in range(11)]
    pts = [(O2[0] + r * math.cos(a), O2[1] + r * math.sin(a)) for a in angs]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="#b00000", lw=MED)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color="#b00000", lw=MED))


register_renderer("fourbar", draw_fourbar)
