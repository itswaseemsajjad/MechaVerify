"""Renderer for planar mechanism skeletons (mobility task).

Links are drawn as bars (binary) or filled plates (ternary+); the frame is a
hatched ground bar; revolute joints are open circles (fixed ones carry a ground
pivot symbol); prismatic joints are blocks on a guide; higher pairs are drawn as
a cam profile with a marked contact. Links are labelled L1..Ln so the model can
count n, j1 and j2 and apply the Kutzbach-Grubler criterion.
"""

from __future__ import annotations

import math

from mechaverify.render import register_renderer
from mechaverify.render.style import LINE, MED, THIN, RenderConfig, add_distractors
from mechaverify.render.fourbar import _fixed_pivot, _joint


def _centroid(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _ground_bar(ax, p, q):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=LINE, lw=MED, zorder=2)
    n = 10
    for i in range(n + 1):
        t = i / n
        hx = p[0] + t * (q[0] - p[0])
        hy = p[1] + t * (q[1] - p[1])
        ax.plot([hx, hx - 0.22], [hy, hy - 0.22], color=LINE, lw=THIN, zorder=1)


def _plate(ax, pts):
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(pts, closed=True, facecolor="#e8eef5", edgecolor=LINE,
                         lw=MED, zorder=3))


def _bar(ax, p, q):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=LINE, lw=2.4,
            solid_capstyle="round", zorder=3)


def draw_skeleton(ax, spec: dict, cfg: RenderConfig, seed: int) -> None:
    allpts = []
    for link in spec["links"]:
        pts = [tuple(p) for p in link["points"]]
        allpts += pts
        if link.get("ground"):
            _ground_bar(ax, pts[0], pts[1])
            ax.text(_centroid(pts)[0], pts[0][1] - 0.9, "GROUND (frame)",
                    ha="center", fontsize=7, color="#555")
        elif link.get("cam_r"):
            from matplotlib.patches import Circle
            c, r = pts[0], link["cam_r"]
            ax.add_patch(Circle(c, r, fill=True, facecolor="#e8eef5",
                                edgecolor=LINE, lw=MED, zorder=3))
            ax.text(c[0] - r - 0.2, c[1], link["label"], ha="right", fontsize=7.5,
                    color=LINE)
            allpts.append((c[0] + r, c[1] + r))
            allpts.append((c[0] - r, c[1] - r))
        elif link.get("slider"):
            _bar(ax, pts[0], pts[-1])
            cx, cy = _centroid(pts)
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((cx - 0.5, cy - 0.32), 1.0, 0.64, angle=0,
                                   fill=True, facecolor="#dfe6ee", edgecolor=LINE,
                                   lw=MED, zorder=4))
            ax.text(cx + 0.7, cy + 0.5, link["label"], ha="left", fontsize=7.5,
                    color=LINE)
        elif len(pts) >= 3:
            _plate(ax, pts)
            cx, cy = _centroid(pts)
            ax.text(cx, cy, link["label"], ha="center", va="center", fontsize=7.5,
                    color=LINE, fontweight="bold")
        else:
            _bar(ax, pts[0], pts[1])
            mx, my = _centroid(pts)
            ax.text(mx + 0.15, my + 0.2, link["label"], ha="left", fontsize=7.5,
                    color=LINE)

    # prismatic guides
    for j in spec["joints"]:
        if j["kind"] == "prismatic":
            x, y = j["pos"]
            ang = math.radians(j.get("angle", 0.0))
            dx, dy = math.cos(ang), math.sin(ang)
            for s in (-0.4, 0.4):
                ax.plot([x - 1.6 * dx - s * dy, x + 1.6 * dx - s * dy],
                        [y - 1.6 * dy + s * dx, y + 1.6 * dy + s * dx],
                        color=LINE, lw=THIN, ls=(0, (6, 3)), zorder=1)

    # higher pairs: contact marker
    for hp in spec.get("higher_pairs", []):
        cx, cy = hp["contact"]
        ax.plot(cx, cy, marker="^", color="#b00000", markersize=9, zorder=6)
        ax.text(cx + 0.2, cy + 0.1, "higher pair", fontsize=6.5, color="#b00000")
        allpts.append((cx, cy + 0.5))

    # joints
    for j in spec["joints"]:
        if j["kind"] == "revolute":
            _joint(ax, j["pos"][0], j["pos"][1])
            if j.get("fixed"):
                _fixed_pivot(ax, j["pos"][0], j["pos"][1])

    xs = [p[0] for p in allpts]
    ys = [p[1] for p in allpts]
    xmin, xmax = min(xs) - 1.2, max(xs) + 1.2
    ymax = max(ys) + 1.4
    span = xmax - xmin
    ymin = min(ys) - max(2.6, 0.4 * span)
    w, h = xmax - xmin, ymax - ymin
    if h > 1.5 * w:
        pad = (h / 1.5 - w) / 2
        xmin, xmax = xmin - pad, xmax + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    add_distractors(ax, spec, cfg, seed, (xmin + 0.2, xmin + 0.5 * (xmax - xmin)),
                    ymin + 1.2)


register_renderer("linkage_skeleton", draw_skeleton)
