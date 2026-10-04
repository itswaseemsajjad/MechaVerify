"""Shared engineering-drawing style: canvas, title block, dimension lines,
hatching, and controlled rendering perturbations.

These elements are *distractors* (they do not change the answer) and are toggled
per the :class:`RenderConfig` so difficulty can be swept by annotation density
and perturbation magnitude while the underlying mechanism is held fixed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless; no display required
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

LINE = "#1a1a1a"
THIN = 0.8
MED = 1.4
THICK = 2.0


@dataclass
class RenderConfig:
    dpi: int = 110
    show_title_block: bool = True
    show_dimensions: bool = True       # leader/dimension-line distractors
    distractor_density: float = 0.5    # 0..1, scales number of distractor annotations
    perturb_deg: float = 0.0           # small global rotation perturbation (degrees)
    title: str = "MECHAVERIFY"
    fontsize: float = 9.0
    sketch: bool = False               # hand-drawn ("xkcd") line style; labels unchanged


def new_canvas(cfg: RenderConfig):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _draw_title_block(ax, spec: dict, cfg: RenderConfig) -> None:
    """A small ASME-ish title block in the lower-right of the frame."""
    x0, y0, w, h = 0.62, 0.02, 0.36, 0.13
    ax.add_patch(Rectangle((x0, y0), w, h, transform=ax.transAxes,
                           fill=False, lw=MED, ec=LINE))
    ax.plot([x0, x0 + w], [y0 + h * 0.5, y0 + h * 0.5], transform=ax.transAxes,
            color=LINE, lw=THIN)
    ax.plot([x0 + w * 0.72, x0 + w * 0.72], [y0, y0 + h], transform=ax.transAxes,
            color=LINE, lw=THIN)
    kind = spec.get("type", "drawing").replace("_", " ").upper()
    ax.text(x0 + 0.012, y0 + h * 0.72, cfg.title, transform=ax.transAxes,
            fontsize=cfg.fontsize - 2.5, fontweight="bold", va="center")
    ax.text(x0 + 0.012, y0 + h * 0.25, kind, transform=ax.transAxes,
            fontsize=cfg.fontsize - 3, va="center")
    ax.text(x0 + w * 0.76, y0 + h * 0.72, "NTS", transform=ax.transAxes,
            fontsize=cfg.fontsize - 3.5, va="center")
    ax.text(x0 + w * 0.76, y0 + h * 0.25, "mm", transform=ax.transAxes,
            fontsize=cfg.fontsize - 3.5, va="center")


def _draw_frame(ax) -> None:
    ax.add_patch(Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes,
                           fill=False, lw=MED, ec=LINE))


def draw_dimension(ax, x0, y0, x1, y1, text: str, offset: float = 0.0) -> None:
    """A linear dimension with witness lines and arrowheads (data coords)."""
    ax.annotate("", xy=(x1, y1 + offset), xytext=(x0, y0 + offset),
                arrowprops=dict(arrowstyle="<->", color=LINE, lw=THIN))
    if offset:
        ax.plot([x0, x0], [y0, y0 + offset], color=LINE, lw=THIN)
        ax.plot([x1, x1], [y1, y1 + offset], color=LINE, lw=THIN)
    ax.text((x0 + x1) / 2, (y0 + y1) / 2 + offset, text, ha="center",
            va="center", fontsize=7, color=LINE,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))


def add_distractors(ax, spec: dict, cfg: RenderConfig, seed: int,
                    xspan: tuple[float, float], y_base: float) -> None:
    """Add answer-irrelevant dimension lines in a reserved strip below the
    mechanism. Density scales with difficulty; they never overlap the title
    block (kept to the left 60% of the width)."""
    if not cfg.show_dimensions:
        return
    r = random.Random(seed ^ 0x5EED)
    xmin, xmax = xspan
    width = xmax - xmin
    n = int(round(cfg.distractor_density * 3))
    for i in range(n):
        x0 = xmin + r.uniform(0.0, 0.30) * width
        x1 = min(x0 + r.uniform(0.15, 0.35) * width, xmin + 0.58 * width)
        yy = y_base - i * 0.6
        draw_dimension(ax, x0, yy, x1, yy, f"{r.randint(10, 200)}")


def finalize(fig, ax, spec: dict, cfg: RenderConfig, path: str) -> None:
    if cfg.perturb_deg:
        # apply a small rotation perturbation to all artists via axes transform
        from matplotlib.transforms import Affine2D
        t = Affine2D().rotate_deg_around(
            *ax.transData.transform((sum(ax.get_xlim()) / 2,
                                     sum(ax.get_ylim()) / 2)), cfg.perturb_deg)
        for artist in ax.get_children():
            try:
                artist.set_transform(artist.get_transform() + t)
            except Exception:
                pass
    _draw_frame(ax)
    if cfg.show_title_block:
        _draw_title_block(ax, spec, cfg)
    ax.margins(0.08)
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
