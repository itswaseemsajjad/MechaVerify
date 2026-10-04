"""Rendering of symbolic specs into 2-D engineering-style schematics.

Dispatch is keyed on ``spec["type"]``. Each renderer draws onto a Matplotlib
Axes prepared by :func:`mechaverify.render.style.new_canvas` and the instance is
saved to a PNG by :func:`render_to_file`.
"""

from __future__ import annotations

import os
from typing import Callable

from mechaverify.render.style import RenderConfig, finalize, new_canvas

# spec type -> draw(ax, spec, cfg, rng_seed)
_RENDERERS: dict[str, Callable] = {}


def register_renderer(spec_type: str, fn: Callable) -> None:
    _RENDERERS[spec_type] = fn


def render_to_file(spec: dict, path: str, cfg: RenderConfig | None = None, seed: int = 0) -> str:
    """Render ``spec`` to ``path`` (PNG). Returns the path."""
    cfg = cfg or RenderConfig()
    spec_type = spec.get("type")
    if spec_type not in _RENDERERS:
        raise KeyError(f"no renderer for spec type {spec_type!r}; have {sorted(_RENDERERS)}")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    import matplotlib as mpl
    # Hand-drawn aesthetic via path-sketch wiggle (no font dependency); identical
    # geometry and labels, so the closed-form ground truth is unchanged.
    overrides = {"path.sketch": (1.6, 110, 2)} if cfg.sketch else {}
    with mpl.rc_context(overrides):
        fig, ax = new_canvas(cfg)
        _RENDERERS[spec_type](ax, spec, cfg, seed)
        finalize(fig, ax, spec, cfg, path)
    return path


# Importing a renderer module registers it.
from mechaverify.render import gears  # noqa: E402,F401
from mechaverify.render import fourbar  # noqa: E402,F401
from mechaverify.render import mobility  # noqa: E402,F401
from mechaverify.render import dfma  # noqa: E402,F401
