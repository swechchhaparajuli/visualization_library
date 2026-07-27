"""Flat vector "diorama" motifs, drawn with matplotlib patches.

These replace glossy emoji with simple, cohesive silhouettes in the style of
a flat isometric infographic: layered conifers, round trees, flames, logs,
stumps, crop rows, seedlings, a barn. Each motif draws centered on a ground
point ``(x, y)`` and rises to about ``s`` data units tall.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon

# Cohesive, muted palette (flat fills).
_C = {
    "tree1": "#24463f", "tree2": "#2f6a52", "trunk": "#6b4a2f",
    "flame_o": "#e8542c", "flame_i": "#f4a83a",
    "log": "#8a5a34", "log_end": "#c1935b",
    "stalk": "#6f8f37", "head": "#d8a63a",
    "leaf": "#3f9e5e", "stem": "#4f7a37",
    "barn": "#a5493a", "roof": "#4a3324",
    "char": "#1b1813",  # charred, near-black
}
_SHADOW = "#1c2b22"


def _poly(ax, x, y, pts, s, color, t, z, alpha=1.0):
    arr = np.asarray(pts, float) * s + [x, y]
    ax.add_patch(Polygon(arr, closed=True, facecolor=color, edgecolor="none",
                         alpha=alpha, transform=t, zorder=z))


def _seg(ax, x, y, p0, p1, w, s, color, t, z):
    """Draw a tapered limb (thin quad) from p0 to p1 in unit coords."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.hypot(*d)) or 1.0
    n = np.array([-d[1], d[0]]) / length * w
    _poly(ax, x, y, [p0 + n, p1 + n * 0.35, p1 - n * 0.35, p0 - n], s, color, t, z)


def _shadow(ax, x, y, w, s, t, z):
    ax.add_patch(Ellipse((x, y), w * s, w * s * 0.32, facecolor=_SHADOW,
                         alpha=0.13, edgecolor="none", transform=t, zorder=z))


def evergreen(ax, x, y, s, t, z):
    _shadow(ax, x, y, 0.55, s, t, z)
    _poly(ax, x, y, [(-0.05, 0), (0.05, 0), (0.05, 0.2), (-0.05, 0.2)], s, _C["trunk"], t, z + 0.01)
    _poly(ax, x, y, [(-0.30, 0.15), (0.30, 0.15), (0, 0.55)], s, _C["tree1"], t, z + 0.02)
    _poly(ax, x, y, [(-0.26, 0.40), (0.26, 0.40), (0, 0.78)], s, _C["tree1"], t, z + 0.03)
    _poly(ax, x, y, [(-0.20, 0.62), (0.20, 0.62), (0, 1.0)], s, _C["tree1"], t, z + 0.04)


def round_tree(ax, x, y, s, t, z):
    _shadow(ax, x, y, 0.55, s, t, z)
    _poly(ax, x, y, [(-0.05, 0), (0.05, 0), (0.05, 0.35), (-0.05, 0.35)], s, _C["trunk"], t, z + 0.01)
    for ox, oy, r in ((-0.16, 0.50, 0.20), (0.16, 0.50, 0.20), (0.0, 0.64, 0.30)):
        ax.add_patch(Circle((x + ox * s, y + oy * s), r * s, facecolor=_C["tree2"],
                            edgecolor="none", transform=t, zorder=z + 0.02))


def burnt_tree(ax, x, y, s, t, z):
    """A charred, bare tree: black tapered trunk with leafless branches."""
    _shadow(ax, x, y, 0.4, s, t, z)
    c = _C["char"]
    _poly(ax, x, y, [(-0.05, 0), (0.05, 0), (0.022, 0.9), (-0.022, 0.9)], s, c, t, z + 0.01)
    limbs = [
        ((0.0, 0.42), (0.28, 0.66)), ((0.0, 0.52), (-0.26, 0.78)),
        ((0.0, 0.6), (0.2, 0.9)), ((0.0, 0.34), (-0.2, 0.5)),
        ((0.0, 0.72), (-0.05, 1.02)), ((0.0, 0.7), (0.12, 0.98)),
    ]
    for p0, p1 in limbs:
        _seg(ax, x, y, p0, p1, 0.03, s, c, t, z + 0.02)
        # a small forked twig near the tip
        tip = np.array(p1)
        _seg(ax, x, y, tuple(tip), (tip[0] + (0.08 if tip[0] >= 0 else -0.08), tip[1] + 0.1),
             0.018, s, c, t, z + 0.02)


def burnt_snag(ax, x, y, s, t, z):
    """A short, broken charred stump."""
    _shadow(ax, x, y, 0.34, s, t, z)
    c = _C["char"]
    _poly(ax, x, y, [(-0.06, 0), (0.06, 0), (0.04, 0.34), (0.0, 0.46), (-0.05, 0.3)], s, c, t, z + 0.01)
    _seg(ax, x, y, (0.0, 0.28), (0.2, 0.44), 0.025, s, c, t, z + 0.02)


def flame(ax, x, y, s, t, z):
    _shadow(ax, x, y, 0.4, s, t, z)
    outer = [(0, 0), (0.22, 0.08), (0.26, 0.30), (0.10, 0.46), (0.16, 0.64),
             (0, 1.0), (-0.16, 0.64), (-0.10, 0.46), (-0.26, 0.30), (-0.22, 0.08)]
    _poly(ax, x, y, outer, s, _C["flame_o"], t, z + 0.01)
    inner = [(px * 0.55, py * 0.62 + 0.04) for px, py in outer]
    _poly(ax, x, y, inner, s, _C["flame_i"], t, z + 0.02)


def log(ax, x, y, s, t, z):
    _shadow(ax, x, y, 0.75, s, t, z)
    _poly(ax, x, y, [(-0.35, 0.02), (0.35, 0.02), (0.35, 0.22), (-0.35, 0.22)], s, _C["log"], t, z + 0.01)
    ax.add_patch(Ellipse((x + 0.35 * s, y + 0.12 * s), 0.13 * s, 0.20 * s, facecolor=_C["log"],
                         edgecolor="none", transform=t, zorder=z + 0.02))
    ax.add_patch(Ellipse((x - 0.35 * s, y + 0.12 * s), 0.13 * s, 0.20 * s, facecolor=_C["log_end"],
                         edgecolor="none", transform=t, zorder=z + 0.02))


def stump(ax, x, y, s, t, z):
    _shadow(ax, x, y, 0.4, s, t, z)
    _poly(ax, x, y, [(-0.16, 0), (0.16, 0), (0.16, 0.28), (-0.16, 0.28)], s, _C["log"], t, z + 0.01)
    ax.add_patch(Ellipse((x, y + 0.28 * s), 0.32 * s, 0.12 * s, facecolor=_C["log_end"],
                         edgecolor="none", transform=t, zorder=z + 0.02))


def crop(ax, x, y, s, t, z):
    _shadow(ax, x, y, 0.42, s, t, z)
    for off, h in ((-0.13, 0.55), (0.0, 0.72), (0.13, 0.60)):
        xs = x + off * s
        _poly(ax, xs, y, [(-0.022, 0), (0.022, 0), (0.022, h), (-0.022, h)], s, _C["stalk"], t, z + 0.01)
        ax.add_patch(Ellipse((xs, y + h * s), 0.10 * s, 0.17 * s, facecolor=_C["head"],
                             edgecolor="none", transform=t, zorder=z + 0.02))


def seedling(ax, x, y, s, t, z):
    _shadow(ax, x, y, 0.3, s, t, z)
    _poly(ax, x, y, [(-0.022, 0), (0.022, 0), (0.022, 0.4), (-0.022, 0.4)], s, _C["stem"], t, z + 0.01)
    ax.add_patch(Ellipse((x - 0.10 * s, y + 0.36 * s), 0.20 * s, 0.10 * s, angle=32,
                         facecolor=_C["leaf"], edgecolor="none", transform=t, zorder=z + 0.02))
    ax.add_patch(Ellipse((x + 0.10 * s, y + 0.36 * s), 0.20 * s, 0.10 * s, angle=-32,
                         facecolor=_C["leaf"], edgecolor="none", transform=t, zorder=z + 0.02))


def barn(ax, x, y, s, t, z):
    _shadow(ax, x, y, 0.7, s, t, z)
    _poly(ax, x, y, [(-0.30, 0), (0.30, 0), (0.30, 0.42), (-0.30, 0.42)], s, _C["barn"], t, z + 0.01)
    _poly(ax, x, y, [(-0.34, 0.42), (0.34, 0.42), (0.20, 0.62), (-0.20, 0.62)], s, _C["roof"], t, z + 0.02)
    _poly(ax, x, y, [(-0.08, 0), (0.08, 0), (0.08, 0.24), (-0.08, 0.24)], s, _C["roof"], t, z + 0.03)


MOTIF = {
    "evergreen": evergreen, "round_tree": round_tree, "flame": flame,
    "burnt_tree": burnt_tree, "burnt_snag": burnt_snag,
    "log": log, "stump": stump, "crop": crop, "seedling": seedling, "barn": barn,
}

# Which motifs (and relative sizes) compose each driver's diorama.
SCENES = {
    "Permanent agriculture": [("crop", 1.0), ("crop", 1.05), ("crop", 0.95), ("barn", 1.3), ("seedling", 0.7)],
    "Shifting cultivation": [("seedling", 0.95), ("seedling", 0.85), ("flame", 0.8), ("crop", 0.9)],
    "Wildfire": [("burnt_tree", 1.05), ("burnt_tree", 0.9), ("flame", 1.0),
                 ("flame", 0.85), ("flame", 0.7), ("burnt_snag", 0.8)],
    "Logging": [("evergreen", 1.05), ("stump", 0.9), ("log", 1.0), ("stump", 0.8)],
    "Other natural disturbances": [("flame", 0.9), ("evergreen", 0.85)],
    "Hard commodities": [("stump", 0.85), ("log", 0.95)],
    "Settlements & Infrastructure": [("barn", 1.1), ("barn", 0.9)],
}
