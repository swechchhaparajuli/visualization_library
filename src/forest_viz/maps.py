"""A "pop-out" world map of the top tree-cover-loss countries.

The top-N countries are lifted off the base map with a drop shadow (as if
pulled out), and each carries a small donut showing its loss split by
driver -- the same driver colors as the bar charts.

Geometry is a Natural Earth 110m countries GeoJSON, parsed directly (no
geopandas). Coordinates are drawn as plate-carree (x = longitude,
y = latitude), which is enough for a highlight map.
"""

from __future__ import annotations

import json

import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import PathPatch, Patch, Wedge
from matplotlib.path import Path

from forest_viz import data as _data
from forest_viz import palette

# GeoJSON "ADMIN" names that differ from the data's country names.
_DATA_TO_NE = {
    "United States": "United States of America",
    "México": "Mexico",
}

# Map chrome (kept local to the map look).
_LIGHT = {"land": "#dcdbd4", "lift": "#ffffff", "border": "#b7b6ae", "shadow": "#0b0b0b"}
_DARK = {"land": "#33332f", "lift": "#57574f", "border": "#4a4a47", "shadow": "#000000"}


def load_country_geometry(path) -> dict[str, list[list[tuple[float, float]]]]:
    """Parse a countries GeoJSON into ``ADMIN name -> list of exterior rings``.

    Interior holes are dropped (immaterial at 110m for a highlight map).
    """
    gj = json.load(open(path))
    out: dict[str, list] = {}
    for feat in gj["features"]:
        name = feat["properties"].get("ADMIN", "")
        geom = feat["geometry"]
        rings: list = []
        if geom["type"] == "Polygon":
            rings.append(geom["coordinates"][0])
        elif geom["type"] == "MultiPolygon":
            rings.extend(poly[0] for poly in geom["coordinates"])
        out[name] = [np.asarray(r, dtype=float) for r in rings]
    return out


def _ring_area_centroid(ring: np.ndarray) -> tuple[float, float, float]:
    """Signed-area centroid of a ring; returns (area, cx, cy)."""
    x, y = ring[:, 0], ring[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = cross.sum() / 2.0
    if area == 0:
        return 0.0, float(x.mean()), float(y.mean())
    cx = ((x[:-1] + x[1:]) * cross).sum() / (6 * area)
    cy = ((y[:-1] + y[1:]) * cross).sum() / (6 * area)
    return abs(area), cx, cy


def _representative_point(rings: list[np.ndarray]) -> tuple[float, float]:
    """Centroid of the largest ring (mainland), for glyph placement."""
    best = max((_ring_area_centroid(r) for r in rings), key=lambda t: t[0])
    return best[1], best[2]


def _patch(ring: np.ndarray, transform, **kw) -> PathPatch:
    return PathPatch(Path(ring), transform=transform, **kw)


def _declutter(xy: np.ndarray, min_dist: float, bounds, iters: int = 400) -> np.ndarray:
    """Nudge overlapping points apart (simple pairwise repulsion)."""
    pos = xy.astype(float).copy()
    (xlo, xhi, ylo, yhi) = bounds
    for _ in range(iters):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[j] - pos[i]
                dist = float(np.hypot(*d)) or 1e-6
                if dist < min_dist:
                    push = (min_dist - dist) / 2.0
                    step = d / dist * push
                    pos[i] -= step
                    pos[j] += step
                    moved = True
        pos[:, 0] = np.clip(pos[:, 0], xlo, xhi)
        pos[:, 1] = np.clip(pos[:, 1], ylo, yhi)
        if not moved:
            break
    return pos


def plot_top_countries_map(
    drivers,
    geometry,
    n: int = 15,
    year_range=None,
    ax=None,
    dark: bool = False,
    donut_radius: float = 6.0,
):
    """Draw the pop-out map of the top ``n`` loss countries with driver donuts.

    ``geometry`` is either a path to a countries GeoJSON or the dict returned
    by :func:`load_country_geometry`.
    """
    chrome = palette.chrome(dark=dark)
    colors = palette.driver_colors(dark=dark)
    mc = _DARK if dark else _LIGHT
    if isinstance(geometry, (str, bytes)) or hasattr(geometry, "__fspath__"):
        geometry = load_country_geometry(geometry)

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 7))
    fig = ax.figure

    # Base map: every country, muted.
    for rings in geometry.values():
        for ring in rings:
            if len(ring) >= 3:
                ax.add_patch(_patch(ring, ax.transData, facecolor=mc["land"],
                                    edgecolor=mc["border"], linewidth=0.4, zorder=1))

    wide = _data.top_countries_drivers(drivers, n=n, year_range=year_range)
    order = [d for d in palette.DRIVER_ORDER if d in wide.columns]

    # Lift/shadow offsets in points (shadow down-right, country up).
    shadow_t = mtransforms.offset_copy(ax.transData, fig=fig, x=5, y=-6, units="points")
    lift_t = mtransforms.offset_copy(ax.transData, fig=fig, x=0, y=3, units="points")

    # Resolve geometry + centroid for each country, then declutter the donuts.
    resolved = []
    for country in wide.index:
        rings = geometry.get(_DATA_TO_NE.get(country, country))
        if rings:
            resolved.append((country, rings, _representative_point(rings)))

    origin = np.array([c for _, _, c in resolved], dtype=float)
    donut_xy = _declutter(origin, min_dist=donut_radius * 2.3,
                          bounds=(-168, 183, -55, 80))

    # Lift + shadow every highlighted country.
    for country, rings, _c in resolved:
        for ring in rings:
            if len(ring) < 3:
                continue
            ax.add_patch(_patch(ring, shadow_t, facecolor=mc["shadow"], edgecolor="none",
                                alpha=0.28, zorder=3))
            ax.add_patch(_patch(ring, lift_t, facecolor=mc["lift"],
                                edgecolor=chrome["text_secondary"], linewidth=0.8, zorder=4))

    # Draw donuts (+ leader line if displaced) and labels on top of all shapes.
    for (country, _rings, (ox, oy)), (dx, dy) in zip(resolved, donut_xy):
        if np.hypot(dx - ox, dy - oy) > donut_radius * 0.6:
            ax.plot([ox, dx], [oy, dy], transform=lift_t, color=chrome["muted"],
                    linewidth=0.8, zorder=4.5)
        shares = wide.loc[country, order]
        total = float(shares.sum()) or 1.0
        start = 90.0
        for drv in order:
            frac = float(shares[drv]) / total
            if frac <= 0:
                continue
            end = start - frac * 360.0
            ax.add_patch(Wedge((dx, dy), donut_radius, end, start, width=donut_radius * 0.5,
                               facecolor=colors[drv], edgecolor=chrome["surface"],
                               linewidth=0.6, transform=lift_t, zorder=5))
            start = end
        label = ax.text(dx, dy - donut_radius - 1.2, country, transform=lift_t, ha="center",
                        va="top", fontsize=7.5, color=chrome["text"], fontweight="bold", zorder=6)
        label.set_path_effects([mpe.withStroke(linewidth=2.2, foreground=chrome["surface"])])

    ax.set_xlim(-170, 185)
    ax.set_ylim(-58, 84)
    ax.set_aspect("equal")
    ax.axis("off")
    rng = f" ({year_range[0]}–{year_range[1]})" if year_range else ""
    ax.set_title(f"Top {n} countries by tree cover loss, split by driver{rng}",
                 color=chrome["text"], fontweight="bold", fontsize=15)

    handles = [Patch(facecolor=colors[d], edgecolor=chrome["surface"], label=d) for d in order]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              frameon=False, fontsize=9, labelcolor=chrome["text_secondary"], ncol=4)
    return ax
