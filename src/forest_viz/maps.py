"""A "pop-out" world map of the top tree-cover-loss countries.

The top-N countries are lifted off the base map with a drop shadow (as if
pulled out), and each country is *filled with its own driver breakdown*:
stacked horizontal bands, colored by driver, clipped to the country's
outline, band heights proportional to each driver's share of the country's
loss -- the same driver colors as the bar charts.

Bordering countries are separated by a background-colored gap so their
fills never bleed together.

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
from matplotlib.patches import PathPatch, Patch, Rectangle
from matplotlib.path import Path

from forest_viz import data as _data
from forest_viz import palette

# GeoJSON "ADMIN" names that differ from the data's country names.
_DATA_TO_NE = {
    "United States": "United States of America",
    "México": "Mexico",
}

# Map chrome (kept local to the map look).
_LIGHT = {"land": "#dcdbd4", "border": "#b7b6ae", "shadow": "#0b0b0b"}
_DARK = {"land": "#33332f", "border": "#4a4a47", "shadow": "#000000"}


def load_country_geometry(path) -> dict[str, list]:
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


def _representative_point(rings: list) -> tuple[float, float]:
    """Centroid of the largest ring (mainland), for label placement."""
    best = max((_ring_area_centroid(r) for r in rings), key=lambda t: t[0])
    return best[1], best[2]


def _shrink(ring: np.ndarray, gap: float) -> np.ndarray:
    """Inset a ring toward its own centroid so neighbors don't touch.

    ``gap`` is a small fraction (e.g. 0.06); large countries are inset less
    (a fraction of a smaller reference size) so they aren't badly distorted.
    """
    c = ring[:-1].mean(axis=0)
    d = ring - c
    reach = float(np.abs(d).max()) or 1.0
    # Convert the desired absolute gap (in degrees) into a per-ring factor,
    # capped so big countries shrink only slightly.
    factor = max(1.0 - gap, 1.0 - (gap * 12.0) / reach)
    return c + d * factor


def _compound(rings: list) -> Path:
    return Path.make_compound_path(*[Path(r) for r in rings])


def _patch(path: Path, transform, **kw) -> PathPatch:
    return PathPatch(path, transform=transform, **kw)


def plot_top_countries_map(
    drivers,
    geometry,
    n: int = 15,
    year_range=None,
    ax=None,
    dark: bool = False,
    gap: float = 0.06,
):
    """Pop-out map of the top ``n`` loss countries, each filled by driver share.

    ``geometry`` is either a path to a countries GeoJSON or the dict returned
    by :func:`load_country_geometry`. ``gap`` insets each highlighted country
    slightly so bordering countries are separated by background.
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
                ax.add_patch(_patch(Path(ring), ax.transData, facecolor=mc["land"],
                                    edgecolor=mc["border"], linewidth=0.4, zorder=1))

    wide = _data.top_countries_drivers(drivers, n=n, year_range=year_range)
    order = [d for d in palette.DRIVER_ORDER if d in wide.columns]

    # Lift/shadow offsets in points (shadow down-right, country up).
    shadow_t = mtransforms.offset_copy(ax.transData, fig=fig, x=5, y=-6, units="points")
    lift_t = mtransforms.offset_copy(ax.transData, fig=fig, x=0, y=3, units="points")

    for country in wide.index:
        rings = geometry.get(_DATA_TO_NE.get(country, country))
        if not rings:
            continue
        rings = [_shrink(r, gap) for r in rings if len(r) >= 3]
        if not rings:
            continue
        clip = _compound(rings)

        # Drop shadow (whole silhouette), then the driver bands on top.
        ax.add_patch(_patch(clip, shadow_t, facecolor=mc["shadow"], edgecolor="none",
                            alpha=0.28, zorder=3))

        allpts = np.vstack(rings)
        minx, maxx = allpts[:, 0].min(), allpts[:, 0].max()
        miny, maxy = allpts[:, 1].min(), allpts[:, 1].max()
        height = (maxy - miny) or 1.0

        shares = wide.loc[country, order]
        total = float(shares.sum()) or 1.0
        cum = 0.0
        for drv in order:
            frac = float(shares[drv]) / total
            if frac <= 0:
                continue
            y0 = miny + cum * height
            band = Rectangle((minx, y0), maxx - minx, frac * height,
                             facecolor=colors[drv], edgecolor=chrome["surface"],
                             linewidth=0.6, transform=lift_t, zorder=4)
            ax.add_patch(band)
            band.set_clip_path(clip, lift_t)
            cum += frac

        # Crisp country outline (background-colored gap + thin definition line).
        ax.add_patch(_patch(clip, lift_t, facecolor="none", edgecolor=chrome["surface"],
                            linewidth=2.4, zorder=5))
        ax.add_patch(_patch(clip, lift_t, facecolor="none", edgecolor=chrome["text_secondary"],
                            linewidth=0.5, zorder=5.1))

        cx, cy = _representative_point(rings)
        label = ax.text(cx, cy, country, transform=lift_t, ha="center", va="center",
                        fontsize=7.5, color=chrome["text"], fontweight="bold", zorder=6)
        label.set_path_effects([mpe.withStroke(linewidth=2.4, foreground=chrome["surface"])])

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
