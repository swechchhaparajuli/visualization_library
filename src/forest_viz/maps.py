"""A "pop-out" world map of the top tree-cover-loss countries.

The top-N countries are enlarged and lifted off the base map with a drop
shadow (as if pulled out). Each country's own shape is sliced into pie
wedges, colored by driver and clipped to the country outline, with wedge
angles proportional to each driver's share of the country's loss -- the
same driver colors as the bar charts. Percentage labels sit inside each
country. Bordering countries are pushed apart so a clear gap separates
them.

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

# Countries reduced to their single largest polygon (drops outlying parts,
# e.g. Alaska and Hawaii for the United States).
_MAINLAND_ONLY = {"United States of America"}

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


def _largest_ring(rings: list) -> np.ndarray:
    return max(rings, key=lambda r: _ring_area_centroid(r)[0])


def _representative_point(rings: list) -> tuple[float, float]:
    _, cx, cy = _ring_area_centroid(_largest_ring(rings))
    return cx, cy


def _enlarge(rings: list, factor: float, cap_deg: float) -> list:
    """Scale each ring about its own centroid; big countries grow less."""
    reach = max(float(np.abs(r - r[:-1].mean(axis=0)).max()) for r in rings) or 1.0
    f = 1.0 + min(factor - 1.0, cap_deg / reach)
    out = []
    for r in rings:
        c = r[:-1].mean(axis=0)
        out.append(c + (r - c) * f)
    return out


def _bbox(rings: list) -> np.ndarray:
    pts = np.vstack(rings)
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def _separate(centers, halfs, gap, iters=600, max_disp=14.0):
    """Push overlapping axis-aligned boxes apart, leaving ``gap`` between them.

    Returns the per-box displacement (new_center - original_center), capped
    at ``max_disp`` so nothing flies across the map.
    """
    c = centers.astype(float).copy()
    for _ in range(iters):
        moved = False
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                d = c[j] - c[i]
                need = halfs[i] + halfs[j] + gap  # [need_x, need_y]
                pen = need - np.abs(d)
                if pen[0] > 0 and pen[1] > 0:  # boxes (plus gap) overlap
                    axis = 0 if pen[0] < pen[1] else 1
                    sign = 1.0 if d[axis] >= 0 else -1.0
                    shift = pen[axis] / 2.0 * sign
                    c[i, axis] -= shift
                    c[j, axis] += shift
                    moved = True
        if not moved:
            break
    disp = c - centers
    mag = np.hypot(disp[:, 0], disp[:, 1])
    scale = np.where(mag > max_disp, max_disp / np.maximum(mag, 1e-9), 1.0)
    return disp * scale[:, None]


def _inside(rings: list, pt) -> bool:
    return any(Path(r).contains_point(pt) for r in rings)


def _rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _darken(color, factor: float) -> tuple:
    r, g, b = color if isinstance(color, tuple) else _rgb(color)
    return (r * factor, g * factor, b * factor)


def _lerp(c0: tuple, c1: tuple, t: float) -> tuple:
    return tuple(a + (b - a) * t for a, b in zip(c0, c1))


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
    enlarge: float = 1.6,
    gap: float = 3.5,
    label_min_share: float = 0.12,
    depth: float = 15.0,
):
    """Pop-out map of the top ``n`` loss countries, each sliced by driver.

    ``geometry`` is a path to a countries GeoJSON or the dict from
    :func:`load_country_geometry`. ``enlarge`` scales the highlighted
    countries (capped for large ones); ``gap`` is the degrees of space kept
    between bordering countries; ``depth`` is the extruded 3-D thickness in
    points.
    """
    chrome = palette.chrome(dark=dark)
    colors = palette.driver_colors(dark=dark)
    mc = _DARK if dark else _LIGHT
    if isinstance(geometry, (str, bytes)) or hasattr(geometry, "__fspath__"):
        geometry = load_country_geometry(geometry)

    if ax is None:
        _, ax = plt.subplots(figsize=(15, 8))
    fig = ax.figure

    # Base map: every country, muted.
    for rings in geometry.values():
        for ring in rings:
            if len(ring) >= 3:
                ax.add_patch(_patch(Path(ring), ax.transData, facecolor=mc["land"],
                                    edgecolor=mc["border"], linewidth=0.4, zorder=1))

    wide = _data.top_countries_drivers(drivers, n=n, year_range=year_range)
    order = [d for d in palette.DRIVER_ORDER if d in wide.columns]

    # Resolve + enlarge each country's geometry.
    items = []  # (country, enlarged_rings)
    for country in wide.index:
        rings = geometry.get(_DATA_TO_NE.get(country, country))
        if not rings:
            continue
        ne = _DATA_TO_NE.get(country, country)
        if ne in _MAINLAND_ONLY:
            rings = [_largest_ring(rings)]
        rings = [r for r in rings if len(r) >= 3]
        items.append((country, _enlarge(rings, enlarge, cap_deg=9.0)))

    # Push bordering countries apart, then translate their geometry.
    boxes = np.array([_bbox(r) for _, r in items])
    centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])
    halfs = np.column_stack([(boxes[:, 2] - boxes[:, 0]) / 2, (boxes[:, 3] - boxes[:, 1]) / 2])
    disp = _separate(centers, halfs, gap=gap)
    items = [(c, [r + d for r in rings]) for (c, rings), d in zip(items, disp)]

    def _off(x, y):
        return mtransforms.offset_copy(ax.transData, fig=fig, x=x, y=y, units="points")

    # The extruded top face is lifted `depth` points up; side-wall layers fill
    # the space down to the footprint on the map.
    top_t = _off(0, depth)
    n_layers = max(int(depth), 8)

    for country, rings in items:
        clip = _compound(rings)

        cx, cy = _representative_point(rings)
        radius = max(float(np.hypot(*(np.vstack(rings) - [cx, cy]).T).max()), 1.0) * 1.05

        shares = wide.loc[country, order]
        total = float(shares.sum()) or 1.0
        dominant = max(order, key=lambda d: float(shares[d]))

        # Contact shadow on the map, then the extruded side wall (dark base ->
        # lighter near the top), giving each country a solid 3-D thickness.
        ax.add_patch(_patch(clip, _off(4, -3), facecolor=mc["shadow"], edgecolor="none",
                            alpha=0.22, zorder=2.8))
        wall_lo = _darken(colors[dominant], 0.38)
        wall_hi = _darken(colors[dominant], 0.68)
        for i in range(n_layers + 1):
            t = i / n_layers
            ax.add_patch(_patch(clip, _off(0, depth * t), facecolor=_lerp(wall_lo, wall_hi, t),
                                edgecolor="none", zorder=3 + t))

        # Top face: pie wedges clipped to the country outline.
        start = 90.0
        for drv in order:
            frac = float(shares[drv]) / total
            if frac <= 0:
                continue
            end = start - frac * 360.0
            wedge = Wedge((cx, cy), radius, end, start, facecolor=colors[drv],
                          edgecolor=chrome["surface"], linewidth=0.6, transform=top_t, zorder=4.5)
            ax.add_patch(wedge)
            wedge.set_clip_path(clip, top_t)
            # Percentage label inside the country, along the wedge mid-angle.
            if frac >= label_min_share:
                mid = np.radians((start + end) / 2.0)
                pt = None
                for rr in (0.6, 0.72, 0.46, 0.82, 0.34):
                    cand = (cx + radius * rr * np.cos(mid), cy + radius * rr * np.sin(mid))
                    if _inside(rings, cand):
                        pt = cand
                        break
                if pt is not None:
                    t = ax.text(pt[0], pt[1], f"{frac * 100:.0f}%", transform=top_t,
                                ha="center", va="center", fontsize=7.5, zorder=6,
                                color=chrome["text"], fontweight="bold")
                    t.set_path_effects([mpe.withStroke(linewidth=2.0, foreground=chrome["surface"])])
            start = end

        # Crisp top-edge outline (surface gap + thin definition line).
        ax.add_patch(_patch(clip, top_t, facecolor="none", edgecolor=chrome["surface"],
                            linewidth=2.6, zorder=5))
        ax.add_patch(_patch(clip, top_t, facecolor="none", edgecolor=chrome["text_secondary"],
                            linewidth=0.5, zorder=5.1))

        # Country label at the centroid (countries are displaced enough to
        # keep names from colliding).
        lbl = ax.text(cx, cy, country, transform=top_t, ha="center", va="center",
                      fontsize=8, color=chrome["text"], fontweight="bold", zorder=6)
        lbl.set_path_effects([mpe.withStroke(linewidth=2.8, foreground=chrome["surface"])])

    ax.set_xlim(-175, 190)
    ax.set_ylim(-60, 88)
    ax.set_aspect("equal")
    ax.axis("off")
    rng = f" ({year_range[0]}–{year_range[1]})" if year_range else ""
    ax.set_title(f"Top {n} countries by tree cover loss, split by driver{rng}",
                 color=chrome["text"], fontweight="bold", fontsize=15)

    handles = [Patch(facecolor=colors[d], edgecolor=chrome["surface"], label=d) for d in order]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              frameon=False, fontsize=9, labelcolor=chrome["text_secondary"], ncol=4)
    return ax
