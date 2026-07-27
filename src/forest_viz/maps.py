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
import zlib

import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Ellipse, PathPatch, Polygon, Rectangle, Wedge
from matplotlib.path import Path

from forest_viz import data as _data
from forest_viz import motifs as _motifs
from forest_viz import palette

# Each driver's primary sector is drawn as a little diorama composed from a
# SET of flat vector motifs (see forest_viz.motifs), not one repeated icon.
DRIVER_SCENE = _motifs.SCENES

# One representative motif per driver, for the legend key.
_LEGEND_MOTIF = {
    "Permanent agriculture": "crop",
    "Shifting cultivation": "seedling",
    "Wildfire": "flame",
    "Logging": "log",
    "Other natural disturbances": "tornado",
    "Hard commodities": "oil_derrick",
    "Settlements & Infrastructure": "cityscape",
}


def _scene(driver: str):
    """Return the diorama motif spec for a driver, or None."""
    return DRIVER_SCENE.get(driver) or None

# GeoJSON "ADMIN" names that differ from the data's country names.
_DATA_TO_NE = {
    "United States": "United States of America",
    "México": "Mexico",
}

# Countries reduced to their single largest polygon (drops outlying parts,
# e.g. Alaska and Hawaii for the United States).
_MAINLAND_ONLY = {"United States of America"}

# Force a label side for specific countries ("left" / "right"); otherwise the
# label goes above and falls back to a clear side automatically.
_LABEL_SIDE = {"Argentina": "left"}

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


def _drop_small_islands(rings: list, min_frac: float) -> list:
    """Drop outlying rings smaller than ``min_frac`` of the largest ring.

    Keeps major landmasses (so archipelagos like Indonesia stay intact) while
    removing tiny outlying islands that clutter a big country's silhouette.
    """
    if len(rings) <= 1:
        return rings
    areas = [_ring_area_centroid(r)[0] for r in rings]
    biggest = max(areas) or 1.0
    kept = [r for r, a in zip(rings, areas) if a / biggest >= min_frac]
    return kept or [_largest_ring(rings)]


def _enlarge(rings: list, factor: float, cap_deg: float) -> list:
    """Scale each ring about its own centroid; big countries grow less."""
    reach = max(float(np.abs(r - r[:-1].mean(axis=0)).max()) for r in rings) or 1.0
    f = 1.0 + min(factor - 1.0, cap_deg / reach)
    out = []
    for r in rings:
        c = r[:-1].mean(axis=0)
        out.append(c + (r - c) * f)
    return out


def _separate(list_of_rings, gap, iters=300, max_disp=18.0, step=0.7):
    """Nudge countries apart until bordering outlines are ``gap`` degrees apart.

    Unlike a bounding-box scheme, this measures actual outline proximity (and
    containment), so only countries whose shapes touch or overlap move, and
    only by the minimum needed -- distant countries stay put. Returns a
    per-country (dx, dy) displacement.
    """
    n = len(list_of_rings)
    cents = np.array([_representative_point(r) for r in list_of_rings], dtype=float)
    pts, paths = [], []
    for rings in list_of_rings:
        p = np.vstack(rings)
        k = max(1, len(p) // 90)
        pts.append(p[::k])
        paths.append(_compound(rings))

    disp = np.zeros((n, 2))
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = pts[i] + disp[i], pts[j] + disp[j]
                d = bi[:, None, :] - bj[None, :, :]
                dmin = float(np.sqrt((d ** 2).sum(-1).min()))
                overlap = dmin >= gap and (
                    paths[i].contains_points(bj - disp[i]).any()
                    or paths[j].contains_points(bi - disp[j]).any()
                )
                if dmin < gap or overlap:
                    vec = (cents[i] + disp[i]) - (cents[j] + disp[j])
                    norm = float(np.hypot(*vec)) or 1.0
                    unit = vec / norm if norm > 1e-6 else np.array([1.0, 0.0])
                    shift = step if overlap else (gap - dmin)
                    disp[i] += unit * shift / 2
                    disp[j] -= unit * shift / 2
                    moved = True
        mag = np.hypot(disp[:, 0], disp[:, 1])
        over = mag > max_disp
        if over.any():
            disp[over] *= (max_disp / mag[over])[:, None]
        if not moved:
            break
    return disp


def _inside(rings: list, pt) -> bool:
    return any(Path(r).contains_point(pt) for r in rings)


def _inside_any(list_of_rings: list, pt) -> bool:
    return any(_inside(rings, pt) for rings in list_of_rings)


def _band_hits(list_of_rings: list, x0: float, x1: float, y: float, n: int = 7) -> bool:
    """True if any sample point along the horizontal span [x0, x1] at y lands
    inside one of the countries (used to test if a label would overlap)."""
    return any(_inside_any(list_of_rings, (x, y)) for x in np.linspace(x0, x1, n))


def _rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _darken(color, factor: float) -> tuple:
    r, g, b = color if isinstance(color, tuple) else _rgb(color)
    return (r * factor, g * factor, b * factor)


def _lerp(c0: tuple, c1: tuple, t: float) -> tuple:
    return tuple(a + (b - a) * t for a, b in zip(c0, c1))


def _pale(hex_color: str, t: float = 0.6) -> tuple:
    """Lighten a color toward white (for the backdrop under tiled icons)."""
    return _lerp(_rgb(hex_color), (1.0, 1.0, 1.0), t)


def _primary_set(fracs: dict, tol: float = 0.05) -> set:
    """Drivers that are the primary one -- the top share plus any driver
    within ``tol`` (share points) of it (an effective tie)."""
    if not fracs:
        return set()
    smax = max(fracs.values())
    return {d for d, f in fracs.items() if f > 0 and smax - f <= tol}


def _diorama(ax, scene, cx, cy, radius, start_deg, frac, rings, transform, size, seed,
             spacing=None, y_scale=1.0, y_offset=0.0, z0=5.4):
    """Compose a little diorama across a pie sector (clipped to ``rings``).

    Points in the sector are populated with a mix of the driver's flat vector
    ``scene`` motifs at varied sizes and jittered positions, drawn
    back-to-front for depth. Motif count scales with area. ``seed`` makes the
    layout deterministic; ``size`` is the base motif height in data units;
    ``spacing`` overrides the grid spacing (default scales with ``radius``).
    """
    rng = np.random.RandomState(seed)
    if spacing is None:
        spacing = min(6.5, max(2.6, radius * 0.20))
    span = frac * 360.0

    def _emit(x, y):
        key, w = scene[rng.randint(len(scene))]
        s = size * w * (0.82 + rng.rand() * 0.4)
        return (y, x, s, key)

    placed = []
    xs = np.arange(cx - radius, cx + radius + 1e-9, spacing)
    ys = np.arange(cy - radius, cy + radius + 1e-9, spacing)
    for x in xs:
        for y in ys:
            jx = x + (rng.rand() - 0.5) * spacing * 0.75
            jy = y + (rng.rand() - 0.5) * spacing * 0.75
            dx, dy = jx - cx, jy - cy
            if np.hypot(dx, dy) > radius * 0.95:
                continue
            if (start_deg - np.degrees(np.arctan2(dy, dx))) % 360 > span:
                continue
            if not _inside(rings, (jx, jy)):
                continue
            placed.append(_emit(jx, jy))

    if not placed:  # tiny sector: guarantee at least one element
        mid = np.radians(start_deg - span / 2.0)
        for rr in (0.5, 0.35, 0.65):
            x, y = cx + radius * rr * np.cos(mid), cy + radius * rr * np.sin(mid)
            if _inside(rings, (x, y)):
                placed.append(_emit(x, y))
                break

    # Back-to-front: higher latitude first, lower (nearer) drawn last / on top.
    # y_scale < 1 places motifs on a vertically-squashed (isometric) surface
    # while they are still drawn upright.
    for i, (y, x, s, key) in enumerate(sorted(placed, key=lambda t: -t[0])):
        _motifs.MOTIF[key](ax, x, y * y_scale + y_offset, s, transform, z0 + i * 0.003)


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
    enlarge: float = 1.35,
    gap: float = 3.5,
    depth: float = 15.0,
    island_min_frac: float = 0.01,
    icon_scale: float = 1.0,
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
    theme = palette.driver_theme(dark=dark)
    mc = _DARK if dark else _LIGHT
    if isinstance(geometry, (str, bytes)) or hasattr(geometry, "__fspath__"):
        geometry = load_country_geometry(geometry)

    if ax is None:
        _, ax = plt.subplots(figsize=(16.5, 9))
    fig = ax.figure

    # View window with generous north/south whitespace; a surface-colored
    # backdrop spans it so the margin survives a tight-bbox save.
    xmin, xmax, ymin, ymax = -186, 200, -116, 120
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           facecolor=chrome["surface"], edgecolor="none", zorder=0))

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
        rings = _drop_small_islands([r for r in rings if len(r) >= 3], island_min_frac)
        items.append((country, _enlarge(rings, enlarge, cap_deg=9.0)))

    # Nudge only the countries whose outlines touch/overlap, minimally.
    disp = _separate([r for _, r in items], gap=gap)
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
        fracs = {d: float(shares[d]) / total for d in order}
        dominant = max(order, key=lambda d: fracs[d])
        # Primary driver(s): the top share, plus any within ~5 points of it.
        primary = _primary_set(fracs, tol=0.05)

        # Contact shadow on the map, then the extruded side wall (dark base ->
        # lighter near the top), giving each country a solid 3-D thickness.
        ax.add_patch(_patch(clip, _off(4, -3), facecolor=mc["shadow"], edgecolor="none",
                            alpha=0.22, zorder=2.8))
        wall_lo = _darken(theme[dominant], 0.38)
        wall_hi = _darken(theme[dominant], 0.68)
        for i in range(n_layers + 1):
            t = i / n_layers
            ax.add_patch(_patch(clip, _off(0, depth * t), facecolor=_lerp(wall_lo, wall_hi, t),
                                edgecolor="none", zorder=3 + t))

        # Top face: pie wedges clipped to the country outline. The primary
        # driver's sector is filled with a flat-motif diorama, not a flat color.
        icon_size = min(6.5, max(2.6, radius * 0.20)) * icon_scale
        start = 90.0
        for drv in order:
            frac = fracs[drv]
            if frac <= 0:
                continue
            end = start - frac * 360.0
            scene = _scene(drv) if drv in primary else None
            # Primary driver: themed "biome" color behind its diorama.
            # Secondary drivers: their muted pastel-green shade.
            face = _lerp(_rgb(theme[drv]), (1.0, 1.0, 1.0), 0.12) if scene else colors[drv]
            wedge = Wedge((cx, cy), radius, end, start, facecolor=face,
                          edgecolor="none", transform=top_t, zorder=4.5)
            ax.add_patch(wedge)
            wedge.set_clip_path(clip, top_t)
            if scene:
                seed = zlib.crc32(f"{country}:{drv}".encode()) & 0xFFFFFFFF
                _diorama(ax, scene, cx, cy, radius, start, frac, rings, top_t, icon_size, seed)
            # Percentage label for the primary driver only, inside the country.
            if drv in primary:
                mid = np.radians((start + end) / 2.0)
                pt = None
                for rr in (0.6, 0.72, 0.46, 0.82, 0.34):
                    cand = (cx + radius * rr * np.cos(mid), cy + radius * rr * np.sin(mid))
                    if _inside(rings, cand):
                        pt = cand
                        break
                if pt is not None:
                    t = ax.text(pt[0], pt[1], f"{frac * 100:.0f}%", transform=top_t,
                                ha="center", va="center", fontsize=10.5, zorder=6,
                                color=chrome["text"], fontweight="bold")
                    t.set_path_effects([mpe.withStroke(linewidth=2.6, foreground=chrome["surface"])])
            start = end

        # Country label above the country; if "above" would land on another
        # country, place it beside instead.
        pts = np.vstack(rings)
        minx, maxx = pts[:, 0].min(), pts[:, 0].max()
        maxy = pts[:, 1].max()
        others = [r for c2, r in items if c2 != country]
        halfw = len(country) * 1.2  # ~half the label's width, in degrees
        ly_above = maxy + 2.8
        override = _LABEL_SIDE.get(country)
        if override == "left":
            lx, ly, ha, va = minx - 2.5, cy, "right", "center"
        elif override == "right":
            lx, ly, ha, va = maxx + 2.5, cy, "left", "center"
        elif not _band_hits(others, cx - halfw, cx + halfw, ly_above):
            lx, ly, ha, va = cx, ly_above, "center", "bottom"
        elif not _band_hits(others, maxx + 2.5, maxx + 2.5 + 2 * halfw, cy):
            lx, ly, ha, va = maxx + 2.5, cy, "left", "center"
        elif not _band_hits(others, minx - 2.5 - 2 * halfw, minx - 2.5, cy):
            lx, ly, ha, va = minx - 2.5, cy, "right", "center"
        else:
            lx, ly, ha, va = cx, ly_above, "center", "bottom"
        lbl = ax.text(lx, ly, country, transform=top_t, ha=ha, va=va,
                      fontsize=11.5, color=chrome["text"], fontweight="bold", zorder=6)
        lbl.set_path_effects([mpe.withStroke(linewidth=3.4, foreground=chrome["surface"])])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    rng = f" ({year_range[0]}–{year_range[1]})" if year_range else ""
    ax.set_title(f"Top {n} countries by tree cover loss, split by driver{rng}",
                 color=chrome["text"], fontweight="bold", fontsize=15)

    # Legend: draw each driver's motif (all drivers, primary or not) with its
    # name, in a grid along the bottom whitespace.
    cols = 4
    col_x = np.linspace(-158, 66, cols)
    row_y = (-88, -102)
    for i, d in enumerate(palette.DRIVER_ORDER):
        x, y = col_x[i % cols], row_y[i // cols]
        key = _LEGEND_MOTIF.get(d)
        if key:
            _motifs.MOTIF[key](ax, x, y - 4.5, 10.0, ax.transData, 7)
        ax.text(x + 9, y, d, ha="left", va="center", fontsize=10.5,
                color=chrome["text_secondary"], zorder=7)
    return ax


def plot_driver_pie(drivers, country=None, year_range=None, ax=None, dark=False,
                    tilt=0.52, depth=0.22):
    """Isometric pie of loss by driver; each slice filled with driver motifs.

    The pie is tilted into an ellipse and extruded into a 3-D disc: each slice
    takes the driver's themed biome color, its front rim is a darker side
    wall, and its surface is filled with a diorama of that driver's motifs
    (drawn upright), sized to its share. Name + percentage label each slice.
    """
    chrome = palette.chrome(dark=dark)
    theme = palette.driver_theme(dark=dark)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    totals = _data.driver_totals(drivers, country=country, year_range=year_range)
    order = [d for d in palette.DRIVER_ORDER if d in totals.index and totals[d] > 0]
    total = float(totals.sum()) or 1.0

    R = 1.0
    ang = np.linspace(0, 2 * np.pi, 181)
    circle = [np.column_stack([R * np.cos(ang), R * np.sin(ang)])]  # flat, for clipping

    def rim(a):  # base-plane rim point at angle a (degrees)
        return np.array([R * np.cos(np.radians(a)), R * np.sin(np.radians(a)) * tilt])

    # Slice angles + a distinct extruded height per slice (taller = bigger share).
    spans, start = [], 90.0
    fmax = max((totals[d] / total for d in order), default=1.0)
    for d in order:
        frac = totals[d] / total
        h = depth * (0.6 + 1.1 * frac / fmax)  # height varies by share
        spans.append((d, frac, start, start - frac * 360.0, h))
        start -= frac * 360.0

    # Soft ground shadow under the disc.
    ax.add_patch(Ellipse((0.06, -0.06), 2 * R * 1.02, 2 * R * tilt * 1.02, facecolor="#1c1c1a",
                         alpha=0.12, edgecolor="none", zorder=0.5))

    # Draw back-to-front (slices with higher mid-latitude are farther away) so
    # nearer/taller slices overlap correctly; each slice gets its own z band.
    ordered = sorted(enumerate(spans), key=lambda t: -np.sin(np.radians((t[1][2] + t[1][3]) / 2.0)))
    mids = []
    for zi, (_, (d, frac, a0, a1, h)) in enumerate(ordered):
        zbase = 2 + zi * 0.5
        up = np.array([0, h])
        # Radial cut walls (the straight sides), shaded darker; skip clearly
        # back-facing edges. Left/right edges get slightly different shading.
        for edge, shade in ((a0, 0.52), (a1, 0.44)):
            if np.sin(np.radians(edge)) < 0.35:
                base_r = rim(edge)
                quad = [(0, 0), base_r, base_r + up, (0, h)]
                ax.add_patch(Polygon(quad, closed=True, facecolor=_darken(theme[d], shade),
                                     edgecolor="none", zorder=zbase))
        # Outer arc wall: front-facing arc, base plane up to this slice's height.
        ths = np.linspace(a0, a1, 60)
        ths = ths[np.sin(np.radians(ths)) < 0.03]
        if len(ths) >= 2:
            top = np.column_stack([R * np.cos(np.radians(ths)), R * np.sin(np.radians(ths)) * tilt + h])
            base = top - up
            ax.add_patch(Polygon(np.vstack([top, base[::-1]]), closed=True,
                                 facecolor=_darken(theme[d], 0.6), edgecolor="none", zorder=zbase + 0.05))
            # Inner shadow: darker band along the bottom of the arc wall.
            mid_band = (top + base) / 2
            ax.add_patch(Polygon(np.vstack([mid_band, base[::-1]]), closed=True,
                                 facecolor="#000000", alpha=0.16, edgecolor="none", zorder=zbase + 0.06))
        # Top face, lifted by h.
        lift = mtransforms.Affine2D().scale(1.0, tilt).translate(0, h) + ax.transData
        ax.add_patch(Wedge((0, 0), R, a1, a0, facecolor=_lerp(_rgb(theme[d]), (1, 1, 1), 0.12),
                           edgecolor="none", transform=lift, zorder=zbase + 0.1))
        scene = _scene(d)
        if scene:
            seed = zlib.crc32(d.encode()) & 0xFFFFFFFF
            _diorama(ax, scene, 0.0, 0.0, R, a0, frac, circle, ax.transData, size=0.15,
                     seed=seed, spacing=0.22, y_scale=tilt, y_offset=h, z0=zbase + 0.2)
        mids.append((np.radians((a0 + a1) / 2.0), d, frac, h))

    # 4) Outside labels with leaders, spread evenly down each side.
    right = [it for it in mids if np.cos(it[0]) >= 0]
    left = [it for it in mids if np.cos(it[0]) < 0]
    for items, is_right in ((right, True), (left, False)):
        items = sorted(items, key=lambda it: np.sin(it[0]), reverse=True)
        n = len(items)
        ys = np.linspace(1.05, -0.95, n) if n > 1 else [np.sin(items[0][0]) * tilt if items else 0.0]
        lx = 1.4 if is_right else -1.4
        for (mid, d, frac, h), ly in zip(items, ys):
            ax.annotate(f"{d}  {frac * 100:.0f}%",
                        xy=(0.92 * R * np.cos(mid), 0.92 * R * np.sin(mid) * tilt + h),
                        xytext=(lx, ly), ha="left" if is_right else "right", va="center",
                        fontsize=10.5, color=chrome["text"], fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=chrome["muted"], lw=0.8))

    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-1.15, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    rng = f" ({year_range[0]}–{year_range[1]})" if year_range else ""
    scope = country if country else "Global"
    ax.set_title(f"{scope} primary-forest loss by driver{rng}",
                 color=chrome["text"], fontweight="bold", fontsize=15, pad=14)
    return ax
