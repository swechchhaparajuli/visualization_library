"""Chart functions for tree-cover loss and its drivers.

Each function takes a tidy frame from :mod:`forest_viz.data`, draws onto a
matplotlib ``Axes`` (creating one if none is passed), and returns that
``Axes``. Colors come from :mod:`forest_viz.palette`; call
:func:`forest_viz.theme.apply_theme` first for the full look.

Design notes
------------
* Drivers are categorical, so each keeps a fixed color regardless of rank.
* Multi-series charts always carry a legend; single-series charts name the
  series in the title instead.
* Bars and driver-composition charts carry direct value labels (the
  "relief" for hues that sit below 3:1 on the light surface).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from forest_viz import data as _data
from forest_viz import palette


def _fmt_ha(value, _pos=None) -> str:
    """Compact hectare label: 1_500_000 -> '1.5M', 30_000_000 -> '30M'."""
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= div:
            return f"{value / div:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:.0f}"


def _new_ax(ax, figsize):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _readable_text(hex_color: str) -> str:
    """Black or white ink, whichever has more contrast on ``hex_color``."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#0b0b0b" if luminance > 0.4 else "#ffffff"


def _scope_label(country: str | None) -> str:
    return country if country else "Global"


def plot_loss_trend(tcl, country=None, ax=None, dark=False):
    """Annual tree-cover loss over time as an area + line (single series)."""
    chrome = palette.chrome(dark=dark)
    accent = palette.ACCENT_DARK if dark else palette.ACCENT
    ax = _new_ax(ax, (9, 5))

    series = _data.loss_by_year(tcl, country=country)
    years, loss = series["year"].to_numpy(), series["tc_loss_ha"].to_numpy()

    ax.fill_between(years, loss, color=accent, alpha=0.18, zorder=1)
    ax.plot(years, loss, color=accent, linewidth=2, zorder=2)

    # Direct-label the peak year (selective, not every point).
    peak = int(series["tc_loss_ha"].idxmax())
    px, py = series.loc[peak, "year"], series.loc[peak, "tc_loss_ha"]
    ax.scatter([px], [py], s=36, color=accent, zorder=3)
    ax.annotate(
        f"{int(px)}: {_fmt_ha(py)} ha",
        (px, py),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9.5,
        color=chrome["text"],
        fontweight="bold",
    )

    ax.set_title(f"{_scope_label(country)} tree cover loss, {years.min()}–{years.max()}")
    ax.set_ylabel("Tree cover loss (ha/yr)")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.02)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ha))
    return ax


def plot_top_countries(tcl, n=15, year_range=None, ax=None, dark=False):
    """Top ``n`` countries by total tree-cover loss as a ranked bar chart."""
    chrome = palette.chrome(dark=dark)
    accent = palette.ACCENT_DARK if dark else palette.ACCENT
    ax = _new_ax(ax, (9, 6))

    ranked = _data.loss_by_country(tcl, year_range=year_range).head(n).iloc[::-1]
    labels = ranked["country"].to_numpy()
    values = ranked["tc_loss_ha"].to_numpy()

    ax.barh(labels, values, color=accent, height=0.72, zorder=2)
    span = values.max() if len(values) else 1
    for y, v in enumerate(values):
        ax.text(
            v + span * 0.01, y, f"{_fmt_ha(v)}",
            va="center", ha="left", fontsize=9, color=chrome["text_secondary"],
        )

    rng = f" ({year_range[0]}–{year_range[1]})" if year_range else ""
    ax.set_title(f"Top {n} countries by tree cover loss{rng}")
    ax.set_xlabel("Tree cover loss (ha)")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_ha))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    ax.tick_params(axis="y", length=0)
    return ax


def plot_top_countries_drivers(
    drivers, n=15, year_range=None, ax=None, dark=False, min_label_share=0.08
):
    """Top ``n`` countries by loss, each bar split into its driver shares.

    Bar length is total loss (so the ranking still reads as magnitude);
    the colored segments show the driver composition, and segments at or
    above ``min_label_share`` of a country's loss are labeled with their
    percentage. A legend maps color to driver.
    """
    chrome = palette.chrome(dark=dark)
    colors = palette.driver_colors(dark=dark)
    ax = _new_ax(ax, (10.5, 7))

    wide = _data.top_countries_drivers(drivers, n=n, year_range=year_range)
    order = [d for d in palette.DRIVER_ORDER if d in wide.columns]
    wide = wide[order].iloc[::-1]  # largest at top of horizontal bars
    countries = list(wide.index)
    totals = wide.sum(axis=1).to_numpy()
    span = totals.max() if len(totals) else 1.0

    left = [0.0] * len(countries)
    for driver in order:
        vals = wide[driver].to_numpy()
        ax.barh(
            countries, vals, left=left, color=colors[driver], height=0.74,
            edgecolor=chrome["surface"], linewidth=0.8, label=driver, zorder=2,
        )
        ink = _readable_text(colors[driver])
        for i, (v, l, t) in enumerate(zip(vals, left, totals)):
            # Label a segment only when it is both a meaningful share of its
            # country AND wide enough (vs the axis) to hold the text.
            if t > 0 and v / t >= min_label_share and v >= span * 0.03:
                ax.text(
                    l + v / 2, i, f"{v / t * 100:.0f}%",
                    ha="center", va="center", fontsize=8, color=ink,
                )
        left = [l + v for l, v in zip(left, vals)]

    for i, t in enumerate(totals):
        ax.text(
            t + span * 0.01, i, f"{_fmt_ha(t)} ha",
            va="center", ha="left", fontsize=9, color=chrome["text_secondary"],
        )

    rng = f" ({year_range[0]}–{year_range[1]})" if year_range else ""
    ax.set_title(f"Top {n} countries by tree cover loss, by driver{rng}")
    ax.set_xlabel("Tree cover loss (ha)")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_ha))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.14)
    ax.tick_params(axis="y", length=0)
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labs[::-1], loc="lower right", ncol=1)
    return ax


def plot_drivers_over_time(drivers, country=None, ax=None, dark=False):
    """Loss by driver over time as a stacked area chart (composition)."""
    chrome = palette.chrome(dark=dark)
    colors = palette.driver_colors(dark=dark)
    ax = _new_ax(ax, (9.5, 5.5))

    wide = _data.drivers_by_year(drivers, country=country)
    order = [d for d in palette.DRIVER_ORDER if d in wide.columns]
    wide = wide[order]
    years = wide.index.to_numpy()

    ax.stackplot(
        years,
        [wide[d].to_numpy() for d in order],
        labels=order,
        colors=[colors[d] for d in order],
        edgecolor=chrome["surface"],
        linewidth=1.2,  # surface-colored seam between bands
    )

    ax.set_title(f"{_scope_label(country)} primary-forest loss by driver")
    ax.set_ylabel("Tree cover loss (ha/yr)")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.02)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ha))
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labs[::-1], loc="upper left", ncol=1)
    return ax


def plot_driver_composition(drivers, country=None, year_range=None, ax=None, dark=False):
    """Total loss per driver as a ranked bar chart (each driver its own color)."""
    chrome = palette.chrome(dark=dark)
    colors = palette.driver_colors(dark=dark)
    ax = _new_ax(ax, (9, 5))

    totals = _data.driver_totals(drivers, country=country, year_range=year_range)
    totals = totals.iloc[::-1]  # largest at top of horizontal bars
    labels = totals.index.to_numpy()
    values = totals.to_numpy()
    total = values.sum() or 1.0

    ax.barh(labels, values, color=[colors[d] for d in labels], height=0.72, zorder=2)
    span = values.max() if len(values) else 1
    for y, v in enumerate(values):
        ax.text(
            v + span * 0.01, y, f"{_fmt_ha(v)} ha ({v / total * 100:.0f}%)",
            va="center", ha="left", fontsize=9, color=chrome["text_secondary"],
        )

    rng = f" ({year_range[0]}–{year_range[1]})" if year_range else ""
    ax.set_title(f"{_scope_label(country)} primary-forest loss by driver{rng}")
    ax.set_xlabel("Tree cover loss (ha)")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_ha))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.18)
    ax.tick_params(axis="y", length=0)
    return ax
