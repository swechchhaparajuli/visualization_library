"""Tests for forest_viz.

Charts are exercised on a tiny synthetic frame (no 18MB workbook needed);
we assert on structure rather than pixels.
"""

import matplotlib

matplotlib.use("Agg")  # headless

import pandas as pd
import pytest

from forest_viz import data, palette, plots, theme


@pytest.fixture
def tcl():
    return pd.DataFrame(
        {
            "country": ["A", "A", "B", "B"],
            "year": [2001, 2002, 2001, 2002],
            "tc_loss_ha": [10.0, 20.0, 5.0, 1.0],
        }
    )


@pytest.fixture
def drivers():
    rows = []
    for year in (2002, 2003):
        for drv, val in [("Permanent agriculture", 100.0), ("Wildfire", 40.0), ("Logging", 10.0)]:
            rows.append({"country": "A", "driver": drv, "year": year, "tc_loss_ha": val})
    return pd.DataFrame(rows)


def test_loss_by_year_sums_across_countries(tcl):
    out = data.loss_by_year(tcl)
    assert list(out["year"]) == [2001, 2002]
    assert list(out["tc_loss_ha"]) == [15.0, 21.0]


def test_loss_by_year_country_filter(tcl):
    out = data.loss_by_year(tcl, country="A")
    assert list(out["tc_loss_ha"]) == [10.0, 20.0]


def test_loss_by_country_ranked_and_year_range(tcl):
    out = data.loss_by_country(tcl)
    assert list(out["country"]) == ["A", "B"]  # A (30) > B (6)
    only_2001 = data.loss_by_country(tcl, year_range=(2001, 2001))
    assert dict(zip(only_2001["country"], only_2001["tc_loss_ha"])) == {"A": 10.0, "B": 5.0}


def test_drivers_by_year_pivot(drivers):
    wide = data.drivers_by_year(drivers)
    assert list(wide.index) == [2002, 2003]
    assert set(wide.columns) == {"Permanent agriculture", "Wildfire", "Logging"}
    assert wide.loc[2002, "Permanent agriculture"] == 100.0


def test_driver_totals_sorted(drivers):
    totals = data.driver_totals(drivers)
    assert list(totals.index) == ["Permanent agriculture", "Wildfire", "Logging"]
    assert totals.iloc[0] == 200.0


def test_palette_covers_all_drivers():
    assert len(palette.DRIVER_ORDER) == 7
    assert set(palette.driver_colors()) == set(palette.DRIVER_ORDER)
    assert set(palette.driver_colors(dark=True)) == set(palette.DRIVER_ORDER)


@pytest.mark.parametrize("dark", [False, True])
def test_plot_functions_return_axes(tcl, drivers, dark):
    theme.apply_theme(dark=dark)
    assert plots.plot_loss_trend(tcl, dark=dark) is not None
    assert plots.plot_top_countries(tcl, n=2, dark=dark) is not None
    assert plots.plot_drivers_over_time(drivers, dark=dark) is not None
    ax = plots.plot_driver_composition(drivers, dark=dark)
    # legend/labels present so identity is never color-alone
    assert ax.get_xlabel() != ""


def test_fmt_ha_is_compact():
    assert plots._fmt_ha(30_000_000) == "30M"
    assert plots._fmt_ha(29_674_764) == "29.7M"
    assert plots._fmt_ha(107_384) == "107.4k"
    assert plots._fmt_ha(0) == "0"
