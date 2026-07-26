"""Tests for forest_viz.

Charts are exercised on a tiny synthetic frame (no 18MB workbook needed);
we assert on structure rather than pixels.
"""

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd
import pytest

from forest_viz import data, maps, motifs, palette, plots, theme


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


def test_top_countries_drivers_ranks_and_pivots():
    df = pd.DataFrame(
        [
            {"country": "A", "driver": "Wildfire", "year": 2002, "tc_loss_ha": 100.0},
            {"country": "A", "driver": "Logging", "year": 2002, "tc_loss_ha": 100.0},
            {"country": "B", "driver": "Wildfire", "year": 2002, "tc_loss_ha": 10.0},
        ]
    )
    wide = data.top_countries_drivers(df, n=2)
    assert list(wide.index) == ["A", "B"]  # A (200) ranks above B (10)
    assert wide.loc["A", "Logging"] == 100.0
    # per-driver share is row-normalized
    shares = wide.loc["A"] / wide.loc["A"].sum()
    assert shares["Wildfire"] == 0.5


def test_readable_text_contrast():
    assert plots._readable_text("#ffffff") == "#0b0b0b"  # black on light
    assert plots._readable_text("#2a78d6") == "#ffffff"  # white on mid-blue


def test_enlarge_grows_ring_about_centroid():
    ring = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]])
    out = maps._enlarge([ring], factor=1.5, cap_deg=100.0)[0]
    # grown outward past the original bounding box, centroid preserved
    assert out[:, 0].max() > 10.0 and out[:, 0].min() < 0.0
    assert np.allclose(out[:-1].mean(axis=0), ring[:-1].mean(axis=0))


def test_primary_set_single_and_tie():
    # clear single primary
    assert maps._primary_set({"a": 0.70, "b": 0.20, "c": 0.10}) == {"a"}
    # two effectively tied (within 5 points) -> both primary
    assert maps._primary_set({"a": 0.40, "b": 0.37, "c": 0.23}) == {"a", "b"}
    # 6-point gap is not a tie
    assert maps._primary_set({"a": 0.40, "b": 0.34}) == {"a"}
    # zero-share drivers never count
    assert maps._primary_set({"a": 0.0, "b": 0.0}) == set()


def test_every_driver_has_a_diorama_scene():
    for driver in palette.DRIVER_ORDER:
        assert driver in maps.DRIVER_SCENE
        scene = maps._scene(driver)
        assert scene, f"{driver} has no diorama"
        assert len(scene) >= 2, f"{driver} diorama needs a variety of motifs"
        # every motif key resolves to a known drawing function
        assert all(key in motifs.MOTIF for key, _w in scene)


def test_drop_small_islands_keeps_major_rings():
    big = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]])
    mid = np.array([[20.0, 0.0], [26.0, 0.0], [26.0, 6.0], [20.0, 6.0], [20.0, 0.0]])
    tiny = np.array([[40.0, 40.0], [40.5, 40.0], [40.5, 40.5], [40.0, 40.5], [40.0, 40.0]])
    kept = maps._drop_small_islands([big, mid, tiny], min_frac=0.01)
    assert len(kept) == 2  # big + mid kept, tiny (0.25%) dropped
    # never returns empty even if everything is below threshold
    assert len(maps._drop_small_islands([big, tiny], min_frac=0.9)) == 1


def test_separate_disconnects_overlapping_countries():
    a = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]])
    b = a + np.array([2.0, 0.0])  # overlaps a
    disp = maps._separate([[a], [b]], gap=1.5)
    va, vb = a + disp[0], b + disp[1]
    dmin = np.sqrt(((va[:, None, :] - vb[None, :, :]) ** 2).sum(-1).min())
    assert dmin >= 1.0  # outlines end up disconnected, ~gap apart


def test_map_plots_with_synthetic_geometry():
    # Two triangle "countries"; name mapping exercises the override too.
    geometry = {
        "United States of America": [np.array([[-100, 30], [-90, 30], [-95, 40], [-100, 30]])],
        "Brazil": [np.array([[-55, -10], [-45, -10], [-50, 0], [-55, -10]])],
    }
    drivers = pd.DataFrame(
        [
            {"country": "United States", "driver": "Wildfire", "year": 2001, "tc_loss_ha": 80.0},
            {"country": "United States", "driver": "Logging", "year": 2001, "tc_loss_ha": 20.0},
            {"country": "Brazil", "driver": "Permanent agriculture", "year": 2001, "tc_loss_ha": 90.0},
        ]
    )
    theme.apply_theme()
    ax = maps.plot_top_countries_map(drivers, geometry, n=2)
    assert ax.get_legend() is not None


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
    ax2 = plots.plot_top_countries_drivers(drivers, n=1, dark=dark)
    assert ax2.get_legend() is not None  # driver legend present


def test_fmt_ha_is_compact():
    assert plots._fmt_ha(30_000_000) == "30M"
    assert plots._fmt_ha(29_674_764) == "29.7M"
    assert plots._fmt_ha(107_384) == "107.4k"
    assert plots._fmt_ha(0) == "0"
