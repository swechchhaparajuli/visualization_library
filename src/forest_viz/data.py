"""Load and tidy Global Forest Watch tree-cover-loss data.

The source workbook stores tree-cover loss in *wide* form (one
``tc_loss_ha_YYYY`` column per year) and driver attribution in *long*
form. These loaders return tidy, long-format DataFrames that the plotting
functions consume, plus a few convenience aggregations.

The MVP supports the GFW country-level workbook. Pass ``threshold=30`` to
match the driver sheets (which are published at the 30% canopy-density
threshold only).
"""

from __future__ import annotations

import re

import pandas as pd

_YEAR_COL = re.compile(r"tc_loss_ha_(\d{4})")

# Sheet names in the GFW global workbook.
SHEET_TREE_COVER_LOSS = "Country tree cover loss"
SHEET_DRIVERS = "Country drivers"
SHEET_PRIMARY_DRIVERS = "Country primary drivers"


def _melt_years(df: pd.DataFrame, id_vars: list[str]) -> pd.DataFrame:
    """Melt wide ``tc_loss_ha_YYYY`` columns into (year, tc_loss_ha) rows."""
    year_cols = [c for c in df.columns if _YEAR_COL.fullmatch(str(c))]
    tidy = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year",
        value_name="tc_loss_ha",
    )
    tidy["year"] = tidy["year"].str.extract(_YEAR_COL).astype(int)
    tidy["tc_loss_ha"] = pd.to_numeric(tidy["tc_loss_ha"], errors="coerce").fillna(0.0)
    return tidy


def load_tree_cover_loss(path, threshold: int = 30) -> pd.DataFrame:
    """Load annual tree-cover loss per country.

    Returns a tidy frame with columns ``country``, ``year``, ``tc_loss_ha``.
    """
    df = pd.read_excel(path, sheet_name=SHEET_TREE_COVER_LOSS)
    df = df[df["threshold"] == threshold]
    return _melt_years(df, id_vars=["country"]).sort_values(["country", "year"]).reset_index(drop=True)


def load_drivers(path, primary: bool = True, threshold: int = 30) -> pd.DataFrame:
    """Load tree-cover loss attributed to each driver, per country and year.

    Parameters
    ----------
    primary:
        When ``True`` (default) load loss of *primary* forest only
        ("Country primary drivers"); otherwise load all tree-cover loss
        ("Country drivers").

    Returns a tidy frame with columns ``country``, ``driver``, ``year``,
    ``tc_loss_ha``.
    """
    sheet = SHEET_PRIMARY_DRIVERS if primary else SHEET_DRIVERS
    df = pd.read_excel(path, sheet_name=sheet)
    df = df[df["threshold"] == threshold].copy()
    df["tc_loss_ha"] = pd.to_numeric(df["tc_loss_ha"], errors="coerce").fillna(0.0)
    keep = ["country", "driver", "year", "tc_loss_ha"]
    return df[keep].sort_values(["country", "year", "driver"]).reset_index(drop=True)


# --- aggregations -----------------------------------------------------------

def loss_by_year(tcl: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Total tree-cover loss per year (global, or one country).

    Returns columns ``year``, ``tc_loss_ha``.
    """
    df = tcl if country is None else tcl[tcl["country"] == country]
    out = df.groupby("year", as_index=False)["tc_loss_ha"].sum()
    return out.sort_values("year").reset_index(drop=True)


def loss_by_country(
    tcl: pd.DataFrame, year_range: tuple[int, int] | None = None
) -> pd.DataFrame:
    """Total tree-cover loss per country, most-loss first.

    ``year_range`` is an inclusive ``(start, end)`` filter on years.
    Returns columns ``country``, ``tc_loss_ha``.
    """
    df = tcl
    if year_range is not None:
        lo, hi = year_range
        df = df[(df["year"] >= lo) & (df["year"] <= hi)]
    out = df.groupby("country", as_index=False)["tc_loss_ha"].sum()
    return out.sort_values("tc_loss_ha", ascending=False).reset_index(drop=True)


def drivers_by_year(drivers: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Pivot driver loss to a year x driver matrix (global, or one country).

    Returns a DataFrame indexed by ``year`` with one column per driver.
    """
    df = drivers if country is None else drivers[drivers["country"] == country]
    wide = df.pivot_table(
        index="year", columns="driver", values="tc_loss_ha", aggfunc="sum", fill_value=0.0
    )
    wide.columns.name = None
    return wide


def driver_totals(
    drivers: pd.DataFrame,
    country: str | None = None,
    year_range: tuple[int, int] | None = None,
) -> pd.Series:
    """Total loss per driver, most-loss first (global, or one country)."""
    df = drivers if country is None else drivers[drivers["country"] == country]
    if year_range is not None:
        lo, hi = year_range
        df = df[(df["year"] >= lo) & (df["year"] <= hi)]
    return df.groupby("driver")["tc_loss_ha"].sum().sort_values(ascending=False)
