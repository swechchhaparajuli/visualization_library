"""Core exploratory-data-analysis helpers for :mod:`simple_eda`.

The MVP supports pandas :class:`~pandas.DataFrame` inputs only. Polars,
Spark, Dask, and Arrow are intentionally not handled yet -- fewer choices
means fewer bugs.
"""

from __future__ import annotations

import pandas as pd


def summarize(df):
    """Return the shape, columns, and dtypes of a DataFrame."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "types": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def missing(df):
    """Return null counts per column, most-missing first."""
    counts = df.isna().sum()
    return counts.sort_values(ascending=False)


def numeric_columns(df):
    """Return the names of the numeric columns."""
    sel = df.select_dtypes(include="number")
    return list(sel.columns)


def categorical_columns(df):
    """Return the names of the object/category columns."""
    sel = df.select_dtypes(include=["object", "category"])
    return list(sel.columns)
