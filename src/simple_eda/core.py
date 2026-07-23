"""Core exploratory-data-analysis helpers for :mod:`simple_eda`.

The MVP supports pandas :class:`~pandas.DataFrame` inputs only. Polars,
Spark, Dask, and Arrow are intentionally not handled yet -- fewer choices
means fewer bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Summary:
    """A lightweight summary of a :class:`~pandas.DataFrame`.

    Instances render as a human-readable report when printed, while the
    underlying pieces stay available as attributes for programmatic use.
    """

    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    missing: dict[str, int]
    numeric: pd.DataFrame = field(repr=False)

    @property
    def missing_total(self) -> int:
        """Total number of missing (NaN/None) cells across the frame."""
        return int(sum(self.missing.values()))

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a plain, JSON-friendly dictionary."""
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": list(self.columns),
            "dtypes": dict(self.dtypes),
            "missing": dict(self.missing),
            "numeric": self.numeric.to_dict(),
        }

    def __str__(self) -> str:
        lines = [
            "simple_eda summary",
            "==================",
            f"shape: {self.n_rows} rows x {self.n_cols} columns",
            f"missing cells: {self.missing_total}",
            "",
            "columns:",
        ]
        width = max((len(c) for c in self.columns), default=0)
        for col in self.columns:
            miss = self.missing.get(col, 0)
            miss_note = f"  ({miss} missing)" if miss else ""
            lines.append(f"  {col:<{width}}  {self.dtypes[col]}{miss_note}")

        if not self.numeric.empty:
            lines += ["", "numeric columns:", self.numeric.to_string()]

        return "\n".join(lines)


def summarize(df: pd.DataFrame) -> Summary:
    """Summarize a pandas :class:`~pandas.DataFrame`.

    Parameters
    ----------
    df:
        The DataFrame to describe. Must be a pandas ``DataFrame``.

    Returns
    -------
    Summary
        Shape, per-column dtypes, missing-value counts, and descriptive
        statistics for the numeric columns. Print the result for a
        readable report, or call :meth:`Summary.to_dict` for the raw data.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas ``DataFrame``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "simple_eda.summarize() supports pandas DataFrames only; "
            f"got {type(df).__name__!r}."
        )

    columns = [str(c) for c in df.columns]
    dtypes = {str(c): str(dt) for c, dt in df.dtypes.items()}
    missing = {str(c): int(n) for c, n in df.isna().sum().items()}

    numeric_df = df.select_dtypes(include="number")
    numeric = numeric_df.describe().transpose() if not numeric_df.empty else pd.DataFrame()

    return Summary(
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        columns=columns,
        dtypes=dtypes,
        missing=missing,
        numeric=numeric,
    )
