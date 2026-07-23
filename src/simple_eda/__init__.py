"""simple_eda: the simplest exploratory-data-analysis API.

    import simple_eda as eda
    eda.summarize(df)
"""

from simple_eda.core import (
    categorical_columns,
    missing,
    numeric_columns,
    summarize,
)

__all__ = ["summarize", "missing", "numeric_columns", "categorical_columns"]
__version__ = "0.1.1"
