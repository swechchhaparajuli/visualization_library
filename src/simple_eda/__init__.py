"""simple_eda: the simplest exploratory-data-analysis API.

    import simple_eda as eda
    eda.summarize(df)
"""

from simple_eda.core import Summary, summarize

__all__ = ["Summary", "summarize"]
__version__ = "0.1.0"
