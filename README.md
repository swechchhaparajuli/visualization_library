# simple_eda

The simplest exploratory-data-analysis API. One import, one call.

```python
import simple_eda as eda

eda.summarize(df)
```

## Install

```bash
pip install -e .
```

## Usage

`summarize()` takes a pandas DataFrame and returns a `Summary` you can print
for a readable report or convert to a dict for programmatic use.

```python
import pandas as pd
import simple_eda as eda

df = pd.DataFrame({"a": [1, 2, 3, None], "b": ["x", "y", "z", "z"]})

summary = eda.summarize(df)
print(summary)          # human-readable report
summary.to_dict()       # raw values (shape, dtypes, missing, numeric stats)
```

The report includes:

- **shape** — row and column counts
- **dtypes** — the type of each column
- **missing** — missing-value count per column (and a total)
- **numeric** — descriptive statistics for numeric columns

## Scope (MVP)

Input is **pandas DataFrames only**. Polars, Spark, Dask, and Arrow are not
supported yet — fewer choices means fewer bugs.

## Project layout

```
simple-eda-project/
├── src/
│   └── simple_eda/
│       ├── __init__.py
│       └── core.py
├── README.md
├── pyproject.toml
└── LICENSE
```
