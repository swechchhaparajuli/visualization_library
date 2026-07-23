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

```python
import pandas as pd
import simple_eda as eda

df = pd.DataFrame({
    "name": ["Ana", "Bo", None],
    "age": [25, None, 31],
})
print(eda.summarize(df))
print(eda.missing(df))
```

## Core functions

- **`summarize(df)`** — shape, columns, and dtypes (as a dict)
- **`missing(df)`** — null count per column, most-missing first (a Series)
- **`numeric_columns(df)`** — names of the numeric columns (a list)
- **`categorical_columns(df)`** — names of the object/category columns (a list)

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
