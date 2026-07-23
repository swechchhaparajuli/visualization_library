# visualization_library

A data science project structured with [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Project documentation (mkdocs site).
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
│
├── notebooks          <- Jupyter notebooks for exploration only. Naming convention is a
│                         number (for ordering), the creator's initials, and a short
│                         description, e.g. `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for the
│                         src package and configuration for tools like black.
│
├── references         <- Data dictionaries, manuals, and other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures used in reporting.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
│
└── src/visualization_library   <- Source code for use in this project.
    ├── __init__.py             <- Makes src a Python module.
    ├── config.py               <- Store useful variables and configuration.
    ├── dataset.py              <- Scripts to download or generate data.
    ├── features.py             <- Code to create features for modeling.
    ├── plots.py                <- Code to create visualizations.
    └── modeling
        ├── __init__.py
        ├── predict.py          <- Code to run model inference with trained models.
        └── train.py            <- Code to train models.
```
