# madrid-congestion

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Advanced project - RUC

Early warning of district congestion onset in Madrid, from the City Council's open loop-detector data
(2022-2024).

## Status (2026-10-20)

- Done: download and conversion of the 36 monthly files (`madrid_congestion/dataset.py`, with tests).
- In progress: exploratory analysis in `notebooks/1.0-anh-initial-exploration.ipynb` (2022-2023 only;
  2024 is the locked test year).

## Reproduce (so far)

```powershell
git clone https://github.com/AnhTRJotnar/madrid-congestion.git
cd madrid-congestion
uv sync
uv run python -m madrid_congestion.dataset download     # 36 monthly zips, about 2.7 GB, into data/raw/
uv run python -m madrid_congestion.dataset convert      # one Parquet file per month, into data/interim/
uv run pytest
```

The data are not in git (`/data/` is ignored); these commands rebuild them from datos.madrid.es.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         madrid_congestion and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── madrid_congestion   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes madrid_congestion a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
