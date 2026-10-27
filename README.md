# madrid-congestion

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Advanced project - RUC

Early warning of an operational district congestion proxy in Madrid, from the City Council's open
loop-detector data (2022-2024). The URB data give flow and occupancy but no independent congestion
measure, so the event predicted is the onset of **persistent unusually high district occupancy** (each
district's 2022 95th percentile, at least 30 minutes). Three warnings are compared 30 and 60 minutes
ahead: forecasting flow, forecasting occupancy, and classifying the onset directly, with the same
inputs. Train 2022, tune 2023, test 2024 once; SARIMA is a separate classical benchmark.

## Status (2026-10-26)

- Done: download and conversion of the 36 monthly files (`madrid_congestion/dataset.py`, with tests).
- Done: exploratory analysis and data rules in `notebooks/1.0-anh-initial-exploration.ipynb` (2022-2023
  only; 2024 is the locked test year). The rules are constants in `madrid_congestion/config.py`.
- Done: fixed sensor panel and district series for 2022-2023 (`madrid_congestion/districts.py`, with
  tests).
- Done: 2022 labels, calendar baseline and design checks in `notebooks/2.0-anh-labels.ipynb` (19
  districts, 5,466 onsets; label rules in `config.py`). `notebooks/archive/` holds the superseded critical-point attempt.
- Next: `labels.py` and `features.py` in the package, then models.

## Reproduce (so far)

```powershell
git clone https://github.com/AnhTRJotnar/madrid-congestion.git
cd madrid-congestion
uv sync
uv run python -m madrid_congestion.dataset download            # 36 monthly zips, about 2.7 GB, into data/raw/
uv run python -m madrid_congestion.dataset convert             # one Parquet file per month, into data/interim/
uv run python -m madrid_congestion.dataset download-locations  # two sensor-location files, into data/external/
uv run python -m madrid_congestion.districts build-panel       # fixed sensor panel chosen on 2022, into data/processed/
uv run python -m madrid_congestion.districts build-district-series  # district series 2022-2023, into data/processed/
uv run pytest
```

The data are not in git (`/data/` is ignored); these commands rebuild them from datos.madrid.es.
`build-panel` should report c = 0.66 and 2,865 sensors in 20 districts.

**Data rules** (decided in the notebook, applied by the package): urban sensors only; clock-change rows
removed; a valid reading has 15 one-minute samples, flow and occupancy present, and flow at most 5,000
vehicles per hour; district values are means over a fixed sensor panel; a district slot is valid if at
least 30 panel sensors and at least half of the panel report; invalid slots are never filled in.

**Label rules** (notebook 2.0): a district is studied if at least half of its 2022 slots are valid; a
high-occupancy slot is a valid slot at or above the district's 2022 95th-percentile occupancy; an onset
starts a run of at least two such slots after a valid slot below the threshold; origins are calm when
the previous hour holds no high-occupancy slot.

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
    ├── districts.py            <- Fixed sensor panel and district series
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
