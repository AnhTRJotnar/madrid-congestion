# madrid-congestion

Advanced project - RUC

Early warning of an operational district congestion proxy in Madrid, from the City Council's open
loop-detector data (2022-2024). The URB data give flow and occupancy but no independent congestion
measure, so the event predicted is the onset of **persistent unusually high district occupancy** (each
district's 2022 95th percentile, at least 30 minutes). Three warnings are compared 30 and 60 minutes
ahead: forecasting flow, forecasting occupancy, and classifying the onset directly, with the same
inputs. Train 2022, tune 2023, test 2024 once; SARIMA is a separate classical benchmark.

## Repository structure

```
├── README.md
├── pyproject.toml, uv.lock     <- Dependencies (uv, Python 3.12)
├── madrid_congestion/          <- Package: data pipeline and shared settings
│   ├── config.py               <- Paths, source file ids, and every data and label rule as a constant
│   ├── dataset.py              <- Commands: download the monthly zips and location files, convert to Parquet
│   ├── districts.py            <- Commands: fixed sensor panel and district series
│   └── plots.py                <- Shared style for the report figures
├── tests/                      <- pytest tests for dataset.py and districts.py
├── notebooks/                  <- Analysis, run in numerical order (see Workflow)
├── reports/figures/            <- Report figures, written by the notebooks
└── data/                       <- Not in git; rebuilt by the commands below
    ├── raw/                    <- 36 monthly zips, 2022-2024
    ├── interim/                <- One Parquet file per month
    ├── external/               <- Sensor location snapshots, December 2021 and December 2022
    └── processed/              <- Sensor panel and district series
```

## Workflow

The years have fixed roles: 2022 trains the models and fixes every rule and threshold, 2023 tunes the
models and selects the warning threshold, and 2024 is the test year, run once after the design freeze.
The package refuses to build a 2024 district series before then.

| Stage | Location | Status |
|---|---|---|
| Download and conversion of the monthly files | `madrid_congestion/dataset.py` | Done |
| Data-quality analysis and data rules, 2022-2023 | `notebooks/1.0-anh-data-quality.ipynb`, rules in `config.py` | Done |
| Fixed sensor panel and district series | `madrid_congestion/districts.py` | Done for 2022-2023 |
| Label construction, calendar baseline and design checks, 2022 | `notebooks/2.0-anh-label-construction.ipynb` | Done |
| Superseded critical-point label (paper Section 3.4) | `notebooks/2.1-anh-superseded-label-check.ipynb` | Done, not part of the pipeline |
| Features, model fitting (2022), tuning and warning-threshold selection (2023) | modelling notebook | Planned |
| Final evaluation on the locked 2024 test year | separate test notebook | Planned, run once |

## Setup and reproduction

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

`build-panel` should report c = 0.66 and 2,865 sensors in 20 districts. Then run the notebooks in
order (1.0, 2.0, 2.1) with the project's kernel. Notebook 1.0 checks that the package's panel and
district series match the rules it derives, so it needs the commands above to have run first.

**Data rules** (decided in notebook 1.0, applied by the package): urban sensors only; clock-change rows
removed; a valid reading has 15 one-minute samples, flow and occupancy present, and flow at most 5,000
vehicles per hour; district values are means over a fixed sensor panel; a district slot is valid if at
least 30 panel sensors and at least half of the panel report; invalid slots are never filled in.

**Label rules** (notebook 2.0): a district is studied if at least half of its 2022 slots are valid; a
high-occupancy slot is a valid slot at or above the district's 2022 95th-percentile occupancy; an onset
starts a run of at least two such slots after a valid slot below the threshold; origins are calm when
the previous hour holds no high-occupancy slot.
