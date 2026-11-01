from collections.abc import Mapping, Sequence
import warnings

from joblib import Parallel, delayed
import numpy as np
from numpy.typing import ArrayLike
import polars as pl
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace import kalman_filter as kf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from madrid_congestion.config import HORIZONS
from madrid_congestion.features import SLOTS_PER_DAY

# The classical benchmark, one model per district and variable: SARIMA(1,0,1)(0,1,1) with a daily season of
# 96 slots. A pragmatic fixed order, set before validation and not tuned (paper Section 3.7).
VARIABLES = ("occupancy", "flow")  # one model per district and relative variable
ORDER = (1, 0, 1)
SEASONAL_ORDER = (0, 1, 1, SLOTS_PER_DAY)
# What the forecast filter keeps: the predicted states only. The state covariances would need about 10 GB per
# series-year (194 x 194 per slot) and the forecasts do not use them.
KEEP_PREDICTED_STATE = (
    kf.MEMORY_NO_FORECAST
    | kf.MEMORY_NO_FILTERED
    | kf.MEMORY_NO_PREDICTED_COV
    | kf.MEMORY_NO_GAIN
    | kf.MEMORY_NO_SMOOTHING
    | kf.MEMORY_NO_STD_FORECAST
)


def sarima(values: ArrayLike, simple_differencing: bool = False) -> SARIMAX:
    # Invalid slots are NaN: the Kalman filter skips them, nothing is filled in. No constant (the default).
    return SARIMAX(
        np.asarray(values, dtype=float),
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        simple_differencing=simple_differencing,
    )


def fit_sarima(values: ArrayLike) -> dict[str, float]:
    # Maximum likelihood on the fitting year. The likelihood is computed on the seasonally differenced series
    # (simple differencing: 98 states instead of 194, about 8 times faster); the model and its parameters are
    # the same, and a slot is skipped when it or the slot one day earlier is invalid.
    model = sarima(values, simple_differencing=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        params = model.fit(disp=False, return_params=True)
    return dict(zip(model.param_names, np.asarray(params, dtype=float).tolist(), strict=True))


def forecast_paths(values: ArrayLike, params: dict[str, float], steps: int) -> np.ndarray:
    # Row t: the forecasts of slots t+1 ... t+steps from the observations up to and including t, with the
    # parameters fixed. One Kalman filter pass gives the predicted state of t+1 for every t; a forecast h steps
    # ahead applies the transition h-1 more times and reads the series value from the state.
    model = sarima(values)
    ssm = model.ssm
    model.update(np.array([params[name] for name in model.param_names]))
    result = ssm.filter(conserve_memory=KEEP_PREDICTED_STATE)
    # 1 x states; the model is time-invariant, so one matrix serves every slot
    design = ssm["design"]
    transition = ssm["transition"]
    rows = [design]
    for _ in range(steps - 1):
        rows.append(rows[-1] @ transition)
    # steps x states: row h-1 reads y(t+h) from the predicted state of t+1
    readout = np.vstack(rows)
    return (readout @ result.predicted_state[:, 1:]).T


def fit_and_forecast(
    fit_values: ArrayLike, all_values: ArrayLike, steps: int
) -> tuple[dict[str, float], np.ndarray]:
    # One district and variable: parameters from the fitting year, then forecast paths over every slot of
    # all_values (the fitting year followed by the later years), with the parameters fixed.
    params = fit_sarima(fit_values)
    return params, forecast_paths(all_values, params, steps)


def fit_and_forecast_all(
    values: Mapping[tuple[int, str], np.ndarray], fit_slots: int, steps: int, n_jobs: int = 8
) -> dict[tuple[int, str], tuple[dict[str, float], np.ndarray]]:
    # Every district and variable: fitted on its first fit_slots values (the fitting year), then forecast over all
    # of them. The models are independent, so they run in parallel processes.
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_and_forecast)(series[:fit_slots], series, steps) for series in values.values()
    )
    fitted = {}
    for key, result in zip(values, results, strict=True):
        assert result is not None, key
        params, paths = result
        fitted[key] = (params, np.asarray(paths))
    return fitted


def sarima_grid(labels: Sequence[pl.DataFrame]) -> pl.DataFrame:
    # The labels of consecutive years, from 2022 on, as one continuous 15-minute grid per district: the relative
    # occupancy and flow of valid slots, NaN for invalid ones (the Kalman filter skips them).
    grid = (
        pl.concat(labels)
        .select(
            "distrito",
            "slot_start",
            *[
                pl.when(pl.col("valid_slot")).then(pl.col(f"relative_{variable}")).alias(variable)
                for variable in VARIABLES
            ],
        )
        .sort("distrito", "slot_start")
        .with_columns(pl.col(VARIABLES).fill_null(np.nan))
    )
    gaps = grid.group_by("distrito").agg(steps=pl.col("slot_start").diff().drop_nulls().n_unique())
    assert (gaps["steps"] == 1).all(), "the years are not consecutive"
    return grid


def series_values(grid: pl.DataFrame, district: int, variable: str) -> np.ndarray:
    return grid.filter(pl.col("distrito") == district)[variable].to_numpy()


def sarima_scores(
    grid: pl.DataFrame,
    paths: Mapping[tuple[int, str], np.ndarray],
    rows: pl.DataFrame,
    minutes: int,
) -> pl.DataFrame:
    # The SARIMA warning scores of each usable calm origin, in row order: the largest forecast relative value in
    # t+1 ... t+H, from the forecast paths of every (district, variable) over the grid.
    H = HORIZONS[minutes]
    maxima = pl.concat(
        [
            grid.filter(pl.col("distrito") == district)
            .select("distrito", "slot_start")
            .with_columns(
                **{
                    f"sarima_{variable}": pl.Series(paths[district, variable][:, :H].max(axis=1))
                    for variable in VARIABLES
                }
            )
            for district in grid["distrito"].unique(maintain_order=True)
        ]
    )
    columns = [f"sarima_{variable}" for variable in VARIABLES]
    scores = rows.select("distrito", "slot_start").join(
        maxima, on=["distrito", "slot_start"], how="left", maintain_order="left"
    )
    missing = scores.select(pl.col(columns).fill_nan(None).null_count()).sum_horizontal().item()
    assert missing == 0, "an origin has no SARIMA forecast"
    return scores.select(columns)
