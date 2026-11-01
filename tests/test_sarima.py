from datetime import date

import numpy as np
import polars as pl
import pytest

from madrid_congestion.sarima import forecast_paths, sarima, sarima_scores

PARAMS = {"ar.L1": 0.6, "ma.L1": 0.2, "ma.S.L96": -0.7, "sigma2": 0.003}


@pytest.mark.filterwarnings("ignore")
def test_forecast_paths_match_statsmodels_forecasts_after_gaps():
    # eight days of a daily cycle with noise, an invalid stretch and one invalid slot
    rng = np.random.default_rng(0)
    slots = np.arange(96 * 8)
    values = 1 + 0.3 * np.sin(2 * np.pi * slots / 96) + 0.05 * rng.normal(size=slots.size)
    values[200:230] = np.nan
    values[500] = np.nan
    paths = forecast_paths(values, PARAMS, 4)
    for origin in (150, 199, 231, 499, 763):  # before, inside, after the gaps, near the end
        own = sarima(values[: origin + 1]).filter(np.array(list(PARAMS.values())))
        assert np.allclose(paths[origin], own.forecast(4), atol=1e-12)


def test_sarima_scores_take_the_window_maximum_of_each_origin():
    # Two districts, four slots each; the path of slot t holds the forecasts of t+1 ... t+4
    slots = pl.datetime_range(
        date(2022, 3, 1), date(2022, 3, 2), "15m", closed="left", eager=True
    )[:4]
    grid = pl.DataFrame({"distrito": [1] * 4 + [2] * 4, "slot_start": list(slots) * 2})
    path = np.array([[1.0, 3.0, 2.0, 9.0]] * 4)
    paths = {
        (1, "occupancy"): path,
        (1, "flow"): path / 10,
        (2, "occupancy"): path * 2,
        (2, "flow"): path,
    }
    rows = pl.DataFrame(
        {"distrito": [2, 1], "slot_start": [slots[1], slots[0]]}
    )  # scored origins, any order
    scores = sarima_scores(grid, paths, rows, 30)  # 30 minutes: the first two steps
    assert scores["sarima_occupancy"].to_list() == [6.0, 3.0]
    assert scores["sarima_flow"].to_list() == [3.0, 0.3]
    assert sarima_scores(grid, paths, rows, 60)["sarima_occupancy"].to_list() == [18.0, 9.0]
