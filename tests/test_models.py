from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier, XGBRegressor

from madrid_congestion.features import feature_names
from madrid_congestion.models import (
    SEARCH_SPACE,
    TRIALS,
    at_edge,
    fit_model,
    score_model,
    tune_xgboost,
    xgboost_candidates,
)

DISTRICTS = pl.Series("distrito", [1, 2])
CHEAP = {
    "max_depth": 2,
    "learning_rate": 0.3,
    "n_estimators": 20,
    "subsample": 1.0,
    "min_child_weight": 1,
}


def synthetic_rows(n: int, seed: int) -> pl.DataFrame:
    # Rows with every model input; the onset depends on the first occupancy lag, the forecast targets are constant.
    rng = np.random.default_rng(seed)
    inputs = {name: rng.normal(size=n) for name in feature_names(DISTRICTS)}
    target = inputs["occupancy_lag0"] + rng.normal(scale=0.5, size=n) > 1
    return pl.DataFrame(inputs).with_columns(
        target_30=pl.Series(target), occupancy_max_30=pl.lit(5.0), flow_max_30=pl.lit(7.0)
    )


def test_candidates_are_reproducible_and_inside_the_search_space():
    candidates = xgboost_candidates()
    assert candidates == xgboost_candidates()  # the same list for every route and ablation variant
    assert len(candidates) == TRIALS
    for params in candidates:
        assert list(params) == list(SEARCH_SPACE)
        for name, (_, low, high) in SEARCH_SPACE.items():
            assert low <= params[name] <= high
    assert at_edge({**candidates[0], "max_depth": 8, "min_child_weight": 1}) == [
        "max_depth",
        "min_child_weight",
    ]


def test_each_route_gets_its_own_model_type_target_and_inputs():
    rows = synthetic_rows(300, seed=0)
    direct = fit_model("xgboost", "direct", 30, rows, DISTRICTS, CHEAP)
    # a classifier of the binary onset target
    assert isinstance(direct, XGBClassifier) and set(direct.classes_) == {0, 1}
    for family, params in (("xgboost", CHEAP), ("linear", None)):
        # the forecasts are regressions on their own continuous targets (5 and 7 here)
        for name, value in (("occupancy", 5.0), ("flow", 7.0)):
            model = fit_model(family, name, 30, rows, DISTRICTS, params)
            assert not hasattr(model, "classes_") and np.allclose(
                score_model(model, name, rows, DISTRICTS), value, atol=0.01
            )
    assert isinstance(fit_model("xgboost", "occupancy", 30, rows, DISTRICTS, CHEAP), XGBRegressor)
    no_flow = fit_model("xgboost", "direct_no_flow", 30, rows, DISTRICTS, CHEAP)
    # the four flow lags are left out
    assert no_flow.n_features_in_ == len(feature_names(DISTRICTS)) - 4


def test_tuning_keeps_the_candidate_with_the_highest_validation_ap():
    fit_rows, validation_rows = synthetic_rows(300, seed=1), synthetic_rows(300, seed=2)
    names = list(SEARCH_SPACE)
    small: list[dict[str, Any]] = [
        dict(zip(names, values, strict=True))
        for values in ((1, 0.05, 3, 1.0, 1), (3, 0.3, 20, 1.0, 1), (2, 0.1, 10, 0.8, 5))
    ]
    model, trials, selected = tune_xgboost(
        "direct", 30, fit_rows, validation_rows, DISTRICTS, small
    )
    assert trials.height == len(small)
    ap = average_precision_score(
        validation_rows["target_30"], score_model(model, "direct", validation_rows, DISTRICTS)
    )
    assert np.isclose(ap, selected["ap_2023"]) and selected["ap_2023"] == trials["ap"].max()
    assert {name: selected[name] for name in names} == small[selected["trial"]]
