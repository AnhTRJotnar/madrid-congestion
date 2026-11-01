import math
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from madrid_congestion.features import feature_names

TRIALS = 30  # the tuning budget of every XGBoost model (paper Section 3.7)
SEED = 1  # draws the candidates and seeds every XGBoost fit, whose row subsampling is random
# The XGBoost search space: setting -> (how it is drawn, lowest, highest), both ends included. The order is the
# order of the draws, so changing it would change the candidates.
SEARCH_SPACE: dict[str, tuple[str, float, float]] = {
    "max_depth": ("integer", 2, 8),
    "learning_rate": ("log-uniform", 0.02, 0.3),
    "n_estimators": ("integer", 100, 800),
    "subsample": ("uniform", 0.6, 1.0),  # share of rows each tree sees
    "min_child_weight": ("integer", 1, 20),
}

# What each warning route is fitted to (column name + "_<minutes>"): the largest relative occupancy or flow in
# t+1 ... t+H for the two forecasts (regression), the binary onset target for the direct classification.
ROUTE_TARGETS = {"occupancy": "occupancy_max", "flow": "flow_max", "direct": "target"}
# Every model of the study: name -> (route, whether the four flow lags are inputs). direct_no_flow is variant A of
# the flow ablation (XGBoost only); variant B is the full direct model.
MODELS = {
    "occupancy": ("occupancy", True),
    "flow": ("flow", True),
    "direct": ("direct", True),
    "direct_no_flow": ("direct", False),
}
LINEAR_MODELS = ["occupancy", "flow", "direct"]
XGBOOST_MODELS = list(MODELS)

Model = XGBClassifier | XGBRegressor | Pipeline


def fit_target(route: str, minutes: int) -> str:
    return f"{ROUTE_TARGETS[route]}_{minutes}"


def is_classifier(model_name: str) -> bool:
    return MODELS[model_name][0] == "direct"


def model_inputs(model_name: str, districts: pl.Series) -> list[str]:
    # The 34 inputs of paper Table 3.4, or 30 without the flow lags for the ablation variant.
    return feature_names(districts, flow=MODELS[model_name][1])


def linear_model(route: str) -> Pipeline:
    # Untuned linear models; the inputs are standardised inside the pipeline, so the scaling is learned from 2022.
    # L2 is scikit-learn's default penalty; 1,000 iterations leave the solver room to converge on 475,000 rows.
    if route == "direct":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    return make_pipeline(StandardScaler(), LinearRegression())


def xgboost_candidates(trials: int = TRIALS, seed: int = SEED) -> list[dict[str, Any]]:
    # The random-search candidates, drawn once from SEARCH_SPACE. Every route, horizon and both flow-ablation
    # variants receive this same list, so no model gets a larger or luckier budget.
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(trials):
        params: dict[str, Any] = {}
        for name, (kind, low, high) in SEARCH_SPACE.items():
            if kind == "integer":  # integers() leaves out its upper bound, hence high + 1
                params[name] = int(rng.integers(int(low), int(high) + 1))
            elif kind == "log-uniform":
                params[name] = math.exp(rng.uniform(math.log(low), math.log(high)))
            else:
                params[name] = float(rng.uniform(low, high))
        candidates.append(params)
    return candidates


def at_edge(params: dict[str, Any]) -> list[str]:
    # The integer settings that sit on an end of the search space (continuous ones cannot sit exactly on one).
    return [
        name
        for name, (kind, low, high) in SEARCH_SPACE.items()
        if kind == "integer" and params[name] in (low, high)
    ]


def xgboost_model(classifier: bool, params: dict[str, Any]) -> XGBClassifier | XGBRegressor:
    # Direct classification: binary log loss. Forecasts: squared error. No rebalancing of the rare positives.
    # "hist" is XGBoost's histogram method, its fast default for large tables; n_jobs=-1 uses every core.
    common = {"tree_method": "hist", "random_state": SEED, "n_jobs": -1, **params}
    if classifier:
        return XGBClassifier(objective="binary:logistic", **common)
    return XGBRegressor(objective="reg:squarederror", **common)


def new_model(family: str, model_name: str, params: dict[str, Any] | None = None) -> Model:
    # An unfitted model: "linear" (no settings) or "xgboost" (one candidate's settings).
    if family == "linear":
        assert params is None and model_name in LINEAR_MODELS, (family, model_name)
        return linear_model(MODELS[model_name][0])
    assert family == "xgboost" and params is not None, (family, model_name)
    return xgboost_model(is_classifier(model_name), params)


def model_data(
    model_name: str, minutes: int, rows: pl.DataFrame, districts: pl.Series
) -> tuple[np.ndarray, np.ndarray]:
    # The model's inputs and what it is fitted to, for every row (usable calm origin).
    route = MODELS[model_name][0]
    return (
        rows.select(model_inputs(model_name, districts)).to_numpy(),
        rows[fit_target(route, minutes)].to_numpy(),
    )


def fit_model(
    family: str,
    model_name: str,
    minutes: int,
    rows: pl.DataFrame,
    districts: pl.Series,
    params: dict[str, Any] | None = None,
) -> Model:
    # Fit one model on the given rows (the 2022 usable calm origins at this horizon).
    model = new_model(family, model_name, params)
    model.fit(*model_data(model_name, minutes, rows, districts))
    return model


def warning_score(model: Model, X: np.ndarray) -> np.ndarray:
    # The classifier's probability of an onset, or the forecast largest relative value; higher = stronger warning.
    # (a linear forecast is a pipeline without predict_proba)
    if isinstance(model, XGBRegressor) or not hasattr(model, "predict_proba"):
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


def score_model(
    model: Model, model_name: str, rows: pl.DataFrame, districts: pl.Series
) -> np.ndarray:
    # The warning score of every row, from the model's own inputs.
    return warning_score(model, rows.select(model_inputs(model_name, districts)).to_numpy())


def tune_xgboost(
    model_name: str,
    minutes: int,
    fit_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    districts: pl.Series,
    candidates: list[dict[str, Any]],
) -> tuple[Model, pl.DataFrame, dict[str, Any]]:
    # Fit every candidate on the fitting-year rows and keep the one whose warning score has the highest AP on the
    # validation-year rows (the one selection rule; a tie keeps the earlier trial). A forecast is judged by the
    # warning it gives, not by its own error. Returns the kept model, every trial, and the kept trial's record.
    X_fit, y_fit = model_data(model_name, minutes, fit_rows, districts)
    X_validation = validation_rows.select(model_inputs(model_name, districts)).to_numpy()
    target = validation_rows[f"target_{minutes}"].to_numpy()
    best, selected, trials = None, {"ap_2023": -math.inf}, []
    for trial, params in enumerate(candidates):
        model = new_model("xgboost", model_name, params)
        model.fit(X_fit, y_fit)
        ap = float(average_precision_score(target, warning_score(model, X_validation)))
        trials.append({"trial": trial, **params, "ap": ap})
        if ap > selected["ap_2023"]:
            best, selected = model, {"trial": trial, **params, "ap_2023": ap}
    assert best is not None, "no candidates"
    return best, pl.DataFrame(trials), selected
