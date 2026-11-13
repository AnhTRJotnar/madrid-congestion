import json
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from madrid_congestion import bootstrap
from madrid_congestion.config import (
    CALM_SLOTS,
    DISTRICT_MINIMUM,
    HIGH_OCC_QUANTILE,
    HORIZONS,
    MAX_FLOW,
    MIN_EPISODE_SLOTS,
    MIN_SAMPLES,
    MODELS_DIR,
    PANEL_YEAR,
    SUPPORT_SHARE,
    TEST_YEAR,
    VALID_SLOT_PANEL_SHARE,
    VALIDATION_YEAR,
    WEEKEND_FROM,
)
from madrid_congestion.evaluation import best_csi_threshold
from madrid_congestion.features import feature_names
from madrid_congestion.labels import (
    add_labels,
    baseline_scores,
    calendar_baseline,
    references,
    studied_districts,
    year_grid,
)
from madrid_congestion.models import (
    LINEAR_MODELS,
    SEARCH_SPACE,
    SEED,
    TRIALS,
    XGBOOST_MODELS,
    Model,
    fit_model,
    score_model,
    xgboost_candidates,
)
from madrid_congestion.sarima import (
    ORDER,
    SEASONAL_ORDER,
    VARIABLES,
    forecast_paths,
    sarima_scores,
    series_values,
)

# The frozen design (29 October 2026): what the 2022/2023 notebooks chose, saved in models/, and the checks that
# the final test run passes before it may read 2024.
FREEZE_DATE = "2026-10-29"
SCORES = [
    "proximity",
    "calendar",
    *[f"linear_{name}" for name in LINEAR_MODELS],
    *[f"xgboost_{name}" for name in XGBOOST_MODELS],
    *[f"sarima_{variable}" for variable in VARIABLES],
]
# The comparisons of paper Section 5.1: difference = AP(first) - AP(second)
CONTRASTS = {
    "primary: direct - occupancy": ("xgboost_direct", "xgboost_occupancy"),
    "secondary: occupancy - flow": ("xgboost_occupancy", "xgboost_flow"),
    "secondary: with flow - without flow": ("xgboost_direct", "xgboost_direct_no_flow"),
}
FILES = {
    "design": "design.json",  # notebook 3.0
    "references": "references-2022.parquet",  # notebook 3.0
    "calendar": "calendar-2022.parquet",  # notebook 3.0
    "xgboost": "xgboost-selected.json",  # notebook 3.1
    "ablation": "xgboost-ablation.json",  # notebook 3.2
    "sarima": "sarima-2022.json",  # notebook 3.3
    "thresholds": "thresholds-2023.json",  # notebook 3.4
}
REPLAY_TOLERANCE = 1e-9  # largest absolute difference from a frozen 2023 AP or warning threshold


class FrozenDesignError(Exception):
    pass


def save_frozen(name: str, value: Any) -> None:
    # A frozen 2022/2023 choice, written once by its notebook and only read by the final test run.
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / name).write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")


def load_frozen(name: str) -> Any:
    return json.loads((MODELS_DIR / name).read_text(encoding="utf-8"))


def design_manifest(districts: pl.Series) -> dict[str, Any]:
    # Everything that identifies the frozen experiment, built from the code's own constants.
    manifest = {
        "frozen": FREEZE_DATE,
        "years": {"fit": PANEL_YEAR, "validation": VALIDATION_YEAR, "test": TEST_YEAR},
        "label": {
            "high_occupancy_quantile": HIGH_OCC_QUANTILE,  # theta and the flow reference, per district, 2022
            "min_episode_slots": MIN_EPISODE_SLOTS,
            "calm_slots": CALM_SLOTS,  # no high slot in t-3 ... t, all of t-3 ... t+H+1 valid
            "support_share": SUPPORT_SHARE,  # studied district: this share of its 2022 slots valid
            "valid_slot": {
                "min_samples": MIN_SAMPLES,
                "max_flow": MAX_FLOW,
                "district_minimum": DISTRICT_MINIMUM,
                "panel_share": VALID_SLOT_PANEL_SHARE,
            },
            "weekend_from": WEEKEND_FROM,
        },
        "horizons": {str(minutes): slots for minutes, slots in HORIZONS.items()},
        "districts": districts.to_list(),
        "features": feature_names(districts),
        "features_without_flow": feature_names(districts, flow=False),
        "models": {"linear": LINEAR_MODELS, "xgboost": XGBOOST_MODELS},
        "seed": SEED,
        "trials": TRIALS,
        "search_space": SEARCH_SPACE,
        "candidates": xgboost_candidates(),
        "sarima": {"order": ORDER, "seasonal_order": SEASONAL_ORDER, "variables": VARIABLES},
        "scores": SCORES,
        "contrasts": CONTRASTS,
        "bootstrap": {
            "weeks": bootstrap.WEEKS,
            "repetitions": bootstrap.REPETITIONS,
            "large_repetitions": bootstrap.LARGE_REPETITIONS,
            "seeds": bootstrap.SEEDS,
            "move_limit": bootstrap.MOVE_LIMIT,
            "level": bootstrap.LEVEL,
        },
    }
    return json.loads(json.dumps(manifest))  # tuples as lists, exactly as the file holds them


def write_design(districts: pl.Series) -> None:
    save_frozen(FILES["design"], design_manifest(districts))


def check_design(districts: pl.Series) -> None:
    # The saved manifest must equal the one the current code builds: any drift after the freeze fails here.
    if not (MODELS_DIR / FILES["design"]).exists():
        raise FrozenDesignError(f"missing {FILES['design']}")
    saved, current = load_frozen(FILES["design"]), design_manifest(districts)
    differing = sorted(
        key for key in saved.keys() | current.keys() if saved.get(key) != current.get(key)
    )
    if differing:
        raise FrozenDesignError(f"the code no longer matches {FILES['design']}: {differing}")


def check_frozen(districts: pl.Series | None = None) -> None:
    # Every frozen file present, complete and consistent with the manifest; raises with the list of problems.
    # Without districts, the studied districts are those of the frozen references.
    missing = [name for name in FILES.values() if not (MODELS_DIR / name).exists()]
    if missing:
        raise FrozenDesignError(f"missing frozen files: {missing}")
    refs = pl.read_parquet(MODELS_DIR / FILES["references"])
    if districts is None:
        districts = refs["distrito"]
    check_design(districts)
    problems = []
    if (
        refs["distrito"].to_list() != districts.to_list()
        or refs.null_count().sum_horizontal().item()
    ):
        problems.append("references do not cover the studied districts")
    calendar = pl.read_parquet(MODELS_DIR / FILES["calendar"])
    cells = (
        len(districts) * 2 * 96 * len(HORIZONS)
    )  # district x weekday/weekend x slot of day x horizon
    if calendar.height != cells or calendar["calendar_rate"].null_count():
        problems.append("calendar shares incomplete")
    candidates = xgboost_candidates()
    selected = load_frozen(FILES["xgboost"])
    ablation = load_frozen(FILES["ablation"])
    for minutes in map(str, HORIZONS):
        for name in XGBOOST_MODELS:
            record = (
                (ablation if name == "direct_no_flow" else selected).get(minutes, {}).get(name)
            )
            if record is None:
                problems.append(f"no XGBoost settings for {name}, {minutes} min")
            elif {key: record[key] for key in SEARCH_SPACE} != candidates[record["trial"]]:
                problems.append(
                    f"XGBoost settings of {name}, {minutes} min are not candidate {record['trial']}"
                )
    sarima = load_frozen(FILES["sarima"])
    for variable in VARIABLES:
        if sorted(map(int, sarima.get(variable, {}))) != districts.to_list():
            problems.append(f"SARIMA parameters missing for {variable}")
    thresholds = load_frozen(FILES["thresholds"])
    for minutes in map(str, HORIZONS):
        if sorted(thresholds.get(minutes, {})) != sorted(SCORES):
            problems.append(f"warning thresholds incomplete at {minutes} min")
    if problems:
        raise FrozenDesignError("; ".join(problems))


def check_references(series: pl.DataFrame) -> None:
    # The 2022 choices the test run loads, rebuilt from the 2022 district series by the package's own rules: studied
    # districts, thresholds, flow references and calendar shares must equal the frozen files exactly.
    districts = studied_districts(series)
    grid = year_grid(series, districts, PANEL_YEAR)
    refs = references(grid)
    calendar = calendar_baseline(add_labels(grid, refs), districts)
    problems = []
    if not refs.equals(pl.read_parquet(MODELS_DIR / FILES["references"])):
        problems.append(f"{FILES['references']} differs from the references rebuilt from 2022")
    if not calendar.equals(pl.read_parquet(MODELS_DIR / FILES["calendar"])):
        problems.append(f"{FILES['calendar']} differs from the calendar shares rebuilt from 2022")
    if problems:
        raise FrozenDesignError("; ".join(problems))


def frozen_models(
    fit_rows: dict[int, pl.DataFrame], districts: pl.Series
) -> dict[tuple[int, str], Model]:
    # Every linear and XGBoost model refitted on the fitting-year rows from the frozen settings; nothing is chosen.
    settings = load_frozen(FILES["xgboost"])
    for minutes, record in load_frozen(FILES["ablation"]).items():
        settings[minutes] |= record
    models = {}
    for minutes, rows in fit_rows.items():
        for name in LINEAR_MODELS:
            models[minutes, f"linear_{name}"] = fit_model("linear", name, minutes, rows, districts)
        for name in XGBOOST_MODELS:
            params = {key: settings[str(minutes)][name][key] for key in SEARCH_SPACE}
            models[minutes, f"xgboost_{name}"] = fit_model(
                "xgboost", name, minutes, rows, districts, params
            )
    return models


def frozen_sarima_paths(grid: pl.DataFrame, n_jobs: int = 8) -> dict[tuple[int, str], np.ndarray]:
    # Forecast paths of every district and variable over the grid, with the frozen 2022 parameters.
    params = load_frozen(FILES["sarima"])
    keys = [(int(district), variable) for variable in VARIABLES for district in params[variable]]
    steps = max(HORIZONS.values())
    paths = Parallel(n_jobs=n_jobs)(
        delayed(forecast_paths)(
            series_values(grid, district, variable), params[variable][str(district)], steps
        )
        for district, variable in keys
    )
    return {key: np.asarray(path) for key, path in zip(keys, paths, strict=True)}


def score_year(
    rows: pl.DataFrame,
    minutes: int,
    districts: pl.Series,
    models: dict[tuple[int, str], Model],
    calendar: pl.DataFrame,
    grid: pl.DataFrame,
    paths: dict[tuple[int, str], np.ndarray],
) -> pl.DataFrame:
    # Every warning score of the study for one year's usable calm origins at one horizon, by the same functions
    # for the 2023 replay and the 2024 test. A model's score name is its family and model name, e.g. xgboost_direct.
    model_scores = {
        name: score_model(models[minutes, name], name.split("_", 1)[1], rows, districts)
        for name in SCORES
        if (minutes, name) in models
    }
    return pl.concat(
        [
            rows.select("distrito", "slot_start", target=pl.col(f"target_{minutes}")),
            baseline_scores(rows, calendar, minutes),
            pl.DataFrame(model_scores),
            sarima_scores(grid, paths, rows, minutes),
        ],
        how="horizontal",
    ).select("distrito", "slot_start", "target", *SCORES)


def replay_validation(scores_by_horizon: dict[int, pl.DataFrame]) -> pl.DataFrame:
    # The 2023 AP and highest-CSI warning threshold of every score, recomputed by the final-test path, against
    # those frozen by notebook 3.4. Raises unless every score reproduces both; only then may the 2024 stage go on.
    thresholds = load_frozen(FILES["thresholds"])
    rows = []
    for minutes, scores in scores_by_horizon.items():
        for name in SCORES:
            frozen = thresholds[str(minutes)][name]
            rows.append(
                {
                    "minutes": minutes,
                    "score": name,
                    "ap_replay": float(average_precision_score(scores["target"], scores[name])),
                    "ap_frozen": frozen["ap_2023"],
                    "threshold_replay": best_csi_threshold(scores["target"], scores[name]),
                    "threshold_frozen": frozen["warning_threshold"],
                }
            )
    table = pl.DataFrame(rows).with_columns(
        difference=(pl.col("ap_replay") - pl.col("ap_frozen")).abs(),
        threshold_difference=(pl.col("threshold_replay") - pl.col("threshold_frozen")).abs(),
    )
    failed = table.filter(
        (pl.col("difference") > REPLAY_TOLERANCE)
        | (pl.col("threshold_difference") > REPLAY_TOLERANCE)
    )
    if failed.height:
        raise FrozenDesignError(
            f"the 2023 replay does not reproduce the frozen AP or warning threshold:\n{failed}"
        )
    return table


def compare(scores_by_horizon: dict[int, pl.DataFrame], year: int) -> pl.DataFrame:
    # The comparisons of paper Section 5.1 at each horizon: the AP difference on the 52 complete weeks of `year`
    # with its paired week-block bootstrap interval (Section 5.3). The same function serves the 2023 debugging run
    # and the 2024 test, so the test year meets no untried code.
    rows = []
    for minutes, scores in scores_by_horizon.items():
        assert (scores["slot_start"].dt.year() == year).all(), "origins from another year"
        assert scores["target"].dtype == pl.Boolean, "the target must be the binary onset label"
        # a null or NaN score would be ranked last without any error, so every compared score must be finite
        for name in sorted({name for pair in CONTRASTS.values() for name in pair}):
            if not np.isfinite(scores[name].to_numpy()).all():
                raise ValueError(f"{name} has a missing or non-finite score at {minutes} min")
        block = bootstrap.week_block(scores["slot_start"], year)
        for name, (first, second) in CONTRASTS.items():
            rows.append(
                {
                    "minutes": minutes,
                    "comparison": name,
                    **bootstrap.bootstrap_difference(
                        scores["target"], scores[first], scores[second], block
                    ),
                }
            )
    return pl.DataFrame(rows)
