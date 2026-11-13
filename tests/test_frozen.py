from datetime import date
import json
import re

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from madrid_congestion import bootstrap, frozen
from madrid_congestion.config import HORIZONS, PANEL_YEAR
from madrid_congestion.evaluation import best_csi_threshold
from madrid_congestion.frozen import (
    CONTRASTS,
    FILES,
    SCORES,
    FrozenDesignError,
    check_frozen,
    check_references,
    compare,
    replay_validation,
    write_design,
)
from madrid_congestion.labels import add_labels, calendar_baseline, references, year_grid
from madrid_congestion.models import SEARCH_SPACE, xgboost_candidates

DISTRICTS = pl.Series("distrito", [1, 2])


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    # A complete, consistent set of frozen files for two districts, in a temporary models folder
    monkeypatch.setattr(frozen, "MODELS_DIR", tmp_path)
    write_design(DISTRICTS)
    pl.DataFrame(
        {"distrito": [1, 2], "threshold": [10.0, 12.0], "flow_reference": [500.0, 600.0]}
    ).write_parquet(tmp_path / FILES["references"])
    cells = pl.DataFrame({"distrito": [1, 2]}).join(
        pl.DataFrame({"weekend": [False, True]}), how="cross"
    )
    cells = cells.join(pl.DataFrame({"slot_of_day": range(96)}), how="cross")
    cells = cells.join(pl.DataFrame({"horizon": list(HORIZONS)}), how="cross")
    cells.with_columns(calendar_rate=pl.lit(0.01)).write_parquet(tmp_path / FILES["calendar"])
    record = {"trial": 0, **xgboost_candidates()[0], "ap_2023": 0.5}
    write(
        tmp_path,
        "xgboost",
        {str(m): {name: record for name in ("occupancy", "flow", "direct")} for m in HORIZONS},
    )
    write(tmp_path, "ablation", {str(m): {"direct_no_flow": record} for m in HORIZONS})
    params = {"ar.L1": 0.5, "ma.L1": 0.1, "ma.S.L96": -0.8, "sigma2": 0.01}
    write(
        tmp_path,
        "sarima",
        {variable: {"1": params, "2": params} for variable in ("occupancy", "flow")},
    )
    entry = {"warning_threshold": 0.3, "ap_2023": 0.5}
    write(tmp_path, "thresholds", {str(m): {name: entry for name in SCORES} for m in HORIZONS})
    return tmp_path


def write(folder, key, value):
    (folder / FILES[key]).write_text(json.dumps(value), encoding="utf-8")


def edit(folder, key, change):
    value = json.loads((folder / FILES[key]).read_text(encoding="utf-8"))
    change(value)
    write(folder, key, value)


def test_a_complete_frozen_design_passes(models_dir):
    check_frozen(DISTRICTS)
    check_frozen()  # districts taken from the frozen references


@pytest.mark.parametrize(
    ("key", "change", "message"),
    [
        ("thresholds", lambda value: value.pop("60"), "thresholds incomplete at 60"),
        ("thresholds", lambda value: value["30"].pop("calendar"), "thresholds incomplete at 30"),
        ("xgboost", lambda value: value["30"].pop("flow"), "no XGBoost settings for flow"),
        ("ablation", lambda value: value["60"].clear(), "no XGBoost settings for direct_no_flow"),
        ("xgboost", lambda value: value["30"]["direct"].update(max_depth=9), "not candidate 0"),
        ("design", lambda value: value.update(seed=2), "['seed']"),
        (
            "design",
            lambda value: value["search_space"]["max_depth"].__setitem__(2, 10),
            "['search_space']",
        ),
        ("design", lambda value: value["candidates"].pop(), "['candidates']"),
        ("design", lambda value: value["label"].update(min_episode_slots=3), "['label']"),
        ("design", lambda value: value["features"].pop(), "['features']"),
    ],
)
def test_drift_after_the_freeze_is_rejected(models_dir, key, change, message):
    edit(models_dir, key, change)
    with pytest.raises(FrozenDesignError, match=re.escape(message)):
        check_frozen(DISTRICTS)


def test_other_districts_or_a_missing_file_are_rejected(models_dir):
    with pytest.raises(FrozenDesignError, match="districts"):
        check_frozen(pl.Series("distrito", [1, 3]))
    (models_dir / FILES["sarima"]).unlink()
    with pytest.raises(FrozenDesignError, match="missing frozen files"):
        check_frozen(DISTRICTS)


def test_the_manifest_holds_the_search_space_and_its_candidates(models_dir):
    design = json.loads((models_dir / FILES["design"]).read_text(encoding="utf-8"))
    assert design["seed"] == 1 and design["trials"] == 30 and len(design["candidates"]) == 30
    assert list(design["search_space"]) == list(SEARCH_SPACE)


def scores_with_frozen_values(seed):
    # Random scores per horizon, and the AP and highest-CSI threshold each one has, as notebook 3.4 would freeze them
    rng = np.random.default_rng(seed)
    scores, values = {}, {}
    for minutes in HORIZONS:
        target = rng.random(300) < 0.2
        scores[minutes] = pl.DataFrame(
            {"target": target, **{name: rng.random(300) + 0.3 * target for name in SCORES}}
        )
        values[str(minutes)] = {
            name: {
                "warning_threshold": best_csi_threshold(target, scores[minutes][name]),
                "ap_2023": float(average_precision_score(target, scores[minutes][name])),
            }
            for name in SCORES
        }
    return scores, values


def test_the_replay_passes_only_when_every_ap_and_threshold_reproduces(models_dir):
    scores, values = scores_with_frozen_values(0)
    write(models_dir, "thresholds", values)
    assert replay_validation(scores).height == len(HORIZONS) * len(SCORES)
    for field, change in (("ap_2023", 1e-6), ("warning_threshold", 1e-3)):
        drifted = json.loads(
            json.dumps(values)
        )  # one frozen value off, everything else as computed
        drifted["60"]["calendar"][field] += change
        write(models_dir, "thresholds", drifted)
        with pytest.raises(FrozenDesignError, match="does not reproduce"):
            replay_validation(scores)


def test_references_and_calendar_are_rebuilt_from_2022_and_compared(models_dir):
    rng = np.random.default_rng(0)
    slots = pl.datetime_range(
        date(PANEL_YEAR, 1, 3), date(PANEL_YEAR, 1, 10), "15m", closed="left", eager=True
    )
    series = pl.concat(
        pl.DataFrame(
            {
                "distrito": district,
                "slot_start": slots,
                "occupancy": rng.random(slots.len()) * 20,
                "flow": rng.random(slots.len()) * 900,
                "valid_slot": True,
            }
        )
        for district in DISTRICTS
    )
    grid = year_grid(series, DISTRICTS, PANEL_YEAR)
    refs = references(grid)
    refs.write_parquet(models_dir / FILES["references"])
    calendar_baseline(add_labels(grid, refs), DISTRICTS).write_parquet(
        models_dir / FILES["calendar"]
    )
    check_references(series)
    refs.with_columns(pl.col("threshold") + 0.1).write_parquet(models_dir / FILES["references"])
    with pytest.raises(FrozenDesignError, match="references rebuilt from 2022"):
        check_references(series)


def test_compare_gives_each_comparison_at_each_horizon_on_the_52_weeks(monkeypatch):
    monkeypatch.setattr(bootstrap, "REPETITIONS", 50)  # few repetitions keep the test fast
    monkeypatch.setattr(bootstrap, "MOVE_LIMIT", float("inf"))  # and no rerun with 10,000
    rng = np.random.default_rng(0)
    slot_start = pl.datetime_range(
        date(2023, 1, 1), date(2024, 1, 1), "1h", closed="left", eager=True
    )
    target = rng.random(slot_start.len()) < 0.1
    names = {name for pair in CONTRASTS.values() for name in pair}
    scores = pl.DataFrame(
        {
            "slot_start": slot_start,
            "target": target,
            **{name: rng.random(slot_start.len()) + 0.2 * target for name in names},
        }
    )
    table = compare({30: scores, 60: scores}, 2023)
    assert table.height == len(HORIZONS) * len(CONTRASTS)
    inside = scores.filter(
        pl.col("slot_start").dt.ordinal_day() <= 364
    )  # 31 December 2023 is outside the weeks
    for row in table.iter_rows(named=True):
        first, second = CONTRASTS[row["comparison"]]
        expected = average_precision_score(
            inside["target"], inside[first]
        ) - average_precision_score(inside["target"], inside[second])
        assert np.isclose(row["difference"], expected)
        assert row["low"] <= row["high"] and row["repetitions"] == 50
    with pytest.raises(AssertionError, match="another year"):
        compare({30: scores}, 2024)


def test_compare_refuses_a_missing_score_instead_of_ranking_it_last():
    # A null or NaN score would sort to the end of the ranking without any error; compare must stop instead.
    slot_start = pl.datetime_range(
        date(2023, 1, 1), date(2023, 1, 8), "1h", closed="left", eager=True
    )
    names = sorted({name for pair in CONTRASTS.values() for name in pair})
    scores = pl.DataFrame(
        {"slot_start": slot_start, "target": [k % 5 == 0 for k in range(slot_start.len())]}
        | {name: np.linspace(0, 1, slot_start.len()) for name in names}
    )
    gap = pl.int_range(pl.len()) == 3  # one origin loses its score
    with_nan = scores.with_columns(
        pl.when(gap).then(float("nan")).otherwise(pl.col(names[0])).alias(names[0])
    )
    with pytest.raises(ValueError, match=f"{names[0]} has a missing or non-finite score"):
        compare({30: with_nan}, 2023)
    with_null = scores.with_columns(
        pl.when(gap).then(None).otherwise(pl.col(names[-1])).alias(names[-1])
    )
    with pytest.raises(ValueError, match=f"{names[-1]} has a missing or non-finite score"):
        compare({30: with_null}, 2023)
