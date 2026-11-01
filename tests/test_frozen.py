import json
import re

import polars as pl
import pytest

from madrid_congestion import frozen
from madrid_congestion.config import HORIZONS
from madrid_congestion.frozen import FILES, SCORES, FrozenDesignError, check_frozen, write_design
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
