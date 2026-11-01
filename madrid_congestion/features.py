import math

import polars as pl

from madrid_congestion.config import CALM_SLOTS, HORIZONS
from madrid_congestion.labels import SLOT_OF_DAY, later

LAGS = range(CALM_SLOTS)  # t, t-1, t-2, t-3: the same hour the calm rule looks at
SLOTS_PER_DAY = 96  # 24 hours of 15-minute slots
WEEKDAYS = range(2, 8)  # polars weekday 2 (Tuesday) ... 7 (Sunday); Monday is the reference day


def feature_names(districts: pl.Series, flow: bool = True) -> list[str]:
    # The model inputs in a fixed order (paper Table 3.4); the first studied district is the reference district.
    # flow=False leaves out the four flow lags: variant A of the flow ablation.
    return (
        [f"occupancy_lag{k}" for k in LAGS]
        + ([f"flow_lag{k}" for k in LAGS] if flow else [])
        + ["slot_sin", "slot_cos"]
        + [f"weekday_{d}" for d in WEEKDAYS]
        + [f"district_{d}" for d in districts.sort()[1:]]
    )


def add_features(labels: pl.DataFrame, districts: pl.Series) -> pl.DataFrame:
    # Every feature at origin t uses only slots up to and including t. Needs rows sorted by district and slot.
    angle = 2 * math.pi * SLOT_OF_DAY / SLOTS_PER_DAY
    weekday = pl.col("slot_start").dt.weekday()
    return labels.with_columns(
        *[later("relative_occupancy", -k).alias(f"occupancy_lag{k}") for k in LAGS],
        *[later("relative_flow", -k).alias(f"flow_lag{k}") for k in LAGS],
        angle.sin().alias("slot_sin"),
        angle.cos().alias("slot_cos"),
        *[(weekday == d).cast(pl.Float64).alias(f"weekday_{d}") for d in WEEKDAYS],
        *[
            (pl.col("distrito") == d).cast(pl.Float64).alias(f"district_{d}")
            for d in districts.sort()[1:]
        ],
    )


def add_forecast_targets(labels: pl.DataFrame) -> pl.DataFrame:
    # What the two forecasting approaches predict: the largest relative occupancy and relative flow in t+1 ... t+H.
    return labels.with_columns(
        *[
            pl.max_horizontal([later(f"relative_{variable}", k) for k in range(1, H + 1)]).alias(
                f"{variable}_max_{minutes}"
            )
            for minutes, H in HORIZONS.items()
            for variable in ("occupancy", "flow")
        ]
    )


def origin_rows(labels: pl.DataFrame, districts: pl.Series, minutes: int) -> pl.DataFrame:
    # One row per usable calm origin at this horizon, with the inputs and the two forecast targets.
    rows = add_forecast_targets(add_features(labels, districts))
    return rows.filter(pl.col(f"usable_{minutes}"))
