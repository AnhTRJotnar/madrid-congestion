from datetime import date

import polars as pl

from madrid_congestion.config import (
    CALM_SLOTS,
    HIGH_OCC_QUANTILE,
    HORIZONS,
    MIN_EPISODE_SLOTS,
    SUPPORT_SHARE,
    WEEKEND_FROM,
)

# Calendar keys of an origin: weekday/weekend and the 15-minute slot of day (0 = 00:00 ... 95 = 23:45)
WEEKEND = pl.col("slot_start").dt.weekday() >= WEEKEND_FROM
SLOT_OF_DAY = (
    pl.col("slot_start").dt.hour().cast(pl.Int32) * 4 + pl.col("slot_start").dt.minute() // 15
)


def later(column: str, k: int) -> pl.Expr:
    # The value k slots later (k > 0) or earlier (k < 0) in the same district; outside the year it is null.
    # Needs the rows sorted by district and slot start.
    return pl.col(column).shift(-k).over("distrito")


def studied_districts(series: pl.DataFrame) -> pl.Series:
    # Districts with at least SUPPORT_SHARE of their slots valid; decided on the 2022 series only.
    support = series.group_by("distrito").agg(valid_share=pl.col("valid_slot").mean())
    return support.filter(pl.col("valid_share") >= SUPPORT_SHARE)["distrito"].sort()


def year_grid(series: pl.DataFrame, districts: pl.Series, year: int) -> pl.DataFrame:
    # Every slot start of `year` on a regular 15-minute grid, for every studied district. Slots missing from
    # the series (clock-change rows, the year's last slot) count as invalid, so no value is ever filled in.
    # The last slot of a year (31 Dec 23:45) is in no series, so it is always invalid: building each year on
    # its own grid drops exactly the origins whose history or target window would cross the year boundary.
    grid = pl.datetime_range(
        date(year, 1, 1), date(year + 1, 1, 1), "15m", closed="left", eager=True
    )
    return (
        districts.to_frame()
        .join(grid.alias("slot_start").to_frame(), how="cross")
        .join(
            series.select("distrito", "slot_start", "occupancy", "flow", "valid_slot"),
            on=["distrito", "slot_start"],
            how="left",
        )
        .with_columns(pl.col("valid_slot").fill_null(False))
        .sort("distrito", "slot_start")
    )


def references(grid: pl.DataFrame) -> pl.DataFrame:
    # Per district, over its valid slots: the high-occupancy threshold and the flow reference,
    # both the 95th percentile (polars default quantile, nearest rank). Fixed on 2022.
    return (
        grid.filter(pl.col("valid_slot"))
        .group_by("distrito")
        .agg(
            valid_slots=pl.len(),
            threshold=pl.col("occupancy").quantile(HIGH_OCC_QUANTILE),
            flow_reference=pl.col("flow").quantile(HIGH_OCC_QUANTILE),
        )
        .sort("distrito")
    )


def add_labels(grid: pl.DataFrame, refs: pl.DataFrame) -> pl.DataFrame:
    # High-occupancy slots, onsets, calm origins, and per horizon the target and the usable-origin flag.
    assert MIN_EPISODE_SLOTS == 2  # the onset rule below checks exactly one following slot
    labels = (
        grid.join(refs.select("distrito", "threshold", "flow_reference"), on="distrito")
        .sort("distrito", "slot_start")  # a join does not keep row order, and later() needs it
        .with_columns(
            high=pl.col("valid_slot") & (pl.col("occupancy") >= pl.col("threshold")),
            relative_occupancy=pl.col("occupancy") / pl.col("threshold"),
            relative_flow=pl.col("flow") / pl.col("flow_reference"),
        )
    )
    labels = labels.with_columns(
        # onset: this slot and the next are high, and the slot before is valid and below the threshold
        onset=pl.col("high")
        & later("high", 1).fill_null(False)
        & later("valid_slot", -1).fill_null(False)
        & ~later("high", -1).fill_null(True),
        # calm: none of t-3 ... t is high; history_valid: all of t-3 ... t are valid
        calm=pl.all_horizontal([~later("high", -k).fill_null(True) for k in range(CALM_SLOTS)]),
        history_valid=pl.all_horizontal(
            [later("valid_slot", -k).fill_null(False) for k in range(CALM_SLOTS)]
        ),
    )
    for minutes, H in HORIZONS.items():
        labels = labels.with_columns(
            # target: an onset starts in t+1 ... t+H
            pl.any_horizontal([later("onset", k).fill_null(False) for k in range(1, H + 1)]).alias(
                f"target_{minutes}"
            ),
            # usable: calm, and every slot from t-3 to t+H+1 valid (the last one confirms an episode at the window's end)
            (
                pl.col("calm")
                & pl.col("history_valid")
                & pl.all_horizontal(
                    [later("valid_slot", k).fill_null(False) for k in range(1, H + 2)]
                )
            ).alias(f"usable_{minutes}"),
        )
    return labels


def calendar_baseline(labels: pl.DataFrame, districts: pl.Series) -> pl.DataFrame:
    # Per district x weekday/weekend x slot of day and horizon: the share of usable calm origins with a positive
    # target. A cell without any usable origin takes the share of its district and day type over all slots.
    cells = (
        districts.to_frame()
        .join(pl.DataFrame({"weekend": [False, True]}), how="cross")
        .join(
            pl.int_range(0, 96, dtype=pl.Int32, eager=True).alias("slot_of_day").to_frame(),
            how="cross",
        )
    )
    calendar = []
    for minutes in HORIZONS:
        usable = labels.filter(pl.col(f"usable_{minutes}")).with_columns(
            weekend=WEEKEND, slot_of_day=SLOT_OF_DAY
        )
        by_cell = usable.group_by("distrito", "weekend", "slot_of_day").agg(
            origins=pl.len(), positives=pl.col(f"target_{minutes}").sum()
        )
        by_day_type = usable.group_by("distrito", "weekend").agg(
            day_type_rate=pl.col(f"target_{minutes}").mean()
        )
        calendar.append(
            cells.join(by_cell, on=["distrito", "weekend", "slot_of_day"], how="left")
            .join(by_day_type, on=["distrito", "weekend"])
            .with_columns(pl.col("origins", "positives").fill_null(0), horizon=pl.lit(minutes))
            .with_columns(
                calendar_rate=pl.when(pl.col("origins") > 0)
                .then(pl.col("positives") / pl.col("origins"))
                .otherwise(pl.col("day_type_rate"))
            )
        )
    return pl.concat(calendar).sort("horizon", "distrito", "weekend", "slot_of_day")


def baseline_scores(rows: pl.DataFrame, calendar: pl.DataFrame, minutes: int) -> pl.DataFrame:
    # The two baseline warning scores of each usable calm origin, in row order. Proximity: the relative occupancy
    # at t (the naive forecast that nothing changes). Calendar: the 2022 share of its district x weekday/weekend x
    # slot-of-day cell at this horizon; an empty cell already holds its district x day-type share (above).
    keys = ["distrito", "weekend", "slot_of_day"]
    shares = calendar.filter(pl.col("horizon") == minutes).select(*keys, calendar="calendar_rate")
    scores = (
        rows.select(
            "distrito", proximity="relative_occupancy", weekend=WEEKEND, slot_of_day=SLOT_OF_DAY
        )
        .join(shares, on=keys, how="left", maintain_order="left")  # keeps the row order of `rows`
        .select("proximity", "calendar")
    )
    assert scores.height == rows.height and scores["calendar"].null_count() == 0, (
        "an origin has no calendar cell"
    )
    return scores
