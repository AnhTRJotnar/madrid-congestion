from datetime import date, timedelta

import polars as pl

from madrid_congestion.labels import (
    SLOT_OF_DAY,
    add_labels,
    baseline_scores,
    calendar_baseline,
    references,
    year_grid,
)

# High-occupancy threshold of the synthetic district: occupancy 12 is high, 5 is not
THRESHOLD = 10.0
REFS = pl.DataFrame({"distrito": [1], "threshold": [THRESHOLD], "flow_reference": [100.0]})


def labelled(occupancy: list[float], invalid: tuple[int, ...] = ()) -> pl.DataFrame:
    # One district, consecutive 15-minute slots from 1 March 2022 00:00; slots listed in `invalid` are invalid.
    slots = pl.datetime_range(date(2022, 3, 1), date(2022, 3, 2), "15m", closed="left", eager=True)
    grid = pl.DataFrame(
        {
            "distrito": [1] * len(occupancy),
            "slot_start": slots[: len(occupancy)],
            "occupancy": occupancy,
            "flow": [50.0] * len(occupancy),
            "valid_slot": [i not in invalid for i in range(len(occupancy))],
        }
    )
    return add_labels(grid, REFS)


def test_two_high_slots_make_an_onset():
    rows = labelled([5, 5, 5, 5, 12, 12, 5, 5])
    assert rows["onset"].to_list() == [False] * 4 + [True] + [False] * 3


def test_one_high_slot_is_no_episode():
    rows = labelled([5, 5, 5, 5, 12, 5, 5, 5])
    assert rows["high"].sum() == 1
    assert not rows["onset"].any()


def test_invalid_slot_before_hides_the_onset():
    # the two high slots still form an episode, but its start cannot be observed
    rows = labelled([5, 5, 5, 5, 12, 12, 5, 5], invalid=(3,))
    assert rows["high"].to_list()[4:6] == [True, True]
    assert not rows["onset"].any()


def test_onset_at_t_plus_2_is_positive_at_30_minutes():
    # onset at slot 6: origin 4 (t+2) is positive, origin 3 (t+3) is not
    rows = labelled([5, 5, 5, 5, 5, 5, 12, 12, 5, 5])
    assert rows["target_30"].to_list()[3:5] == [False, True]
    assert rows["usable_30"][4]


def test_invalid_confirming_slot_makes_the_origin_unusable():
    # origin 4 needs slots 1 ... 7 valid (t-3 ... t+H+1); slot 7 confirms an episode starting at t+2
    occupancy: list[float] = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    assert labelled(occupancy)["usable_30"][4]
    assert not labelled(occupancy, invalid=(7,))["usable_30"][4]


def test_high_slot_in_the_previous_hour_makes_the_origin_unusable():
    # origin 5 looks back to slots 2 ... 5; a high slot at 2 breaks the calm rule, origin 6 is calm again
    rows = labelled([5, 5, 12, 5, 5, 5, 5, 5, 5, 5])
    assert rows["usable_30"].to_list()[5:7] == [False, True]


def test_onset_at_t_plus_4_is_positive_at_60_minutes_and_t_plus_5_is_not():
    # onset at slot 8: origin 4 (t+4) is positive at 60 minutes, origin 3 (t+5) is not
    rows = labelled([5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5])
    assert rows["target_60"].to_list()[3:5] == [False, True]
    assert rows["usable_60"][4]


def test_supplied_references_are_applied_unchanged_to_another_year():
    # 2022-like slots set the threshold; a later year with a very different level keeps it
    slots = pl.datetime_range(
        date(2022, 3, 1), date(2022, 3, 2), "15m", closed="left", eager=True
    )[:20]
    design_year = pl.DataFrame(
        {
            "distrito": [1] * 20,
            "slot_start": slots,
            "occupancy": [float(v) for v in range(1, 21)],
            "flow": [100.0] * 20,
            "valid_slot": [True] * 20,
        }
    )
    refs = references(design_year)
    assert refs["threshold"].item() == 19.0  # 95th percentile (nearest rank) of 1 ... 20
    # its own 95th percentile would be 49
    later_year = design_year.with_columns(pl.col("occupancy") + 30)
    labels = add_labels(later_year, refs)
    assert labels["threshold"].unique().to_list() == [19.0]
    assert labels["high"].all()  # every slot is above the 2022 threshold


def test_empty_calendar_cell_takes_the_district_day_type_share():
    # Tuesday 08:00-08:45: four usable origins, one positive. The 09:00 cell has no usable origin.
    slots = pl.datetime_range(date(2022, 3, 1), date(2022, 3, 2), "15m", closed="left", eager=True)
    rows = pl.DataFrame({"distrito": [1] * 96, "slot_start": slots}).with_columns(
        usable=SLOT_OF_DAY.is_between(32, 35),
        target=SLOT_OF_DAY == 32,
    )
    labels = rows.select(
        "distrito",
        "slot_start",
        usable_30="usable",
        target_30="target",
        usable_60="usable",
        target_60="target",
    )
    calendar = calendar_baseline(labels, pl.Series("distrito", [1])).filter(
        pl.col("horizon") == 30, ~pl.col("weekend")
    )
    rate = dict(zip(calendar["slot_of_day"], calendar["calendar_rate"], strict=True))
    assert rate[32] == 1.0  # a cell with origins keeps its own share
    assert rate[33] == 0.0  # a genuine zero share is not replaced
    assert rate[36] == 0.25  # an empty cell takes the weekday share of the district: 1 of 4


def test_baseline_scores_give_every_origin_its_cell_or_the_fallback():
    # Calendar from one Tuesday as above; scored origins on the next Tuesday at 08:00, 08:15 and 09:00
    slots = pl.datetime_range(date(2022, 3, 1), date(2022, 3, 2), "15m", closed="left", eager=True)
    labels = pl.DataFrame({"distrito": [1] * 96, "slot_start": slots}).with_columns(
        usable_30=SLOT_OF_DAY.is_between(32, 35),
        target_30=SLOT_OF_DAY == 32,
        usable_60=SLOT_OF_DAY.is_between(32, 35),
        target_60=SLOT_OF_DAY == 32,
    )
    calendar = calendar_baseline(labels, pl.Series("distrito", [1]))
    rows = pl.DataFrame(
        {
            "distrito": [1, 1, 1],
            "slot_start": [
                slots[32] + timedelta(days=7),
                slots[33] + timedelta(days=7),
                slots[36] + timedelta(days=7),
            ],
            "relative_occupancy": [0.5, 0.7, 0.9],
        }
    )
    scores = baseline_scores(rows, calendar, 30)
    assert scores["proximity"].to_list() == [
        0.5,
        0.7,
        0.9,
    ]  # the relative occupancy at t, in row order
    # own share, a genuine zero kept, an empty cell taking the district's weekday share (1 of 4)
    assert scores["calendar"].to_list() == [1.0, 0.0, 0.25]


def test_no_origin_reaches_into_the_next_year():
    # The last slot of a year (31 Dec 23:45) is in no series, so it is invalid on the year's own grid
    last_day = pl.datetime_range(
        date(2022, 12, 31), date(2023, 1, 1), "15m", closed="left", eager=True
    )
    series = pl.DataFrame(
        {
            "distrito": [1] * 95,
            "slot_start": last_day[:95],  # up to 23:30, as in the real series
            "occupancy": [5.0] * 95,
            "flow": [100.0] * 95,
            "valid_slot": [True] * 95,
        }
    )
    grid = year_grid(series, pl.Series("distrito", [1]), 2022)
    labels = add_labels(grid, REFS).filter(pl.col("slot_start").dt.date() == date(2022, 12, 31))
    usable = dict(zip(labels["slot_start"], labels["usable_30"], strict=True))
    end = last_day[0] + timedelta(hours=23)  # 31 Dec 23:00
    assert usable[end - timedelta(minutes=15)]  # 22:45 needs slots up to 23:30
    assert not usable[end]  # 23:00 needs 23:45, which belongs to no year
    assert grid["slot_start"].max() == last_day[95]  # the grid ends at 23:45 of the same year
