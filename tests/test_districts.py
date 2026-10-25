from datetime import datetime

import polars as pl
import pytest
import typer

from madrid_congestion.config import DST_ROWS, MAX_FLOW
from madrid_congestion.districts import (
    VALID_READING,
    build_district_series,
    by_slot_start,
    combine_locations,
    district_series,
    panel_cutoff,
    select_panel,
    urb_rows,
)

# district 1: three sensors; district 2: three sensors; district 3: one sensor (never eligible with minimum 2)
SHARES = pl.DataFrame({
    "id": [1, 2, 3, 4, 5, 6, 7],
    "distrito": [1, 1, 1, 2, 2, 2, 3],
    "valid_share": [0.9, 0.8, 0.6, 0.95, 0.7, 0.65, 0.99],
})


def test_valid_reading_rules():
    rows = pl.DataFrame({
        "periodo_integracion": pl.Series([15, 14, 16, 15, 15], dtype=pl.Int8),
        "ocupacion": pl.Series([10, 10, 10, None, 10], dtype=pl.Int16),
        "intensidad": pl.Series([300, 300, 300, 300, MAX_FLOW + 1], dtype=pl.Int32),
    })
    # complete; too few samples; clock-change artefact (16); occupancy missing; implausible flow
    assert rows.select(valid=VALID_READING)["valid"].to_list() == [True, False, False, False, False]


def test_urb_rows_drops_m30_and_clock_change_rows():
    rows = pl.LazyFrame({
        "tipo_elem": ["URB", "M30", "URB"],
        "fecha": [datetime(2022, 3, 1, 8, 0), datetime(2022, 3, 1, 8, 0), DST_ROWS[0]],
    })
    assert urb_rows(rows).collect().height == 1


def test_combine_locations_latest_wins_and_keeps_both():
    latest = pl.DataFrame({"id": [1, 2], "distrito": [5, None]})
    earlier = pl.DataFrame({"id": [1, 3], "distrito": [4, 7]})
    assert combine_locations(latest, earlier).sort("id").rows() == [(1, 5), (2, None), (3, 7)]


def test_panel_cutoff_is_highest_c_keeping_every_eligible_district():
    # at 0.71 district 2 keeps only one sensor, so 0.70 is the highest c that keeps districts 1 and 2
    assert panel_cutoff(SHARES, minimum=2, cutoffs=[0.5, 0.6, 0.7, 0.71, 0.8]) == 0.7


def test_panel_cutoff_raises_when_no_cutoff_keeps_the_districts():
    with pytest.raises(ValueError):
        panel_cutoff(SHARES, minimum=2, cutoffs=[0.96])


def test_select_panel_drops_districts_below_the_minimum():
    # sensor 7 passes c but district 3 has only one sensor
    assert select_panel(SHARES, c=0.7, minimum=2)["id"].to_list() == [1, 2, 4, 5]


def test_district_series_means_counts_and_valid_slots():
    t1, t2, t3 = datetime(2022, 3, 1, 8, 0), datetime(2022, 3, 1, 8, 15), datetime(2022, 3, 1, 8, 30)
    panel = pl.DataFrame({"id": pl.Series([1, 2, 3, 4], dtype=pl.Int32), "distrito": [1, 1, 1, 1]})
    readings = pl.DataFrame({
        "id": pl.Series([1, 2, 3, 1, 9], dtype=pl.Int32),  # sensor 9 is not in the panel
        "slot_start": [t1, t1, t1, t2, t1],
        "intensidad": pl.Series([100, 200, 300, 100, 999], dtype=pl.Int32),
        "ocupacion": pl.Series([10, 20, 30, 10, 99], dtype=pl.Int16),
        "periodo_integracion": pl.Series([15, 15, 15, 15, 15], dtype=pl.Int8),
    })
    slots = pl.DataFrame({"slot_start": [t1, t2, t3]})
    series = district_series(readings, panel, slots, minimum=2, panel_share=0.5)
    assert series["sensors"].to_list() == [3, 1, 0]  # t3 has no reading but is kept
    assert series["occupancy"].to_list() == [20.0, 10.0, None]  # never filled in
    assert series["valid_slot"].to_list() == [True, False, False]


def test_build_district_series_refuses_the_test_year():
    with pytest.raises(typer.BadParameter):
        build_district_series([2024])


def test_by_slot_start_keys_on_slot_start_and_keeps_the_year():
    rows = pl.DataFrame({"fecha": [datetime(2022, 1, 1, 0, 0), datetime(2022, 1, 1, 0, 15), datetime(2023, 1, 1, 0, 0)]})
    # the first row ends the last slot of 2021; the last row ends the last slot of 2022
    assert by_slot_start(rows, 2022)["slot_start"].to_list() == [datetime(2022, 1, 1, 0, 0), datetime(2022, 12, 31, 23, 45)]

