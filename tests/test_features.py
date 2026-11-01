from datetime import date

import polars as pl

from madrid_congestion.features import add_features, feature_names
from madrid_congestion.labels import add_labels

REFS = pl.DataFrame(
    {"distrito": [1, 2], "threshold": [10.0, 10.0], "flow_reference": [100.0, 100.0]}
)
DISTRICTS = pl.Series("distrito", [1, 2])


def features(occupancy: dict[int, list[float]]) -> pl.DataFrame:
    # Features of consecutive 15-minute slots from 1 March 2022 for each district, all slots valid.
    slots = pl.datetime_range(date(2022, 3, 1), date(2022, 3, 2), "15m", closed="left", eager=True)
    grid = pl.concat(
        [
            pl.DataFrame(
                {
                    "distrito": [district] * len(values),
                    "slot_start": slots[: len(values)],
                    "occupancy": values,
                    "flow": [10.0 * v for v in values],
                    "valid_slot": [True] * len(values),
                }
            )
            for district, values in occupancy.items()
        ]
    )
    return add_features(add_labels(grid, REFS), DISTRICTS)


def test_values_after_the_origin_do_not_change_its_features():
    # Origin 5: changing every slot after it must leave its features unchanged
    before = features({1: [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    after = features({1: [1, 2, 3, 4, 5, 6, 70, 80, 90]})
    names = feature_names(DISTRICTS)
    assert before.select(names).row(5) == after.select(names).row(5)


def test_lags_never_cross_into_another_district():
    rows = features({1: [1, 2, 3, 4, 5, 6], 2: [7, 8, 9, 10, 11, 12]})
    district_2 = rows.filter(pl.col("distrito") == 2)
    assert district_2["occupancy_lag1"][0] is None  # not district 1's last slot
    assert district_2["occupancy_lag3"].to_list()[:4] == [None, None, None, 0.7]
    assert district_2["flow_lag2"][1] is None
