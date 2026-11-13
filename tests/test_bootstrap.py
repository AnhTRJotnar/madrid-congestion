import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from madrid_congestion import bootstrap
from madrid_congestion.bootstrap import RankedScore, paired_differences, week_block


def test_weighted_ap_matches_scikit_learn_with_ties_and_zero_weights():
    rng = np.random.default_rng(0)
    target = rng.random(500) < 0.1
    score = rng.integers(0, 40, size=500) / 40  # many tied scores
    weight = rng.integers(0, 4, size=500).astype(
        float
    )  # some origins weigh 0, others count twice or more
    expected = average_precision_score(target, score, sample_weight=weight)
    assert np.isclose(RankedScore(target, score).ap(weight), expected)


def test_week_blocks_cover_52_complete_weeks():
    slots = pl.Series(
        [
            "2024-01-01 00:00",
            "2024-01-07 23:45",
            "2024-01-08 00:00",
            "2024-12-29 23:45",
            "2024-12-30 00:00",
            "2024-12-31 12:00",
        ]
    ).str.to_datetime()
    assert week_block(slots, 2024).tolist() == [0, 0, 1, 51, -1, -1]


def test_paired_differences_repeat_with_the_same_seed_and_are_zero_for_equal_scores():
    rng = np.random.default_rng(1)
    target = rng.random(700) < 0.2
    score = rng.random(700)
    block = np.repeat(np.arange(-1, 52), 14)[:700]
    first = paired_differences(target, score + target, score, block, 20, seed=1)
    assert np.array_equal(
        first, paired_differences(target, score + target, score, block, 20, seed=1)
    )
    assert np.allclose(paired_differences(target, score, score, block, 20, seed=1), 0)


def test_bootstrap_difference_moves_to_more_repetitions_only_when_the_second_seed_moves(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap, "REPETITIONS", 20)  # small numbers keep the test fast
    monkeypatch.setattr(bootstrap, "LARGE_REPETITIONS", 40)
    rng = np.random.default_rng(2)
    target = rng.random(700) < 0.2
    first, second = rng.random(700) + 0.3 * target, rng.random(700) + 0.1 * target
    block = np.repeat(np.arange(-1, 52), 14)[:700]
    inside = block >= 0
    expected = average_precision_score(target[inside], first[inside]) - average_precision_score(
        target[inside], second[inside]
    )
    monkeypatch.setattr(bootstrap, "MOVE_LIMIT", float("inf"))  # never unstable: 20 repetitions
    stable = bootstrap.bootstrap_difference(target, first, second, block)
    assert np.isclose(stable["difference"], expected) and stable["repetitions"] == 20
    monkeypatch.setattr(bootstrap, "MOVE_LIMIT", 0.0)  # always unstable: rerun with 40
    unstable = bootstrap.bootstrap_difference(target, first, second, block)
    assert unstable["repetitions"] == 40 and unstable["low"] <= unstable["high"]
