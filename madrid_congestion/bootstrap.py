from datetime import date

import numpy as np
from numpy.typing import ArrayLike
import polars as pl

# Paired week-block bootstrap of an AP difference (paper Section 5.3)
WEEKS = 52  # complete seven-day blocks from 1 January; the last one or two days of the year are left out
REPETITIONS = 2_000
LARGE_REPETITIONS = (
    10_000  # used for both seeds when the second seed moves an endpoint by MOVE_LIMIT or more
)
SEEDS = (1, 2)  # the fixed seed and the second seed of the stability check
MOVE_LIMIT = 0.005
LEVEL = 0.95  # percentile interval


def week_block(slot_start: pl.Series, year: int) -> np.ndarray:
    # Block of each origin by its slot start: days since 1 January // 7, so 0 ... 51; -1 for the days after
    # the 52 complete weeks (30-31 December in 2024, 31 December in 2023), which are outside the sample.
    days = (slot_start.dt.date() - date(year, 1, 1)).dt.total_days().to_numpy()
    block = days // 7
    return np.where(block < WEEKS, block, -1)


class RankedScore:
    # One score's ranking, sorted once; every repetition only changes the weights of the origins.
    def __init__(self, target: ArrayLike, score: ArrayLike):
        score = np.asarray(score, dtype=float)
        self.order = np.argsort(-score, kind="stable")
        self.positive = np.asarray(target, dtype=bool)[self.order]
        # last position of each run of equal scores: tied origins enter the ranking together
        self.group_end = np.r_[np.flatnonzero(np.diff(score[self.order])), score.size - 1]

    def ap(self, weight: np.ndarray) -> float:
        # Weighted step-wise AP, as scikit-learn's average_precision_score with sample_weight:
        # the sum over thresholds of (recall gain) x precision.
        w = weight[self.order]
        hits = np.cumsum(w * self.positive)[self.group_end]
        warned = np.cumsum(w)[self.group_end]
        keep = (
            warned > 0
        )  # thresholds where only zero-weight origins have been passed give no precision
        hits, warned = hits[keep], warned[keep]
        recall_gain = np.diff(np.r_[0.0, hits]) / hits[-1]
        return float(np.sum(recall_gain * hits / warned))


def paired_differences(
    target: ArrayLike,
    first: ArrayLike,
    second: ArrayLike,
    block: np.ndarray,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    # AP(first) - AP(second) for each repetition. A repetition draws WEEKS blocks with replacement; an origin's
    # weight is the number of times its block was drawn, so all districts of a week move together, and both
    # scores get the same weights (paired). Origins outside the blocks always weigh 0.
    a, b = RankedScore(target, first), RankedScore(target, second)
    rng = np.random.default_rng(seed)
    inside = block >= 0
    differences = np.empty(repetitions)
    for repetition in range(repetitions):
        draws = np.bincount(rng.integers(0, WEEKS, size=WEEKS), minlength=WEEKS)
        weight = np.where(inside, draws[np.maximum(block, 0)], 0).astype(float)
        differences[repetition] = a.ap(weight) - b.ap(weight)
    return differences


def interval(differences: np.ndarray) -> tuple[float, float]:
    tail = (1 - LEVEL) / 2 * 100
    low, high = np.percentile(differences, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_difference(
    target: ArrayLike, first: ArrayLike, second: ArrayLike, block: np.ndarray
) -> dict[str, float | int]:
    # The point estimate on the 52 complete weeks and its 95% interval, with the second-seed check: if an
    # endpoint moves by MOVE_LIMIT or more, both seeds are rerun with LARGE_REPETITIONS and those are reported.
    inside = (block >= 0).astype(float)
    estimate = RankedScore(target, first).ap(inside) - RankedScore(target, second).ap(inside)

    def seeds_run(repetitions: int) -> tuple[list[tuple[float, float]], float]:
        runs = [
            interval(paired_differences(target, first, second, block, repetitions, seed))
            for seed in SEEDS
        ]
        return runs, max(abs(runs[1][0] - runs[0][0]), abs(runs[1][1] - runs[0][1]))

    repetitions = REPETITIONS
    runs, move = seeds_run(repetitions)
    if move >= MOVE_LIMIT:
        repetitions = LARGE_REPETITIONS
        runs, move = seeds_run(repetitions)
    return {
        "difference": estimate,
        "low": runs[0][0],
        "high": runs[0][1],
        "repetitions": repetitions,
        "second_seed_move": move,
    }
