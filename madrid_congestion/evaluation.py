from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike
import polars as pl
from sklearn.metrics import average_precision_score

# Every count here is a count of usable calm origins, not of onsets: one onset can make several origins positive.


def warning_metrics(
    target: ArrayLike, score: ArrayLike, warning_threshold: float
) -> dict[str, float]:
    # Origin-level counts and measures when a warning is issued at every origin with score >= warning_threshold.
    # (The warning threshold is a cut on a warning score, chosen on 2023; not the high-occupancy threshold theta.)
    target, score = np.asarray(target, dtype=bool), np.asarray(score, dtype=float)
    warned = score >= warning_threshold
    hits = int(np.sum(warned & target))
    misses = int(np.sum(~warned & target))
    false_alarms = int(np.sum(warned & ~target))
    return {
        "warning_threshold": float(warning_threshold),
        # POD: share of positive origins with a warning; FAR: share of warnings given at negative origins
        "pod": hits / (hits + misses),
        "far": false_alarms / (hits + false_alarms) if hits + false_alarms else float("nan"),
        "csi": hits / (hits + misses + false_alarms),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
    }


def best_csi_threshold(target: ArrayLike, score: ArrayLike) -> float:
    # The threshold with the highest CSI: every distinct score is tried, warning at score >= threshold.
    # Chosen on the validation year and then held fixed for the test year.
    target, score = np.asarray(target, dtype=bool), np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="stable")
    sorted_score, sorted_target = score[order], target[order]
    hits = np.cumsum(sorted_target)  # hits if every origin up to this position gets a warning
    false_alarms = np.arange(1, len(score) + 1) - hits
    # last position of each distinct score, and the CSI there: hits / (hits + misses + false alarms)
    last = np.r_[np.flatnonzero(np.diff(sorted_score)), len(score) - 1]
    csi = hits[last] / (sorted_target.sum() + false_alarms[last])
    return float(sorted_score[last[np.argmax(csi)]])


def warning_table(
    target: ArrayLike,
    scores: Mapping[str, ArrayLike],
    warning_thresholds: Mapping[str, float] | None = None,
) -> pl.DataFrame:
    # AP and the warning measures of every score. Without warning thresholds (validation year), each score gets
    # its highest-CSI threshold; with them (test year), the given ones are applied unchanged.
    target = np.asarray(target, dtype=bool)
    rows = []
    for name, score in scores.items():
        if warning_thresholds is None:
            warning_threshold = best_csi_threshold(target, score)
        else:
            warning_threshold = warning_thresholds[name]
        # AP is read against the base rate, its expected value for a random ranking
        ap = float(average_precision_score(target, np.asarray(score, dtype=float)))
        rows.append({"score": name, "ap": ap, **warning_metrics(target, score, warning_threshold)})
    return pl.DataFrame(rows)


def forecast_mae(
    forecast: ArrayLike, actual: ArrayLike, naive: ArrayLike, reference: ArrayLike
) -> dict[str, float]:
    # MAE of a relative forecast and of the naive forecast (the value at t). Relative MAE is dimensionless.
    # For the original units, every value is first multiplied by its row's district reference (threshold:
    # occupancy in percentage points; flow reference: vehicles per hour), and the MAE is taken afterwards.
    forecast, actual, naive, reference = (
        np.asarray(x, dtype=float) for x in (forecast, actual, naive, reference)
    )
    mae = float(np.mean(np.abs(forecast - actual)))
    naive_mae = float(np.mean(np.abs(naive - actual)))
    return {
        "mae_relative": mae,
        "naive_mae_relative": naive_mae,
        "mae_original_units": float(np.mean(np.abs(forecast * reference - actual * reference))),
        "naive_mae_original_units": float(np.mean(np.abs(naive * reference - actual * reference))),
        "mae_reduction": 1 - mae / naive_mae,  # relative to the naive forecast
    }
