import numpy as np

from madrid_congestion.evaluation import best_csi_threshold, forecast_mae, warning_metrics

TARGET = np.array([True, False, True, False, False, True])
SCORE = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])


def test_best_csi_threshold_matches_a_brute_force_search():
    by_threshold = {t: warning_metrics(TARGET, SCORE, t)["csi"] for t in SCORE}
    assert best_csi_threshold(TARGET, SCORE) == max(by_threshold, key=lambda t: by_threshold[t])
    # warn at >= 0.7: 2 hits, 1 miss, 1 false alarm
    assert warning_metrics(TARGET, SCORE, 0.7) == {
        "warning_threshold": 0.7,
        "pod": 2 / 3,
        "far": 1 / 3,
        "csi": 0.5,
        "hits": 2,
        "misses": 1,
        "false_alarms": 1,
    }


def test_forecast_mae_converts_each_row_with_its_own_reference():
    # relative errors 0.1 and 0.2; references 10 and 20 percent: original-unit errors 1 and 4 points
    result = forecast_mae([1.0, 0.5], [0.9, 0.7], [0.9, 0.5], [10.0, 20.0])
    assert np.isclose(result["mae_relative"], 0.15)
    assert np.isclose(result["mae_original_units"], 2.5)
    assert np.isclose(result["naive_mae_relative"], 0.1)
