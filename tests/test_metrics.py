from __future__ import annotations

import pandas as pd

from ai_text_detector.metrics import (
    bootstrap_metric_intervals,
    compute_binary_metrics,
    subgroup_fpr,
    threshold_for_target_fpr,
)


def test_threshold_for_target_fpr_keeps_human_false_positives_low():
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]
    threshold = threshold_for_target_fpr(y_true, y_score, target_fpr=0.25)
    assert 0.3 < threshold <= 0.4
    metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)
    assert metrics["fpr"] <= 0.25


def test_bootstrap_metric_intervals_can_cap_large_samples():
    labels = [0, 1] * 100
    scores = [0.1, 0.9] * 100
    intervals = bootstrap_metric_intervals(
        labels,
        scores,
        threshold=0.5,
        iterations=3,
        seed=7,
        max_samples=20,
    )

    assert intervals["bootstrap_population_n"] == 200
    assert intervals["bootstrap_sample_n"] == 20
    assert intervals["bootstrap_sample_capped"] is True


def test_subgroup_fpr_human_only():
    df = pd.DataFrame(
        {
            "label": [0, 0, 0, 1],
            "native_status": ["learner", "learner", "native", "learner"],
        }
    )
    result = subgroup_fpr(
        df,
        [0.9, 0.1, 0.8, 0.2],
        threshold=0.5,
        group_cols=["native_status"],
        min_group_size=1,
    )
    learner = result[result["group_value"] == "learner"].iloc[0]
    assert learner["n_human"] == 2
    assert learner["false_positives"] == 1
