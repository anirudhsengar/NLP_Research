from __future__ import annotations

import pandas as pd

from ai_text_detector.metrics import (
    calibration_diagnostics,
    compute_binary_metrics,
    mcnemar_test,
    paired_bootstrap_auroc_difference,
    subgroup_fpr,
    subgroup_fpr_intervals,
    threshold_for_target_fpr,
)


def test_threshold_for_target_fpr_keeps_human_false_positives_low():
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]
    threshold = threshold_for_target_fpr(y_true, y_score, target_fpr=0.25)
    assert 0.3 < threshold <= 0.4
    metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)
    assert metrics["fpr"] <= 0.25
    assert metrics["accuracy"] == 5 / 6
    assert metrics["fnr"] == 0.0


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


def test_statistical_helpers_return_paired_results():
    y_true = [0, 0, 1, 1, 1, 0]
    pred_a = [0, 0, 1, 1, 0, 0]
    pred_b = [0, 1, 1, 0, 0, 0]
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert result["discordant"] == 2
    assert 0.0 <= result["mcnemar_p_value"] <= 1.0

    auc_result = paired_bootstrap_auroc_difference(
        y_true,
        [0.1, 0.2, 0.9, 0.8, 0.4, 0.3],
        [0.1, 0.7, 0.8, 0.3, 0.4, 0.2],
        iterations=20,
        seed=42,
    )
    assert auc_result["auroc_diff_bootstrap_n"] > 0
    assert auc_result["auroc_a"] is not None


def test_subgroup_intervals_and_calibration_diagnostics():
    df = pd.DataFrame(
        {
            "label": [0, 0, 0, 0],
            "native_status": ["learner", "learner", "native", "native"],
        }
    )
    intervals = subgroup_fpr_intervals(
        df,
        [0.9, 0.1, 0.8, 0.2],
        threshold=0.5,
        group_cols=["native_status"],
        min_group_size=1,
    )
    assert {"fpr_ci_low", "fpr_ci_high"}.issubset(intervals.columns)
    assert intervals["fpr_ci_low"].between(0, 1).all()

    diagnostics = calibration_diagnostics([0, 0, 1, 1], [0.1, 0.4, 0.8, 0.9], n_bins=2)
    assert diagnostics["brier"] < 0.1
    assert diagnostics["ece"] >= 0.0
