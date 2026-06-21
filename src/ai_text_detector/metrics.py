from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def threshold_for_target_fpr(y_true: Iterable[int], y_score: Iterable[float], target_fpr: float = 0.01) -> float:
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score), dtype=float)
    human_scores = y_score[y_true == 0]
    if len(human_scores) == 0:
        return 0.5
    max_false_positives = int(np.floor(max(0.0, target_fpr) * len(human_scores)))
    descending = np.sort(human_scores)[::-1]
    if max_false_positives == 0:
        boundary = descending[0]
    elif max_false_positives >= len(descending):
        return 0.0
    else:
        boundary = descending[max_false_positives]
    return float(np.nextafter(boundary, np.inf))


def tpr_at_fpr(y_true: Iterable[int], y_score: Iterable[float], target_fpr: float = 0.01) -> float | None:
    threshold = threshold_for_target_fpr(y_true, y_score, target_fpr)
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score), dtype=float)
    positives = y_true == 1
    if positives.sum() == 0:
        return None
    return float((y_score[positives] >= threshold).mean())


def compute_binary_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    threshold: float,
    target_fpr: float = 0.01,
) -> dict[str, float | None]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_score = np.asarray(list(y_score), dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    return {
        "n": float(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float((y_pred == y_true).mean()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "human_precision": float(precision[0]),
        "human_recall": float(recall[0]),
        "human_f1": float(f1[0]),
        "ai_precision": float(precision[1]),
        "ai_recall": float(recall[1]),
        "ai_f1": float(f1[1]),
        "auroc": _safe_metric(roc_auc_score, y_true, y_score),
        "aupr": _safe_metric(average_precision_score, y_true, y_score),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "fpr": float(fp / max(1, fp + tn)),
        "fnr": float(fn / max(1, fn + tp)),
        "tpr": float(tp / max(1, tp + fn)),
        f"tpr_at_fpr_{target_fpr:g}": tpr_at_fpr(y_true, y_score, target_fpr),
    }


def compute_human_only_metrics(
    y_score: Iterable[float],
    *,
    threshold: float,
    target_fpr: float = 0.01,
) -> dict[str, float | bool | None]:
    """Metrics for human-only fairness audits where recall/AUROC are undefined."""
    y_score = np.asarray(list(y_score), dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    fp = int(y_pred.sum())
    tn = int(len(y_pred) - fp)
    return {
        "human_only": True,
        "n": float(len(y_score)),
        "threshold": float(threshold),
        "target_fpr": float(target_fpr),
        "tn": float(tn),
        "fp": float(fp),
        "accuracy": float(tn / max(1, len(y_score))),
        "fpr": float(fp / max(1, len(y_score))),
        "fnr": None,
        "human_recall": float(tn / max(1, len(y_score))),
        "human_precision": 1.0,
        "f1_macro": None,
        "ai_precision": None,
        "ai_recall": None,
        "ai_f1": None,
        "auroc": None,
        "aupr": None,
        "tpr": None,
        f"tpr_at_fpr_{target_fpr:g}": None,
    }


def bootstrap_metric_intervals(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    threshold: float,
    target_fpr: float = 0.01,
    iterations: int = 0,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Bootstrap confidence intervals for report-critical metrics."""
    if iterations <= 0:
        return {}
    y_true = np.asarray(list(y_true), dtype=int)
    y_score = np.asarray(list(y_score), dtype=float)
    if len(y_true) == 0:
        return {}

    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "accuracy": [],
        "fpr": [],
        "fnr": [],
        "tpr": [],
        "f1_macro": [],
        "human_f1": [],
        "ai_f1": [],
        "auroc": [],
        "aupr": [],
    }
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_labels = y_true[indices]
        sample_scores = y_score[indices]
        if set(np.unique(sample_labels)) == {0}:
            metrics = compute_human_only_metrics(
                sample_scores,
                threshold=threshold,
                target_fpr=target_fpr,
            )
        elif len(set(np.unique(sample_labels))) >= 2:
            metrics = compute_binary_metrics(
                sample_labels,
                sample_scores,
                threshold=threshold,
                target_fpr=target_fpr,
            )
        else:
            continue
        for key in values:
            value = metrics.get(key)
            if value is not None and np.isfinite(float(value)):
                values[key].append(float(value))

    intervals: dict[str, float | int | None] = {"bootstrap_iterations": int(iterations)}
    for key, metric_values in values.items():
        intervals[f"{key}_bootstrap_n"] = len(metric_values)
        if metric_values:
            intervals[f"{key}_ci_low"] = float(np.percentile(metric_values, 2.5))
            intervals[f"{key}_ci_high"] = float(np.percentile(metric_values, 97.5))
        else:
            intervals[f"{key}_ci_low"] = None
            intervals[f"{key}_ci_high"] = None
    return intervals


def mcnemar_test(
    y_true: Iterable[int],
    y_pred_a: Iterable[int],
    y_pred_b: Iterable[int],
) -> dict[str, float | int]:
    """McNemar paired test for two classifiers evaluated on the same rows."""
    y_true = np.asarray(list(y_true), dtype=int)
    pred_a = np.asarray(list(y_pred_a), dtype=int)
    pred_b = np.asarray(list(y_pred_b), dtype=int)
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("McNemar inputs must have the same length.")

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    discordant = b + c
    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = float((abs(b - c) - 1) ** 2 / discordant)
        p_value = float(stats.chi2.sf(statistic, df=1))
    return {
        "n": int(len(y_true)),
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "discordant": discordant,
        "mcnemar_statistic": statistic,
        "mcnemar_p_value": p_value,
    }


def paired_bootstrap_auroc_difference(
    y_true: Iterable[int],
    y_score_a: Iterable[float],
    y_score_b: Iterable[float],
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Paired bootstrap interval for AUROC(A) - AUROC(B)."""
    y_true = np.asarray(list(y_true), dtype=int)
    score_a = np.asarray(list(y_score_a), dtype=float)
    score_b = np.asarray(list(y_score_b), dtype=float)
    if not (len(y_true) == len(score_a) == len(score_b)):
        raise ValueError("Paired AUROC inputs must have the same length.")
    if len(np.unique(y_true)) < 2:
        return {
            "auroc_a": None,
            "auroc_b": None,
            "auroc_diff_a_minus_b": None,
            "auroc_diff_ci_low": None,
            "auroc_diff_ci_high": None,
            "auroc_diff_bootstrap_n": 0,
            "auroc_diff_p_value": None,
        }

    auc_a = float(roc_auc_score(y_true, score_a))
    auc_b = float(roc_auc_score(y_true, score_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(max(0, iterations)):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_labels = y_true[indices]
        if len(np.unique(sample_labels)) < 2:
            continue
        diffs.append(
            float(
                roc_auc_score(sample_labels, score_a[indices])
                - roc_auc_score(sample_labels, score_b[indices])
            )
        )

    diff = auc_a - auc_b
    result: dict[str, float | int | None] = {
        "auroc_a": auc_a,
        "auroc_b": auc_b,
        "auroc_diff_a_minus_b": float(diff),
        "auroc_diff_bootstrap_n": len(diffs),
    }
    if diffs:
        diffs_array = np.asarray(diffs)
        result["auroc_diff_ci_low"] = float(np.percentile(diffs_array, 2.5))
        result["auroc_diff_ci_high"] = float(np.percentile(diffs_array, 97.5))
        centered = diffs_array - diff
        result["auroc_diff_p_value"] = float(np.mean(np.abs(centered) >= abs(diff)))
    else:
        result["auroc_diff_ci_low"] = None
        result["auroc_diff_ci_high"] = None
        result["auroc_diff_p_value"] = None
    return result


def subgroup_fpr_intervals(
    df: pd.DataFrame,
    y_score: Iterable[float],
    *,
    threshold: float,
    group_cols: list[str],
    min_group_size: int = 30,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Subgroup false-positive rates with Wilson confidence intervals."""
    rows = subgroup_fpr(
        df,
        y_score,
        threshold=threshold,
        group_cols=group_cols,
        min_group_size=min_group_size,
    )
    if rows.empty:
        return rows
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    ci_low = []
    ci_high = []
    for _, row in rows.iterrows():
        low, high = _wilson_interval(
            successes=int(row["false_positives"]),
            total=int(row["n_human"]),
            z=z,
        )
        ci_low.append(low)
        ci_high.append(high)
    rows["confidence"] = float(confidence)
    rows["fpr_ci_low"] = ci_low
    rows["fpr_ci_high"] = ci_high
    return rows


def calibration_diagnostics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    n_bins: int = 10,
) -> dict[str, float | int]:
    """Brier score and expected calibration error for binary scores."""
    y_true = np.asarray(list(y_true), dtype=int)
    y_score = np.asarray(list(y_score), dtype=float)
    if len(y_true) == 0:
        return {"n": 0, "brier": float("nan"), "ece": float("nan"), "n_bins": int(n_bins)}

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_ids = np.clip(np.digitize(y_score, bins, right=True) - 1, 0, int(n_bins) - 1)
    ece = 0.0
    for bin_idx in range(int(n_bins)):
        mask = bin_ids == bin_idx
        if not mask.any():
            continue
        confidence = float(y_score[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(confidence - accuracy)
    return {
        "n": int(len(y_true)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "ece": float(ece),
        "n_bins": int(n_bins),
    }


def subgroup_fpr(
    df: pd.DataFrame,
    y_score: Iterable[float],
    *,
    threshold: float,
    group_cols: list[str],
    min_group_size: int = 30,
) -> pd.DataFrame:
    work = df.copy()
    work["score_ai"] = np.asarray(list(y_score), dtype=float)
    work["pred_label"] = (work["score_ai"] >= threshold).astype(int)
    work = work[work["label"] == 0].copy()
    rows = []
    for group_col in group_cols:
        if group_col not in work.columns:
            continue
        for group_value, group in work.groupby(group_col):
            if len(group) < min_group_size:
                continue
            fp = int((group["pred_label"] == 1).sum())
            rows.append(
                {
                    "group_col": group_col,
                    "group_value": group_value,
                    "n_human": len(group),
                    "false_positives": fp,
                    "fpr": fp / len(group),
                }
            )
    return pd.DataFrame(rows)


def confusion_df(y_true: Iterable[int], y_score: Iterable[float], *, threshold: float) -> pd.DataFrame:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = (np.asarray(list(y_score), dtype=float) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(matrix, index=["true_human", "true_ai"], columns=["pred_human", "pred_ai"])


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _safe_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        return float(metric_fn(y_true, y_score))
    except ValueError:
        return None


def _wilson_interval(*, successes: int, total: int, z: float) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((proportion * (1 - proportion) / total) + (z**2 / (4 * total**2))) / denominator
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))
