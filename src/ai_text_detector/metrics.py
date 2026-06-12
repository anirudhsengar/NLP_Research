from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
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
        "fpr": float(fp / max(1, len(y_score))),
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
        "fpr": [],
        "tpr": [],
        "f1_macro": [],
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
