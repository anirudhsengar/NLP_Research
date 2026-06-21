from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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


def compute_selective_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, float | bool | None]:
    """Metrics for a three-way human/review/high-confidence-AI policy."""
    y_true = np.asarray(list(y_true), dtype=int)
    y_score = np.asarray(list(y_score), dtype=float)
    if len(y_true) == 0:
        return {
            "selective_n": 0.0,
            "review_zone_rate": None,
            "coverage": None,
            "selective_risk": None,
            "high_confidence_ai_rate": None,
            "human_auto_accept_rate": None,
            "high_confidence_ai_fpr": None,
            "high_confidence_ai_recall": None,
        }

    low_threshold = float(low_threshold)
    high_threshold = max(float(high_threshold), low_threshold)
    review = (y_score >= low_threshold) & (y_score < high_threshold)
    pred_human = y_score < low_threshold
    pred_ai = y_score >= high_threshold
    decided = pred_human | pred_ai
    errors = ((pred_human & (y_true == 1)) | (pred_ai & (y_true == 0))) & decided
    humans = y_true == 0
    ai = y_true == 1
    return {
        "selective_n": float(len(y_true)),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "review_zone_rate": float(review.mean()),
        "coverage": float(decided.mean()),
        "selective_risk": float(errors.sum() / max(1, decided.sum())),
        "high_confidence_ai_rate": float(pred_ai.mean()),
        "human_auto_accept_rate": float(pred_human.mean()),
        "high_confidence_ai_fpr": float((pred_ai & humans).sum() / max(1, humans.sum())),
        "high_confidence_ai_recall": (
            float((pred_ai & ai).sum() / max(1, ai.sum())) if ai.sum() else None
        ),
    }


def bootstrap_metric_intervals(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    threshold: float,
    target_fpr: float = 0.01,
    iterations: int = 0,
    seed: int = 42,
    max_samples: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, float | int | None]:
    """Bootstrap confidence intervals for report-critical metrics."""
    if iterations <= 0:
        return {}
    y_true = np.asarray(list(y_true), dtype=int)
    y_score = np.asarray(list(y_score), dtype=float)
    if len(y_true) == 0:
        return {}

    rng = np.random.default_rng(seed)
    population_n = len(y_true)
    if max_samples and population_n > max_samples:
        indices = _stratified_sample_indices(y_true, max_samples=int(max_samples), rng=rng)
        y_true = y_true[indices]
        y_score = y_score[indices]

    values: dict[str, list[float]] = {
        "fpr": [],
        "tpr": [],
        "f1_macro": [],
        "auroc": [],
        "aupr": [],
    }
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=iterations)
    if n_jobs is None:
        n_jobs = 1
    if int(n_jobs) == 1:
        metric_rows = [
            _bootstrap_metric_row(
                y_true,
                y_score,
                threshold=threshold,
                target_fpr=target_fpr,
                seed=int(iteration_seed),
            )
            for iteration_seed in seeds
        ]
    else:
        metric_rows = Parallel(n_jobs=int(n_jobs), prefer="threads")(
            delayed(_bootstrap_metric_row)(
                y_true,
                y_score,
                threshold=threshold,
                target_fpr=target_fpr,
                seed=int(iteration_seed),
            )
            for iteration_seed in seeds
        )
    for metrics in metric_rows:
        if not metrics:
            continue
        for key in values:
            value = metrics.get(key)
            if value is not None and np.isfinite(float(value)):
                values[key].append(float(value))

    intervals: dict[str, float | int | None] = {
        "bootstrap_iterations": int(iterations),
        "bootstrap_population_n": int(population_n),
        "bootstrap_sample_n": int(len(y_true)),
        "bootstrap_sample_capped": bool(len(y_true) < population_n),
    }
    for key, metric_values in values.items():
        intervals[f"{key}_bootstrap_n"] = len(metric_values)
        if metric_values:
            intervals[f"{key}_ci_low"] = float(np.percentile(metric_values, 2.5))
            intervals[f"{key}_ci_high"] = float(np.percentile(metric_values, 97.5))
        else:
            intervals[f"{key}_ci_low"] = None
            intervals[f"{key}_ci_high"] = None
    return intervals


def _bootstrap_metric_row(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    target_fpr: float,
    seed: int,
) -> dict[str, float | bool | None]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(y_true), size=len(y_true))
    sample_labels = y_true[indices]
    sample_scores = y_score[indices]
    if set(np.unique(sample_labels)) == {0}:
        return compute_human_only_metrics(
            sample_scores,
            threshold=threshold,
            target_fpr=target_fpr,
        )
    if len(set(np.unique(sample_labels))) >= 2:
        return compute_binary_metrics(
            sample_labels,
            sample_scores,
            threshold=threshold,
            target_fpr=target_fpr,
        )
    return {}


def _stratified_sample_indices(y_true: np.ndarray, *, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples <= 0 or len(y_true) <= max_samples:
        return np.arange(len(y_true))
    sampled = []
    for label in np.unique(y_true):
        label_indices = np.flatnonzero(y_true == label)
        n_label = max(1, int(round(max_samples * len(label_indices) / len(y_true))))
        sampled.append(rng.choice(label_indices, size=min(len(label_indices), n_label), replace=False))
    indices = np.concatenate(sampled)
    if len(indices) > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False)
    elif len(indices) < max_samples:
        remaining = np.setdiff1d(np.arange(len(y_true)), indices, assume_unique=False)
        if len(remaining):
            extra = rng.choice(remaining, size=min(len(remaining), max_samples - len(indices)), replace=False)
            indices = np.concatenate([indices, extra])
    rng.shuffle(indices)
    return indices


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
