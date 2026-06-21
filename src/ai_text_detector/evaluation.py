from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from ai_text_detector.metrics import (
    bootstrap_metric_intervals,
    compute_binary_metrics,
    compute_human_only_metrics,
    confusion_df,
    subgroup_fpr,
)
from ai_text_detector.model import DetectorBundle


def evaluate_bundle(
    bundle: DetectorBundle,
    df: pd.DataFrame,
    *,
    name: str,
    config: dict[str, Any],
    threshold: float | None = None,
    threshold_policy: str = "default_hc3_validation",
) -> dict[str, Path]:
    scores = bundle.score_frame(df, batch_size=int(config["evaluation"].get("batch_size", 5000)))
    threshold = bundle.threshold if threshold is None else float(threshold)
    return evaluate_scores(
        df,
        scores,
        name=name,
        config=config,
        threshold=threshold,
        threshold_policy=threshold_policy,
    )


def evaluate_scores(
    df: pd.DataFrame,
    scores,
    *,
    name: str,
    config: dict[str, Any],
    threshold: float,
    threshold_policy: str = "default_hc3_validation",
) -> dict[str, Path]:
    """Evaluate precomputed AI scores and save report-ready artifacts."""
    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    figures_dir = Path(config["paths"]["reports_dir"]) / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    scores = np.asarray(scores, dtype=float)
    y_true = df["label"].astype(int)
    target_fpr = float(config["model"].get("target_fpr", 0.01))
    if set(y_true.unique()) == {0}:
        metrics = compute_human_only_metrics(scores, threshold=threshold, target_fpr=target_fpr)
    else:
        metrics = compute_binary_metrics(y_true, scores, threshold=threshold, target_fpr=target_fpr)
    metrics["threshold_policy"] = threshold_policy
    bootstrap_iterations = int(config.get("evaluation", {}).get("bootstrap_iterations", 0))
    if bootstrap_iterations > 0:
        metrics.update(
            bootstrap_metric_intervals(
                y_true,
                scores,
                threshold=threshold,
                target_fpr=target_fpr,
                iterations=bootstrap_iterations,
                seed=int(config.get("evaluation", {}).get("bootstrap_seed", 42)),
            )
        )

    metrics_path = results_dir / f"{name}_metrics.json"
    _write_json(metrics_path, metrics)

    metadata_cols = [
        "sample_id",
        "dataset",
        "split",
        "label",
        "source",
        "domain",
        "native_status",
        "country",
        "l1",
        "cefr_level",
        "length_bin",
    ]
    predictions = df[[col for col in metadata_cols if col in df.columns]].copy()
    predictions["score_ai"] = scores
    predictions["threshold"] = threshold
    predictions["threshold_policy"] = threshold_policy
    predictions["pred_label"] = (predictions["score_ai"] >= threshold).astype(int)
    predictions_path = results_dir / f"{name}_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    confusion = confusion_df(y_true, scores, threshold=threshold)
    confusion_path = results_dir / f"{name}_confusion.csv"
    confusion.to_csv(confusion_path)
    confusion_fig_path = figures_dir / f"{name}_confusion.png"
    _save_confusion_plot(confusion, confusion_fig_path, title=f"{name} confusion matrix")

    curves_path = figures_dir / f"{name}_roc_pr.png"
    _save_curves(y_true, scores, curves_path, title=name)

    subgroup_path = results_dir / f"{name}_subgroup_fpr.csv"
    subgroup = subgroup_fpr(
        df,
        scores,
        threshold=threshold,
        group_cols=["native_status", "country", "l1", "cefr_level", "length_bin", "domain"],
        min_group_size=int(config["evaluation"].get("min_subgroup_size", 30)),
    )
    subgroup.to_csv(subgroup_path, index=False)

    errors_path = results_dir / f"{name}_errors.csv"
    _save_error_analysis(
        predictions,
        errors_path,
        max_rows=int(config.get("evaluation", {}).get("error_analysis_rows", 100)),
    )

    outputs = {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "confusion": confusion_path,
        "confusion_figure": confusion_fig_path,
        "curves_figure": curves_path,
        "subgroup_fpr": subgroup_path,
        "errors": errors_path,
    }
    fairness_summary_path = _save_fairness_summary(subgroup, name=name, results_dir=results_dir)
    if fairness_summary_path is not None:
        outputs["fairness_summary"] = fairness_summary_path
    return outputs


def save_evaluation_summary(
    outputs: dict[str, dict[str, Path]],
    config: dict[str, Any],
    *,
    filename: str = "summary_table.csv",
) -> Path:
    """Create one compact table for report-ready model comparison."""
    rows = []
    for eval_name, paths in outputs.items():
        metrics_path = paths.get("metrics")
        if metrics_path is None or not Path(metrics_path).exists():
            continue
        with Path(metrics_path).open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {
            "eval_set": eval_name,
            "n": metrics.get("n"),
            "human_only": bool(metrics.get("human_only", False)),
            "threshold_policy": metrics.get("threshold_policy", ""),
            "threshold": metrics.get("threshold"),
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "human_f1": metrics.get("human_f1"),
            "ai_f1": metrics.get("ai_f1"),
            "auroc": metrics.get("auroc"),
            "aupr": metrics.get("aupr"),
            "fpr": metrics.get("fpr"),
            "fnr": metrics.get("fnr"),
            "tpr": metrics.get("tpr"),
            "tpr_at_target_fpr": metrics.get(f"tpr_at_fpr_{float(config['model'].get('target_fpr', 0.01)):g}"),
        }
        fairness_path = paths.get("fairness_summary")
        if fairness_path is not None and Path(fairness_path).exists():
            with Path(fairness_path).open("r", encoding="utf-8") as f:
                fairness = json.load(f)
            row.update(
                {
                    "native_fpr": fairness.get("native_fpr"),
                    "learner_fpr": fairness.get("learner_fpr"),
                    "fpr_gap_learner_minus_native": fairness.get("fpr_gap_learner_minus_native"),
                }
            )
        rows.append(row)

    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / filename
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def save_feature_importance(bundle: DetectorBundle, config: dict[str, Any], *, top_k: int = 50) -> Path:
    import numpy as np

    from ai_text_detector.features import feature_names

    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    figures_dir = Path(config["paths"]["reports_dir"]) / "figures"
    features_step = bundle.pipeline.named_steps["features"]
    classifier = bundle.pipeline.named_steps["classifier"]
    names = feature_names(features_step)
    coefs = classifier.coef_[0]
    df = pd.DataFrame({"feature": names, "coefficient": coefs})
    df["feature_family"] = df["feature"].map(_feature_family)
    df["abs_coefficient"] = df["coefficient"].abs()
    output = results_dir / "logreg_feature_coefficients.csv"
    df.sort_values("abs_coefficient", ascending=False).to_csv(output, index=False)

    group_output = results_dir / "logreg_feature_group_summary.csv"
    grouped = (
        df.assign(weighted_abs=df["abs_coefficient"])
        .groupby("feature_family", as_index=False)
        .agg(
            n_features=("feature", "count"),
            coefficient_l1=("abs_coefficient", "sum"),
            coefficient_mean_abs=("abs_coefficient", "mean"),
        )
        .sort_values("coefficient_l1", ascending=False)
    )
    top_by_group = (
        df.sort_values("abs_coefficient", ascending=False)
        .drop_duplicates("feature_family")[["feature_family", "feature", "coefficient"]]
        .rename(columns={"feature": "top_feature", "coefficient": "top_feature_coefficient"})
    )
    grouped.merge(top_by_group, on="feature_family", how="left").to_csv(group_output, index=False)

    top = pd.concat(
        [
            df.sort_values("coefficient", ascending=False).head(top_k).assign(direction="ai"),
            df.sort_values("coefficient", ascending=True).head(top_k).assign(direction="human"),
        ]
    )
    fig_path = figures_dir / "logreg_top_coefficients.png"
    plt.figure(figsize=(12, 10))
    plot_df = top.sort_values("coefficient")
    colors = np.where(plot_df["coefficient"] > 0, "#b2182b", "#2166ac")
    plt.barh(plot_df["feature"].astype(str), plot_df["coefficient"], color=colors)
    plt.xlabel("Logistic regression coefficient")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    _save_grouped_importance_plot(grouped, figures_dir / "logreg_feature_group_summary.png")
    return output


def _save_confusion_plot(confusion: pd.DataFrame, path: Path, *, title: str) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_curves(y_true, scores, path: Path, *, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    y_true_values = np.asarray(list(y_true), dtype=int)
    if len(set(y_true_values)) < 2:
        axes[0].text(0.5, 0.5, "ROC unavailable", ha="center", va="center")
        axes[1].text(0.5, 0.5, "PR unavailable", ha="center", va="center")
    else:
        RocCurveDisplay.from_predictions(y_true_values, scores, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_true_values, scores, ax=axes[1])
    axes[0].set_title(f"{title} ROC")
    axes[1].set_title(f"{title} PR")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_error_analysis(predictions: pd.DataFrame, path: Path, *, max_rows: int) -> None:
    work = predictions.copy()
    if "label" not in work.columns or "pred_label" not in work.columns:
        work.head(0).to_csv(path, index=False)
        return
    work["error_type"] = ""
    work.loc[(work["label"] == 0) & (work["pred_label"] == 1), "error_type"] = "false_positive"
    work.loc[(work["label"] == 1) & (work["pred_label"] == 0), "error_type"] = "false_negative"
    errors = work[work["error_type"] != ""].copy()
    if errors.empty:
        errors.head(0).to_csv(path, index=False)
        return
    false_positives = errors[errors["error_type"] == "false_positive"].sort_values(
        "score_ai",
        ascending=False,
    )
    false_negatives = errors[errors["error_type"] == "false_negative"].sort_values(
        "score_ai",
        ascending=True,
    )
    pd.concat([false_positives.head(max_rows), false_negatives.head(max_rows)]).to_csv(path, index=False)


def _save_fairness_summary(subgroup: pd.DataFrame, *, name: str, results_dir: Path) -> Path | None:
    if subgroup.empty:
        return None
    native_status = subgroup[subgroup["group_col"] == "native_status"].copy()
    required = {"native", "learner"}
    if not required.issubset(set(native_status["group_value"].astype(str))):
        return None

    by_status = native_status.set_index("group_value")
    summary = {
        "eval_set": name,
        "native_n_human": int(by_status.loc["native", "n_human"]),
        "native_false_positives": int(by_status.loc["native", "false_positives"]),
        "native_fpr": float(by_status.loc["native", "fpr"]),
        "learner_n_human": int(by_status.loc["learner", "n_human"]),
        "learner_false_positives": int(by_status.loc["learner", "false_positives"]),
        "learner_fpr": float(by_status.loc["learner", "fpr"]),
    }
    summary["fpr_gap_learner_minus_native"] = summary["learner_fpr"] - summary["native_fpr"]
    output = results_dir / f"{name}_fairness_summary.json"
    _write_json(output, summary)
    return output


def _feature_family(feature: str) -> str:
    text = str(feature)
    if "__" in text:
        return text.split("__", 1)[0]
    if ":" in text:
        return text.split(":", 1)[0]
    return "other"


def _save_grouped_importance_plot(grouped: pd.DataFrame, path: Path) -> None:
    if grouped.empty:
        return
    plot_df = grouped.sort_values("coefficient_l1", ascending=True)
    plt.figure(figsize=(8, 4))
    plt.barh(plot_df["feature_family"], plot_df["coefficient_l1"], color="#4c78a8")
    plt.xlabel("Sum of absolute logistic-regression coefficients")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True, allow_nan=False)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
