from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ai_text_detector.baselines import run_transformer_baselines
from ai_text_detector.evaluation import evaluate_bundle, save_evaluation_summary
from ai_text_detector.model import DetectorBundle


def create_comparison_frames(
    eval_frames: dict[str, pd.DataFrame],
    *,
    max_samples_per_set: int | None,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Create fixed model-comparison subsets from evaluation frames."""
    if not max_samples_per_set:
        return {name: frame.copy().reset_index(drop=True) for name, frame in eval_frames.items()}
    return {
        name: _sample_frame(frame, max_samples=max_samples_per_set, seed=seed).reset_index(drop=True)
        for name, frame in eval_frames.items()
    }


def save_comparison_sample_ids(frames: dict[str, pd.DataFrame], config: dict[str, Any]) -> Path:
    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for eval_set, frame in frames.items():
        for _, row in frame.iterrows():
            rows.append(
                {
                    "eval_set": eval_set,
                    "sample_id": row.get("sample_id", ""),
                    "dataset": row.get("dataset", ""),
                    "label": row.get("label", ""),
                    "native_status": row.get("native_status", ""),
                    "domain": row.get("domain", ""),
                    "length_bin": row.get("length_bin", ""),
                }
            )
    output = results_dir / "comparison_sample_ids.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def evaluate_classical_on_comparison(
    bundle: DetectorBundle,
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    *,
    model_name: str = "classical_logreg",
) -> dict[str, dict[str, Path]]:
    outputs = {}
    for eval_name, frame in frames.items():
        outputs[f"comparison_{model_name}_{eval_name}"] = evaluate_bundle(
            bundle,
            frame,
            name=f"comparison_{model_name}_{eval_name}",
            config=config,
            threshold=bundle.threshold,
            threshold_policy="default_hc3_validation",
        )
        education_threshold = bundle.education_threshold
        if education_threshold > bundle.threshold:
            outputs[f"comparison_{model_name}_education_mitigated_{eval_name}"] = evaluate_bundle(
                bundle,
                frame,
                name=f"comparison_{model_name}_education_mitigated_{eval_name}",
                config=config,
                threshold=education_threshold,
                threshold_policy="education_mitigated",
            )
    save_evaluation_summary(outputs, config, filename="comparison_classical_summary_table.csv")
    return outputs


def run_comparison_baselines(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, Path]:
    return run_transformer_baselines(
        frames,
        config,
        max_samples=0,
        output_prefix="comparison_baseline",
        summary_filename="comparison_baseline_summary_table.csv",
        threshold_max_samples=int(config.get("baselines", {}).get("threshold_calibration_max_samples", 2000)),
    )


def save_model_comparison_table(
    classical_outputs: dict[str, dict[str, Path]],
    baseline_summary: Path | None,
    config: dict[str, Any],
) -> Path:
    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    rows = []
    rows.extend(_comparison_metric_rows(classical_outputs, model_family="classical"))
    if baseline_summary is not None and baseline_summary.exists():
        baseline_df = pd.read_csv(baseline_summary)
        for _, row in baseline_df.iterrows():
            rows.append(
                {
                    "model_family": "baseline",
                    "model": _baseline_model_from_eval_set(str(row["eval_set"])),
                    "eval_set": _baseline_eval_set(str(row["eval_set"])),
                    "threshold_policy": row.get("threshold_policy", ""),
                    "n": row.get("n"),
                    "human_only": row.get("human_only"),
                    "threshold": row.get("threshold"),
                    "f1_macro": row.get("f1_macro"),
                    "auroc": row.get("auroc"),
                    "aupr": row.get("aupr"),
                    "fpr": row.get("fpr"),
                    "tpr": row.get("tpr"),
                    "tpr_at_target_fpr": row.get("tpr_at_target_fpr"),
                    "native_fpr": row.get("native_fpr"),
                    "learner_fpr": row.get("learner_fpr"),
                    "fpr_gap_learner_minus_native": row.get("fpr_gap_learner_minus_native"),
                }
            )
    output = results_dir / "model_comparison_table.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def save_paper_readiness_summary(
    config: dict[str, Any],
    *,
    comparison_table: Path | None = None,
    sample_ids: Path | None = None,
    baseline_outputs: dict[str, Path] | None = None,
) -> Path:
    """Write a concise machine-readable checklist of paper artifacts."""
    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    payload = {
        "required_artifacts": {
            "data_profile": _exists(results_dir / "data_profile.json"),
            "run_manifest": _exists(results_dir / "run_manifest.json"),
            "summary_table": _exists(results_dir / "summary_table.csv"),
            "calibrated_summary_table": _exists(results_dir / "calibrated_summary_table.csv"),
            "ablation_metrics": _exists(results_dir / "ablation_metrics.csv"),
            "feature_coefficients": _exists(results_dir / "logreg_feature_coefficients.csv"),
            "feature_group_summary": _exists(results_dir / "logreg_feature_group_summary.csv"),
            "comparison_sample_ids": _exists(sample_ids) if sample_ids is not None else False,
            "model_comparison_table": _exists(comparison_table) if comparison_table is not None else False,
            "paper_results_notes": _exists(results_dir / "paper_results_notes.md"),
        },
        "baseline_outputs": {key: str(value) for key, value in (baseline_outputs or {}).items()},
        "known_limitations": [
            "Full 300k-row transformer baseline inference is computationally heavy; use fixed sampled comparison unless GPU time is available.",
            "Default HC3-calibrated threshold has high OOD and learner-English false-positive rates; report calibrated education policy as a separate manual-review policy.",
            "Optional spaCy and GPT-2 feature ablations require extra dependencies and should be reported only if rerun successfully.",
        ],
    }
    output = results_dir / "paper_readiness_summary.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
    return output


def _sample_frame(df: pd.DataFrame, *, max_samples: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_samples:
        return df.copy()
    if (
        "label" in df.columns
        and df["label"].nunique(dropna=False) <= 1
        and "native_status" in df.columns
        and {"native", "learner"}.issubset(set(df["native_status"].astype(str)))
    ):
        return _balanced_sample_by_column(df, column="native_status", max_samples=max_samples, seed=seed)
    if "label" not in df.columns or df["label"].nunique(dropna=False) <= 1:
        return df.sample(n=max_samples, random_state=seed).copy()
    return _balanced_sample_by_column(df, column="label", max_samples=max_samples, seed=seed)


def _balanced_sample_by_column(
    df: pd.DataFrame,
    *,
    column: str,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    sampled_indices = []
    per_group = max(1, max_samples // df[column].nunique())
    for _, group in df.groupby(column):
        sampled_indices.extend(group.sample(min(len(group), per_group), random_state=seed).index.tolist())
    sampled = df.loc[sampled_indices]
    if len(sampled) < max_samples:
        remaining = df.drop(sampled.index, errors="ignore")
        if not remaining.empty:
            sampled = pd.concat(
                [sampled, remaining.sample(min(len(remaining), max_samples - len(sampled)), random_state=seed)]
            )
    return sampled.sample(frac=1.0, random_state=seed).copy()


def _comparison_metric_rows(outputs: dict[str, dict[str, Path]], *, model_family: str) -> list[dict[str, Any]]:
    rows = []
    for eval_name, paths in outputs.items():
        metrics_path = paths.get("metrics")
        if metrics_path is None or not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        fairness = {}
        fairness_path = paths.get("fairness_summary")
        if fairness_path is not None and fairness_path.exists():
            with fairness_path.open("r", encoding="utf-8") as f:
                fairness = json.load(f)
        rows.append(
            {
                "model_family": model_family,
                "model": _classical_model_from_eval_set(eval_name),
                "eval_set": _classical_eval_set(eval_name),
                "threshold_policy": metrics.get("threshold_policy", ""),
                "n": metrics.get("n"),
                "human_only": bool(metrics.get("human_only", False)),
                "threshold": metrics.get("threshold"),
                "f1_macro": metrics.get("f1_macro"),
                "auroc": metrics.get("auroc"),
                "aupr": metrics.get("aupr"),
                "fpr": metrics.get("fpr"),
                "tpr": metrics.get("tpr"),
                "tpr_at_target_fpr": metrics.get("tpr_at_fpr_0.01"),
                "native_fpr": fairness.get("native_fpr"),
                "learner_fpr": fairness.get("learner_fpr"),
                "fpr_gap_learner_minus_native": fairness.get("fpr_gap_learner_minus_native"),
            }
        )
    return rows


def _classical_model_from_eval_set(eval_name: str) -> str:
    prefix = "comparison_"
    if eval_name.startswith(prefix):
        rest = eval_name[len(prefix) :]
    else:
        rest = eval_name
    for suffix in ("_hc3_test", "_gpt_wiki_intro_ood", "_icnale_fairness"):
        if rest.endswith(suffix):
            return rest[: -len(suffix)]
    return rest


def _classical_eval_set(eval_name: str) -> str:
    for suffix in ("hc3_test", "gpt_wiki_intro_ood", "icnale_fairness"):
        if eval_name.endswith(suffix):
            return suffix
    return eval_name


def _baseline_model_from_eval_set(eval_name: str) -> str:
    prefix = "comparison_baseline_"
    rest = eval_name[len(prefix) :] if eval_name.startswith(prefix) else eval_name
    for suffix in ("_hc3_test", "_gpt_wiki_intro_ood", "_icnale_fairness"):
        if rest.endswith(suffix):
            return rest[: -len(suffix)]
    return rest


def _baseline_eval_set(eval_name: str) -> str:
    for suffix in ("hc3_test", "gpt_wiki_intro_ood", "icnale_fairness"):
        if eval_name.endswith(suffix):
            return suffix
    return eval_name


def _exists(path: Path | None) -> bool:
    return bool(path and path.exists())
