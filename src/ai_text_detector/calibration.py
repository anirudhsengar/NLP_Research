from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ai_text_detector.metrics import threshold_for_target_fpr
from ai_text_detector.model import DetectorBundle


def split_icnale_calibration_audit(
    icnale: pd.DataFrame,
    *,
    calibration_fraction: float,
    seed: int,
    group_col: str = "group_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ICNALE by participant group for threshold calibration and held-out audit."""
    if icnale.empty:
        return icnale.copy(), icnale.copy()
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    groups = icnale[[group_col, "native_status"]].drop_duplicates(group_col).reset_index(drop=True)
    stratify = _safe_stratify(groups["native_status"])
    calibration_groups, audit_groups = train_test_split(
        groups,
        train_size=calibration_fraction,
        random_state=seed,
        stratify=stratify,
    )
    calibration = icnale[icnale[group_col].isin(calibration_groups[group_col])].copy()
    audit = icnale[icnale[group_col].isin(audit_groups[group_col])].copy()
    return calibration.reset_index(drop=True), audit.reset_index(drop=True)


def calibrate_thresholds(
    bundle: DetectorBundle,
    *,
    hc3: pd.DataFrame,
    icnale_calibration: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build conservative threshold policies for academic-integrity use."""
    target_fpr = float(config.get("calibration", {}).get("target_fpr", config["model"].get("target_fpr", 0.01)))
    batch_size = int(config["evaluation"].get("batch_size", 5000))

    hc3_validation_human = hc3[(hc3["split"] == "validation") & (hc3["label"].astype(int) == 0)].copy()
    pooled_humans = pd.concat([hc3_validation_human, icnale_calibration], ignore_index=True)
    learner_humans = icnale_calibration[icnale_calibration["native_status"] == "learner"].copy()

    default_threshold = float(bundle.threshold)
    pooled_threshold = _threshold_from_human_frame(
        bundle,
        pooled_humans,
        target_fpr=target_fpr,
        batch_size=batch_size,
        fallback=default_threshold,
    )
    learner_threshold = _threshold_from_human_frame(
        bundle,
        learner_humans,
        target_fpr=target_fpr,
        batch_size=batch_size,
        fallback=pooled_threshold,
    )
    education_threshold = max(default_threshold, pooled_threshold, learner_threshold)

    thresholds = {
        "default_hc3_validation": default_threshold,
        "pooled_human_calibrated": pooled_threshold,
        "learner_human_calibrated": learner_threshold,
        "education_mitigated": education_threshold,
    }
    policy = {
        "target_fpr": target_fpr,
        "thresholds": thresholds,
        "review_zone": {
            "low": default_threshold,
            "high": education_threshold,
            "action": "manual_review",
        },
        "calibration_counts": {
            "hc3_validation_human": int(len(hc3_validation_human)),
            "icnale_calibration_human": int(len(icnale_calibration)),
            "icnale_calibration_learner": int(len(learner_humans)),
            "pooled_human": int(len(pooled_humans)),
        },
    }
    rows = []
    for source_name, frame in {
        "hc3_validation_human": hc3_validation_human,
        "icnale_calibration_human": icnale_calibration,
        "icnale_calibration_learner": learner_humans,
        "pooled_human": pooled_humans,
    }.items():
        if frame.empty:
            continue
        scores = bundle.score_frame(frame, batch_size=batch_size)
        for policy_name, threshold in thresholds.items():
            rows.append(
                _calibration_row(
                    source_name=source_name,
                    policy_name=policy_name,
                    threshold=threshold,
                    scores=scores,
                    target_fpr=target_fpr,
                )
            )
    return policy, pd.DataFrame(rows)


def save_calibration_report(
    bundle: DetectorBundle,
    *,
    policy: dict[str, Any],
    report: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "calibration_threshold_report.csv"
    json_path = results_dir / "calibration_policy.json"
    report.to_csv(csv_path, index=False)
    bundle.metadata["calibration"] = policy

    import json

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, sort_keys=True, allow_nan=False)
    return csv_path, json_path


def _threshold_from_human_frame(
    bundle: DetectorBundle,
    frame: pd.DataFrame,
    *,
    target_fpr: float,
    batch_size: int,
    fallback: float,
) -> float:
    if frame.empty:
        return fallback
    scores = bundle.score_frame(frame, batch_size=batch_size)
    y_true = [0] * len(scores)
    return threshold_for_target_fpr(y_true, scores, target_fpr=target_fpr)


def _calibration_row(
    *,
    source_name: str,
    policy_name: str,
    threshold: float,
    scores,
    target_fpr: float,
) -> dict[str, float | str]:
    false_positives = int((scores >= threshold).sum())
    n_human = int(len(scores))
    return {
        "source": source_name,
        "threshold_policy": policy_name,
        "threshold": float(threshold),
        "target_fpr": float(target_fpr),
        "n_human": n_human,
        "false_positives": false_positives,
        "fpr": false_positives / max(1, n_human),
    }


def _safe_stratify(values: pd.Series) -> pd.Series | None:
    counts = values.fillna("unknown").astype(str).value_counts()
    if len(counts) <= 1 or counts.min() < 2:
        return None
    return values.fillna("unknown").astype(str)
