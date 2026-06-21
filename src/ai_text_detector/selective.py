from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ai_text_detector.metrics import compute_selective_metrics, threshold_for_target_fpr
from ai_text_detector.model import DetectorBundle


@dataclass
class SelectivePolicy:
    model_name: str
    low_threshold: float
    high_threshold: float
    target_fpr: float
    learner_target_fpr: float
    calibration_human_count: int
    calibration_learner_count: int
    calibration_ai_count: int
    pooled_human_fpr: float
    learner_human_fpr: float | None
    validation_ai_recall_high_confidence: float | None
    review_zone_rate_validation: float | None
    coverage_validation: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_selective_policy(
    bundle: DetectorBundle,
    *,
    model_name: str,
    validation: pd.DataFrame,
    human_calibration: pd.DataFrame,
    target_fpr: float,
    learner_target_fpr: float,
    batch_size: int,
) -> SelectivePolicy:
    """Create a three-way policy with subgroup-constrained high-confidence AI threshold."""
    validation_scores = bundle.score_frame(validation, batch_size=batch_size)
    validation_labels = validation["label"].astype(int).to_numpy()
    low_threshold = float(bundle.threshold)

    human_calibration = human_calibration.copy()
    human_scores = bundle.score_frame(human_calibration, batch_size=batch_size)
    pooled_threshold = threshold_for_target_fpr(
        [0] * len(human_scores),
        human_scores,
        target_fpr=target_fpr,
    )

    learner_mask = human_calibration.get("native_status", pd.Series("", index=human_calibration.index))
    learner_mask = learner_mask.astype(str) == "learner"
    learner_scores = human_scores[learner_mask.to_numpy()]
    if len(learner_scores):
        learner_threshold = threshold_for_target_fpr(
            [0] * len(learner_scores),
            learner_scores,
            target_fpr=learner_target_fpr,
        )
    else:
        learner_threshold = pooled_threshold
    high_threshold = max(low_threshold, float(pooled_threshold), float(learner_threshold))

    pooled_fpr = float((human_scores >= high_threshold).mean()) if len(human_scores) else 0.0
    learner_fpr = float((learner_scores >= high_threshold).mean()) if len(learner_scores) else None
    selective = compute_selective_metrics(
        validation_labels,
        validation_scores,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    ai_mask = validation_labels == 1
    ai_recall = (
        float((validation_scores[ai_mask] >= high_threshold).mean()) if int(ai_mask.sum()) else None
    )
    return SelectivePolicy(
        model_name=model_name,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        target_fpr=float(target_fpr),
        learner_target_fpr=float(learner_target_fpr),
        calibration_human_count=int(len(human_scores)),
        calibration_learner_count=int(len(learner_scores)),
        calibration_ai_count=int(ai_mask.sum()),
        pooled_human_fpr=pooled_fpr,
        learner_human_fpr=learner_fpr,
        validation_ai_recall_high_confidence=ai_recall,
        review_zone_rate_validation=_maybe_float(selective.get("review_zone_rate")),
        coverage_validation=_maybe_float(selective.get("coverage")),
    )


def policy_satisfies_constraints(policy: SelectivePolicy) -> bool:
    learner_ok = (
        policy.learner_human_fpr is None
        or policy.learner_human_fpr <= policy.learner_target_fpr + 1e-12
    )
    return policy.pooled_human_fpr <= policy.target_fpr + 1e-12 and learner_ok


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if np.isnan(numeric):
        return None
    return numeric
