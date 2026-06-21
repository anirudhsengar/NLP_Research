from __future__ import annotations

import numpy as np
import pandas as pd

from ai_text_detector.metrics import compute_selective_metrics
from ai_text_detector.selective import calibrate_selective_policy, policy_satisfies_constraints


class ScoreBundle:
    threshold = 0.4

    def score_frame(self, df: pd.DataFrame, *, batch_size: int | None = None):
        return np.asarray(df["score_hint"], dtype=float)


def test_compute_selective_metrics_tracks_review_zone_and_risk():
    metrics = compute_selective_metrics(
        [0, 0, 1, 1],
        [0.1, 0.5, 0.6, 0.9],
        low_threshold=0.4,
        high_threshold=0.8,
    )

    assert metrics["review_zone_rate"] == 0.5
    assert metrics["coverage"] == 0.5
    assert metrics["high_confidence_ai_fpr"] == 0.0
    assert metrics["high_confidence_ai_recall"] == 0.5


def test_calibrate_selective_policy_respects_learner_constraint():
    validation = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "native_status": ["", "", "", ""],
            "score_hint": [0.1, 0.2, 0.7, 0.95],
        }
    )
    human_calibration = pd.DataFrame(
        {
            "label": [0, 0, 0, 0],
            "native_status": ["learner", "learner", "native", "native"],
            "score_hint": [0.3, 0.85, 0.2, 0.4],
        }
    )

    policy = calibrate_selective_policy(
        ScoreBundle(),
        model_name="fixture",
        validation=validation,
        human_calibration=human_calibration,
        target_fpr=0.0,
        learner_target_fpr=0.0,
        batch_size=16,
    )

    assert policy.high_threshold > 0.85
    assert policy.learner_human_fpr == 0.0
    assert policy_satisfies_constraints(policy)
