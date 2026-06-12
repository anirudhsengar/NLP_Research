from __future__ import annotations

import numpy as np
import pandas as pd

from ai_text_detector.calibration import calibrate_thresholds, split_icnale_calibration_audit
from ai_text_detector.schema import ensure_canonical


class ScoreBundle:
    threshold = 0.5
    metadata = {}

    def score_frame(self, df: pd.DataFrame, *, batch_size: int | None = None):
        return np.asarray(df["score_hint"], dtype=float)


def test_split_icnale_calibration_audit_keeps_groups_separate():
    rows = []
    for idx in range(20):
        rows.append(
            {
                "sample_id": f"s{idx}",
                "dataset": "icnale",
                "split": "fairness",
                "text": f"Human essay number {idx} with enough words for testing.",
                "label": 0,
                "source": "WE",
                "domain": "PTJ0",
                "group_id": f"g{idx}",
                "native_status": "learner" if idx % 2 else "native",
            }
        )
    df = ensure_canonical(pd.DataFrame(rows))

    calibration, audit = split_icnale_calibration_audit(df, calibration_fraction=0.3, seed=7)

    assert set(calibration["group_id"]).isdisjoint(set(audit["group_id"]))
    assert len(calibration) == 6
    assert len(audit) == 14


def test_calibrate_thresholds_adds_conservative_education_policy():
    hc3 = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "label": [0, 0, 1, 1],
            "score_hint": [0.1, 0.3, 0.8, 0.9],
        }
    )
    icnale = pd.DataFrame(
        {
            "label": [0, 0, 0, 0],
            "native_status": ["learner", "learner", "native", "native"],
            "score_hint": [0.2, 0.7, 0.4, 0.6],
        }
    )
    config = {
        "model": {"target_fpr": 0.01},
        "calibration": {"target_fpr": 0.25},
        "evaluation": {"batch_size": 32},
    }

    policy, report = calibrate_thresholds(ScoreBundle(), hc3=hc3, icnale_calibration=icnale, config=config)

    thresholds = policy["thresholds"]
    assert thresholds["education_mitigated"] >= thresholds["default_hc3_validation"]
    assert thresholds["education_mitigated"] >= thresholds["learner_human_calibrated"]
    assert set(report["threshold_policy"]) == set(thresholds)
