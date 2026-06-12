from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_text_detector.evaluation import evaluate_bundle, save_evaluation_summary
from ai_text_detector.schema import ensure_canonical


class StaticBundle:
    threshold = 0.5

    def score_frame(self, df: pd.DataFrame, *, batch_size: int | None = None):
        return np.asarray([0.6, 0.2, 0.7], dtype=float)


def test_human_only_evaluation_writes_defined_fairness_metrics(tmp_path: Path):
    df = ensure_canonical(
        pd.DataFrame(
            [
                {
                    "sample_id": "h1",
                    "dataset": "fixture",
                    "split": "fairness",
                    "text": "This is a human learner essay with enough words for a fixture.",
                    "label": 0,
                    "source": "fixture",
                    "domain": "essay",
                    "group_id": "g1",
                    "native_status": "learner",
                },
                {
                    "sample_id": "h2",
                    "dataset": "fixture",
                    "split": "fairness",
                    "text": "This is a human native essay with enough words for a fixture.",
                    "label": 0,
                    "source": "fixture",
                    "domain": "essay",
                    "group_id": "g2",
                    "native_status": "native",
                },
                {
                    "sample_id": "h3",
                    "dataset": "fixture",
                    "split": "fairness",
                    "text": "Another human learner essay with enough words for a fixture.",
                    "label": 0,
                    "source": "fixture",
                    "domain": "essay",
                    "group_id": "g3",
                    "native_status": "learner",
                },
            ]
        )
    )
    config = {
        "paths": {"reports_dir": str(tmp_path)},
        "model": {"target_fpr": 0.01},
        "evaluation": {"batch_size": 16, "min_subgroup_size": 1},
    }

    outputs = evaluate_bundle(StaticBundle(), df, name="fixture_fairness", config=config)
    summary_path = save_evaluation_summary({"fixture_fairness": outputs}, config)

    metrics = json.loads(outputs["metrics"].read_text(encoding="utf-8"))
    assert metrics["human_only"] is True
    assert metrics["fp"] == 2.0
    assert metrics["fpr"] == 2 / 3
    assert metrics["auroc"] is None
    assert metrics["tpr"] is None
    assert summary_path.exists()
    assert outputs["errors"].exists()
