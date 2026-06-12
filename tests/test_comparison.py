from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ai_text_detector.comparison import (
    create_comparison_frames,
    save_comparison_sample_ids,
    save_model_comparison_table,
)
from ai_text_detector.schema import ensure_canonical


def test_create_comparison_frames_uses_balanced_binary_sample():
    rows = []
    for idx in range(10):
        rows.append(
            {
                "sample_id": f"h{idx}",
                "dataset": "fixture",
                "split": "test",
                "text": f"Human text {idx}",
                "label": 0,
                "source": "fixture",
                "domain": "essay",
                "group_id": f"h{idx}",
            }
        )
        rows.append(
            {
                "sample_id": f"a{idx}",
                "dataset": "fixture",
                "split": "test",
                "text": f"Generated text {idx}",
                "label": 1,
                "source": "fixture",
                "domain": "essay",
                "group_id": f"a{idx}",
            }
        )
    frames = create_comparison_frames(
        {"fixture": ensure_canonical(pd.DataFrame(rows))},
        max_samples_per_set=8,
        seed=42,
    )

    sampled = frames["fixture"]
    assert len(sampled) == 8
    assert sampled["label"].value_counts().to_dict() == {0: 4, 1: 4}


def test_create_comparison_frames_balances_native_status_for_human_only_fairness():
    rows = []
    for idx in range(20):
        rows.append(
            {
                "sample_id": f"n{idx}",
                "dataset": "icnale",
                "split": "fairness",
                "text": f"Native essay {idx}",
                "label": 0,
                "source": "fixture",
                "domain": "essay",
                "group_id": f"n{idx}",
                "native_status": "native",
            }
        )
    for idx in range(80):
        rows.append(
            {
                "sample_id": f"l{idx}",
                "dataset": "icnale",
                "split": "fairness",
                "text": f"Learner essay {idx}",
                "label": 0,
                "source": "fixture",
                "domain": "essay",
                "group_id": f"l{idx}",
                "native_status": "learner",
            }
        )
    frames = create_comparison_frames(
        {"icnale": ensure_canonical(pd.DataFrame(rows))},
        max_samples_per_set=40,
        seed=42,
    )

    sampled = frames["icnale"]
    assert sampled["native_status"].value_counts().to_dict() == {"native": 20, "learner": 20}


def test_save_comparison_sample_ids_and_table(tmp_path: Path):
    frame = ensure_canonical(
        pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "dataset": "fixture",
                    "split": "test",
                    "text": "Human text.",
                    "label": 0,
                    "source": "fixture",
                    "domain": "essay",
                    "group_id": "g1",
                }
            ]
        )
    )
    config = {"paths": {"reports_dir": str(tmp_path)}}
    sample_path = save_comparison_sample_ids({"fixture": frame}, config)
    assert sample_path.exists()

    metrics = tmp_path / "classical_metrics.json"
    metrics.write_text(
        json.dumps({"n": 1, "threshold": 0.5, "fpr": 0.0, "tpr": None}),
        encoding="utf-8",
    )
    table = save_model_comparison_table({"comparison_classical_logreg_fixture": {"metrics": metrics}}, None, config)
    output = pd.read_csv(table)
    assert output.loc[0, "model_family"] == "classical"
    assert output.loc[0, "model"] == "classical_logreg_fixture"
