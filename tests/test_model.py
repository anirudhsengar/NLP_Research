from __future__ import annotations

import pandas as pd

from ai_text_detector.model import train_detector
from ai_text_detector.schema import ensure_canonical


def test_train_detector_tiny_cycle():
    rows = []
    for split in ("train", "validation"):
        for idx in range(12):
            rows.append(
                {
                    "sample_id": f"{split}-h-{idx}",
                    "dataset": "hc3",
                    "split": split,
                    "text": "I personally think this essay has specific human details and varied wording.",
                    "label": 0,
                    "source": "fixture",
                    "domain": "fixture",
                    "group_id": f"{split}-h-{idx}",
                }
            )
            rows.append(
                {
                    "sample_id": f"{split}-a-{idx}",
                    "dataset": "hc3",
                    "split": split,
                    "text": "In conclusion, this comprehensive response highlights important aspects effectively.",
                    "label": 1,
                    "source": "fixture",
                    "domain": "fixture",
                    "group_id": f"{split}-a-{idx}",
                }
            )
    config = {
        "random_seed": 42,
        "features": {
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 4],
            "max_word_features": 100,
            "max_char_features": 100,
            "min_df": 1,
            "lowercase": True,
            "use_spacy": False,
        },
        "model": {"class_weight": "balanced", "max_iter": 200, "target_fpr": 0.1, "name": "test"},
    }
    bundle, metrics = train_detector(ensure_canonical(pd.DataFrame(rows)), config)
    explanation = bundle.explain_text("This response is comprehensive and highlights aspects effectively.")
    assert 0.0 <= explanation["score_ai"] <= 1.0
    assert explanation["feature_group_contributions"]
    assert "feature_family" in explanation["feature_group_contributions"][0]
    assert "f1_macro" in metrics
