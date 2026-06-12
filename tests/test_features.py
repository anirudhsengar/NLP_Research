from __future__ import annotations

import pandas as pd
import pytest

from ai_text_detector.features import BasicStatsTransformer, feature_names, make_feature_pipeline


def test_basic_stats_includes_explicit_burstiness_features():
    names = set(BasicStatsTransformer().get_feature_names_out())
    assert "stats:burstiness_index" in names
    assert "stats:sentence_len_delta_mean" in names
    assert "stats:word_frequency_entropy" in names


def test_feature_pipeline_can_run_stats_only():
    df = pd.DataFrame({"text": ["A short sentence. A much longer sentence follows here."]})
    pipeline = make_feature_pipeline(
        {
            "use_word_tfidf": False,
            "use_char_tfidf": False,
            "use_basic_stats": True,
            "use_spacy": False,
            "use_lm_stats": False,
        }
    )
    matrix = pipeline.fit_transform(df)
    assert matrix.shape[0] == 1
    assert "basic_stats__stats:burstiness_index" in set(feature_names(pipeline))


def test_feature_pipeline_rejects_empty_feature_set():
    with pytest.raises(ValueError, match="At least one feature family"):
        make_feature_pipeline(
            {
                "use_word_tfidf": False,
                "use_char_tfidf": False,
                "use_basic_stats": False,
                "use_spacy": False,
                "use_lm_stats": False,
            }
        )
