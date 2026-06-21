from __future__ import annotations

import pandas as pd

from ai_text_detector.augmentation import transform_text, transformed_frame


def test_deterministic_style_transforms():
    assert transform_text("A   spaced\nsentence.", "whitespace") == "A spaced sentence."
    assert transform_text("Hello!!!World", "punctuation") == "Hello! World"
    assert transform_text("\u201cQuoted\u201d", "quotes") == '"Quoted"'


def test_transformed_frame_updates_text_and_ids():
    df = pd.DataFrame({"sample_id": ["s1"], "text": ["Hello!!!World"]})
    changed = transformed_frame(df, "punctuation")

    assert changed.loc[0, "text"] == "Hello! World"
    assert changed.loc[0, "sample_id"] == "s1:punctuation"
