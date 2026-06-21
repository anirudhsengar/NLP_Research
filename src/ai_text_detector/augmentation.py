from __future__ import annotations

import re

import pandas as pd


QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
)


def transform_text(text: str, transform: str) -> str:
    """Apply deterministic style-only transforms used in the paper study."""
    value = str(text)
    if transform == "whitespace":
        return normalize_whitespace(value)
    if transform == "quotes":
        return normalize_quotes(value)
    if transform == "punctuation":
        return normalize_punctuation(value)
    if transform == "style_normalized":
        return normalize_punctuation(normalize_quotes(normalize_whitespace(value)))
    raise ValueError(f"Unknown augmentation transform: {transform}")


def transformed_frame(df: pd.DataFrame, transform: str) -> pd.DataFrame:
    work = df.copy()
    work["text"] = work["text"].map(lambda text: transform_text(str(text), transform))
    work["sample_id"] = work["sample_id"].astype(str) + f":{transform}"
    return work


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_quotes(text: str) -> str:
    return str(text).translate(QUOTE_TRANSLATION)


def normalize_punctuation(text: str) -> str:
    value = str(text)
    value = re.sub(r"([!?.,;:])\1+", r"\1", value)
    value = re.sub(r"\s+([!?.,;:])", r"\1", value)
    value = re.sub(r"([!?.,;:])(?=\S)", r"\1 ", value)
    return normalize_whitespace(value)
