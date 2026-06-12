from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_HUMAN = 0
LABEL_AI = 1

CANONICAL_COLUMNS = [
    "sample_id",
    "dataset",
    "split",
    "text",
    "label",
    "source",
    "domain",
    "group_id",
    "native_status",
    "country",
    "l1",
    "cefr_level",
    "length_bin",
    "license_tag",
]


def word_count(text: str) -> int:
    return len(str(text).split())


def length_bin_from_words(n_words: int) -> str:
    if n_words < 100:
        return "short_lt_100"
    if n_words < 300:
        return "medium_100_299"
    if n_words < 700:
        return "long_300_699"
    return "very_long_700_plus"


def ensure_canonical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df["text"] = df["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df["length_bin"] = df["text"].map(lambda text: length_bin_from_words(word_count(text)))
    return df[CANONICAL_COLUMNS]


def write_jsonl(df: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ensure_canonical(df).to_json(output, orient="records", lines=True, force_ascii=False)
    return output


def read_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")
    return ensure_canonical(pd.read_json(path, orient="records", lines=True))

