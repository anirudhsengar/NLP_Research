from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_group(
    df: pd.DataFrame,
    *,
    group_col: str = "group_id",
    stratify_col: str = "domain",
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/validation/test while keeping paired samples in one split."""
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("Split sizes must sum to 1.0")

    df = df.copy()
    groups = df[[group_col, stratify_col]].drop_duplicates(group_col).reset_index(drop=True)
    stratify = _safe_stratify(groups[stratify_col])

    train_groups, temp_groups = train_test_split(
        groups,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
    )

    temp_test_fraction = test_size / (validation_size + test_size)
    temp_stratify = _safe_stratify(temp_groups[stratify_col])
    validation_groups, test_groups = train_test_split(
        temp_groups,
        test_size=temp_test_fraction,
        random_state=seed,
        stratify=temp_stratify,
    )

    split_map = {group: "train" for group in train_groups[group_col]}
    split_map.update({group: "validation" for group in validation_groups[group_col]})
    split_map.update({group: "test" for group in test_groups[group_col]})
    df["split"] = df[group_col].map(split_map)
    return df


def _safe_stratify(values: pd.Series) -> pd.Series | None:
    counts = values.fillna("unknown").astype(str).value_counts()
    if len(counts) <= 1 or counts.min() < 2:
        return None
    return values.fillna("unknown").astype(str)

