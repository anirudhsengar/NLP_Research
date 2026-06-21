from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ANCHOR_TRAIN_SOURCES = [
    "HC3_reddit_eli5",
    "HC3_finance",
    "DAIGT_v2_Distance learning",
    "DAIGT_v2_Seeking multiple opinions",
    "HC3_open_qa",
]

ANCHOR_VALIDATION_SOURCES = [
    "DAIGT_v2_Car-free cities",
    "DAIGT_v2_Does the electoral college work?",
    "DAIGT_v2_Facial action coding system",
    "DAIGT_v2_Mandatory extracurricular activities",
    "DAIGT_v2_Summer projects",
    "HC3_medicine",
    "DAIGT_v2_Driverless cars",
    "DAIGT_v2_Exploring Venus",
]

ANCHOR_TEST_SOURCES = [
    "DAIGT_v2_Cell phones at school",
    "DAIGT_v2_Grades for extracurricular activities",
    "DAIGT_v2_Community service",
    'DAIGT_v2_"A Cowboy Who Rode the Waves"',
    "DAIGT_v2_The Face on Mars",
    "HC3_wiki_csai",
    "DAIGT_v2_Phones and driving",
]


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


def split_by_topic(
    df: pd.DataFrame,
    *,
    topic_col: str = "domain",
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign whole topics/domains to train/validation/test partitions."""
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("Split sizes must sum to 1.0")
    if topic_col not in df.columns:
        raise ValueError(f"Topic column not found: {topic_col}")

    work = df.copy()
    topics = (
        work[topic_col]
        .fillna("unknown")
        .astype(str)
        .replace("", "unknown")
        .drop_duplicates()
        .to_numpy()
    )
    if len(topics) < 3:
        raise ValueError("At least three topics are required for a topic-holdout split.")

    rng = np.random.default_rng(seed)
    shuffled = topics.copy()
    rng.shuffle(shuffled)

    n_topics = len(shuffled)
    n_train = _bounded_count(round(n_topics * train_size), minimum=1, maximum=n_topics - 2)
    n_validation = _bounded_count(
        round(n_topics * validation_size),
        minimum=1,
        maximum=n_topics - n_train - 1,
    )
    train_topics = set(shuffled[:n_train])
    validation_topics = set(shuffled[n_train : n_train + n_validation])
    test_topics = set(shuffled[n_train + n_validation :])
    if not test_topics:
        test_topics = {validation_topics.pop()}

    work["_topic_key"] = work[topic_col].fillna("unknown").astype(str).replace("", "unknown")
    split_map = {topic: "train" for topic in train_topics}
    split_map.update({topic: "validation" for topic in validation_topics})
    split_map.update({topic: "test" for topic in test_topics})
    work["split"] = work["_topic_key"].map(split_map)
    return work.drop(columns=["_topic_key"])


def split_by_anchor_source(
    df: pd.DataFrame,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Use the final manual source split from the public anchor notebooks."""
    work = df.copy()
    keys = anchor_source_keys(work)
    split_map = {source: "train" for source in ANCHOR_TRAIN_SOURCES}
    split_map.update({source: "validation" for source in ANCHOR_VALIDATION_SOURCES})
    split_map.update({source: "test" for source in ANCHOR_TEST_SOURCES})
    assigned = keys.map(split_map)
    missing = sorted(set(keys[assigned.isna()].astype(str)))
    if strict and missing:
        raise ValueError(f"Anchor source split has no assignment for source(s): {missing}")
    work["split"] = assigned.fillna("train")
    return work


def anchor_source_keys(df: pd.DataFrame) -> pd.Series:
    """Return source labels compatible with crusnix/ai_text_detector_final notebooks."""
    if "dataset" not in df.columns:
        raise ValueError("dataset column is required for anchor source keys")
    if "domain" not in df.columns and "source" not in df.columns:
        raise ValueError("domain or source column is required for anchor source keys")
    dataset = df["dataset"].fillna("").astype(str).str.lower()
    topic = df.get("domain", df.get("source")).fillna("").astype(str)
    fallback = df.get("source", topic).fillna("").astype(str)
    topic = topic.where(topic.str.len() > 0, fallback)
    return pd.Series(
        np.select(
            [dataset.eq("hc3"), dataset.eq("daigt_v2")],
            ["HC3_" + topic, "DAIGT_v2_" + topic],
            default=dataset + "_" + topic,
        ),
        index=df.index,
    )


def leave_one_domain_out_frames(
    df: pd.DataFrame,
    *,
    domain_col: str = "domain",
    validation_size: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Return split-labelled frames where each domain is held out as test once."""
    if domain_col not in df.columns:
        raise ValueError(f"Domain column not found: {domain_col}")
    frames: dict[str, pd.DataFrame] = {}
    domains = sorted(value for value in df[domain_col].dropna().astype(str).unique() if value)
    for domain in domains:
        work = df.copy()
        held_out = work[domain_col].astype(str) == domain
        work.loc[held_out, "split"] = "test"
        remaining = work[~held_out].copy()
        groups = remaining[["group_id", domain_col]].drop_duplicates("group_id").reset_index(drop=True)
        validation_groups, _ = train_test_split(
            groups,
            train_size=min(max(validation_size, 0.01), 0.99),
            random_state=seed,
            stratify=_safe_stratify(groups[domain_col]),
        )
        validation_ids = set(validation_groups["group_id"])
        work.loc[~held_out, "split"] = "train"
        work.loc[work["group_id"].isin(validation_ids), "split"] = "validation"
        frames[domain] = work.reset_index(drop=True)
    return frames


def _safe_stratify(values: pd.Series) -> pd.Series | None:
    counts = values.fillna("unknown").astype(str).value_counts()
    if len(counts) <= 1 or counts.min() < 2:
        return None
    return values.fillna("unknown").astype(str)


def _bounded_count(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
