from __future__ import annotations

import pandas as pd

from ai_text_detector.splitting import (
    ANCHOR_TEST_SOURCES,
    ANCHOR_TRAIN_SOURCES,
    ANCHOR_VALIDATION_SOURCES,
    anchor_source_keys,
    leave_one_domain_out_frames,
    split_by_anchor_source,
    split_by_group,
    split_by_topic,
)


def test_split_by_group_has_no_group_leakage():
    rows = []
    for group_idx in range(60):
        for label in (0, 1):
            rows.append(
                {
                    "group_id": f"group-{group_idx}",
                    "domain": f"domain-{group_idx % 3}",
                    "label": label,
                    "text": f"sample {group_idx} {label}",
                }
            )
    df = split_by_group(pd.DataFrame(rows), seed=7)
    group_counts = df.groupby("group_id")["split"].nunique()
    assert group_counts.max() == 1
    assert set(df["split"]) == {"train", "validation", "test"}


def test_split_by_topic_has_no_topic_leakage():
    rows = []
    for topic_idx in range(8):
        for item_idx in range(4):
            for label in (0, 1):
                rows.append(
                    {
                        "group_id": f"topic-{topic_idx}-item-{item_idx}",
                        "domain": f"topic-{topic_idx}",
                        "label": label,
                        "text": f"topic {topic_idx} sample {item_idx} label {label}",
                    }
                )
    df = split_by_topic(pd.DataFrame(rows), seed=11)
    topic_counts = df.groupby("domain")["split"].nunique()
    assert topic_counts.max() == 1
    assert set(df["split"]) == {"train", "validation", "test"}


def test_leave_one_domain_out_frames_hold_out_each_domain():
    rows = []
    for domain in ("finance", "medicine", "open_qa"):
        for idx in range(10):
            for label in (0, 1):
                rows.append(
                    {
                        "group_id": f"{domain}-{idx}",
                        "domain": domain,
                        "label": label,
                        "text": f"{domain} sample {idx} label {label}",
                    }
                )
    frames = leave_one_domain_out_frames(pd.DataFrame(rows), seed=5)
    assert set(frames) == {"finance", "medicine", "open_qa"}
    for domain, frame in frames.items():
        test_domains = set(frame.loc[frame["split"] == "test", "domain"])
        assert test_domains == {domain}
        assert {"train", "validation", "test"}.issubset(set(frame["split"]))


def test_split_by_anchor_source_matches_public_notebook_lists():
    rows = []
    for source in [*ANCHOR_TRAIN_SOURCES, *ANCHOR_VALIDATION_SOURCES, *ANCHOR_TEST_SOURCES]:
        dataset, domain = source.split("_", 1)
        if dataset == "DAIGT":
            dataset = "daigt_v2"
            domain = domain.removeprefix("v2_")
        else:
            dataset = "hc3"
        rows.append(
            {
                "dataset": dataset,
                "domain": domain,
                "source": domain,
                "group_id": source,
                "label": 0,
                "text": f"{source} sample",
            }
        )
    split = split_by_anchor_source(pd.DataFrame(rows))
    source_to_split = dict(zip(anchor_source_keys(split), split["split"], strict=False))
    assert {source_to_split[source] for source in ANCHOR_TRAIN_SOURCES} == {"train"}
    assert {source_to_split[source] for source in ANCHOR_VALIDATION_SOURCES} == {"validation"}
    assert {source_to_split[source] for source in ANCHOR_TEST_SOURCES} == {"test"}
