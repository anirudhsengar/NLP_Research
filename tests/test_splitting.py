from __future__ import annotations

import pandas as pd

from ai_text_detector.splitting import split_by_group


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

