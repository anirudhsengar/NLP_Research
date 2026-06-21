from __future__ import annotations

import pandas as pd

from ai_text_detector.data import load_anchor_merged, load_daigt_v2


def test_load_daigt_v2_canonicalizes_local_csv(tmp_path):
    csv_path = tmp_path / "train_v2_drcat_02.csv"
    pd.DataFrame(
        {
            "text": ["A human essay.", "An AI generated essay."],
            "label": [0, 1],
            "prompt_name": ["phones", "phones"],
            "source": ["student", "chatgpt"],
        }
    ).to_csv(csv_path, index=False)
    config = {
        "_project_root": str(tmp_path),
        "random_seed": 42,
        "datasets": {"daigt_v2": {"path": csv_path.name}},
    }

    df = load_daigt_v2(config)

    assert list(df["dataset"].unique()) == ["daigt_v2"]
    assert set(df["label"].astype(int)) == {0, 1}
    assert set(df["domain"]) == {"phones"}
    assert df["sample_id"].str.startswith("daigt_v2:phones").all()


def test_load_anchor_merged_canonicalizes_public_notebook_csv(tmp_path):
    csv_path = tmp_path / "merged_dataset.csv"
    pd.DataFrame(
        {
            "text": ["HC3 human answer.", "DAIGT generated essay."],
            "label": [0, 1],
            "source": ["HC3_finance", "DAIGT_v2_Distance learning"],
        }
    ).to_csv(csv_path, index=False)
    config = {
        "_project_root": str(tmp_path),
        "random_seed": 42,
        "datasets": {"anchor_merged": {"path": csv_path.name}},
    }

    df = load_anchor_merged(config)

    assert list(df["dataset"]) == ["hc3", "daigt_v2"]
    assert list(df["domain"]) == ["finance", "Distance learning"]
    assert list(df["label"].astype(int)) == [0, 1]
