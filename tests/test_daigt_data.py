from __future__ import annotations

import pandas as pd

from ai_text_detector.data import load_daigt_v2


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
