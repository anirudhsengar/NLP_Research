from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ai_text_detector.anchor_study import (
    ANCHOR_EXPECTED_SPLIT_COUNTS,
    ANCHOR_TEST_SOURCES,
    ANCHOR_TRAIN_SOURCES,
    ANCHOR_VALIDATION_SOURCES,
    apply_anchor_source_split,
    make_alikhanov_baseline_pipeline,
)
from ai_text_detector.cli import app


ANCHOR_SOURCE_COUNTS = {
    'DAIGT_v2_"A Cowboy Who Rode the Waves"': 1896,
    "DAIGT_v2_Car-free cities": 4716,
    "DAIGT_v2_Cell phones at school": 2119,
    "DAIGT_v2_Community service": 2092,
    "DAIGT_v2_Distance learning": 5554,
    "DAIGT_v2_Does the electoral college work?": 4432,
    "DAIGT_v2_Driverless cars": 2250,
    "DAIGT_v2_Exploring Venus": 2176,
    "DAIGT_v2_Facial action coding system": 3084,
    "DAIGT_v2_Grades for extracurricular activities": 2116,
    "DAIGT_v2_Mandatory extracurricular activities": 3077,
    "DAIGT_v2_Phones and driving": 1582,
    "DAIGT_v2_Seeking multiple opinions": 5176,
    "DAIGT_v2_Summer projects": 2701,
    "DAIGT_v2_The Face on Mars": 1893,
    "HC3_finance": 8393,
    "HC3_medicine": 2551,
    "HC3_open_qa": 4689,
    "HC3_reddit_eli5": 62085,
    "HC3_wiki_csai": 1613,
}


def test_anchor_split_counts_and_no_source_overlap():
    sources = []
    for source, count in ANCHOR_SOURCE_COUNTS.items():
        sources.extend([source] * count)
    df = pd.DataFrame(
        {
            "text": ["sample text"] * len(sources),
            "label": [0] * len(sources),
            "source": sources,
        }
    )
    split = apply_anchor_source_split(df)
    assert split["split"].value_counts().to_dict() == ANCHOR_EXPECTED_SPLIT_COUNTS
    assert set(ANCHOR_TRAIN_SOURCES).isdisjoint(ANCHOR_VALIDATION_SOURCES)
    assert set(ANCHOR_TRAIN_SOURCES).isdisjoint(ANCHOR_TEST_SOURCES)
    assert set(ANCHOR_VALIDATION_SOURCES).isdisjoint(ANCHOR_TEST_SOURCES)


def test_alikhanov_baseline_pipeline_matches_paper_settings():
    pipeline = make_alikhanov_baseline_pipeline(random_seed=42)
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    assert tfidf.stop_words == "english"
    assert tfidf.ngram_range == (1, 2)
    assert clf.solver == "liblinear"
    assert clf.class_weight is None
    assert clf.random_state == 42


def test_anchor_study_cli_smoke_on_tiny_dataset(tmp_path: Path):
    anchor_csv = tmp_path / "anchor.csv"
    rows = []
    all_sources = ANCHOR_TRAIN_SOURCES + ANCHOR_VALIDATION_SOURCES + ANCHOR_TEST_SOURCES
    for source in all_sources:
        for idx in range(10):
            label = idx % 2
            token = "generated synthetic formal conclusion" if label else "human personal varied detail"
            rows.append({"text": f"{token} {source} {idx}", "label": label, "source": source})
    pd.DataFrame(rows).to_csv(anchor_csv, index=False)

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "default.yaml"
    config_path.write_text(
        f"""
random_seed: 42
paths:
  raw_dir: {tmp_path}
  processed_dir: {tmp_path / "processed"}
  artifacts_dir: {tmp_path / "artifacts"}
  reports_dir: {tmp_path / "reports"}
features:
  use_word_tfidf: true
  use_char_tfidf: true
  use_basic_stats: true
  word_ngram_range: [1, 2]
  char_ngram_range: [3, 4]
  max_word_features: 50
  max_char_features: 50
  min_df: 1
  lowercase: true
model:
  class_weight: balanced
  max_iter: 200
  target_fpr: 0.1
evaluation:
  batch_size: 100
  bootstrap_iterations: 0
  min_subgroup_size: 1
calibration:
  icnale_calibration_fraction: 0.3
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "anchor-study",
            "--config",
            str(config_path),
            "--anchor-data-path",
            str(anchor_csv),
            "--models",
            "m0",
            "--max-external-samples",
            "0",
            "--paired-bootstrap-iterations",
            "5",
            "--skip-leave-one-domain",
            "--no-verify-anchor-counts",
            "--permutation-sample-size",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "reports" / "results" / "anchor_study_summary.csv").exists()
    assert (tmp_path / "reports" / "results" / "anchor_grid_results.csv").exists()
