from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ai_text_detector.reporting import save_data_profile, save_run_manifest
from ai_text_detector.schema import ensure_canonical


def test_save_data_profile_records_core_counts(tmp_path: Path):
    frame = ensure_canonical(
        pd.DataFrame(
            [
                {
                    "sample_id": "h1",
                    "dataset": "fixture",
                    "split": "test",
                    "text": "Human text for profiling.",
                    "label": 0,
                    "source": "fixture",
                    "domain": "essay",
                    "group_id": "g1",
                    "native_status": "learner",
                },
                {
                    "sample_id": "a1",
                    "dataset": "fixture",
                    "split": "test",
                    "text": "Generated text for profiling.",
                    "label": 1,
                    "source": "fixture",
                    "domain": "essay",
                    "group_id": "g1",
                },
            ]
        )
    )
    config = {
        "paths": {
            "reports_dir": str(tmp_path),
            "processed_dir": str(tmp_path / "processed"),
        },
        "reporting": {"version_packages": ["pandas"]},
    }

    output = save_data_profile({"fixture": frame}, config)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["datasets"]["fixture"]["n_rows"] == 2
    assert payload["datasets"]["fixture"]["label_counts"] == {"0": 1, "1": 1}


def test_save_run_manifest_writes_versions_and_outputs(tmp_path: Path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "fixture.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    config = {
        "paths": {
            "reports_dir": str(tmp_path),
            "processed_dir": str(processed),
        },
        "reporting": {"version_packages": ["pandas"]},
    }

    output = save_run_manifest(config, command="aidetect fixture", outputs={"x": tmp_path / "x.txt"})
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["command"] == "aidetect fixture"
    assert payload["package_versions"]["pandas"] != "not-installed"
    assert payload["processed_files"]["fixture.jsonl"]["rows"] == 1
