from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from ai_text_detector.icnale import load_icnale, parse_icnale_filename


def test_parse_icnale_we_filename():
    meta = parse_icnale_filename("ICNALE_WE_2.6/WE_0_Unclassified_Unmerged/WE_CHN_PTJ0_001_B1_1.txt")
    assert meta["module"] == "WE"
    assert meta["country_code"] == "CHN"
    assert meta["participant_code"] == "WE_CHN_001"
    assert meta["cefr_level_from_filename"] == "B1_1"


def test_parse_icnale_native_filename():
    meta = parse_icnale_filename("WE_ENS_PTJ0_001_XX_1.txt")
    assert meta["participant_code"] == "WE_ENS_001"
    assert meta["cefr_level_from_filename"] == "N/A"


def test_load_icnale_from_zip_without_survey(tmp_path: Path):
    zip_path = tmp_path / "mini_we.zip"
    with ZipFile(zip_path, "w") as z:
        z.writestr(
            "ICNALE_WE_2.6/WE_0_Unclassified_Unmerged/WE_ENS_PTJ0_001_XX_1.txt",
            "This is a native speaker essay. It has a few words.",
        )
        z.writestr(
            "ICNALE_WE_2.6/WE_0_Unclassified_Unmerged/WE_CHN_PTJ0_001_B1_1.txt",
            "This is a learner essay. It has a few words.",
        )

    config = {
        "paths": {"raw_dir": str(tmp_path)},
        "datasets": {
            "icnale": {
                "survey_xlsx": "missing.xlsx",
                "we_zip": "mini_we.zip",
                "include_wep": False,
                "include_weuae": False,
            }
        },
    }
    df = load_icnale(config)
    assert len(df) == 2
    assert set(df["native_status"]) == {"native", "learner"}
    assert set(df["label"]) == {0}


def test_load_icnale_from_extracted_dir_without_password(tmp_path: Path):
    we_dir = tmp_path / "ICNALE_WE_2.6" / "WE_0_Unclassified_Unmerged"
    we_dir.mkdir(parents=True)
    (we_dir / "WE_ENS_PTJ0_001_XX_1.txt").write_text(
        "This is a native speaker essay. It has a few words.",
        encoding="utf-8",
    )
    (we_dir / "WE_CHN_PTJ0_001_B1_1.txt").write_text(
        "This is a learner essay. It has a few words.",
        encoding="utf-8",
    )

    config = {
        "paths": {"raw_dir": str(tmp_path)},
        "datasets": {
            "icnale": {
                "survey_xlsx": "missing.xlsx",
                "we_dir": "ICNALE_WE_2.6",
                "we_zip": "missing.zip",
                "include_wep": False,
                "include_weuae": False,
            }
        },
    }
    df = load_icnale(config)
    assert len(df) == 2
    assert set(df["native_status"]) == {"native", "learner"}
