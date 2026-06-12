from __future__ import annotations

import os
import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from ai_text_detector.config import resolve_raw_path
from ai_text_detector.schema import LABEL_HUMAN, ensure_canonical


TEXT_SUFFIX = ".txt"


def load_icnale(config: dict, *, include_optional: bool | None = None) -> pd.DataFrame:
    """Load ICNALE human essays for native-vs-learner false-positive auditing."""
    icnale_cfg = config["datasets"]["icnale"]
    survey = _load_survey(resolve_raw_path(config, icnale_cfg["survey_xlsx"]))
    password = _zip_password(icnale_cfg)

    data_specs: list[tuple[str, Path | None, Path | None]] = [
        (
            "WE",
            resolve_raw_path(config, icnale_cfg.get("we_dir")),
            resolve_raw_path(config, icnale_cfg.get("we_zip")),
        )
    ]
    include_wep = include_optional if include_optional is not None else bool(icnale_cfg.get("include_wep", True))
    if include_wep and (icnale_cfg.get("wep_dir") or icnale_cfg.get("wep_zip")):
        data_specs.append(
            (
                "WEP",
                resolve_raw_path(config, icnale_cfg.get("wep_dir")),
                resolve_raw_path(config, icnale_cfg.get("wep_zip")),
            )
        )
    include_weuae = include_optional if include_optional is not None else bool(icnale_cfg.get("include_weuae", True))
    if include_weuae and (icnale_cfg.get("weuae_dir") or icnale_cfg.get("weuae_zip")):
        data_specs.append(
            (
                "WEUAE",
                resolve_raw_path(config, icnale_cfg.get("weuae_dir")),
                resolve_raw_path(config, icnale_cfg.get("weuae_zip")),
            )
        )

    frames = []
    for expected_module, dir_path, zip_path in data_specs:
        if dir_path is not None and dir_path.exists():
            frames.append(_load_text_dir(dir_path, expected_module=expected_module, survey=survey))
        elif zip_path is not None and zip_path.exists():
            frames.append(_load_text_zip(zip_path, expected_module=expected_module, survey=survey, password=password))

    if not frames:
        raise FileNotFoundError("No ICNALE essay directories or archives were found from config paths.")
    return ensure_canonical(pd.concat(frames, ignore_index=True).drop_duplicates("sample_id"))


def parse_icnale_filename(filename: str, expected_module: str | None = None) -> dict[str, str]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected ICNALE filename: {filename}")

    if parts[0] == "W" and len(parts) >= 6 and parts[1] == "UAE":
        module = "WEUAE"
        country = "UAE"
        prompt = parts[2]
        participant_no = parts[3]
        cefr_level = "_".join(parts[4:6])
        participant_code = f"WEUAE_UAE_{participant_no}"
    else:
        module = parts[0]
        country = parts[1]
        prompt = parts[2]
        participant_no = parts[3]
        cefr_level = "_".join(parts[4:6]) if len(parts) >= 6 else parts[4]
        participant_code = f"{module}_{country}_{participant_no}"

    if expected_module and module != expected_module:
        module = expected_module

    if cefr_level.startswith("XX"):
        cefr_level = "N/A"

    return {
        "module": module,
        "country_code": country,
        "prompt": prompt,
        "participant_no": participant_no,
        "participant_code": participant_code,
        "cefr_level_from_filename": cefr_level,
    }


def _load_survey(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="Participants", dtype=str).fillna("N/A")
    keep = [
        "Module",
        "Code",
        "Country",
        "L1",
        "Sex",
        "Age",
        "ENS Type",
        "Academic Genre",
        "CEFR Level",
        "Vocabulary Size Test Score",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = "N/A"
    return df[keep].rename(
        columns={
            "Module": "survey_module",
            "Code": "participant_code",
            "Country": "country",
            "L1": "l1",
            "CEFR Level": "cefr_level",
            "ENS Type": "ens_type",
            "Academic Genre": "academic_genre",
            "Vocabulary Size Test Score": "vocabulary_size_score",
        }
    )


def _load_text_dir(dir_path: Path, *, expected_module: str, survey: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path in dir_path.rglob("*.txt"):
        member = path.relative_to(dir_path).as_posix()
        if not _should_read_member(member, expected_module):
            continue
        try:
            meta = parse_icnale_filename(member, expected_module=expected_module)
        except ValueError:
            continue
        rows.append(_essay_row(member, _decode_text(path.read_bytes()), meta, expected_module))
    return _finalize_essay_rows(rows, expected_module=expected_module, survey=survey)


def _load_text_zip(
    zip_path: Path,
    *,
    expected_module: str,
    survey: pd.DataFrame,
    password: bytes | None = None,
) -> pd.DataFrame:
    rows = []
    with ZipFile(zip_path) as z:
        encrypted_members = [
            info.filename
            for info in z.infolist()
            if info.filename.lower().endswith(TEXT_SUFFIX) and info.flag_bits & 0x1
        ]
        if encrypted_members and password is None:
            raise RuntimeError(
                f"{zip_path.name} is encrypted. Set ICNALE_ZIP_PASSWORD before running ICNALE preparation."
            )
        for member in z.namelist():
            if not _should_read_member(member, expected_module):
                continue
            try:
                meta = parse_icnale_filename(member, expected_module=expected_module)
            except ValueError:
                continue
            rows.append(_essay_row(member, _decode_text(z.read(member, pwd=password)), meta, expected_module))
    return _finalize_essay_rows(rows, expected_module=expected_module, survey=survey)


def _finalize_essay_rows(rows: list[dict], *, expected_module: str, survey: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(rows).drop_duplicates("sample_id")
    if df.empty:
        return ensure_canonical(df)

    if not survey.empty and expected_module in {"WE", "WEP"}:
        df = df.merge(survey, on="participant_code", how="left")
    else:
        df["country"] = df["country_code"]
        df["l1"] = "N/A"
        df["cefr_level"] = df["cefr_level_from_filename"]

    df["country"] = df["country"].fillna(df["country_code"]).map(_clean_cell)
    df["l1"] = df["l1"].fillna("N/A").map(_clean_cell)
    df["cefr_level"] = df["cefr_level"].fillna(df["cefr_level_from_filename"]).map(_clean_cell)
    df["native_status"] = df.apply(_native_status, axis=1)
    return ensure_canonical(df)


def _essay_row(member: str, text: str, meta: dict[str, str], expected_module: str) -> dict:
    return {
        **meta,
        "sample_id": f"icnale:{Path(member).stem}",
        "dataset": "icnale",
        "split": "fairness",
        "text": text,
        "label": LABEL_HUMAN,
        "source": expected_module,
        "domain": meta["prompt"],
        "group_id": meta["participant_code"],
        "license_tag": "ICNALE local use only",
    }


def _should_read_member(member: str, expected_module: str) -> bool:
    lower = member.lower()
    if not lower.endswith(TEXT_SUFFIX) or lower.endswith(":zone.identifier"):
        return False
    if expected_module in {"WE", "WEP"}:
        return "unclassified_unmerged" in lower
    if expected_module == "WEUAE":
        return "/unmerged/" in lower or lower.startswith("unmerged/")
    return "unmerged" in lower and "merged" not in lower.replace("unmerged", "")


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding).replace("\ufeff", "").strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").strip()


def _zip_password(icnale_cfg: dict) -> bytes | None:
    env_name = icnale_cfg.get("password_env", "ICNALE_ZIP_PASSWORD")
    password = os.environ.get(env_name, "")
    if not password:
        return None
    return password.encode("utf-8")


def _clean_cell(value: object) -> str:
    text = str(value).strip()
    return re.sub(r"\s+", " ", text) if text else "N/A"


def _native_status(row: pd.Series) -> str:
    country = str(row.get("country", "")).strip()
    l1 = str(row.get("l1", "")).strip().lower()
    participant_code = str(row.get("participant_code", ""))
    if country.startswith("ENS") or l1 == "english" or "_ENS_" in participant_code:
        return "native"
    return "learner"
