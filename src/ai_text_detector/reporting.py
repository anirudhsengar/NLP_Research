from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_text_detector.schema import word_count


DEFAULT_VERSION_PACKAGES = [
    "datasets",
    "gradio",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "seaborn",
    "torch",
    "transformers",
]


def save_data_profile(frames: dict[str, pd.DataFrame], config: dict[str, Any]) -> Path:
    """Save dataset counts and distribution summaries for paper provenance."""
    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / "data_profile.json"
    payload = {
        "created_at_utc": _now(),
        "datasets": {name: _frame_profile(frame) for name, frame in frames.items()},
    }
    _write_json(output, payload)
    return output


def save_run_manifest(
    config: dict[str, Any],
    *,
    command: str,
    outputs: dict[str, Any] | None = None,
    model_path: str | Path | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a compact reproducibility manifest for a completed command."""
    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / "run_manifest.json"
    payload = {
        "created_at_utc": _now(),
        "command": command,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "package_versions": _package_versions(
            config.get("reporting", {}).get("version_packages", DEFAULT_VERSION_PACKAGES)
        ),
        "config": _public_config(config),
        "processed_files": _processed_file_manifest(config),
        "model_path": str(model_path) if model_path is not None else None,
        "model_metadata": model_metadata or {},
        "outputs": _stringify_paths(outputs or {}),
    }
    _write_json(output, payload)
    return output


def _frame_profile(df: pd.DataFrame) -> dict[str, Any]:
    words = df["text"].map(word_count) if "text" in df.columns else pd.Series(dtype=float)
    profile = {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "label_counts": _counts(df, "label"),
        "split_counts": _counts(df, "split"),
        "source_counts": _counts(df, "source", limit=40),
        "domain_counts": _counts(df, "domain", limit=40),
        "native_status_counts": _counts(df, "native_status"),
        "country_counts": _counts(df, "country", limit=80),
        "l1_counts": _counts(df, "l1", limit=80),
        "cefr_level_counts": _counts(df, "cefr_level"),
        "length_bin_counts": _counts(df, "length_bin"),
        "license_tag_counts": _counts(df, "license_tag", limit=20),
        "word_count": {
            "min": _safe_number(words.min()),
            "p25": _safe_number(words.quantile(0.25)) if not words.empty else None,
            "median": _safe_number(words.median()) if not words.empty else None,
            "p75": _safe_number(words.quantile(0.75)) if not words.empty else None,
            "max": _safe_number(words.max()),
        },
    }
    if {"split", "label"}.issubset(df.columns):
        profile["split_label_counts"] = _nested_counts(df, ["split", "label"])
    if {"native_status", "label"}.issubset(df.columns):
        profile["native_status_label_counts"] = _nested_counts(df, ["native_status", "label"])
    return profile


def _counts(df: pd.DataFrame, column: str, *, limit: int = 20) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].fillna("").astype(str).value_counts(dropna=False).head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def _nested_counts(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    counts = df.groupby(columns, dropna=False).size()
    return {" | ".join(str(part) for part in key): int(value) for key, value in counts.items()}


def _processed_file_manifest(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    processed_dir = Path(config["paths"]["processed_dir"])
    manifest = {}
    for path in sorted(processed_dir.glob("*.jsonl")):
        row_count = None
        try:
            row_count = sum(1 for _ in path.open("r", encoding="utf-8"))
        except OSError:
            pass
        manifest[path.name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "modified_at_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            "rows": row_count,
        }
    return manifest


def _package_versions(packages: list[str]) -> dict[str, str]:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _stringify_paths(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    return value


def _safe_number(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
