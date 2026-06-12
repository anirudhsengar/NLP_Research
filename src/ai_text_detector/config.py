from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML config and resolve project-relative paths."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    project_root = path.resolve().parent.parent if path.parent.name == "configs" else Path.cwd()
    config["_project_root"] = str(project_root)

    paths = config.setdefault("paths", {})
    for key in ("raw_dir", "processed_dir", "artifacts_dir", "reports_dir"):
        value = Path(paths.get(key, key))
        if not value.is_absolute():
            value = project_root / value
        paths[key] = str(value)

    return config


def resolve_raw_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["paths"]["raw_dir"]) / path


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ("processed_dir", "artifacts_dir", "reports_dir"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    (Path(config["paths"]["artifacts_dir"]) / "models").mkdir(parents=True, exist_ok=True)
    (Path(config["paths"]["reports_dir"]) / "results").mkdir(parents=True, exist_ok=True)
    (Path(config["paths"]["reports_dir"]) / "figures").mkdir(parents=True, exist_ok=True)

