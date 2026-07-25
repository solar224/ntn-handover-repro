from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("configs/paper_result.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(config_path.parent.parent.resolve())
    if overrides:
        cfg = merge_dicts(cfg, overrides)
    return cfg


def merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def project_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg.get("_root", ".")).resolve()


def artifact_dir(cfg: dict[str, Any]) -> Path:
    path = project_root(cfg) / cfg["project"]["artifact_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(cfg: dict[str, Any]) -> Path:
    data_dir = project_root(cfg) / cfg["data"]["dataset_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / cfg["data"]["dataset_file"]


def metadata_path(cfg: dict[str, Any]) -> Path:
    data_dir = project_root(cfg) / cfg["data"]["dataset_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / cfg["data"]["metadata_file"]


def checkpoint_dir(cfg: dict[str, Any]) -> Path:
    path = artifact_dir(cfg) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_dir(cfg: dict[str, Any]) -> Path:
    path = artifact_dir(cfg) / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(cfg: dict[str, Any]) -> Path:
    path = artifact_dir(cfg) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path

