"""Loading of config/domain.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "domain.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load domain.yaml and resolve relative paths against the repo root."""
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg["paths"].items():
        if key != "ecmwf_pattern":
            cfg["paths"][key] = str((REPO_ROOT / value).resolve())

    return cfg
