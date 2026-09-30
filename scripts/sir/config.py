"""Config loading. All paths in config.json are relative to the project root."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def project_path(rel: str) -> Path:
    return ROOT / rel
