"""Raw float32 embedding files + JSON manifests (shared by image and query embeddings)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .config import ROOT


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def write_manifest(bin_path: Path, manifest_path: Path, *, model: str, dim: int, rows: int,
                   normalized: bool, extra: dict | None = None) -> dict:
    size = bin_path.stat().st_size
    if size != rows * dim * 4:
        raise RuntimeError(f"{bin_path}: size {size} != rows*dim*4 = {rows * dim * 4}")
    m = {
        "model": model,
        "dim": dim,
        "rows": rows,
        "l2_normalized": normalized,
        "dtype": "float32",
        "byte_order": "little",
        "layout": "row-major [rows x dim], no header",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "sha256": sha256_file(bin_path),
    }
    if extra:
        m.update(extra)
    manifest_path.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    return m


def read_embeddings(bin_path: Path, dim: int) -> np.ndarray:
    a = np.fromfile(bin_path, dtype="<f4")
    if a.size % dim:
        raise RuntimeError(f"{bin_path}: {a.size} floats is not a multiple of dim {dim}")
    return a.reshape(-1, dim)
