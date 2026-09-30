#!/usr/bin/env python3
"""Flickr30k ingestion, validation, cleaning and metadata generation.

Steps
  1. Extract flickr30k-images.zip into data/raw/flickr30k/ (once), then make data/raw
     read-only so nothing downstream can modify it.
  2. Build the candidate list: every annotation-CSV row, plus every image file on disk
     that has no CSV row. Each candidate ends up either valid or excluded, exactly once.
  3. Validate every image file (missing, zero bytes, unsupported format, undecodable,
     truncated/corrupt, zero dimensions) in parallel.
  4. Duplicates: duplicate image IDs (keep first CSV occurrence) and duplicate file
     content by SHA-256 (keep the lowest image ID in sorted order).
  5. Captions: NFC, whitespace collapse, strip, drop empties; zero captions => excluded.
     Per-caption flag for CLIP's 77-token truncation (BOS/EOS included).
  6. Assign contiguous internal IDs 0..N-1 to valid images in sorted image-ID order.

Outputs: data/processed/metadata.json, data/processed/exclusion_report.csv,
         data/processed/dataset_report.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import sys
import unicodedata
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.config import ROOT, load_config  # noqa: E402
from sir.preprocess import ImageDecodeError, decode_rgb  # noqa: E402

SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP", "TIFF"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WS = re.compile(r"\s+")


def extract_if_needed(cfg: dict) -> None:
    raw = ROOT / cfg["paths"]["raw_dir"]
    images = ROOT / cfg["paths"]["images_dir"]
    marker = raw / ".extracted"
    if marker.exists():
        return
    zpath = raw / "flickr30k-images.zip"
    print(f"extracting {zpath} ...", flush=True)
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"zip CRC failure at member {bad}")
        members = [m for m in z.namelist() if not m.startswith("__MACOSX/")]
        z.extractall(raw, members=members)
    if not images.is_dir():
        raise RuntimeError(f"expected {images} after extraction")
    marker.write_text("ok\n")
    make_read_only(raw)


def make_read_only(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        for name in dirnames:
            p = Path(dirpath) / name
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    root.chmod(root.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def clean_caption(text: str) -> str:
    return WS.sub(" ", unicodedata.normalize("NFC", str(text))).strip()


def validate_file(path_str: str) -> dict:
    """Return {ok, reason, sha256, width, height, format}. Never raises."""
    p = Path(path_str)
    out = {"ok": False, "reason": "", "sha256": "", "width": 0, "height": 0, "format": ""}
    try:
        if not p.exists():
            out["reason"] = "missing_file"
            return out
        data = p.read_bytes()
        if len(data) == 0:
            out["reason"] = "zero_bytes"
            return out
        out["sha256"] = hashlib.sha256(data).hexdigest()
        try:
            with Image.open(io.BytesIO(data)) as im:
                out["format"] = im.format or ""
                if im.format not in SUPPORTED_FORMATS:
                    out["reason"] = f"unsupported_format_{im.format}"
                    return out
                im.load()  # full decode: catches truncated / corrupt streams
                w, h = im.size
        except Exception as e:  # PIL raises many types for corrupt data
            out["reason"] = f"corrupt:{type(e).__name__}"
            return out
        if w == 0 or h == 0:
            out["reason"] = "zero_dimensions"
            return out
        try:
            rgb = decode_rgb(data)  # the decoder the embedding pipeline uses
        except ImageDecodeError as e:
            out["reason"] = f"opencv_{e}"
            return out
        if rgb.shape[:2] != (h, w):
            out["reason"] = f"decoder_size_mismatch_pil{w}x{h}_cv{rgb.shape[1]}x{rgb.shape[0]}"
            return out
        out.update(ok=True, width=w, height=h)
        return out
    except Exception as e:  # last-resort guard: a bad image must never crash the run
        out["reason"] = f"unreadable:{type(e).__name__}"
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    extract_if_needed(cfg)

    images_dir = ROOT / cfg["paths"]["images_dir"]
    df = pd.read_csv(ROOT / cfg["paths"]["annotations_csv"], dtype=str, keep_default_na=False)
    print(f"annotation rows: {len(df)}", flush=True)

    # --- candidates -------------------------------------------------------------
    candidates = []  # dicts: image_id, filename, captions_raw, split, source
    seen_ids: dict[str, int] = {}
    exclusions = []  # (image_id, path, reason)
    for row in df.itertuples(index=False):
        fname = row.filename.strip()
        image_id = Path(fname).stem
        try:
            caps = json.loads(row.raw)
        except json.JSONDecodeError:
            caps = []
        cand = {"image_id": image_id, "filename": fname, "captions_raw": caps,
                "split": row.split, "source": "csv"}
        if image_id in seen_ids:
            exclusions.append((image_id, f"{cfg['paths']['images_dir']}/{fname}",
                               f"duplicate_of={image_id}"))
            continue
        seen_ids[image_id] = len(candidates)
        candidates.append(cand)
    n_dup_id = len(exclusions)

    on_disk = sorted(p.name for p in images_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    other_files = sorted(p.name for p in images_dir.iterdir()
                         if p.is_file() and p.suffix.lower() not in IMAGE_EXTS)
    for fname in on_disk:
        if Path(fname).stem not in seen_ids:
            candidates.append({"image_id": Path(fname).stem, "filename": fname,
                               "captions_raw": [], "split": "", "source": "disk_only"})
            seen_ids[Path(fname).stem] = len(candidates) - 1
    total_found = len(candidates) + n_dup_id
    print(f"image files on disk: {len(on_disk)} (non-image files ignored: {len(other_files)})")
    print(f"candidates: {total_found} (duplicate-ID CSV rows: {n_dup_id})", flush=True)

    # --- validation (parallel) --------------------------------------------------
    paths = [str(images_dir / c["filename"]) for c in candidates]
    with Pool(args.workers) as pool:
        results = pool.map(validate_file, paths, chunksize=64)

    # --- captions + truncation ----------------------------------------------------
    import open_clip  # imported late: only needed for the tokenizer
    tok = open_clip.get_tokenizer(cfg["model"]["name"])
    ctx = cfg["model"]["context_length"]

    valid = []
    for c, r in zip(candidates, results):
        rel = f"{cfg['paths']['images_dir']}/{c['filename']}"
        if not r["ok"]:
            exclusions.append((c["image_id"], rel, r["reason"]))
            continue
        caps = [clean_caption(t) for t in c["captions_raw"]]
        caps = [t for t in caps if t]
        if not caps:
            exclusions.append((c["image_id"], rel, "no_captions"))
            continue
        # tok.encode applies the same cleaning as tokenisation; +2 for BOS/EOS
        trunc = [len(tok.encode(t)) + 2 > ctx for t in caps]
        valid.append({**c, "path": rel, "captions": caps, "caption_truncated": trunc,
                      "width": r["width"], "height": r["height"], "sha256": r["sha256"],
                      "format": r["format"]})

    # --- content duplicates: keep lowest image_id in sorted order -------------------
    valid.sort(key=lambda v: v["image_id"])
    first_by_hash: dict[str, str] = {}
    kept = []
    for v in valid:
        h = v["sha256"]
        if h in first_by_hash:
            exclusions.append((v["image_id"], v["path"], f"duplicate_of={first_by_hash[h]}"))
            continue
        first_by_hash[h] = v["image_id"]
        kept.append(v)

    records = []
    for i, v in enumerate(kept):
        records.append({"internal_id": i, "image_id": v["image_id"], "path": v["path"],
                        "width": v["width"], "height": v["height"], "split": v["split"],
                        "format": v["format"], "sha256": v["sha256"],
                        "captions": v["captions"], "caption_truncated": v["caption_truncated"]})

    # --- outputs -----------------------------------------------------------------
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": "flickr30k (HF nlphuji/flickr30k)", "count": len(records),
            "id_order": "sorted image_id (lexicographic string order)",
            "records": records}
    (ROOT / cfg["paths"]["metadata"]).write_text(json.dumps(meta, ensure_ascii=False),
                                                 encoding="utf-8")
    exclusions.sort(key=lambda e: (e[0], e[2]))
    with open(ROOT / cfg["paths"]["exclusion_report"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "path", "reason"])
        w.writerows(exclusions)

    n_caps = sum(len(r["captions"]) for r in records)
    n_trunc = sum(sum(r["caption_truncated"]) for r in records)
    reasons = Counter(e[2].split("=")[0] if e[2].startswith("duplicate_of") else e[2]
                      for e in exclusions)
    report = {"total_found": total_found, "total_valid": len(records),
              "total_excluded": len(exclusions),
              "sum_check": len(records) + len(exclusions) == total_found,
              "annotation_rows": len(df), "image_files_on_disk": len(on_disk),
              "captions_total": n_caps, "captions_truncated_at_77_tokens": n_trunc,
              "exclusion_reasons": dict(reasons),
              "splits": dict(Counter(r["split"] for r in records))}
    (proc / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["sum_check"]:
        raise SystemExit("ERROR: found != valid + excluded")


if __name__ == "__main__":
    main()
