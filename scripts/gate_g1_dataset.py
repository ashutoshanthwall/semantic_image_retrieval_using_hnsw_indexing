#!/usr/bin/env python3
"""G1: dataset counts + 5 random images with captions for manual inspection."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.config import ROOT, load_config  # noqa: E402

cfg = load_config()
rep = json.loads((ROOT / "data/processed/dataset_report.json").read_text())
meta = json.loads((ROOT / cfg["paths"]["metadata"]).read_text(encoding="utf-8"))
print("== G1 dataset")
for k in ["total_found", "total_valid", "total_excluded", "sum_check", "annotation_rows",
          "image_files_on_disk", "captions_total", "captions_truncated_at_77_tokens",
          "exclusion_reasons", "splits"]:
    print(f"{k}: {rep[k]}")
unreadable = sum(v for k, v in rep["exclusion_reasons"].items()
                 if k.startswith(("corrupt", "unreadable", "opencv", "zero_", "unsupported")))
print(f"unreadable/corrupt files: {unreadable}")
rng = random.Random(cfg["evaluation"]["seed"])
for r in rng.sample(meta["records"], 5):
    print(f"\n[{r['internal_id']}] {r['path']} ({r['width']}x{r['height']})")
    for c in r["captions"]:
        print(f"   - {c}")
