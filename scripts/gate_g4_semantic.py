#!/usr/bin/env python3
"""G4: semantic sanity — CLIP text queries vs the full embedding matrix (NumPy)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.clip_model import encode_texts, l2_normalize, load_clip  # noqa: E402
from sir.config import ROOT, load_config  # noqa: E402
from sir.embio import read_embeddings  # noqa: E402

QUERIES = ["a dog playing in snow", "two men playing guitars on a stage",
           "a child on a red bicycle", "people eating at an outdoor restaurant",
           "a surfer riding a large wave"]
cfg = load_config()
meta = json.loads((ROOT / cfg["paths"]["metadata"]).read_text(encoding="utf-8"))
E = read_embeddings(ROOT / cfg["paths"]["embeddings"], cfg["model"]["embedding_dim"])
model, _, tok, dev = load_clip(cfg)
Q = l2_normalize(encode_texts(model, tok, QUERIES, dev))
S = Q.astype(np.float64) @ E.astype(np.float64).T
print("== G4 semantic sanity (NumPy, float64 scores)")
for q, s in zip(QUERIES, S):
    top = np.lexsort((np.arange(len(s)), -s))[:5]
    print(f"\nquery: {q}")
    for rank, i in enumerate(top, 1):
        r = meta["records"][i]
        print(f"  {rank}. {s[i]:.4f}  {r['path']}   | {r['captions'][0]}")
