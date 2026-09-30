#!/usr/bin/env python3
"""Comparability check for the absolute-quality track (section 9).

Published zero-shot Flickr30k retrieval numbers use the Karpathy 1K test protocol: gallery
= the 1000 test-split images, queries = all 5 captions of each (5000 queries). The main
benchmark instead searches the full 31k gallery with one held-out caption per image, which
is a much harder task, so its R@1 is not directly comparable. This script evaluates the
standard protocol with exact (NumPy, float64) search over our own embeddings."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.clip_model import encode_texts, l2_normalize, load_clip  # noqa: E402
from sir.config import ROOT, load_config  # noqa: E402
from sir.embio import read_embeddings  # noqa: E402

cfg = load_config()
meta = json.loads((ROOT / cfg["paths"]["metadata"]).read_text(encoding="utf-8"))
E = read_embeddings(ROOT / cfg["paths"]["embeddings"], cfg["model"]["embedding_dim"])
test = [r for r in meta["records"] if r["split"] == "test"]
gallery = E[[r["internal_id"] for r in test]].astype(np.float64)
caps, owner = [], []
for g, r in enumerate(test):
    for c in r["captions"]:
        caps.append(c)
        owner.append(g)
owner = np.array(owner)
model, _, tok, dev = load_clip(cfg)
Q = l2_normalize(np.concatenate([encode_texts(model, tok, caps[i:i + 500], dev)
                                 for i in range(0, len(caps), 500)])).astype(np.float64)
S = Q @ gallery.T
# rank of the true image (ties broken by ascending gallery index, like the C++ engine)
true = S[np.arange(len(caps)), owner]
rank = (S > true[:, None]).sum(1) + ((S == true[:, None]) & (np.arange(len(test))[None, :]
                                                                < owner[:, None])).sum(1)
out = {f"R@{k}": float((rank < k).mean()) for k in (1, 5, 10)}
print(f"== Standard Flickr30k 1K test protocol: {len(test)} images, {len(caps)} captions, "
      f"exact search")
print("text->image " + "  ".join(f"{k}={v:.4f}" for k, v in out.items()))
(ROOT / "evaluation/results/standard_1k_protocol.json").write_text(
    json.dumps({"gallery_images": len(test), "queries": len(caps), **out}, indent=2) + "\n")
