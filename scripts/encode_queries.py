#!/usr/bin/env python3
"""Precompute CLIP text embeddings for evaluation/queries.txt (same raw float32 format and
manifest schema as the image embeddings). Benchmarks read this file, so the C++ benchmark
path never touches Python."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.clip_model import encode_texts, l2_normalize, load_clip  # noqa: E402
from sir.config import ROOT, load_config  # noqa: E402
from sir.embio import sha256_file, write_manifest  # noqa: E402

cfg = load_config()
qpath = ROOT / cfg["paths"]["queries"]
queries = [l for l in qpath.read_text(encoding="utf-8").split("\n") if l]
model, _, tok, dev = load_clip(cfg)
embs = [encode_texts(model, tok, queries[i:i + 256], dev) for i in range(0, len(queries), 256)]
Q = l2_normalize(np.concatenate(embs))
out = ROOT / cfg["paths"]["query_embeddings"]
Q.astype("<f4").tofile(out)
m = write_manifest(out, ROOT / cfg["paths"]["query_manifest"], model=cfg["model"]["name"],
                   dim=Q.shape[1], rows=Q.shape[0], normalized=True,
                   extra={"pretrained": cfg["model"]["pretrained"],
                          "queries_file": cfg["paths"]["queries"],
                          "queries_sha256": sha256_file(qpath), "device": dev})
print(f"wrote {Q.shape[0]} x {Q.shape[1]} query embeddings, sha256={m['sha256'][:16]}...")
