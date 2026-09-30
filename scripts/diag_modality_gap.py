#!/usr/bin/env python3
"""Diagnostic (analysis only, not a search feature): how far text queries sit from the
image embeddings. Compares each text query's best image similarity with image-to-nearest-
image similarity. Text queries lying far from the data manifold make graph search harder
than the synthetic in-distribution tests suggest."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.config import ROOT, load_config  # noqa: E402
from sir.embio import read_embeddings  # noqa: E402

cfg = load_config()
E = read_embeddings(ROOT / cfg["paths"]["embeddings"], 512).astype(np.float64)
Q = read_embeddings(ROOT / cfg["paths"]["query_embeddings"], 512).astype(np.float64)
top = (Q @ E.T).max(1)
idx = np.random.default_rng(0).choice(len(E), 1000, replace=False)
S = E[idx] @ E.T
S[np.arange(1000), idx] = -np.inf
nn = S.max(1)
print(f"text query -> best image cosine ({len(Q)} queries): median {np.median(top):.3f}, "
      f"p5 {np.percentile(top, 5):.3f}, p95 {np.percentile(top, 95):.3f}")
print(f"image -> nearest other image cosine (1000 sampled): median {np.median(nn):.3f}, "
      f"p5 {np.percentile(nn, 5):.3f}, p95 {np.percentile(nn, 95):.3f}")
