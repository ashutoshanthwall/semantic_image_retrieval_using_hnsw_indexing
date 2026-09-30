#!/usr/bin/env python3
"""G3: embedding file integrity + row-order check by re-embedding 5 random rows."""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.clip_model import encode_images, l2_normalize, load_clip  # noqa: E402
from sir.config import ROOT, load_config  # noqa: E402
from sir.embio import read_embeddings, sha256_file  # noqa: E402
from sir.preprocess import preprocess_file  # noqa: E402

cfg = load_config()
dim = cfg["model"]["embedding_dim"]
meta = json.loads((ROOT / cfg["paths"]["metadata"]).read_text(encoding="utf-8"))
man = json.loads((ROOT / cfg["paths"]["manifest"]).read_text())
E = read_embeddings(ROOT / cfg["paths"]["embeddings"], dim)
norms = np.linalg.norm(E.astype(np.float64), axis=1)
checks = {
    "rows == valid images": (E.shape[0], meta["count"], E.shape[0] == meta["count"]),
    "dim == config dim": (E.shape[1], dim, E.shape[1] == dim),
    "manifest rows/dim": ((man["rows"], man["dim"]), (E.shape[0], dim),
                          (man["rows"], man["dim"]) == (E.shape[0], dim)),
    "manifest sha256": (man["sha256"][:16], "", sha256_file(ROOT / cfg["paths"]["embeddings"])
                        == man["sha256"]),
    "NaN count": (int(np.isnan(E).sum()), 0, not np.isnan(E).any()),
    "Inf count": (int(np.isinf(E).sum()), 0, not np.isinf(E).any()),
    "zero-norm rows": (int((norms == 0).sum()), 0, not (norms == 0).any()),
    "norms within 1 +- 1e-5": (f"min={norms.min():.8f} max={norms.max():.8f} "
                               f"mean={norms.mean():.8f}", "",
                               bool(np.all(np.abs(norms - 1) <= 1e-5))),
}
print("== G3 embeddings")
ok = True
for k, (got, exp, passed) in checks.items():
    ok &= bool(passed)
    print(f"{k:28s} {str(got):50s} {'PASS' if passed else 'FAIL'}")
model, _, _, dev = load_clip(cfg)
rng = random.Random(cfg["evaluation"]["seed"] + 3)
print("row-order check (re-embed from scratch, batch of 1):")
for i in sorted(rng.sample(range(E.shape[0]), 5)):
    r = meta["records"][i]
    e = l2_normalize(encode_images(model, preprocess_file(ROOT / r["path"])[None], dev))[0]
    cos = float(np.dot(e.astype(np.float64), E[i].astype(np.float64)))
    passed = cos >= 0.9999
    ok &= passed
    print(f"  row {i:6d} image_id={r['image_id']:>12s} cosine={cos:.7f} "
          f"{'PASS' if passed else 'FAIL'}")
print("G3:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
