#!/usr/bin/env python3
"""G2: OpenCV preprocessing vs the reference open_clip processor on 10 random images.
Gates: max |tensor diff| < 1e-4 and embedding cosine >= 0.9999, per image."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.clip_model import encode_images, l2_normalize, load_clip  # noqa: E402
from sir.config import ROOT, load_config  # noqa: E402
from sir.preprocess import preprocess_file  # noqa: E402

cfg = load_config()
meta = json.loads((ROOT / cfg["paths"]["metadata"]).read_text(encoding="utf-8"))
model, ref_pre, _, dev = load_clip(cfg)
rng = random.Random(cfg["evaluation"]["seed"] + 2)
ok = True
print("== G2 preprocessing fidelity (ours = OpenCV decode + Pillow-exact resample)")
print(f"{'internal_id':>11} {'size':>9}  {'max_abs_diff':>12}  {'cosine':>10}")
for r in rng.sample(meta["records"], 10):
    p = ROOT / r["path"]
    ours = preprocess_file(p)
    with Image.open(p) as im:
        ref = ref_pre(im).numpy()
    d = float(np.abs(ours - ref).max())
    e = l2_normalize(encode_images(model, np.stack([ours, ref]), dev))
    cos = float(np.dot(e[0].astype(np.float64), e[1].astype(np.float64)))
    passed = d < 1e-4 and cos >= 0.9999
    ok &= passed
    print(f"{r['internal_id']:>11} {r['width']}x{r['height']:<5} {d:12.3e}  {cos:10.7f}  "
          f"{'PASS' if passed else 'FAIL'}")
print("G2:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
