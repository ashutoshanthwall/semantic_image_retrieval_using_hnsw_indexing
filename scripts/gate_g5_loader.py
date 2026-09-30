#!/usr/bin/env python3
"""G5: C++ EmbeddingStore reads back byte-identical vectors (max |diff| must be 0.0)."""
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.config import ROOT, load_config  # noqa: E402
from sir.embio import read_embeddings  # noqa: E402

cfg = load_config()
E = read_embeddings(ROOT / cfg["paths"]["embeddings"], cfg["model"]["embedding_dim"])
rows = sorted(random.Random(cfg["evaluation"]["seed"] + 5).sample(range(E.shape[0]), 5))
out = subprocess.run([str(ROOT / "build/sir_gate"), "dump-rows", *map(str, rows)], cwd=ROOT,
                     capture_output=True, text=True, check=True).stdout.splitlines()
print("== G5 C++ loader vs NumPy")
print("C++:", out[0])
ok = True
for line in out[1:]:
    parts = line.split()
    i = int(parts[0])
    cpp = np.array([float.fromhex(x) for x in parts[1:]], dtype=np.float32)
    d = float(np.abs(cpp - E[i]).max())
    same_bits = np.array_equal(cpp.view(np.uint32), E[i].view(np.uint32))
    ok &= d == 0.0 and same_bits
    print(f"row {i:6d}: max|diff| = {d!r}  bit-identical={same_bits}")
i = rows[0]
cpp0 = [float.fromhex(x) for x in out[1].split()[1:5]]
print(f"row {i} first 4 elements  C++  : {[f'{v:.9g}' for v in cpp0]}")
print(f"row {i} first 4 elements  NumPy: {[f'{v:.9g}' for v in E[i][:4].tolist()]}")
print("G5:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
