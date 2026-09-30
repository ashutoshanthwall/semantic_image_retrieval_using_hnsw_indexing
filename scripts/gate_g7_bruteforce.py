#!/usr/bin/env python3
"""G7: C++ brute force vs NumPy on the same embeddings and the same 50 query vectors, K=10.
Ids must match exactly (rank by rank); scores within 1e-5."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.config import ROOT, load_config  # noqa: E402
from sir.embio import read_embeddings  # noqa: E402

N_Q, K = 50, 10
cfg = load_config()
dim = cfg["model"]["embedding_dim"]
E = read_embeddings(ROOT / cfg["paths"]["embeddings"], dim)
Q = read_embeddings(ROOT / cfg["paths"]["query_embeddings"], dim)[:N_Q]
out = Path(tempfile.mkdtemp()) / "bf.txt"
subprocess.run([str(ROOT / "build/sir_gate"), "bf-topk", "--n", str(N_Q), "--k", str(K),
                "--out", str(out)], cwd=ROOT, check=True, capture_output=True)
cpp = {}
for line in out.read_text().splitlines():
    q, i, s = line.split()
    cpp.setdefault(int(q), []).append((int(i), float.fromhex(s)))

S = Q.astype(np.float64) @ E.astype(np.float64).T  # float64 reference scores
id_mismatch, max_dev, min_gap = 0, 0.0, np.inf
ids = np.arange(E.shape[0])
for q in range(N_Q):
    order = np.lexsort((ids, -S[q]))[:K + 1]
    ref_ids = order[:K].tolist()
    got_ids = [i for i, _ in cpp[q]]
    id_mismatch += sum(a != b for a, b in zip(ref_ids, got_ids)) + abs(len(got_ids) - K)
    for i, s in cpp[q]:
        max_dev = max(max_dev, abs(s - S[q, i]))
    min_gap = min(min_gap, float(np.min(-np.diff(S[q, order]))))
print("== G7 C++ brute force vs NumPy (50 queries, K=10)")
print(f"id mismatches (rank-by-rank): {id_mismatch}")
print(f"max |score deviation|: {max_dev:.3e}  (tolerance 1e-5)")
print(f"smallest adjacent-score gap in reference top-11: {min_gap:.3e}")
ok = id_mismatch == 0 and max_dev <= 1e-5
print("G7:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
