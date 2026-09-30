#!/usr/bin/env bash
# End-to-end reproduction from a clean checkout. Each step is a gate; set -e stops on the
# first failure. Expect roughly 20-30 minutes on a laptop with a CUDA GPU (CPU-only CLIP
# inference works but the embedding step takes much longer).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PIN_CPU=${PIN_CPU:-4}   # a performance core; check `lscpu -e` on hybrid CPUs

[ -x "$PY" ] || python3.11 -m venv .venv
$PY -m pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

scripts/download_dataset.sh                      # verified by SHA-256
$PY scripts/prepare_dataset.py                   # extract, validate, clean, metadata
$PY scripts/test_python_pipeline.py              # bad-data / odd-mode / resampler tests
$PY scripts/gate_g1_dataset.py                   # G1 (then inspect the 5 printed images)
$PY scripts/gate_g2_preprocess.py                # G2
$PY scripts/generate_embeddings.py               # resumable: rerun after a crash
$PY scripts/gate_g3_embeddings.py                # G3
$PY scripts/gate_g4_semantic.py                  # G4 (then inspect the printed images)
$PY scripts/make_queries.py
$PY scripts/encode_queries.py

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure        # G6, G8 + all unit tests
$PY scripts/gate_g5_loader.py                    # G5
$PY scripts/gate_g7_bruteforce.py                # G7
build/sir_build_index                            # G9 (builds data/index/hnsw.idx)

TMP=$(mktemp -d)                                 # G10
build/sir_gate persist-save --index "$TMP/g10.idx" --out "$TMP/expected.txt"
build/sir_gate persist-check --index "$TMP/g10.idx" --expect "$TMP/expected.txt"
$PY - "$TMP/g10.idx" <<'PYEOF'
import sys; p = sys.argv[1]; b = bytearray(open(p, "rb").read()); b[len(b) // 2] ^= 0xFF
open(p, "wb").write(b)
PYEOF
if build/sir_gate load-index --index "$TMP/g10.idx"; then echo "G10 FAIL: corrupt index loaded"; exit 1; fi

build/sir_benchmark --pin-cpu "$PIN_CPU" --measure-encoder
$PY scripts/plot_benchmark.py
$PY scripts/eval_standard_1k.py
$PY scripts/diag_modality_gap.py
build/sir_cli --k 5 a dog playing in snow         # G11
