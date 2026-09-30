#!/usr/bin/env python3
"""CLIP image embeddings for every valid image, in internal-ID order, resumable.

Row i of image_embeddings.bin is internal ID i (metadata.json order). Rows are written
sequentially to <bin>.partial; every `checkpoint_every_batches` batches the data file is
fsync'ed and then a checkpoint JSON (rows_done) is atomically replaced. On restart the
partial file is truncated back to the checkpointed row count and generation continues
from there. Batches are always aligned to multiples of batch_size from row 0, so a
resumed run processes exactly the same batches as an uninterrupted one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.clip_model import encode_images, l2_normalize, load_clip  # noqa: E402
from sir.config import ROOT, load_config  # noqa: E402
from sir.embio import sha256_file, write_manifest  # noqa: E402
from sir.preprocess import preprocess_file  # noqa: E402


class ImageSet(Dataset):
    def __init__(self, paths: list[Path], size: int):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(preprocess_file(self.paths[i], self.size))


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ecfg = cfg["embedding"]
    seed = ecfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    meta_path = ROOT / cfg["paths"]["metadata"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    records = meta["records"]
    n = len(records)
    assert all(r["internal_id"] == i for i, r in enumerate(records)), "metadata not in ID order"
    paths = [ROOT / r["path"] for r in records]

    dim = cfg["model"]["embedding_dim"]
    out_bin = ROOT / cfg["paths"]["embeddings"]
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    partial = out_bin.with_suffix(".bin.partial")
    ckpt_path = out_bin.with_suffix(".ckpt.json")
    meta_sha = sha256_file(meta_path)
    ident = {"model": cfg["model"]["name"], "pretrained": cfg["model"]["pretrained"],
             "dim": dim, "rows_total": n, "metadata_sha256": meta_sha,
             "batch_size": ecfg["batch_size"]}

    start = 0
    if ckpt_path.exists() and partial.exists():
        ck = json.loads(ckpt_path.read_text())
        if {k: ck.get(k) for k in ident} == ident:
            start = ck["rows_done"]
            with open(partial, "r+b") as f:
                f.truncate(start * dim * 4)
            print(f"RESUME from checkpoint: rows_done={start}/{n}", flush=True)
        else:
            print("checkpoint does not match current config/metadata; starting fresh")
    if start == 0:
        partial.write_bytes(b"")
        atomic_write_json(ckpt_path, {**ident, "rows_done": 0})

    model, _, _, device = load_clip(cfg)
    bs = ecfg["batch_size"]
    assert start % bs == 0
    loader = DataLoader(ImageSet(paths[start:], cfg["preprocess"]["image_size"]),
                        batch_size=bs, shuffle=False, num_workers=ecfg["num_workers"],
                        pin_memory=(device == "cuda"))
    rows_done = start
    t0 = time.time()
    with open(partial, "ab") as f:
        for b, batch in enumerate(loader, 1):
            emb = l2_normalize(encode_images(model, batch.numpy(), device))
            f.write(emb.astype("<f4", copy=False).tobytes(order="C"))
            rows_done += emb.shape[0]
            if b % ecfg["checkpoint_every_batches"] == 0 or rows_done == n:
                f.flush()
                os.fsync(f.fileno())
                atomic_write_json(ckpt_path, {**ident, "rows_done": rows_done})
                rate = (rows_done - start) / (time.time() - t0)
                print(f"  {rows_done}/{n} rows  ({rate:.0f} img/s)", flush=True)
    if rows_done != n:
        raise SystemExit(f"ERROR: wrote {rows_done} rows, expected {n}")
    os.replace(partial, out_bin)
    import open_clip
    m = write_manifest(out_bin, ROOT / cfg["paths"]["manifest"], model=cfg["model"]["name"],
                       dim=dim, rows=n, normalized=True, extra={
                           "pretrained": cfg["model"]["pretrained"],
                           "force_quick_gelu": cfg["model"]["pretrained"] == "openai",
                           "open_clip_version": open_clip.__version__,
                           "torch_version": torch.__version__, "device": device,
                           "metadata_sha256": meta_sha, "seed": seed,
                           "preprocessing": "OpenCV decode (IMREAD_UNCHANGED) -> RGB; "
                                            "Pillow-exact antialiased bicubic shorter-side "
                                            "resize to 224; centre crop 224; /255; CLIP "
                                            "mean/std normalisation"})
    ckpt_path.unlink()
    print(f"done: {n} rows in {time.time() - t0:.1f}s; sha256={m['sha256']}")


if __name__ == "__main__":
    main()
