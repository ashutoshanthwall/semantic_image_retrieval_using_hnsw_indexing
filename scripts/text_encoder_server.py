#!/usr/bin/env python3
"""Persistent CLIP text encoder for the interactive C++ CLI.

Protocol (all integers little-endian uint32), over stdin/stdout:
  startup     server -> client : dim
  request     client -> server : n, then n bytes of UTF-8 query text   (n == 0: shut down)
  response ok server -> client : 4*dim, then dim float32 (raw CLIP embedding, NOT normalised;
                                 the C++ SearchEngine validates and normalises)
  response err                 : 0x80000000 | m, then m bytes of UTF-8 error message
stdout carries only protocol frames; every log line goes to stderr.
"""
import struct
import sys
from pathlib import Path

proto_in, proto_out = sys.stdin.buffer, sys.stdout.buffer
sys.stdout = sys.stderr  # nothing else may write to the protocol stream

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from sir.clip_model import encode_texts, load_clip  # noqa: E402
from sir.config import load_config  # noqa: E402

ERR = 0x80000000


def read_exact(n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = proto_in.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def send(header: int, payload: bytes) -> None:
    proto_out.write(struct.pack("<I", header) + payload)
    proto_out.flush()


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    torch.set_num_threads(4)
    model, _, tok, dev = load_clip(cfg, device="cpu")
    dim = cfg["model"]["embedding_dim"]
    send(dim, b"")
    print(f"[text_encoder_server] ready (dim={dim}, device={dev})", file=sys.stderr, flush=True)
    while True:
        hdr = read_exact(4)
        if hdr is None:
            return
        (n,) = struct.unpack("<I", hdr)
        if n == 0:
            return
        raw = read_exact(n)
        if raw is None:
            return
        try:
            text = raw.decode("utf-8")
            if not text.strip():
                raise ValueError("empty query")
            e = encode_texts(model, tok, [text], dev)[0].astype("<f4")
            if e.shape[0] != dim:
                raise ValueError(f"encoder produced dim {e.shape[0]}, expected {dim}")
            send(4 * dim, e.tobytes())
        except Exception as ex:  # report to the client instead of dying
            msg = f"{type(ex).__name__}: {ex}".encode("utf-8")
            send(ERR | len(msg), msg)


if __name__ == "__main__":
    main()
