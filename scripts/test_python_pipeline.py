#!/usr/bin/env python3
"""Robustness tests for the Python data path: bad files never crash validation and get a
specific reason; odd colour modes convert to RGB; the resampler is bit-exact vs Pillow."""
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import clean_caption, validate_file  # noqa: E402
from sir.preprocess import decode_rgb, preprocess_rgb, resize_pil_bicubic  # noqa: E402

rng = np.random.default_rng(0)
tmp = Path(tempfile.mkdtemp())
base = Image.fromarray(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8))


def save(name, img, fmt, **kw):
    p = tmp / name
    img.save(p, fmt, **kw)
    return p


def enc(img, fmt):
    b = io.BytesIO()
    img.save(b, fmt)
    return b.getvalue()


fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += not cond


# --- bad files -> specific reasons, no exceptions ----------------------------------
good = save("good.jpg", base, "JPEG")
(tmp / "empty.jpg").write_bytes(b"")
(tmp / "garbage.jpg").write_bytes(b"this is not an image at all" * 10)
(tmp / "trunc.jpg").write_bytes(good.read_bytes()[: len(good.read_bytes()) // 2])
gif = save("anim.gif", base.convert("P"), "GIF")
cases = {"good.jpg": "", "missing.jpg": "missing_file", "empty.jpg": "zero_bytes",
         "garbage.jpg": "corrupt", "trunc.jpg": "corrupt", "anim.gif": "unsupported_format"}
for name, want in cases.items():
    r = validate_file(str(tmp / name))
    ok = r["ok"] if want == "" else (not r["ok"] and r["reason"].startswith(want))
    check(ok, f"validate {name:12s} -> ok={r['ok']} reason={r['reason']!r}")

# --- odd modes -> 3-channel RGB -------------------------------------------------------
g = decode_rgb(enc(base.convert("L"), "PNG"))
check(g.shape == (60, 80, 3) and np.array_equal(g[..., 0], g[..., 2]), "grayscale -> RGB")
p = decode_rgb(enc(base.convert("P"), "PNG"))
check(np.array_equal(p, np.asarray(base.convert("P").convert("RGB"))), "palette -> RGB exact")
rgba = base.convert("RGBA")
a = np.asarray(rgba).copy()
a[..., 3] = 0
a[:10, :10, 3] = 255
out = decode_rgb(enc(Image.fromarray(a), "PNG"))
check(np.all(out[20:, 20:] == 255) and np.array_equal(out[:10, :10], a[:10, :10, :3]),
      "alpha composited over white")
cmyk = decode_rgb(enc(base.convert("CMYK"), "JPEG"))
ref = np.asarray(Image.open(io.BytesIO(enc(base.convert("CMYK"), "JPEG"))).convert("RGB"))
check(cmyk.shape == (60, 80, 3) and np.abs(cmyk.astype(int) - ref).mean() < 3,
      f"CMYK JPEG -> RGB (mean |diff| vs PIL {np.abs(cmyk.astype(int) - ref).mean():.2f})")
t = preprocess_rgb(g)
check(t.shape == (3, 224, 224) and t.dtype == np.float32, "tensor shape/dtype")

# --- resampler exactness (downscale + upscale + extreme aspect) ------------------------
bad = 0
for h, w in [(375, 500), (60, 80), (1000, 30), (224, 224), (500, 333), (7, 300)]:
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    nh, nw = (int(224 * h / w), 224) if w <= h else (224, int(224 * w / h))
    bad += not np.array_equal(resize_pil_bicubic(img, nw, nh),
                              np.asarray(Image.fromarray(img).resize((nw, nh), Image.BICUBIC)))
check(bad == 0, "Pillow-exact bicubic on 6 shapes")

# --- captions -----------------------------------------------------------------------
check(clean_caption("  Á  dog \t in\n snow ") == "Á dog in snow", "caption NFC+ws")
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
