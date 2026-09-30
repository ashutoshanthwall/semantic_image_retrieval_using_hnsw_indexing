"""CLIP image preprocessing: OpenCV decode + a bit-exact re-implementation of the
reference resampler.

The reference CLIP processor (open_clip / torchvision on PIL images) does:
  Resize(224, bicubic)  -> PIL Image.resize, which is an *antialiased* separable
                           bicubic (a = -0.5) evaluated in 22-bit fixed point with a
                           uint8 round-trip between the horizontal and vertical pass
  CenterCrop(224)       -> offsets computed with Python's round() (half-to-even)
  ToTensor              -> float32 / 255
  Normalize(mean, std)

cv2.resize(INTER_CUBIC) does not antialias when downscaling and differs from the
reference by several uint8 levels (~0.05 after normalisation), so it cannot meet the
1e-4 fidelity gate. Instead the resampling below reproduces Pillow's algorithm
(libImaging/Resample.c: precompute_coeffs, normalize_coeffs_8bpc, clip8) exactly.
Decoding and colour conversion are done with OpenCV.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
PRECISION_BITS = 32 - 8 - 2  # Pillow's fixed-point precision for 8-bit images


class ImageDecodeError(Exception):
    pass


# --------------------------------------------------------------------------- decode
def decode_rgb(data: bytes | np.ndarray) -> np.ndarray:
    """Decode an encoded image to HxWx3 uint8 RGB.

    Grayscale, palette, CMYK and alpha images are converted to 3-channel RGB; alpha is
    composited over white. IMREAD_UNCHANGED is used so that EXIF orientation is *not*
    applied, matching PIL.Image.open (which the reference processor uses).
    """
    buf = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray)) else data
    if buf.size == 0:
        raise ImageDecodeError("zero_bytes")
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageDecodeError("undecodable")
    if img.dtype == np.uint16:  # 16-bit PNG/TIFF: scale to 8 bits
        img = np.round(img.astype(np.float64) / 257.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ImageDecodeError(f"unsupported_dtype_{img.dtype}")
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img.reshape(img.shape[0], img.shape[1]), cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:
        img = composite_over_white(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    else:
        raise ImageDecodeError(f"unsupported_channels_{img.shape[2]}")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ImageDecodeError("zero_dimensions")
    return np.ascontiguousarray(img)


def composite_over_white(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float64)
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    out = rgb * a + 255.0 * (1.0 - a)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------- resample
def _bicubic(x: float) -> float:
    # Pillow's bicubic_filter with a = -0.5; operation order kept identical.
    a = -0.5
    if x < 0.0:
        x = -x
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


@lru_cache(maxsize=4096)
def _coeffs(in_size: int, out_size: int):
    """Pillow precompute_coeffs + normalize_coeffs_8bpc, computed in IEEE double with the
    same operation order as the C code so the integer weights are identical."""
    scale = filterscale = float(in_size) / out_size
    if filterscale < 1.0:
        filterscale = 1.0
    support = 2.0 * filterscale  # bicubic support = 2
    ksize = int(math.ceil(support)) * 2 + 1
    starts = np.zeros(out_size, dtype=np.int64)
    weights = np.zeros((out_size, ksize), dtype=np.int64)
    one = float(1 << PRECISION_BITS)
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        ss = 1.0 / filterscale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        k = []
        ww = 0.0
        for x in range(xmax):  # sequential sum, as in C
            w = _bicubic((x + xmin - center + 0.5) * ss)
            k.append(w)
            ww += w
        for x in range(xmax):
            kx = k[x] / ww if ww != 0.0 else k[x]
            weights[xx, x] = int(-0.5 + kx * one) if kx < 0 else int(0.5 + kx * one)
        starts[xx] = xmin
    idx = starts[:, None] + np.arange(ksize)[None, :]
    idx = np.minimum(idx, in_size - 1)  # taps past xmax have weight 0
    return idx, weights


def _resample_axis(img: np.ndarray, axis: int, out_size: int) -> np.ndarray:
    in_size = img.shape[axis]
    idx, w = _coeffs(in_size, out_size)
    src = img.astype(np.int64)
    if axis == 1:
        g = src[:, idx, :]                       # H, out, k, C
        acc = (g * w[None, :, :, None]).sum(axis=2)
    else:
        g = src[idx, :, :]                       # out, k, W, C
        acc = (g * w[:, :, None, None]).sum(axis=1)
    acc += 1 << (PRECISION_BITS - 1)
    return np.clip(acc >> PRECISION_BITS, 0, 255).astype(np.uint8)  # == Pillow clip8


def resize_pil_bicubic(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Bit-exact equivalent of PIL Image.resize((out_w, out_h), BICUBIC) for uint8 RGB."""
    h, w = img.shape[:2]
    out = img
    if out_w != w:
        out = _resample_axis(out, 1, out_w)
    if out_h != h:
        out = _resample_axis(out, 0, out_h)
    return out if out is not img else img.copy()


def shorter_side_size(h: int, w: int, target: int) -> tuple[int, int]:
    """torchvision Resize(int) output size: shorter side -> target, longer side truncated."""
    if w <= h:
        return int(target * h / w), target  # (new_h, new_w)
    return target, int(target * w / h)


# --------------------------------------------------------------------------- tensor
def preprocess_rgb(rgb: np.ndarray, size: int = 224,
                   mean: np.ndarray = MEAN, std: np.ndarray = STD) -> np.ndarray:
    """HxWx3 uint8 RGB -> 3xSxS float32 normalised CLIP input tensor."""
    h, w = rgb.shape[:2]
    new_h, new_w = shorter_side_size(h, w, size)
    img = resize_pil_bicubic(rgb, new_w, new_h)
    top = int(round((new_h - size) / 2.0))    # Python round == torchvision center_crop
    left = int(round((new_w - size) / 2.0))
    img = img[top:top + size, left:left + size]
    x = img.astype(np.float32) / np.float32(255.0)
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess_file(path, size: int = 224) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    return preprocess_rgb(decode_rgb(data), size)
