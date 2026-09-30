"""CLIP model loading. The OpenAI ViT-B/32 weights were trained with QuickGELU; open_clip
>= 3 builds 'ViT-B-32' with standard GELU unless forced, which silently changes the
embeddings. force_quick_gelu=True restores the architecture the weights expect."""
from __future__ import annotations

import numpy as np
import open_clip
import torch


def load_clip(cfg: dict, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name, pretrained = cfg["model"]["name"], cfg["model"]["pretrained"]
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, force_quick_gelu=(pretrained == "openai"))
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(name)
    dim = model.visual.output_dim
    if dim != cfg["model"]["embedding_dim"]:
        raise RuntimeError(f"model output dim {dim} != config embedding_dim "
                           f"{cfg['model']['embedding_dim']}")
    return model, preprocess, tokenizer, device


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalise rows in float64, return float32."""
    x64 = x.astype(np.float64)
    n = np.linalg.norm(x64, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero-norm embedding row(s): "
                         f"{np.nonzero(n[:, 0] == 0)[0][:10].tolist()}")
    return (x64 / n).astype(np.float32)


@torch.no_grad()
def encode_images(model, batch: np.ndarray, device: str) -> np.ndarray:
    t = torch.from_numpy(batch).to(device)
    return model.encode_image(t).float().cpu().numpy()


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], device: str) -> np.ndarray:
    tokens = tokenizer(texts).to(device)
    return model.encode_text(tokens).float().cpu().numpy()
