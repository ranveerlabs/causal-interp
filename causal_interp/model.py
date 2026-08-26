"""Model loading for interpretability runs."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


def best_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_name: str = "gpt2-small", device: str | None = None) -> HookedTransformer:
    return HookedTransformer.from_pretrained(model_name, device=device or best_device())
