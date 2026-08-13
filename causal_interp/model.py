"""Model loading for interpretability runs."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


def best_device() -> str:
    """Return the device to run on, preferring the GPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_name: str = "gpt2-small", device: str | None = None) -> HookedTransformer:
    """Load a HookedTransformer with hooks available for intervention.

    Phase 1 targets `gpt2-small`, whose IOI circuit and induction heads are
    already documented and serve as ground truth for validating the system's
    autonomous conclusions.
    """
    return HookedTransformer.from_pretrained(model_name, device=device or best_device())
