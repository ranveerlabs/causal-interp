"""task-agnostic corruption: substituting a uniformly drawn vocabulary token."""

from __future__ import annotations

import torch
from torch import Tensor

SEED_OFFSET = 777


def random_vocab_corruption(
    clean_tokens: Tensor,
    lengths: Tensor,
    d_vocab: int,
    seed: int,
    anchor: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """replace one token per prompt with a uniformly drawn vocabulary entry."""
    generator = torch.Generator().manual_seed(seed + SEED_OFFSET)
    tokens = clean_tokens.clone()
    n = tokens.shape[0]
    indices = torch.zeros(n, dtype=torch.long, device=tokens.device)

    for i in range(n):
        if anchor is not None:
            index = int(anchor[i])
        else:
            index = int(torch.randint(1, int(lengths[i]), (1,), generator=generator))
        original = int(tokens[i, index])
        replacement = original
        while replacement == original:
            replacement = int(torch.randint(0, d_vocab, (1,), generator=generator))
        tokens[i, index] = replacement
        indices[i] = index

    return tokens, indices
