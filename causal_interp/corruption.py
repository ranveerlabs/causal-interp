"""Task-agnostic corruption: substituting a uniformly drawn vocabulary token.

Phase 5 introduced this as a method on `IOIDataset`, to measure what the
hand-built corruption schemes were actually buying. Phase 6 needs the identical
procedure on a second task, and rewriting it there would have reintroduced
exactly the hand-tuning the phase exists to detect — a "generic" corruption
written twice is two hand-built corruptions that happen to resemble each other.

So the code moved here unchanged and both tasks call it. The extraction is
verified bit-identical for IOI: the corrupted token tensors under both generic
schemes hash the same before and after the move.

Nothing in this module knows what any token means. It is handed a batch of clean
tokens and, optionally, one anchor index per prompt; everything else it draws
uniformly. That is the whole point — the position may be supplied (Phase 4 showed
positions are searchable rather than privileged knowledge) or drawn uniformly too,
in which case nothing about the task is supplied at all.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Offset applied to the caller's seed so the substitution stream is independent of
# whatever else that seed drives. Preserved from the original implementation
# because changing it would silently change every Phase 5 number.
SEED_OFFSET = 777


def random_vocab_corruption(
    clean_tokens: Tensor,
    lengths: Tensor,
    d_vocab: int,
    seed: int,
    anchor: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Replace one token per prompt with a uniformly drawn vocabulary entry.

    `anchor` gives the token index to corrupt in each prompt. When it is None the
    index is drawn uniformly from the real tokens instead — excluding index 0,
    which is the BOS and not part of the prompt. The final token is deliberately
    eligible: a corruption that supplies no task knowledge has no reason to know
    the last position is where the answer is read off, and protecting it would be
    smuggling that knowledge back in.

    The replacement is redrawn until it differs from the original, so every prompt
    is genuinely corrupted and a no-op draw cannot quietly weaken the measured
    effect.

    Returns the corrupted tokens and the index changed in each prompt.
    """
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
