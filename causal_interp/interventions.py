"""Intervention primitives — the causal core of the system.

Tracing and visualization show what correlates with a behaviour. These primitives
exist to answer the stronger question: if this component is changed, does the
behaviour change? Every claim the system emits must be backed by one of these.

Phase 1 is not yet built; these are the intended entry points.
"""

from __future__ import annotations

from transformer_lens import HookedTransformer


def activation_patch(model: HookedTransformer, clean_prompt: str, corrupted_prompt: str, node: str):
    """Patch one node's activation from a clean run into a corrupted run.

    The effect on the output logits is the causal effect attributed to `node`.
    """
    raise NotImplementedError("Phase 1")


def ablate(model: HookedTransformer, prompt: str, node: str):
    """Knock out a node and measure the resulting change in behaviour."""
    raise NotImplementedError("Phase 1")


def prune_circuit(model: HookedTransformer, task, threshold: float):
    """Iteratively remove components that carry no causal weight for `task`.

    What survives pruning is the candidate circuit: a minimal subgraph that
    still reproduces the behaviour.
    """
    raise NotImplementedError("Phase 1")
