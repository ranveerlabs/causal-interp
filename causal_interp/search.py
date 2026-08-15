"""Searching for receiver specifications instead of being told them.

Phases 1-3 asked about receiver inputs the paper had already named. This module
searches for them: given no account of the mechanism, which of a head's inputs,
at which token position, carries information that matters?

**This module must never import `causal_interp.ground_truth`.** The search has to
be able to run on a circuit with no published answer, and a search that consults
the answer key is not a search. The comparison against the published circuit
happens afterwards, in the Phase 4 script, on the output this module produces.

Two stages, sized by measured cost (see `results/PHASE4_SEARCH_SPACE.md`):

- `screen_specs` scores every receiver specification with one forward pass, so the
  whole grid is affordable and is searched exhaustively.
- `confirm_spec` sweeps all 144 heads as senders into one specification, which
  costs about 18 seconds and so runs only on the specifications that survive the
  screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

from causal_interp.interventions import (
    Baseline,
    Patch,
    Receiver,
    patch_effect,
    path_patch,
    path_signal,
)
from causal_interp.ioi import IOIDataset

RECEIVER_INPUTS = ("q", "k", "v")


@dataclass(frozen=True)
class ReceiverSpec:
    """One point in the search space: a head input, at a token position."""

    layer: int
    head: int
    input: str
    position: str

    def as_patch(self) -> Patch:
        return Patch(layer=self.layer, kind=self.input, position=self.position, head=self.head)

    def as_receiver(self) -> Receiver:
        return Receiver(layer=self.layer, head=self.head, position=self.position, input=self.input)

    def __str__(self) -> str:
        return f"{self.layer}.{self.head}.{self.input}@{self.position}"


def absolute_positions(ds: IOIDataset) -> list[str]:
    """Register one position name per token index and return the names.

    Only valid when every prompt has the same length, which is why the
    absolute-position search runs on a single template: otherwise index *k* would
    refer to a different word in every row and the search would be averaging over
    incomparable things.

    The names are deliberately opaque (`t0`, `t1`, ...). Nothing about the task's
    structure — which token is the indirect object, where the subject repeats — is
    available to a search that uses these.
    """
    lengths = ds.lengths
    if int(lengths.min()) != int(lengths.max()):
        raise ValueError(
            f"absolute positions need a fixed prompt length; got {int(lengths.min())}"
            f"-{int(lengths.max())}. Build the dataset with a single template."
        )
    length = int(lengths.min())
    device = ds.clean_tokens.device
    names = []
    for index in range(length):
        name = f"t{index}"
        ds.positions[name] = torch.full((len(ds),), index, dtype=torch.long, device=device)
        names.append(name)
    return names


def screen_specs(
    model: HookedTransformer,
    ds: IOIDataset,
    clean_cache: ActivationCache,
    baseline: Baseline,
    positions: Sequence[str],
    inputs: Sequence[str] = RECEIVER_INPUTS,
    progress: Callable[[int, int], None] | None = None,
) -> dict[ReceiverSpec, float]:
    """Stage A: score every receiver specification, one forward pass each.

    The score is the normalized recovery of the logit difference when that single
    input, at that single position, is spliced from the clean run into the
    corrupted one. It asks whether the information arriving on that wire is enough
    to move the behaviour.

    This is a logit-effect screen and inherits the blind spot Phase 3 documented:
    a receiver whose incoming information matters to the mechanism but not to the
    output will score low here and never reach stage B.
    """
    out: dict[ReceiverSpec, float] = {}
    total = model.cfg.n_layers * model.cfg.n_heads * len(inputs) * len(positions)
    done = 0
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            for kind in inputs:
                for position in positions:
                    spec = ReceiverSpec(layer, head, kind, position)
                    out[spec] = patch_effect(model, ds, clean_cache, [spec.as_patch()], baseline)
                    done += 1
                    if progress is not None:
                        progress(done, total)
    return out


def rank_specs_for_head(
    scores: dict[ReceiverSpec, float], layer: int, head: int
) -> list[tuple[ReceiverSpec, float]]:
    """Every specification for one head, best first by absolute score."""
    subset = [(s, v) for s, v in scores.items() if s.layer == layer and s.head == head]
    return sorted(subset, key=lambda pair: abs(pair[1]), reverse=True)


def confirm_spec(
    model: HookedTransformer,
    ds: IOIDataset,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    baseline: Baseline,
    spec: ReceiverSpec,
    signal_threshold: float,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Stage B: sweep every head as a sender into `spec`.

    Scores each sender with both criteria the project already has — delivery to
    the receiver (`path_signal`, against Phase 3's recorded threshold) and effect
    on the output logits — and reports them separately, because Phase 3 showed
    they disagree.
    """
    receivers = [spec.as_receiver()]
    signals: dict[tuple[int, int], float] = {}
    effects: dict[tuple[int, int], float] = {}
    total = spec.layer * model.cfg.n_heads
    done = 0

    for layer in range(spec.layer):  # a sender must sit strictly below its receiver
        for head in range(model.cfg.n_heads):
            sender = Patch(layer=layer, kind="z", position=spec.position, head=head)
            value = path_signal(model, ds, clean_cache, corrupted_cache, sender, receivers)
            if value == value:  # not NaN: clean and corrupted differ at the receiver
                signals[(layer, head)] = value
                effects[(layer, head)] = path_patch(
                    model, ds, clean_cache, corrupted_cache, baseline, sender, receivers
                )
            done += 1
            if progress is not None:
                progress(done, total)

    return {
        "spec": str(spec),
        "senders_tested": len(signals),
        "signals": {f"{l}.{h}": v for (l, h), v in signals.items()},
        "effects": {f"{l}.{h}": v for (l, h), v in effects.items()},
        "senders_clearing_signal": sorted(
            f"{l}.{h}" for (l, h), v in signals.items() if abs(v) >= signal_threshold
        ),
        "_signals": signals,
        "_effects": effects,
    }
