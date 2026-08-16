"""Scoring a patched run without knowing what the right answer is.

Phases 1-4 all scored recovery as a logit difference between two specific tokens:
the indirect object and the subject. That quantity cannot be built without already
knowing which token is correct, which is exactly the dependency Phase 5 is
auditing.

The metrics here need only the clean and corrupted runs. They compare the *whole*
next-token distribution rather than two hand-picked entries, so nothing about the
task's answer enters. Both are defined in `results/PHASE5_AUDIT.md`, written
before either was run.

All three metrics share the same normalization, so their numbers sit on one scale:

    recovery = 1 - D(clean, patched) / D(clean, corrupted)

    0.0   the patched run is as far from clean as the corrupted run is
    1.0   the patched run has closed the whole gap

Being distribution-wide is the point and also the risk: these metrics see every
way the two runs differ, not only the part that implements the behaviour. Whether
that dilutes the signal is an empirical question, and Phase 5 answers it rather
than assuming either way.
"""

from __future__ import annotations

import torch
from torch import Tensor

from causal_interp.ioi import IOIDataset

METRICS = ("logit_diff", "kl", "tv")


def final_log_probs(ds: IOIDataset, logits: Tensor) -> Tensor:
    """Log next-token distribution at each prompt's final position, (batch, d_vocab)."""
    rows = torch.arange(len(ds), device=logits.device)
    return logits[rows, ds.positions["END"]].log_softmax(dim=-1)


def kl_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    """D_KL(P || Q) per prompt, computed in log space so no probability underflows."""
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def total_variation(log_p: Tensor, log_q: Tensor) -> Tensor:
    """½ Σ |P − Q| per prompt. Bounded in [0, 1], unlike KL."""
    return 0.5 * (log_p.exp() - log_q.exp()).abs().sum(dim=-1)


class DistributionalBaseline:
    """The clean and corrupted reference distributions a patched run is scored against.

    Holds the clean log-probabilities and the clean-to-corrupted divergence that
    every later measurement is normalized by, so those are computed once.
    """

    def __init__(self, ds: IOIDataset, clean_logits: Tensor, corrupted_logits: Tensor) -> None:
        self.ds = ds
        self.clean = final_log_probs(ds, clean_logits)
        corrupted = final_log_probs(ds, corrupted_logits)
        self.span = {
            "kl": kl_divergence(self.clean, corrupted).mean().item(),
            "tv": total_variation(self.clean, corrupted).mean().item(),
        }
        for name, value in self.span.items():
            if value <= 0:
                raise ValueError(
                    f"{name} divergence between the clean and corrupted runs is {value}; "
                    "the corruption changes nothing measurable and cannot be normalized against"
                )

    def recovery(self, patched_logits: Tensor) -> dict[str, float]:
        """Normalized recovery under each distributional metric."""
        patched = final_log_probs(self.ds, patched_logits)
        return {
            "kl": 1.0 - kl_divergence(self.clean, patched).mean().item() / self.span["kl"],
            "tv": 1.0 - total_variation(self.clean, patched).mean().item() / self.span["tv"],
        }


def all_metrics(
    ds: IOIDataset, patched_logits: Tensor, logit_baseline, distributional: DistributionalBaseline
) -> dict[str, float]:
    """Score one patched run under all three metrics at once.

    Computing them from the same forward pass is what makes the comparison fair:
    the three numbers describe the identical intervention, so any difference
    between them is the metric and not the run.
    """
    scores = distributional.recovery(patched_logits)
    scores["logit_diff"] = logit_baseline.normalize(ds.logit_diff(patched_logits).item())
    return scores
