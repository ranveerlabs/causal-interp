"""scoring a patched run without knowing what the right answer is."""

from __future__ import annotations

import torch
from torch import Tensor

from causal_interp.ioi import IOIDataset

METRICS = ("logit_diff", "kl", "tv")


def final_log_probs(ds: IOIDataset, logits: Tensor) -> Tensor:
    rows = torch.arange(len(ds), device=logits.device)
    return logits[rows, ds.positions["END"]].log_softmax(dim=-1)


def kl_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def total_variation(log_p: Tensor, log_q: Tensor) -> Tensor:
    return 0.5 * (log_p.exp() - log_q.exp()).abs().sum(dim=-1)


class DistributionalBaseline:
    """The clean and corrupted reference distributions a patched run is scored against."""

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
        patched = final_log_probs(self.ds, patched_logits)
        return {
            "kl": 1.0 - kl_divergence(self.clean, patched).mean().item() / self.span["kl"],
            "tv": 1.0 - total_variation(self.clean, patched).mean().item() / self.span["tv"],
        }


def all_metrics(
    ds: IOIDataset, patched_logits: Tensor, logit_baseline, distributional: DistributionalBaseline
) -> dict[str, float]:
    scores = distributional.recovery(patched_logits)
    scores["logit_diff"] = logit_baseline.normalize(ds.logit_diff(patched_logits).item())
    return scores
