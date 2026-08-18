"""An induced structure, wrapped as a task the existing pipeline can run — Phase 10.

`induction.py` turns example prompts into slots and proposed counterfactuals. This
module turns those into the two things the rest of the repository already knows how to
consume:

- an **`AutoDataset`**, satisfying the dataset contract all three hand-built task
  modules satisfy — `clean_tokens`, `corrupted_tokens`, `lengths`, `positions`,
  `logit_diff`, `__len__`;
- a **`TaskSpec`**, so `pipeline.discover()` sweeps it under every proposed scheme with
  no change to `pipeline.py`, `interventions.py`, `agreement.py` or `metrics.py`.

That is the whole design goal. Phases 6 and 7 measured their generality by how little
existing code they had to touch; Phase 10 is built to be measured the same way, and the
runner prints `git diff --stat` over the pre-existing modules for exactly that reason.

Two pieces here are Phase 10's own proposals rather than transcriptions of existing
practice, and both are fixed in sections 3.4 and 3.5 of `results/PHASE10_PLAN.md`:

**The primary scheme is chosen by measurement.** Each proposed counterfactual is scored
by the mean KL divergence between the clean and corrupted next-token distributions, and
the largest wins. No answer key is involved — only the two runs the counterfactual
already provides.

**The metric is `clean_argmax_logprob`**: the mean log-probability the model assigns, at
the final position, to whatever token it *itself* predicted on the clean prompt. Phase 5
established that a metric needs no answer key to locate a circuit; what it did not
supply is a scalar for a task that has no hand-written one. This is that scalar. It
measures restoration of the model's own clean behaviour, which is not the same thing as
restoration of the correct answer, and the gap between the two is measured once at the
end against the published task rather than assumed away.

**This module must never import a `ground_truth` module.** The Phase 10 runner asserts
it, for the reason `search.py` and `agreement.py` carry the same prohibition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Sequence

import torch
from torch import Tensor

from causal_interp.corruption import random_vocab_corruption
from causal_interp.induction import (
    END,
    FILTER_LENGTH,
    Generated,
    Proposal,
    Structure,
    apply_proposal,
    corrupted_round_trip_rate,
    generate,
    induce,
    propose,
)
from causal_interp.schemes import Scheme, TaskSpec

# The one generic scheme an induced task registers. `random_vocab_any` needs no anchor
# and therefore no knowledge of the task at all, which is why it is the one that
# transfers; `random_vocab_<pivot>` would need to be told where the pivot is, and on an
# induced task nothing knows that before the primary has been selected.
GENERIC_SCHEME = "random_vocab_any"


class AutoDataset:
    """A batch of induced clean/corrupted pairs, in the shape the pipeline expects.

    Mirrors `GreaterThanDataset` and `DocstringDataset` field for field. The clean rows
    are generated once by `build()` and handed in, so every scheme in a multi-scheme
    sweep sees the *identical* clean sample and only the counterfactual differs —
    `greater_than.py` arranges the same invariant by hand with a second RNG, and
    `scripts/check_schemes.py` exists because getting it wrong silently changes what a
    cross-scheme comparison is comparing.
    """

    def __init__(
        self,
        model: Any,
        *,
        structure: Structure,
        generated: Generated,
        proposals: Sequence[Proposal],
        n: int = 128,
        corruption: str = GENERIC_SCHEME,
        seed: int = 0,
    ) -> None:
        by_name = {p.name: p for p in proposals}
        if corruption != GENERIC_SCHEME and corruption not in by_name:
            raise ValueError(
                f"corruption must be {GENERIC_SCHEME} or one of {sorted(by_name)}, "
                f"got {corruption!r}"
            )
        if n != generated.count:
            raise ValueError(
                f"this task was built with {generated.count} generated prompts but was "
                f"asked for {n}. The clean sample is generated once so that every scheme "
                "sees the same prompts; rebuild the task to change n."
            )

        self.model = model
        self.structure = structure
        self.corruption = corruption
        self.seed = seed
        self.rows = generated.rows

        device = model.cfg.device
        self.clean_tokens = torch.tensor(list(self.rows), device=device)
        length = structure.length
        self.lengths = torch.full((len(self.rows),), length, dtype=torch.long, device=device)

        if corruption == GENERIC_SCHEME:
            self.corrupted_tokens, self.corrupted_indices = random_vocab_corruption(
                clean_tokens=self.clean_tokens,
                lengths=self.lengths,
                d_vocab=model.cfg.d_vocab,
                seed=seed,
            )
            self.corrupted_round_trip = float("nan")
        else:
            corrupted, changed = apply_proposal(by_name[corruption], structure, self.rows, seed)
            self.corrupted_tokens = torch.tensor(corrupted, device=device)
            self.corrupted_indices = torch.tensor(changed, device=device)
            # Reported, never filtered: a corrupted row has to stay aligned with the
            # clean row it is paired with, so it cannot be rejected and redrawn.
            self.corrupted_round_trip = corrupted_round_trip_rate(model, corrupted)

        if self.clean_tokens.shape != self.corrupted_tokens.shape:
            raise AssertionError(
                f"clean/corrupted shape mismatch: {tuple(self.clean_tokens.shape)} vs "
                f"{tuple(self.corrupted_tokens.shape)}"
            )

        self.positions = self._locate_positions(device)
        self.target_ids = self._clean_argmax()

    def __len__(self) -> int:
        return len(self.rows)

    # -- positions ----------------------------------------------------------

    def _locate_positions(self, device: str) -> dict[str, Tensor]:
        """Every induced position, plus END, as a (batch,) index tensor.

        Every generated prompt has the same length by construction, so each index is
        constant down the batch. The tensor shape is kept anyway, because that is the
        contract `interventions.py` indexes against and a task that returned a scalar
        here would work until something batched it.
        """
        n = len(self.rows)
        out: dict[str, Tensor] = {}
        for label in self.structure.positions:
            index = self.structure.position_index(label)
            out[label] = torch.full((n,), index, dtype=torch.long, device=device)
        out.setdefault(
            END, torch.full((n,), self.structure.length - 1, dtype=torch.long, device=device)
        )
        return out

    # -- metric -------------------------------------------------------------

    def _clean_argmax(self) -> Tensor:
        """The token the model itself predicts at END on each clean prompt.

        One forward pass at construction. This is the stand-in for the answer key: the
        induced task has no idea what the *right* continuation is, so it takes the
        model's own clean continuation as the thing an intervention is trying to
        restore.
        """
        with torch.no_grad():
            logits = self.model(self.clean_tokens)
        rows = torch.arange(len(self), device=logits.device)
        return logits[rows, self.positions[END]].argmax(dim=-1)

    def logit_diff(self, logits: Tensor, per_prompt: bool = False) -> Tensor:
        """`clean_argmax_logprob` — mean log p(the clean argmax) at END.

        Named `logit_diff` to satisfy the interface `interventions.py` calls, exactly as
        `greater_than.py` does for its probability difference. It is neither a logit nor
        a difference: it is a log-probability, and it is the largest it can be on the
        clean run by construction, which is what makes the clean-to-corrupted span
        positive without having to be told what the task is.
        """
        rows = torch.arange(len(self), device=logits.device)
        log_probs = logits[rows, self.positions[END]].log_softmax(dim=-1)
        values = log_probs[rows, self.target_ids]
        return values if per_prompt else values.mean()

    def auto_rank_stats(self, logits: Tensor) -> dict[str, float]:
        """Does the run still predict what the clean run predicted?

        The counterpart of `year_rank_stats` and `io_rank_stats`, and the only accuracy
        notion available without an answer key. `agrees_with_clean` is 1.0 on the clean
        run by definition; its value on the corrupted run is what says whether the
        counterfactual did anything.
        """
        rows = torch.arange(len(self), device=logits.device)
        final = logits[rows, self.positions[END]]
        probs = final.softmax(dim=-1)
        return {
            "agrees_with_clean": (final.argmax(dim=-1) == self.target_ids).float().mean().item(),
            "target_prob": probs[rows, self.target_ids].mean().item(),
        }


# ---------------------------------------------------------------------------
# Building the TaskSpec
# ---------------------------------------------------------------------------


def _kl_at_end(ds: AutoDataset, clean_logits: Tensor, corrupted_logits: Tensor) -> float:
    """Mean KL(clean || corrupted) over the full next-token distribution at END.

    The selection statistic of section 3.4. Computed here rather than imported from
    `metrics.py` only because `DistributionalBaseline` bundles it with a normalization
    this step does not want — the raw divergence is the quantity being ranked.
    """
    rows = torch.arange(len(ds), device=clean_logits.device)
    end = ds.positions[END]
    log_p = clean_logits[rows, end].log_softmax(dim=-1)
    log_q = corrupted_logits[rows, end].log_softmax(dim=-1)
    return float((log_p.exp() * (log_p - log_q)).sum(dim=-1).mean())


@dataclass
class AutoTask:
    """Everything `build()` produced, so a report can print it without re-deriving it."""

    name: str
    structure: Structure
    generated: Generated
    proposals: tuple[Proposal, ...]
    primary: str
    divergences: dict[str, float]
    task: TaskSpec
    corrupted_round_trip: dict[str, float] = field(default_factory=dict)
    accuracy: dict[str, dict] = field(default_factory=dict)

    def as_dict(self, decode: Any = None) -> dict:
        return {
            "name": self.name,
            "structure": self.structure.as_dict(decode),
            "generation": self.generated.as_dict(),
            "proposals": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "columns": list(p.columns),
                    "label": p.label,
                    "breaks": p.breaks,
                }
                for p in self.proposals
            ],
            "primary": self.primary,
            "divergences": self.divergences,
            "corrupted_round_trip": self.corrupted_round_trip,
            "accuracy": self.accuracy,
            "discovery_schemes": list(self.task.discovery_schemes),
        }


def select_primary(
    model: Any,
    structure: Structure,
    generated: Generated,
    proposals: Sequence[Proposal],
    *,
    seed: int,
) -> tuple[str, dict[str, float], dict[str, float], dict[str, dict]]:
    """Section 3.4: rank the proposed counterfactuals by measured divergence.

    Returns the winner, every candidate's divergence, every candidate's corrupted
    round-trip rate, and the clean/corrupted agreement statistics — the last three
    reported rather than used, so the selection rule stays a single argmax with nothing
    in it to tune.
    """
    divergences: dict[str, float] = {}
    round_trip: dict[str, float] = {}
    accuracy: dict[str, dict] = {}

    for proposal in proposals:
        ds = AutoDataset(
            model,
            structure=structure,
            generated=generated,
            proposals=proposals,
            n=generated.count,
            corruption=proposal.name,
            seed=seed,
        )
        with torch.no_grad():
            clean_logits = model(ds.clean_tokens)
            corrupted_logits = model(ds.corrupted_tokens)
        divergences[proposal.name] = _kl_at_end(ds, clean_logits, corrupted_logits)
        round_trip[proposal.name] = ds.corrupted_round_trip
        accuracy[proposal.name] = {
            "clean": ds.auto_rank_stats(clean_logits),
            "corrupted": ds.auto_rank_stats(corrupted_logits),
        }

    # argmax, ties broken by lowest anchor column — fixed in the plan so that a tie
    # cannot be resolved by whichever scheme happens to score better afterwards.
    order = {p.name: p.columns[0] for p in proposals}
    primary = min(divergences, key=lambda name: (-divergences[name], order[name]))
    return primary, divergences, round_trip, accuracy


def build(
    model: Any,
    examples: Sequence[str],
    *,
    name: str,
    n: int = 128,
    seed: int = 0,
    model_alias: str = "gpt2-small",
    filter_mode: str = FILTER_LENGTH,
) -> AutoTask:
    """Induce, generate, propose, select — the whole of section 3, end to end.

    The returned `AutoTask.task` is a `TaskSpec` that `pipeline.discover()` accepts
    without knowing it was not written by hand.

    `filter_mode` defaults to the pre-registered rule; the amendment's repair has to be
    named explicitly by the caller.
    """
    structure = induce(model, examples, filter_mode=filter_mode)
    generated = generate(model, structure, n=n, seed=seed)
    if generated.count < 2:
        raise ValueError(
            f"{name}: generation produced {generated.count} distinct prompts from "
            f"{len(structure.slots)} slots; there is no dataset here to run on"
        )

    proposals = propose(structure)
    primary, divergences, round_trip, accuracy = select_primary(
        model, structure, generated, proposals, seed=seed
    )

    schemes: dict[str, Scheme] = {
        p.name: Scheme(
            name=p.name,
            # `authored`, not `generic`: these encode no knowledge of what the task
            # means, but they do encode the human's examples, and Phase 8's provenance
            # field exists precisely so that distinction is not quietly lost.
            provenance="authored",
            breaks=p.breaks,
            # Not knowable without the answer key. Declared False for every induced
            # scheme and reported as unavailable — a real loss against Phase 8's
            # registry, recorded rather than papered over.
            preserves_answer=False,
            primary=(p.name == primary),
        )
        for p in proposals
    }
    schemes[GENERIC_SCHEME] = Scheme(
        name=GENERIC_SCHEME,
        provenance="generic",
        breaks="substitutes a uniformly drawn vocabulary token anywhere in the prompt",
        preserves_answer=False,
    )

    spec = TaskSpec(
        name=name,
        dataset=partial(
            AutoDataset, structure=structure, generated=generated, proposals=proposals
        ),
        positions=structure.positions,
        schemes=schemes,
        discovery_schemes=tuple(p.name for p in proposals) + (GENERIC_SCHEME,),
        metric_label="log p(the model's own clean argmax) at END",
        model_alias=model_alias,
    )

    return AutoTask(
        name=name,
        structure=structure,
        generated=generated,
        proposals=proposals,
        primary=primary,
        divergences=divergences,
        task=spec,
        corrupted_round_trip=round_trip,
        accuracy=accuracy,
    )
