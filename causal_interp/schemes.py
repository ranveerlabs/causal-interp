"""Counterfactual schemes, registered per task — Phase 8's structural change.

Every phase before this one treated **one** counterfactual as canonical per task and
ran everything downstream against it. Phase 7 showed what that costs: the primary
docstring counterfactual replaces the answer token, which makes the circuit's routing
heads causally invisible to a metric read off that token, and the pipeline's output
said nothing about it. A second published counterfactual recovered 5 of 6 heads where
the primary found 3 — but only because a human saw a low recall number and went
looking.

This module holds the inert half of the fix: the vocabulary a task uses to declare
*which* counterfactuals it has, and what each one breaks. It contains no measurement,
imports nothing from the rest of the package, and must never import a ground-truth
module — the agreement analysis it feeds runs before any answer key is consulted.

The teeth are in `TaskSpec.__post_init__`: a task cannot be registered with fewer than
two discovery schemes. Multi-scheme discovery is therefore not an option a future phase
can leave unset; skipping it means deleting a check.

    from causal_interp.schemes import Scheme, TaskSpec

    SCHEMES = {
        "yy01": Scheme("yy01", "published", "sets the start year to 01", False, primary=True),
        ...
    }
    TASK = TaskSpec(name="greater-than", dataset=GreaterThanDataset, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

# What a scheme's definition can be traced back to. The distinction is load-bearing
# rather than decorative: Phase 8's whole caveat is that `authored` schemes are
# hand-built task knowledge of the same kind Phase 5 audited, and a report that does
# not separate them from `published` ones would hide that.
PROVENANCES = ("published", "authored", "generic")


@dataclass(frozen=True)
class Scheme:
    """One named counterfactual, described in the terms Phase 7 showed matter.

    `breaks` is one sentence naming which aspect of the prompt this scheme destroys.
    `preserves_answer` records whether the correct answer survives in the corrupted
    prompt — the axis along which `random_random` and `random_def` differ, and the
    reason one of them can see the docstring circuit's routing heads and the other
    cannot. Both are declarations *about* the scheme, written when it is registered;
    nothing here is inferred from a run.
    """

    name: str
    provenance: str
    breaks: str
    preserves_answer: bool
    primary: bool = False

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCES:
            raise ValueError(f"provenance must be one of {PROVENANCES}, got {self.provenance!r}")

    @property
    def is_hand_built(self) -> bool:
        """Does this scheme encode knowledge of the task? Generic ones do not."""
        return self.provenance in ("published", "authored")


@dataclass(frozen=True)
class TaskSpec:
    """Everything the pipeline needs to run discovery on a task under every scheme.

    `dataset` is called as `dataset(model, n=..., corruption=..., seed=...)`, which is
    the constructor signature all three task modules already share.

    `discovery_schemes` is the set the pipeline sweeps **by default**. It must name at
    least two, and one of them must be the primary: a single-counterfactual task cannot
    be expressed in this type at all.
    """

    name: str
    dataset: Callable[..., Any]
    positions: Sequence[str]
    schemes: Mapping[str, Scheme]
    discovery_schemes: Sequence[str]
    metric_label: str
    model_alias: str

    def __post_init__(self) -> None:
        unknown = [s for s in self.discovery_schemes if s not in self.schemes]
        if unknown:
            raise ValueError(f"{self.name}: unregistered discovery schemes {unknown}")
        if len(self.discovery_schemes) < 2:
            raise ValueError(
                f"{self.name}: registered {len(self.discovery_schemes)} discovery scheme(s). "
                "A task must register at least two counterfactual schemes — Phase 7 showed a "
                "single scheme decides which parts of a circuit are visible at all, and "
                "nothing in a one-scheme run reveals that. See results/PHASE8_PLAN.md."
            )
        primaries = [name for name in self.discovery_schemes if self.schemes[name].primary]
        if len(primaries) != 1:
            raise ValueError(
                f"{self.name}: expected exactly one primary discovery scheme, got {primaries}"
            )

    @property
    def primary_scheme(self) -> str:
        return next(name for name in self.discovery_schemes if self.schemes[name].primary)

    def scheme(self, name: str) -> Scheme:
        return self.schemes[name]

    def table_rows(self) -> list[dict]:
        """The registration table, for reports. Inert data, safe to serialize."""
        return [
            {
                "scheme": name,
                "provenance": self.schemes[name].provenance,
                "breaks": self.schemes[name].breaks,
                "preserves_answer": self.schemes[name].preserves_answer,
                "primary": self.schemes[name].primary,
            }
            for name in self.discovery_schemes
        ]
