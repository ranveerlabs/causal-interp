"""Counterfactual schemes, registered per task, phase 8's structural change."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

PROVENANCES = ("published", "authored", "generic")


@dataclass(frozen=True)
class Scheme:
    """One named counterfactual, described in the terms phase 7 showed matter."""

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
        return self.provenance in ("published", "authored")


@dataclass(frozen=True)
class TaskSpec:
    """everything the pipeline needs to run discovery on a task under every scheme."""

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
                "A task must register at least two counterfactual schemes, Phase 7 showed a "
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
        """the registration table, for reports. inert data, safe to serialize."""
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
