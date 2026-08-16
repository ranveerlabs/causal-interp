"""Scoring a discovered head set against a published circuit.

Kept separate from the run itself so that the comparison is pure set arithmetic
over a ground-truth module and cannot be tuned by anything the run observed.

Two comparisons are produced, because either one alone is misleading:

- **Threshold-based** — every head whose effect exceeds a cutoff. Honest about
  set size, but the cutoff is a free parameter, so it is always reported as a
  sweep rather than a single number.
- **Size-matched** — the top-*k* heads, *k* being the published circuit's size.
  Has no free parameter, so it cannot be tuned; precision and recall coincide.

**Which circuit** is a parameter, added in Phase 6 when a second published circuit
arrived. It defaults to IOI, so every call written in Phases 1-5 keeps its exact
meaning and none of them were touched. A ground-truth module qualifies as an
argument here if it exposes `CIRCUIT`, `ALL_HEADS` and `classify` — both
`ground_truth` and `ground_truth_greater_than` do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType

from causal_interp import ground_truth as _default_circuit
from causal_interp.ground_truth import Head


@dataclass
class Comparison:
    """One scoring of a discovered head set against the published circuit."""

    label: str
    discovered: set[Head]
    matches: list[Head] = field(default_factory=list)
    misses: list[Head] = field(default_factory=list)
    extras: list[Head] = field(default_factory=list)
    per_class: dict[str, tuple[int, int]] = field(default_factory=dict)
    circuit: ModuleType = _default_circuit

    @property
    def precision(self) -> float:
        return len(self.matches) / len(self.discovered) if self.discovered else 0.0

    @property
    def recall(self) -> float:
        return len(self.matches) / len(self.circuit.ALL_HEADS)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def compare(discovered: set[Head], label: str, circuit: ModuleType = _default_circuit) -> Comparison:
    """Score `discovered` against the published circuit in `circuit`."""
    result = Comparison(label=label, discovered=set(discovered), circuit=circuit)
    result.matches = sorted(discovered & circuit.ALL_HEADS)
    result.misses = sorted(circuit.ALL_HEADS - discovered)
    result.extras = sorted(discovered - circuit.ALL_HEADS)
    result.per_class = {
        cls: (sum(1 for h in heads if h in discovered), len(heads))
        for cls, heads in circuit.CIRCUIT.items()
    }
    return result


def threshold_set(effects: dict[Head, float], threshold: float) -> set[Head]:
    """Heads whose absolute effect reaches `threshold`.

    Absolute value matters: a head that reliably pushes the model *away* from the
    correct answer (the published negative name movers do exactly this) is
    causally involved, and a signed cutoff would discard it.
    """
    return {head for head, effect in effects.items() if abs(effect) >= threshold}


def top_k_set(effects: dict[Head, float], k: int) -> set[Head]:
    """The `k` heads with the largest absolute effect."""
    ranked = sorted(effects, key=lambda h: abs(effects[h]), reverse=True)
    return set(ranked[:k])


def threshold_sweep(
    effects: dict[Head, float], thresholds: list[float], circuit: ModuleType = _default_circuit
) -> list[Comparison]:
    """Score the discovered set at several cutoffs, so no single one has to be trusted."""
    # Labels avoid the |...| notation on purpose: these strings are rendered into
    # markdown table cells, where a literal pipe splits the column.
    return [
        compare(threshold_set(effects, t), label=f"abs(effect) >= {t:g}", circuit=circuit)
        for t in thresholds
    ]


def miss_report(
    effects: dict[Head, float],
    positions: dict[Head, str],
    discovered: set[Head],
    circuit: ModuleType = _default_circuit,
) -> list[dict]:
    """For each published head not discovered, what the run actually measured.

    Reporting the measured value for every miss keeps a near-threshold miss from
    being presented the same way as a head with no detectable effect at all.
    """
    rows = []
    for head in sorted(circuit.ALL_HEADS - discovered):
        rows.append(
            {
                "head": f"{head[0]}.{head[1]}",
                "class": circuit.classify(head),
                "effect": effects.get(head, 0.0),
                "best_position": positions.get(head, "-"),
            }
        )
    return sorted(rows, key=lambda r: abs(r["effect"]), reverse=True)
