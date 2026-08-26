"""scoring a discovered head set against a published circuit."""

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
    return {head for head, effect in effects.items() if abs(effect) >= threshold}


def top_k_set(effects: dict[Head, float], k: int) -> set[Head]:
    ranked = sorted(effects, key=lambda h: abs(effects[h]), reverse=True)
    return set(ranked[:k])


def threshold_sweep(
    effects: dict[Head, float], thresholds: list[float], circuit: ModuleType = _default_circuit
) -> list[Comparison]:
    # labels avoid the |...| notation on purpose
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
    """for each published head not discovered, what the run actually measured."""
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
