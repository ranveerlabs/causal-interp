"""What the pipeline says when two counterfactual schemes disagree about a head.

Phase 7's docstring result — 3 of 6 published heads under the primary counterfactual,
5 of 6 under a different one — was diagnosed by a human who noticed a low recall
number and knew which alternative to try. This module is the attempt to make that
diagnosis a routine output instead.

**It must never import a `ground_truth` module**, and the Phase 8 runner asserts that
at startup, for the same reason `search.py` carries the same prohibition: the whole
value of a disagreement report is that it is available on a circuit with no published
answer. Every quantity here is computed from measured effects and one inherited
threshold. Nothing knows which heads are "right", and nothing knows which scheme is.

Three things come out, and the order matters:

- **per-head verdicts** — `robust` (found under every scheme) or `scheme-dependent`
  (found under some and missed under others), with the full presence vector and the
  per-scheme effect. Never averaged, never collapsed onto one scheme's numbers.
- **blind spots** — for *every* scheme, the heads some other scheme found and it did
  not. Asymmetric on purpose: "what can this experiment not see" is the question
  Phase 7 could not answer from its own output.
- **the flag** — fires when the primary scheme's blind spot is non-empty. A bare
  non-emptiness test with no cutoff, so there is nothing in it to tune. It is
  deliberately noisy in one direction: a fired flag says the answer depends on the
  experiment, not that the primary scheme is wrong.

`power` is reported beside all of it and gates nothing. A scheme whose corrupted run
sits close to its clean run has a small denominator and therefore noisy normalized
effects; that is worth knowing and is not grounds for dropping a scheme, because a
power gate would be a free parameter that could be tuned until the flag fired only
where it was wanted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

Head = tuple[int, int]

# Below this fraction of the primary scheme's span, a scheme is labelled low-power in
# every table it appears in. Fixed in results/PHASE8_PLAN.md before any Phase 8 run.
# An annotation, never a gate: no code path excludes a scheme on this basis.
LOW_POWER = 0.10

ROBUST = "robust"
SCHEME_DEPENDENT = "scheme-dependent"


def _head_str(head: Head) -> str:
    return f"{head[0]}.{head[1]}"


@dataclass(frozen=True)
class SchemePower:
    """How much room a scheme leaves between its clean and corrupted runs."""

    scheme: str
    span: float
    power: float          # span / primary span
    low_power: bool

    def as_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "span": self.span,
            "power": self.power,
            "low_power": self.low_power,
        }


@dataclass(frozen=True)
class HeadVerdict:
    """One head's standing across every scheme discovery was run under."""

    head: Head
    status: str
    found_in: tuple[str, ...]
    missing_in: tuple[str, ...]
    effects: Mapping[str, float]

    @property
    def missed_by_primary(self) -> bool:
        return self.status == SCHEME_DEPENDENT

    def as_dict(self) -> dict:
        return {
            "head": _head_str(self.head),
            "status": self.status,
            "found_in": list(self.found_in),
            "missing_in": list(self.missing_in),
            "effects": dict(self.effects),
        }


@dataclass
class AgreementReport:
    """The cross-scheme comparison for one discovery channel.

    `channel` names what produced the effects — "activation patching", "path chain" —
    so the same analysis can be run at more than one level of the pipeline and the
    results kept apart.
    """

    channel: str
    threshold: float
    primary: str
    schemes: tuple[str, ...]
    verdicts: list[HeadVerdict] = field(default_factory=list)
    per_scheme: dict[str, list[Head]] = field(default_factory=dict)
    blind_spots: dict[str, list[Head]] = field(default_factory=dict)
    only_in: dict[str, list[Head]] = field(default_factory=dict)
    power: dict[str, SchemePower] = field(default_factory=dict)

    # -- the headline ------------------------------------------------------

    @property
    def union(self) -> list[Head]:
        return sorted({h for heads in self.per_scheme.values() for h in heads})

    @property
    def intersection(self) -> list[Head]:
        if not self.per_scheme:
            return []
        sets = [set(v) for v in self.per_scheme.values()]
        return sorted(set.intersection(*sets))

    @property
    def scheme_dependent(self) -> list[Head]:
        return [v.head for v in self.verdicts if v.status == SCHEME_DEPENDENT]

    @property
    def primary_blind_spot(self) -> list[Head]:
        return self.blind_spots.get(self.primary, [])

    @property
    def flag(self) -> bool:
        return bool(self.primary_blind_spot)

    @property
    def flag_text(self) -> str:
        if not self.flag:
            return (
                f"no scheme found a head the primary scheme ({self.primary}) missed — "
                f"the {len(self.union)} discovered heads are what every counterfactual sees"
            )
        heads = ", ".join(_head_str(h) for h in self.primary_blind_spot)
        return (
            f"COUNTERFACTUAL-SCHEME-DEPENDENT: {len(self.primary_blind_spot)} head(s) "
            f"[{heads}] are found under another counterfactual and missed under the "
            f"primary one ({self.primary}). The head list under the primary "
            f"counterfactual is not the circuit; it is what this counterfactual can see."
        )

    def as_dict(self) -> dict:
        return {
            "channel": self.channel,
            "threshold": self.threshold,
            "primary": self.primary,
            "schemes": list(self.schemes),
            "flag": self.flag,
            "flag_text": self.flag_text,
            "union": [_head_str(h) for h in self.union],
            "intersection": [_head_str(h) for h in self.intersection],
            "scheme_dependent": [_head_str(h) for h in self.scheme_dependent],
            "per_scheme": {k: [_head_str(h) for h in v] for k, v in self.per_scheme.items()},
            "blind_spots": {k: [_head_str(h) for h in v] for k, v in self.blind_spots.items()},
            "only_in": {k: [_head_str(h) for h in v] for k, v in self.only_in.items()},
            "power": {k: v.as_dict() for k, v in self.power.items()},
            "verdicts": [v.as_dict() for v in self.verdicts],
        }


def discovered_set(effects: Mapping[Head, float], threshold: float) -> set[Head]:
    """Heads whose absolute effect reaches `threshold`, under one scheme.

    Absolute value for the reason `comparison.threshold_set` gives: a head that pushes
    the model away from the answer is causally involved.
    """
    return {head for head, value in effects.items() if abs(value) >= threshold}


def compare_schemes(
    effects_by_scheme: Mapping[str, Mapping[Head, float]],
    *,
    threshold: float,
    primary: str,
    channel: str,
    spans: Mapping[str, float] | None = None,
) -> AgreementReport:
    """Cross-scheme agreement for one discovery channel.

    `effects_by_scheme[scheme][head]` is that head's effect under that scheme, already
    collapsed to whatever summary the channel uses (for activation patching: the head's
    effect at its own best position). `spans` are the clean-minus-corrupted differences
    used only for the power annotation.

    Schemes contributing no measurements at all — a path chain that halted, say — are
    kept in the report as empty sets rather than dropped, so a scheme that measured
    nothing cannot be mistaken for one that measured nothing *interesting*.
    """
    schemes = tuple(effects_by_scheme)
    if primary not in schemes:
        raise ValueError(f"primary scheme {primary!r} is not among {schemes}")
    if len(schemes) < 2:
        raise ValueError(
            f"cross-scheme agreement needs at least two schemes, got {schemes}. "
            "A single-scheme run cannot report what it cannot see."
        )

    per_scheme = {s: discovered_set(effects_by_scheme[s], threshold) for s in schemes}
    union = sorted({h for heads in per_scheme.values() for h in heads})

    verdicts: list[HeadVerdict] = []
    for head in union:
        found = tuple(s for s in schemes if head in per_scheme[s])
        missing = tuple(s for s in schemes if head not in per_scheme[s])
        verdicts.append(
            HeadVerdict(
                head=head,
                status=ROBUST if not missing else SCHEME_DEPENDENT,
                found_in=found,
                missing_in=missing,
                effects={s: float(effects_by_scheme[s].get(head, float("nan"))) for s in schemes},
            )
        )

    blind_spots = {
        s: sorted({h for other in schemes if other != s for h in per_scheme[other]} - per_scheme[s])
        for s in schemes
    }
    only_in = {
        s: sorted(per_scheme[s] - {h for other in schemes if other != s for h in per_scheme[other]})
        for s in schemes
    }

    power: dict[str, SchemePower] = {}
    if spans:
        reference = abs(spans.get(primary, 0.0))
        for s in schemes:
            span = float(spans.get(s, float("nan")))
            ratio = abs(span) / reference if reference else float("nan")
            power[s] = SchemePower(
                scheme=s, span=span, power=ratio, low_power=bool(ratio == ratio and ratio < LOW_POWER)
            )

    return AgreementReport(
        channel=channel,
        threshold=threshold,
        primary=primary,
        schemes=schemes,
        verdicts=verdicts,
        per_scheme={s: sorted(v) for s, v in per_scheme.items()},
        blind_spots=blind_spots,
        only_in=only_in,
        power=power,
    )


def pairwise_overlap(report: AgreementReport) -> list[dict]:
    """Jaccard overlap between every pair of schemes' discovered sets.

    A summary, not a criterion: nothing in the flag depends on it. It exists because
    "the schemes disagreed" is much less useful than "these two agreed and that one
    stands apart".
    """
    rows = []
    names = report.schemes
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = set(report.per_scheme[a]), set(report.per_scheme[b])
            union = sa | sb
            rows.append({
                "a": a,
                "b": b,
                "shared": len(sa & sb),
                "only_a": len(sa - sb),
                "only_b": len(sb - sa),
                "jaccard": len(sa & sb) / len(union) if union else float("nan"),
            })
    return rows


# ---------------------------------------------------------------------------
# The same question, one level down: receiver specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecVerdict:
    """Which receiver specification wins for one head, under each scheme."""

    head: Head
    top_spec: Mapping[str, str]     # scheme -> "input@position"
    top_score: Mapping[str, float]
    status: str                     # "robust" | "scheme-dependent"

    def as_dict(self) -> dict:
        return {
            "head": _head_str(self.head),
            "status": self.status,
            "top_spec": dict(self.top_spec),
            "top_score": dict(self.top_score),
        }


def compare_spec_rankings(
    top_by_scheme: Mapping[str, Mapping[Head, tuple[str, float]]],
    *,
    primary: str,
    heads: Sequence[Head] | None = None,
) -> dict:
    """Do the schemes agree on which input each head receives its signal on?

    `top_by_scheme[scheme][head]` is `(spec_label, score)` — the highest-scoring
    receiver specification for that head under that scheme, as the search ranked it.
    A head is `scheme-dependent` when the argmax differs between any two schemes: a
    bare inequality, no threshold, no answer key.

    Phase 7 found this failure one level below the head list — for both docstring
    argument movers the search ranked the wire carrying the answer above the wire
    choosing it — so the same comparison is run here rather than left to a reader.
    """
    schemes = tuple(top_by_scheme)
    if primary not in schemes:
        raise ValueError(f"primary scheme {primary!r} is not among {schemes}")
    if heads is None:
        heads = sorted({h for per_head in top_by_scheme.values() for h in per_head})

    verdicts: list[SpecVerdict] = []
    for head in heads:
        labels = {s: top_by_scheme[s][head][0] for s in schemes if head in top_by_scheme[s]}
        scores = {s: top_by_scheme[s][head][1] for s in schemes if head in top_by_scheme[s]}
        if not labels:
            continue
        status = ROBUST if len(set(labels.values())) == 1 else SCHEME_DEPENDENT
        verdicts.append(SpecVerdict(head=head, top_spec=labels, top_score=scores, status=status))

    dependent = [v for v in verdicts if v.status == SCHEME_DEPENDENT]
    disagree_with_primary = [
        v for v in dependent
        if primary in v.top_spec and any(l != v.top_spec[primary] for l in v.top_spec.values())
    ]
    return {
        "primary": primary,
        "schemes": list(schemes),
        "n_heads": len(verdicts),
        "n_scheme_dependent": len(dependent),
        "scheme_dependent": [_head_str(v.head) for v in dependent],
        "disagree_with_primary": [_head_str(v.head) for v in disagree_with_primary],
        "verdicts": [v.as_dict() for v in verdicts],
    }
