"""What the pipeline says when two counterfactual schemes disagree about a head."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

Head = tuple[int, int]

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
    """one head's standing across every scheme discovery was run under."""

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
    """the cross-scheme comparison for one discovery channel."""

    channel: str
    threshold: float | Mapping[str, float]
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
                f"no scheme found a head the primary scheme ({self.primary}) missed, "
                f"the {len(self.union)} discovered heads are what every counterfactual sees"
            )
        heads = ", ".join(_head_str(h) for h in self.primary_blind_spot)
        return (
            f"COUNTERFACTUAL-SCHEME-DEPENDENT: {len(self.primary_blind_spot)} head(s) "
            f"[{heads}] are found under another counterfactual and missed under the "
            f"primary one ({self.primary}). the head list under the primary "
            f"counterfactual isnt the circuit, it is what this counterfactual can see."
        )

    def as_dict(self) -> dict:
        return {
            "channel": self.channel,
            "threshold": dict(self.threshold) if not isinstance(self.threshold, float)
                         else self.threshold,
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
    return {head for head, value in effects.items() if abs(value) >= threshold}


def compare_schemes(
    effects_by_scheme: Mapping[str, Mapping[Head, float]],
    *,
    threshold: float | Mapping[str, float],
    primary: str,
    channel: str,
    spans: Mapping[str, float] | None = None,
) -> AgreementReport:
    """Cross-scheme agreement for one discovery channel."""
    schemes = tuple(effects_by_scheme)
    if primary not in schemes:
        raise ValueError(f"primary scheme {primary!r} is not among {schemes}")
    if len(schemes) < 2:
        raise ValueError(
            f"cross-scheme agreement needs at least two schemes, got {schemes}. "
            "A single-scheme run cannot report what it cannot see."
        )

    per_threshold = (
        {s: float(threshold) for s in schemes}
        if isinstance(threshold, (int, float))
        else {s: float(threshold[s]) for s in schemes}
    )
    per_scheme = {s: discovered_set(effects_by_scheme[s], per_threshold[s]) for s in schemes}
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
        threshold=per_threshold,
        primary=primary,
        schemes=schemes,
        verdicts=verdicts,
        per_scheme={s: sorted(v) for s, v in per_scheme.items()},
        blind_spots=blind_spots,
        only_in=only_in,
        power=power,
    )


def pairwise_overlap(report: AgreementReport) -> list[dict]:
    """Jaccard overlap between every pair of schemes' discovered sets."""
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


# the same question, one level down: receiver specifications


@dataclass(frozen=True)
class SpecVerdict:
    """which receiver specification wins for one head, under each scheme."""

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
    """do the schemes agree on which input each head receives its signal on?"""
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
