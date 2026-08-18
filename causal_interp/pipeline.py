"""Discovery under every registered counterfactual scheme — the default path.

Phases 1-7 each wrote their own loop over corruption schemes and then reported one of
them as the result. This module makes the loop the pipeline's own: given a `TaskSpec`,
it sweeps every head at every position under **every** registered discovery scheme, and
returns the per-scheme effects together with the cross-scheme agreement analysis from
`causal_interp.agreement`.

There is deliberately **no single-scheme entry point**. `TaskSpec` refuses to be
constructed with fewer than two discovery schemes and `compare_schemes` refuses to
score fewer than two, so the multi-scheme comparison is not something a caller can
switch off — which is the difference between Phase 7's one-off diagnostic and a
standard part of the method.

Nothing here imports a ground-truth module. Scoring the discovered heads against a
published circuit happens afterwards, in the phase script, on this module's output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import torch
from transformer_lens import HookedTransformer

from causal_interp.agreement import AgreementReport, compare_schemes
from causal_interp.interventions import (
    Patch,
    baseline_for,
    clean_cache_for,
    derangement,
    run_patched,
    sweep_heads_null,
)
from causal_interp.metrics import METRICS, DistributionalBaseline, all_metrics
from causal_interp.schemes import TaskSpec

Head = tuple[int, int]


def sweep_all_metrics(
    model: HookedTransformer,
    ds,
    cache,
    logit_baseline,
    dist_baseline,
    positions: Sequence[str],
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, torch.Tensor]:
    """Patch every head at every position, scoring each run under all three metrics.

    Phase 5's sweep, lifted out of the phase scripts that had a copy each and made
    task-agnostic by taking the position vocabulary as an argument. One forward pass
    yields all three metrics, so any difference between them is the metric and not the
    run.
    """
    grids = {
        name: torch.zeros(model.cfg.n_layers, model.cfg.n_heads, len(positions))
        for name in METRICS
    }
    total = model.cfg.n_layers * model.cfg.n_heads * len(positions)
    done = 0
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            for p, position in enumerate(positions):
                logits = run_patched(model, ds, cache, [Patch(layer, "z", position, head)])
                scores = all_metrics(ds, logits, logit_baseline, dist_baseline)
                for name in METRICS:
                    grids[name][layer, head, p] = scores[name]
                done += 1
                if progress is not None:
                    progress(done, total)
    return grids


def collapse_positions(
    grid: torch.Tensor, positions: Sequence[str]
) -> tuple[dict[Head, float], dict[Head, str]]:
    """Summarise each head by the position where its effect is largest in absolute value.

    Phase 1's rule, so a head acting at one position only is not diluted by the
    positions where it does nothing.
    """
    effects: dict[Head, float] = {}
    best: dict[Head, str] = {}
    for layer in range(grid.shape[0]):
        for head in range(grid.shape[1]):
            row = grid[layer, head]
            p = int(row.abs().argmax())
            effects[(layer, head)] = float(row[p])
            best[(layer, head)] = positions[p]
    return effects, best


def rank_stats(ds, logits) -> dict[str, float]:
    """Whatever the task calls its "does the model actually solve this" check.

    The three task modules name it `io_rank_stats`, `year_rank_stats` and
    `answer_rank_stats`; the pipeline does not need to know which, and a task that
    provides none simply reports nothing here.
    """
    name = next((n for n in dir(ds) if n.endswith("_rank_stats")), None)
    return {} if name is None else getattr(ds, name)(logits)


@dataclass
class SchemeRun:
    """One scheme's discovery sweep: the grids, the collapsed effects, the baselines."""

    scheme: str
    n_prompts: int
    clean: float
    corrupted: float
    accuracy: dict = field(default_factory=dict)
    grids: dict[str, list] = field(default_factory=dict)
    effects: dict[str, dict[Head, float]] = field(default_factory=dict)
    best_positions: dict[str, dict[Head, str]] = field(default_factory=dict)
    exact_zeros: dict[str, list[int]] = field(default_factory=dict)

    @property
    def span(self) -> float:
        return self.clean - self.corrupted

    def as_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "n_prompts": self.n_prompts,
            "clean": self.clean,
            "corrupted": self.corrupted,
            "span": self.span,
            "accuracy": self.accuracy,
            "exact_zeros": self.exact_zeros,
            "effects": {
                metric: {f"{l}.{h}": v for (l, h), v in per_head.items()}
                for metric, per_head in self.effects.items()
            },
            "best_positions": {
                metric: {f"{l}.{h}": v for (l, h), v in per_head.items()}
                for metric, per_head in self.best_positions.items()
            },
            "grids": self.grids,
        }


@dataclass
class Discovery:
    """Multi-scheme discovery for one task: every scheme's run, plus the comparison."""

    task: str
    threshold: float
    primary: str
    runs: dict[str, SchemeRun] = field(default_factory=dict)
    agreement: dict[str, AgreementReport] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "threshold": self.threshold,
            "primary": self.primary,
            "runs": {k: v.as_dict() for k, v in self.runs.items()},
            "agreement": {k: v.as_dict() for k, v in self.agreement.items()},
        }


def discover(
    model: HookedTransformer,
    task: TaskSpec,
    *,
    n: int,
    seed: int,
    threshold: float,
    metrics: Sequence[str] = METRICS,
    progress: Callable[[int, int], None] | None = None,
    announce: Callable[[str], None] | None = None,
) -> Discovery:
    """Run activation-patching discovery under every registered scheme, then compare.

    Returns per-scheme effects *and* an `AgreementReport` per metric. The agreement
    report is not optional and not computed on request: a caller that wants the head
    list gets the disagreement analysis in the same object, because the Phase 7 failure
    was precisely that the head list was available on its own.
    """
    say = announce or (lambda _text: None)
    result = Discovery(task=task.name, threshold=threshold, primary=task.primary_scheme)

    for scheme in task.discovery_schemes:
        meta = task.scheme(scheme)
        say(f"\n{'=' * 72}\nscheme: {scheme}  [{meta.provenance}]  {meta.breaks}\n{'=' * 72}")
        ds = task.dataset(model, n=n, corruption=scheme, seed=seed)
        logit_baseline, clean_logits, corrupted_logits = baseline_for(model, ds)
        dist_baseline = DistributionalBaseline(ds, clean_logits, corrupted_logits)
        say(
            f"  clean {logit_baseline.clean_logit_diff:+.4f}   "
            f"corrupted {logit_baseline.corrupted_logit_diff:+.4f}   "
            f"span {logit_baseline.span:+.4f}"
        )

        cache, _ = clean_cache_for(model, ds)
        grids = sweep_all_metrics(
            model, ds, cache, logit_baseline, dist_baseline, task.positions, progress
        )

        run = SchemeRun(
            scheme=scheme,
            n_prompts=len(ds),
            clean=logit_baseline.clean_logit_diff,
            corrupted=logit_baseline.corrupted_logit_diff,
            accuracy={
                "clean": rank_stats(ds, clean_logits),
                "corrupted": rank_stats(ds, corrupted_logits),
            },
            exact_zeros={
                position: [
                    int((grids["logit_diff"][:, :, p] == 0).sum()),
                    grids["logit_diff"][:, :, p].numel(),
                ]
                for p, position in enumerate(task.positions)
            },
        )
        for name in metrics:
            effects, best = collapse_positions(grids[name], task.positions)
            run.effects[name] = effects
            run.best_positions[name] = best
            run.grids[name] = grids[name].tolist()
        result.runs[scheme] = run

    spans = {name: run.span for name, run in result.runs.items()}
    for name in metrics:
        result.agreement[name] = compare_schemes(
            {scheme: run.effects[name] for scheme, run in result.runs.items()},
            threshold=threshold,
            primary=task.primary_scheme,
            channel=f"activation patching / {name}",
            spans=spans,
        )
    return result


def agreement_rows(report: AgreementReport, classify: Callable[[Head], str | None]) -> list[dict]:
    """Flatten a report to CSV rows, annotating each head with a published class.

    `classify` is passed in by the phase script rather than imported, so this module
    still knows nothing about any answer key: the annotation is added on the way out,
    after every verdict has been decided.
    """
    rows = []
    for verdict in report.verdicts:
        row = {
            "head": f"{verdict.head[0]}.{verdict.head[1]}",
            "status": verdict.status,
            "found_in": " ".join(verdict.found_in),
            "missing_in": " ".join(verdict.missing_in),
            "published_class": classify(verdict.head) or "",
        }
        row.update({f"effect_{s}": f"{v:.6f}" for s, v in verdict.effects.items()})
        rows.append(row)
    return rows


def as_head_effects(effects: Mapping[Head, float]) -> dict[str, float]:
    return {f"{l}.{h}": v for (l, h), v in effects.items()}


# ---------------------------------------------------------------------------
# Phase 9 — a discovery criterion in each scheme's own units
# ---------------------------------------------------------------------------

# Phase 3's rule, unchanged: the 99th percentile of a shuffled-source null, rounded up
# to two significant figures. Phase 9 applies it to activation patching instead of
# `path_signal`, per scheme, because normalized recovery divides by that scheme's own
# span and a shared cutoff therefore means different things under different
# counterfactuals. Fixed in results/PHASE9_PLAN.md before any of it was measured.
NULL_QUANTILE = 0.99
SIGNIFICANT_FIGURES = 2
NULL_SEED = 20260815


def round_up_sigfigs(value: float, digits: int = SIGNIFICANT_FIGURES) -> float:
    """Round up, so a threshold never claims more precision than its null supports."""
    if value <= 0:
        return 0.0
    exponent = math.floor(math.log10(value)) - (digits - 1)
    step = 10 ** exponent
    return math.ceil(value / step) * step


def null_floor(
    model: HookedTransformer,
    task: TaskSpec,
    scheme: str,
    *,
    n: int,
    seed: int,
    null_seed: int = NULL_SEED,
    quantile: float = NULL_QUANTILE,
    sigfigs: int = SIGNIFICANT_FIGURES,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """How much apparent recovery this scheme manufactures from a mismatched activation.

    Runs the head sweep with the spliced clean value drawn from a deranged prompt order
    and returns the calibrated discovery criterion for this scheme, along with the null
    distribution behind it so the number can be checked rather than trusted.

    The unit is one (head, position) cell — the unit the sweep measures. The real
    per-head statistic is a *maximum* over positions, so comparing it against a per-cell
    null keeps more heads than a like-for-like comparison would; that direction is
    recorded in the plan and is against this criterion's own hypothesis.
    """
    ds = task.dataset(model, n=n, corruption=scheme, seed=seed)
    baseline, _, _ = baseline_for(model, ds)
    cache, _ = clean_cache_for(model, ds)
    permutation = derangement(len(ds), seed=null_seed)

    grid = sweep_heads_null(
        model, ds, cache, baseline, task.positions, permutation, progress
    )
    values = grid.abs().flatten()
    raw = float(torch.quantile(values.sort().values, quantile))
    per_head_max = grid.abs().amax(dim=-1)

    return {
        "scheme": scheme,
        "threshold": round_up_sigfigs(raw, sigfigs),
        "raw_quantile": raw,
        "quantile": quantile,
        "null_seed": null_seed,
        "n_cells": int(values.numel()),
        "null_median": float(values.median()),
        "null_mean": float(values.mean()),
        "null_max": float(values.max()),
        "null_per_head_max_median": float(per_head_max.median()),
        "span": baseline.span,
        "grid": grid.tolist(),
    }


def calibrate(
    model: HookedTransformer,
    task: TaskSpec,
    *,
    n: int,
    seed: int,
    null_seed: int = NULL_SEED,
    progress: Callable[[int, int], None] | None = None,
    announce: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    """`null_floor` for every registered discovery scheme.

    Nothing here consults a real measurement or an answer key: the null sweep never
    pairs a prompt with its own clean activation, so no result of the actual experiment
    can reach the threshold that will judge it.
    """
    say = announce or (lambda _text: None)
    floors: dict[str, dict] = {}
    for scheme in task.discovery_schemes:
        say(f"  null sweep: {scheme} ")
        floors[scheme] = null_floor(
            model, task, scheme, n=n, seed=seed, null_seed=null_seed, progress=progress
        )
        block = floors[scheme]
        say(f"    theta({scheme}) = {block['threshold']:g}"
            f"   (raw {block['raw_quantile']:.4f}, null median "
            f"{block['null_median']:.4f}, null max {block['null_max']:.3f})")
    return floors
