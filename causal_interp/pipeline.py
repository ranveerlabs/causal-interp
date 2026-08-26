"""Discovery under every registered counterfactual scheme, the default path."""

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
    """Patch every head at every position, scoring each run under all three metrics."""
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
    """summarise each head by the position where its effect is largest in absolute value."""
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
    """multi-scheme discovery for one task: every scheme's run, plus the comparison."""

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
    """Run activation-patching discovery under every registered scheme, then compare."""
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
    """flatten a report to CSV rows, annotating each head with a published class."""
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


# phase 9, a discovery criterion in each scheme's own unitsts

NULL_QUANTILE = 0.99
SIGNIFICANT_FIGURES = 2
NULL_SEED = 20260815


def round_up_sigfigs(value: float, digits: int = SIGNIFICANT_FIGURES) -> float:
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
    """how much apparent recovery this scheme manufactures from a mismatched activation."""
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
    """`null_floor` for every registered discovery scheme."""
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
