"""Phase 7: run the Phases 1-6 pipeline, unmodified, against a circuit in a *different model*.

    python scripts/run_phase7_docstring.py --stage sweep   # patching + path chain
    python scripts/run_phase7_docstring.py --preregister    # recalibrate the null
    python scripts/run_phase7_docstring.py                  # apply, search, report
    python scripts/run_phase7_docstring.py --report-only

The target, the ground truth, the scoring rules, an advance audit of what in the
code is GPT-2-shaped, and seven predictions were fixed in `results/PHASE7_PLAN.md`,
committed before this file existed.

Phase 6 showed the pipeline was not fitted to IOI — but greater-than and IOI both
live in GPT-2 small, so every number this project has produced comes from one
12-layer, 12-head, MLP-bearing model with one tokenizer. This script points the
same machinery at the docstring circuit in `attn-only-4l`: 4 layers, 8 heads, **no
MLP blocks**, a different tokenizer, a different corpus.

**Every threshold, cutoff, width and margin below is inherited verbatim from the
phase that introduced it**, including `COMPONENT_KINDS`, which still contains
`mlp_out` even though this model has no MLPs — the plan predicts that sweep returns
exact zeros rather than raising, and removing it in advance would have hidden the
finding instead of measuring it. The only number recalibrated is the receiver-side
threshold, whose *rule* is Phase 3's and whose *value* must be recomputed because
the null is model- and task-specific.

The three stages are separate commands for the reason Phase 3 split `--preregister`
out: the null threshold has to be on record before the comparison it will be judged
by exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from causal_interp import comparison, search
from causal_interp import ground_truth_docstring as gt
from causal_interp.docstring import CORRUPTIONS, POSITIONS, DocstringDataset
from causal_interp.ground_truth import Head
from causal_interp.interventions import (
    ALL_POSITIONS,
    LOGITS,
    Patch,
    Receiver,
    baseline_for,
    cache_for,
    clean_cache_for,
    derangement,
    greedy_select,
    patch_effect,
    path_signal,
    run_patched,
    sweep_component,
    sweep_path_senders,
    sweep_path_signal,
)
from causal_interp.metrics import METRICS, DistributionalBaseline, all_metrics
from causal_interp.model import load
from causal_interp.search import ReceiverSpec, absolute_positions, screen_specs

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SWEEP_JSON = RESULTS_DIR / "phase7_sweep.json"
PREREG_JSON = RESULTS_DIR / "phase7_preregistration.json"
RESULTS_JSON = RESULTS_DIR / "phase7_results.json"

# The model this phase exists to test transfer to. Not GPT-2 small.
MODEL = "attn-only-4l"

# The authors' benchmark default (`dataset_version="random_random"`), which is why
# it is primary — a fact about the code release, decided in the plan before any
# patching ran.
PRIMARY_CORRUPTION = "random_random"

# ---------------------------------------------------------------------------
# Inherited constants. Not one of these was chosen for this model or this task.
# ---------------------------------------------------------------------------
HEADLINE_THRESHOLD = 0.02                       # Phase 1
THRESHOLD_SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]  # Phase 1
COMPONENT_KINDS = ("resid_pre", "attn_out", "mlp_out")   # Phase 1 — mlp_out kept deliberately
CHAIN_WIDTH = 4                                 # Phase 2
NULL_QUANTILE = 0.99                            # Phase 3
SIGNIFICANT_FIGURES = 2                         # Phase 3
NULL_SEED = 20260815                            # Phase 3
TOP_K_CONFIRM = 20                              # Phase 4
AMBIGUITY_MARGIN = 0.20                         # Phase 4
AMBIGUITY_MAX_RANK = 3                          # Phase 4
GREEDY_CANDIDATES = 20                          # Phase 1
GREEDY_MAX = 10                                 # Phase 1

CACHE_KINDS = ("z", "q", "k", "v")


@dataclass(frozen=True)
class Round:
    """One step of the iterative path-patching chain: what we ask, and where."""

    name: str
    question: str
    receiver_input: str | None  # None => the logits themselves
    position: str
    expected: str


# The receiver input and position for each round come from the published account of
# this circuit, exactly as Phase 2's rounds came from the IOI paper's and Phase 6's
# from the greater-than paper's. This circuit has four levels of composition rather
# than IOI's three, so the ladder is one round longer, and the released 37-edge
# graph names which input each stage arrives on. Which heads turn up is not
# constrained: all 32 are swept as senders in every round.
ROUNDS: tuple[Round, ...] = (
    Round(
        name="direct effect on the logits",
        question="which heads move the prediction without another head relaying it?",
        receiver_input=None,
        position="END",
        expected="argument mover (3.0, 3.6)",
    ),
    Round(
        name="keys of the round-0 heads at C_def",
        question="which heads feed those heads' keys at C_def?",
        receiver_input="k",
        position="C_def",
        expected="fuzzy previous token (2.0), duplicate token (1.2)",
    ),
    Round(
        name="values of the round-1 heads at comma_B",
        question="which heads feed *those* heads' values at the comma before C_def?",
        receiver_input="v",
        position="comma_B",
        expected="induction + fuzzy previous token (1.4)",
    ),
    Round(
        name="values of the round-2 heads at B_def",
        question="which heads feed *those* heads' values at B_def?",
        receiver_input="v",
        position="B_def",
        expected="duplicate token (0.5)",
    ),
)


def assert_search_is_blind() -> None:
    """Fail loudly if the search module can see any answer key.

    Phase 6 widened Phase 4's check from `ground_truth` to any module whose name
    starts with `ground_truth`, so a third circuit needs no change here — which is
    the point of having widened it.
    """
    source = (Path(__file__).resolve().parents[1] / "causal_interp" / "search.py").read_text(
        encoding="utf-8"
    )
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and "ground_truth" in stripped:
            raise SystemExit(f"search.py imports ground truth: {stripped!r}")
    print("search.py does not import any ground_truth module — ok")


def _progress(done: int, total: int) -> None:
    if done % max(1, total // 20) == 0:
        print(".", end="", flush=True)


def _round_up_sigfigs(value: float, digits: int) -> float:
    """Round up so the threshold never claims more precision than the null supports."""
    if value <= 0:
        return 0.0
    exponent = math.floor(math.log10(value)) - (digits - 1)
    step = 10 ** exponent
    return math.ceil(value / step) * step


def _extended_heads() -> list[Head]:
    return sorted({h for heads in gt.EXTENDED_CIRCUIT.values() for h in heads})


# ---------------------------------------------------------------------------
# Stage 1 — activation patching, component sweeps, and the path-patching chain
# ---------------------------------------------------------------------------


def sweep_all_metrics(model, ds, cache, logit_baseline, dist_baseline, progress=None):
    """Patch every head at every position, scoring each run under all three metrics.

    Phase 5's function, re-expressed over this task's position vocabulary. One
    forward pass yields all three metrics, so any difference between them is the
    metric and not the run.
    """
    grids = {
        name: torch.zeros(model.cfg.n_layers, model.cfg.n_heads, len(POSITIONS))
        for name in METRICS
    }
    total = model.cfg.n_layers * model.cfg.n_heads * len(POSITIONS)
    done = 0
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            for p, position in enumerate(POSITIONS):
                logits = run_patched(model, ds, cache, [Patch(layer, "z", position, head)])
                scores = all_metrics(ds, logits, logit_baseline, dist_baseline)
                for name in METRICS:
                    grids[name][layer, head, p] = scores[name]
                done += 1
                if progress is not None:
                    progress(done, total)
    return grids


def run_corruption(model, corruption: str, n: int, seed: int) -> dict:
    """Activation patching under one corruption scheme, scored under all three metrics."""
    print(f"\n{'=' * 72}\ncorruption scheme: {corruption}\n{'=' * 72}")
    ds = DocstringDataset(model, n=n, corruption=corruption, seed=seed)
    logit_baseline, clean_logits, corrupted_logits = baseline_for(model, ds)
    dist_baseline = DistributionalBaseline(ds, clean_logits, corrupted_logits)
    clean_stats = ds.answer_rank_stats(clean_logits)
    corrupted_stats = ds.answer_rank_stats(corrupted_logits)
    print(f"  clean logit diff      {logit_baseline.clean_logit_diff:+.4f}"
          f"   (answer is top token {clean_stats['answer_is_top_token']:.1%})")
    print(f"  corrupted logit diff  {logit_baseline.corrupted_logit_diff:+.4f}"
          f"   (answer is top token {corrupted_stats['answer_is_top_token']:.1%})")

    cache, _ = clean_cache_for(model, ds)

    print("  component sweeps ...", end="", flush=True)
    t0 = time.time()
    components = {
        kind: sweep_component(model, ds, cache, logit_baseline, kind, POSITIONS).tolist()
        for kind in COMPONENT_KINDS
    }
    print(f" {time.time() - t0:.0f}s")

    print("  head sweep ", end="", flush=True)
    t0 = time.time()
    grids = sweep_all_metrics(model, ds, cache, logit_baseline, dist_baseline, _progress)
    print(f" {time.time() - t0:.0f}s")

    # Collapse positions: each head is summarised by the position where its effect
    # is largest in absolute value, so a head acting only at C_def is not diluted.
    per_metric = {}
    for name, grid in grids.items():
        effects: dict[Head, float] = {}
        best_positions: dict[Head, str] = {}
        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                row = grid[layer, head]
                p = int(row.abs().argmax())
                effects[(layer, head)] = float(row[p])
                best_positions[(layer, head)] = POSITIONS[p]

        headline = comparison.compare(
            comparison.threshold_set(effects, HEADLINE_THRESHOLD),
            f"{corruption}/{name} abs(effect) >= {HEADLINE_THRESHOLD:g}", circuit=gt,
        )
        sized = comparison.compare(
            comparison.top_k_set(effects, gt.PUBLISHED_HEAD_COUNT),
            f"{corruption}/{name} top {gt.PUBLISHED_HEAD_COUNT}", circuit=gt,
        )
        # The secondary 8-head set the plan fixed in advance, scored here so it
        # exists whichever way it comes out.
        extended = _extended_heads()
        sized_ext = comparison.top_k_set(effects, len(extended))
        per_metric[name] = {
            "effects": {f"{l}.{h}": v for (l, h), v in effects.items()},
            "best_positions": {f"{l}.{h}": v for (l, h), v in best_positions.items()},
            "grid": grid.tolist(),
            "headline": _comparison_dict(headline),
            "size_matched": _comparison_dict(sized),
            "extended": {
                "n_published": len(extended),
                "matches": sorted(f"{l}.{h}" for l, h in set(sized_ext) & set(extended)),
                "auxiliary_found": sorted(
                    f"{l}.{h}" for l, h in set(sized_ext) & set(gt.AUXILIARY_HEADS)
                ),
            },
            "auxiliary_effects": {
                f"{l}.{h}": effects[(l, h)] for l, h in gt.AUXILIARY_HEADS
            },
            "sweep": [
                {
                    "threshold": t,
                    "matched": len(comparison.compare(
                        comparison.threshold_set(effects, t), "", circuit=gt).matches),
                    "discovered": len(comparison.threshold_set(effects, t)),
                }
                for t in THRESHOLD_SWEEP
            ],
            "_effects": effects,
            "_best_positions": best_positions,
        }
        print(f"    {name:10} recall {headline.recall:.2f}  precision {headline.precision:.2f}"
              f"  ({len(headline.discovered)} discovered)"
              f"  top{gt.PUBLISHED_HEAD_COUNT} {len(sized.matches)}/{gt.PUBLISHED_HEAD_COUNT}")

    # How many head/position cells are *exact* zeros. Prediction 1 in the plan says
    # there should be none under the primary scheme, and a full block at A_def,
    # B_def and comma_B under `random_vocab_cdef`. Measured, not asserted.
    exact_zeros = {
        position: [
            int((grids["logit_diff"][:, :, p] == 0).sum()),
            grids["logit_diff"][:, :, p].numel(),
        ]
        for p, position in enumerate(POSITIONS)
    }

    result = {
        "corruption": corruption,
        "n_prompts": len(ds),
        "baseline": {
            "clean_logit_diff": logit_baseline.clean_logit_diff,
            "corrupted_logit_diff": logit_baseline.corrupted_logit_diff,
            "kl_span": dist_baseline.span["kl"],
            "tv_span": dist_baseline.span["tv"],
        },
        "accuracy": {"clean": clean_stats, "corrupted": corrupted_stats},
        "exact_zeros": exact_zeros,
        "components": components,
        "metrics": per_metric,
    }

    # Joint narrowing, on the hand-built metric only — Phase 1 ran it once per
    # scheme and Phases 6 and 7 keep that shape.
    if corruption == PRIMARY_CORRUPTION:
        effects = per_metric["logit_diff"]["_effects"]
        best_positions = per_metric["logit_diff"]["_best_positions"]
        ranked = sorted(effects, key=lambda h: abs(effects[h]), reverse=True)
        candidates = [
            Patch(layer=l, kind="z", position=best_positions[(l, h)], head=h)
            for (l, h) in ranked[:GREEDY_CANDIDATES]
        ]
        print(f"  greedy narrowing over {len(candidates)} candidates ...", end="", flush=True)
        t0 = time.time()
        trace = greedy_select(model, ds, cache, logit_baseline, candidates, max_size=GREEDY_MAX)
        print(f" {time.time() - t0:.0f}s -> {len(trace)} nodes"
              + (f", recovery {trace[-1][1]:.3f}" if trace else ""))
        result["greedy"] = [{"node": str(p), "cumulative_recovery": s} for p, s in trace]

        result["all_heads_patched_effect"] = patch_effect(
            model, ds, cache,
            [Patch(l, "z", ALL_POSITIONS, h)
             for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)],
            logit_baseline,
        )

    return result


def run_path_chain(model, n: int, seed: int) -> dict:
    """Phase 2's iterative path-patching chain, on this circuit's four rounds."""
    print(f"\n{'=' * 72}\npath patching chain: {PRIMARY_CORRUPTION}\n{'=' * 72}")
    ds = DocstringDataset(model, n=n, corruption=PRIMARY_CORRUPTION, seed=seed)
    baseline, _, _ = baseline_for(model, ds)
    clean_cache, _ = cache_for(model, ds.clean_tokens, CACHE_KINDS)
    corrupted_cache, _ = cache_for(model, ds.corrupted_tokens, CACHE_KINDS)
    n_heads_total = model.cfg.n_layers * model.cfg.n_heads

    rounds: list[dict] = []
    receivers: list[Receiver] | str = LOGITS
    carried: list[Head] = []

    for index, spec in enumerate(ROUNDS):
        if index > 0:
            if not carried:
                rounds.append({
                    "index": index, "name": spec.name, "question": spec.question,
                    "expected": spec.expected, "position": spec.position,
                    "receivers": [], "effects": {}, "discovered": [], "halted": True,
                })
                print(f"  round {index}: no receivers carried forward — chain halted")
                break
            receivers = [
                Receiver(layer=l, head=h, position=spec.position, input=spec.receiver_input)
                for (l, h) in carried
            ]

        label = "logits" if index == 0 else ", ".join(str(r) for r in receivers)
        print(f"  round {index} ({spec.name}) receivers: {label}")
        print(f"    sweeping {n_heads_total} senders ", end="", flush=True)
        t0 = time.time()
        grid = sweep_path_senders(
            model, ds, clean_cache, corrupted_cache, baseline,
            receivers=receivers, sender_position=spec.position, progress=_progress,
        )
        print(f" {time.time() - t0:.0f}s")

        effects: dict[Head, float] = {
            (l, h): float(grid[l, h])
            for l in range(model.cfg.n_layers)
            for h in range(model.cfg.n_heads)
            if not torch.isnan(grid[l, h])
        }
        if not effects:
            print("    no eligible senders — chain halted")
            rounds.append({
                "index": index, "name": spec.name, "question": spec.question,
                "expected": spec.expected, "position": spec.position,
                "receivers": [str(r) for r in receivers], "effects": {},
                "discovered": [], "halted": True,
            })
            break

        discovered = sorted(comparison.threshold_set(effects, HEADLINE_THRESHOLD))
        ranked = sorted(effects, key=lambda k: abs(effects[k]), reverse=True)

        signals: dict[Head, float] = {}
        if index > 0:
            for head in ranked[:CHAIN_WIDTH]:
                signals[head] = path_signal(
                    model, ds, clean_cache, corrupted_cache,
                    Patch(head[0], "z", spec.position, head[1]), receivers,
                )

        carried = ranked[:CHAIN_WIDTH]
        top = ", ".join(f"{l}.{h} {effects[(l, h)]:+.3f}" for l, h in ranked[:5])
        print(f"    top senders: {top}")
        print(f"    cleared cutoff: {len(discovered)}")

        rounds.append({
            "index": index,
            "senders_tested": len(effects),
            "name": spec.name,
            "question": spec.question,
            "expected": spec.expected,
            "position": spec.position,
            "receivers": [] if index == 0 else [str(r) for r in receivers],
            "effects": {f"{l}.{h}": v for (l, h), v in effects.items()},
            "discovered": [f"{l}.{h}" for l, h in discovered],
            "carried": [f"{l}.{h}" for l, h in carried],
            "signals": {f"{l}.{h}": v for (l, h), v in signals.items()},
            "halted": False,
        })

    union: set[Head] = set()
    for entry in rounds:
        union |= {tuple(map(int, h.split("."))) for h in entry["discovered"]}

    cmp = comparison.compare(union, "path patching, all rounds", circuit=gt)
    print(f"\n  path patching union: {len(cmp.matches)}/{gt.PUBLISHED_HEAD_COUNT}"
          f"  precision {cmp.precision:.2f}")

    return {
        "corruption": PRIMARY_CORRUPTION,
        "rounds": rounds,
        "discovered": [f"{l}.{h}" for l, h in sorted(union)],
        "comparison": _comparison_dict(cmp),
    }


def stage_sweep(model, n: int, seed: int) -> int:
    started = time.time()
    corruptions = {c: run_corruption(model, c, n, seed) for c in CORRUPTIONS}
    chain = run_path_chain(model, n, seed)

    payload = {
        "meta": _meta(model, n, seed, started),
        "corruptions": {
            c: {
                **{k: v for k, v in block.items() if k != "metrics"},
                "metrics": {
                    name: {k: v for k, v in entry.items() if not k.startswith("_")}
                    for name, entry in block["metrics"].items()
                },
            }
            for c, block in corruptions.items()
        },
        "path_chain": chain,
    }
    SWEEP_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_sweep_csvs(corruptions, RESULTS_DIR)
    print(f"\nwrote {SWEEP_JSON}  ({payload['meta']['runtime_seconds']}s)")
    print("run --preregister next, and commit its output before the main run.")
    return 0


# ---------------------------------------------------------------------------
# Stage 2 — recalibrate the receiver-side null under Phase 3's rule
# ---------------------------------------------------------------------------


def receiver_sets_from_chain() -> list[dict]:
    """The receiver groups the path chain arrived at, read back from its output.

    Phase 3 took its receivers from Phase 2's committed results rather than from the
    answer key; Phases 6 and 7 do the same. The groups come from a chain that was
    never told which heads to look for.
    """
    if not SWEEP_JSON.exists():
        raise SystemExit(f"missing {SWEEP_JSON}; run --stage sweep first")
    data = json.loads(SWEEP_JSON.read_text(encoding="utf-8"))
    sets = []
    for entry in data["path_chain"]["rounds"]:
        if entry.get("halted") or not entry.get("receivers"):
            continue  # round 0's receiver is the logits, where the two measures coincide
        sets.append({
            "label": f"round {entry['index']}",
            "position": entry["position"],
            "receivers": entry["receivers"],
        })
    if not sets:
        raise SystemExit("path chain produced no receiver groups to calibrate against")
    return sets


def _parse_receiver(text: str) -> Receiver:
    """'3.0.q@END' -> Receiver(layer=3, head=0, input='q', position='END')."""
    node, position = text.split("@")
    layer, head, kind = node.split(".")
    return Receiver(layer=int(layer), head=int(head), position=position, input=kind)


def preregister(model, n: int, seed: int) -> int:
    """Compute the null, derive the threshold, write it down, and stop.

    Phase 3's rule verbatim. Deliberately computes no real measurement: nothing in
    this function can see how any head scores on the actual data.
    """
    print("PRE-REGISTRATION — null distribution only, no real measurements\n")
    groups = receiver_sets_from_chain()
    ds = DocstringDataset(model, n=n, corruption=PRIMARY_CORRUPTION, seed=seed)
    clean_cache, _ = cache_for(model, ds.clean_tokens, CACHE_KINDS)
    corrupted_cache, _ = cache_for(model, ds.corrupted_tokens, CACHE_KINDS)
    permutation = derangement(n, seed=NULL_SEED)

    pooled: list[float] = []
    per_group = []
    for group in groups:
        receivers = [_parse_receiver(r) for r in group["receivers"]]
        print(f"  null sweep: {group['label']} ", end="", flush=True)
        t0 = time.time()
        grid = sweep_path_signal(
            model, ds, clean_cache, corrupted_cache,
            receivers=receivers, sender_position=group["position"],
            source_permutation=permutation, progress=_progress,
        )
        values = [
            abs(float(grid[l, h]))
            for l in range(model.cfg.n_layers)
            for h in range(model.cfg.n_heads)
            if not torch.isnan(grid[l, h])
        ]
        pooled.extend(values)
        print(f" {time.time() - t0:.0f}s  n={len(values)}"
              + (f"  max|null|={max(values):.4f}" if values else "  (no eligible senders)"))
        group_raw = (
            float(torch.quantile(torch.tensor(sorted(values)), NULL_QUANTILE)) if values else None
        )
        per_group.append({
            "label": group["label"],
            "position": group["position"],
            "receivers": group["receivers"],
            "n_senders": len(values),
            "max_abs_null": max(values) if values else None,
            "mean_abs_null": (sum(values) / len(values)) if values else None,
            "group_raw_quantile": group_raw,
            "group_threshold": _round_up_sigfigs(group_raw, SIGNIFICANT_FIGURES) if group_raw else None,
        })

    if not pooled:
        raise SystemExit("null distribution is empty; cannot pre-register a threshold")

    tensor = torch.tensor(sorted(pooled))
    raw = float(torch.quantile(tensor, NULL_QUANTILE))
    threshold = _round_up_sigfigs(raw, SIGNIFICANT_FIGURES)

    payload = {
        "task": "docstring",
        "model": model.cfg.model_name,
        "rule": (
            f"threshold = {NULL_QUANTILE:.0%} percentile of |path_signal| under a "
            f"shuffled-source null, rounded up to {SIGNIFICANT_FIGURES} significant figures"
        ),
        "rule_source": "Phase 3, inherited verbatim; only the null is recalibrated",
        "null_quantile": NULL_QUANTILE,
        "significant_figures": SIGNIFICANT_FIGURES,
        "null_seed": NULL_SEED,
        "n_prompts": n,
        "dataset_seed": seed,
        "n_null_measurements": len(pooled),
        "raw_quantile": raw,
        "threshold": threshold,
        "null_max": float(tensor.max()),
        "null_median": float(tensor.median()),
        "null_mean": float(tensor.mean()),
        "per_group": per_group,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "Computed before any real path_signal measurement on this model, and "
            "before the receiver search was run. The rule is Phase 3's and was not "
            "modified; the number differs because the null is model- and task-specific."
        ),
    }
    PREREG_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n  null measurements pooled : {len(pooled)}")
    print(f"  null median |signal|     : {payload['null_median']:.4f}")
    print(f"  null max |signal|        : {payload['null_max']:.4f}")
    print(f"  {NULL_QUANTILE:.0%} percentile (raw)      : {raw:.4f}")
    print(f"\n  RECALIBRATED THRESHOLD   : {threshold}   (Phase 3's was 0.11, Phase 6's 0.046)")
    print(f"\nwrote {PREREG_JSON}")
    print("commit this file before running the comparison.")
    return 0


# ---------------------------------------------------------------------------
# Stage 3 — receiver-input search, receiver-side criterion, comparison
# ---------------------------------------------------------------------------


def run_screen(model, label: str, ds, positions) -> dict:
    baseline, _, _ = baseline_for(model, ds)
    clean_cache, _ = cache_for(model, ds.clean_tokens, CACHE_KINDS)
    n_specs = len(positions) * 3 * model.cfg.n_layers * model.cfg.n_heads
    print(f"\n  {label}: {len(positions)} positions x 3 inputs x "
          f"{model.cfg.n_layers * model.cfg.n_heads} heads = {n_specs} specs ", end="", flush=True)
    t0 = time.time()
    scores = screen_specs(model, ds, clean_cache, baseline, positions, progress=_progress)
    print(f" {time.time() - t0:.0f}s")
    ranked = sorted(scores, key=lambda s: abs(scores[s]), reverse=True)
    print("    top: " + ", ".join(f"{s} {scores[s]:+.3f}" for s in ranked[:5]))
    return {"label": label, "scores": scores, "ranked": ranked, "baseline": baseline, "ds": ds}


def _score_spec(ranked, want, top_score) -> tuple[int | None, float, float]:
    """Rank, score and relative gap of one published (input, position) in a head's ranking."""
    want_input, want_position = want
    rank = next(
        (i for i, (spec, _) in enumerate(ranked)
         if spec.input == want_input and spec.position == want_position),
        None,
    )
    score = ranked[rank][1] if rank is not None else 0.0
    gap = (abs(top_score) - abs(score)) / abs(top_score) if top_score else 1.0
    return rank, score, gap


def rediscovery_check(scores: dict[ReceiverSpec, float]) -> list[dict]:
    """Where does the published receiver spec sit in the search's own ranking?

    Phase 4's function, unchanged apart from which ground-truth module it reads and
    one addition the plan fixed in advance: this circuit has *alternative* published
    inputs for two of its classes, and where they exist the row records how the best
    of them would have scored. That column is reported separately and never merged
    into the headline.
    """
    rows = []
    for head in sorted(gt.ALL_HEADS):
        published = gt.receiver_spec(head)
        if published is None:
            rows.append({
                "head": f"{head[0]}.{head[1]}", "class": gt.classify(head),
                "published_spec": None, "outcome": "no published spec",
            })
            continue

        ranked = search.rank_specs_for_head(scores, head[0], head[1])
        if not ranked:
            rows.append({
                "head": f"{head[0]}.{head[1]}", "class": gt.classify(head),
                "published_spec": f"{published[0]}@{published[1]}", "outcome": "not scored",
            })
            continue

        top_spec, top_score = ranked[0]
        rank, published_score, gap = _score_spec(ranked, published, top_score)

        if rank == 0:
            outcome = "agreement"
        elif published_score == 0.0:
            outcome = "unmeasurable"
        elif rank is not None and rank < AMBIGUITY_MAX_RANK and gap <= AMBIGUITY_MARGIN:
            outcome = "ambiguous"
        else:
            outcome = "disagreement"

        alternatives = []
        for alt in gt.alternative_specs(head):
            alt_rank, alt_score, _ = _score_spec(ranked, alt, top_score)
            alternatives.append({
                "spec": f"{alt[0]}@{alt[1]}",
                "rank": None if alt_rank is None else alt_rank + 1,
                "score": alt_score,
            })

        rows.append({
            "head": f"{head[0]}.{head[1]}",
            "class": gt.classify(head),
            "published_spec": f"{published[0]}@{published[1]}",
            "published_rank": None if rank is None else rank + 1,
            "published_score": published_score,
            "top_spec": f"{top_spec.input}@{top_spec.position}",
            "top_score": top_score,
            "relative_gap": gap,
            "outcome": outcome,
            "alternatives": alternatives,
        })
    return rows


def receiver_side_criterion(model, n: int, seed: int, threshold: float) -> dict:
    """Phase 3's criterion: which senders deliver signal to the chain's receivers?

    Scored against the recalibrated threshold, on the same receiver groups the null
    was calibrated on.
    """
    groups = receiver_sets_from_chain()
    ds = DocstringDataset(model, n=n, corruption=PRIMARY_CORRUPTION, seed=seed)
    clean_cache, _ = cache_for(model, ds.clean_tokens, CACHE_KINDS)
    corrupted_cache, _ = cache_for(model, ds.corrupted_tokens, CACHE_KINDS)

    per_group = []
    discovered: set[Head] = set()
    for group in groups:
        receivers = [_parse_receiver(r) for r in group["receivers"]]
        print(f"  real sweep: {group['label']} ", end="", flush=True)
        t0 = time.time()
        grid = sweep_path_signal(
            model, ds, clean_cache, corrupted_cache,
            receivers=receivers, sender_position=group["position"], progress=_progress,
        )
        signals = {
            (l, h): float(grid[l, h])
            for l in range(model.cfg.n_layers)
            for h in range(model.cfg.n_heads)
            if not torch.isnan(grid[l, h])
        }
        cleared = {h for h, v in signals.items() if abs(v) >= threshold}
        discovered |= cleared
        print(f" {time.time() - t0:.0f}s  {len(cleared)} of {len(signals)} clear {threshold}")
        per_group.append({
            "label": group["label"],
            "position": group["position"],
            "receivers": group["receivers"],
            "signals": {f"{l}.{h}": v for (l, h), v in signals.items()},
            "cleared": sorted(f"{l}.{h}" for l, h in cleared),
        })

    cmp = comparison.compare(discovered, f"receiver-side, signal >= {threshold}", circuit=gt)
    return {
        "threshold": threshold,
        "groups": per_group,
        "discovered": sorted(f"{l}.{h}" for l, h in discovered),
        "comparison": _comparison_dict(cmp),
    }


def stage_main(model, n: int, seed: int) -> int:
    if not SWEEP_JSON.exists():
        raise SystemExit(f"missing {SWEEP_JSON}; run --stage sweep first")
    if not PREREG_JSON.exists():
        raise SystemExit(
            f"missing {PREREG_JSON}\n"
            "run with --preregister first, and commit the result, so the threshold is "
            "on record before the comparison it will be judged by."
        )
    prereg = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    if prereg["n_prompts"] != n or prereg["dataset_seed"] != seed:
        raise SystemExit(
            f"pre-registration was computed for n={prereg['n_prompts']}, "
            f"seed={prereg['dataset_seed']}, but this run is n={n}, seed={seed}. "
            "Re-run --preregister for these settings."
        )
    threshold = prereg["threshold"]
    print(f"using recalibrated threshold {threshold} ({prereg['rule']})")

    started = time.time()
    sweep = json.loads(SWEEP_JSON.read_text(encoding="utf-8"))

    # -- receiver-spec search, both screens ---------------------------------
    print(f"\n{'=' * 72}\nreceiver-specification search\n{'=' * 72}")
    semantic_ds = DocstringDataset(model, n=n, corruption=PRIMARY_CORRUPTION, seed=seed)
    semantic = run_screen(model, "semantic positions", semantic_ds, POSITIONS)

    # As for greater-than, no restriction to one template is needed: every prompt is
    # the same frame with single-token substitutions, so index k means the same
    # thing in every row.
    absolute_ds = DocstringDataset(model, n=n, corruption=PRIMARY_CORRUPTION, seed=seed)
    abs_positions = absolute_positions(absolute_ds)
    absolute = run_screen(model, "absolute positions", absolute_ds, abs_positions)

    # -- stage B on the survivors -------------------------------------------
    print(f"\n  stage B: confirming top {TOP_K_CONFIRM} semantic specs")
    clean_cache, _ = cache_for(model, semantic_ds.clean_tokens, CACHE_KINDS)
    corrupted_cache, _ = cache_for(model, semantic_ds.corrupted_tokens, CACHE_KINDS)
    confirmations = []
    for spec in semantic["ranked"][:TOP_K_CONFIRM]:
        if spec.layer == 0:
            continue  # nothing upstream to sweep
        result = search.confirm_spec(
            model, semantic_ds, clean_cache, corrupted_cache, semantic["baseline"],
            spec, threshold,
        )
        result["screen_score"] = semantic["scores"][spec]
        confirmations.append(result)
        top = sorted(result["_signals"], key=lambda k: abs(result["_signals"][k]), reverse=True)[:3]
        print(f"    {spec}: top senders " + (", ".join(
            f"{l}.{h} sig {result['_signals'][(l, h)]:+.2f}" for l, h in top) or "none"))

    # -- receiver-side criterion --------------------------------------------
    print(f"\n{'=' * 72}\nreceiver-side criterion at {threshold}\n{'=' * 72}")
    receiver_side = receiver_side_criterion(model, n, seed, threshold)

    # -- only now: the answer key -------------------------------------------
    check = rediscovery_check(semantic["scores"])
    abs_labels = {
        index: semantic_of_absolute(absolute_ds, index) for index in range(len(abs_positions))
    }

    meta = _meta(model, n, seed, started)
    meta["signal_threshold"] = threshold
    meta["top_k_confirmed"] = TOP_K_CONFIRM
    meta["ambiguity_rule"] = (
        f"published spec in top {AMBIGUITY_MAX_RANK} and within {AMBIGUITY_MARGIN:.0%} of the best"
    )

    _write_search_csvs(semantic, absolute, RESULTS_DIR)

    payload = {
        "meta": meta,
        "sweep": sweep,
        "preregistration": prereg,
        "absolute_labels": {str(k): v for k, v in abs_labels.items()},
        "semantic_top": [
            {"spec": str(s), "score": semantic["scores"][s]} for s in semantic["ranked"][:60]
        ],
        "absolute_top": [
            {"spec": str(s), "score": absolute["scores"][s], "index": int(s.position[1:]),
             "is_semantically": semantic_of_absolute(absolute_ds, int(s.position[1:]))}
            for s in absolute["ranked"][:60]
        ],
        "rediscovery": check,
        "receiver_side": receiver_side,
        "confirmations": [
            {k: v for k, v in c.items() if not k.startswith("_")} for c in confirmations
        ],
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    from phase7_report import write_report  # noqa: PLC0415

    write_report(RESULTS_DIR / "PHASE7_REPORT.md", payload)
    print(f"\nwrote {RESULTS_DIR / 'PHASE7_REPORT.md'}  ({meta['runtime_seconds']}s)")
    return 0


def semantic_of_absolute(ds, index: int) -> str:
    """What a bare token index turns out to be, used only to interpret results."""
    labels = [name for name in POSITIONS if bool((ds.positions[name] == index).all())]
    return "/".join(labels) if labels else "—"


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------


def _comparison_dict(c: comparison.Comparison) -> dict:
    return {
        "label": c.label,
        "n_discovered": len(c.discovered),
        "matches": [f"{l}.{h}" for l, h in c.matches],
        "misses": [f"{l}.{h}" for l, h in c.misses],
        "extras": [f"{l}.{h}" for l, h in c.extras],
        "per_class": {k: list(v) for k, v in c.per_class.items()},
        "precision": c.precision,
        "recall": c.recall,
        "f1": c.f1,
    }


def _meta(model, n: int, seed: int, started: float) -> dict:
    return {
        "model": model.cfg.model_name,
        "model_alias": MODEL,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "attn_only": bool(model.cfg.attn_only),
        "d_model": model.cfg.d_model,
        "d_vocab": model.cfg.d_vocab,
        "tokenizer": model.cfg.tokenizer_name,
        "task": "docstring (Heimersheim and Janiak 2023)",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "transformer_lens": version("transformer_lens"),
        "python": platform.python_version(),
        "prompts": n,
        "seed": seed,
        "headline_threshold": HEADLINE_THRESHOLD,
        "published_head_count": gt.PUBLISHED_HEAD_COUNT,
        "runtime_seconds": round(time.time() - started, 1),
    }


def _write_sweep_csvs(corruptions: dict, out_dir: Path) -> None:
    with (out_dir / "phase7_head_effects.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "metric", "layer", "head", "position", "effect", "published_class"])
        for corruption, block in corruptions.items():
            for name, entry in block["metrics"].items():
                for layer, per_head in enumerate(entry["grid"]):
                    for head, per_pos in enumerate(per_head):
                        cls = gt.classify((layer, head)) or ""
                        for p, value in enumerate(per_pos):
                            w.writerow([corruption, name, layer, head, POSITIONS[p],
                                        f"{value:.6f}", cls])

    with (out_dir / "phase7_component_effects.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "component", "layer", "position", "effect"])
        for corruption, block in corruptions.items():
            for kind, rows in block["components"].items():
                for layer, per_pos in enumerate(rows):
                    for p, value in enumerate(per_pos):
                        w.writerow([corruption, kind, layer, POSITIONS[p], f"{value:.6f}"])


def _write_search_csvs(semantic: dict, absolute: dict, out_dir: Path) -> None:
    for name, block in (("semantic", semantic), ("absolute", absolute)):
        with (out_dir / f"phase7_receiver_search_{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.writer(f)
            w.writerow(["layer", "head", "input", "position", "screen_score", "published_class"])
            for spec, score in block["scores"].items():
                w.writerow([spec.layer, spec.head, spec.input, spec.position,
                            f"{score:.6f}", gt.classify((spec.layer, spec.head)) or ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stage", choices=("sweep",), help="run the patching stage and stop")
    parser.add_argument("--preregister", action="store_true",
                        help="recalibrate the null threshold; compute nothing real")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    n = 16 if args.quick else args.n
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.report_only:
        from phase7_report import write_report  # noqa: PLC0415

        payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        write_report(RESULTS_DIR / "PHASE7_REPORT.md", payload)
        print(f"rebuilt {RESULTS_DIR / 'PHASE7_REPORT.md'} from stored results (no GPU work)")
        return 0

    assert_search_is_blind()
    model = load(MODEL)

    if args.stage == "sweep":
        return stage_sweep(model, n, args.seed)
    if args.preregister:
        return preregister(model, n, args.seed)
    return stage_main(model, n, args.seed)


if __name__ == "__main__":
    sys.exit(main())
