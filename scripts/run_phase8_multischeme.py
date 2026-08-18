"""Phase 8: discovery under every registered counterfactual, and what disagrees.

    python scripts/run_phase8_multischeme.py --circuit docstring
    python scripts/run_phase8_multischeme.py --circuit greater_than
    python scripts/run_phase8_multischeme.py --report-only

The design, the scheme registrations, the decision rules and eight predictions were
fixed in `results/PHASE8_PLAN.md`, committed before this file existed.

Phase 7 found that the primary counterfactual decides which parts of a circuit are
visible at all — its routing heads are invisible to a metric read off a token the
counterfactual replaces — and that a different published counterfactual recovers 5 of
6 heads where the primary finds 3. That was a one-off diagnostic, run because a human
saw a low recall number and knew what to try.

This script re-runs Phase 6's circuit and Phase 7's circuit through
`causal_interp.pipeline`, which sweeps **every** registered scheme and returns the
cross-scheme agreement analysis in the same object as the head list. Three channels
are compared, in order of the pipeline's own dependency chain:

    1. activation patching   — which heads clear the threshold under each scheme
    2. the path chain        — which senders each scheme's chain arrives at
    3. the receiver search   — which input each head's signal arrives on, per scheme

**The answer key is not consulted until every verdict above has been decided.** The
scoring section is last on purpose and is separated by a banner in the output, because
the phase's actual question is whether the structure catches Phase 7's blindness
*without* a human knowing to look for it.

Every threshold and cutoff is inherited: 0.02 from Phase 1, the chain width from
Phase 2, the position vocabularies from the task modules. Nothing is recalibrated,
and this phase adds no free parameter — the flag is a non-emptiness test.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from causal_interp import agreement, comparison, pipeline, search
from causal_interp import ground_truth_docstring as gt_docstring
from causal_interp import ground_truth_greater_than as gt_greater_than
from causal_interp.docstring import TASK as DOCSTRING_TASK
from causal_interp.greater_than import TASK as GREATER_THAN_TASK
from causal_interp.interventions import (
    LOGITS,
    Patch,
    Receiver,
    baseline_for,
    cache_for,
    sweep_path_senders,
)
from causal_interp.model import load

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ---------------------------------------------------------------------------
# Inherited constants. Not one of these was chosen for this phase.
# ---------------------------------------------------------------------------
HEADLINE_THRESHOLD = 0.02   # Phase 1
CHAIN_WIDTH = 4             # Phase 2
PRIMARY_METRIC = "logit_diff"


@dataclass(frozen=True)
class Round:
    """One step of the iterative path-patching chain: what we ask, and where."""

    name: str
    question: str
    receiver_input: str | None  # None => the logits themselves
    position: str


# The chain rounds are copied verbatim from `run_phase6_greater_than.py` and
# `run_phase7_docstring.py`. They come from each paper's account of the mechanism, as
# they did in Phases 2, 6 and 7; nothing about them is new here, and re-deriving them
# would have made this phase's chains incomparable with those phases'.
GREATER_THAN_ROUNDS: tuple[Round, ...] = (
    Round("direct effect on the logits",
          "which heads move the prediction without another head relaying it?", None, "END"),
    Round("values of the round-0 heads at YY",
          "which heads feed those heads' values at YY?", "v", "YY"),
    Round("values of the round-1 heads at YY",
          "which heads feed *those* heads' values at YY?", "v", "YY"),
)

DOCSTRING_ROUNDS: tuple[Round, ...] = (
    Round("direct effect on the logits",
          "which heads move the prediction without another head relaying it?", None, "END"),
    Round("keys of the round-0 heads at C_def",
          "which heads feed those heads' keys at C_def?", "k", "C_def"),
    Round("values of the round-1 heads at comma_B",
          "which heads feed *those* heads' values at the comma before C_def?", "v", "comma_B"),
    Round("values of the round-2 heads at B_def",
          "which heads feed *those* heads' values at B_def?", "v", "B_def"),
)

CIRCUITS = {
    "docstring": {
        "task": DOCSTRING_TASK,
        "ground_truth": gt_docstring,
        "model": "attn-only-4l",
        "rounds": DOCSTRING_ROUNDS,
        "cache_kinds": ("z", "q", "k", "v"),
        "phase": "7",
    },
    "greater_than": {
        "task": GREATER_THAN_TASK,
        "ground_truth": gt_greater_than,
        "model": "gpt2-small",
        "rounds": GREATER_THAN_ROUNDS,
        "cache_kinds": ("z", "mlp_out", "q", "k", "v"),
        "phase": "6",
    },
}


def assert_analysis_is_blind() -> None:
    """Fail loudly if anything in the discovery path can see an answer key.

    Phase 4 introduced this check for `search.py`; Phase 6 widened it to any module
    whose name starts with `ground_truth`. Phase 8 adds the three modules that decide
    the disagreement verdicts, because a flag computed with the answer key in reach
    would prove nothing about what the pipeline can see on its own.
    """
    for name in ("search.py", "agreement.py", "pipeline.py", "schemes.py"):
        source = (Path(__file__).resolve().parents[1] / "causal_interp" / name).read_text(
            encoding="utf-8"
        )
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and "ground_truth" in stripped:
                raise SystemExit(f"{name} imports ground truth: {stripped!r}")
    print("search.py, agreement.py, pipeline.py, schemes.py import no ground_truth — ok")


def _progress(done: int, total: int) -> None:
    if done % max(1, total // 20) == 0:
        print(".", end="", flush=True)


def _say(text: str) -> None:
    print(text, flush=True)


def _head_str(head: tuple[int, int]) -> str:
    return f"{head[0]}.{head[1]}"


def _parse_head(text: str) -> tuple[int, int]:
    layer, head = text.split(".")
    return int(layer), int(head)


# ---------------------------------------------------------------------------
# Channel 2 — the path-patching chain, run under every scheme
# ---------------------------------------------------------------------------


def run_chain(model, task, rounds, cache_kinds, scheme: str, n: int, seed: int) -> dict:
    """Phase 2's iterative chain, on one scheme.

    Identical in structure to the chains in Phases 2, 6 and 7 — each round's receivers
    are the previous round's top senders, and the answer key is never consulted to
    choose them. The only difference is that it runs once per registered scheme rather
    than once on the primary.
    """
    ds = task.dataset(model, n=n, corruption=scheme, seed=seed)
    baseline, _, _ = baseline_for(model, ds)
    clean_cache, _ = cache_for(model, ds.clean_tokens, cache_kinds)
    corrupted_cache, _ = cache_for(model, ds.corrupted_tokens, cache_kinds)

    out_rounds: list[dict] = []
    receivers: list[Receiver] | str = LOGITS
    carried: list[tuple[int, int]] = []

    for index, spec in enumerate(rounds):
        if index > 0:
            if not carried:
                out_rounds.append({"index": index, "name": spec.name, "halted": True,
                                   "reason": "no receivers carried forward"})
                print(f"    round {index}: chain halted (nothing carried forward)")
                break
            receivers = [
                Receiver(layer=l, head=h, position=spec.position, input=spec.receiver_input)
                for (l, h) in carried
            ]

        print(f"    round {index} ({spec.name}) ", end="", flush=True)
        grid = sweep_path_senders(
            model, ds, clean_cache, corrupted_cache, baseline,
            receivers=receivers, sender_position=spec.position, progress=_progress,
        )
        effects = {
            (l, h): float(grid[l, h])
            for l in range(model.cfg.n_layers)
            for h in range(model.cfg.n_heads)
            if not torch.isnan(grid[l, h])
        }
        if not effects:
            print(" halted (no eligible senders)")
            out_rounds.append({"index": index, "name": spec.name, "halted": True,
                               "reason": "no eligible senders"})
            break

        discovered = sorted(agreement.discovered_set(effects, HEADLINE_THRESHOLD))
        ranked = sorted(effects, key=lambda k: abs(effects[k]), reverse=True)
        carried = ranked[:CHAIN_WIDTH]
        print(f" top: " + ", ".join(f"{_head_str(h)} {effects[h]:+.3f}" for h in ranked[:4]))

        out_rounds.append({
            "index": index,
            "name": spec.name,
            "question": spec.question,
            "position": spec.position,
            "senders_tested": len(effects),
            "receivers": [] if index == 0 else [str(r) for r in receivers],
            "effects": {_head_str(h): v for h, v in effects.items()},
            "discovered": [_head_str(h) for h in discovered],
            "carried": [_head_str(h) for h in carried],
            "halted": False,
        })

    union: dict[tuple[int, int], float] = {}
    for entry in out_rounds:
        for head_text, value in entry.get("effects", {}).items():
            head = _parse_head(head_text)
            if abs(value) > abs(union.get(head, 0.0)):
                union[head] = value

    return {
        "scheme": scheme,
        "rounds": out_rounds,
        "halted": any(r.get("halted") for r in out_rounds),
        "discovered": sorted(
            _head_str(h) for h, v in union.items() if abs(v) >= HEADLINE_THRESHOLD
        ),
        "_best_effects": union,
    }


# ---------------------------------------------------------------------------
# Channel 3 — the receiver-specification search, run under every scheme
# ---------------------------------------------------------------------------


def run_spec_search(model, task, cache_kinds, scheme: str, n: int, seed: int) -> dict:
    """Screen every receiver specification under one scheme, and keep each head's best.

    Phase 4's screen, unchanged. What Phase 8 adds is running it once per scheme and
    comparing the argmaxes: Phase 7 found the blindness repeats here, with the wire
    carrying the answer outranking the wire that chooses it.
    """
    ds = task.dataset(model, n=n, corruption=scheme, seed=seed)
    baseline, _, _ = baseline_for(model, ds)
    clean_cache, _ = cache_for(model, ds.clean_tokens, cache_kinds)
    print(f"    screening {model.cfg.n_layers * model.cfg.n_heads * 3 * len(task.positions)} specs ",
          end="", flush=True)
    scores = search.screen_specs(model, ds, clean_cache, baseline, task.positions,
                                 progress=_progress)
    top: dict[tuple[int, int], tuple[str, float]] = {}
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            ranked = search.rank_specs_for_head(scores, layer, head)
            if ranked:
                spec, value = ranked[0]
                top[(layer, head)] = (f"{spec.input}@{spec.position}", value)
    best = sorted(top, key=lambda h: abs(top[h][1]), reverse=True)[:4]
    print(" top: " + ", ".join(f"{_head_str(h)} {top[h][0]} {top[h][1]:+.3f}" for h in best))
    return {
        "scheme": scheme,
        "top_per_head": {_head_str(h): {"spec": s, "score": v} for h, (s, v) in top.items()},
        "all_scores": {str(spec): value for spec, value in scores.items()},
        "_top": top,
        "_scores": scores,
    }


# ---------------------------------------------------------------------------
# Scoring — everything below this line consults the published circuit
# ---------------------------------------------------------------------------


def _comparison_dict(c: comparison.Comparison) -> dict:
    """The same serializer Phases 6 and 7 each carry, copied rather than shared.

    `comparison.py` is the module every phase scores through, and Phase 8 has no reason
    to edit it.
    """
    return {
        "label": c.label,
        "n_discovered": len(c.discovered),
        "matches": [_head_str(h) for h in c.matches],
        "misses": [_head_str(h) for h in c.misses],
        "extras": [_head_str(h) for h in c.extras],
        "per_class": {k: list(v) for k, v in c.per_class.items()},
        "precision": c.precision,
        "recall": c.recall,
        "f1": c.f1,
    }


def score_against_circuit(report: agreement.AgreementReport, gt) -> dict:
    """What the blind verdicts turn out to mean, once the answer key is opened."""
    published = set(gt.ALL_HEADS)
    per_scheme = {
        scheme: _comparison_dict(comparison.compare(set(heads), scheme, circuit=gt))
        for scheme, heads in report.per_scheme.items()
    }
    union_cmp = comparison.compare(set(report.union), "union of all schemes", circuit=gt)
    primary_cmp = comparison.compare(set(report.per_scheme[report.primary]),
                                     f"{report.primary} alone", circuit=gt)

    blind_spot = report.primary_blind_spot
    return {
        "per_scheme": per_scheme,
        "union": _comparison_dict(union_cmp),
        "primary": _comparison_dict(primary_cmp),
        "blind_spot": {
            "heads": [_head_str(h) for h in blind_spot],
            "published_heads_in_blind_spot": sorted(
                _head_str(h) for h in blind_spot if h in published
            ),
            "unpublished_heads_in_blind_spot": sorted(
                _head_str(h) for h in blind_spot if h not in published
            ),
        },
        "published_head_status": [
            {
                "head": _head_str(head),
                "class": gt.classify(head),
                "status": next(
                    (v.status for v in report.verdicts if v.head == head), "found by no scheme"
                ),
                "found_in": next(
                    (list(v.found_in) for v in report.verdicts if v.head == head), []
                ),
            }
            for head in sorted(published)
        ],
        "recall_primary": primary_cmp.recall,
        "recall_union": union_cmp.recall,
        "precision_primary": primary_cmp.precision,
        "precision_union": union_cmp.precision,
    }


def score_spec_agreement(spec_reports: dict, spec_agreement: dict, gt) -> dict:
    """Where the published receiver specification sits, per scheme, for each head.

    The counterpart of Phase 4's rediscovery check, asked once per scheme so that
    "the search disagrees with the paper" can be separated from "the search agrees
    with the paper under a different counterfactual".
    """
    rows = []
    for head in sorted(gt.ALL_HEADS):
        published = gt.receiver_spec(head)
        if published is None:
            rows.append({"head": _head_str(head), "class": gt.classify(head),
                         "published_spec": None})
            continue
        want = f"{published[0]}@{published[1]}"
        per_scheme = {}
        for scheme, block in spec_reports.items():
            ranked = search.rank_specs_for_head(block["_scores"], head[0], head[1])
            rank = next(
                (i + 1 for i, (spec, _) in enumerate(ranked)
                 if spec.input == published[0] and spec.position == published[1]),
                None,
            )
            per_scheme[scheme] = {
                "top_spec": block["_top"].get(head, ("-", 0.0))[0],
                "published_rank": rank,
                "agrees": bool(rank == 1),
            }
        rows.append({
            "head": _head_str(head),
            "class": gt.classify(head),
            "published_spec": want,
            "per_scheme": per_scheme,
            "any_scheme_agrees": any(v["agrees"] for v in per_scheme.values()),
            "primary_agrees": per_scheme.get(spec_agreement["primary"], {}).get("agrees", False),
        })
    scoreable = [r for r in rows if r.get("published_spec")]
    return {
        "rows": rows,
        "n_scoreable": len(scoreable),
        "n_primary_agrees": sum(1 for r in scoreable if r["primary_agrees"]),
        "n_any_scheme_agrees": sum(1 for r in scoreable if r["any_scheme_agrees"]),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_circuit(circuit: str, n: int, seed: int) -> int:
    config = CIRCUITS[circuit]
    task = config["task"]
    gt = config["ground_truth"]
    started = time.time()

    model = load(config["model"])
    print(f"\n{'#' * 72}")
    print(f"# Phase 8 — {task.name} ({config['model']}), "
          f"{len(task.discovery_schemes)} registered schemes")
    print(f"# primary: {task.primary_scheme}   "
          f"threshold: {HEADLINE_THRESHOLD} (Phase 1's, inherited)")
    print(f"{'#' * 72}")
    for row in task.table_rows():
        mark = " (primary)" if row["primary"] else ""
        print(f"  {row['scheme']:18} [{row['provenance']:9}] answer preserved: "
              f"{str(row['preserves_answer']):5}{mark}")

    # -- channel 1: activation patching under every scheme -------------------
    discovery = pipeline.discover(
        model, task, n=n, seed=seed, threshold=HEADLINE_THRESHOLD,
        progress=_progress, announce=_say,
    )
    head_report = discovery.agreement[PRIMARY_METRIC]
    print(f"\n{'=' * 72}\ncross-scheme agreement — activation patching\n{'=' * 72}")
    for scheme in head_report.schemes:
        power = head_report.power[scheme]
        label = "  LOW-POWER" if power.low_power else ""
        print(f"  {scheme:18} found {len(head_report.per_scheme[scheme]):3}   "
              f"span {power.span:+8.3f}  power {power.power:.2f}{label}")
    print(f"\n  robust           : {len(head_report.union) - len(head_report.scheme_dependent)}")
    print(f"  scheme-dependent : {len(head_report.scheme_dependent)}")
    print(f"\n  {head_report.flag_text}\n")

    # -- channel 2: the path chain under every scheme -------------------------
    print(f"{'=' * 72}\npath-patching chain, once per scheme\n{'=' * 72}")
    chains = {}
    for scheme in task.discovery_schemes:
        print(f"  scheme {scheme}")
        chains[scheme] = run_chain(
            model, task, config["rounds"], config["cache_kinds"], scheme, n, seed
        )
    chain_report = agreement.compare_schemes(
        {scheme: block["_best_effects"] for scheme, block in chains.items()},
        threshold=HEADLINE_THRESHOLD, primary=task.primary_scheme, channel="path chain",
        spans={scheme: discovery.runs[scheme].span for scheme in chains},
    )
    print(f"\n  {chain_report.flag_text}\n")

    # -- channel 3: the receiver-spec search under every scheme ---------------
    print(f"{'=' * 72}\nreceiver-specification search, once per scheme\n{'=' * 72}")
    spec_runs = {}
    for scheme in task.discovery_schemes:
        print(f"  scheme {scheme}")
        spec_runs[scheme] = run_spec_search(
            model, task, config["cache_kinds"], scheme, n, seed
        )
    spec_report = agreement.compare_spec_rankings(
        {scheme: block["_top"] for scheme, block in spec_runs.items()},
        primary=task.primary_scheme,
    )
    print(f"\n  receiver specs that change with the counterfactual: "
          f"{spec_report['n_scheme_dependent']} of {spec_report['n_heads']} heads")

    # -- and only now, the answer key ----------------------------------------
    print(f"\n{'=' * 72}\nSCORING — the published circuit is opened only here\n{'=' * 72}")
    scored_heads = score_against_circuit(head_report, gt)
    scored_chain = score_against_circuit(chain_report, gt)
    scored_specs = score_spec_agreement(spec_runs, spec_report, gt)

    print(f"  recall under the primary scheme alone : "
          f"{len(scored_heads['primary']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}"
          f"  (precision {scored_heads['precision_primary']:.2f})")
    print(f"  recall under the union of all schemes : "
          f"{len(scored_heads['union']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}"
          f"  (precision {scored_heads['precision_union']:.2f})")
    print(f"  published heads inside the flagged blind spot: "
          f"{scored_heads['blind_spot']['published_heads_in_blind_spot'] or 'none'}")
    print(f"  unpublished heads inside it                  : "
          f"{len(scored_heads['blind_spot']['unpublished_heads_in_blind_spot'])}")

    payload = {
        "meta": {
            "circuit": circuit,
            "task": task.name,
            "model": model.cfg.model_name,
            "model_alias": config["model"],
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
            "reruns_phase": config["phase"],
            "prompts": n,
            "seed": seed,
            "threshold": HEADLINE_THRESHOLD,
            "primary_metric": PRIMARY_METRIC,
            "published_head_count": gt.PUBLISHED_HEAD_COUNT,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "transformer_lens": version("transformer_lens"),
            "python": platform.python_version(),
            "runtime_seconds": round(time.time() - started, 1),
        },
        "schemes": task.table_rows(),
        "discovery": discovery.as_dict(),
        "head_agreement": head_report.as_dict(),
        "head_overlap": agreement.pairwise_overlap(head_report),
        "chain": {scheme: {k: v for k, v in block.items() if not k.startswith("_")}
                  for scheme, block in chains.items()},
        "chain_agreement": chain_report.as_dict(),
        "chain_overlap": agreement.pairwise_overlap(chain_report),
        "spec_search": {
            scheme: {k: v for k, v in block.items() if not k.startswith("_")}
            for scheme, block in spec_runs.items()
        },
        "spec_agreement": spec_report,
        "scored": {
            "heads": scored_heads,
            "chain": scored_chain,
            "specs": scored_specs,
        },
    }
    out = RESULTS_DIR / f"phase8_{circuit}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csvs(circuit, head_report, chain_report, spec_report, gt)
    print(f"\nwrote {out}  ({payload['meta']['runtime_seconds']}s)")
    return 0


def _write_csvs(circuit, head_report, chain_report, spec_report, gt) -> None:
    with (RESULTS_DIR / f"phase8_{circuit}_head_agreement.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        schemes = list(head_report.schemes)
        w = csv.writer(f)
        w.writerow(["channel", "head", "status", "found_in", "missing_in", "published_class"]
                   + [f"effect_{s}" for s in schemes])
        for report in (head_report, chain_report):
            for verdict in report.verdicts:
                w.writerow(
                    [report.channel, _head_str(verdict.head), verdict.status,
                     " ".join(verdict.found_in), " ".join(verdict.missing_in),
                     gt.classify(verdict.head) or ""]
                    + [f"{verdict.effects.get(s, float('nan')):.6f}" for s in schemes]
                )

    with (RESULTS_DIR / f"phase8_{circuit}_spec_agreement.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        schemes = list(spec_report["schemes"])
        w = csv.writer(f)
        w.writerow(["head", "status", "published_class"] + [f"top_spec_{s}" for s in schemes])
        for verdict in spec_report["verdicts"]:
            head = _parse_head(verdict["head"])
            w.writerow([verdict["head"], verdict["status"], gt.classify(head) or ""]
                       + [verdict["top_spec"].get(s, "-") for s in schemes])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", choices=tuple(CIRCUITS), help="which circuit to re-run")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    n = 16 if args.quick else args.n
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.report_only:
        from phase8_report import write_report  # noqa: PLC0415

        write_report(RESULTS_DIR / "PHASE8_REPORT.md", RESULTS_DIR)
        print(f"rebuilt {RESULTS_DIR / 'PHASE8_REPORT.md'} from stored results (no GPU work)")
        return 0

    if not args.circuit:
        parser.error("--circuit is required unless --report-only is given")

    assert_analysis_is_blind()
    return run_circuit(args.circuit, n, args.seed)


if __name__ == "__main__":
    sys.exit(main())
