"""Phase 5: how much does the hand-built task, corruption and metric actually buy?

    python scripts/run_phase5_scoping.py            # full run, ~9 min
    python scripts/run_phase5_scoping.py --quick
    python scripts/run_phase5_scoping.py --report-only

Phase 4 closed the receiver-specification gap and named what remained: the task,
the corruption schemes and the metric are all still built from knowledge of what
the model does. `results/PHASE5_AUDIT.md` itemises what each one encodes, and was
written before this ran.

This script measures two of the three. Every combination of four corruption
schemes and three metrics is run through the same head sweep Phase 1 used, and
each is scored against the same published circuit. One forward pass yields all
three metrics, so any difference between them is the metric and not the run.

Task construction is deliberately not attempted; the audit explains why.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from causal_interp import comparison, ground_truth
from causal_interp.ground_truth import Head, classify
from causal_interp.interventions import Patch, baseline_for, clean_cache_for, run_patched
from causal_interp.ioi import POSITIONS, IOIDataset
from causal_interp.metrics import METRICS, DistributionalBaseline, all_metrics
from causal_interp.model import load

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Phase 1's cutoff, reused unchanged. All three metrics are normalized onto the
# same 0-to-1 scale, so the same number means the same thing on each; picking a
# different cutoff per metric would let the comparison be tuned.
THRESHOLD = 0.02
THRESHOLD_SWEEP = [0.01, 0.02, 0.05, 0.10]

CORRUPTIONS = ("s2_swap", "abc", "random_vocab_s2", "random_vocab_any")


def sweep_all_metrics(model, ds, cache, logit_baseline, dist_baseline, progress=None):
    """Patch every head at every position, scoring each run under all three metrics."""
    grids = {
        name: torch.zeros(model.cfg.n_layers, model.cfg.n_heads, len(POSITIONS))
        for name in METRICS
    }
    total = model.cfg.n_layers * model.cfg.n_heads * len(POSITIONS)
    done = 0
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            for p, position in enumerate(POSITIONS):
                logits = run_patched(
                    model, ds, cache, [Patch(layer, "z", position, head)]
                )
                scores = all_metrics(ds, logits, logit_baseline, dist_baseline)
                for name in METRICS:
                    grids[name][layer, head, p] = scores[name]
                done += 1
                if progress is not None:
                    progress(done, total)
    return grids


def run_corruption(model, corruption: str, n: int, seed: int) -> dict:
    print(f"\n  {corruption} ", end="", flush=True)
    ds = IOIDataset(model, n=n, corruption=corruption, seed=seed)
    logit_baseline, clean_logits, corrupted_logits = baseline_for(model, ds)
    dist_baseline = DistributionalBaseline(ds, clean_logits, corrupted_logits)
    cache, _ = clean_cache_for(model, ds)

    t0 = time.time()

    def progress(done: int, total: int) -> None:
        if done % max(1, total // 20) == 0:
            print(".", end="", flush=True)

    grids = sweep_all_metrics(model, ds, cache, logit_baseline, dist_baseline, progress)
    print(f" {time.time() - t0:.0f}s")

    per_metric = {}
    for name, grid in grids.items():
        effects: dict[Head, float] = {}
        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                row = grid[layer, head]
                effects[(layer, head)] = float(row[int(row.abs().argmax())])
        cmp = comparison.compare(comparison.threshold_set(effects, THRESHOLD), f"{corruption}/{name}")
        # Size-matched to the published circuit. A fixed cutoff compares metrics on
        # scales that need not agree — a metric whose effects run larger discovers
        # more heads at the same number without discriminating any better. Taking
        # the top 26 removes the scale entirely and leaves only the ranking.
        sized = comparison.compare(
            comparison.top_k_set(effects, ground_truth.PUBLISHED_HEAD_COUNT), f"{corruption}/{name}/top26"
        )
        per_metric[name] = {
            "effects": {f"{l}.{h}": v for (l, h), v in effects.items()},
            "matched": len(cmp.matches),
            "discovered": len(cmp.discovered),
            "precision": cmp.precision,
            "recall": cmp.recall,
            "size_matched": len(sized.matches),
            "per_class": {k: list(v) for k, v in cmp.per_class.items()},
            "sweep": [
                {
                    "threshold": t,
                    "matched": len(comparison.compare(
                        comparison.threshold_set(effects, t), "").matches),
                    "discovered": len(comparison.threshold_set(effects, t)),
                }
                for t in THRESHOLD_SWEEP
            ],
            "_effects": effects,
        }
        print(f"    {name:10} {cmp.recall:.2f} recall  {cmp.precision:.2f} precision"
              f"  ({len(cmp.discovered)} discovered)  top26 {len(sized.matches)}/26")

    return {
        "corruption": corruption,
        "logit_span": logit_baseline.span,
        "clean_logit_diff": logit_baseline.clean_logit_diff,
        "corrupted_logit_diff": logit_baseline.corrupted_logit_diff,
        "kl_span": dist_baseline.span["kl"],
        "tv_span": dist_baseline.span["tv"],
        "metrics": per_metric,
    }


def rank_agreement(a: dict[Head, float], b: dict[Head, float]) -> float:
    """Spearman correlation between two metrics' head rankings, on |effect|."""
    heads = sorted(set(a) & set(b))
    if len(heads) < 3:
        return float("nan")
    va = torch.tensor([abs(a[h]) for h in heads])
    vb = torch.tensor([abs(b[h]) for h in heads])
    ra = va.argsort().argsort().float()
    rb = vb.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    n = 16 if args.quick else args.n
    RESULTS_DIR.mkdir(exist_ok=True)

    from phase5_report import write_report  # noqa: PLC0415

    if args.report_only:
        payload = json.loads((RESULTS_DIR / "phase5_results.json").read_text(encoding="utf-8"))
        for block in payload["corruptions"].values():
            for entry in block["metrics"].values():
                entry["_effects"] = {
                    tuple(map(int, k.split("."))): v for k, v in entry["effects"].items()
                }
        write_report(RESULTS_DIR / "PHASE5_REPORT.md", payload["meta"], payload["corruptions"],
                     payload["agreement"], THRESHOLD)
        print("rebuilt PHASE5_REPORT.md from stored results (no GPU work)")
        return 0

    started = time.time()
    model = load()
    results = {c: run_corruption(model, c, n, args.seed) for c in CORRUPTIONS}

    agreement = {}
    for corruption, block in results.items():
        effects = {name: block["metrics"][name]["_effects"] for name in METRICS}
        agreement[corruption] = {
            "logit_diff vs kl": rank_agreement(effects["logit_diff"], effects["kl"]),
            "logit_diff vs tv": rank_agreement(effects["logit_diff"], effects["tv"]),
            "kl vs tv": rank_agreement(effects["kl"], effects["tv"]),
        }

    meta = {
        "model": model.cfg.model_name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "transformer_lens": version("transformer_lens"),
        "python": platform.python_version(),
        "prompts": n,
        "seed": args.seed,
        "threshold": THRESHOLD,
        "runtime_seconds": round(time.time() - started, 1),
    }

    with (RESULTS_DIR / "phase5_metric_effects.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "metric", "layer", "head", "effect", "published_class"])
        for corruption, block in results.items():
            for name, entry in block["metrics"].items():
                for key, value in entry["effects"].items():
                    layer, head = map(int, key.split("."))
                    w.writerow([corruption, name, layer, head, f"{value:.6f}",
                                classify((layer, head)) or ""])

    write_report(RESULTS_DIR / "PHASE5_REPORT.md", meta, results, agreement, THRESHOLD)
    payload = {
        "meta": meta,
        "agreement": agreement,
        "corruptions": {
            c: {
                **{k: v for k, v in block.items() if k != "metrics"},
                "metrics": {
                    name: {k: v for k, v in entry.items() if not k.startswith("_")}
                    for name, entry in block["metrics"].items()
                },
            }
            for c, block in results.items()
        },
    }
    (RESULTS_DIR / "phase5_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR / 'PHASE5_REPORT.md'}  ({meta['runtime_seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
