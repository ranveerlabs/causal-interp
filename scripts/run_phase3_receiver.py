"""Phase 3: a pre-registered receiver-side discovery criterion.

    python scripts/run_phase3_receiver.py --preregister   # fix the threshold, write it down
    python scripts/run_phase3_receiver.py                 # apply it, compare, report

Phases 1 and 2 both scored a head as "found" by what it does to the output logit
difference, and both landed on 20/26. Phase 2 also computed `path_signal` — how
much of a receiver's clean-vs-corrupted difference a path actually delivers — and
that diagnostic scored several of the missing heads well.

Turning a diagnostic that was observed to look good into a discovery criterion is
exactly the move that invalidates a validation exercise, so the threshold is fixed
first, by a rule, in a separate step that never computes a real measurement:

    threshold = 99th percentile of |path_signal| under a shuffled-source null,
                rounded up to two significant figures

The null runs the identical procedure but draws the sender's clean value from a
*different* prompt in the batch. The path, the freezing, the receiver and the
projection are all unchanged; only the correspondence between the value carried
and the prompt it belongs to is destroyed. Whatever projection that still
produces is what the method manufactures from nothing, and the 99th percentile
fixes the false-positive rate at about one in a hundred before any real number is
looked at.

`--preregister` writes results/phase3_preregistration.json and exits. The main run
refuses to start without it. Committing that file before the comparison exists is
what makes the ordering auditable in git history rather than merely asserted.
"""

from __future__ import annotations

import argparse
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

from causal_interp import comparison, ground_truth
from causal_interp.ground_truth import Head, classify
from causal_interp.interventions import (
    Receiver,
    cache_for,
    derangement,
    sweep_path_signal,
)
from causal_interp.ioi import IOIDataset
from causal_interp.model import load

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PHASE2_JSON = RESULTS_DIR / "phase2_results.json"
PREREG_JSON = RESULTS_DIR / "phase3_preregistration.json"

# The pre-registered rule. These two numbers define the threshold; the threshold
# itself is whatever they produce. Changing them after seeing a result would be
# the thing this whole phase exists to avoid.
NULL_QUANTILE = 0.99
SIGNIFICANT_FIGURES = 2
NULL_SEED = 20260815

CACHE_KINDS = ("z", "mlp_out", "q", "k", "v")


@dataclass(frozen=True)
class ReceiverSet:
    """One receiver group to sweep senders against, recovered from Phase 2's chain."""

    label: str
    scheme: str
    position: str
    receivers: tuple[Receiver, ...]


def _parse_receiver(text: str) -> Receiver:
    """'9.9.q@END' -> Receiver(layer=9, head=9, input='q', position='END')."""
    node, position = text.split("@")
    layer, head, kind = node.split(".")
    return Receiver(layer=int(layer), head=int(head), position=position, input=kind)


def receiver_sets() -> list[ReceiverSet]:
    """The receiver groups Phase 2 arrived at, read back from its committed results.

    Reusing them rather than re-deriving them keeps this phase asking about the
    same paths Phase 2 asked about, so the two criteria differ only in how a
    result is scored. It also means the receivers still come from Phase 2's
    non-circular chain and not from the answer key.
    """
    if not PHASE2_JSON.exists():
        raise SystemExit(f"missing {PHASE2_JSON}; run scripts/run_phase2_paths.py first")
    data = json.loads(PHASE2_JSON.read_text(encoding="utf-8"))

    sets: list[ReceiverSet] = []
    for scheme, res in data["schemes"].items():
        for entry in res["rounds"]:
            # Round 0's receiver is the logits, where a receiver-side measure and
            # the logit measure are the same quantity by construction.
            if entry.get("halted") or not entry.get("receivers"):
                continue
            sets.append(ReceiverSet(
                label=f"{scheme} round {entry['index']}",
                scheme=scheme,
                position=entry["position"],
                receivers=tuple(_parse_receiver(r) for r in entry["receivers"]),
            ))

    probe = data.get("previous_token_probe", {})
    for variant in probe.get("variants", []):
        sets.append(ReceiverSet(
            label=f"previous-token probe (receivers layer >= {variant['receiver_floor']})",
            scheme="abc",
            position="S1+1",
            receivers=tuple(_parse_receiver(r) for r in variant["receivers"]),
        ))
    return sets


def _contexts(model, n: int, seed: int) -> dict[str, tuple]:
    """Dataset and caches per corruption scheme, built once and shared."""
    out = {}
    for scheme in ("s2_swap", "abc"):
        ds = IOIDataset(model, n=n, corruption=scheme, seed=seed)
        clean_cache, _ = cache_for(model, ds.clean_tokens, CACHE_KINDS)
        corrupted_cache, _ = cache_for(model, ds.corrupted_tokens, CACHE_KINDS)
        out[scheme] = (ds, clean_cache, corrupted_cache)
    return out


def _sweep(model, contexts, group: ReceiverSet, permutation) -> dict[Head, float]:
    ds, clean_cache, corrupted_cache = contexts[group.scheme]

    def progress(done: int, total: int) -> None:
        if done % max(1, total // 20) == 0:
            print(".", end="", flush=True)

    grid = sweep_path_signal(
        model, ds, clean_cache, corrupted_cache,
        receivers=list(group.receivers), sender_position=group.position,
        source_permutation=permutation, progress=progress,
    )
    return {
        (l, h): float(grid[l, h])
        for l in range(model.cfg.n_layers)
        for h in range(model.cfg.n_heads)
        if not torch.isnan(grid[l, h])
    }


def _round_up_sigfigs(value: float, digits: int) -> float:
    """Round up so the threshold never claims more precision than the null supports."""
    if value <= 0:
        return 0.0
    import math

    exponent = math.floor(math.log10(value)) - (digits - 1)
    step = 10 ** exponent
    return math.ceil(value / step) * step


def preregister(model, n: int, seed: int) -> int:
    """Compute the null, derive the threshold, write it down, and stop.

    Deliberately computes no real measurement. Nothing in this function can see
    how any head scores on the actual data.
    """
    print("PRE-REGISTRATION — null distribution only, no real measurements\n")
    groups = receiver_sets()
    contexts = _contexts(model, n, seed)
    permutation = derangement(n, seed=NULL_SEED)

    pooled: list[float] = []
    per_group = []
    for group in groups:
        print(f"  null sweep: {group.label} ", end="", flush=True)
        t0 = time.time()
        null = _sweep(model, contexts, group, permutation)
        values = [abs(v) for v in null.values()]
        pooled.extend(values)
        print(f" {time.time() - t0:.0f}s  n={len(values)}"
              + (f"  max|null|={max(values):.4f}" if values else "  (no eligible senders)"))
        # The same rule applied within each group. Fixed here, alongside the primary,
        # so both are on record before any real measurement. The pooled threshold is
        # the pre-registered criterion; these are reported next to it because pooling
        # necessarily gives a group with a wide null a lenient bar relative to its own
        # noise, and a reader should be able to see which discoveries depend on that.
        group_raw = (
            float(torch.quantile(torch.tensor(sorted(values)), NULL_QUANTILE)) if values else None
        )
        per_group.append({
            "label": group.label,
            "scheme": group.scheme,
            "position": group.position,
            "receivers": [str(r) for r in group.receivers],
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
        "rule": (
            f"threshold = {NULL_QUANTILE:.0%} percentile of |path_signal| under a "
            f"shuffled-source null, rounded up to {SIGNIFICANT_FIGURES} significant figures"
        ),
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
            "Computed before any real path_signal measurement in this phase. The "
            "author had already seen a handful of Phase 2 path_signal values, so "
            "this is not a blind pre-registration; the mitigation is that the "
            "number is produced by a fixed rule rather than chosen."
        ),
    }
    PREREG_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n  null measurements pooled : {len(pooled)}")
    print(f"  null median |signal|     : {payload['null_median']:.4f}")
    print(f"  null max |signal|        : {payload['null_max']:.4f}")
    print(f"  {NULL_QUANTILE:.0%} percentile (raw)      : {raw:.4f}")
    print(f"\n  PRE-REGISTERED THRESHOLD : {threshold}")
    print(f"\nwrote {PREREG_JSON}")
    print("commit this file before running the comparison.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--preregister", action="store_true",
                        help="derive and record the threshold; compute nothing real")
    args = parser.parse_args()

    n = 16 if args.quick else args.n
    RESULTS_DIR.mkdir(exist_ok=True)
    model = load()

    if args.preregister:
        return preregister(model, n, args.seed)

    if not PREREG_JSON.exists():
        raise SystemExit(
            f"missing {PREREG_JSON}\n"
            "run with --preregister first, and commit the result, so the threshold is "
            "on record before the comparison it will be judged by."
        )
    prereg = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    # A threshold calibrated on a different dataset size is not the threshold that
    # was pre-registered for this run, and quietly reusing it would defeat the point.
    if prereg["n_prompts"] != n or prereg["dataset_seed"] != args.seed:
        raise SystemExit(
            f"pre-registration was computed for n={prereg['n_prompts']}, "
            f"seed={prereg['dataset_seed']}, but this run is n={n}, seed={args.seed}. "
            "Re-run --preregister for these settings."
        )
    print(f"using pre-registered threshold {prereg['threshold']} ({prereg['rule']})")

    from phase3_analysis import run_analysis  # noqa: PLC0415 - split for auditability

    return run_analysis(model, prereg, n, args.seed, RESULTS_DIR)


if __name__ == "__main__":
    sys.exit(main())
