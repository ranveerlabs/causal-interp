"""phase 11, step 2, re-run discovery R times under independent resampling."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_interp import pipeline
from causal_interp.docstring import TASK as DOCSTRING_TASK
from causal_interp.greater_than import TASK as GREATER_THAN_TASK
from causal_interp.model import load

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# every one of these is inherited from the plan, not chosen here.
SEEDS = tuple(range(10))          # R = 10
N_PROMPTS = 128                   # phase 8's n, held fixed
METRIC = "logit_diff"             # the plan's metric
PHASE8_THRESHOLD = 0.02           # passed to discover() only cuz it demands one.
                                  # no phase 11 statistic reads the agreement report

CIRCUITS = {
    "docstring": {"task": DOCSTRING_TASK, "model": "attn-only-4l"},
    "greater_than": {"task": GREATER_THAN_TASK, "model": "gpt2-small"},
}


def _progress(done: int, total: int) -> None:
    if done % max(1, total // 20) == 0:
        print(".", end="", flush=True)


def assert_measurement_is_blind() -> None:
    """no module on this path, including this script, may import an answer key."""
    targets = [ROOT / "causal_interp" / n for n in
               ("search.py", "agreement.py", "pipeline.py", "schemes.py", "interventions.py",
                "metrics.py", "docstring.py", "greater_than.py")]
    targets.append(Path(__file__).resolve())
    for path in targets:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and "ground_truth" in stripped:
                raise SystemExit(f"{path.name} imports ground truth: {stripped!r}")
    print("nothing on the measurement path imports a ground_truth module, ok")


def out_path(circuit: str, seed: int) -> Path:
    return RESULTS_DIR / f"phase11_{circuit}_seed{seed}.json"


def run_one(model, task, circuit: str, seed: int) -> float:
    started = time.time()
    discovery = pipeline.discover(
        model, task, n=N_PROMPTS, seed=seed, threshold=PHASE8_THRESHOLD,
        progress=_progress, announce=lambda text: print(text, flush=True),
    )
    elapsed = round(time.time() - started, 1)

    payload = {
        "meta": {
            "circuit": circuit,
            "task": task.name,
            "seed": seed,
            "prompts": N_PROMPTS,
            "metric": METRIC,
            "model": model.cfg.model_name,
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
            "positions": list(task.positions),
            "primary": task.primary_scheme,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "transformer_lens": version("transformer_lens"),
            "python": platform.python_version(),
            "runtime_seconds": elapsed,
        },
        "runs": {
            scheme: {
                "clean": run.clean,
                "corrupted": run.corrupted,
                "span": run.span,
                "accuracy": run.accuracy,
                # (n_layers, n_heads, n_positions), the plan's e(h, s, r) is thehe
                # max-|.| collapse of this, and the footnote variant reads it directly.
                "grid": run.grids[METRIC],
            }
            for scheme, run in discovery.runs.items()
        },
    }
    out_path(circuit, seed).write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {out_path(circuit, seed).name}  ({elapsed}s)")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", choices=tuple(CIRCUITS), required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--force", action="store_true",
                        help="re-measure seeds whose payload already exists")
    args = parser.parse_args()

    assert_measurement_is_blind()
    RESULTS_DIR.mkdir(exist_ok=True)

    config = CIRCUITS[args.circuit]
    task = config["task"]
    print(f"\n{'#' * 72}")
    print(f"# Phase 11, {task.name} ({config['model']})")
    print(f"# {len(task.discovery_schemes)} schemes x {len(args.seeds)} resamples, "
          f"n={N_PROMPTS}, metric {METRIC}")
    print(f"{'#' * 72}")

    model = load(config["model"])
    total = 0.0
    for seed in args.seeds:
        if out_path(args.circuit, seed).exists() and not args.force:
            print(f"\nseed {seed}: already measured, skipping")
            continue
        print(f"\n{'~' * 72}\nresample {seed}\n{'~' * 72}")
        total += run_one(model, task, args.circuit, seed)
    print(f"\nPhase 11 measurement for {args.circuit} done, {round(total, 1)}s of new sweeps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
