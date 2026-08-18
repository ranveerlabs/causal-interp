"""Phase 9: does a per-scheme null floor separate a real blind spot from noise?

    python scripts/run_phase9_calibration.py --circuit docstring
    python scripts/run_phase9_calibration.py --circuit greater_than
    python scripts/run_phase9_calibration.py --circuit ioi        # the holdout
    python scripts/run_phase9_calibration.py --report-only

The rule, the scoring table, the holdout and eight predictions were fixed in
`results/PHASE9_PLAN.md`, committed before this file existed; the characterization it
rests on was committed before that.

Phase 8's flag compared every scheme against one shared cutoff of 0.02. But normalized
recovery divides by each scheme's own clean-vs-corrupted span, so that number does not
mean the same thing under two counterfactuals. This script replaces it with Phase 3's
rule applied per scheme:

    theta(s) = 99th percentile of |normalized recovery| under a shuffled-source null,
               rounded up to two significant figures

and recomputes Phase 8's verdicts with **nothing else changed**, so the two runs differ
in the criterion alone.

For docstring and greater-than the real sweeps are **read back from the committed Phase
8 payloads** rather than repeated: re-running them could only introduce a difference
this phase would then have to disentangle from the criterion. Only the null sweeps are
new. IOI has never been run through the multi-scheme pipeline at all, so it gets both.

The answer key is opened at the end, after every calibrated verdict exists.
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

from causal_interp import agreement, comparison, pipeline
from causal_interp import ground_truth as gt_ioi
from causal_interp import ground_truth_docstring as gt_docstring
from causal_interp import ground_truth_greater_than as gt_greater_than
from causal_interp.docstring import TASK as DOCSTRING_TASK
from causal_interp.greater_than import TASK as GREATER_THAN_TASK
from causal_interp.ioi import TASK as IOI_TASK
from causal_interp.model import load

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Inherited, all of them. Phase 1's cutoff is kept only as the *comparison* baseline —
# the number this phase is testing a replacement for.
PHASE8_THRESHOLD = 0.02
METRIC = "logit_diff"

CIRCUITS = {
    "docstring": {
        "task": DOCSTRING_TASK, "ground_truth": gt_docstring, "model": "attn-only-4l",
        "phase8": "phase8_docstring.json", "role": "known real blind spot (Phase 7)",
    },
    "greater_than": {
        "task": GREATER_THAN_TASK, "ground_truth": gt_greater_than, "model": "gpt2-small",
        "phase8": "phase8_greater_than.json", "role": "known noise (Phase 6)",
    },
    "ioi": {
        "task": IOI_TASK, "ground_truth": gt_ioi, "model": "gpt2-small",
        "phase8": None, "role": "HOLDOUT — never run through the multi-scheme pipeline",
    },
}


def _progress(done: int, total: int) -> None:
    if done % max(1, total // 20) == 0:
        print(".", end="", flush=True)


def _say(text: str) -> None:
    print(text, flush=True)


def _head(text: str) -> tuple[int, int]:
    layer, head = text.split(".")
    return int(layer), int(head)


def _hs(head: tuple[int, int]) -> str:
    return f"{head[0]}.{head[1]}"


def assert_analysis_is_blind() -> None:
    """The calibration path must not be able to see an answer key.

    Phase 4 introduced this for `search.py`, Phase 6 widened it, Phase 8 extended it to
    the three modules that decide the disagreement verdicts. Phase 9 adds nothing new to
    the list — `pipeline.null_floor` lives in a module already on it — and re-runs the
    check because the threshold is now computed there too.
    """
    for name in ("search.py", "agreement.py", "pipeline.py", "schemes.py",
                 "interventions.py"):
        source = (Path(__file__).resolve().parents[1] / "causal_interp" / name).read_text(
            encoding="utf-8"
        )
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and "ground_truth" in stripped:
                raise SystemExit(f"{name} imports ground truth: {stripped!r}")
    print("no module on the calibration path imports a ground_truth module — ok")


# ---------------------------------------------------------------------------
# the real effects: reused where they exist, measured where they do not
# ---------------------------------------------------------------------------


def stored_effects(circuit: str) -> tuple[dict[str, dict[tuple[int, int], float]], dict[str, float], dict]:
    """Phase 8's committed per-scheme effects, spans and scheme table."""
    payload = json.loads((RESULTS_DIR / CIRCUITS[circuit]["phase8"]).read_text(encoding="utf-8"))
    runs = payload["discovery"]["runs"]
    effects = {
        scheme: {_head(h): v for h, v in block["effects"][METRIC].items()}
        for scheme, block in runs.items()
    }
    spans = {scheme: block["span"] for scheme, block in runs.items()}
    return effects, spans, payload


def measure_effects(model, task, n: int, seed: int) -> tuple[dict, dict, dict]:
    """The standard multi-scheme discovery path, for a circuit with no stored run."""
    discovery = pipeline.discover(
        model, task, n=n, seed=seed, threshold=PHASE8_THRESHOLD,
        progress=_progress, announce=_say,
    )
    effects = {scheme: run.effects[METRIC] for scheme, run in discovery.runs.items()}
    spans = {scheme: run.span for scheme, run in discovery.runs.items()}
    return effects, spans, discovery.as_dict()


# ---------------------------------------------------------------------------
# scoring — the answer key is opened only inside these two functions
# ---------------------------------------------------------------------------


def _comparison_dict(c: comparison.Comparison) -> dict:
    return {
        "label": c.label,
        "n_discovered": len(c.discovered),
        "matches": [_hs(h) for h in c.matches],
        "misses": [_hs(h) for h in c.misses],
        "precision": c.precision,
        "recall": c.recall,
    }


def score(report: agreement.AgreementReport, gt) -> dict:
    published = set(gt.ALL_HEADS)
    blind = report.primary_blind_spot
    return {
        "flag": report.flag,
        "n_flagged": len(blind),
        "flagged": [_hs(h) for h in blind],
        "published_in_blind_spot": sorted(_hs(h) for h in blind if h in published),
        "unpublished_in_blind_spot": sorted(_hs(h) for h in blind if h not in published),
        "per_scheme": {
            scheme: _comparison_dict(comparison.compare(set(heads), scheme, circuit=gt))
            for scheme, heads in report.per_scheme.items()
        },
        "primary": _comparison_dict(
            comparison.compare(set(report.per_scheme[report.primary]), "primary", circuit=gt)
        ),
        "union": _comparison_dict(
            comparison.compare(set(report.union), "union", circuit=gt)
        ),
        "blind_spot_contributions": {
            scheme: sorted(
                _hs(h) for h in blind if h in set(report.per_scheme[scheme])
            )
            for scheme in report.schemes if scheme != report.primary
        },
    }


# ---------------------------------------------------------------------------


def run_circuit(circuit: str, n: int, seed: int) -> int:
    config = CIRCUITS[circuit]
    task = config["task"]
    gt = config["ground_truth"]
    started = time.time()

    print(f"\n{'#' * 72}")
    print(f"# Phase 9 — {task.name} ({config['model']})   [{config['role']}]")
    print(f"# {len(task.discovery_schemes)} schemes, primary {task.primary_scheme}")
    print(f"{'#' * 72}")

    model = load(config["model"])

    if config["phase8"]:
        print("\nreal effects: read back from the committed Phase 8 payload (not re-run)")
        effects, spans, source = stored_effects(circuit)
        source_kind = "phase8"
    else:
        print("\nreal effects: measuring — this circuit has no stored multi-scheme run")
        effects, spans, source = measure_effects(model, task, n, seed)
        source_kind = "measured"

    # -- the uncalibrated comparison, Phase 8's criterion --------------------
    before = agreement.compare_schemes(
        effects, threshold=PHASE8_THRESHOLD, primary=task.primary_scheme,
        channel=f"activation patching / {METRIC} / shared 0.02", spans=spans,
    )

    # -- the calibration ----------------------------------------------------
    print(f"\n{'=' * 72}\nnull calibration — Phase 3's rule, per scheme\n{'=' * 72}")
    floors = pipeline.calibrate(
        model, task, n=n, seed=seed, progress=_progress, announce=_say
    )
    thresholds = {scheme: block["threshold"] for scheme, block in floors.items()}

    after = agreement.compare_schemes(
        effects, threshold=thresholds, primary=task.primary_scheme,
        channel=f"activation patching / {METRIC} / per-scheme null floor", spans=spans,
    )

    print(f"\n{'=' * 72}\nbefore and after\n{'=' * 72}")
    for scheme in before.schemes:
        print(f"  {scheme:20} power {before.power[scheme].power:.2f}   "
              f"theta {thresholds[scheme]:<8g} found "
              f"{len(before.per_scheme[scheme]):3} -> {len(after.per_scheme[scheme]):3}")
    print(f"\n  flagged (primary blind spot): {len(before.primary_blind_spot)} -> "
          f"{len(after.primary_blind_spot)}")
    print(f"  {after.flag_text}")

    # -- and only now the answer key ----------------------------------------
    print(f"\n{'=' * 72}\nSCORING — the published circuit is opened only here\n{'=' * 72}")
    scored_before, scored_after = score(before, gt), score(after, gt)
    print(f"  published heads in the flagged set: "
          f"{scored_before['published_in_blind_spot'] or 'none'} -> "
          f"{scored_after['published_in_blind_spot'] or 'none'}")
    print(f"  unpublished heads in it           : "
          f"{len(scored_before['unpublished_in_blind_spot'])} -> "
          f"{len(scored_after['unpublished_in_blind_spot'])}")
    print(f"  primary scheme recall             : "
          f"{len(scored_before['primary']['matches'])}/{gt.PUBLISHED_HEAD_COUNT} -> "
          f"{len(scored_after['primary']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}")

    payload = {
        "meta": {
            "circuit": circuit,
            "task": task.name,
            "role": config["role"],
            "model": model.cfg.model_name,
            "model_alias": config["model"],
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
            "prompts": n,
            "seed": seed,
            "metric": METRIC,
            "phase8_threshold": PHASE8_THRESHOLD,
            "null_quantile": pipeline.NULL_QUANTILE,
            "null_seed": pipeline.NULL_SEED,
            "significant_figures": pipeline.SIGNIFICANT_FIGURES,
            "real_effects_source": source_kind,
            "published_head_count": gt.PUBLISHED_HEAD_COUNT,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "transformer_lens": version("transformer_lens"),
            "python": platform.python_version(),
            "runtime_seconds": round(time.time() - started, 1),
        },
        "schemes": task.table_rows(),
        "floors": {
            scheme: {k: v for k, v in block.items() if k != "grid"}
            for scheme, block in floors.items()
        },
        "before": before.as_dict(),
        "after": after.as_dict(),
        "scored_before": scored_before,
        "scored_after": scored_after,
        "effects": {
            scheme: {_hs(h): v for h, v in per_head.items()}
            for scheme, per_head in effects.items()
        },
    }
    if source_kind == "measured":
        payload["discovery"] = source

    out = RESULTS_DIR / f"phase9_{circuit}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(circuit, before, after, thresholds, gt)
    print(f"\nwrote {out}  ({payload['meta']['runtime_seconds']}s)")
    return 0


def _write_csv(circuit, before, after, thresholds, gt) -> None:
    schemes = list(before.schemes)
    path = RESULTS_DIR / f"phase9_{circuit}_calibration.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["head", "published_class", "flagged_before", "flagged_after"]
                   + [f"effect_{s}" for s in schemes]
                   + [f"over_theta_{s}" for s in schemes])
        heads = sorted(set(before.union) | set(after.union))
        b, a = set(before.primary_blind_spot), set(after.primary_blind_spot)
        for head in heads:
            row_before = next((v for v in before.verdicts if v.head == head), None)
            effects = row_before.effects if row_before else {}
            w.writerow(
                [_hs(head), gt.classify(head) or "", head in b, head in a]
                + [f"{effects.get(s, float('nan')):.6f}" for s in schemes]
                + [abs(effects.get(s, 0.0)) >= thresholds[s] for s in schemes]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", choices=tuple(CIRCUITS))
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    n = 16 if args.quick else args.n
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.report_only:
        from phase9_report import write_report  # noqa: PLC0415

        write_report(RESULTS_DIR / "PHASE9_REPORT.md", RESULTS_DIR)
        print(f"rebuilt {RESULTS_DIR / 'PHASE9_REPORT.md'} from stored results")
        return 0

    if not args.circuit:
        parser.error("--circuit is required unless --report-only is given")

    assert_analysis_is_blind()
    return run_circuit(args.circuit, n, args.seed)


if __name__ == "__main__":
    sys.exit(main())
