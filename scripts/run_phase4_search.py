"""Phase 4 end to end: search for receiver specifications, then check the search.

    python scripts/run_phase4_search.py            # full run
    python scripts/run_phase4_search.py --quick    # smoke test

The space and the budget were fixed in `results/PHASE4_SEARCH_SPACE.md`, committed
before this script was written. Two exhaustive stage-A screens are run:

- **semantic** — all 8 templates, positions labelled IO / S1 / S2 / END and so on.
  Comparable with Phases 2-3, but those labels encode knowledge of the task.
- **absolute** — one template, positions labelled `t0 … tN` with no meaning
  attached. Nothing about the task's structure is supplied, which makes this the
  one that actually tests whether the search can stand on its own.

The published circuit is consulted only after both searches have produced output.
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

from causal_interp import ground_truth, search
from causal_interp.ground_truth import Head, classify
from causal_interp.interventions import baseline_for, cache_for
from causal_interp.ioi import POSITIONS, TEMPLATES, IOIDataset
from causal_interp.model import load
from causal_interp.search import ReceiverSpec, absolute_positions, screen_specs

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PREREG_JSON = RESULTS_DIR / "phase3_preregistration.json"

CACHE_KINDS = ("z", "mlp_out", "q", "k", "v")
TOP_K_CONFIRM = 20  # fixed in the search-space document, before any result

# How the rediscovery check labels an outcome. Fixed here rather than
# pre-registered: the search-space document named the three outcomes but not the
# margin, so this rule is stated openly and the raw ranks are reported beside it.
AMBIGUITY_MARGIN = 0.20  # within 20% of the top score counts as not distinguished
AMBIGUITY_MAX_RANK = 3


def assert_search_is_blind() -> None:
    """Fail loudly if the search module can see the answer key.

    The claim that the search does not consult ground truth should be checkable
    rather than promised, and this is the cheapest way to check it.
    """
    source = (Path(__file__).resolve().parents[1] / "causal_interp" / "search.py").read_text(
        encoding="utf-8"
    )
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and "ground_truth" in stripped:
            raise SystemExit(f"search.py imports ground truth: {stripped!r}")
    print("search.py does not import ground_truth — ok")


def run_screen(model, label: str, ds, positions, progress_every: int = 20) -> dict:
    baseline, _, _ = baseline_for(model, ds)
    clean_cache, _ = cache_for(model, ds.clean_tokens, CACHE_KINDS)
    print(f"\n  {label}: {len(positions)} positions x 3 inputs x 144 heads "
          f"= {len(positions) * 3 * 144} specs ", end="", flush=True)
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        if done % max(1, total // progress_every) == 0:
            print(".", end="", flush=True)

    scores = screen_specs(model, ds, clean_cache, baseline, positions, progress=progress)
    print(f" {time.time() - t0:.0f}s")
    ranked = sorted(scores, key=lambda s: abs(scores[s]), reverse=True)
    top = ", ".join(f"{s} {scores[s]:+.3f}" for s in ranked[:5])
    print(f"    top: {top}")
    return {"label": label, "scores": scores, "ranked": ranked, "baseline": baseline, "ds": ds}


def rediscovery_check(scores: dict[ReceiverSpec, float]) -> list[dict]:
    """Where does the paper's named receiver spec sit in the search's own ranking?

    Runs only after the search is complete. For each published head that has a
    published receiver specification, rank that head's 21 candidate specs by the
    search's score and report where the published one landed.
    """
    rows = []
    for head in sorted(ground_truth.ALL_HEADS):
        published = ground_truth.receiver_spec(head)
        if published is None:
            rows.append({
                "head": f"{head[0]}.{head[1]}",
                "class": classify(head),
                "published_spec": None,
                "outcome": "no published spec",
            })
            continue

        ranked = search.rank_specs_for_head(scores, head[0], head[1])
        if not ranked:
            rows.append({
                "head": f"{head[0]}.{head[1]}", "class": classify(head),
                "published_spec": f"{published[0]}@{published[1]}", "outcome": "not scored",
            })
            continue

        want_input, want_position = published
        rank = next(
            (i for i, (spec, _) in enumerate(ranked)
             if spec.input == want_input and spec.position == want_position),
            None,
        )
        top_spec, top_score = ranked[0]
        published_score = ranked[rank][1] if rank is not None else 0.0
        gap = (abs(top_score) - abs(published_score)) / abs(top_score) if top_score else 1.0

        if rank == 0:
            outcome = "agreement"
        elif published_score == 0.0:
            # The corruption scheme makes this specification bit-identical between
            # the two runs, so the screen cannot score it at all. Calling that a
            # disagreement would blame the search for a property of the
            # counterfactual it was handed.
            outcome = "unmeasurable"
        elif rank is not None and rank < AMBIGUITY_MAX_RANK and gap <= AMBIGUITY_MARGIN:
            outcome = "ambiguous"
        else:
            outcome = "disagreement"

        rows.append({
            "head": f"{head[0]}.{head[1]}",
            "class": classify(head),
            "published_spec": f"{want_input}@{want_position}",
            "published_rank": None if rank is None else rank + 1,
            "published_score": published_score,
            "top_spec": f"{top_spec.input}@{top_spec.position}",
            "top_score": top_score,
            "relative_gap": gap,
            "outcome": outcome,
        })
    return rows


def semantic_of_absolute(ds: IOIDataset, index: int) -> str:
    """What a bare token index turns out to be, used only to interpret results.

    A label is reported only if it holds for every prompt, not just the first.
    """
    labels = [name for name in POSITIONS if bool((ds.positions[name] == index).all())]
    return "/".join(labels) if labels else "—"


def _load_scores(path: Path) -> dict:
    """Rebuild a stage-A score table from the CSV a previous run wrote."""
    scores: dict[ReceiverSpec, float] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            spec = ReceiverSpec(
                int(row["layer"]), int(row["head"]), row["input"], row["position"]
            )
            scores[spec] = float(row["screen_score"])
    ranked = sorted(scores, key=lambda s: abs(scores[s]), reverse=True)
    return {"scores": scores, "ranked": ranked}


def rebuild_report() -> int:
    """Regenerate the report from a previous run's artefacts.

    The stage-A grids are the expensive part and they are already on disk, so a
    change to how the results are written should not cost another half hour of
    GPU time. Nothing is recomputed; the rediscovery check is re-derived from the
    stored scores, which is what picks up changes to how outcomes are labelled.
    """
    payload = json.loads((RESULTS_DIR / "phase4_results.json").read_text(encoding="utf-8"))
    semantic = _load_scores(RESULTS_DIR / "receiver_search_semantic.csv")
    absolute = _load_scores(RESULTS_DIR / "receiver_search_absolute.csv")
    if "absolute_labels" in payload:
        abs_labels = {int(k): v for k, v in payload["absolute_labels"].items()}
    else:
        # A run from before the labels were stored separately: recover them from the
        # per-specification records, which carry the same mapping.
        abs_labels = {
            entry["index"]: entry["is_semantically"] for entry in payload.get("absolute_top", [])
        }
        payload["absolute_labels"] = {str(k): v for k, v in abs_labels.items()}
    # The report reads tuple-keyed views; build them on copies so the payload that
    # gets written back stays JSON-serialisable.
    confirmations = [
        {
            **entry,
            "_signals": {tuple(map(int, k.split("."))): v for k, v in entry["signals"].items()},
            "_effects": {tuple(map(int, k.split("."))): v for k, v in entry["effects"].items()},
        }
        for entry in payload["confirmations"]
    ]

    check = rediscovery_check(semantic["scores"])
    payload["rediscovery"] = check

    from phase4_report import write_report  # noqa: PLC0415

    write_report(
        RESULTS_DIR / "PHASE4_REPORT.md", payload["meta"], semantic, absolute, abs_labels,
        check, confirmations, payload["meta"]["signal_threshold"],
    )
    (RESULTS_DIR / "phase4_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"rebuilt {RESULTS_DIR / 'PHASE4_REPORT.md'} from stored results (no GPU work)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--report-only", action="store_true",
        help="rebuild the report from the CSVs and JSON of a previous run, without the GPU sweep",
    )
    args = parser.parse_args()
    n = 16 if args.quick else args.n

    assert_search_is_blind()
    if args.report_only:
        return rebuild_report()
    if not PREREG_JSON.exists():
        raise SystemExit("missing Phase 3 pre-registration; run Phase 3 first")
    threshold = json.loads(PREREG_JSON.read_text(encoding="utf-8"))["threshold"]
    print(f"reusing Phase 3's recorded signal threshold {threshold}")

    RESULTS_DIR.mkdir(exist_ok=True)
    started = time.time()
    model = load()

    # -- stage A, twice ------------------------------------------------------
    semantic_ds = IOIDataset(model, n=n, corruption="s2_swap", seed=args.seed)
    semantic = run_screen(model, "semantic positions", semantic_ds, POSITIONS)

    # One template *and* one name order: the two orders put the indirect object and
    # the subject at swapped token indices, so mixing them would make a bare index
    # mean two different things and the search would average over both.
    absolute_ds = IOIDataset(
        model, n=n, corruption="s2_swap", seed=args.seed,
        templates=(TEMPLATES[0],), orders=("ABB",),
    )
    abs_positions = absolute_positions(absolute_ds)
    absolute = run_screen(model, "absolute positions", absolute_ds, abs_positions)

    # -- stage B on the survivors -------------------------------------------
    print(f"\n  stage B: confirming top {TOP_K_CONFIRM} semantic specs")
    clean_cache, _ = cache_for(model, semantic_ds.clean_tokens, CACHE_KINDS)
    corrupted_cache, _ = cache_for(model, semantic_ds.corrupted_tokens, CACHE_KINDS)
    confirmations = []
    for spec in semantic["ranked"][:(3 if args.quick else TOP_K_CONFIRM)]:
        if spec.layer == 0:
            continue  # nothing upstream to sweep
        result = search.confirm_spec(
            model, semantic_ds, clean_cache, corrupted_cache, semantic["baseline"],
            spec, threshold,
        )
        result["screen_score"] = semantic["scores"][spec]
        confirmations.append(result)
        top = sorted(result["_signals"], key=lambda k: abs(result["_signals"][k]), reverse=True)[:3]
        print(f"    {spec}: top senders " + ", ".join(
            f"{l}.{h} sig {result['_signals'][(l, h)]:+.2f}" for l, h in top) or "none")

    # -- only now: the answer key -------------------------------------------
    semantic_check = rediscovery_check(semantic["scores"])

    meta = {
        "model": model.cfg.model_name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "transformer_lens": version("transformer_lens"),
        "python": platform.python_version(),
        "prompts": n,
        "seed": args.seed,
        "signal_threshold": threshold,
        "top_k_confirmed": TOP_K_CONFIRM,
        "ambiguity_rule": f"published spec in top {AMBIGUITY_MAX_RANK} and within "
                          f"{AMBIGUITY_MARGIN:.0%} of the best score",
        "runtime_seconds": None,
    }

    _write_csv(semantic, absolute, RESULTS_DIR)
    meta["runtime_seconds"] = round(time.time() - started, 1)

    from phase4_report import write_report  # noqa: PLC0415

    abs_labels = {
        index: semantic_of_absolute(absolute_ds, index) for index in range(len(abs_positions))
    }
    write_report(
        RESULTS_DIR / "PHASE4_REPORT.md", meta, semantic, absolute, abs_labels,
        semantic_check, confirmations, threshold,
    )

    payload = {
        "meta": meta,
        "absolute_labels": {str(k): v for k, v in abs_labels.items()},
        "semantic_top": [
            {"spec": str(s), "score": semantic["scores"][s]} for s in semantic["ranked"][:60]
        ],
        "absolute_top": [
            {"spec": str(s), "score": absolute["scores"][s],
             "index": int(s.position[1:]), "is_semantically": semantic_of_absolute(absolute_ds, int(s.position[1:]))}
            for s in absolute["ranked"][:60]
        ],
        "rediscovery": semantic_check,
        "confirmations": [
            {k: v for k, v in c.items() if not k.startswith("_")} for c in confirmations
        ],
    }
    (RESULTS_DIR / "phase4_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR / 'PHASE4_REPORT.md'}")
    print(f"total runtime {meta['runtime_seconds']}s")
    return 0


def _write_csv(semantic: dict, absolute: dict, results_dir: Path) -> None:
    for name, block in (("semantic", semantic), ("absolute", absolute)):
        with (results_dir / f"receiver_search_{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.writer(f)
            w.writerow(["layer", "head", "input", "position", "screen_score", "published_class"])
            for spec, score in block["scores"].items():
                w.writerow([
                    spec.layer, spec.head, spec.input, spec.position,
                    f"{score:.6f}", classify((spec.layer, spec.head)) or "",
                ])


if __name__ == "__main__":
    sys.exit(main())
