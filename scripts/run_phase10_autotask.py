"""Phase 10: build the task from example prompts, then run the existing pipeline on it.

    python scripts/run_phase10_autotask.py --fixture frame_same --induction plan
    python scripts/run_phase10_autotask.py --fixture frame_same --induction shape
    python scripts/run_phase10_autotask.py --fixture frame_own  --induction plan
    python scripts/run_phase10_autotask.py --fixture frame_own  --induction shape
    python scripts/run_phase10_autotask.py --stage ksweep

`--induction plan` is section 3 of `results/PHASE10_PLAN.md` as pre-registered, and is
the phase's headline. `--induction shape` is the single repair fixed in
`results/PHASE10_AMENDMENT.md` after step 1 measured what the pre-registered rule costs;
it is post-hoc, it is labelled that way in every table it appears in, and it does not
replace the headline.

Nothing in this run is retuned. `n = 128`, `seed = 0`, the 0.02 cutoff from Phase 1, the
size-matched top-7 from Phase 6 — every one of them the setting Phases 6 and 8 used, so
the only thing that differs from Phase 6's greater-than run is that the task was induced
from 32 lines a person typed instead of written by hand.

**The answer key is not opened until every verdict above it has been decided.** The
scoring section is last and is separated by a banner, exactly as in Phase 8, because the
phase's question is whether an induced task locates the circuit — not whether it can be
made to after someone checks.
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

from causal_interp import autotask, comparison, induction, pipeline
from causal_interp import ground_truth_greater_than as gt
from causal_interp.metrics import METRICS
from causal_interp.model import load
from causal_interp.pipeline import collapse_positions, sweep_all_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIXTURES = ROOT / "fixtures"

# ---------------------------------------------------------------------------
# Inherited constants. Not one of these was chosen for this phase.
# ---------------------------------------------------------------------------
N_PROMPTS = 128            # Phases 1, 6, 8
SEED = 0                   # Phases 1, 6, 8
HEADLINE_THRESHOLD = 0.02  # Phase 1
PRIMARY_METRIC = "logit_diff"
SIZE_MATCHED_K = len(gt.ALL_HEADS)   # 7 — the published circuit's size

K_SWEEP = (2, 4, 8, 16, 32)

INDUCTION_MODES = {
    "plan": induction.FILTER_LENGTH,
    "shape": induction.FILTER_SHAPE,
}

# Where each fixture's start-year token sits, for the *scoring* section only. This is
# answer-key knowledge about the task — the same category as the published head list —
# and it is used nowhere before the banner.
FIXTURES_CONFIG = {
    "frame_same": {"file": FIXTURES / "greater_than_frame_same.txt", "yy_column": 8},
    "frame_own": {"file": FIXTURES / "greater_than_frame_own.txt", "yy_column": 9},
}


def _progress(done: int, total: int) -> None:
    if done % max(1, total // 20) == 0:
        print(".", end="", flush=True)


def _say(text: str) -> None:
    print(text, flush=True)


def _head_str(head: tuple[int, int]) -> str:
    return f"{head[0]}.{head[1]}"


def read_fixture(name: str, limit: int | None = None) -> list[str]:
    path = FIXTURES_CONFIG[name]["file"]
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[:limit] if limit else lines


def assert_firewall() -> None:
    """Neither induced-task module may import an answer key.

    Phases 4, 6 and 7 assert the same thing about `search.py`, and Phase 8 about
    `agreement.py`. A task *built* with the published circuit in reach would prove
    nothing about what can be built without one, so the check runs before the model
    loads rather than being promised in a docstring.
    """
    for name in ("induction", "autotask"):
        text = (ROOT / "causal_interp" / f"{name}.py").read_text(encoding="utf-8")
        assert "from causal_interp.ground_truth" not in text, f"{name}.py imports a ground truth"
        assert "import ground_truth" not in text, f"{name}.py imports a ground truth"


# ---------------------------------------------------------------------------
# Scoring — everything below here may consult the published circuit
# ---------------------------------------------------------------------------


def task_validity(model, ds, yy_column: int) -> dict:
    """Does the model actually perform greater-than on the *generated* clean prompts?

    Answer-key territory, and run only in the scoring section. The induced task has no
    idea what a correct continuation is; this asks how far its `clean_argmax_logprob`
    target — the model's own clean prediction — is from the published task's notion of
    a right answer.
    """
    with torch.no_grad():
        logits = model(ds.clean_tokens)
    rows = torch.arange(len(ds), device=logits.device)
    top = logits[rows, ds.positions["END"]].argmax(dim=-1)

    two_digit = 0
    greater = 0
    for i in range(len(ds)):
        text = model.to_string(top[i : i + 1])
        if len(text) == 2 and text.isdigit():
            two_digit += 1
            yy_text = model.to_string(ds.clean_tokens[i, yy_column : yy_column + 1])
            if yy_text.strip().isdigit() and int(text) > int(yy_text.strip()[-2:]):
                greater += 1
    return {
        "top_is_two_digit_year": two_digit / len(ds),
        "top_year_exceeds_start": greater / len(ds),
    }


def score(effects: dict, best_positions: dict, label: str) -> dict:
    """Size-matched and threshold comparisons against the published greater-than heads."""
    size_matched = comparison.compare(
        comparison.top_k_set(effects, SIZE_MATCHED_K), f"{label} top-{SIZE_MATCHED_K}", circuit=gt
    )
    at_cutoff = comparison.compare(
        comparison.threshold_set(effects, HEADLINE_THRESHOLD),
        f"{label} >= {HEADLINE_THRESHOLD}",
        circuit=gt,
    )
    ranked = sorted(effects, key=lambda h: abs(effects[h]), reverse=True)[:12]
    return {
        "label": label,
        "size_matched": {
            "recovered": len(size_matched.matches),
            "of": SIZE_MATCHED_K,
            "matches": [_head_str(h) for h in size_matched.matches],
            "misses": [_head_str(h) for h in size_matched.misses],
        },
        "at_cutoff": {
            "discovered": len(at_cutoff.discovered),
            "recovered": len(at_cutoff.matches),
            "precision": at_cutoff.precision,
        },
        "top_heads": [
            {
                "head": _head_str(h),
                "effect": effects[h],
                "position": best_positions.get(h, "-"),
                "published": gt.classify(h) or "",
            }
            for h in ranked
        ],
    }


# ---------------------------------------------------------------------------
# Run B / C — full multi-scheme discovery on one induced task
# ---------------------------------------------------------------------------


def run_discovery(fixture: str, mode: str, n: int, seed: int) -> int:
    assert_firewall()
    started = time.time()
    filter_mode = INDUCTION_MODES[mode]

    print(f"\n{'#' * 72}")
    print(f"# Phase 10 — induced greater-than task from fixtures/{fixture}")
    print(f"# induction: {mode} ({filter_mode} filter)"
          + ("   [PRE-REGISTERED]" if mode == "plan" else "   [POST-HOC — amendment]"))
    print(f"# threshold {HEADLINE_THRESHOLD} (Phase 1), size-matched top-{SIZE_MATCHED_K} (Phase 6)")
    print(f"{'#' * 72}")

    model = load("gpt2-small")
    examples = read_fixture(fixture)

    built = autotask.build(
        model, examples, name=f"induced-{fixture}-{mode}", n=n, seed=seed,
        filter_mode=filter_mode,
    )
    task = built.task
    structure = built.structure

    print(f"\n  examples kept        {structure.n_examples_kept} / {structure.n_examples_given}")
    for dropped in structure.dropped:
        print(f"    dropped ({dropped['reason']}): {dropped['text']!r}")
    print(f"  slots                {len(structure.slots)}")
    for slot in structure.slots:
        print(f"    {slot.label:5s} columns={list(slot.columns)} tied={slot.is_tied} "
              f"values={len(slot.values)}")
    print(f"  positions            {list(task.positions)}")
    print(f"  generated            {built.generated.count} distinct "
          f"(round-trip rejected {built.generated.rejected_round_trip})")
    print(f"  schemes              {list(task.discovery_schemes)}")
    for scheme, value in sorted(built.divergences.items(), key=lambda kv: -kv[1]):
        mark = "  <- primary" if scheme == built.primary else ""
        print(f"    {scheme:20s} KL {value:7.3f}{mark}")

    discovery = pipeline.discover(
        model, task, n=built.generated.count, seed=seed,
        threshold=HEADLINE_THRESHOLD, progress=_progress, announce=_say,
    )

    report = discovery.agreement[PRIMARY_METRIC]
    print(f"\n{'=' * 72}\ncross-scheme agreement — activation patching\n{'=' * 72}")
    for scheme in report.schemes:
        power = report.power[scheme]
        run = discovery.runs[scheme]
        flag = "  LOW-POWER" if power.low_power else ""
        span_flag = "   SPAN <= 0 — not normalizable" if run.span <= 0 else ""
        print(f"  {scheme:20} found {len(report.per_scheme[scheme]):3}   "
              f"span {power.span:+8.3f}  power {power.power:.2f}{flag}{span_flag}")

    # ---------------------------------------------------------------------
    print(f"\n{'#' * 72}\n# ANSWER KEY OPENS HERE — nothing above consulted it\n{'#' * 72}")

    primary_run = discovery.runs[built.primary]
    scored = {
        metric: score(
            primary_run.effects[metric], primary_run.best_positions[metric],
            f"{built.primary} / {metric}",
        )
        for metric in METRICS
    }

    print(f"\nprimary scheme: {built.primary}")
    for metric in METRICS:
        block = scored[metric]
        print(f"  {metric:12s} size-matched {block['size_matched']['recovered']}/{SIZE_MATCHED_K}"
              f"   at {HEADLINE_THRESHOLD}: {block['at_cutoff']['recovered']}"
              f"/{block['at_cutoff']['discovered']} discovered"
              f"   precision {block['at_cutoff']['precision']:.2f}")

    print("\ntop heads under the primary scheme, headline metric:")
    for row in scored[PRIMARY_METRIC]["top_heads"]:
        mark = f"  PUBLISHED ({row['published']})" if row["published"] else ""
        print(f"  {row['head']:6s} {row['effect']:+.4f}  @{row['position']:5s}{mark}")

    per_scheme = {
        scheme: score(run.effects[PRIMARY_METRIC], run.best_positions[PRIMARY_METRIC], scheme)
        for scheme, run in discovery.runs.items()
    }
    print("\nevery scheme, headline metric, size-matched:")
    for scheme, block in per_scheme.items():
        mark = " (primary)" if scheme == built.primary else ""
        print(f"  {scheme:20s} {block['size_matched']['recovered']}/{SIZE_MATCHED_K}"
              f"   precision {block['at_cutoff']['precision']:.2f}{mark}")

    union: set = set()
    for run in discovery.runs.values():
        union |= comparison.threshold_set(run.effects[PRIMARY_METRIC], HEADLINE_THRESHOLD)
    union_cmp = comparison.compare(union, "union across schemes", circuit=gt)
    print(f"\nunion across all schemes at {HEADLINE_THRESHOLD}: "
          f"{len(union_cmp.matches)}/{SIZE_MATCHED_K} of the published heads, "
          f"{len(union)} discovered, precision {union_cmp.precision:.2f}")

    primary_ds = task.dataset(
        model, n=built.generated.count, corruption=built.primary, seed=seed
    )
    validity = task_validity(model, primary_ds, FIXTURES_CONFIG[fixture]["yy_column"])
    print(f"\nis the generated task actually greater-than?")
    print(f"  clean top-1 is a two-digit year      {validity['top_is_two_digit_year']:.0%}")
    print(f"  clean top-1 exceeds the start year   {validity['top_year_exceeds_start']:.0%}")

    payload = {
        "meta": {
            "fixture": fixture,
            "induction": mode,
            "filter_mode": filter_mode,
            "pre_registered": mode == "plan",
            "n_requested": n,
            "seed": seed,
            "threshold": HEADLINE_THRESHOLD,
            "size_matched_k": SIZE_MATCHED_K,
            "model": "gpt2-small",
            "torch": torch.__version__,
            "transformer_lens": version("transformer_lens"),
            "platform": platform.platform(),
            "runtime_seconds": round(time.time() - started, 1),
        },
        "built": built.as_dict(lambda t: model.to_string([t])),
        "discovery": discovery.as_dict(),
        "scored_primary": scored,
        "scored_per_scheme": per_scheme,
        "union": {
            "discovered": sorted(_head_str(h) for h in union),
            "recovered": len(union_cmp.matches),
            "precision": union_cmp.precision,
        },
        "task_validity": validity,
    }
    out = RESULTS / f"phase10_{fixture}_{mode}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}   ({payload['meta']['runtime_seconds']}s)")

    rows = [
        {
            "scheme": scheme,
            "size_matched": block["size_matched"]["recovered"],
            "precision": f"{block['at_cutoff']['precision']:.4f}",
            "discovered": block["at_cutoff"]["discovered"],
            "primary": scheme == built.primary,
        }
        for scheme, block in per_scheme.items()
    ]
    csv_path = RESULTS / f"phase10_{fixture}_{mode}_schemes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# Run D — how many examples does the human have to write?
# ---------------------------------------------------------------------------


def run_ksweep(n: int, seed: int) -> int:
    """Sweep the primary scheme only, for k example lines, k in K_SWEEP.

    A recovery curve against the size of the human's input, not a circuit claim, which
    is why it sweeps one scheme rather than every scheme — declared as an exception in
    section 6 of the plan before it ran.
    """
    assert_firewall()
    started = time.time()
    model = load("gpt2-small")
    out: dict = {"meta": {"seed": seed, "k_values": list(K_SWEEP)}, "rows": []}

    for mode, filter_mode in INDUCTION_MODES.items():
        for k in K_SWEEP:
            label = f"k={k:<3} induction={mode}"
            print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
            examples = read_fixture("frame_same", limit=k)
            try:
                built = autotask.build(
                    model, examples, name=f"induced-k{k}-{mode}", n=n, seed=seed,
                    filter_mode=filter_mode,
                )
            except ValueError as exc:
                print(f"  BUILD FAILED: {exc}")
                out["rows"].append(
                    {"k": k, "induction": mode, "failed": str(exc)}
                )
                continue

            print(f"  kept {built.structure.n_examples_kept}/{k}   "
                  f"slots {len(built.structure.slots)}   "
                  f"tied {sum(1 for s in built.structure.slots if s.is_tied)}   "
                  f"generated {built.generated.count}/{n}   "
                  f"primary {built.primary}")

            ds = built.task.dataset(
                model, n=built.generated.count, corruption=built.primary, seed=seed
            )
            from causal_interp.interventions import baseline_for, clean_cache_for
            from causal_interp.metrics import DistributionalBaseline

            baseline, clean_logits, corrupted_logits = baseline_for(model, ds)
            dist = DistributionalBaseline(ds, clean_logits, corrupted_logits)
            cache, _ = clean_cache_for(model, ds)
            grids = sweep_all_metrics(
                model, ds, cache, baseline, dist, built.task.positions, _progress
            )
            effects, best = collapse_positions(grids[PRIMARY_METRIC], built.task.positions)
            block = score(effects, best, label)
            print(f"\n  size-matched {block['size_matched']['recovered']}/{SIZE_MATCHED_K}"
                  f"   precision {block['at_cutoff']['precision']:.2f}"
                  f"   span {baseline.span:+.3f}")

            out["rows"].append(
                {
                    "k": k,
                    "induction": mode,
                    "kept": built.structure.n_examples_kept,
                    "slots": len(built.structure.slots),
                    "tied": sum(1 for s in built.structure.slots if s.is_tied),
                    "generated": built.generated.count,
                    "primary": built.primary,
                    "span": baseline.span,
                    "size_matched": block["size_matched"]["recovered"],
                    "matches": block["size_matched"]["matches"],
                    "precision": block["at_cutoff"]["precision"],
                    "discovered": block["at_cutoff"]["discovered"],
                }
            )

    out["meta"]["runtime_seconds"] = round(time.time() - started, 1)
    path = RESULTS / "phase10_ksweep.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(ROOT)}   ({out['meta']['runtime_seconds']}s)")
    return 0


def run_pairs(n: int, seed: int) -> int:
    """Post-hoc robustness check: does the k = 2 result depend on *which* two lines?

    Not in the plan. Added because run D came out inverted — two examples recovered more
    published heads than thirty-two — and a surprising number resting on one arbitrary
    pair of sentences is worth trying to break before it is reported. Every pair below
    is a contiguous slice of the fixture, chosen by position and not by result, and all
    of them are reported.
    """
    assert_firewall()
    started = time.time()
    model = load("gpt2-small")
    all_lines = read_fixture("frame_same")
    # Fixed before running: five contiguous pairs spread across the file, plus the two
    # pairs that straddle the tokenizer-odd lines 4 and 6, which are the interesting
    # adversarial cases rather than the flattering ones.
    pairs = [(0, 1), (4, 5), (6, 7), (10, 11), (20, 21), (30, 31), (2, 3)]
    out: dict = {"meta": {"seed": seed, "n": n}, "rows": []}

    for first, second in pairs:
        examples = [all_lines[first], all_lines[second]]
        for mode, filter_mode in INDUCTION_MODES.items():
            label = f"lines {first},{second}  induction={mode}"
            print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
            for text in examples:
                print(f"    {text}")
            try:
                built = autotask.build(
                    model, examples, name=f"pair-{first}-{second}-{mode}", n=n, seed=seed,
                    filter_mode=filter_mode,
                )
            except ValueError as exc:
                print(f"  BUILD FAILED: {exc}")
                out["rows"].append(
                    {"pair": [first, second], "induction": mode, "failed": str(exc)}
                )
                continue

            tied = sum(1 for s in built.structure.slots if s.is_tied)
            print(f"  kept {built.structure.n_examples_kept}/2   "
                  f"slots {len(built.structure.slots)}   tied {tied}   "
                  f"generated {built.generated.count}   primary {built.primary}")

            ds = built.task.dataset(
                model, n=built.generated.count, corruption=built.primary, seed=seed
            )
            from causal_interp.interventions import baseline_for, clean_cache_for
            from causal_interp.metrics import DistributionalBaseline

            baseline, clean_logits, corrupted_logits = baseline_for(model, ds)
            dist = DistributionalBaseline(ds, clean_logits, corrupted_logits)
            cache, _ = clean_cache_for(model, ds)
            grids = sweep_all_metrics(
                model, ds, cache, baseline, dist, built.task.positions, _progress
            )
            effects, best = collapse_positions(grids[PRIMARY_METRIC], built.task.positions)
            block = score(effects, best, label)
            validity = task_validity(model, ds, FIXTURES_CONFIG["frame_same"]["yy_column"])
            print(f"\n  size-matched {block['size_matched']['recovered']}/{SIZE_MATCHED_K}"
                  f"   precision {block['at_cutoff']['precision']:.2f}"
                  f"   span {baseline.span:+.3f}"
                  f"   task valid {validity['top_year_exceeds_start']:.0%}")

            out["rows"].append(
                {
                    "pair": [first, second],
                    "induction": mode,
                    "kept": built.structure.n_examples_kept,
                    "slots": len(built.structure.slots),
                    "tied": tied,
                    "generated": built.generated.count,
                    "primary": built.primary,
                    "span": baseline.span,
                    "size_matched": block["size_matched"]["recovered"],
                    "matches": block["size_matched"]["matches"],
                    "precision": block["at_cutoff"]["precision"],
                    "task_valid": validity["top_year_exceeds_start"],
                }
            )

    out["meta"]["runtime_seconds"] = round(time.time() - started, 1)
    path = RESULTS / "phase10_pairs.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(ROOT)}   ({out['meta']['runtime_seconds']}s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=sorted(FIXTURES_CONFIG))
    parser.add_argument("--induction", choices=sorted(INDUCTION_MODES), default="plan")
    parser.add_argument("--stage", choices=("discover", "ksweep", "pairs"), default="discover")
    parser.add_argument("--n", type=int, default=N_PROMPTS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.stage == "ksweep":
        return run_ksweep(args.n, args.seed)
    if args.stage == "pairs":
        return run_pairs(args.n, args.seed)
    if not args.fixture:
        parser.error("--fixture is required unless --stage ksweep or --stage pairs")
    return run_discovery(args.fixture, args.induction, args.n, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
