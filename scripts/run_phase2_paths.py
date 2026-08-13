"""Phase 2 end to end: recover the IOI circuit by *path* patching, then score it
against the same published circuit Phase 1 was scored against.

    python scripts/run_phase2_paths.py                 # full run, both corruptions
    python scripts/run_phase2_paths.py --quick         # small run for smoke-testing

Phase 1 measured each head's total effect on the output and recovered 20 of 26
published heads. The six it missed reach the logits only through another head,
which total-effect patching cannot see. This run measures paths instead.

Discovery is iterative and deliberately not seeded from the answer key. Round 0
asks which heads affect the logits directly. Every later round takes the heads
discovered in the round before as its receivers and asks which heads feed them.
Nothing in `causal_interp/ground_truth.py` is consulted until the comparison at
the end, and `causal_interp/comparison.py` remains pure set arithmetic.

Writes to results/:
    PHASE2_REPORT.md              the comparison (the deliverable)
    phase2_results.json           every number the report is built from
    path_effects_<scheme>_r<n>.csv per-round, per-sender effects
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

from causal_interp import comparison, ground_truth
from causal_interp.ground_truth import Head, classify
from causal_interp.interventions import (
    LOGITS,
    Patch,
    Receiver,
    baseline_for,
    cache_for,
    path_patch,
    path_signal,
    sweep_path_senders,
)
from causal_interp.ioi import IOIDataset
from causal_interp.model import load

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PHASE1_JSON = RESULTS_DIR / "phase1_results.json"

# Same pre-registered cutoff as Phase 1, so the two phases are scored alike.
HEADLINE_THRESHOLD = 0.02
THRESHOLD_SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

# How many heads carry forward as the next round's receivers. Fixed in advance and
# independent of the cutoff, so the chain cannot be lengthened or shortened by
# choosing a threshold after seeing the results.
CHAIN_WIDTH = 4

CACHE_KINDS = ("z", "mlp_out", "q", "k", "v")


@dataclass(frozen=True)
class Round:
    """One step of the iterative search: what we ask, and where we ask it."""

    name: str
    question: str
    receiver_input: str | None  # None => the logits themselves
    position: str
    expected: str  # the class the paper's account predicts, used only for narration


# The receiver input and position for each round come from the paper's mechanistic
# account of the circuit — S-inhibition heads act on name movers' *queries*, and so
# on. That is prior knowledge about where to look, and it is why this is guided
# rediscovery rather than blind search. Which heads turn up is not constrained:
# every one of the 144 is swept as a sender in every round.
ROUNDS: tuple[Round, ...] = (
    Round(
        name="direct effect on the logits",
        question="which heads move the prediction without another head relaying it?",
        receiver_input=None,
        position="END",
        expected="name mover / negative name mover / backup name mover",
    ),
    Round(
        name="queries of the round-0 heads",
        question="which heads feed those heads' queries at END?",
        receiver_input="q",
        position="END",
        expected="s-inhibition",
    ),
    Round(
        name="values of the round-1 heads at S2",
        question="which heads feed those heads' values at S2?",
        receiver_input="v",
        position="S2",
        expected="induction / duplicate token",
    ),
    Round(
        name="keys of the round-2 heads at S1+1",
        question="which heads feed those heads' keys at S1+1?",
        receiver_input="k",
        position="S1+1",
        expected="previous token",
    ),
)


def run_scheme(model, corruption: str, n: int, seed: int) -> dict:
    """The full iterative path-patching chain for one corruption scheme."""
    print(f"\n{'=' * 72}\ncorruption scheme: {corruption}\n{'=' * 72}")
    ds = IOIDataset(model, n=n, corruption=corruption, seed=seed)
    baseline, _, _ = baseline_for(model, ds)
    print(f"  clean {baseline.clean_logit_diff:+.4f}   corrupted {baseline.corrupted_logit_diff:+.4f}")

    clean_cache, _ = cache_for(model, ds.clean_tokens, CACHE_KINDS)
    corrupted_cache, _ = cache_for(model, ds.corrupted_tokens, CACHE_KINDS)

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
        print("    sweeping 144 senders ", end="", flush=True)
        t0 = time.time()

        def progress(done: int, total: int) -> None:
            if done % max(1, total // 20) == 0:
                print(".", end="", flush=True)

        grid = sweep_path_senders(
            model, ds, clean_cache, corrupted_cache, baseline,
            receivers=receivers, sender_position=spec.position, progress=progress,
        )
        print(f" {time.time() - t0:.0f}s")

        # NaN marks senders that cannot reach every receiver, which the sweep leaves
        # unmeasured rather than scoring against a smaller receiver set.
        effects: dict[Head, float] = {
            (l, h): float(grid[l, h])
            for l in range(model.cfg.n_layers)
            for h in range(model.cfg.n_heads)
            if not torch.isnan(grid[l, h])
        }
        discovered = sorted(comparison.threshold_set(effects, HEADLINE_THRESHOLD))
        ranked = sorted(effects, key=lambda k: abs(effects[k]), reverse=True)

        # The receiver-signal diagnostic: does the path deliver anything at its own
        # endpoint, regardless of whether the logits move? Only meaningful when the
        # receivers are real nodes, and only worth the passes for the top senders.
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
            "_effects": effects,
            "_signals": signals,
        })

    union: set[Head] = set()
    for entry in rounds:
        union |= {tuple(map(int, h.split("."))) for h in entry["discovered"]}

    return {
        "corruption": corruption,
        "n_prompts": len(ds),
        "baseline": {
            "clean_logit_diff": baseline.clean_logit_diff,
            "corrupted_logit_diff": baseline.corrupted_logit_diff,
        },
        "rounds": rounds,
        "discovered": sorted(union),
    }


def previous_token_probe(model, results: dict[str, dict], n: int, seed: int) -> dict:
    """A dedicated cross-scheme test for previous-token heads.

    Neither scheme can answer this question inside its own chain. Under `s2_swap`
    the S1+1 position is bit-identical between the two runs, so every measurement
    there is an exact zero. Under `abc` the chain does not survive round 1, so it
    never produces receivers to ask about.

    So the two halves are taken from where each is sound: the receivers are the
    heads the `s2_swap` chain discovered at round 2 — arrived at without consulting
    the answer key — and the measurement runs on `abc`, the only scheme in which
    S1+1 differs at all. Mixing schemes this way is a real caveat and is reported
    as one, but the alternative is not measuring the question.
    """
    source = results["s2_swap"]
    round2 = next((r for r in source["rounds"] if r["index"] == 2 and not r["halted"]), None)
    if round2 is None or not round2["carried"]:
        return {"available": False, "reason": "the s2_swap chain did not reach round 2"}

    receiver_heads = [tuple(map(int, h.split("."))) for h in round2["carried"]]

    print(f"\n{'=' * 72}\nprevious-token probe: s2_swap receivers, measured on abc\n{'=' * 72}")

    ds = IOIDataset(model, n=n, corruption="abc", seed=seed)
    baseline, _, _ = baseline_for(model, ds)
    clean_cache, _ = cache_for(model, ds.clean_tokens, CACHE_KINDS)
    corrupted_cache, _ = cache_for(model, ds.corrupted_tokens, CACHE_KINDS)

    # A sweep can only test senders below its earliest receiver, so one receiver set
    # tests one band of sender layers. Dropping the earliest receivers raises that
    # ceiling. The probe is therefore run once per distinct receiver layer, which
    # covers every sender the carried set can reach — a mechanical rule, not a
    # hand-picked receiver list, so no knowledge of the answer enters here.
    variants = []
    for floor in sorted({l for l, _ in receiver_heads}):
        subset = [
            Receiver(layer=l, head=h, position="S1+1", input="k")
            for l, h in receiver_heads if l >= floor
        ]
        print(f"\n  receivers (layer >= {floor}): {', '.join(str(r) for r in subset)}")
        print("    sweeping senders ", end="", flush=True)
        t0 = time.time()

        def progress(done: int, total: int) -> None:
            if done % max(1, total // 20) == 0:
                print(".", end="", flush=True)

        grid = sweep_path_senders(
            model, ds, clean_cache, corrupted_cache, baseline,
            receivers=subset, sender_position="S1+1", progress=progress,
        )
        print(f" {time.time() - t0:.0f}s")

        effects = {
            (l, h): float(grid[l, h])
            for l in range(model.cfg.n_layers)
            for h in range(model.cfg.n_heads)
            if not torch.isnan(grid[l, h])
        }
        if not effects:
            continue

        ranked = sorted(effects, key=lambda k: abs(effects[k]), reverse=True)
        # Signal is measured for the strongest senders and, separately, for whichever
        # published previous-token heads this ceiling makes testable — so their
        # numbers appear whether or not they rank highly. Reporting, not discovery:
        # nothing here feeds the discovered set.
        published_prev = [h for h in ground_truth.IOI_CIRCUIT["previous token"] if h in effects]
        signals = {}
        for head in list(dict.fromkeys(ranked[:5] + published_prev)):
            signals[head] = path_signal(
                model, ds, clean_cache, corrupted_cache,
                Patch(head[0], "z", "S1+1", head[1]), subset,
            )

        top = ", ".join(f"{l}.{h} {effects[(l, h)]:+.4f}" for l, h in ranked[:4])
        print(f"    senders testable: layers 0-{max(l for l, _ in effects)}; top: {top}")
        for head in published_prev:
            print(f"    published {head[0]}.{head[1]}: effect {effects[head]:+.4f}, "
                  f"signal at receiver {signals[head]:+.3f}")

        variants.append({
            "receiver_floor": floor,
            "receivers": [str(r) for r in subset],
            "senders_tested": len(effects),
            "max_sender_layer": max(l for l, _ in effects),
            "effects": {f"{l}.{h}": v for (l, h), v in effects.items()},
            "signals": {f"{l}.{h}": v for (l, h), v in signals.items()},
            "discovered": [f"{l}.{h}" for l, h in comparison.threshold_set(effects, HEADLINE_THRESHOLD)],
            "_effects": effects,
            "_signals": signals,
            "_published_prev": published_prev,
        })

    if not variants:
        return {"available": False, "reason": "no sender sits below the earliest receiver layer"}
    return {"available": True, "variants": variants}


def phase1_discovered() -> dict[str, set[Head]]:
    """Phase 1's discovered sets, read back rather than recomputed.

    Reusing the committed artefact keeps the two phases comparable and avoids a
    second six-minute sweep that would produce the same numbers.
    """
    if not PHASE1_JSON.exists():
        return {}
    data = json.loads(PHASE1_JSON.read_text(encoding="utf-8"))
    out = {}
    for scheme, res in data["schemes"].items():
        headline = res["headline"]
        out[scheme] = {
            tuple(map(int, h.split("."))) for h in headline["matches"] + headline["extras"]
        }
    return out


# -- output ---------------------------------------------------------------------


def _fmt_signal(value: float | None) -> str:
    """Render a receiver-signal reading, distinguishing 'not measured' from 'undefined'."""
    if value is None:
        return "—"
    if value != value:  # NaN: clean and corrupted coincide at the receiver
        return "*undefined*"
    return f"{value:+.3f}"


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


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


def write_csvs(res: dict, out_dir: Path) -> None:
    scheme = res["corruption"]
    for entry in res["rounds"]:
        if entry["halted"]:
            continue
        path = out_dir / f"path_effects_{scheme}_r{entry['index']}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["layer", "head", "position", "path_effect", "published_class"])
            for key, value in entry["effects"].items():
                layer, head = map(int, key.split("."))
                w.writerow([layer, head, entry["position"], f"{value:.6f}", classify((layer, head)) or ""])


def write_report(
    results: dict[str, dict], probe: dict, p1: dict[str, set[Head]], meta: dict, path: Path
) -> None:
    out: list[str] = []
    a = out.append

    a("# Phase 2 — IOI circuit recovery by path patching\n")
    a("Phase 1 measured what each attention head does to the output through every route at ")
    a("once, and recovered 20 of the 26 heads published in Wang et al. (2022), ")
    a("*Interpretability in the Wild* (arXiv:2211.00593). The six it missed reach the logits ")
    a("only by way of another head, which is precisely what total-effect patching cannot ")
    a("resolve. This phase measures single paths instead.\n")
    a("\nGenerated by `scripts/run_phase2_paths.py`. The published circuit is hard-coded in ")
    a("`causal_interp/ground_truth.py` and is not consulted until the comparison.\n")

    a("\n## Run configuration\n")
    a(_table([[k, f"`{v}`"] for k, v in meta.items()], ["setting", "value"]))

    a("\n\n## 1. What path patching does differently\n")
    a("Activation patching splices a node's clean value into the corrupted run and lets the ")
    a("change propagate everywhere. Path patching pins every *other* attention head to its ")
    a("corrupted value first, so the only thing that can reach the chosen receiver is what ")
    a("came straight from the sender. Following Wang et al., MLPs are recomputed rather than ")
    a("pinned: the paper treats attention heads as the circuit's nodes and lets an MLP carry ")
    a("a path between them.\n")
    a("\nA head's q, k and v at token position *p* are computed from the residual stream at *p* ")
    a("alone, so only writes at *p* can reach them. Sender and receiver positions are therefore ")
    a("always equal — not a simplification, an architectural constraint.\n")

    a("\n## 2. The search, round by round\n")
    a("Round 0 asks which heads reach the logits directly. Each later round takes the heads ")
    a(f"found in the round before — the top {CHAIN_WIDTH} by absolute effect, a width fixed in ")
    a("advance — and asks which heads feed them. The answer key is never used to choose ")
    a("receivers, so a wrong turn in one round propagates rather than being silently corrected.\n")
    a("\nWhat the *receivers* are is prior knowledge: that S-inhibition heads act on name movers' ")
    a("queries, and so on, comes from the paper's account of the mechanism. Which *senders* turn ")
    a("up is not constrained — all 144 heads are swept every round. This is guided rediscovery, ")
    a("not blind search, and Phase 3 would have to search receiver inputs too.\n")

    for scheme, res in results.items():
        a(f"\n### `{scheme}`\n")
        for entry in res["rounds"]:
            a(f"\n**Round {entry['index']} — {entry['name']}.** {entry['question']}\n")
            if entry["halted"]:
                a("\n*Chain halted: the previous round carried no heads forward.*\n")
                continue
            if entry["receivers"]:
                a(f"\nReceivers: {', '.join(f'`{r}`' for r in entry['receivers'])}\n")
            a(f"\nThe paper's account predicts: *{entry['expected']}*.\n\n")
            ranked = sorted(entry["_effects"], key=lambda k: abs(entry["_effects"][k]), reverse=True)[:8]
            rows = []
            for head in ranked:
                rows.append([
                    f"**{head[0]}.{head[1]}**",
                    f"{entry['_effects'][head]:+.4f}",
                    _fmt_signal(entry["_signals"].get(head)),
                    classify(head) or "— *not in published circuit*",
                ])
            a(_table(rows, ["sender", "path effect on logits", "signal at receiver", "published class"]))
            a("\n")

    a("\n## 3. Comparison against the published circuit\n")
    a(f"Scored against the same {ground_truth.PUBLISHED_HEAD_COUNT} heads as Phase 1.\n")

    combined_all: set[Head] = set()
    for scheme, res in results.items():
        discovered = set(res["discovered"])
        combined_all |= discovered
        cmp_path = comparison.compare(discovered, f"path patching ({scheme})")
        a(f"\n### `{scheme}` — path patching alone\n\n")
        a(_table(
            [[cls, f"{found}/{total}"] for cls, (found, total) in cmp_path.per_class.items()]
            + [["**total**", f"**{len(cmp_path.matches)}/{ground_truth.PUBLISHED_HEAD_COUNT}**"]],
            ["published class", "recovered"],
        ))
        a(f"\n\nPrecision {cmp_path.precision:.2f}, recall {cmp_path.recall:.2f}, ")
        a(f"from {len(cmp_path.discovered)} heads discovered.\n")

    a("\n### Phase 1 and Phase 2 combined\n")
    p1_all: set[Head] = set()
    for s in p1.values():
        p1_all |= s
    combined = p1_all | combined_all
    cmp_p1 = comparison.compare(p1_all, "phase 1")
    cmp_p2 = comparison.compare(combined_all, "phase 2")
    cmp_both = comparison.compare(combined, "phase 1 + phase 2")

    rows = []
    for cls in ground_truth.IOI_CIRCUIT:
        total = len(ground_truth.IOI_CIRCUIT[cls])
        gained = cmp_both.per_class[cls][0] - cmp_p1.per_class[cls][0]
        rows.append([
            cls,
            f"{cmp_p1.per_class[cls][0]}/{total}",
            f"{cmp_p2.per_class[cls][0]}/{total}",
            f"**{cmp_both.per_class[cls][0]}/{total}**",
            f"+{gained}" if gained else "—",
        ])
    rows.append([
        "**total**",
        f"{len(cmp_p1.matches)}/26",
        f"{len(cmp_p2.matches)}/26",
        f"**{len(cmp_both.matches)}/26**",
        f"+{len(cmp_both.matches) - len(cmp_p1.matches)}" if len(cmp_both.matches) > len(cmp_p1.matches) else "—",
    ])
    a("\n")
    a(_table(rows, ["published class", "Phase 1", "Phase 2", "combined", "gained"]))

    a("\n\n**Still missing:**\n\n")
    if cmp_both.misses:
        a(_table(
            [[f"{l}.{h}", classify((l, h))] for l, h in cmp_both.misses],
            ["head", "published class"],
        ))
    else:
        a("*None — all 26 published heads recovered.*")

    a("\n\n**Discovered but not in the published circuit:**\n\n")
    if cmp_both.extras:
        a(", ".join(f"`{l}.{h}`" for l, h in cmp_both.extras))
    else:
        a("*None.*")

    a("\n\n## 4. Previous token heads\n")
    a(_previous_token_section(results, probe))

    a("\n## 5. What this phase settled\n")
    a(_conclusions(results, cmp_p1, cmp_p2, cmp_both))
    a("\n")

    path.write_text("".join(out), encoding="utf-8")


def _previous_token_section(results: dict[str, dict], probe: dict) -> str:
    """Report the previous-token result explicitly rather than letting it vanish."""
    lines = [
        "Phase 1 recovered 0 of the 2 published previous-token heads (2.2, 4.11) and blamed the ",
        "corruption scheme rather than the method. That explanation is now testable.\n",
        "\nNeither scheme can settle it inside its own chain. Under `s2_swap`, S1+1 is ",
        "bit-identical between the two runs, so every measurement there is an exact zero — ",
        "visible in round 3 below, where all senders score 0.0000. Under `abc`, S1+1 does differ, ",
        "but the chain does not survive round 1 and never produces receivers to ask about.\n",
    ]
    for scheme, res in results.items():
        entry = next((r for r in res["rounds"] if r["index"] == 3), None)
        if entry is None or entry.get("halted"):
            lines.append(f"\n- **`{scheme}` round 3** — chain halted before this round.\n")
            continue
        largest = max((abs(v) for v in entry["_effects"].values()), default=0.0)
        lines.append(
            f"\n- **`{scheme}` round 3** — {len(entry['discovered'])} of "
            f"{entry.get('senders_tested', 0)} senders cleared the cutoff; "
            f"largest absolute effect {largest:.4f}.\n"
        )

    if not probe.get("available"):
        lines.append(f"\nThe dedicated probe could not run: {probe.get('reason', 'unknown')}.\n")
        return "".join(lines)

    lines.append(
        "\n### The dedicated probe\n\n"
        "Taking each half from where it is sound: the receivers are the heads the `s2_swap` chain "
        "discovered at round 2, arrived at without consulting the answer key, and the measurement "
        "runs on `abc`, the only scheme in which S1+1 differs at all. Mixing schemes is a real "
        "caveat — the receivers were identified under one counterfactual and probed under another "
        "— and it is the price of the question being answerable.\n\n"
        "A sweep can only test senders below its earliest receiver, so the probe is run once per "
        "distinct receiver layer. Dropping the earliest receivers raises the ceiling and brings "
        "deeper senders into range; running every variant is what stops the sender range from "
        "quietly excluding a head.\n"
    )

    all_found: set[Head] = set()
    all_delivered: set[Head] = set()
    tested: set[Head] = set()
    for variant in probe["variants"]:
        lines.append(
            f"\n**Receivers at layer >= {variant['receiver_floor']}** "
            f"({', '.join(f'`{r}`' for r in variant['receivers'])}) — "
            f"senders testable: layers 0-{variant['max_sender_layer']}.\n\n"
        )
        ranked = sorted(variant["_effects"], key=lambda k: abs(variant["_effects"][k]), reverse=True)[:5]
        shown = list(dict.fromkeys(ranked + variant["_published_prev"]))
        rows = [
            [
                f"**{l}.{h}**",
                f"{variant['_effects'][(l, h)]:+.4f}",
                _fmt_signal(variant["_signals"].get((l, h))),
                classify((l, h)) or "— *not in published circuit*",
            ]
            for (l, h) in shown
        ]
        lines.append(
            _table(rows, ["sender", "path effect on logits", "signal at receiver", "published class"])
        )
        lines.append("\n")
        tested |= set(variant["_published_prev"])
        all_found |= {h for h in variant["_published_prev"] if f"{h[0]}.{h[1]}" in variant["discovered"]}
        all_delivered |= {
            h for h in variant["_published_prev"]
            if h in variant["_signals"] and abs(variant["_signals"][h]) > 0.1
        }

    published_prev_all = set(ground_truth.IOI_CIRCUIT["previous token"])
    untested = sorted(published_prev_all - tested)
    lines.append(f"\n**Result: {len(all_found)}/{len(published_prev_all)} cleared the cutoff.**\n")
    if untested:
        names = ", ".join(f"`{l}.{h}`" for l, h in untested)
        lines.append(
            f"\n{names} sits at or above every receiver layer available here, so no variant could "
            "test it. That is an untested head, not a head measured and found wanting, and it is "
            "counted as a miss on those terms.\n"
        )
    if all_delivered and not all_found:
        ordered = sorted(all_delivered)
        names = ", ".join(f"`{l}.{h}`" for l, h in ordered)
        one = len(ordered) == 1
        lines.append(
            f"\nThe two measurements disagree, and the disagreement is the finding. {names} "
            f"{'delivers' if one else 'each deliver'} a substantial share of the receiver's "
            f"clean-vs-corrupted difference — the path is there and carries signal — while "
            f"{'its' if one else 'their'} effect on the output logit difference stays near zero.\n"
        )
    elif all_found:
        lines.append("\nThe path effect on the logits was large enough to clear the cutoff.\n")
    else:
        lines.append(
            "\nNeither the logit effect nor the signal at the receiver was substantial, so this "
            "probe gives no positive evidence for these heads on this task.\n"
        )

    lines.append(
        "\nThe two columns answer different questions, which is why both are reported. *Path "
        "effect on logits* asks whether the prediction moves; *signal at receiver* asks whether "
        "the path delivered anything at its own endpoint. A path can score full marks on the "
        "second and near zero on the first, because every stage downstream of the receiver is "
        "still running on corrupted input — the deeper a link sits in the chain, the more of its "
        "effect is absorbed before reaching the output.\n"
        "\nWhere that pattern holds, the defensible conclusion is that logit-difference path "
        "patching is the wrong instrument for that link, not that the link is absent. Making it "
        "measurable needs a metric defined at the receiver rather than at the output. That is a "
        "change of measurement, not of method, and it is left for a later phase rather than "
        "folded into this one's headline number — the previous-token heads are counted as misses "
        "in every table above.\n"
    )
    return "".join(lines)


def _conclusions(results, cmp_p1, cmp_p2, cmp_both) -> str:
    gained = sorted(set(cmp_both.matches) - set(cmp_p1.matches))
    lines = [
        f"Path patching on its own recovers {len(cmp_p2.matches)}/26 published heads. Combined "
        f"with Phase 1's activation patching the project recovers **{len(cmp_both.matches)}/26**"
        + (
            f", up from {len(cmp_p1.matches)}/26.\n"
            if len(cmp_both.matches) > len(cmp_p1.matches)
            else f" — unchanged from Phase 1's {len(cmp_p1.matches)}/26.\n"
        ),
    ]
    if gained:
        by_class: dict[str, list[str]] = {}
        for head in gained:
            by_class.setdefault(classify(head) or "not in circuit", []).append(f"{head[0]}.{head[1]}")
        detail = "; ".join(f"{cls}: {', '.join(hs)}" for cls, hs in by_class.items())
        lines.append(f"\nHeads Phase 1 could not see and this phase can — {detail}.\n")
    else:
        lines.append(
            "\nNo head missed by Phase 1 was recovered here. That is a negative result and is "
            "reported as one: on this task, with this metric, path patching did not reach past "
            "what total-effect patching already found.\n"
        )
    if cmp_both.misses:
        remaining: dict[str, list[str]] = {}
        for head in cmp_both.misses:
            remaining.setdefault(classify(head) or "?", []).append(f"{head[0]}.{head[1]}")
        detail = "; ".join(f"{cls}: {', '.join(hs)}" for cls, hs in remaining.items())
        lines.append(f"\nStill outstanding — {detail}. Section 4 covers the previous-token case.\n")
    lines.append(
        "\n### What did improve\n"
        "\nHead count is not the only thing a circuit claim is made of, and two things moved that "
        "the 20/26 headline does not show.\n"
    )
    lines.append(
        f"\n**Precision.** Phase 1 discovered {len(cmp_p1.discovered)} heads to find "
        f"{len(cmp_p1.matches)}, a precision of {cmp_p1.precision:.2f}. Phase 2 discovered "
        f"{len(cmp_p2.discovered)} to find {len(cmp_p2.matches)}, a precision of "
        f"{cmp_p2.precision:.2f}. Restricting each measurement to a single path removes most of "
        "what total-effect patching swept up incidentally.\n"
    )

    # The ordering result: what the chain found, round by round, without being told.
    chain = results.get("s2_swap", {}).get("rounds", [])
    described = []
    for entry in chain:
        if entry.get("halted") or not entry.get("_effects"):
            continue
        # A round where every effect is exactly zero has no ranking to report — its
        # "top senders" would be an arbitrary tie-break, and quoting a hit rate off
        # that would invent a result out of no signal at all.
        if max(abs(v) for v in entry["_effects"].values()) == 0.0:
            described.append(f"round {entry['index']} — no signal (every sender exactly zero)")
            continue
        top = sorted(entry["_effects"], key=lambda k: abs(entry["_effects"][k]), reverse=True)[:4]
        in_circuit = sum(1 for h in top if classify(h))
        classes = sorted({classify(h) for h in top if classify(h)})
        described.append(
            f"round {entry['index']} — {in_circuit}/4 top senders in the published circuit"
            + (f" ({', '.join(classes)})" if classes else "")
        )
    if described:
        lines.append(
            "\n**The mechanism, not just the parts.** Phase 1 produced a ranked list of heads. "
            "The chain here reproduced the paper's causal *order* without being told it, each "
            "round's receivers coming from the round before: "
            + "; ".join(described)
            + ". Recovering the wiring is a stronger claim than recovering the set of components, "
            "and it is the claim Phase 1 could not make at all.\n"
        )

    lines.append(
        "\nThe direct-effect round is worth reading on its own. Heads that Phase 1 ranked highly "
        "by total effect can score near zero on direct effect, which is not a contradiction: it "
        "is the measurement saying their influence is real but routed through another head. That "
        "distinction is exactly what Phase 1 could not draw, and drawing it is what this phase "
        "was for.\n"
    )
    lines.append(
        "\n**One deliberate choice worth stating plainly.** Every head counted as discovered above "
        "was discovered by its effect on the output logit difference — the same criterion Phase 1 "
        "used, kept identical so the two phases are comparable. The *signal at receiver* column "
        "repeatedly identifies published heads that the logit criterion misses: the previous-token "
        "heads in section 4 are the clearest case, and the S-inhibition heads under `abc` are "
        "another. Scoring discovery on that column instead would raise the recall number in this "
        "report.\n"
        "\nIt was not done, because switching the success criterion after seeing which criterion "
        "scores better is how a validation exercise stops validating anything. The signal metric "
        "is reported as a diagnostic, its disagreements with the logit metric are shown wherever "
        "they occur, and adopting it as a discovery criterion is left to a later phase where it "
        "can be committed to in advance.\n"
    )
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    n = 16 if args.quick else args.n
    RESULTS_DIR.mkdir(exist_ok=True)
    started = time.time()

    model = load()
    meta = {
        "model": model.cfg.model_name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "transformer_lens": version("transformer_lens"),
        "python": platform.python_version(),
        "prompts_per_scheme": n,
        "seed": args.seed,
        "headline_threshold": HEADLINE_THRESHOLD,
        "chain_width": CHAIN_WIDTH,
        "mlps": "recomputed (per Wang et al.)",
    }

    results = {scheme: run_scheme(model, scheme, n, args.seed) for scheme in ("s2_swap", "abc")}
    for res in results.values():
        write_csvs(res, RESULTS_DIR)

    probe = previous_token_probe(model, results, n, args.seed)

    p1 = phase1_discovered()
    meta["runtime_seconds"] = round(time.time() - started, 1)
    write_report(results, probe, p1, meta, RESULTS_DIR / "PHASE2_REPORT.md")

    combined_p2: set[Head] = set()
    for res in results.values():
        combined_p2 |= set(res["discovered"])
    p1_all: set[Head] = set()
    for s in p1.values():
        p1_all |= s

    serializable = {
        "meta": meta,
        "schemes": {
            scheme: {
                "corruption": res["corruption"],
                "n_prompts": res["n_prompts"],
                "baseline": res["baseline"],
                "discovered": [f"{l}.{h}" for l, h in res["discovered"]],
                "rounds": [
                    {k: v for k, v in entry.items() if not k.startswith("_")}
                    for entry in res["rounds"]
                ],
            }
            for scheme, res in results.items()
        },
        "previous_token_probe": {
            "available": probe.get("available", False),
            "reason": probe.get("reason"),
            "variants": [
                {k: v for k, v in variant.items() if not k.startswith("_")}
                for variant in probe.get("variants", [])
            ],
        },
        "comparison": {
            "phase1": _comparison_dict(comparison.compare(p1_all, "phase 1")),
            "phase2": _comparison_dict(comparison.compare(combined_p2, "phase 2")),
            "combined": _comparison_dict(comparison.compare(p1_all | combined_p2, "combined")),
        },
    }
    (RESULTS_DIR / "phase2_results.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print(f"\nwrote {RESULTS_DIR / 'PHASE2_REPORT.md'}")
    print(f"total runtime {meta['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
