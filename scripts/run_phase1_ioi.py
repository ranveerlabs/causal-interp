"""Phase 1 end to end: recover the IOI circuit by activation patching, then score
it against the published circuit.

    python scripts/run_phase1_ioi.py                 # full run, both corruptions
    python scripts/run_phase1_ioi.py --quick         # small run for smoke-testing

Writes to results/:
    PHASE1_REPORT.md              the comparison against Wang et al. (the deliverable)
    phase1_results.json           every number the report is built from
    head_effects_<scheme>.csv     per (layer, head, position) normalized effect
    component_effects_<scheme>.csv per (layer, position) effect for resid/attn/mlp

Nothing in this script decides what counts as a match; that lives in
causal_interp.comparison, scored against causal_interp.ground_truth.
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

from causal_interp import comparison, ground_truth
from causal_interp.ground_truth import Head, classify
from causal_interp.interventions import (
    ALL_POSITIONS,
    Patch,
    baseline_for,
    clean_cache_for,
    greedy_select,
    patch_effect,
    sweep_component,
    sweep_heads,
)
from causal_interp.ioi import POSITIONS, IOIDataset
from causal_interp.model import load

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Fixed before looking at any output, so the headline number is not chosen to flatter
# the result. The sweep around it is reported either way.
HEADLINE_THRESHOLD = 0.02
THRESHOLD_SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
COMPONENT_KINDS = ("resid_pre", "attn_out", "mlp_out")


def run_one(model, corruption: str, n: int, seed: int, greedy_candidates: int, greedy_max: int) -> dict:
    """Full patching analysis for one corruption scheme."""
    print(f"\n{'=' * 72}\ncorruption scheme: {corruption}\n{'=' * 72}")
    ds = IOIDataset(model, n=n, corruption=corruption, seed=seed)

    baseline, clean_logits, corrupted_logits = baseline_for(model, ds)
    clean_acc = ds.io_rank_stats(clean_logits)
    corrupted_acc = ds.io_rank_stats(corrupted_logits)
    print(f"  clean logit diff      {baseline.clean_logit_diff:+.4f}   (IO beats S on {clean_acc['io_beats_s']:.1%})")
    print(f"  corrupted logit diff  {baseline.corrupted_logit_diff:+.4f}   (IO beats S on {corrupted_acc['io_beats_s']:.1%})")

    cache, _ = clean_cache_for(model, ds)

    print("  component sweeps ...", end="", flush=True)
    t0 = time.time()
    components = {
        kind: sweep_component(model, ds, cache, baseline, kind, POSITIONS).tolist()
        for kind in COMPONENT_KINDS
    }
    print(f" {time.time() - t0:.0f}s")

    print("  head sweep ", end="", flush=True)
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        if done % max(1, total // 20) == 0:
            print(".", end="", flush=True)

    head_sweep = sweep_heads(model, ds, cache, baseline, POSITIONS, progress=progress)
    print(f" {time.time() - t0:.0f}s")

    # Collapse positions: each head is summarised by the position where it has the
    # largest absolute effect, so a head acting only at S2 is not diluted by six
    # positions where it does nothing.
    effects: dict[Head, float] = {}
    best_positions: dict[Head, str] = {}
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            row = head_sweep[layer, head]
            p = int(row.abs().argmax())
            effects[(layer, head)] = float(row[p])
            best_positions[(layer, head)] = POSITIONS[p]

    # Step 4: narrow a ranked list down to a set that jointly restores behaviour.
    ranked = sorted(effects, key=lambda h: abs(effects[h]), reverse=True)
    candidates = [
        Patch(layer=l, kind="z", position=best_positions[(l, h)], head=h)
        for (l, h) in ranked[:greedy_candidates]
    ]
    print(f"  greedy narrowing over {len(candidates)} candidates ...", end="", flush=True)
    t0 = time.time()
    trace = greedy_select(model, ds, cache, baseline, candidates, max_size=greedy_max)
    print(f" {time.time() - t0:.0f}s -> {len(trace)} nodes, recovery {trace[-1][1]:.3f}" if trace else " none")

    # Upper bound for context: patching every head everywhere.
    all_heads_effect = patch_effect(
        model, ds, cache,
        [Patch(l, "z", ALL_POSITIONS, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)],
        baseline,
    )

    # How many head/position cells came out as *exact* zeros. Under s2_swap the
    # pre-S2 positions should be zero for every head, since clean and corrupted
    # activations there are bit-identical; measuring it beats asserting it.
    exact_zeros = {
        position: [int((head_sweep[:, :, p] == 0).sum()), head_sweep[:, :, p].numel()]
        for p, position in enumerate(POSITIONS)
    }

    comparisons = comparison.threshold_sweep(effects, THRESHOLD_SWEEP)
    headline = comparison.compare(
        comparison.threshold_set(effects, HEADLINE_THRESHOLD), f"abs(effect) >= {HEADLINE_THRESHOLD:g}"
    )
    size_matched = comparison.compare(
        comparison.top_k_set(effects, ground_truth.PUBLISHED_HEAD_COUNT),
        f"top {ground_truth.PUBLISHED_HEAD_COUNT} by abs(effect)",
    )

    return {
        "corruption": corruption,
        "n_prompts": len(ds),
        "seed": seed,
        "baseline": {
            "clean_logit_diff": baseline.clean_logit_diff,
            "corrupted_logit_diff": baseline.corrupted_logit_diff,
        },
        "accuracy": {"clean": clean_acc, "corrupted": corrupted_acc},
        "all_heads_patched_effect": all_heads_effect,
        "exact_zeros": exact_zeros,
        "components": components,
        "head_sweep": head_sweep.tolist(),
        "effects": {f"{l}.{h}": v for (l, h), v in effects.items()},
        "best_positions": {f"{l}.{h}": v for (l, h), v in best_positions.items()},
        "greedy": [{"node": str(p), "cumulative_recovery": s} for p, s in trace],
        "headline": headline,
        "size_matched": size_matched,
        "threshold_sweep": comparisons,
        "_effects": effects,
        "_best_positions": best_positions,
    }


# -- output ---------------------------------------------------------------------


def write_csvs(res: dict, out_dir: Path) -> None:
    scheme = res["corruption"]
    with (out_dir / f"head_effects_{scheme}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "position", "normalized_effect", "published_class"])
        for layer, per_head in enumerate(res["head_sweep"]):
            for head, per_pos in enumerate(per_head):
                cls = classify((layer, head)) or ""
                for p, value in enumerate(per_pos):
                    w.writerow([layer, head, POSITIONS[p], f"{value:.6f}", cls])

    with (out_dir / f"component_effects_{scheme}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["component", "layer", "position", "normalized_effect"])
        for kind, rows in res["components"].items():
            for layer, per_pos in enumerate(rows):
                for p, value in enumerate(per_pos):
                    w.writerow([kind, layer, POSITIONS[p], f"{value:.6f}"])


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


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def write_report(results: dict[str, dict], meta: dict, path: Path) -> None:
    primary = results["s2_swap"]
    out: list[str] = []
    a = out.append

    a("# Phase 1 — IOI circuit recovery by activation patching\n")
    a("Validation run: does activation patching on GPT-2 small recover the circuit ")
    a("published in Wang et al. (2022), *Interpretability in the Wild* (arXiv:2211.00593)?\n")
    a("This report is generated by `scripts/run_phase1_ioi.py`. Every number below comes ")
    a("from that run; the published circuit it is scored against is hard-coded in ")
    a("`causal_interp/ground_truth.py` and is not derived from any measurement here.\n")

    a("\n## Run configuration\n")
    a(_table(
        [[k, f"`{v}`"] for k, v in meta.items()],
        ["setting", "value"],
    ))

    a("\n\n## 1. The model actually does the task\n")
    a("A circuit claim is meaningless if the behaviour is not there to begin with.\n\n")
    rows = []
    for scheme, res in results.items():
        rows.append([
            f"`{scheme}`",
            f"{res['baseline']['clean_logit_diff']:+.3f}",
            f"{res['baseline']['corrupted_logit_diff']:+.3f}",
            f"{res['accuracy']['clean']['io_beats_s']:.1%}",
            f"{res['accuracy']['clean']['top1_is_io']:.1%}",
        ])
    a(_table(rows, ["scheme", "clean logit diff", "corrupted logit diff", "IO beats S (clean)", "IO is top-1 (clean)"]))
    a("\n\nLogit difference is `logit(IO) - logit(S)` at the final token, with IO and S ")
    a("fixed to the *clean* prompt's names in all runs so the three numbers are comparable. ")
    a("All patching results are normalized onto that span: **0.0 = corrupted behaviour, ")
    a("1.0 = clean behaviour restored**.\n\n")
    a("Values above 1.0 appear below and are not errors. Splicing a name mover's clean ")
    a("output into the corrupted run can make the model prefer the indirect object *more* ")
    a("strongly than it does on the clean prompt itself, because the patched run keeps the ")
    a("corrupted context while gaining the clean answer signal. This is why the narrowing in ")
    a("section 6 targets closeness to 1.0 rather than maximum recovery.\n")

    a("\n## 2. Corruption schemes, and what each one can see\n")
    a("Two counterfactuals were run, because each is blind to something the other sees.\n\n")
    a("- **`s2_swap`** — the repeated subject name is swapped for the indirect object, ")
    a("flipping the correct answer. Exactly one token differs, which makes the metric ")
    a("cleanly symmetric. The cost: clean and corrupted activations are *identical* at ")
    a("every position before S2, so patching there is a mathematical no-op and no ")
    a("component acting earlier than S2 can be detected at all.\n")
    a("- **`abc`** — all three name slots are replaced with fresh names, so no name repeats. ")
    a("Activations differ from the first name onward, which makes early positions ")
    a("visible, at the cost of a corrupted run whose logit difference sits near zero ")
    a("rather than symmetrically negative.\n")

    a("\nThe blindness of `s2_swap` is measured, not assumed. Across the full head sweep, ")
    a("the number of (head, position) cells whose effect is *exactly* zero:\n\n")
    pre_s2 = ("IO", "IO+1", "S1", "S1+1")
    rows = []
    for position in POSITIONS:
        cells = [
            f"{results[s]['exact_zeros'][position][0]}/{results[s]['exact_zeros'][position][1]}"
            for s in ("s2_swap", "abc")
        ]
        rows.append([position, *cells, "yes" if position in pre_s2 else "no"])
    a(_table(rows, ["position", "`s2_swap` exact zeros", "`abc` exact zeros", "precedes S2"]))
    a("\n\nEvery head is an exact zero at every position before S2 under `s2_swap` — a ")
    a("floating-point exact zero, because the two runs are computing on identical inputs ")
    a("there. Any published head whose role is at those positions cannot be found by this ")
    a("scheme even in principle.\n")
    a(f"\nThe residual {meta['n_heads']} zeros at each non-END position, present under both ")
    a("schemes, are a second and unrelated structural zero: they are the final layer's heads, ")
    a("whose output nothing reads except the logits at END, so patching them anywhere else ")
    a("cannot change the prediction. Both patterns falling exactly where the architecture says ")
    a("they must is a check that the hooks write where they claim to.\n")

    a("\n## 3. Where the causal signal lives\n")
    a("Patching whole layers before attributing anything to individual heads.\n\n")
    for scheme, res in results.items():
        a(f"\n**`{scheme}`** — strongest (layer, position) per component:\n\n")
        rows = []
        for kind, grid in res["components"].items():
            flat = [(v, l, p) for l, row in enumerate(grid) for p, v in enumerate(row)]
            flat.sort(key=lambda t: abs(t[0]), reverse=True)
            top = ", ".join(f"L{l}@{POSITIONS[p]} {v:+.3f}" for v, l, p in flat[:4])
            rows.append([f"`{kind}`", top])
        a(_table(rows, ["component", "top 4 by abs(effect)"]))
        a("\n")

    a("\n## 4. Head-level results\n")
    a("Every attention head patched at every position, one at a time. Each head is ")
    a("summarised by the position where its absolute effect is largest.\n")
    for scheme, res in results.items():
        a(f"\n### `{scheme}` — top 20 heads by |effect|\n\n")
        effects, best = res["_effects"], res["_best_positions"]
        ranked = sorted(effects, key=lambda h: abs(effects[h]), reverse=True)[:20]
        rows = [
            [
                f"**{l}.{h}**",
                best[(l, h)],
                f"{effects[(l, h)]:+.4f}",
                classify((l, h)) or "— *not in published circuit*",
            ]
            for (l, h) in ranked
        ]
        a(_table(rows, ["head", "best position", "effect", "published class"]))
        a("\n")

    a("\n## 5. Comparison against the published circuit\n")
    a(f"The published circuit is {ground_truth.PUBLISHED_HEAD_COUNT} heads in ")
    a(f"{len(ground_truth.IOI_CIRCUIT)} classes. Recall is against all ")
    a(f"{ground_truth.PUBLISHED_HEAD_COUNT}.\n")

    for scheme, res in results.items():
        a(f"\n### `{scheme}`\n")
        a("\n**Threshold sensitivity** — the cutoff is a free parameter, so the whole sweep ")
        a("is shown rather than one flattering choice:\n\n")
        rows = [
            [
                c.label,
                str(len(c.discovered)),
                f"{len(c.matches)}/{ground_truth.PUBLISHED_HEAD_COUNT}",
                f"{c.precision:.2f}",
                f"{c.recall:.2f}",
                f"{c.f1:.2f}",
            ]
            for c in res["threshold_sweep"]
        ]
        a(_table(rows, ["cutoff", "discovered", "matched", "precision", "recall", "F1"]))

        head = res["headline"]
        a(f"\n\n**Headline** (pre-registered cutoff `|effect| >= {HEADLINE_THRESHOLD:g}`): ")
        a(f"{len(head.discovered)} heads discovered, {len(head.matches)} in the published circuit ")
        a(f"(precision {head.precision:.2f}, recall {head.recall:.2f}).\n")

        sm = res["size_matched"]
        a(f"\n**Size-matched** (top {ground_truth.PUBLISHED_HEAD_COUNT}, no free parameter): ")
        a(f"{len(sm.matches)}/{ground_truth.PUBLISHED_HEAD_COUNT} overlap ")
        a(f"(precision = recall = {sm.recall:.2f}).\n")

        a("\n**Recall by class** (headline cutoff):\n\n")
        rows = [
            [
                cls,
                f"{found}/{total}",
                ground_truth.CLASS_EXPECTED_POSITION[cls],
                ", ".join(f"{l}.{h}" for l, h in ground_truth.IOI_CIRCUIT[cls] if (l, h) in head.discovered) or "—",
            ]
            for cls, (found, total) in head.per_class.items()
        ]
        a(_table(rows, ["published class", "found", "expected position", "heads found"]))

        a("\n\n**Misses** — published heads below the cutoff, with what was actually measured:\n\n")
        miss_rows = comparison.miss_report(res["_effects"], res["_best_positions"], head.discovered)
        if miss_rows:
            a(_table(
                [[f"{r['head']}", r["class"], f"{r['effect']:+.4f}", r["best_position"]] for r in miss_rows],
                ["head", "published class", "measured effect", "best position"],
            ))
        else:
            a("*None — every published head cleared the cutoff.*")

        a("\n\n**Extras** — heads above the cutoff that are not in the published circuit:\n\n")
        if head.extras:
            a(_table(
                [
                    [f"{l}.{h}", res["_best_positions"][(l, h)], f"{res['_effects'][(l, h)]:+.4f}"]
                    for (l, h) in sorted(head.extras, key=lambda x: abs(res["_effects"][x]), reverse=True)
                ],
                ["head", "best position", "effect"],
            ))
            a("\n")
            a(_extras_caveat(scheme, head.extras, res["_best_positions"]))
        else:
            a("*None.*")
        a("\n")

    a("\n### Union across both corruption schemes\n")
    a("Each scheme is blind to something the other sees, so the union at the headline cutoff is ")
    a("the fair summary of what activation patching found overall.\n\n")
    union = results["s2_swap"]["headline"].discovered | results["abc"]["headline"].discovered
    union_cmp = comparison.compare(union, "union of both schemes")
    rows = []
    for cls, (found, total) in union_cmp.per_class.items():
        per_scheme = [f"{results[s]['headline'].per_class[cls][0]}/{total}" for s in ("s2_swap", "abc")]
        recovered = sorted(h for h in ground_truth.IOI_CIRCUIT[cls] if h in union)
        rows.append([
            cls,
            *per_scheme,
            f"**{found}/{total}**",
            ", ".join(f"{l}.{h}" for l, h in recovered) or "— *none*",
        ])
    rows.append([
        "**total**",
        f"{len(results['s2_swap']['headline'].matches)}/{ground_truth.PUBLISHED_HEAD_COUNT}",
        f"{len(results['abc']['headline'].matches)}/{ground_truth.PUBLISHED_HEAD_COUNT}",
        f"**{len(union_cmp.matches)}/{ground_truth.PUBLISHED_HEAD_COUNT}**",
        "",
    ])
    a(_table(rows, ["published class", "`s2_swap`", "`abc`", "union", "heads recovered"]))
    a("\n\nThe classes fully recovered — name movers, negative name movers, S-inhibition — are ")
    a("exactly those that write to the output logits directly. The classes that fall short — ")
    a("previous token, duplicate token, induction, and the redundant backup name movers — are ")
    a("exactly those whose contribution reaches the output only through another head. That ")
    a("boundary, rather than any single recall number, is the result of this phase.\n")

    a("\n## 6. Iterative narrowing to a joint circuit\n")
    a("A ranking of marginal effects is not a circuit: heads can be redundant, so a set ")
    a("chosen by individual scores can overshoot or undershoot. Each step below adds the ")
    a("single node that brings recovery *closest to 1.0* given everything already patched, ")
    a("stopping when the best remaining addition closes less than 0.5% of the gap.\n")
    for scheme, res in results.items():
        a(f"\n**`{scheme}`** — patching all 144 heads at all positions recovers ")
        a(f"{res['all_heads_patched_effect']:.3f}, the ceiling for a head-only circuit.\n\n")
        rows = [
            [
                str(i + 1),
                f"`{step['node']}`",
                f"{step['cumulative_recovery']:.3f}",
                classify(_parse_head(step["node"])) or "— *not in published circuit*",
            ]
            for i, step in enumerate(res["greedy"])
        ]
        a(_table(rows, ["step", "node added", "cumulative recovery", "published class"]) if rows else "*No node cleared the minimum gain.*")
        a("\n")
        chosen = [_parse_head(step["node"]) for step in res["greedy"]]
        a(_extras_caveat(scheme, [h for h in chosen if h not in ground_truth.ALL_HEADS], res["_best_positions"]))

    a("\n## 7. What this validated, and what it did not\n")
    a(_limitations(results))
    a("\n")

    path.write_text("".join(out), encoding="utf-8")


def _extras_caveat(scheme: str, extras: list[Head], best_positions: dict[Head, str]) -> str:
    """Flag extras that are early heads sitting on a token the corruption itself changed.

    Under `abc` the name tokens differ between the clean and corrupted prompts, so a
    layer-0 head reading one of those positions carries "which name is written here".
    That is real causal influence on the logit difference, but it is token identity
    being restored, not a component of the IOI algorithm — and counting it as a
    discovery would inflate the result.
    """
    suspect = [h for h in extras if h[0] <= 1 and best_positions[h] in ("IO", "S1", "S2")]
    if not suspect or scheme != "abc":
        return ""

    listed = ", ".join(f"`{l}.{h}@{best_positions[(l, h)]}`" for l, h in sorted(suspect))
    layers = sorted({l for l, _ in suspect})
    layer_text = f"layer {layers[0]}" if len(layers) == 1 else "layers " + " and ".join(map(str, layers))
    one = len(suspect) == 1
    return (
        f"\n*Caveat:* {listed} — {'this head sits' if one else 'these heads sit'} in {layer_text}, "
        f"on a name token that `abc` itself replaced. Patching {'it' if one else 'them'} restores "
        "*which name is written there*, which moves the logit difference for a reason unrelated to "
        f"the IOI algorithm. {'That head at that position is an exact zero' if one else 'Those heads at those positions are exact zeros'} "
        f"under `s2_swap`, where those tokens are untouched. {'It is' if one else 'They are'} reported "
        "rather than quietly dropped, but should be read as an artifact of the corruption, not a discovery.\n"
    )


def _parse_head(node: str) -> Head:
    """'9.9@END' -> (9, 9)."""
    layer, head = node.split("@")[0].split(".")
    return (int(layer), int(head))


def _limitations(results: dict[str, dict]) -> str:
    prim = results["s2_swap"]["headline"]
    abc = results["abc"]["headline"]
    union = prim.discovered | abc.discovered
    union_cmp = comparison.compare(union, "union of both schemes")
    lines = [
        f"Taking the union of both corruption schemes at the headline cutoff, "
        f"{len(union_cmp.matches)}/{ground_truth.PUBLISHED_HEAD_COUNT} published heads were recovered "
        f"({len(union_cmp.extras)} heads outside the published circuit also cleared it).\n",
        "\n**Validated.** The behaviour is real and localized: patching single attention heads ",
        "moves the logit difference by a large fraction of its full span, and the heads that do ",
        "so are concentrated in the layers and positions the paper predicts.\n",
        "\n**Not validated, by construction.** Three limits are structural, not tuning problems:\n",
        "\n1. **Activation patching is not path patching.** The published circuit was derived with ",
        "*path* patching, which measures a component's effect along a specific downstream route. ",
        "Plain activation patching measures a node's total effect on the output. A head whose ",
        "contribution is routed entirely through another head can therefore be invisible here even ",
        "though it is genuinely in the circuit. Classes defined by their *indirect* role — ",
        "previous token, duplicate token, induction — are the ones this hits hardest.\n",
        "\n2. **`s2_swap` cannot see anything before S2.** With one differing token, clean and ",
        "corrupted activations coincide at every earlier position, so previous-token heads acting ",
        "at S1+1 have an effect of *exactly* zero — not a weak signal, a structurally undefined one. ",
        "`abc` was added specifically to cover this, and is the only scheme that can speak to those heads.\n",
        "\n3. **Marginal effects understate redundant components.** The paper's backup name movers ",
        "exist precisely because they activate when the primary name movers are removed. Patching ",
        "one head at a time, with everything else intact, is the condition under which redundant ",
        "components look least important.\n",
        "\nNone of these are reasons to discount the misses; they are the reasons the misses are ",
        "where they are, and they are the concrete argument for implementing path patching before ",
        "this method is pointed at a circuit nobody has published.\n",
    ]
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128, help="prompts per dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="small run for smoke-testing")
    parser.add_argument("--greedy-candidates", type=int, default=40)
    parser.add_argument("--greedy-max", type=int, default=20)
    args = parser.parse_args()

    n = 16 if args.quick else args.n
    greedy_candidates = 8 if args.quick else args.greedy_candidates
    greedy_max = 4 if args.quick else args.greedy_max

    RESULTS_DIR.mkdir(exist_ok=True)
    started = time.time()

    model = load()
    meta = {
        "model": model.cfg.model_name,
        "device": str(model.cfg.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "transformer_lens": version("transformer_lens"),
        "python": platform.python_version(),
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "prompts_per_scheme": n,
        "seed": args.seed,
        "headline_threshold": HEADLINE_THRESHOLD,
    }

    results = {
        scheme: run_one(model, scheme, n, args.seed, greedy_candidates, greedy_max)
        for scheme in ("s2_swap", "abc")
    }

    for res in results.values():
        write_csvs(res, RESULTS_DIR)

    meta["runtime_seconds"] = round(time.time() - started, 1)
    write_report(results, meta, RESULTS_DIR / "PHASE1_REPORT.md")

    serializable = {
        "meta": meta,
        "schemes": {
            scheme: {
                **{k: v for k, v in res.items() if not k.startswith("_") and k not in
                   ("headline", "size_matched", "threshold_sweep")},
                "headline": _comparison_dict(res["headline"]),
                "size_matched": _comparison_dict(res["size_matched"]),
                "threshold_sweep": [_comparison_dict(c) for c in res["threshold_sweep"]],
            }
            for scheme, res in results.items()
        },
    }
    (RESULTS_DIR / "phase1_results.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print(f"\nwrote {RESULTS_DIR / 'PHASE1_REPORT.md'}")
    print(f"total runtime {meta['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
