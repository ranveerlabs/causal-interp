"""Phase 3 analysis: apply the pre-registered receiver-side threshold and report.

Imported only by the main run of `run_phase3_receiver.py`, never by its
`--preregister` path, so the step that fixes the threshold cannot reach any of
this code.

The output deliberately does not merge the two criteria into one score. A head
found by its effect on the output and a head found by what it delivers to its
receiver are answering different questions about what "part of the circuit"
means, and collapsing them would hide the disagreement that is the result.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import torch

from causal_interp import comparison, ground_truth
from causal_interp.ground_truth import Head, classify
from causal_interp.interventions import derangement

import run_phase3_receiver as phase3


def _load_phase2(results_dir: Path) -> dict:
    return json.loads((results_dir / "phase2_results.json").read_text(encoding="utf-8"))


def _heads(names) -> set[Head]:
    return {tuple(map(int, n.split("."))) for n in names}


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def run_analysis(model, prereg: dict, n: int, seed: int, results_dir: Path) -> int:
    threshold = prereg["threshold"]
    groups = phase3.receiver_sets()
    contexts = phase3._contexts(model, n, seed)
    phase2 = _load_phase2(results_dir)
    started = time.time()

    # -- measure ------------------------------------------------------------
    per_group: list[dict] = []
    signal_by_head: dict[Head, float] = {}
    for group in groups:
        print(f"  signal sweep: {group.label} ", end="", flush=True)
        t0 = time.time()
        signals = phase3._sweep(model, contexts, group, permutation=None)
        print(f" {time.time() - t0:.0f}s  n={len(signals)}")
        for head, value in signals.items():
            if head not in signal_by_head or abs(value) > abs(signal_by_head[head]):
                signal_by_head[head] = value
        per_group.append({
            "label": group.label,
            "scheme": group.scheme,
            "position": group.position,
            "receivers": [str(r) for r in group.receivers],
            "signals": {f"{l}.{h}": v for (l, h), v in signals.items()},
            "discovered": [
                f"{l}.{h}" for l, h in sorted(comparison.threshold_set(signals, threshold))
            ],
            "_signals": signals,
        })

    discovered_signal = set(comparison.threshold_set(signal_by_head, threshold))

    # -- the two criteria ---------------------------------------------------
    logit_all: set[Head] = set()
    logit_rounds1plus: set[Head] = set()
    for res in phase2["schemes"].values():
        for entry in res["rounds"]:
            if entry.get("halted"):
                continue
            found = _heads(entry.get("discovered", []))
            logit_all |= found
            if entry["index"] > 0:
                logit_rounds1plus |= found

    cmp_logit = comparison.compare(logit_all, "logit criterion (all rounds)")
    cmp_logit_r1 = comparison.compare(logit_rounds1plus, "logit criterion (rounds 1+)")
    cmp_signal = comparison.compare(discovered_signal, "receiver-side criterion")

    phase1 = json.loads((results_dir / "phase1_results.json").read_text(encoding="utf-8"))
    p1: set[Head] = set()
    for res in phase1["schemes"].values():
        p1 |= _heads(res["headline"]["matches"] + res["headline"]["extras"])
    previously_missing = sorted(ground_truth.ALL_HEADS - (p1 | logit_all))

    meta = {
        "model": model.cfg.model_name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "transformer_lens": version("transformer_lens"),
        "python": platform.python_version(),
        "prompts_per_scheme": n,
        "seed": seed,
        "pre_registered_threshold": threshold,
        "threshold_rule": prereg["rule"],
        "runtime_seconds": None,
    }

    _write_csv(per_group, results_dir)
    meta["runtime_seconds"] = round(time.time() - started, 1)
    _write_report(
        results_dir / "PHASE3_REPORT.md", meta, prereg, per_group, phase2,
        cmp_logit, cmp_logit_r1, cmp_signal, previously_missing, signal_by_head,
    )

    payload = {
        "meta": meta,
        "preregistration": prereg,
        "groups": [{k: v for k, v in g.items() if not k.startswith("_")} for g in per_group],
        "best_signal_per_head": {f"{l}.{h}": v for (l, h), v in signal_by_head.items()},
        "criteria": {
            "logit_all_rounds": _cmp_dict(cmp_logit),
            "logit_rounds_1plus": _cmp_dict(cmp_logit_r1),
            "receiver_side": _cmp_dict(cmp_signal),
        },
        "previously_missing": [f"{l}.{h}" for l, h in previously_missing],
    }
    (results_dir / "phase3_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {results_dir / 'PHASE3_REPORT.md'}")
    return 0


def _cmp_dict(c: comparison.Comparison) -> dict:
    return {
        "label": c.label,
        "n_discovered": len(c.discovered),
        "matches": [f"{l}.{h}" for l, h in c.matches],
        "misses": [f"{l}.{h}" for l, h in c.misses],
        "extras": [f"{l}.{h}" for l, h in c.extras],
        "per_class": {k: list(v) for k, v in c.per_class.items()},
        "precision": c.precision,
        "recall": c.recall,
    }


def _write_csv(per_group: list[dict], results_dir: Path) -> None:
    path = results_dir / "receiver_signals.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "scheme", "position", "layer", "head", "path_signal", "published_class"])
        for group in per_group:
            for key, value in group["signals"].items():
                layer, head = map(int, key.split("."))
                w.writerow([
                    group["label"], group["scheme"], group["position"],
                    layer, head, f"{value:.6f}", classify((layer, head)) or "",
                ])


def _write_report(
    path, meta, prereg, per_group, phase2,
    cmp_logit, cmp_logit_r1, cmp_signal, previously_missing, signal_by_head,
) -> None:
    out: list[str] = []
    a = out.append
    threshold = prereg["threshold"]

    a("# Phase 3 — a pre-registered receiver-side criterion\n")
    a("Phases 1 and 2 both scored a head as *found* by what it does to the output logit ")
    a("difference, and both arrived at 20/26. Phase 2 also computed `path_signal` — how much ")
    a("of a receiver's clean-vs-corrupted difference a path actually delivers — and noticed that ")
    a("this diagnostic scored several of the missing heads well.\n")
    a("\nThat observation is exactly what makes it dangerous to adopt. This phase fixes the ")
    a("threshold first, by a rule, in a step that computes no real measurement, and then applies ")
    a("it without adjustment.\n")

    a("\n## 1. The pre-registration\n")
    a(f"\n> {prereg['rule']}\n")
    a("\nThe null runs the identical procedure — same freezing, same path, same receiver, same ")
    a("projection — but draws the sender's clean value from a *different* prompt in the batch. ")
    a("The value carried is a real activation of the right kind; only its correspondence to the ")
    a("prompt is destroyed. Whatever projection survives that is what the method manufactures ")
    a("from nothing, so the 99th percentile of it fixes the false-positive rate at about one in a ")
    a("hundred, in advance, the same role Phase 1's 0.02 cutoff played.\n")
    a("\n")
    a(_table(
        [
            ["null measurements pooled", str(prereg["n_null_measurements"])],
            ["null median |signal|", f"{prereg['null_median']:.4f}"],
            ["null mean |signal|", f"{prereg['null_mean']:.4f}"],
            ["null max |signal|", f"{prereg['null_max']:.4f}"],
            ["99th percentile (raw)", f"{prereg['raw_quantile']:.4f}"],
            ["**threshold (rounded up, 2 s.f.)**", f"**{threshold}**"],
        ],
        ["null statistic", "value"],
    ))
    a(f"\n\nRecorded in `results/phase3_preregistration.json` and committed before the comparison ")
    a("below was run, so the ordering is visible in git history rather than merely claimed.\n")

    a("\n### What this pre-registration is not\n")
    a("It is not blind. Phase 2 printed a handful of real `path_signal` values — the ")
    a("previous-token heads among them — and those numbers were known before this threshold was ")
    a("derived. A genuinely blind pre-registration was no longer available once Phase 2 was ")
    a("published.\n")
    a("\nWhat is claimed instead is narrower and checkable: the number was produced by a fixed ")
    a("rule applied to a null distribution, not selected; the rule and its two free parameters ")
    a("were written into the code before the null was run; and the number was not adjusted after ")
    a("the comparison. A reader who suspects the rule itself was reverse-engineered should weigh ")
    a("the sensitivity table in section 5, which shows what a stricter per-group bar would do.\n")

    a("\n## 2. What the criterion can and cannot see\n")
    a("Two limits are structural and worth stating before any number.\n")
    a("\n- **The output round is out of scope.** Round 0 asks what a head does to the logits ")
    a("directly. Its receiver *is* the output, where a receiver-side measure and the logit ")
    a("measure are the same quantity — there is no independent second opinion to take. Heads ")
    a("found only by direct effect (the name movers) therefore cannot be found by this criterion, ")
    a("and the like-for-like comparison in section 4 restricts the logit criterion to the same ")
    a("rounds to keep that from reading as a failure.\n")
    a("\n- **Under `s2_swap`, S1+1 is undefined rather than zero.** Clean and corrupted coincide ")
    a("there, so the quantity this criterion normalizes by is exactly zero and the ratio does not ")
    a("exist. Those cells are dropped, not scored as misses.\n")

    a("\n## 3. Results by receiver group\n")
    for group in per_group:
        a(f"\n**{group['label']}** — receivers {', '.join(f'`{r}`' for r in group['receivers'])}, ")
        a(f"senders at {group['position']}.\n\n")
        if not group["_signals"]:
            a("*No eligible sender with a defined measurement.*\n")
            continue
        ranked = sorted(group["_signals"], key=lambda k: abs(group["_signals"][k]), reverse=True)[:6]
        rows = [
            [
                f"**{l}.{h}**",
                f"{group['_signals'][(l, h)]:+.3f}",
                "yes" if abs(group["_signals"][(l, h)]) >= threshold else "no",
                classify((l, h)) or "— *not in published circuit*",
            ]
            for (l, h) in ranked
        ]
        a(_table(rows, ["sender", "path signal", f"clears {threshold}", "published class"]))
        a("\n")

    a("\n## 4. The two criteria, side by side\n")
    a("These are two definitions of *found*, reported as two columns rather than one merged ")
    a("score. A head that clears one and not the other is a genuine disagreement about what ")
    a("counts as being part of the circuit, not noise to be averaged away.\n\n")
    rows = []
    for cls in ground_truth.IOI_CIRCUIT:
        total = len(ground_truth.IOI_CIRCUIT[cls])
        rows.append([
            cls,
            f"{cmp_logit.per_class[cls][0]}/{total}",
            f"{cmp_logit_r1.per_class[cls][0]}/{total}",
            f"{cmp_signal.per_class[cls][0]}/{total}",
        ])
    rows.append([
        "**total**",
        f"**{len(cmp_logit.matches)}/26**",
        f"**{len(cmp_logit_r1.matches)}/26**",
        f"**{len(cmp_signal.matches)}/26**",
    ])
    rows.append([
        "precision",
        f"{cmp_logit.precision:.2f}",
        f"{cmp_logit_r1.precision:.2f}",
        f"{cmp_signal.precision:.2f}",
    ])
    a(_table(rows, [
        "published class",
        "logit (all rounds)",
        "logit (rounds 1+)",
        f"receiver-side (>= {threshold})",
    ]))
    a("\n\nThe middle column is the like-for-like one: same rounds, same receivers, same paths, ")
    a("scored by effect on the output instead of delivery to the receiver.\n")

    a("\n### The six heads neither Phase 1 nor Phase 2 found\n\n")
    if previously_missing:
        rows = []
        for head in previously_missing:
            best = signal_by_head.get(head)
            rows.append([
                f"{head[0]}.{head[1]}",
                classify(head) or "?",
                "—" if best is None else f"{best:+.3f}",
                "not measurable" if best is None else ("**yes**" if abs(best) >= threshold else "no"),
            ])
        a(_table(rows, ["head", "published class", "best path signal", f"clears {threshold}"]))
        recovered = [
            h for h in previously_missing
            if signal_by_head.get(h) is not None and abs(signal_by_head[h]) >= threshold
        ]
        a(f"\n\n**{len(recovered)} of {len(previously_missing)}** are recovered by the ")
        a("pre-registered receiver-side criterion")
        if recovered:
            a(": " + ", ".join(f"`{l}.{h}` ({classify((l, h))})" for l, h in recovered) + ".\n")
        else:
            a(". The criterion was fixed in advance and is reported as it fell.\n")
    else:
        a("*None — the earlier phases recovered all 26.*\n")

    a("\n### Where the criteria disagree\n\n")
    only_logit = sorted(cmp_logit.discovered - cmp_signal.discovered)
    only_signal = sorted(cmp_signal.discovered - cmp_logit.discovered)
    both = sorted(cmp_logit.discovered & cmp_signal.discovered)
    a(_table(
        [
            ["found by both", str(len(both)), ", ".join(f"`{l}.{h}`" for l, h in both) or "—"],
            ["logit only", str(len(only_logit)),
             ", ".join(f"`{l}.{h}`" for l, h in only_logit) or "—"],
            ["receiver-side only", str(len(only_signal)),
             ", ".join(f"`{l}.{h}`" for l, h in only_signal) or "—"],
        ],
        ["agreement", "count", "heads"],
    ))
    a("\n\nThe *logit only* set is dominated by heads whose receiver is the output, which the ")
    a("receiver-side criterion cannot evaluate at all. The *receiver-side only* set is the ")
    a("interesting one: paths that demonstrably deliver their content and still do not move the ")
    a("prediction.\n")

    a("\n## 5. Sensitivity: a stricter per-group bar\n")
    a("Pooling the null across receiver groups controls the overall false-positive rate, which is ")
    a("the right target for a single criterion. It also means a group with a wide null gets a bar ")
    a("that is lenient relative to its own noise. Both thresholds were fixed by the same rule at ")
    a("the same time, so the comparison below is not a second bite at the cherry.\n\n")
    rows = []
    for entry in prereg["per_group"]:
        if entry["group_threshold"] is None:
            rows.append([entry["label"], "—", "—", "—"])
            continue
        group = next((g for g in per_group if g["label"] == entry["label"]), None)
        survivors = []
        if group:
            survivors = sorted(
                h for h, v in group["_signals"].items() if abs(v) >= entry["group_threshold"]
            )
        rows.append([
            entry["label"],
            f"{entry['max_abs_null']:.3f}",
            f"{entry['group_threshold']}",
            ", ".join(f"`{l}.{h}`" for l, h in survivors) or "—",
        ])
    a(_table(rows, ["receiver group", "null max", "per-group threshold", "clears it"]))
    a("\n\nWhere a head clears the pooled threshold but not its own group's, that is worth ")
    a("knowing, and it is why both numbers are on record.\n")

    a("\n## 6. Two limitations carried forward, not fixed\n")
    a(_limitations(phase2, per_group, threshold))
    a("\n")

    path.write_text("".join(out), encoding="utf-8")


def _limitations(phase2: dict, per_group: list[dict], threshold: float) -> str:
    lines: list[str] = []

    lines.append("\n### (a) Why the `abc` chain dies at round 1\n")
    abc = phase2["schemes"].get("abc", {})
    round1 = next((r for r in abc.get("rounds", []) if r["index"] == 1), None)
    group = next((g for g in per_group if g["label"] == "abc round 1"), None)
    if round1 and group and group["_signals"]:
        rows = []
        for head, sig in sorted(group["_signals"].items(), key=lambda kv: -abs(kv[1]))[:5]:
            logit = round1["effects"].get(f"{head[0]}.{head[1]}")
            rows.append([
                f"`{head[0]}.{head[1]}`",
                "—" if logit is None else f"{logit:+.4f}",
                f"{sig:+.3f}",
                classify(head) or "— *not in published circuit*",
            ])
        lines.append(
            "\nThe chain dies on the *logit* criterion, and the receiver-side numbers show it is "
            "not because the paths are absent:\n\n"
        )
        lines.append(_table(
            rows, ["sender", "logit effect (Phase 2)", "path signal", "published class"]
        ))
        lines.append(
            "\n\nThe paths deliver. The prediction does not move because `abc` replaces all three "
            "names, so the tokens the logit difference is defined on — the clean prompt's IO and S "
            "— are not in the corrupted prompt at all. Restoring a name mover's query makes it "
            "attend to the right *position*, but the token sitting there is a different name, so "
            "there is no logit difference to restore. That is a property of the counterfactual, "
            "not of the circuit, and it is why `abc` contributes little past round 0 where the "
            "answer tokens still carry the measurement.\n"
        )
    else:
        lines.append("\n*Not measurable from the available Phase 2 record.*\n")

    lines.append(
        "\n### (b) Receiver inputs are still supplied, not searched\n"
        "\nEvery round in Phases 2 and 3 was told *where to look*: that S-inhibition heads act on "
        "name movers' queries, that duplicate-token information arrives as a value at S2, that "
        "induction keys live at S1+1. Those choices come from the paper's account of the "
        "mechanism. Which heads turn up was never constrained — all 144 are swept every round — "
        "but the question asked of them was.\n"
        "\nThat is the line this project has not yet crossed. Everything so far is **guided "
        "rediscovery**: given the right question, the method finds the right components, and "
        "finds them in the right causal order. The autonomous loop the README describes has to "
        "generate the questions too — searching over receiver inputs, positions and depths "
        "without being handed the mechanism first — and nothing here demonstrates that.\n"
        "\nThe distinction matters most exactly where the project is aimed. On a circuit nobody "
        "has published there is no paper to supply the receiver inputs, so a method that needs "
        "them supplied does not yet transfer. Search over receiver specifications is the concrete "
        "next problem, and it is a larger one than either patching primitive: the space is the "
        "product of receiver head, input, and position, and unlike this phase there is no answer "
        "key to check the search against.\n"
    )
    return "".join(lines)
