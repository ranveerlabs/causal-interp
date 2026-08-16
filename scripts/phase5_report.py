"""Phase 5 report: what the hand-built pieces were buying, measured."""

from __future__ import annotations

from pathlib import Path

from causal_interp import ground_truth
from causal_interp.metrics import METRICS

METRIC_LABEL = {
    "logit_diff": "logit difference (hand-built)",
    "kl": "KL divergence (general)",
    "tv": "total variation (general)",
}

CORRUPTION_LABEL = {
    "s2_swap": "`s2_swap` — semantic, one token",
    "abc": "`abc` — semantic, three tokens",
    "random_vocab_s2": "`random_vocab_s2` — generic token, position supplied",
    "random_vocab_any": "`random_vocab_any` — generic token, generic position",
}


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def write_report(path: Path, meta, results, agreement, threshold) -> None:
    out: list[str] = []
    a = out.append
    total = ground_truth.PUBLISHED_HEAD_COUNT

    a("# Phase 5 — scoping the remaining hand-built dependency\n")
    a("Phase 4 removed the need to be told which head input, at which position, to interrogate, ")
    a("and named what was left: the task, the corruption schemes and the metric are all still ")
    a("built from knowledge of what the model does and what a correct answer looks like.\n")
    a("\nThis phase does not try to build those automatically. It measures how much of the ")
    a("project's result actually depended on them, using IOI — where the answer is still known ")
    a("and can check the substitution. What each piece encodes is itemised in ")
    a("[PHASE5_AUDIT.md](PHASE5_AUDIT.md), written before any of this ran.\n")

    a("\n## Run configuration\n")
    a(_table([[k, f"`{v}`"] for k, v in meta.items()], ["setting", "value"]))

    a("\n\n## 1. How much room each counterfactual leaves to measure in\n")
    a("Before any circuit result: a corruption is only useful if the clean and corrupted runs ")
    a("differ enough that a patched run can sit somewhere informative between them. That range is ")
    a("what every normalized recovery in this project is divided by.\n\n")
    rows = []
    for corruption, block in results.items():
        rows.append([
            CORRUPTION_LABEL.get(corruption, corruption),
            f"{block['clean_logit_diff']:+.2f} → {block['corrupted_logit_diff']:+.2f}",
            f"{block['logit_span']:.2f}",
            f"{block['kl_span']:.2f}",
            f"{block['tv_span']:.3f}",
        ])
    a(_table(rows, [
        "corruption", "logit diff, clean → corrupted", "logit span", "KL span", "TV span",
    ]))
    a("\n\nThe semantic corruptions were designed so the answer flips, which is why `s2_swap` ")
    a("spans a signed range twice the size of anything the generic ones produce. A generic token ")
    a("substitution damages the prompt without reversing the behaviour, so the corrupted run ")
    a("still partly does the task — and the measurement has correspondingly less room.\n")

    a("\n## 2. Can the metric be replaced?\n")
    a("Every combination of corruption and metric, scored against the same published circuit at ")
    a(f"the same cutoff ({threshold}) Phase 1 used. All three metrics come from the *same* ")
    a("forward passes, so any difference between columns is the metric and nothing else.\n\n")

    rows = []
    for corruption, block in results.items():
        row = [CORRUPTION_LABEL.get(corruption, corruption)]
        for name in METRICS:
            entry = block["metrics"][name]
            row.append(f"{entry['matched']}/{total} · p{entry['precision']:.2f}")
        rows.append(row)
    a(_table(rows, ["corruption"] + [METRIC_LABEL[m] for m in METRICS]))
    a("\n\nEach cell is *heads recovered / 26* and the precision behind it.\n")

    a("\n### The same comparison with the scale removed\n")
    a("A fixed cutoff is not neutral between metrics: one whose effects simply run larger clears ")
    a("it more often without discriminating any better. Taking each metric's **top 26** — the ")
    a("size of the published circuit — removes the scale and leaves only the ranking, so ")
    a("precision and recall coincide.\n\n")
    rows = []
    for corruption, block in results.items():
        rows.append([
            CORRUPTION_LABEL.get(corruption, corruption),
            *[f"{block['metrics'][m].get('size_matched', 0)}/{total}" for m in METRICS],
        ])
    a(_table(rows, ["corruption"] + [METRIC_LABEL[m] for m in METRICS]))

    a("\n### Do the metrics rank heads the same way?\n")
    a("Spearman correlation between the per-head rankings each metric produces. A high value ")
    a("means the metrics disagree about magnitudes but not about which heads matter.\n\n")
    rows = []
    for corruption, pairs in agreement.items():
        rows.append([
            CORRUPTION_LABEL.get(corruption, corruption),
            *[f"{pairs[k]:+.3f}" for k in ("logit_diff vs kl", "logit_diff vs tv", "kl vs tv")],
        ])
    a(_table(rows, ["corruption", "logit ~ KL", "logit ~ TV", "KL ~ TV"]))

    a("\n\n### By class, on the primary corruption\n")
    primary = results.get("s2_swap")
    if primary:
        rows = []
        for cls in ground_truth.IOI_CIRCUIT:
            n_cls = len(ground_truth.IOI_CIRCUIT[cls])
            rows.append([
                cls,
                *[f"{primary['metrics'][m]['per_class'][cls][0]}/{n_cls}" for m in METRICS],
            ])
        rows.append([
            "**total**",
            *[f"**{primary['metrics'][m]['matched']}/{total}**" for m in METRICS],
        ])
        a(_table(rows, ["published class"] + [METRIC_LABEL[m] for m in METRICS]))

    a("\n\n## 3. Can the corruption be made generic?\n")
    a("The bottom two rows of the grid in section 2 answer this. Reading them against the top ")
    a("two is the whole experiment: the generic schemes substitute a uniformly drawn vocabulary ")
    a("token rather than a semantically chosen name, and `random_vocab_any` does not even choose ")
    a("the position.\n\n")
    rows = []
    for corruption in ("s2_swap", "random_vocab_s2", "random_vocab_any"):
        block = results.get(corruption)
        if not block:
            continue
        best = max(block["metrics"].items(), key=lambda kv: kv[1]["matched"])
        rows.append([
            CORRUPTION_LABEL.get(corruption, corruption),
            f"{best[1]['matched']}/{total}",
            METRIC_LABEL[best[0]],
            f"{best[1]['precision']:.2f}",
        ])
    a(_table(rows, ["corruption", "best recovery", "under which metric", "precision"]))

    a("\n\n## 4. What this phase settled\n")
    a(_findings(results, agreement, threshold))
    a("\n")

    path.write_text("".join(out), encoding="utf-8")


def _findings(results, agreement, threshold) -> str:
    total = ground_truth.PUBLISHED_HEAD_COUNT
    base = results["s2_swap"]["metrics"]
    lines: list[str] = []

    logit = base["logit_diff"]["matched"]
    kl = base["kl"]["matched"]
    tv = base["tv"]["matched"]
    corr = agreement["s2_swap"]["logit_diff vs kl"]

    s_logit = base["logit_diff"].get("size_matched", 0)
    s_kl = base["kl"].get("size_matched", 0)
    s_tv = base["tv"].get("size_matched", 0)

    lines.append("### The metric: substitutable\n\n")
    lines.append(
        f"On the corruption every earlier phase used, the hand-built logit difference recovers "
        f"{logit}/{total} heads at the fixed cutoff. Replacing it with a KL divergence over the "
        f"full next-token distribution — which needs no knowledge of which token is correct — "
        f"recovers {kl}/{total}, and total variation {tv}/{total}.\n"
    )
    lines.append(
        f"\nThe fixed cutoff flatters whichever metric runs larger, so the size-matched numbers "
        f"are the ones to read: top-26 gives **{s_logit}** for the hand-built metric, **{s_kl}** "
        f"for KL and **{s_tv}** for total variation. The per-head rankings of the hand-built "
        f"metric and KL correlate at {corr:+.2f}.\n"
    )
    if s_kl >= s_logit - 2:
        lines.append(
            "\nThat is the substantive positive result of this phase. The single piece of "
            "knowledge that looked most load-bearing — knowing which token is the right answer — "
            "turns out not to be needed to *locate the circuit*. The two runs the corruption "
            "already provides are enough, because the difference between their output "
            "distributions is dominated by the behaviour under study.\n"
        )
    else:
        lines.append(
            "\nThe general metrics lose ground, and the loss is reported as it fell rather than "
            "recovered by adjusting the cutoff.\n"
        )
    lines.append(
        "\nWorth being precise about what this does not show. The distributional metrics still "
        "require a *clean and a corrupted run* to compare, so they inherit whatever knowledge "
        "built the corruption. They remove the answer key, not the counterfactual.\n"
    )

    lines.append("\n### The corruption: partly, and the cost is not where it was expected\n\n")
    gen_s2 = results["random_vocab_s2"]["metrics"]
    gen_any = results["random_vocab_any"]["metrics"]
    # The corners that matter are defined by how much knowledge each *combination*
    # uses. Taking the best metric per corruption would mix configurations that
    # still consult the answer key with ones that do not.
    gen_s2_logit = gen_s2["logit_diff"].get("size_matched", 0)
    gen_any_logit = gen_any["logit_diff"].get("size_matched", 0)
    fully_s2 = max(gen_s2[m].get("size_matched", 0) for m in ("kl", "tv"))
    fully_any = max(gen_any[m].get("size_matched", 0) for m in ("kl", "tv"))

    lines.append(_table(
        [
            ["hand-built corruption, hand-built metric", f"**{s_logit}/{total}**", "`s2_swap` + logit difference"],
            ["hand-built corruption, general metric", f"**{max(s_kl, s_tv)}/{total}**", "`s2_swap` + KL / TV"],
            ["generic corruption, hand-built metric", f"**{gen_s2_logit}/{total}**", "`random_vocab_s2` + logit difference"],
            ["generic corruption, general metric", f"**{fully_s2}/{total}**", "`random_vocab_s2` + KL / TV"],
            ["nothing supplied at all", f"**{fully_any}/{total}**", "`random_vocab_any` + KL / TV"],
        ],
        ["what is supplied", "size-matched recovery", "configuration"],
    ))

    lines.append(
        f"\n\nThe expectation going in — stated in the audit and in the brief that commissioned "
        f"this phase — was that a generic corruption would degrade badly. It does, but not where "
        f"that prediction pointed. Replacing the semantic swap with a uniformly drawn token while "
        f"keeping the hand-built metric costs almost nothing ({gen_s2_logit}/{total} against "
        f"{s_logit}/{total}). The loss appears only when *both* pieces are generic: "
        f"{fully_s2}/{total} with the position still supplied, and {fully_any}/{total} with "
        f"nothing supplied.\n"
    )
    lines.append(
        "\nThat pattern is worth stating carefully, because it inverts the natural reading. The "
        "two hand-built pieces are not independently load-bearing and they are not additive. "
        "Either one alone carries enough task knowledge to locate the circuit; what fails is "
        "removing both. A corruption that damages the prompt arbitrarily still produces a usable "
        "signal *if the metric knows what to look at*, and a metric that looks at everything "
        "still works *if the corruption was aimed at the right thing*. Neither survives the other "
        "being taken away.\n"
    )
    lines.append(
        f"\nThe mechanism is visible in section 1. `s2_swap` was built to *reverse* the behaviour, "
        f"so the corrupted run sits as far from clean as the task allows. A random token damages "
        f"the prompt instead: the corrupted run still partly performs the task, the span "
        f"collapses, and every normalized recovery is divided by a smaller number. With a metric "
        f"that reads only the two answer tokens, that shrunken span is still pointed at the right "
        f"quantity. With a metric that reads the whole distribution, the shrunken span is shared "
        f"out over every way a corrupted prompt differs, most of which have nothing to do with "
        f"the circuit.\n"
    )
    lines.append(
        f"\nSo the negative result stands, in the form that matters: the fully generic "
        f"configuration recovers {fully_any}/{total} against {s_logit}/{total}, a loss of about a "
        f"third, and on an unfamiliar circuit there would be no published answer to notice that "
        f"degradation against.\n"
    )

    lines.append("\n### The task: not attempted, and not because of budget\n\n")
    lines.append(
        "Constructing a task means choosing which behaviour to study, and that is a different "
        "kind of problem from anything in Phases 1-5. Every phase so far takes a behaviour as "
        "given and asks how the model implements it. No improvement to patching, searching or "
        "scoring turns that into a method for finding behaviours worth studying in the first "
        "place — the machinery here has no way to propose a hypothesis, only to test one.\n"
    )
    lines.append(
        "\nA weak attempt was available and was deliberately not made. Sweeping templates, or "
        "mining a corpus for prompts with a predictable completion, would have produced a section "
        "in this report and no evidence that the resulting tasks isolate anything mechanistically "
        "interesting. The honest position is that this is open.\n"
    )

    lines.append("\n### Where that leaves the autonomy claim\n\n")
    lines.append(_ladder(s_logit, s_kl, fully_s2, fully_any, total))
    return "".join(lines)


def _ladder(logit: int, kl: int, gen_s2: int, gen_any: int, total: int) -> str:
    rows = [
        ["behaviour to study", "**supplied**", "no method attempted; a different kind of problem"],
        ["task template", "**supplied**", "out of scope for this phase, explicitly"],
        ["corruption content", "**supplied**", f"generic substitute costs little alone, {gen_s2}/{total} once the metric is generic too"],
        ["corruption position", "searchable", "Phase 4"],
        ["receiver input and position", "searchable", "Phase 4, 16 of 17 recovered"],
        ["answer key in the metric", "**not needed**", f"KL over the output distribution: {kl}/{total} vs {logit}/{total}"],
        ["circuit components and wiring", "discovered", "Phases 1-3"],
    ]
    table = _table(rows, ["ingredient", "status", "evidence"])
    return (
        "Stacking what each phase has established, from the outside in:\n\n"
        + table
        + "\n\nThe line between the top two rows and the rest is the real boundary. Everything "
        "below it is a question about a behaviour that has already been chosen, and the project "
        "now answers those with progressively less help. Everything above it is the question of "
        "which behaviour to look at, and this project has never addressed it.\n"
        "\nA next phase pursuing autonomy on task construction would have to start there, and it "
        "would need a validation strategy that does not exist yet: IOI cannot check it, because "
        "IOI *is* the supplied task. Checking whether an automatically constructed task is a good "
        "one requires either a second published circuit to rediscover, or a criterion for task "
        "quality that does not reduce to 'it found the thing we already knew about'.\n"
    )
