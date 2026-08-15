"""Phase 4 report writer, kept out of the search module for the same reason
Phase 3's analysis was: the code that consults the answer key should not be
reachable from the code that searches.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from causal_interp import ground_truth
from causal_interp.ground_truth import classify
from causal_interp.ioi import POSITIONS


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _semantic_of(labels: dict[int, str], index: int) -> str:
    """What a bare token index turned out to be, looked up after the search."""
    return labels.get(index, "—")


def write_report(path: Path, meta, semantic, absolute, abs_labels, check, confirmations, threshold) -> None:
    out: list[str] = []
    a = out.append

    a("# Phase 4 — searching for receiver specifications\n")
    a("Phases 1-3 were told which head input, at which position, to interrogate. Those choices ")
    a("came from the paper's account of the mechanism, which is why nothing so far transfers to a ")
    a("circuit nobody has published. This phase searches for them instead, on the same task, and ")
    a("then checks the search against the answer it was not allowed to see.\n")
    a("\nThe space and budget were fixed in ")
    a("[PHASE4_SEARCH_SPACE.md](PHASE4_SEARCH_SPACE.md), committed before the search code was ")
    a("written. `causal_interp/search.py` does not import `ground_truth`, and the run asserts ")
    a("that before starting.\n")

    a("\n## Run configuration\n")
    a(_table([[k, f"`{v}`"] for k, v in meta.items()], ["setting", "value"]))

    a("\n\n## 1. What the search found, before checking\n")
    a("Stage A scores every receiver specification by splicing that one input, at that one ")
    a("position, from the clean run into the corrupted one. Exhaustive over the grid.\n")

    for label, block, ds in (
        ("Semantic positions", semantic, None),
        ("Absolute positions (single template, no semantic labels)", absolute, abs_labels),
    ):
        a(f"\n**{label}** — top 12 of {len(block['scores'])} specifications:\n\n")
        rows = []
        for spec in block["ranked"][:12]:
            position = spec.position
            if ds is not None:
                position = f"`{spec.position}` (= {_semantic_of(ds, int(spec.position[1:]))})"
            rows.append([
                f"**{spec.layer}.{spec.head}**",
                spec.input,
                position,
                f"{block['scores'][spec]:+.3f}",
                classify((spec.layer, spec.head)) or "— *not in published circuit*",
            ])
        a(_table(rows, ["head", "input", "position", "screen score", "published class"]))
        a("\n")

    a("\n### Which positions the search prefers overall\n")
    a("Counting how often each position appears in the top 50 specifications — the search's own ")
    a("view of where the task's information lives, with no labels supplied in the absolute case.\n\n")
    for label, block, ds in (("semantic", semantic, None), ("absolute", absolute, abs_labels)):
        counts = Counter(spec.position for spec in block["ranked"][:50])
        rendered = ", ".join(
            f"`{pos}`"
            + (f" (= {_semantic_of(ds, int(pos[1:]))})" if ds is not None else "")
            + f" ×{k}"
            for pos, k in counts.most_common(6)
        )
        a(f"- **{label}**: {rendered}\n")

    a("\n## 2. The rediscovery check\n")
    a("Only now is the published circuit consulted. For each published head that has a published ")
    a("*receiver* specification, its 21 candidate specifications are ranked by the search's own ")
    a("score, and the published one is located in that ranking.\n")
    a(f"\nOutcome rule, stated openly and fixed at analysis time rather than pre-registered: ")
    a(f"**agreement** if the published spec ranks first, **ambiguous** if it ranks in the top ")
    a(f"{meta['ambiguity_rule'].split('top ')[1].split(' ')[0]} and scores within ")
    a("20% of the best, **disagreement** otherwise. Raw ranks and scores are shown so the labels ")
    a("can be second-guessed.\n\n")

    checkable = [r for r in check if r.get("published_spec")]
    rows = []
    for row in checkable:
        rows.append([
            f"**{row['head']}**",
            row["class"],
            f"`{row['published_spec']}`",
            str(row.get("published_rank") or "—"),
            f"`{row.get('top_spec', '—')}`",
            f"{row.get('top_score', 0.0):+.3f}",
            {"agreement": "✅ agreement", "ambiguous": "◐ ambiguous",
             "unmeasurable": "⊘ unmeasurable", "disagreement": "✗ disagreement"}.get(
                row["outcome"], row["outcome"]),
        ])
    a(_table(rows, [
        "head", "class", "published spec", "its rank", "search's top", "top score", "outcome",
    ]))

    tally = Counter(r["outcome"] for r in checkable)
    total = len(checkable)
    a(f"\n\n**{tally.get('agreement', 0)}/{total} agreement, ")
    a(f"{tally.get('ambiguous', 0)}/{total} ambiguous, ")
    a(f"{tally.get('unmeasurable', 0)}/{total} unmeasurable, ")
    a(f"{tally.get('disagreement', 0)}/{total} disagreement.**\n")

    unmeasurable = [r for r in checkable if r["outcome"] == "unmeasurable"]
    if unmeasurable:
        heads = ", ".join(f"`{r['head']}`" for r in unmeasurable)
        a(f"\n**Unmeasurable is not disagreement.** For {heads} the published specification scores ")
        a("*exactly* zero — not a small number, an exact floating-point zero. Under the `s2_swap` ")
        a("corruption the S1+1 position is bit-identical between the clean and corrupted runs, so ")
        a("every one of the 432 specifications at that position is unscoreable, the published one ")
        a("included. The search did not weigh `k@S1+1` against the alternatives and prefer ")
        a("something else; it was handed a counterfactual that cannot see that position at all. ")
        a("This is the same structural blindness Phase 1 measured (576/576 exact zeros before S2) ")
        a("arriving again, one phase later, in a new guise.\n")
        a("\nCounting these as search failures would credit the search with a defect belonging to ")
        a("the corruption scheme. Counting them as successes would be worse. They are reported as ")
        a("their own category and excluded from both.\n")

    unspecified = [r for r in check if not r.get("published_spec")]
    if unspecified:
        heads = ", ".join(f"`{r['head']}`" for r in unspecified)
        a(f"\n{len(unspecified)} published heads have no published *receiver* specification to ")
        a(f"check against — {heads}. The paper describes them by what they send, not what they ")
        a("receive. A search result for them is unfalsifiable here rather than correct, so they ")
        a("are excluded from the tally rather than counted as successes.\n")

    a("\n## 3. Did the search need the position labels?\n")
    a("The semantic search uses positions named IO, S1, S2 and END — labels that already encode ")
    a("which name is the indirect object and where the subject repeats. The absolute search has ")
    a("only bare token indices on a single template, so it is the one that tests whether the ")
    a("method can find the structure rather than be handed it.\n\n")
    a(_absolute_verdict(absolute, abs_labels))

    a("\n## 4. Stage B — what feeds the specifications the search chose\n")
    a(f"For the top surviving specifications, every head below them was swept as a sender and ")
    a(f"scored with both of the project's criteria: delivery to the receiver (`path_signal`, ")
    a(f"against Phase 3's recorded threshold of {threshold}) and effect on the output logits.\n\n")
    if confirmations:
        rows = []
        for entry in confirmations[:10]:
            top = sorted(entry["_signals"], key=lambda k: abs(entry["_signals"][k]), reverse=True)[:3]
            senders = ", ".join(
                f"{l}.{h} ({entry['_signals'][(l, h)]:+.2f}"
                + (f", {classify((l, h))}" if classify((l, h)) else "")
                + ")"
                for l, h in top
            )
            rows.append([
                f"`{entry['spec']}`",
                f"{entry['screen_score']:+.3f}",
                str(len(entry["senders_clearing_signal"])),
                senders or "—",
            ])
        a(_table(rows, ["specification", "screen score", "senders clearing threshold", "top senders"]))
    else:
        a("*No specification reached stage B.*")

    a("\n\n## 5. Assessment\n")
    a(_assessment(tally, total, semantic, absolute, abs_labels, confirmations))
    a("\n")

    path.write_text("".join(out), encoding="utf-8")


def _absolute_verdict(absolute, abs_labels) -> str:
    """Does the unlabelled search land on the same positions the labelled one uses?"""
    top = absolute["ranked"][:50]
    hits = Counter()
    for spec in top:
        index = int(spec.position[1:])
        hits[_semantic_of(abs_labels, index)] += 1
    named = sum(v for k, v in hits.items() if k != "—")
    lines = [
        f"Of the top 50 specifications the absolute search returned, **{named}** sit at token "
        f"indices that turn out to carry a semantic label, and {50 - named} do not.\n\n",
    ]
    rows = [[k or "—", str(v)] for k, v in hits.most_common()]
    lines.append(_table(rows, ["what that index turns out to be", "count in top 50"]))
    lines.append(
        "\n\nThe labels in that table were attached *after* the search, purely to interpret its "
        "output. The search itself ranked bare indices.\n"
    )
    return "".join(lines)


def _assessment(tally, total, semantic, absolute, abs_labels, confirmations) -> str:
    agree = tally.get("agreement", 0)
    ambiguous = tally.get("ambiguous", 0)
    disagree = tally.get("disagreement", 0)
    unmeasurable = tally.get("unmeasurable", 0)
    scoreable = total - unmeasurable
    lines = []

    lines.append(
        f"On the {total} heads where the paper names a receiver specification, the search agrees "
        f"with it {agree} times, is ambiguous {ambiguous} times, and disagrees "
        f"{disagree} time{'s' if disagree != 1 else ''}. A further {unmeasurable} are unscoreable "
        f"under this corruption scheme, so of the {scoreable} the search could actually weigh, it "
        f"recovered the published specification **{agree}**.\n"
    )

    lines.append(
        "\n**What this does and does not license.** The search recovered receiver specifications "
        "without being told them, on a task where the answer happens to be known. That is the "
        "step Phases 1-3 could not take. It is not the same as autonomous discovery, and three "
        "things still stand between the two.\n"
    )
    lines.append(
        "\n1. **The screen is a logit-effect screen.** A receiver only reaches stage B if splicing "
        "its input moves the output. Phase 3 established that some genuine circuit links do not "
        "move the output — the previous-token heads are exactly that case — so this search "
        "inherits that blind spot by construction, not by accident.\n"
        "\n2. **The task, the counterfactual and the metric are still supplied.** The IOI "
        "templates, both corruption schemes and the logit-difference metric were all designed by "
        "hand from knowledge of what the model is doing. A system pointed at an unfamiliar "
        "circuit would have to construct its own counterfactual, and nothing here does that. It "
        "is the largest remaining gap and it is larger than the one this phase closed.\n"
        "\n3. **There is no answer key on an unfamiliar circuit.** Every outcome above was legible "
        "only because a published specification existed to compare against. The search itself "
        "emits a ranking either way, and nothing in that ranking distinguishes the case where it "
        "is right from the case where it is wrong.\n"
    )

    if ambiguous:
        lines.append(
            "\n**The ambiguity is a substantive finding.** Where several specifications for the "
            "same head score close together, the ranking is not evidence about which one the "
            "mechanism uses. On this task that is visible because the paper says which is right; "
            "on an unfamiliar one it would be invisible, and a confident-looking top-ranked "
            "specification would be reported as the answer.\n"
        )
    if disagree:
        lines.append(
            f"\n**The {disagree} genuine disagreement"
            f"{'s matter' if disagree != 1 else ' matters'} more than the count suggests.** There "
            "the search weighed the published specification against the alternatives and preferred "
            "a different one. On this task that is catchable. On an unfamiliar circuit the same "
            "result would be indistinguishable from a correct answer, since the only thing marking "
            "it wrong is a published account to compare against.\n"
        )
    if not ambiguous:
        lines.append(
            "\nWorth noting what did *not* happen: no published specification landed in the "
            "ambiguous band, where the search ranks it near the top but cannot separate it from a "
            "rival. Rankings here were decisive rather than marginal. That is a better outcome "
            "than the alternative, but it is a property of this task at this sample size and "
            "should not be assumed to hold elsewhere — an autonomous use of the method still "
            "needs a calibrated notion of when its own ranking is uninformative, and this phase "
            "does not provide one.\n"
        )

    return "".join(lines)
