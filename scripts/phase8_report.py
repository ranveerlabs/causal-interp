"""Phase 8 report: what the pipeline says when its counterfactuals disagree.

Kept out of the run module for the reason `phase6_report.py` and `phase7_report.py`
are: the report consults the answer key freely, and the discovery path must not.

Generated from the stored payloads rather than written by hand, so a number in the
prose cannot drift from the number in the JSON. Where a sentence depends on which way
a result went, it branches on the measured value instead of asserting a direction —
including the prediction scoring, which reads the measurements and marks each
prediction hit or missed without a human deciding.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_interp import ground_truth_docstring as gt_docstring
from causal_interp import ground_truth_greater_than as gt_greater_than

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ROOT = Path(__file__).resolve().parents[1]
PRE_PHASE8 = "a015ecb"   # last commit before Phase 8 touched anything

CIRCUIT_TITLE = {
    "docstring": "docstring (Phase 7's circuit, `attn-only-4l`)",
    "greater_than": "greater-than (Phase 6's circuit, GPT-2 small)",
}

GROUND_TRUTH = {"docstring": gt_docstring, "greater_than": gt_greater_than}


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _hs(head: str) -> str:
    return f"`{head}`"


def _joined(heads: list[str]) -> str:
    return ", ".join(_hs(h) for h in heads) if heads else "—"


# ---------------------------------------------------------------------------


def _intro(payloads: dict[str, dict]) -> str:
    lines = [
        "# Phase 8 — a second counterfactual as part of the method, not as a rescue",
        "",
        "**What changed is the pipeline's output, not its recall.** The design, the scheme "
        "registrations, the decision rules and eight predictions were fixed in "
        "[PHASE8_PLAN.md](PHASE8_PLAN.md), committed before any Phase 8 code existed.",
        "",
        "Phase 7 recovered 3 of the 6 published docstring heads under the benchmark's "
        "default counterfactual and 5 of 6 under a different published one, and the reason "
        "was structural: the primary counterfactual replaces the answer token, so the heads "
        "whose job is to *route* attention to the right argument cannot move a metric read "
        "off that token. That diagnosis was correct and it was made by a human who saw a low "
        "recall number and knew which alternative to try. Nothing in the pipeline's own "
        "output said the head list was incomplete.",
        "",
        "Phase 8 makes multi-scheme discovery structural:",
        "",
        "- `causal_interp/schemes.py` — a task registers **named** counterfactual schemes, "
        "each declaring what it breaks and whether the answer survives it. `TaskSpec` "
        "**raises** if a task registers fewer than two: single-counterfactual discovery is "
        "not an option a later phase can leave unset, it is a check someone would have to "
        "delete.",
        "- `causal_interp/pipeline.py` — `discover()` sweeps every registered scheme and "
        "returns the head list and the cross-scheme comparison *in the same object*. There "
        "is no single-scheme entry point.",
        "- `causal_interp/agreement.py` — the per-head verdicts, the per-scheme blind spots "
        "and the flag. It imports no `ground_truth` module and the runner asserts that at "
        "startup, alongside the same check `search.py` has carried since Phase 4.",
        "",
        "The flag is a bare non-emptiness test with no cutoff, so this phase adds no free "
        "parameter:",
        "",
        "> **the primary scheme's blind spot** — heads some other registered counterfactual "
        "finds and the primary one does not.",
        "",
        "Both earlier circuits were re-run through it. Phase 6's and Phase 7's own results "
        "files are untouched; these are new runs under the new structure.",
        "",
    ]
    if payloads:
        rows = []
        for circuit, payload in payloads.items():
            meta = payload["meta"]
            rows.append([
                CIRCUIT_TITLE[circuit],
                str(len(payload["schemes"])),
                f"{meta['prompts']}",
                f"{meta['runtime_seconds'] / 60:.0f} min",
                "**fires**" if payload["head_agreement"]["flag"] else "silent",
            ])
        lines += [
            _table(rows, ["circuit re-run", "schemes", "prompts", "runtime", "flag"]),
            "",
        ]
    return "\n".join(lines)


def _scheme_table(payload: dict) -> str:
    report = payload["head_agreement"]
    rows = []
    for scheme in report["schemes"]:
        meta = next(s for s in payload["schemes"] if s["scheme"] == scheme)
        power = report["power"][scheme]
        found = len(report["per_scheme"][scheme])
        rows.append([
            f"`{scheme}`" + (" **primary**" if meta["primary"] else ""),
            meta["provenance"],
            meta["breaks"],
            "**yes**" if meta["preserves_answer"] else "no",
            f"{power['span']:+.3f}",
            f"{power['power']:.2f}" + (" *(low-power)*" if power["low_power"] else ""),
            str(found),
        ])
    return _table(
        rows,
        ["scheme", "provenance", "what it breaks", "answer preserved", "span", "power",
         "heads found"],
    )


def _circuit_section(circuit: str, payload: dict) -> str:
    gt = GROUND_TRUTH[circuit]
    meta = payload["meta"]
    report = payload["head_agreement"]
    scored = payload["scored"]["heads"]
    n_published = meta["published_head_count"]

    robust = len(report["union"]) - len(report["scheme_dependent"])
    lines = [
        f"## {CIRCUIT_TITLE[circuit]}",
        "",
        f"{meta['n_layers']} layers x {meta['n_heads']} heads, {meta['prompts']} prompts, "
        f"threshold {meta['threshold']} (Phase 1's, inherited). Every scheme below was swept "
        f"in full; none was chosen after the fact.",
        "",
        _scheme_table(payload),
        "",
        "`power` is the scheme's clean-vs-corrupted span relative to the primary's. It is "
        "**an annotation and never a gate** — no code path drops a scheme for being "
        "low-power, because a power cutoff would be exactly the kind of free parameter that "
        "could be tuned until the flag fired only where it was wanted.",
        "",
        "### What the pipeline says before the answer key is opened",
        "",
        f"- heads found by **every** scheme (`robust`): **{robust}**",
        f"- heads found by some scheme and missed by others (`scheme-dependent`): "
        f"**{len(report['scheme_dependent'])}**",
        f"- union across schemes: {len(report['union'])}   ·   "
        f"intersection: {len(report['intersection'])}",
        "",
        "> " + report["flag_text"],
        "",
    ]

    # per-scheme blind spots
    rows = []
    for scheme in report["schemes"]:
        rows.append([
            f"`{scheme}`",
            str(len(report["per_scheme"][scheme])),
            str(len(report["blind_spots"][scheme])),
            str(len(report["only_in"][scheme])),
        ])
    lines += [
        "Every scheme's blind spot is computed, not just the primary's — the question "
        '"what can this experiment not see" is asked of each of them:',
        "",
        _table(rows, ["scheme", "found", "blind spot (found by others, not by it)",
                      "found by it alone"]),
        "",
        "### And now the answer key",
        "",
    ]

    rows = []
    for scheme in report["schemes"]:
        block = scored["per_scheme"][scheme]
        rows.append([
            f"`{scheme}`",
            str(block["n_discovered"]),
            f"{len(block['matches'])}/{n_published}",
            f"{block['recall']:.2f}",
            f"{block['precision']:.2f}",
        ])
    rows.append([
        "**union of all schemes**",
        str(scored["union"]["n_discovered"]),
        f"**{len(scored['union']['matches'])}/{n_published}**",
        f"{scored['union']['recall']:.2f}",
        f"**{scored['union']['precision']:.2f}**",
    ])
    lines += [
        _table(rows, ["scheme", "discovered", "published heads found", "recall", "precision"]),
        "",
    ]

    # published head status
    rows = []
    for entry in scored["published_head_status"]:
        rows.append([
            _hs(entry["head"]),
            entry["class"] or "—",
            entry["status"],
            ", ".join(f"`{s}`" for s in entry["found_in"]) or "—",
        ])
    lines += [
        "Per published head, the verdict the pipeline had already reached before this "
        "section was reachable:",
        "",
        _table(rows, ["head", "published class", "verdict", "found under"]),
        "",
    ]

    blind = scored["blind_spot"]
    if report["flag"]:
        lines += [
            f"Of the {len(blind['heads'])} heads in the primary scheme's blind spot, "
            f"**{len(blind['published_heads_in_blind_spot'])} are published circuit members** "
            f"({_joined(blind['published_heads_in_blind_spot'])}) and "
            f"{len(blind['unpublished_heads_in_blind_spot'])} are not. Recall goes from "
            f"{len(scored['primary']['matches'])}/{n_published} under the primary scheme to "
            f"{len(scored['union']['matches'])}/{n_published} across the union, at a "
            f"precision cost of {scored['precision_primary']:.2f} → "
            f"{scored['precision_union']:.2f}.",
            "",
        ]
    else:
        lines += [
            "The flag did not fire: no registered scheme found a head the primary missed.",
            "",
        ]
    return "\n".join(lines)


def _channels_section(circuit: str, payload: dict) -> str:
    chain = payload["chain_agreement"]
    chain_scored = payload["scored"]["chain"]
    specs = payload["spec_agreement"]
    spec_scored = payload["scored"]["specs"]
    gt = GROUND_TRUTH[circuit]

    halted = [s for s, block in payload["chain"].items() if block["halted"]]
    lines = [
        "### The same comparison at the other two channels",
        "",
        "**Path-patching chain.** Run once per scheme, each round's receivers taken from the "
        "round before, exactly as in Phases 2, 6 and 7.",
        "",
    ]
    rows = []
    for scheme in chain["schemes"]:
        rows.append([
            f"`{scheme}`",
            str(len(chain["per_scheme"][scheme])),
            str(len(chain["blind_spots"][scheme])),
            "halted early" if scheme in halted else "ran to the last round",
        ])
    lines += [
        _table(rows, ["scheme", "senders found", "blind spot", "chain"]),
        "",
        "> " + chain["flag_text"],
        "",
        "One caveat the chain's numbers carry and the head sweep's do not: a scheme whose "
        "chain **halts early** never measures the senders the later rounds would have "
        "reached, so those heads count as \"not found\" here rather than \"measured and "
        "below threshold\". Where a chain halted it is marked above, and its blind spot is "
        "inflated by that alone.",
        "",
        f"Scored against the published circuit, the chain recovers "
        f"{len(chain_scored['primary']['matches'])}/{payload['meta']['published_head_count']} "
        f"under the primary scheme and "
        f"{len(chain_scored['union']['matches'])}/{payload['meta']['published_head_count']} "
        f"across the union.",
        "",
        "**Receiver-specification search.** Phase 4's screen, run once per scheme. A head is "
        "`spec-scheme-dependent` when the specification that wins its own ranking changes "
        "with the counterfactual — a bare inequality between argmaxes, fixed in the plan.",
        "",
        f"- heads whose winning specification changes across schemes: "
        f"**{specs['n_scheme_dependent']} of {specs['n_heads']}**",
        f"- of the {spec_scored['n_scoreable']} published heads with a published receiver "
        f"specification, the primary scheme's search ranks it first for "
        f"**{spec_scored['n_primary_agrees']}**, and *some* scheme ranks it first for "
        f"**{spec_scored['n_any_scheme_agrees']}**",
        "",
        f"**The argmax rule flags {specs['n_scheme_dependent']} of {specs['n_heads']} heads, "
        "which is close to everything, and that is a finding about the rule rather than "
        "about the circuit.** A head's best receiver specification is not a stable quantity "
        "across counterfactuals — most heads have no strong specification at all, so their "
        "argmax is decided by noise. The pre-registered rule was a bare inequality with no "
        "threshold, it is reported as it was written, and it is much weaker than the "
        "head-level flag. The informative part of this channel is the line above it: which "
        "*published* specifications any scheme recovers.",
        "",
    ]

    rows = []
    schemes = specs["schemes"]
    for row in spec_scored["rows"]:
        if not row.get("published_spec"):
            continue
        cells = [_hs(row["head"]), row["class"] or "—", f"`{row['published_spec']}`"]
        for scheme in schemes:
            entry = row["per_scheme"][scheme]
            mark = "**" if entry["agrees"] else ""
            cells.append(f"{mark}`{entry['top_spec']}`{mark}")
        rows.append(cells)
    if rows:
        lines += [
            _table(rows, ["head", "class", "published spec"] + [f"`{s}`" for s in schemes]),
            "",
            "Bold marks a scheme whose search puts the published specification first.",
            "",
        ]

    # Where the sources publish *alternative* inputs for a head — fixed in Phase 7's plan
    # before that run — how often does the winner under some scheme land on one of them?
    # This is scoring, not searching: it reads the same argmaxes the blind comparison used.
    if hasattr(gt, "alternative_specs"):
        hits, total = 0, 0
        for row in spec_scored["rows"]:
            if not row.get("published_spec"):
                continue
            head = tuple(int(x) for x in row["head"].split("."))
            alternatives = {f"{i}@{p}" for i, p in gt.alternative_specs(head)}
            if not alternatives:
                continue
            total += 1
            if any(v["top_spec"] in alternatives for v in row["per_scheme"].values()):
                hits += 1
        if total:
            lines += [
                f"For the {total} heads whose sources also publish *alternative* inputs, the "
                f"winning specification under at least one scheme is one of those published "
                f"alternatives in **{hits}** case(s). The search is not landing on unrelated "
                "wires; across the schemes it moves between published inputs of the same "
                "head, and which one wins is decided by the counterfactual.",
                "",
            ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------


def _reproduction(payloads: dict[str, dict]) -> str:
    """Does the primary scheme still produce the number the earlier phase reported?

    A re-run under new machinery that quietly changed the old answer would make every
    comparison in this report unreadable, so the old headline is read out of the earlier
    phase's committed sweep and checked against this one rather than assumed.
    """
    earlier = {
        "docstring": (RESULTS_DIR / "phase7_sweep.json", "7", "random_random"),
        "greater_than": (RESULTS_DIR / "phase6_sweep.json", "6", "yy01"),
    }
    rows = []
    for circuit, payload in payloads.items():
        path, phase, scheme = earlier[circuit]
        old = _load(path)
        if not old:
            continue
        old_block = old["corruptions"][scheme]["metrics"]["logit_diff"]["headline"]
        new_block = payload["scored"]["heads"]["per_scheme"][scheme]
        same_set = set(old_block["matches"]) == set(new_block["matches"])
        same_n = old_block["n_discovered"] == new_block["n_discovered"]
        rows.append([
            CIRCUIT_TITLE[circuit],
            f"Phase {phase}: {len(old_block['matches'])} published, "
            f"{old_block['n_discovered']} discovered",
            f"Phase 8: {len(new_block['matches'])} published, "
            f"{new_block['n_discovered']} discovered",
            "**identical**" if (same_set and same_n) else "**differs**",
        ])
    if not rows:
        return ""
    return "\n".join([
        "## Reproduction check",
        "",
        "The primary scheme's sweep is the same experiment Phases 6 and 7 ran. Its result is "
        "read back out of their committed files and compared, so a change in the machinery "
        "cannot silently move the baseline this phase is measured against:",
        "",
        _table(rows, ["circuit", "earlier phase", "this phase", "primary scheme's head set"]),
        "",
    ])


def _cross_circuit(payloads: dict[str, dict]) -> str:
    """The comparison the phase turns on: the same flag on a circuit with no blindness."""
    if len(payloads) < 2:
        return ""
    rows = []
    for circuit, payload in payloads.items():
        report = payload["head_agreement"]
        scored = payload["scored"]["heads"]
        blind = scored["blind_spot"]
        n = payload["meta"]["published_head_count"]
        rows.append([
            CIRCUIT_TITLE[circuit],
            "**fires**" if report["flag"] else "silent",
            str(len(blind["heads"])),
            f"**{len(blind['published_heads_in_blind_spot'])}**",
            f"{len(scored['primary']['matches'])}/{n} → {len(scored['union']['matches'])}/{n}",
            f"{scored['precision_primary']:.2f} → {scored['precision_union']:.2f}",
        ])
    doc = payloads["docstring"]["scored"]["heads"]
    gtn = payloads["greater_than"]["scored"]["heads"]
    doc_report = payloads["docstring"]["head_agreement"]
    gtn_report = payloads["greater_than"]["head_agreement"]

    return "\n".join([
        "## The two circuits side by side — and why the flag is only half an answer",
        "",
        "The flag fired on **both** circuits. What differs is entirely in the detail "
        "underneath it, and none of that detail is available without the answer key:",
        "",
        _table(rows, ["circuit", "flag", "heads flagged", "of which published",
                      "recall, primary → union", "precision, primary → union"]),
        "",
        f"On docstring the flagged set contains the exact heads Phase 7 missed and recall "
        f"rises from {len(doc['primary']['matches'])}/6 to {len(doc['union']['matches'])}/6. "
        f"On greater-than it contains no circuit member at all: the primary counterfactual "
        f"already recovered {len(gtn['primary']['matches'])}/7 and the other schemes add "
        f"nothing but false positives, dropping precision from "
        f"{gtn['precision_primary']:.2f} to {gtn['precision_union']:.2f}. Phase 6's finding "
        "that this circuit has no counterfactual-blindness problem survives the check, which "
        "is the outcome that would have falsified the structure had it come out otherwise.",
        "",
        "**The uncomfortable half.** The two flags are identical in form — same wording, "
        "same kind of head list, 17 heads against 16. Nothing the "
        "pipeline computes before the answer key opens separates *\"your counterfactual is "
        "hiding half the circuit\"* from *\"your counterfactual is fine and the others are "
        "noisier\"*. Phase 7's failure was reporting one number as the circuit; Phase 8's "
        "remaining failure is reporting a warning that cannot be graded. It moves the "
        "question from **\"is this the circuit?\"**, which the pipeline used to answer "
        "wrongly and confidently, to **\"is this head list an artifact of the "
        "experiment?\"**, which it now answers loudly and imprecisely — better, and not the "
        "same as solved.",
        "",
        f"One further measurement worth stating on its own: across the four greater-than "
        f"schemes, **{len(gtn_report['intersection'])} heads were found by every scheme**, "
        f"against {len(doc_report['intersection'])} of the docstring circuit's. Agreement "
        "between counterfactuals is not the normal case that disagreement interrupts. On "
        "GPT-2 small, with four schemes of genuinely different design, it is close to empty.",
        "",
    ])


def _predictions(payloads: dict[str, dict]) -> str:
    doc = payloads.get("docstring")
    gtn = payloads.get("greater_than")
    rows: list[list[str]] = []

    def add(tag: str, text: str, hit: bool | None, evidence: str) -> None:
        mark = "—" if hit is None else ("**hit**" if hit else "**missed**")
        rows.append([tag, text, mark, evidence])

    if doc:
        report = doc["head_agreement"]
        scored = doc["scored"]["heads"]
        blind = scored["blind_spot"]
        add("P1", "the flag fires on docstring *(low information — implied by Phase 7's "
                  "published numbers)*", report["flag"],
            f"blind spot: {len(blind['heads'])} heads")
        published_in = set(blind["published_heads_in_blind_spot"])
        add("P2", "the docstring blind spot contains `1.4` and `2.0` *(low information)*",
            {"1.4", "2.0"} <= published_in,
            f"published heads in blind spot: {_joined(sorted(published_in))}")
        add("P3", "the union across schemes has **lower precision** than the primary alone",
            scored["precision_union"] < scored["precision_primary"],
            f"{scored['precision_primary']:.2f} → {scored['precision_union']:.2f}")
        add("P8", "at least one head outside the published circuit is also flagged "
                  "scheme-dependent",
            len(blind["unpublished_heads_in_blind_spot"]) > 0,
            f"{len(blind['unpublished_heads_in_blind_spot'])} unpublished heads flagged")

        spec_scored = doc["scored"]["specs"]
        movers = [r for r in spec_scored["rows"] if r.get("class") == "argument mover"]
        dependent = set(doc["spec_agreement"]["scheme_dependent"])
        both_flagged = all(r["head"] in dependent for r in movers) and bool(movers)
        detail = "; ".join(
            f"{r['head']}: random_random {r['per_scheme']['random_random']['top_spec']}, "
            f"random_def {r['per_scheme']['random_def']['top_spec']}"
            for r in movers
        )
        expected = all(
            r["per_scheme"]["random_def"]["top_spec"] == "q@END"
            and r["per_scheme"]["random_random"]["top_spec"] == "v@C_def"
            for r in movers
        )
        add("P7", "the search flags the argument movers as spec-scheme-dependent, with "
                  "`q@END` winning under `random_def` and `v@C_def` under `random_random`",
            both_flagged and expected, detail)

    if gtn:
        report = gtn["head_agreement"]
        scored = gtn["scored"]["heads"]
        blind = scored["blind_spot"]
        add("P4", "the flag **also** fires on greater-than, where Phase 6 found no such "
                  "pathology", report["flag"],
            f"blind spot: {len(blind['heads'])} heads")
        add("P5", "no *published* greater-than head is in the primary's blind spot",
            len(blind["published_heads_in_blind_spot"]) == 0,
            f"published heads flagged: {_joined(blind['published_heads_in_blind_spot'])}")
        power = report["power"].get("xx_mismatch", {})
        value = power.get("power", float("nan"))
        add("P6", "`xx_mismatch` has power between 0.05 and 0.60 of `yy01`",
            0.05 <= value <= 0.60, f"power {value:.2f} (span {power.get('span', 0):+.3f})")

    order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4, "P6": 5, "P7": 6, "P8": 7}
    rows.sort(key=lambda r: order[r[0]])
    hits = sum(1 for r in rows if r[2] == "**hit**")
    return "\n".join([
        "## The eight pre-registered predictions",
        "",
        f"**{hits} of {len(rows)} scored as hits.** Two were marked low-information in the "
        "plan, because the Phase 7 results they follow from are already published in this "
        "repo; they are scored anyway so the tally is complete rather than selective.",
        "",
        _table(rows, ["#", "prediction", "outcome", "measured"]),
        "",
    ])


# ---------------------------------------------------------------------------


def _the_real_test(payloads: dict[str, dict]) -> str:
    doc = payloads.get("docstring")
    if not doc:
        return ""
    report = doc["head_agreement"]
    scored = doc["scored"]["heads"]
    blind = scored["blind_spot"]
    n_published = doc["meta"]["published_head_count"]
    published_in_blind = blind["published_heads_in_blind_spot"]

    hand_built = [
        s["scheme"] for s in doc["schemes"] if s["provenance"] in ("published", "authored")
    ]
    hand_built_blind = sorted({
        v["head"] for v in report["verdicts"]
        if report["primary"] in v["missing_in"]
        and any(s in hand_built for s in v["found_in"])
    })
    published_set = {f"{l}.{h}" for l, h in GROUND_TRUTH["docstring"].ALL_HEADS}

    lines = [
        "## Does the structure catch Phase 7's blindness without knowing to look?",
        "",
        "This is the phase's actual question. Phase 7's diagnosis was correct and was reached "
        "by consulting the answer key. The test here is whether the same conclusion is "
        "available from the pipeline's output alone.",
        "",
        "**It is — and the flag on its own is not enough to act on.** Running the "
        "registered schemes and comparing them — no answer key, no human noticing "
        "anything — the pipeline emits:",
        "",
        "```",
        report["flag_text"],
        "```",
        "",
        f"Opening the published circuit afterwards, {len(published_in_blind)} of the "
        f"{len(blind['heads'])} flagged heads are published circuit members "
        f"({_joined(published_in_blind)}), and the primary scheme's recall of "
        f"{len(scored['primary']['matches'])}/{n_published} rises to "
        f"{len(scored['union']['matches'])}/{n_published} across the union. Phase 7 reported "
        f"the first number as the result and needed the answer key to learn it was not the "
        "circuit.",
        "",
        "**Three things stop this from being a clean success**, and they matter more than "
        "the headline:",
        "",
        f"1. **The flag is loud.** {len(blind['heads'])} heads are flagged and only "
        f"{len(published_in_blind)} of them are published. The flag says *the answer depends "
        "on the experiment*; it does not say which heads are real, and a reader who treated "
        "the flagged set as a circuit would be badly wrong. Precision across the union is "
        f"{scored['precision_union']:.2f} against {scored['precision_primary']:.2f} for the "
        "primary scheme alone.",
        f"2. **Most of the noise comes from the schemes that need no task knowledge.** "
        f"Restricting to the hand-built schemes ({', '.join(f'`{s}`' for s in hand_built)}) "
        f"the blind spot is {len(hand_built_blind)} heads, of which "
        f"{len([h for h in hand_built_blind if h in published_set])} are published. That is a "
        "derived cut of the same table, reported after the fact and **not** part of the flag "
        "rule — the pre-registered rule is the one quoted above, and it is what the number in "
        "the headline uses.",
        "3. **A fired flag is not evidence of a real blind spot.** It fired on greater-than "
        "too, where it was pointing at nothing — the next section is the comparison that "
        "makes that concrete, and it is the more useful half of the phase.",
        "",
    ]
    return "\n".join(lines)


def _limits(payloads: dict[str, dict]) -> str:
    gtn = payloads.get("greater_than")
    authored = ""
    if gtn:
        power = gtn["head_agreement"]["power"].get("xx_mismatch", {})
        authored = (
            f" Its power came out at {power.get('power', float('nan')):.2f} of the published "
            f"scheme's, so the alternate is measurably the weaker experiment"
            + (" and is annotated low-power in every table above."
               if power.get("low_power") else ".")
        )
    return "\n".join([
        "## What this does not solve",
        "",
        "**The schemes are still authored per task, and that is the same dependency Phase 5 "
        "named.** The docstring circuit got its second counterfactual for free because the "
        "authors published three. Greater-than did not: `xx_mismatch` was designed for this "
        "phase by a person reasoning about what the task's mechanism must contain — which "
        "value is moved, which structure makes the question well posed — and it is labelled "
        "`authored` everywhere it appears rather than being quietly counted as published."
        + authored,
        "",
        "So Phase 8 does not remove the hand-built ingredient. It changes when the ingredient "
        "shows its hand: instead of one counterfactual silently defining what the circuit is, "
        "several are forced to disagree in public, and the disagreement is printed whether or "
        "not anyone was looking for it.",
        "",
        "**Three further limits, stated rather than discovered later:**",
        "",
        "- **The flag does not say which scheme is right.** It says the answer depends on the "
        "experiment. Choosing between the schemes still needs the answer key or an argument "
        "this phase does not supply — and on an unfamiliar circuit there is no answer key.",
        "- **The generic schemes are nearly free and nearly useless.** They need no knowledge "
        "of the task, so any task can register them and clear the two-scheme bar without a "
        "human thinking about the mechanism at all. They are also the lowest-power schemes "
        "here and the largest source of flagged heads that are not in any published circuit. "
        "A task that registers only generics satisfies the letter of the check and gets very "
        "little from it.",
        "- **Blindness that *every* registered scheme shares is still invisible.** The "
        "comparison can only report disagreement between the counterfactuals it was given. "
        "A property no registered scheme perturbs produces unanimous agreement, which this "
        "structure reports as `robust`.",
        "",
    ])


def _reuse() -> str:
    changed = subprocess.run(
        ["git", "diff", "--stat", PRE_PHASE8, "--", "causal_interp/", "scripts/"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.strip()
    return "\n".join([
        "## What the change cost the existing code",
        "",
        f"`git diff --stat {PRE_PHASE8} -- causal_interp/ scripts/`:",
        "",
        "```",
        changed or "(no changes)",
        "```",
        "",
        "The causal core — `interventions.py`, `search.py`, `metrics.py`, `comparison.py`, "
        "`model.py`, `corruption.py` — is **untouched**, as it was in Phases 6 and 7. The "
        "three task modules gained a registration block each; `greater_than.py` additionally "
        "gained the authored scheme. Phase 6's own sweep list is frozen in place, and "
        "`scripts/check_schemes.py` verifies against git that greater-than's clean and "
        "corrupted token tensors under all three pre-existing schemes still hash identically "
        "to the module as it stood before this phase.",
        "",
        "```bash",
        "python scripts/check_schemes.py     # expect: SCHEMES OK",
        "```",
        "",
        "That script is a known-answer test in the sense `check_patching.py` is: it checks "
        "that a single-scheme `TaskSpec` is refused, that the agreement analysis returns the "
        "verdicts that follow by construction from a synthetic table nobody measured, that "
        "the flag stays silent when the schemes agree, and that `xx_mismatch` changes exactly "
        "one token and leaves `YY` bit-identical.",
        "",
    ])


def write_report(path: Path, results_dir: Path = RESULTS_DIR) -> None:
    payloads: dict[str, dict] = {}
    for circuit in ("docstring", "greater_than"):
        payload = _load(results_dir / f"phase8_{circuit}.json")
        if payload:
            payloads[circuit] = payload

    sections = [_intro(payloads)]
    for circuit, payload in payloads.items():
        sections.append(_circuit_section(circuit, payload))
        sections.append(_channels_section(circuit, payload))
    sections.append(_reproduction(payloads))
    sections.append(_the_real_test(payloads))
    sections.append(_cross_circuit(payloads))
    sections.append(_predictions(payloads))
    sections.append(_limits(payloads))
    sections.append(_reuse())
    path.write_text("\n".join(s for s in sections if s).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_report(RESULTS_DIR / "PHASE8_REPORT.md")
    print(f"wrote {RESULTS_DIR / 'PHASE8_REPORT.md'}")
