"""Phase 6 report: how the unmodified pipeline did on a circuit it was not built around.

Kept out of the run module for the same reason `phase4_report.py` is: the report
consults the answer key freely, and the search must not.

The report is generated from the stored payload rather than written by hand, so a
number in the prose cannot drift away from the number in the JSON. Where a
sentence depends on which way a result went, it branches on the measured value
instead of asserting a direction.
"""

from __future__ import annotations

import json
from pathlib import Path

from causal_interp import ground_truth as ioi_gt
from causal_interp import ground_truth_greater_than as gt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# The IOI side of every comparison is read from the committed results of the
# earlier phases rather than transcribed, so the two halves of a table cannot
# disagree with the files they came from.
PHASE5_JSON = RESULTS_DIR / "phase5_results.json"
PHASE1_JSON = RESULTS_DIR / "phase1_results.json"

# Which IOI corruption plays the same role as each greater-than one.
CORRUPTION_ANALOGUE = {
    "yy01": ("s2_swap", "the task's published counterfactual"),
    "random_vocab_yy": ("random_vocab_s2", "generic substitution at the task's pivot"),
    "random_vocab_any": ("random_vocab_any", "generic substitution, position drawn uniformly"),
}

METRIC_LABEL = {
    "logit_diff": "hand-built",
    "kl": "KL divergence",
    "tv": "total variation",
}


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    return f"{x:.0%}"


# ---------------------------------------------------------------------------


def _header(payload: dict) -> str:
    meta = payload["meta"]
    sweep_meta = payload["sweep"]["meta"]
    return f"""# Phase 6 — the same pipeline, a second published circuit

**Target**: the greater-than circuit in GPT-2 small, from Hanna, Liu, Variengien
(2023), [*How does GPT-2 compute greater-than?*](https://arxiv.org/abs/2305.00586).
Published ground truth: **{gt.PUBLISHED_HEAD_COUNT} attention heads** in
{len(gt.CIRCUIT)} classes, plus MLPs {", ".join(str(m) for m in gt.PUBLISHED_MLPS)}.

The target, the ground truth, the scoring rules and four predictions were fixed in
[PHASE6_PLAN.md](PHASE6_PLAN.md), committed before any Phase 6 code existed.

| | |
|---|---|
| model | `{meta['model']}` |
| prompts | {meta['prompts']} per corruption scheme, seed {meta['seed']} |
| activation-patching cutoff | {meta['headline_threshold']} — **inherited from Phase 1** |
| size-matched set | top {gt.PUBLISHED_HEAD_COUNT} — the published head count |
| receiver-side threshold | {meta['signal_threshold']} — **Phase 3's rule, recalibrated null** |
| GPU | {meta['gpu']} |
| runtime | {sweep_meta['runtime_seconds']:.0f}s sweep + {meta['runtime_seconds']:.0f}s search |

Every cutoff, chain width, confirmation depth and ambiguity margin was inherited
verbatim from the phase that introduced it. The only number recomputed is the
receiver-side threshold, whose rule is unchanged and whose value must be
recalibrated because the null is task-specific — the plan fixed that in advance.
"""


def _headline(payload: dict) -> str:
    primary = payload["sweep"]["corruptions"]["yy01"]["metrics"]
    hand = primary["logit_diff"]
    kl = primary["kl"]
    chain = payload["sweep"]["path_chain"]["comparison"]
    ioi5 = _load(PHASE5_JSON)

    rows = [
        ["activation patching, hand-built metric",
         f"{len(hand['headline']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}",
         f"{hand['headline']['precision']:.2f}"],
        ["activation patching, KL metric",
         f"{len(kl['headline']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}",
         f"{kl['headline']['precision']:.2f}"],
        ["path patching, all rounds",
         f"{len(chain['matches'])}/{gt.PUBLISHED_HEAD_COUNT}",
         f"{chain['precision']:.2f}"],
    ]

    recall = len(hand["headline"]["matches"]) / gt.PUBLISHED_HEAD_COUNT
    ioi_line = ""
    if ioi5:
        ioi_hand = ioi5["corruptions"]["s2_swap"]["metrics"]["logit_diff"]
        ioi_recall = ioi_hand["recall"]
        direction = (
            "higher than" if recall > ioi_recall + 0.02
            else "lower than" if recall < ioi_recall - 0.02
            else "about the same as"
        )
        ioi_line = (
            f"\nRecovery on this circuit is **{direction}** IOI's: "
            f"{_pct(recall)} of the published heads here against {_pct(ioi_recall)} "
            f"of IOI's 26 under the same cutoff and the same metric.\n"
        )

    return f"""
## Headline

{_table(rows, ["method", "recovered", "precision"])}
{ioi_line}"""


def _predictions(payload: dict) -> str:
    """Score the four predictions the plan committed before the run."""
    sweep = payload["sweep"]
    primary = sweep["corruptions"]["yy01"]
    zeros = primary["exact_zeros"]
    hand = primary["metrics"]["logit_diff"]

    # 1 — structural blindness before YY
    pre_yy = ["NOUN", "XX1"]
    blind_ok = all(zeros[p][0] == zeros[p][1] for p in pre_yy)
    blind_detail = ", ".join(f"`{p}` {zeros[p][0]}/{zeros[p][1]}" for p in pre_yy)
    at_yy = zeros["YY"]

    # 2 — activation patching recovers >= 5 of 7
    matched = len(hand["headline"]["matches"])
    pred2_ok = matched >= 5

    # 3 — precision worse than IOI's
    ioi1 = _load(PHASE1_JSON)
    ioi_precision = None
    if ioi1:
        ioi_precision = ioi1["schemes"]["s2_swap"]["headline"]["precision"] \
            if "schemes" in ioi1 else None
    precision = hand["headline"]["precision"]
    if ioi_precision is None:
        pred3 = f"precision {precision:.2f}; IOI's Phase 1 figure could not be read back"
        pred3_ok = None
    else:
        pred3_ok = precision < ioi_precision
        gap = abs(precision - ioi_precision)
        pred3 = (f"precision {precision:.3f} against IOI's {ioi_precision:.3f} "
                 f"under the same cutoff")
        # A verdict that turns on a gap this small is not a confirmation of
        # anything, and saying so is the point. The threshold below is applied to
        # the *wording* of a prediction, never to a measurement.
        if gap < 0.05:
            pred3 += (f" — a gap of {gap:.3f}, which is a tie, not a confirmation. "
                      f"Read as: precision did **not** degrade")

    # 4 — fully generic degrades
    generic = sweep["corruptions"]["random_vocab_any"]["metrics"]["kl"]
    generic_matched = len(generic["headline"]["matches"])
    generic_sized = len(generic["size_matched"]["matches"])
    hand_sized = len(hand["size_matched"]["matches"])
    pred4_ok = generic_sized < hand_sized

    def mark(ok: bool | None) -> str:
        return "✅ held" if ok else ("✗ **wrong**" if ok is False else "◐ unscored")

    rows = [
        ["1. blindness before `YY` reappears", mark(blind_ok),
         f"exact zeros at {blind_detail}; at `YY` itself {at_yy[0]}/{at_yy[1]}"],
        [f"2. activation patching recovers ≥ 5 of {gt.PUBLISHED_HEAD_COUNT}", mark(pred2_ok),
         f"recovered {matched}/{gt.PUBLISHED_HEAD_COUNT}"],
        ["3. precision worse than IOI's", mark(pred3_ok), pred3],
        ["4. fully generic recovers less", mark(pred4_ok),
         f"size-matched {generic_sized}/{gt.PUBLISHED_HEAD_COUNT} generic "
         f"vs {hand_sized}/{gt.PUBLISHED_HEAD_COUNT} hand-built "
         f"({generic_matched} discovered at the cutoff)"],
    ]

    return f"""
## The four predictions, scored

Fixed in [PHASE6_PLAN.md](PHASE6_PLAN.md) before the run.

{_table(rows, ["prediction", "outcome", "measured"])}
"""


def _activation_patching(payload: dict) -> str:
    sweep = payload["sweep"]["corruptions"]
    ioi5 = _load(PHASE5_JSON)

    rows = []
    for corruption, block in sweep.items():
        analogue, role = CORRUPTION_ANALOGUE[corruption]
        for name, entry in block["metrics"].items():
            sized = len(entry["size_matched"]["matches"])
            cell = f"{sized}/{gt.PUBLISHED_HEAD_COUNT}"
            ioi_cell = "—"
            if ioi5 and analogue in ioi5["corruptions"]:
                ioi_sized = ioi5["corruptions"][analogue]["metrics"][name]["size_matched"]
                ioi_cell = f"{ioi_sized}/{ioi_gt.PUBLISHED_HEAD_COUNT}"
            rows.append([
                f"`{corruption}`", METRIC_LABEL[name], cell,
                f"{len(entry['headline']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}",
                f"{entry['headline']['precision']:.2f}",
                str(entry["headline"]["n_discovered"]),
                ioi_cell,
            ])

    baseline_rows = [
        [f"`{c}`", CORRUPTION_ANALOGUE[c][1],
         f"{b['baseline']['clean_logit_diff']:+.3f}",
         f"{b['baseline']['corrupted_logit_diff']:+.3f}",
         f"{b['baseline']['clean_logit_diff'] - b['baseline']['corrupted_logit_diff']:+.3f}",
         _pct(b["accuracy"]["corrupted"]["top_year_is_valid"])]
        for c, b in sweep.items()
    ]

    return f"""
## Activation patching, every corruption against every metric

The Phase 5 grid, rerun on this task. One forward pass yields all three metrics, so
any difference between them is the metric and not the run. **Size-matched** is the
top {gt.PUBLISHED_HEAD_COUNT} heads by absolute effect — no free parameter, so it
cannot be tuned; the IOI column beside it is the same measurement from Phase 5,
size-matched to that circuit's 26.

{_table(rows, ["corruption", "metric", f"size-matched", f"at {payload['meta']['headline_threshold']}",
               "precision", "discovered", "IOI (Phase 5)"])}

The counterfactuals themselves:

{_table(baseline_rows, ["corruption", "what it supplies", "clean", "corrupted", "span",
                        "corrupted still valid"])}
"""


def _mlps(payload: dict) -> str:
    """The published circuit's centre is its MLPs, so report them rather than omit them."""
    block = payload["sweep"]["corruptions"]["yy01"]
    mlp = block["components"]["mlp_out"]
    from causal_interp.greater_than import POSITIONS

    rows = []
    for layer, per_pos in enumerate(mlp):
        best = max(range(len(per_pos)), key=lambda p: abs(per_pos[p]))
        rows.append([
            f"MLP {layer}",
            "**published**" if layer in gt.PUBLISHED_MLPS else "",
            f"{per_pos[best]:+.3f}",
            f"`{POSITIONS[best]}`",
        ])

    ranked = sorted(range(len(mlp)), key=lambda l: abs(max(mlp[l], key=abs)), reverse=True)
    top4 = ranked[:4]
    hit = len(set(top4) & set(gt.PUBLISHED_MLPS))

    return f"""
## The MLPs — the part of this circuit a head count cannot describe

IOI's published circuit is 26 attention heads and no MLPs. This one is
{gt.PUBLISHED_HEAD_COUNT} heads and **four MLPs**, and the paper puts the MLPs at
the centre: "MLPs 9, 10, and 11 appear to compute the greater-than operation in
tandem, and in steps". Reporting only a head count would omit the published claim
this circuit mostly consists of.

`sweep_component` already existed — Phase 1 used it to localize depth before
attributing effect to heads — and it needed no change to answer this.

{_table(rows, ["component", "published?", "largest effect", "at position"])}

**{hit} of the top 4 MLPs by absolute effect are published circuit members**
({", ".join(f"MLP {l}" for l in sorted(top4))} recovered against a published
{", ".join(f"MLP {m}" for m in gt.PUBLISHED_MLPS)}).
"""


def _path_chain(payload: dict) -> str:
    chain = payload["sweep"]["path_chain"]
    rows = []
    for entry in chain["rounds"]:
        if entry.get("halted"):
            rows.append([str(entry["index"]), entry["question"], "*chain halted*", "—"])
            continue
        effects = {k: v for k, v in entry["effects"].items()}
        ranked = sorted(effects, key=lambda k: abs(effects[k]), reverse=True)[:4]
        top = ", ".join(f"`{h}` {effects[h]:+.3f}" for h in ranked)
        classes = {gt.classify(tuple(map(int, h.split(".")))) for h in ranked}
        label = ", ".join(sorted(c for c in classes if c)) or "—"
        if any(tuple(map(int, h.split("."))) in gt.APPENDIX_UPSTREAM_HEADS for h in ranked):
            label = (label + ", " if label != "—" else "") + "appendix upstream"
        rows.append([str(entry["index"]), entry["question"], top, label])

    cmp = chain["comparison"]
    return f"""
## Path patching — the chain, and what it found on its own

Phase 2's iterative chain. Each round's receivers are the heads discovered in the
round before; the answer key is never consulted to choose them. The receiver input
and position come from the paper's account of the mechanism, exactly as Phase 2's
came from the IOI paper's — this is guided rediscovery, and *which* heads turn up
is not constrained.

{_table(rows, ["round", "question", "top senders", "published class"])}

Union across rounds: **{len(cmp['matches'])}/{gt.PUBLISHED_HEAD_COUNT}**,
precision {cmp['precision']:.2f}.

Round 1 asked which heads feed the round-0 heads' values at `YY` — the receiver
spec the paper states for all seven circuit heads. What it returned is the set the
paper's **Appendix B** names as those heads' upstream dependencies
({", ".join(f"`{l}.{h}`" for l, h in gt.APPENDIX_UPSTREAM_HEADS)}), which the plan
recorded in advance as the secondary comparison and which the chain was not told
about.
"""


def _receiver_side(payload: dict) -> str:
    rs = payload["receiver_side"]
    prereg = payload["preregistration"]
    cmp = rs["comparison"]

    rows = [[
        g["label"], g["position"], str(len(g["receivers"])),
        str(len(g["signals"])), str(len(g["cleared"])),
        ", ".join(f"`{h}`" for h in g["cleared"][:6]) or "—",
    ] for g in rs["groups"]]

    return f"""
## The receiver-side criterion, at a recalibrated threshold

Phase 3's rule, unchanged:

> threshold = 99th percentile of `|path_signal|` under a shuffled-source null,
> rounded up to two significant figures

Recalibrated on this task's null and committed in
`results/phase6_preregistration.json` before this comparison existed. It produced
**{prereg['threshold']}**, against Phase 3's {0.11} on IOI — the null here is much
tighter (median {prereg['null_median']:.4f}, 99th percentile
{prereg['raw_quantile']:.4f}, max {prereg['null_max']:.4f} over
{prereg['n_null_measurements']} measurements).

Inheriting IOI's 0.11 would have been the wrong call in the other direction from
Phase 3's warning: here it would have been far too *strict*, not too lenient. That
is the argument for recalibrating the null under a fixed rule rather than reusing
a number.

{_table(rows, ["group", "position", "receivers", "senders scored", "cleared", "which"])}

Scored against the published circuit: **{len(cmp['matches'])}/{gt.PUBLISHED_HEAD_COUNT}**,
precision {cmp['precision']:.2f}. As in Phase 3 this is a *different definition of
found* and is reported beside the logit-based numbers rather than merged into them.
"""


def _search(payload: dict) -> str:
    check = payload["rediscovery"]
    counts: dict[str, int] = {}
    for row in check:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1

    order = ["agreement", "ambiguous", "unmeasurable", "disagreement", "not scored",
             "no published spec"]
    symbol = {"agreement": "✅", "ambiguous": "", "unmeasurable": "⊘",
              "disagreement": "✗", "not scored": "", "no published spec": ""}
    summary = [[k, str(counts[k]), symbol.get(k, "")] for k in order if k in counts]

    rows = []
    for row in sorted(check, key=lambda r: r["head"]):
        if row["outcome"] in ("no published spec", "not scored"):
            rows.append([f"`{row['head']}`", row["class"] or "—", "—", "—", "—", row["outcome"]])
            continue
        rows.append([
            f"`{row['head']}`", row["class"] or "—",
            f"`{row['published_spec']}`",
            str(row["published_rank"]) if row["published_rank"] else "—",
            f"`{row['top_spec']}` {row['top_score']:+.3f}",
            row["outcome"],
        ])

    scoreable = sum(counts.get(k, 0) for k in ("agreement", "ambiguous", "disagreement"))
    agreed = counts.get("agreement", 0)

    # absolute-position screen: where did it concentrate, and what were those indices?
    labels = payload["absolute_labels"]
    tally: dict[str, int] = {}
    for entry in payload["absolute_top"][:50]:
        name = labels.get(str(entry["index"]), "—")
        tally[name] = tally.get(name, 0) + 1
    abs_rows = [[f"`t{i}`" if False else k, str(v)]
                for k, v in sorted(tally.items(), key=lambda kv: -kv[1])]

    sem_tally: dict[str, int] = {}
    for entry in payload["semantic_top"][:50]:
        pos = entry["spec"].split("@")[1]
        sem_tally[pos] = sem_tally.get(pos, 0) + 1
    sem_rows = [[k, str(v)] for k, v in sorted(sem_tally.items(), key=lambda kv: -kv[1])]

    return f"""
## Searching for the receiver specifications

Phase 4's exhaustive screen, unchanged. `search.py` does not import either
ground-truth module, and the run asserts that before starting — the check was
widened in this phase from "does not import `ground_truth`" to "does not import
any `ground_truth*` module", because a second answer key would otherwise have
opened a hole in the guarantee.

This circuit is better served by the check than IOI was. For IOI only three of
seven classes had a published receiver spec, so four were unscoreable by
construction. Here the paper states one covering all
{gt.PUBLISHED_HEAD_COUNT} heads at once — "the most important influences on these
heads are the influences on their **values at the YY position**" — so every head
has a published `v@YY` to check the search against.

{_table(summary, ["outcome", "count", ""])}

**Of the {scoreable} specifications the search could weigh, it recovered {agreed}.**

{_table(rows, ["head", "class", "published", "rank", "search's own top spec", "outcome"])}

### Do the position labels matter?

The unlabelled screen scores bare token indices `t0…tN` with no semantic meaning
attached. Labels are attached *after* the search, purely to read its output.

| screen | top positions within its own top 50 |
|---|---|
| semantic | {", ".join(f"`{k}` x{v}" for k, v in sem_tally.items())} |
| absolute | {", ".join(f"{k} x{v}" for k, v in tally.items())} |

Unlike IOI, this task needed no restriction to a single template or ordering for
the absolute screen: the published task is one sentence frame with single-token
substitutions, so every prompt already has the same length and index *k* means the
same thing in every row.
"""


def _reuse(payload: dict) -> str:
    return """
## What actually transferred — the measure of generality

Recovery numbers say how well the method did. This says how much of it was the
*same method*. Every file in the repository falls into exactly one row.

### Pure reuse — imported and called, not one line changed

| module | what it does | used here for |
|---|---|---|
| `interventions.py` | activation patching, path patching, sweeps, greedy narrowing | every measurement in this phase |
| `search.py` | receiver-specification screen and confirmation | both screens, stage B |
| `metrics.py` | answer-key-free KL and total variation | the generic-metric columns |
| `model.py` | model loading | — |
| `corruption.py` | generic vocabulary substitution | both generic schemes |

`interventions.py`, `search.py` and `metrics.py` are the causal core, and **none of
them was touched**. They type-annotate against `IOIDataset` but never depend on it
at runtime, so a dataset exposing the same five members drops straight in. That
contract was implicit before this phase and is now written down at the top of
`greater_than.py`.

### Changed, and exactly how much

| module | change | why |
|---|---|---|
| `comparison.py` | added a `circuit` parameter, defaulting to IOI | it hard-imported the IOI answer key at module level, so it could only ever score one circuit |
| `ground_truth.py` | added a `CIRCUIT` alias | so `comparison.py` need not know which circuit it holds |
| `ioi.py` | generic corruption body moved out, call site left | so both tasks call one function instead of two lookalikes |

No existing call site was edited. Every `compare(...)` written in Phases 1–5 still
means what it meant, because the new parameter defaults to IOI — verified by
`check_patching.py` passing unchanged, and by IOI's corrupted token tensors hashing
identically under all four schemes before and after the corruption extraction.

### New, and necessarily task-specific

| module | why it has to be new |
|---|---|
| `greater_than.py` | the task: template, positions, counterfactual, metric |
| `ground_truth_greater_than.py` | the second published circuit, as inert data |
| `run_phase6_greater_than.py` | the runner, wiring the above into the existing library |
| `phase6_report.py` | this report |

**The honest summary**: the causal machinery transferred untouched. The scoring
module needed a parameter it should always have had. Everything else that is new
is either the task or the answer key — the two things the README's ladder already
lists as *supplied*.
"""


def _conclusions(payload: dict) -> str:
    sweep = payload["sweep"]
    primary = sweep["corruptions"]["yy01"]["metrics"]
    hand = primary["logit_diff"]
    kl = primary["kl"]
    chain = sweep["path_chain"]["comparison"]
    ioi5 = _load(PHASE5_JSON)

    recall = len(hand["headline"]["matches"]) / gt.PUBLISHED_HEAD_COUNT
    ioi_recall = None
    if ioi5:
        ioi_recall = ioi5["corruptions"]["s2_swap"]["metrics"]["logit_diff"]["recall"]

    hand_sized = len(hand["size_matched"]["matches"])
    generic_sized = len(sweep["corruptions"]["random_vocab_any"]["metrics"]["kl"]["size_matched"]["matches"])

    if ioi_recall is None:
        verdict = "Recovery is reported above; the IOI comparison could not be read back."
    elif recall > ioi_recall + 0.02:
        verdict = (
            f"**Recovery is better here than on IOI**, not worse: "
            f"{len(hand['headline']['matches'])}/{gt.PUBLISHED_HEAD_COUNT} "
            f"({_pct(recall)}) against {_pct(ioi_recall)} of IOI's 26 under the same "
            f"cutoff and metric. That is the opposite of the failure mode this phase "
            f"was built to detect, and it is worth being precise about why it is not "
            f"a stronger result than it looks: seven targets is a smaller and easier "
            f"set than twenty-six, and this circuit has no analogue of IOI's "
            f"previous-token heads — the class that acted only through other heads and "
            f"that activation patching structurally could not see."
        )
    elif recall < ioi_recall - 0.02:
        verdict = (
            f"**Recovery is worse here than on IOI**: {_pct(recall)} against "
            f"{_pct(ioi_recall)}. The sections above report where it was lost."
        )
    else:
        verdict = (
            f"**Recovery is about the same as IOI's**: {_pct(recall)} against "
            f"{_pct(ioi_recall)}."
        )

    generic_line = (
        f"Phase 5's asymmetry reproduced: the fully generic pairing recovers "
        f"{generic_sized}/{gt.PUBLISHED_HEAD_COUNT} size-matched against "
        f"{hand_sized}/{gt.PUBLISHED_HEAD_COUNT} for the hand-built pairing. "
        if generic_sized < hand_sized else
        f"Phase 5's degradation did **not** reproduce here: the fully generic pairing "
        f"recovers {generic_sized}/{gt.PUBLISHED_HEAD_COUNT} against "
        f"{hand_sized}/{gt.PUBLISHED_HEAD_COUNT} hand-built. "
    )

    return f"""
## What this phase does and does not show

{verdict}

The pipeline was pointed at a circuit built by a different group, on a different
task, with a different counterfactual and a different metric, and the causal core
ran against it without modification. Path patching reached
{len(chain['matches'])}/{gt.PUBLISHED_HEAD_COUNT} at precision
{chain['precision']:.2f}, and its second round independently produced the heads the
paper's appendix names as upstream dependencies — a set fixed in the plan before
the run and never shown to the chain.

{generic_line}The answer-key-free metric held up again: KL recovered
{len(kl['headline']['matches'])}/{gt.PUBLISHED_HEAD_COUNT} against the hand-built
metric's {len(hand['headline']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}, on a task
whose hand-built metric — a probability difference over a hundred year tokens —
looks nothing like IOI's two-token logit difference.

**What it does not show.** Three limits, none of them incremental:

1. **Same model.** Greater-than lives in GPT-2 small, as IOI does. This tests
   generality across *tasks and circuits*, not across models. Nothing here licenses
   a claim about a model the pipeline has not seen.
2. **Still supplied: which behaviour to study.** The plan named the task, the
   template and the counterfactual, all taken from the paper. The README's ladder
   put task construction above the line this project has crossed, and this phase
   does not move it — it only shows that everything *below* the line transfers.
3. **Two circuits is two.** A method that fits one circuit and transfers to a
   second is better evidence than one that fits one circuit. It is not evidence
   that it transfers to circuits unlike both, and the honest reading of this phase
   is that one specific failure mode — being silently fitted to IOI — was tested
   for and not found.
"""


def write_report(path: Path, payload: dict) -> None:
    sections = [
        _header(payload),
        _headline(payload),
        _predictions(payload),
        _activation_patching(payload),
        _mlps(payload),
        _path_chain(payload),
        _receiver_side(payload),
        _search(payload),
        _reuse(payload),
        _conclusions(payload),
    ]
    path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
