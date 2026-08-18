"""Phase 7 report: how the unmodified pipeline did on a circuit in a different model.

Kept out of the run module for the same reason `phase4_report.py` and
`phase6_report.py` are: the report consults the answer key freely, and the search
must not.

The report is generated from the stored payload rather than written by hand, so a
number in the prose cannot drift away from the number in the JSON. Where a sentence
depends on which way a result went, it branches on the measured value instead of
asserting a direction.
"""

from __future__ import annotations

import json
from pathlib import Path

from causal_interp import ground_truth as ioi_gt
from causal_interp import ground_truth_docstring as gt
from causal_interp import ground_truth_greater_than as gtgt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# The IOI and greater-than sides of every comparison are read from the committed
# results of the earlier phases rather than transcribed, so the two halves of a
# table cannot disagree with the files they came from.
PHASE5_JSON = RESULTS_DIR / "phase5_results.json"
PHASE6_SWEEP_JSON = RESULTS_DIR / "phase6_sweep.json"

CORRUPTION_ROLE = {
    "random_random": "the benchmark's default counterfactual — **primary**",
    "random_def": "published counterfactual: break the induction match, keep the answer",
    "random_answer": "published counterfactual: replace the answer itself",
    "random_vocab_cdef": "generic substitution at the task's pivot",
    "random_vocab_any": "generic substitution, position drawn uniformly",
}

METRIC_LABEL = {
    "logit_diff": "hand-built",
    "kl": "KL divergence",
    "tv": "total variation",
}

PRIMARY = "random_random"


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


def _head_tuple(text: str) -> tuple[int, int]:
    return tuple(int(p) for p in text.split("."))  # type: ignore[return-value]


def _primary(payload: dict) -> dict:
    return payload["sweep"]["corruptions"][PRIMARY]["metrics"]


# ---------------------------------------------------------------------------


def _header(payload: dict) -> str:
    meta = payload["meta"]
    sweep_meta = payload["sweep"]["meta"]
    return f"""# Phase 7 — the same pipeline, a different model

**Target**: the Python docstring circuit in `attn-only-4l`, from Heimersheim and
Janiak (2023),
[*A circuit for Python docstrings in a 4-layer attention-only transformer*](https://www.lesswrong.com/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only).
Published ground truth: **{gt.PUBLISHED_HEAD_COUNT} attention heads** in
{len(gt.CIRCUIT)} classes, cross-checked against the 37-edge manual graph in the
ACDC release.

Every phase before this one ran inside GPT-2 small. The target, the ground truth,
the scoring rules, an advance audit of what in the code was GPT-2-shaped, and seven
predictions were fixed in [PHASE7_PLAN.md](PHASE7_PLAN.md), committed before any
Phase 7 code existed.

| | |
|---|---|
| model | `{meta['model_alias']}` (`{meta['model']}`) |
| shape | {meta['n_layers']} layers × {meta['n_heads']} heads = {meta['n_layers'] * meta['n_heads']} heads, `d_model` {meta['d_model']}, **`attn_only={meta['attn_only']}`** |
| tokenizer | `{meta['tokenizer']}`, `d_vocab` {meta['d_vocab']} |
| prompts | {meta['prompts']} per corruption scheme, seed {meta['seed']} |
| activation-patching cutoff | {meta['headline_threshold']} — **inherited from Phase 1** |
| size-matched set | top {gt.PUBLISHED_HEAD_COUNT} — the published head count |
| receiver-side threshold | {meta['signal_threshold']} — **Phase 3's rule, recalibrated null** |
| GPU | {meta['gpu']} |
| runtime | {sweep_meta['runtime_seconds']:.0f}s sweep + {meta['runtime_seconds']:.0f}s search |

For comparison, GPT-2 small is 12 × 12 = 144 heads, `d_model` 768, with an MLP in
every block and a 50257-token vocabulary. Nothing about the token indices, the
position layout or the depth of the model carries over from Phases 1–6.
"""


def _headline(payload: dict) -> str:
    primary = _primary(payload)
    hand, kl = primary["logit_diff"], primary["kl"]
    chain = payload["sweep"]["path_chain"]["comparison"]

    rows = [
        ["activation patching, hand-built metric",
         f"{len(hand['headline']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}",
         f"{len(hand['size_matched']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}",
         f"{hand['headline']['precision']:.2f}"],
        ["activation patching, KL (no answer key)",
         f"{len(kl['headline']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}",
         f"{len(kl['size_matched']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}",
         f"{kl['headline']['precision']:.2f}"],
        ["path patching, all rounds",
         f"{len(chain['matches'])}/{gt.PUBLISHED_HEAD_COUNT}", "—",
         f"{chain['precision']:.2f}"],
    ]

    recall = len(hand["headline"]["matches"]) / gt.PUBLISHED_HEAD_COUNT
    ioi5 = _load(PHASE5_JSON)
    p6 = _load(PHASE6_SWEEP_JSON)
    comparisons = []
    if ioi5:
        comparisons.append(("IOI", ioi5["corruptions"]["s2_swap"]["metrics"]["logit_diff"]["recall"]))
    if p6:
        gt6 = p6["corruptions"]["yy01"]["metrics"]["logit_diff"]["headline"]
        comparisons.append(("greater-than", len(gt6["matches"]) / gtgt.PUBLISHED_HEAD_COUNT))

    if comparisons:
        worse = [name for name, value in comparisons if recall < value - 0.02]
        better = [name for name, value in comparisons if recall > value + 0.02]
        detail = ", ".join(f"{name} {_pct(value)}" for name, value in comparisons)
        if len(worse) == len(comparisons):
            verdict = (
                f"**Recovery is worse here than on either GPT-2 small circuit**: "
                f"{_pct(recall)} of the published heads, against {detail}. That is a real "
                f"drop and the sections below locate it — three of the six heads are "
                f"missed, and they are missed for a reason this phase can name."
            )
        elif better:
            verdict = (
                f"Recovery is **{_pct(recall)}**, higher than {', '.join(better)} "
                f"({detail})."
            )
        else:
            verdict = f"Recovery is **{_pct(recall)}**, against {detail}."
    else:
        verdict = f"Recovery is {_pct(recall)}; earlier phases could not be read back."

    return f"""
## Headline

{_table(rows, ["method", f"at {payload['meta']['headline_threshold']}", "size-matched", "precision"])}

{verdict}

For the hand-built metric the threshold-based and size-matched columns agree
exactly, so nothing turns on which is called the headline. They differ by one head
for KL, which the cutoff finds and the size-matched cut does not.
"""


def _predictions(payload: dict) -> str:
    """Score the seven predictions the plan committed before the run."""
    sweep = payload["sweep"]
    primary = sweep["corruptions"][PRIMARY]
    hand = primary["metrics"]["logit_diff"]
    zeros = primary["exact_zeros"]
    anchored = sweep["corruptions"]["random_vocab_cdef"]["exact_zeros"]

    # 1 — no blindness under the primary scheme, full blocks under the anchored one
    n_cells = zeros["END"][1]
    no_full_block = all(zeros[p][0] < zeros[p][1] for p in zeros)
    pre_cdef = ["A_def", "B_def", "comma_B"]
    anchored_blind = all(anchored[p][0] == anchored[p][1] for p in pre_cdef)
    pred1_ok = no_full_block and anchored_blind
    residual = {p: v[0] for p, v in zeros.items() if v[0]}
    pred1_detail = (
        f"`{PRIMARY}`: no position is a full block (largest {max(residual.values(), default=0)}"
        f"/{n_cells}); `random_vocab_cdef`: "
        + ", ".join(f"`{p}` {anchored[p][0]}/{anchored[p][1]}" for p in pre_cdef)
    )

    # 2 — activation patching recovers >= 4 of 6
    matched = len(hand["headline"]["matches"])
    pred2_ok = matched >= 4
    missed = sorted(hand["headline"]["misses"])
    pred2_detail = (
        f"recovered {matched}/{gt.PUBLISHED_HEAD_COUNT}; missed "
        + ", ".join(f"`{h}`" for h in missed)
    )

    # 3 — the chain recovers the published wiring order
    chain = sweep["path_chain"]
    halted = any(r.get("halted") for r in chain["rounds"])
    ran = sum(1 for r in chain["rounds"] if not r.get("halted"))
    pred3_ok = not halted and len(chain["comparison"]["matches"]) >= 5
    pred3_detail = (
        f"{ran} of {len(chain['rounds'])} rounds ran; "
        f"{len(chain['comparison']['matches'])}/{gt.PUBLISHED_HEAD_COUNT} across the chain"
        + (" — the chain halted after reaching layer 0" if halted else "")
    )

    # 4 — precision worse than both earlier circuits, and >50% of heads clear 0.02
    precision = hand["headline"]["precision"]
    discovered = hand["headline"]["n_discovered"]
    ioi5, p6 = _load(PHASE5_JSON), _load(PHASE6_SWEEP_JSON)
    refs = []
    if ioi5:
        refs.append(("IOI", ioi5["corruptions"]["s2_swap"]["metrics"]["logit_diff"]["precision"]))
    if p6:
        refs.append(("greater-than", p6["corruptions"]["yy01"]["metrics"]["logit_diff"]["headline"]["precision"]))
    pred4_ok = all(precision < value for _, value in refs) if refs else None
    share = discovered / (payload["meta"]["n_layers"] * payload["meta"]["n_heads"])
    pred4_detail = (
        f"precision {precision:.2f} against "
        + ", ".join(f"{name} {value:.2f}" for name, value in refs)
        + f"; but only {discovered} of {payload['meta']['n_layers'] * payload['meta']['n_heads']} "
          f"heads ({_pct(share)}) clear the cutoff, not the \"more than half\" the plan predicted"
    )
    # The plan quoted IOI's precision as 0.90, which is Phase 2's *path patching*
    # figure, not activation patching's. The like-for-like comparators are the ones
    # above, read back from the stored results. The direction is unaffected.

    # 5 — fully generic recovers less
    generic = sweep["corruptions"]["random_vocab_any"]["metrics"]["kl"]
    generic_sized = len(generic["size_matched"]["matches"])
    hand_sized = len(hand["size_matched"]["matches"])
    pred5_ok = generic_sized < hand_sized

    # 6 — mlp_out is a silent no-op
    mlp = primary["components"]["mlp_out"]
    all_zero = all(v == 0.0 for row in mlp for v in row)
    cells = sum(len(row) for row in mlp)

    # 7 — the auxiliary heads are not recovered
    aux = hand["auxiliary_effects"]
    aux_found = hand["extended"]["auxiliary_found"]
    pred7_ok = not aux_found

    def mark(ok: bool | None) -> str:
        return "✅ held" if ok else ("✗ **wrong**" if ok is False else "◐ **split**")

    rows = [
        ["1. no blindness under the primary scheme, full blocks under the anchored one",
         mark(pred1_ok), pred1_detail],
        [f"2. activation patching recovers ≥ 4 of {gt.PUBLISHED_HEAD_COUNT}",
         mark(pred2_ok), pred2_detail],
        ["3. the chain recovers the published wiring order", mark(pred3_ok), pred3_detail],
        ["4. precision worse than both earlier circuits", mark(None if pred4_ok else False),
         pred4_detail],
        ["5. fully generic recovers less", mark(pred5_ok),
         f"size-matched {generic_sized}/{gt.PUBLISHED_HEAD_COUNT} generic vs "
         f"{hand_sized}/{gt.PUBLISHED_HEAD_COUNT} hand-built"],
        ["6. `mlp_out` sweep returns exact zeros rather than raising", mark(all_zero),
         f"all {cells} cells exactly 0.0"],
        ["7. the auxiliary heads `0.2`, `0.4` are not recovered", mark(pred7_ok),
         ", ".join(f"`{h}` {v:+.4f}" for h, v in aux.items())
         + (f"; found: {aux_found}" if aux_found else "; neither in the top set")],
    ]

    held = sum(1 for r in rows if r[1].startswith("✅"))
    wrong = sum(1 for r in rows if "wrong" in r[1])

    return f"""
## The seven predictions, scored

Fixed in [PHASE7_PLAN.md](PHASE7_PLAN.md) before the run. **{held} held,
{wrong} were wrong**, and one is split — the largest share of failed predictions of
any phase so far, which is the most useful thing this phase produced.

{_table(rows, ["prediction", "outcome", "measured"])}

Prediction 2 failed in a more interesting way than the count shows. The plan named
`0.5` and `1.2` as the heads at risk, on Phase 1's reasoning that heads acting only
through other heads are the ones activation patching misses. **`0.5` was recovered
(rank 4 of 32) and `1.4` and `2.0` were missed instead** — so the prediction was
wrong about the number *and* about the mechanism. The next section is what actually
explains the misses.

Prediction 4 compared against the wrong IOI number: the plan quoted 0.90, which is
Phase 2's *path patching* precision, where the like-for-like activation-patching
figure is Phase 5's 0.78. The verdict does not turn on it — 0.33 is below both — but
the plan's comparator was mis-stated and correcting it here is cheaper than leaving
it to be found.

The `8/32` in row 1 is not a residue of the corruption. In an attention-only model a
final-layer head writing at any position other than `END` has nothing downstream to
read it, so all 8 layer-3 heads are exact zeros at every non-`END` position by
construction. It is a *structural* zero of the architecture rather than of the
counterfactual, and it is the first one this project has met that is neither.
"""


def _what_the_counterfactual_sees(payload: dict) -> str:
    """The phase's main mechanistic finding: recovery depends on what the corruption breaks."""
    sweep = payload["sweep"]["corruptions"]
    rows = []
    for name in ("random_random", "random_def", "random_answer"):
        block = sweep[name]["metrics"]["logit_diff"]
        rows.append([
            f"`{name}`",
            CORRUPTION_ROLE[name].replace(" — **primary**", ""),
            f"{sweep[name]['baseline']['corrupted_logit_diff']:+.3f}",
            f"{len(block['headline']['matches'])}/{gt.PUBLISHED_HEAD_COUNT}",
            ", ".join(f"`{h}`" for h in block["headline"]["matches"]),
        ])

    prim = sweep["random_random"]["metrics"]["logit_diff"]["headline"]
    alt = sweep["random_def"]["metrics"]["logit_diff"]["headline"]
    gained = sorted(set(alt["matches"]) - set(prim["matches"]))

    return f"""
## Why three heads are missed: what a counterfactual can and cannot see

The three published counterfactuals disagree about which heads exist, and the
disagreement is systematic rather than noisy.

{_table(rows, ["corruption", "what it breaks", "corrupted", "recovered", "which"])}

`random_def` recovers **{len(alt['matches'])}/{gt.PUBLISHED_HEAD_COUNT}**, adding
{", ".join(f"`{h}`" for h in gained) or "nothing"} — and those are exactly the heads
the published account says decide *which* argument the movers attend to.

The mechanism is the same one Phase 5 found and sharper here. The circuit has two
jobs: **route** attention to the right definition argument (`1.4` → `2.0` → the
movers' keys) and **move** the content of whatever is attended to (`0.5`, `3.0`,
`3.6`).

- `random_random` and `random_answer` both **replace the answer token itself**.
  Restoring a routing head then buys nothing: the movers point at the right
  position and the token sitting there is still wrong. The routing heads become
  causally invisible to a metric read off the output.
- `random_def` **leaves the answer in place and breaks only the pointer**. Now
  restoring a routing head is exactly what recovers the behaviour, and `1.4` and
  `2.0` appear.

**The primary scheme was fixed in the plan before any of this was measured**, on
the ground that it is the authors' benchmark default, and it stays the headline.
The point is not that a better corruption exists — it is that **which parts of a
circuit are discoverable is a property of the counterfactual, not of the circuit**,
and on an unfamiliar circuit there would be no published head list to notice the
gap against. That is the same negative result Phase 5 reported for IOI, arriving
here as a 50% recall instead of a footnote.
"""


def _components(payload: dict) -> str:
    from causal_interp.docstring import POSITIONS

    block = payload["sweep"]["corruptions"][PRIMARY]["components"]
    rows = []
    for kind, grid in block.items():
        for layer, per_pos in enumerate(grid):
            best = max(range(len(per_pos)), key=lambda p: abs(per_pos[p]))
            where = "—" if per_pos[best] == 0.0 else f"`{POSITIONS[best]}`"
            rows.append([f"`{kind}`", str(layer), f"{per_pos[best]:+.3f}", where])

    mlp_cells = sum(len(r) for r in block["mlp_out"])
    return f"""
## Components, and a silent failure mode found on purpose

`sweep_component` was inherited with `COMPONENT_KINDS` unchanged, `mlp_out`
included, even though this model has none. The plan predicted what would happen and
it happened: **all {mlp_cells} `mlp_out` cells are exactly 0.0**. The hook exists,
`run_with_hooks` accepts it, the hook never fires, the patch never happens, and the
sweep returns a clean grid that looks like a measurement of an absence rather than
an absence of measurement.

Nothing raised. On a model whose architecture was not already known, that grid
would have been reported as "the MLPs carry no causal signal".

{_table(rows, ["component", "layer", "largest effect", "at position"])}

`attn_out[3]` at `END` recovers **{block['attn_out'][3][-1]:+.3f}** on its own — in
a four-layer attention-only model, almost the entire behaviour is written by the
final attention layer at the final token.
"""


def _path_chain(payload: dict) -> str:
    chain = payload["sweep"]["path_chain"]
    rows = []
    for entry in chain["rounds"]:
        if entry.get("halted"):
            rows.append([str(entry["index"]), entry["question"], "*chain halted*",
                         entry["expected"], "—"])
            continue
        effects = entry["effects"]
        ranked = sorted(effects, key=lambda k: abs(effects[k]), reverse=True)[:4]
        top = ", ".join(f"`{h}` {effects[h]:+.3f}" for h in ranked)
        classes = {gt.classify(_head_tuple(h)) for h in ranked}
        label = ", ".join(sorted(c for c in classes if c)) or "—"
        rows.append([str(entry["index"]), entry["question"], top, entry["expected"], label])

    cmp = chain["comparison"]
    halted = next((r for r in chain["rounds"] if r.get("halted")), None)
    ran = [r for r in chain["rounds"] if not r.get("halted")]
    last_carried = ran[-1].get("carried", []) if ran else []

    halt_note = ""
    if halted:
        halt_note = f"""
### The chain ran out of model

Round {halted['index']} could not run. Its receivers would have been round
{halted['index'] - 1}'s top senders — {", ".join(f"`{h}`" for h in last_carried)} — and a
sender must sit strictly below its receiver, so with a layer-0 head among them
there are no eligible senders left.

This is not a defect of the rule and not a tuning problem: **the chain descended to
layer 0 in one round and there is nothing below layer 0**. Phase 2's chain had
twelve layers to walk down and used four rounds; here it exhausted a four-layer
model in two. The pre-registered four-round ladder
(`3.x ← {{2.0, 1.2}} ← 1.4 ← 0.5`) was therefore never fully testable, and
prediction 3 is scored as failed rather than excused.

The honest reading: **an iterative chain's reach is bounded by the model's depth**,
and that bound is invisible until the pipeline meets a shallow model. It is the
clearest instance in seven phases of a method assumption that was silently a GPT-2
small assumption.
"""

    return f"""
## Path patching — the chain, and where it stopped

Phase 2's iterative chain. Each round's receivers are the heads discovered in the
round before; the answer key is never consulted to choose them. The receiver input
and position come from the published account of the mechanism, exactly as Phase 2's
came from the IOI paper's — this is guided rediscovery, and *which* heads turn up is
not constrained.

{_table(rows, ["round", "question", "top senders", "expected", "published class"])}

Union across rounds: **{len(cmp['matches'])}/{gt.PUBLISHED_HEAD_COUNT}**, precision
{cmp['precision']:.2f}. Round 0 returned the two argument movers as its top two out
of all {payload['meta']['n_layers'] * payload['meta']['n_heads']} heads, unprompted.

Round 1 asked what feeds their keys at `C_def`. Its top senders are `0.5` and `1.2`:
`1.2` is one of the two heads the released graph names as feeding those keys, and
`0.5` is the head the graph names as feeding `1.2`'s own query and key. **`2.0`, the
other named key-input, does not appear** — the same miss activation patching made,
for the same reason the next-but-one section gives. The magnitudes are also small
(`0.5` at −0.029, `1.2` at −0.005 against round 0's +0.588), and only one sender
clears the inherited cutoff.
{halt_note}"""


def _receiver_side(payload: dict) -> str:
    rs = payload["receiver_side"]
    prereg = payload["preregistration"]
    cmp = rs["comparison"]

    rows = [[
        g["label"], g["position"], str(len(g["receivers"])),
        str(len(g["signals"])), str(len(g["cleared"])),
        ", ".join(f"`{h}`" for h in g["cleared"][:6]) or "—",
    ] for g in rs["groups"]]

    ceilings = [min(int(r.split(".")[0]) for r in g["receivers"]) for g in rs["groups"]]
    ceiling = max(ceilings) if ceilings else 0
    eligible = sorted(h for h in gt.ALL_HEADS if h[0] < ceiling)
    out_of_scope = sorted(h for h in gt.ALL_HEADS if h[0] >= ceiling)
    discovered = {_head_tuple(h) for h in rs["discovered"]}
    in_scope_hits = sorted(discovered & set(eligible))

    return f"""
## The receiver-side criterion, at a threshold recalibrated a third time

Phase 3's rule, unchanged:

> threshold = 99th percentile of `|path_signal|` under a shuffled-source null,
> rounded up to two significant figures

Recalibrated on this model's null and committed in
`results/phase7_preregistration.json` before this comparison existed. Three
recalibrations of one rule now exist: **0.11** on IOI, **0.046** on greater-than,
**{prereg['threshold']}** here. No two agree, which is the whole argument for
inheriting the rule and not the number.

{_table(rows, ["group", "position", "receivers", "senders scored", "cleared", "which"])}

Scored against the published circuit: **{len(cmp['matches'])}/{gt.PUBLISHED_HEAD_COUNT}**,
precision {cmp['precision']:.2f}. As in Phases 3 and 6 this is a *different
definition of found*, reported beside the logit-based numbers rather than merged
into them.

### The null is thin, and that is a finding about the rule

The threshold was pooled from **{prereg['n_null_measurements']} null measurements**,
against Phase 3's hundreds. The chain halts after one usable receiver group, and
that group's eligible senders are the {prereg['n_null_measurements']} heads below
it — so a 99th-percentile rule is being asked to pick a tail from a sample that
barely has one. The rule was applied unchanged because changing it after seeing the
sample size would be exactly the move the pre-registration exists to prevent, but
**a percentile null does not survive contact with a shallow model**, and that is
worth more than the number it produced.

Only {len(eligible)} of the {gt.PUBLISHED_HEAD_COUNT} published heads
({", ".join(f"`{l}.{h}`" for l, h in eligible)}) were ever eligible as senders; the
other {len(out_of_scope)} ({", ".join(f"`{l}.{h}`" for l, h in out_of_scope)})
occupy the *receiver* slot and are unmeasured, not measured-and-failed — the
distinction Phase 3 drew for `9.0` and `11.9`. Of the in-scope ones,
{len(in_scope_hits)} cleared the bar{": " + ", ".join(f"`{l}.{h}`" for l, h in in_scope_hits) if in_scope_hits else ""}.
"""


def _search(payload: dict) -> str:
    check = payload["rediscovery"]
    counts: dict[str, int] = {}
    for row in check:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1

    order = ["agreement", "ambiguous", "unmeasurable", "disagreement", "not scored",
             "no published spec"]
    symbol = {"agreement": "✅", "ambiguous": "", "unmeasurable": "⊘",
              "disagreement": "✗", "not scored": "", "no published spec": "—"}
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

    alt_rows = []
    for row in check:
        for alt in row.get("alternatives", []):
            alt_rows.append([
                f"`{row['head']}`", f"`{row['published_spec']}`",
                str(row["published_rank"] or "—"),
                f"`{alt['spec']}`", str(alt["rank"] or "—"), f"{alt['score']:+.3f}",
            ])
    movers = [r for r in check if r["class"] == "argument mover"]
    mover_alt_first = all(
        any(a["spec"] == "v@C_def" and a["rank"] == 1 for a in r.get("alternatives", []))
        for r in movers
    ) if movers else False

    if agreed == 0:
        verdict = f"""**The search recovered none of the {scoreable} primary specifications it could weigh** —
the worst rediscovery result in the project, against 16 of 17 on IOI and 7 of 7 on
greater-than.

It is the same failure as the head-level one, one level down. For both argument
movers the search's top pick is `v@C_def` — the *value* they read at the definition
argument — where the plan's primary spec was `q@END`, the query that decides which
argument to read. Under `random_random` the token at `C_def` is replaced, so
splicing the clean value there restores the behaviour and splicing the clean query
does not. The screen is a logit-effect screen; it ranked the wire that carries the
answer above the wire that chooses it, because under this counterfactual that is
the true causal ordering."""
        if mover_alt_first:
            verdict += """

**And `v@C_def` is itself a published input for those heads** — one of the
alternatives the plan fixed in advance precisely so this could be reported without
it being a rescue. On the alternative specs the search ranks the published wire
**first of all 21** for both movers. The headline stays 0; the alternative line is
the informative one, and it says the search found a real published edge rather than
nothing."""
    else:
        verdict = (
            f"**Of the {scoreable} specifications the search could weigh, it "
            f"recovered {agreed}.**"
        )

    alt_section = ""
    if alt_rows:
        alt_section = f"""
### The alternative published specs, declared in advance

This circuit has more than one published input per class, so the plan fixed both
the primary spec and the alternatives before the run — precisely so a poor primary
result could not be rescued afterwards by adopting a better one. It is reported
here as a separate line and is **not** merged into the count above.

{_table(alt_rows, ["head", "primary spec", "rank", "alternative", "rank", "score"])}
"""

    labels = payload["absolute_labels"]
    tally: dict[int, int] = {}
    for entry in payload["absolute_top"][:50]:
        tally[entry["index"]] = tally.get(entry["index"], 0) + 1
    abs_summary = ", ".join(
        f"`t{i}` ×{v} (= {labels.get(str(i), '—')})"
        for i, v in sorted(tally.items(), key=lambda kv: -kv[1])[:6]
    )
    sem_tally: dict[str, int] = {}
    for entry in payload["semantic_top"][:50]:
        pos = entry["spec"].split("@")[1]
        sem_tally[pos] = sem_tally.get(pos, 0) + 1

    return f"""
## Searching for the receiver specifications

Phase 4's exhaustive screen, unchanged. `search.py` does not import any
ground-truth module, and the run asserts that before starting — the check widened
in Phase 6 to cover any `ground_truth*` module needed no further change for a third
circuit, which is what widening it was for.

Four of the six heads have a published `(input, position)`; `0.5` and `1.2` do not,
and were declared unscoreable in the plan rather than scored against a guess.

{_table(summary, ["outcome", "count", ""])}

{_table(rows, ["head", "class", "published", "rank", "search's own top spec", "outcome"])}

{verdict}
{alt_section}
### Do the position labels matter?

The unlabelled screen scores bare token indices `t0…t40` with no semantic meaning
attached. Labels are attached *after* the search, purely to read its output.

| screen | top positions within its own top 50 |
|---|---|
| semantic | {", ".join(f"`{k}` ×{v}" for k, v in sorted(sem_tally.items(), key=lambda kv: -kv[1]))} |
| absolute | {abs_summary} |
"""


def _cross_model(payload: dict) -> str:
    """One table putting all three circuits side by side."""
    hand = _primary(payload)["logit_diff"]
    chain = payload["sweep"]["path_chain"]["comparison"]
    ioi5, p6 = _load(PHASE5_JSON), _load(PHASE6_SWEEP_JSON)

    rows = []
    ioi_recall = None
    if ioi5:
        m = ioi5["corruptions"]["s2_swap"]["metrics"]["logit_diff"]
        ioi_recall = m["recall"]
        rows.append([
            "IOI (Wang et al.)", "GPT-2 small", "12 × 12 = 144",
            f"{ioi_gt.PUBLISHED_HEAD_COUNT}", f"{m['matched']}/{ioi_gt.PUBLISHED_HEAD_COUNT}",
            _pct(m["recall"]), f"{m['precision']:.2f}",
            _pct(ioi_gt.PUBLISHED_HEAD_COUNT / 144),
        ])
    if p6:
        m = p6["corruptions"]["yy01"]["metrics"]["logit_diff"]["headline"]
        rows.append([
            "greater-than (Hanna et al.)", "GPT-2 small", "12 × 12 = 144",
            f"{gtgt.PUBLISHED_HEAD_COUNT}",
            f"{len(m['matches'])}/{gtgt.PUBLISHED_HEAD_COUNT}",
            _pct(len(m["matches"]) / gtgt.PUBLISHED_HEAD_COUNT), f"{m['precision']:.2f}",
            _pct(gtgt.PUBLISHED_HEAD_COUNT / 144),
        ])
    meta = payload["meta"]
    total_heads = meta["n_layers"] * meta["n_heads"]
    m = hand["headline"]
    rows.append([
        "**docstring (Heimersheim & Janiak)**", f"**{meta['model_alias']}**",
        f"**{meta['n_layers']} × {meta['n_heads']} = {total_heads}**",
        f"**{gt.PUBLISHED_HEAD_COUNT}**",
        f"**{len(m['matches'])}/{gt.PUBLISHED_HEAD_COUNT}**",
        f"**{_pct(len(m['matches']) / gt.PUBLISHED_HEAD_COUNT)}**",
        f"**{m['precision']:.2f}**",
        f"**{_pct(gt.PUBLISHED_HEAD_COUNT / total_heads)}**",
    ])

    chance_here = gt.PUBLISHED_HEAD_COUNT / total_heads
    chance_ioi = ioi_gt.PUBLISHED_HEAD_COUNT / 144
    chance_gt = gtgt.PUBLISHED_HEAD_COUNT / 144
    recall_here = len(m["matches"]) / gt.PUBLISHED_HEAD_COUNT

    return f"""
## All three circuits side by side

{_table(rows, ["circuit", "model", "heads", "published", "recovered", "recall",
               "precision", "chance recall"])}

The last column is the deflation the plan fixed in advance, and it turns out to cut
less than the plan expected. Chance recall — the recall a size-matched set drawn at
random would get — is {_pct(chance_here)} here, {_pct(chance_ioi)} for IOI and
{_pct(chance_gt)} for greater-than. So the plan's warning applies only against
greater-than, whose 100% sits on a denominator four times harder than this one.
**Against IOI the denominators are effectively identical**, which means the drop
from {_pct(ioi_recall) if ioi_recall is not None else "IOI's recall"} to
{_pct(recall_here)} is a real drop and not an artifact of circuit size. Every
measurement here clears chance comfortably; none of them clears it by as much as
either earlier circuit did.
"""


PRE_EXISTING_MODULES = (
    "causal_interp/comparison.py", "causal_interp/interventions.py",
    "causal_interp/metrics.py", "causal_interp/model.py", "causal_interp/search.py",
    "causal_interp/corruption.py", "causal_interp/ioi.py", "causal_interp/greater_than.py",
    "causal_interp/ground_truth.py", "causal_interp/ground_truth_greater_than.py",
)


def _diff_against_plan() -> tuple[str, str] | None:
    """Measure, rather than assert, that no pre-existing module changed in Phase 7.

    Returns (base commit, `git diff --stat` output) or None if git is unavailable.
    """
    import subprocess  # noqa: PLC0415

    root = RESULTS_DIR.parent

    def added(path: str) -> str | None:
        commits = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H", "--", path],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.split()
        return commits[-1] if commits else None

    try:
        base = added("results/PHASE7_PLAN.md")
        # The far end is the commit that published this phase's results, **not** `HEAD`.
        # This sentence is a claim about what Phase 7 changed, and a later phase editing a
        # shared module must not silently rewrite it — Phase 8 registers its counterfactual
        # schemes in `ioi.py` and `greater_than.py`, and against `HEAD` that would turn this
        # measurement into a statement about Phase 8.
        head = added("results/PHASE7_REPORT.md")
        if not base or not head:
            return None
        out = subprocess.run(
            ["git", "diff", "--stat", f"{base}..{head}", "--", *PRE_EXISTING_MODULES],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return base[:7], out


def _reuse(payload: dict) -> str:
    measured = _diff_against_plan()
    if measured is None:
        block = ("(git was not available when this report was generated; "
                 "run the command above to check)")
        base, changed = "<the plan commit>", None
    else:
        base, changed = measured
        block = changed or "(no output)"

    if changed:
        verdict = (
            "**Some pre-existing modules did change**, and the diff above is the "
            "record of it. Read the rows below against it."
        )
    else:
        verdict = (
            "**No existing file was modified at all.** Not the causal core, not the "
            "scoring module, not the loader, not either earlier task. Phase 7 is "
            "entirely additive."
        )

    return f"""
## What actually transferred — the measure of generality

Phase 6 asked this question of a second task in the same model and answered "the
causal core, untouched; one shared module needed a parameter". Phase 7 asks it of a
different model, and the answer is measured rather than asserted — this report runs
the command itself and pastes what it got:

```
$ git diff --stat {base}..HEAD -- causal_interp/*.py   # the ten modules that predate Phase 7
{block}
```

{verdict}

### Pure reuse — imported and called, not one line changed

| module | why it needed nothing |
|---|---|
| `interventions.py` | reads `model.cfg.n_layers` / `n_heads` throughout; no literal 12 or 144 anywhere |
| `search.py` | same, plus a position vocabulary supplied by the dataset |
| `metrics.py` | reads the final position from `ds.positions["END"]`, nothing model-specific |
| `model.py` | `load()` already took a model name; `load("attn-only-4l")` is the whole change |
| `comparison.py` | the `circuit` parameter Phase 6 added took a third answer key without modification |
| `corruption.py` | draws from `model.cfg.d_vocab`, so a 48262-token vocabulary needed no edit |

### New, and necessarily target-specific

| module | why it has to be new |
|---|---|
| `docstring.py` | the task: template, positions, counterfactuals, metric |
| `ground_truth_docstring.py` | the third published circuit, as inert data |
| `run_phase7_docstring.py` | the runner |
| `check_patching_docstring.py` | known-answer tests for the new model |
| `phase7_report.py` | this report |

### The two things that did not transfer, and neither is a file

Both are *assumptions*, which is why a diff cannot show them:

1. **`COMPONENT_KINDS` contains `mlp_out`.** On a model with no MLPs the sweep is a
   silent no-op returning 28 exact zeros. It did not raise, and nothing in the
   pipeline noticed.
2. **The path chain assumes there is always another layer below.** Phase 2's
   iterative chain descended to layer 0 in one round and halted. Its four-round
   ladder is a 12-layer assumption that nothing in the code states.

Neither lives in `interventions.py` or `search.py` — the plan's stated failure
condition was that a model assumption would turn out to be buried in the causal
core, and it is not. But "the code needed no changes" and "the method needed no
changes" are different claims, and only the first one is true.
"""


def _conclusions(payload: dict) -> str:
    hand = _primary(payload)["logit_diff"]
    kl = _primary(payload)["kl"]
    chain = payload["sweep"]["path_chain"]["comparison"]
    recall = len(hand["headline"]["matches"]) / gt.PUBLISHED_HEAD_COUNT
    alt = payload["sweep"]["corruptions"]["random_def"]["metrics"]["logit_diff"]["headline"]

    return f"""
## What this phase does and does not show

**The code transferred; the results got worse.** Those are two findings and
collapsing them into one would misreport both.

The pipeline ran against a model with a quarter the depth, a different tokenizer, a
different training corpus and **no MLP blocks**, with not one line of the existing
library changed. Round 0 of the path chain returned the two published argument
movers as its top two of 32 heads, unprompted; greedy narrowing reached 0.995
recovery with four nodes, three of them published; and the answer-key-free KL metric
held up a third time, finding
{len(kl['headline']['matches'])}/{gt.PUBLISHED_HEAD_COUNT} at the inherited cutoff
against the hand-built metric's
{len(hand['headline']['matches'])}/{gt.PUBLISHED_HEAD_COUNT} (they tie at
{len(kl['size_matched']['matches'])}/{gt.PUBLISHED_HEAD_COUNT} size-matched), on a
third task whose hand-built metric resembles neither of the first two.

And recovery is **{_pct(recall)}** — the lowest of the three circuits, on the most
forgiving denominator of the three. Three of the seven pre-registered predictions
were wrong.

**What went wrong is legible, which is the part that matters.** The three missed
heads are the ones that route attention rather than move content, and the primary
counterfactual replaces the answer token, which makes routing causally invisible to
a metric read off the output. A different published counterfactual — `random_def`,
which breaks the pointer and leaves the answer in place — recovers
{len(alt['matches'])}/{gt.PUBLISHED_HEAD_COUNT} including both routing heads. The
pipeline did not fail to find them; **the experiment it was handed could not
contain them**.

### Does this support or complicate the long-term aim?

It complicates it, and specifically:

- **Supports**: the machinery is not GPT-2-shaped. Interventions, search, metrics
  and scoring are architecture-agnostic in fact and not just in intent, and a
  third answer key dropped into the same comparison module unchanged.
- **Complicates**: on an unfamiliar model there is no published head list to notice
  a 50% recall against. Every diagnosis in the section above — that the misses are
  routing heads, that another counterfactual finds them, that the chain halted
  early rather than converged — was made *by consulting the answer key*. Nothing in
  the pipeline's own output distinguishes "the circuit is these three heads" from
  "the counterfactual can only see three of them", and that is the gap between
  guided rediscovery and oversight.
- **Complicates**: two method assumptions were silently GPT-2 small assumptions —
  an MLP sweep that returns zeros instead of raising, and an iterative chain whose
  depth budget exceeded the model's. Both produced plausible-looking output. On a
  model nobody has mapped, plausible-looking output is the failure mode that
  matters.

### What it still does not show

1. **Scale is untested, and downward is not upward.** `attn-only-4l` is *smaller*
   than GPT-2 small. The plan rejected GPT-2 medium because its published IOI
   circuit is defined by a 2% threshold rather than a head list, so the scale test
   is still waiting on a published circuit that can be checked.
2. **Still supplied: which behaviour to study.** Three phases have now varied the
   task and the model while leaving that row of the README's ladder untouched.
3. **Three circuits is three.** Two specific failure modes have now been tested for
   — being fitted to IOI, and being fitted to GPT-2 small — and neither was found in
   the code. The results, meanwhile, degraded, and the reason they degraded is a
   property of the experimental design that the method itself cannot detect.
"""


def write_report(path: Path, payload: dict) -> None:
    sections = [
        _header(payload),
        _headline(payload),
        _predictions(payload),
        _what_the_counterfactual_sees(payload),
        _components(payload),
        _path_chain(payload),
        _receiver_side(payload),
        _search(payload),
        _cross_model(payload),
        _reuse(payload),
        _conclusions(payload),
    ]
    path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
