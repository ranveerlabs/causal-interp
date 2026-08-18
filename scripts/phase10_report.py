"""Phase 10 report: how much of task construction survives being mechanized.

    python scripts/phase10_report.py

Kept out of the run module for the reason every phase since 4 has kept it out: the
report consults the answer key freely, and the induction must not.

Generated from the stored payloads. The pass/partial/negative verdict is *computed*
from the pre-registered scoring table in `PHASE10_PLAN.md`, and so is every prediction
outcome, including the three post-hoc ones from `PHASE10_AMENDMENT.md`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results"

# The hand-built baseline this phase is measured against, read off Phase 6's report
# rather than recomputed: `yy01` + the published probability-difference metric.
BASELINE = {"size_matched": 6, "at_cutoff": 7, "discovered": 9, "precision": 0.78}

# Which induced slot corresponds to the published counterfactual's position. Answer-key
# knowledge, used only to score prediction P4.
YY_SLOT = {"frame_same": "resample_t8", "frame_own": "resample_t9"}

FIXTURES = ("frame_same", "frame_own")
MODES = ("plan", "shape")
METRIC = "logit_diff"


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _sm(payload: dict) -> int:
    return payload["scored_primary"][METRIC]["size_matched"]["recovered"]


def _cut(payload: dict) -> dict:
    return payload["scored_primary"][METRIC]["at_cutoff"]


def _intro(runs: dict) -> str:
    head = runs[("frame_same", "plan")]
    verdict, band = _verdict_for(_sm(head))
    return f"""# Phase 10 — inducing the task instead of writing it

**Target**: the greater-than circuit in GPT-2 small (Hanna, Liu, Variengien 2023), the
same seven published heads Phase 6 recovered — but reached from a task **induced from 32
lines a person typed**, not from `causal_interp/greater_than.py`.

The question, the algorithm, the scoring table and eight predictions were fixed in
[PHASE10_PLAN.md](PHASE10_PLAN.md), committed with the human input in
[`../fixtures/`](../fixtures/) and before any Phase 10 code existed. What the
pre-registered algorithm produces was then measured in
[PHASE10_CHARACTERIZATION.md](PHASE10_CHARACTERIZATION.md), committed before any repair
was designed, and the one repair — plus an explicit refusal to let it become the
headline — in [PHASE10_AMENDMENT.md](PHASE10_AMENDMENT.md).

| | |
|---|---|
| model | `gpt2-small` |
| human input | **64 lines**, two sentence frames, unfiltered, committed before any code |
| prompts | 128 generated per scheme, seed 0 |
| cutoff | 0.02 — **inherited from Phase 1** |
| size-matched set | top 7 — the published head count, **inherited from Phase 6** |
| metric | `clean_argmax_logprob` — no answer key |
| pre-existing modules changed | **none** |

## Headline

**{verdict}** — the pre-registered induction builds a task on which activation patching
recovers **{_sm(head)} of the 7** published heads, size-matched, against the hand-built
task's **{BASELINE['size_matched']} of 7**. Section 7 of the plan calls
{_sm(head)}/7 `{band}`.

The post-hoc repair reaches **{_sm(runs[('frame_same', 'shape')])}/7** size-matched and
**{_cut(runs[('frame_same', 'shape')])['recovered']}/7 at the inherited 0.02 cutoff**,
which is the whole published circuit. It is reported throughout as post-hoc and it does
not replace the line above.

**One of eight pre-registered predictions held.**
"""


def _verdict_for(recovered: int) -> tuple[str, str]:
    if recovered >= 6:
        return "Clean pass.", "clean pass"
    if recovered >= 4:
        return "Partial.", "partial"
    return "Negative, as the plan defines it.", "negative"


def _ladder(runs: dict, char: dict) -> str:
    same_plan = runs[("frame_same", "plan")]
    same_shape = runs[("frame_same", "shape")]
    rows = [
        ["which behaviour to study", "**human**", "the one-sentence hunch; not attempted"],
        ["where to cut the prompt", "**human**", "encoded in the examples, recovered from nothing"],
        ["example prompts", "**human**", "32 lines per frame, unfiltered"],
        ["prompt template", "mechanized",
         f"{len(char['fixtures']['frame_same']['structure']['frame_columns'])} frame columns "
         "found by constancy"],
        ["slot vocabularies", "mechanized", "the observed values are the vocabulary"],
        ["tokenizer filtering", "mechanized (partly)",
         f"round-trip filter rejects {same_shape['built']['generation']['round_trip_rejection_rate']:.0%} "
         "of candidates; it does **not** catch a same-length mis-split"],
        ["repeated-slot constraint", "mechanized (fragile)",
         "co-variation; unanimity-based, and 2 lines in 32 defeat it"],
        ["position vocabulary", "mechanized", "the varying columns, as bare indices"],
        ["counterfactual content", "mechanized", "resample from the slot's own observed values"],
        ["the `xx_mismatch` alternate", "mechanized",
         "falls out of the tie as `desync`, once the tie survives"],
        ["which scheme is primary", "mechanized (**and it chose badly**)",
         "argmax measured divergence; picked the century, not the year"],
        ["the metric", "mechanized", "`clean_argmax_logprob`, no answer key"],
        ["accuracy check", "mechanized", "agreement with the model's own clean prediction"],
    ]
    return f"""
## What got mechanized, and what did not

This is the phase's actual deliverable. `greater_than.py` is 457 lines of hand-built
task; the table says which of it survived being derived from example prompts.

{_table(rows, ["ingredient", "status", "how / what it cost"])}

The two `human` rows at the top are the floor the plan named in advance and did not
attack. Everything below them was mechanized to some degree, and two of the mechanized
rows failed in ways that are worth more than the rows that worked.
"""


def _preregistered(runs: dict, char: dict) -> str:
    plan = runs[("frame_same", "plan")]
    own = runs[("frame_own", "plan")]
    rows = []
    for fixture in FIXTURES:
        payload = runs[(fixture, "plan")]
        built = payload["built"]
        rows.append([
            f"`{fixture}`",
            f"{built['structure']['n_examples_kept']}/{built['structure']['n_examples_given']}",
            len(built["structure"]["slots"]),
            sum(1 for s in built["structure"]["slots"] if s["tied"]),
            f"`{built['primary']}`",
            f"{_sm(payload)}/7",
            f"{_cut(payload)['precision']:.2f}",
            f"{payload['task_validity']['top_year_exceeds_start']:.0%}",
        ])
    header = ["fixture", "lines kept", "slots", "tied", "primary chosen",
              "size-matched", "precision", "task actually greater-than"]

    scheme_rows = [
        [f"`{scheme}`", f"{block['size_matched']['recovered']}/7",
         f"{block['at_cutoff']['precision']:.2f}",
         "**primary**" if scheme == plan["built"]["primary"] else ""]
        for scheme, block in plan["scored_per_scheme"].items()
    ]

    return f"""
## The pre-registered run

{_table(rows, header)}

Both fixtures land on **{_sm(plan)}/7**, and the final column is why: the task the
induction built is not greater-than. With the two century columns treated as
independent slots, generation samples them independently, so a clean prompt reads

```
The pilgrimage lasted from the year 11245 to the year 14
The migration  lasted from the year 1527  to the year 11
```

— mismatched centuries, and in the first case a five-digit year. The model's top
prediction is a two-digit year on 100% of them and *exceeds the start year on
{plan['task_validity']['top_year_exceeds_start']:.0%}*. **The clean condition is a coin
flip**, which is the published task's own `xx_mismatch` counterfactual served as the
control.

### The selection rule is the second failure, and it is separable

Every proposed scheme, scored against the published circuit after the fact:

{_table(scheme_rows, ["scheme", "size-matched", "precision", ""])}

`resample_t8` — redrawing the **start year**, which is the position the published
`yy01` counterfactual acts on — recovers **5/7 at precision 0.50**. The rule picked
`resample_t7` instead, on a divergence three times larger. **The information needed to
choose well was in the candidate set; the answer-key-free rule for choosing did not find
it.** On `frame_own` the same rule picked `resample_END`, a third position again.

That is a cleaner result than the headline number suggests: the counterfactual *content*
mechanized, and the counterfactual *ranking* did not.
"""


def _repair(runs: dict) -> str:
    plan = runs[("frame_same", "plan")]
    rows = []
    for fixture in FIXTURES:
        for mode in MODES:
            payload = runs[(fixture, mode)]
            built = payload["built"]
            rows.append([
                f"`{fixture}`",
                "pre-registered" if mode == "plan" else "**post-hoc**",
                f"{built['structure']['n_examples_kept']}/32",
                sum(1 for s in built["structure"]["slots"] if s["tied"]),
                f"`{built['primary']}`",
                f"{_sm(payload)}/7",
                f"{_cut(payload)['recovered']}/7 of {_cut(payload)['discovered']}",
                f"{_cut(payload)['precision']:.2f}",
                f"{payload['task_validity']['top_year_exceeds_start']:.0%}",
            ])
    rows.append(["hand-built (Phase 6)", "—", "—", "—", "`yy01`",
                 f"{BASELINE['size_matched']}/7",
                 f"{BASELINE['at_cutoff']}/7 of {BASELINE['discovered']}",
                 f"{BASELINE['precision']:.2f}", "100%"])

    return f"""
## The repair — post-hoc, and it works

One change, fixed in the amendment before it ran: keep the largest group of examples
sharing a **column shape** rather than a token **length**. A strict generalization, with
no threshold in it.

{_table(rows, ["fixture", "induction", "kept", "tied slots", "primary", "size-matched",
               "at 0.02", "precision", "task valid"])}

Three things happen at once, and they are the same thing:

1. The two odd lines are dropped, so the century columns tie again.
2. The tie produces `desync_t7` and `desync_END` — **Phase 8's authored `xx_mismatch`,
   re-derived from 30 example sentences with nothing task-specific in the code.**
3. Tying the centuries means `resample_t7` now moves *both* years together, which
   preserves the greater-than relation and collapses its divergence from 0.755 to 0.084.
   The selection rule then picks `resample_t8` on its own — the published
   counterfactual's position — and the generated task becomes 100% valid.

**At the inherited 0.02 cutoff the induced task recovers all seven published heads**, in
14 discovered against the hand-built task's 9. Size-matched it recovers 5, one behind the
hand-built 6; the two it drops below rank 7 are `8.8` and `5.5`, and `8.8` is the head
Phase 6's size-matched comparison missed as well.

This is post-hoc. It carries no pre-registered weight, and the headline stays
{_sm(plan)}/7.
"""


def _ksweep(ksweep: dict, pairs: dict) -> str:
    rows = []
    for mode in MODES:
        entries = [r for r in ksweep["rows"] if r["induction"] == mode]
        rows.append([
            "pre-registered" if mode == "plan" else "**post-hoc**",
            *[f"{r['size_matched']}/7" if "failed" not in r else "fail" for r in entries],
        ])
    header = ["induction", *[f"k = {k}" for k in ksweep["meta"]["k_values"]]]

    pair_rows = []
    for row in pairs["rows"]:
        if row["induction"] != "plan":
            continue
        shape = next(
            (r for r in pairs["rows"]
             if r["induction"] == "shape" and r["pair"] == row["pair"]), None
        )
        pair_rows.append([
            f"{row['pair'][0]}, {row['pair'][1]}",
            row["tied"],
            f"`{row['primary']}`",
            f"{row['size_matched']}/7",
            f"{row['task_valid']:.0%}",
            "refuses to build" if shape and "failed" in shape else f"{shape['size_matched']}/7",
        ])

    plan_entries = {r["k"]: r for r in ksweep["rows"] if r["induction"] == "plan"}
    return f"""
## How many examples does the human have to write? — the curve runs backwards

{_table(rows, header)}

Prediction P7 said this would be flat above k = 8 with k = 2 strictly worse. It is
**inverted**: two example prompts recover **{plan_entries[2]['size_matched']}/7** and
thirty-two recover **{plan_entries[32]['size_matched']}/7**.

The mechanism is visible in the same table's build data. At k = 2 and k = 4 the examples
happen to be homogeneous, so the century tie holds and the rule selects `resample_t8`.
At k = 8 line 5 — `...the year 1509 to the year 15` — enters the sample, the tie
dissolves, the primary flips to `resample_t7`, and recovery collapses to
{plan_entries[8]['size_matched']}/7. **Every additional natural example is another chance
to poison a unanimity-based rule**, and the pre-registered induction has no way to say so.

The post-hoc filter removes the monotone decay but not all of it: 6/7 at k ≤ 8, 5/7 at
k ≥ 16.

### Does k = 2 depend on *which* two lines?

Seven contiguous pairs, chosen by position and not by result, all reported:

{_table(pair_rows, ["fixture lines", "tied", "primary", "size-matched", "task valid",
                    "under the repair"])}

**Five of seven pairs reach 5–6/7.** The two that do not are exactly the two containing a
tokenizer-odd line, and under the repaired filter those two **refuse to build at all** —
one example survives, and the induction raises rather than producing a task. That is the
better failure: Phase 8's standard was that the pipeline should say what it cannot see,
and here it does, at the cost of needing the human to supply more lines.

The caveat is real and not small: a k = 2 task generates at most 8 distinct prompts, so
these are 144-head sweeps over 8 examples. The effects are large — spans of +2.7 to +3.9
against the 32-line task's +2.6 — but the sample is tiny.
"""


def _cross_task(char: dict) -> str:
    doc = char["cross_task"]["docstring"]
    ioi_all = char["cross_task"]["ioi_all_templates"]
    ioi_one = char["cross_task"]["ioi_one_template"]
    tied = [s for s in doc["structure"]["slots"] if s["tied"]]
    return f"""
## Run E — the induction on the other two tasks

Weaker than everything above by construction, and labelled so: these prompts come out of
hand-built generators, so the induction is handed an already-aligned sample. The only
question is whether the structure it induces matches what those modules hand-code.

| case | examples kept | induced | verdict |
|---|---|---|---|
| IOI, all 8 templates | **{ioi_all['structure']['n_examples_kept']}/32** | {len(ioi_all['structure']['slots'])} slots | **fails** — one slot's values are `','` and four names |
| IOI, one template | {ioi_one['structure']['n_examples_kept']}/32 | {len(ioi_one['structure']['slots'])} slots | recovers the three name positions, the place and the object |
| docstring | {doc['structure']['n_examples_kept']}/32 | {len(doc['structure']['slots'])} slots, **{len(tied)} tied** | **the tie rule works on a task it was never pointed at** |

The docstring result is the strongest single piece of evidence here that the mechanism
generalizes. `{tied[0]['label']}`↔column {tied[0]['columns'][1]} and
`{tied[1]['label']}`↔column {tied[1]['columns'][1]} are the argument names that appear
once in the function signature and again in the docstring — the `A_def`/`A_doc` and
`B_def`/`B_doc` pairs `docstring.py` hand-codes as separate named positions. Found from
32 prompts, on a different model with a different tokenizer, with nothing task-specific
in the code.

The IOI failure is the clearest single obstacle to running any of this on prompts a
human did not curate: the eight published templates have different token lengths, the
filter keeps the plurality one, and **nothing in the induction detects that it was handed
more than one frame**. It reports a confident structure over 12 of 32 examples.
"""


def _predictions(runs: dict, char: dict, ksweep: dict) -> str:
    same_plan = runs[("frame_same", "plan")]
    same_shape = runs[("frame_same", "shape")]
    own_plan = runs[("frame_own", "plan")]
    cs = char["fixtures"]["frame_same"]
    co = char["fixtures"]["frame_own"]
    plan_k = {r["k"]: r for r in ksweep["rows"] if r["induction"] == "plan"}
    shape_slots = same_shape["built"]["structure"]["slots"]
    shape_props = [p["name"] for p in same_shape["built"]["proposals"]]

    checks = [
        ("P1", "the length filter drops ≥ 1 line from each fixture",
         len(cs["structure"]["dropped"]) >= 1 and len(co["structure"]["dropped"]) >= 1,
         f"dropped {len(cs['structure']['dropped'])} and {len(co['structure']['dropped'])} — "
         "the mis-split preserves row length, so the filter cannot see it"),
        ("P2", "3 slots on `frame_same`, centuries tied, positions a subset of the hand-built set",
         len(cs["structure"]["slots"]) == 3
         and any(s["tied"] for s in cs["structure"]["slots"]),
         f"{len(cs['structure']['slots'])} slots, none tied. The *positions* half held — "
         "`t2/t7/t8/END` map to `NOUN/XX1/YY/END`, a subset of the hand-built five"),
        ("P3", "a `desync` on the century slot is proposed — `xx_mismatch` re-derived",
         any(p["kind"] == "desync" for p in same_plan["built"]["proposals"]),
         "nothing was tied, so nothing was proposed. **Held under the post-hoc repair**"),
        ("P4", "the selected primary is the `YY` slot",
         same_plan["built"]["primary"] == YY_SLOT["frame_same"],
         f"chose `{same_plan['built']['primary']}` on `frame_same` and "
         f"`{own_plan['built']['primary']}` on `frame_own`. **Held under the repair**"),
        ("P5", "headline recovers ≥ 6/7 size-matched",
         _sm(same_plan) >= 6, f"{_sm(same_plan)}/7 — the plan's `negative` band"),
        ("P6", "precision at 0.02 is below the hand-built 0.78",
         _cut(same_plan)["precision"] < BASELINE["precision"],
         f"{_cut(same_plan)['precision']:.2f} against 0.78"),
        ("P7", "the k-curve is flat above k = 8 and k = 2 is strictly worse",
         plan_k[8]["size_matched"] == plan_k[16]["size_matched"] == plan_k[32]["size_matched"]
         and plan_k[2]["size_matched"] < plan_k[32]["size_matched"],
         f"inverted — k=2 gives {plan_k[2]['size_matched']}/7 and k=32 gives "
         f"{plan_k[32]['size_matched']}/7, monotonically decreasing in between"),
        ("P8", "`frame_own` recovers strictly fewer than `frame_same`",
         _sm(own_plan) < _sm(same_plan),
         f"{_sm(own_plan)}/7 against {_sm(same_plan)}/7 — identical, under both inductions"),
    ]
    posthoc = [
        ("A1", "the repair keeps 30, finds 3 slots with the centuries tied, proposes 5 schemes "
               "including `desync_t7` and `desync_END`",
         same_shape["built"]["structure"]["n_examples_kept"] == 30
         and len(shape_slots) == 3
         and any(s["tied"] for s in shape_slots)
         and {"desync_t7", "desync_END"} <= set(shape_props),
         f"kept 30, 3 slots, 1 tied, proposals {shape_props}"),
        ("A2", "the repaired rule picks a different primary from `resample_t7`",
         same_shape["built"]["primary"] != "resample_t7",
         f"picked `{same_shape['built']['primary']}` — `resample_t7`'s divergence fell "
         "from 0.755 to 0.084 once both centuries moved together"),
        ("A3", "the repaired run recovers strictly more published heads",
         _sm(same_shape) > _sm(same_plan),
         f"{_sm(same_shape)}/7 against {_sm(same_plan)}/7"),
    ]

    def render(items):
        return _table(
            [[tag, text, "✅ held" if hit else "❌ **wrong**", ev] for tag, text, hit, ev in items],
            ["#", "prediction", "outcome", "measured"],
        )

    held = sum(1 for _, _, hit, _ in checks if hit)
    return f"""
## The eight pre-registered predictions

**{held} of 8 held.**

{render(checks)}

### The three post-hoc predictions, from the amendment

All three held, which is what hindsight-informed predictions are supposed to do and is
the reason they are reported separately rather than added to the count above.

{render(posthoc)}
"""


def _conclusion(runs: dict) -> str:
    plan = runs[("frame_same", "plan")]
    shape = runs[("frame_same", "shape")]
    return f"""
## What this phase establishes

**Mechanized, and demonstrated:** the template, the slot vocabularies, the position
vocabulary, the counterfactual *content*, the answer-key-free metric, the accuracy check,
and — when the examples are homogeneous — the repeated-slot constraint and the
`xx_mismatch`-shaped alternate that falls out of it. The whole of it runs through
`pipeline.discover()` with **no pre-existing module changed**, which is the same measure
Phases 6 and 7 used.

**Failed to mechanize, and this is the phase's substance:**

1. **Choosing which counterfactual to trust.** The proposal step put a 5/7 scheme in the
   candidate set on both fixtures and the answer-key-free ranking picked a 3/7 one both
   times. Maximum output divergence is not the right criterion, and this phase does not
   have a better one — replacing it after seeing which scheme it should have picked would
   be fitting the rule to the answer key.
2. **Detecting that the human's examples disagree with each other.** Two lines in
   thirty-two, tokenized in a way no person can see, dissolved a structural constraint and
   turned the clean condition into a coin flip — silently. The repair converts that into a
   drop or a refusal, which is better, and it is post-hoc.
3. **Noticing that more than one template was supplied.** IOI over eight templates
   produces a confident structure over 12 of 32 examples and says nothing.

**The honest summary of the headline:** an induced task recovers
{_sm(plan)}/7 pre-registered and {_sm(shape)}/7 repaired, against a hand-built
{BASELINE['size_matched']}/7, and at the inherited cutoff the repaired version recovers
the entire published circuit at precision {_cut(shape)['precision']:.2f} against 0.78.
Mechanized task construction is **not free and not impossible**. It costs precision, it
costs a head, and it fails in ways that are diagnosable rather than mysterious.

## What it does not establish

- **It does not choose a behaviour.** The hunch and the decision of where to cut the
  prompt were human in every run here, exactly as the plan said they would be. The top
  row of the README ladder is untouched.
- **Rediscovery cannot validate task *invention*.** It validates task *construction*:
  given a behaviour with a published circuit, does mechanized construction reach it. The
  case that matters for oversight has no published anything, and no version of this
  experiment covers it.
- **n = 1 circuit, 1 model, 2 frames.** The k-sweep and the pair check add sample size on
  the cheap measurements only. Run E adds two tasks on the induction alone.
- **The k = 2 results rest on 8 generated prompts.** Large effects, tiny sample.
- **`clean_argmax_logprob` has no guaranteed positive span.** It did not bite on these
  fixtures; the known-answer suite shows a synthetic frame where all four induced schemes
  come out negative, and every normalized number in the repository divides by that span.
"""


def write_report(path: Path) -> None:
    char = _load("phase10_characterization.json")
    ksweep = _load("phase10_ksweep.json")
    pairs = _load("phase10_pairs.json")
    runs = {}
    for fixture in FIXTURES:
        for mode in MODES:
            payload = _load(f"phase10_{fixture}_{mode}.json")
            if payload is None:
                raise SystemExit(f"missing results/phase10_{fixture}_{mode}.json — run it first")
            runs[(fixture, mode)] = payload

    sections = [
        _intro(runs),
        _ladder(runs, char),
        _preregistered(runs, char),
        _repair(runs),
        _ksweep(ksweep, pairs),
        _cross_task(char),
        _predictions(runs, char, ksweep),
        _conclusion(runs),
    ]
    path.write_text("\n\n".join(s.strip() for s in sections) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    write_report(RESULTS / "PHASE10_REPORT.md")
