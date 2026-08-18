# Phase 10 — inducing the task instead of writing it

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

**Negative, as the plan defines it.** — the pre-registered induction builds a task on which activation patching
recovers **3 of the 7** published heads, size-matched, against the hand-built
task's **6 of 7**. Section 7 of the plan calls
3/7 `negative`.

The post-hoc repair reaches **5/7** size-matched and
**7/7 at the inherited 0.02 cutoff**,
which is the whole published circuit. It is reported throughout as post-hoc and it does
not replace the line above.

**One of eight pre-registered predictions held.**

## What got mechanized, and what did not

This is the phase's actual deliverable. `greater_than.py` is 457 lines of hand-built
task; the table says which of it survived being derived from example prompts.

| ingredient | status | how / what it cost |
|---|---|---|
| which behaviour to study | **human** | the one-sentence hunch; not attempted |
| where to cut the prompt | **human** | encoded in the examples, recovered from nothing |
| example prompts | **human** | 32 lines per frame, unfiltered |
| prompt template | mechanized | 9 frame columns found by constancy |
| slot vocabularies | mechanized | the observed values are the vocabulary |
| tokenizer filtering | mechanized (partly) | round-trip filter rejects 7% of candidates; it does **not** catch a same-length mis-split |
| repeated-slot constraint | mechanized (fragile) | co-variation; unanimity-based, and 2 lines in 32 defeat it |
| position vocabulary | mechanized | the varying columns, as bare indices |
| counterfactual content | mechanized | resample from the slot's own observed values |
| the `xx_mismatch` alternate | mechanized | falls out of the tie as `desync`, once the tie survives |
| which scheme is primary | mechanized (**and it chose badly**) | argmax measured divergence; picked the century, not the year |
| the metric | mechanized | `clean_argmax_logprob`, no answer key |
| accuracy check | mechanized | agreement with the model's own clean prediction |

The two `human` rows at the top are the floor the plan named in advance and did not
attack. Everything below them was mechanized to some degree, and two of the mechanized
rows failed in ways that are worth more than the rows that worked.

## The pre-registered run

| fixture | lines kept | slots | tied | primary chosen | size-matched | precision | task actually greater-than |
|---|---|---|---|---|---|---|---|
| `frame_same` | 32/32 | 4 | 0 | `resample_t7` | 3/7 | 0.22 | 48% |
| `frame_own` | 32/32 | 4 | 0 | `resample_END` | 3/7 | 0.40 | 62% |

Both fixtures land on **3/7**, and the final column is why: the task the
induction built is not greater-than. With the two century columns treated as
independent slots, generation samples them independently, so a clean prompt reads

```
The pilgrimage lasted from the year 11245 to the year 14
The migration  lasted from the year 1527  to the year 11
```

— mismatched centuries, and in the first case a five-digit year. The model's top
prediction is a two-digit year on 100% of them and *exceeds the start year on
48%*. **The clean condition is a coin
flip**, which is the published task's own `xx_mismatch` counterfactual served as the
control.

### The selection rule is the second failure, and it is separable

Every proposed scheme, scored against the published circuit after the fact:

| scheme | size-matched | precision |  |
|---|---|---|---|
| `resample_t2` | 2/7 | 0.09 |  |
| `resample_t7` | 3/7 | 0.22 | **primary** |
| `resample_t8` | 5/7 | 0.50 |  |
| `resample_END` | 4/7 | 0.46 |  |
| `random_vocab_any` | 1/7 | 0.12 |  |

`resample_t8` — redrawing the **start year**, which is the position the published
`yy01` counterfactual acts on — recovers **5/7 at precision 0.50**. The rule picked
`resample_t7` instead, on a divergence three times larger. **The information needed to
choose well was in the candidate set; the answer-key-free rule for choosing did not find
it.** On `frame_own` the same rule picked `resample_END`, a third position again.

That is a cleaner result than the headline number suggests: the counterfactual *content*
mechanized, and the counterfactual *ranking* did not.

## The repair — post-hoc, and it works

One change, fixed in the amendment before it ran: keep the largest group of examples
sharing a **column shape** rather than a token **length**. A strict generalization, with
no threshold in it.

| fixture | induction | kept | tied slots | primary | size-matched | at 0.02 | precision | task valid |
|---|---|---|---|---|---|---|---|---|
| `frame_same` | pre-registered | 32/32 | 0 | `resample_t7` | 3/7 | 6/7 of 27 | 0.22 | 48% |
| `frame_same` | **post-hoc** | 30/32 | 1 | `resample_t8` | 5/7 | 7/7 of 14 | 0.50 | 100% |
| `frame_own` | pre-registered | 32/32 | 0 | `resample_END` | 3/7 | 4/7 of 10 | 0.40 | 62% |
| `frame_own` | **post-hoc** | 31/32 | 1 | `resample_t9` | 5/7 | 7/7 of 14 | 0.50 | 100% |
| hand-built (Phase 6) | — | — | — | `yy01` | 6/7 | 7/7 of 9 | 0.78 | 100% |

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
3/7.

## How many examples does the human have to write? — the curve runs backwards

| induction | k = 2 | k = 4 | k = 8 | k = 16 | k = 32 |
|---|---|---|---|---|---|
| pre-registered | 6/7 | 6/7 | 1/7 | 2/7 | 3/7 |
| **post-hoc** | 6/7 | 6/7 | 6/7 | 5/7 | 5/7 |

Prediction P7 said this would be flat above k = 8 with k = 2 strictly worse. It is
**inverted**: two example prompts recover **6/7** and
thirty-two recover **3/7**.

The mechanism is visible in the same table's build data. At k = 2 and k = 4 the examples
happen to be homogeneous, so the century tie holds and the rule selects `resample_t8`.
At k = 8 line 5 — `...the year 1509 to the year 15` — enters the sample, the tie
dissolves, the primary flips to `resample_t7`, and recovery collapses to
1/7. **Every additional natural example is another chance
to poison a unanimity-based rule**, and the pre-registered induction has no way to say so.

The post-hoc filter removes the monotone decay but not all of it: 6/7 at k ≤ 8, 5/7 at
k ≥ 16.

### Does k = 2 depend on *which* two lines?

Seven contiguous pairs, chosen by position and not by result, all reported:

| fixture lines | tied | primary | size-matched | task valid | under the repair |
|---|---|---|---|---|---|
| 0, 1 | 1 | `resample_t8` | 6/7 | 100% | 6/7 |
| 4, 5 | 0 | `resample_t7` | 3/7 | 83% | refuses to build |
| 6, 7 | 0 | `resample_t7` | 2/7 | 67% | refuses to build |
| 10, 11 | 1 | `resample_t8` | 5/7 | 100% | 5/7 |
| 20, 21 | 1 | `desync_t7` | 5/7 | 100% | 5/7 |
| 30, 31 | 1 | `resample_t8` | 5/7 | 100% | 5/7 |
| 2, 3 | 1 | `resample_t8` | 5/7 | 100% | 5/7 |

**Five of seven pairs reach 5–6/7.** The two that do not are exactly the two containing a
tokenizer-odd line, and under the repaired filter those two **refuse to build at all** —
one example survives, and the induction raises rather than producing a task. That is the
better failure: Phase 8's standard was that the pipeline should say what it cannot see,
and here it does, at the cost of needing the human to supply more lines.

The caveat is real and not small: a k = 2 task generates at most 8 distinct prompts, so
these are 144-head sweeps over 8 examples. The effects are large — spans of +2.7 to +3.9
against the 32-line task's +2.6 — but the sample is tiny.

## Run E — the induction on the other two tasks

Weaker than everything above by construction, and labelled so: these prompts come out of
hand-built generators, so the induction is handed an already-aligned sample. The only
question is whether the structure it induces matches what those modules hand-code.

| case | examples kept | induced | verdict |
|---|---|---|---|
| IOI, all 8 templates | **12/32** | 11 slots | **fails** — one slot's values are `','` and four names |
| IOI, one template | 32/32 | 5 slots | recovers the three name positions, the place and the object |
| docstring | 32/32 | 14 slots, **2 tied** | **the tie rule works on a task it was never pointed at** |

The docstring result is the strongest single piece of evidence here that the mechanism
generalizes. `t11`↔column 27 and
`t13`↔column 34 are the argument names that appear
once in the function signature and again in the docstring — the `A_def`/`A_doc` and
`B_def`/`B_doc` pairs `docstring.py` hand-codes as separate named positions. Found from
32 prompts, on a different model with a different tokenizer, with nothing task-specific
in the code.

The IOI failure is the clearest single obstacle to running any of this on prompts a
human did not curate: the eight published templates have different token lengths, the
filter keeps the plurality one, and **nothing in the induction detects that it was handed
more than one frame**. It reports a confident structure over 12 of 32 examples.

## The eight pre-registered predictions

**1 of 8 held.**

| # | prediction | outcome | measured |
|---|---|---|---|
| P1 | the length filter drops ≥ 1 line from each fixture | ❌ **wrong** | dropped 0 and 0 — the mis-split preserves row length, so the filter cannot see it |
| P2 | 3 slots on `frame_same`, centuries tied, positions a subset of the hand-built set | ❌ **wrong** | 4 slots, none tied. The *positions* half held — `t2/t7/t8/END` map to `NOUN/XX1/YY/END`, a subset of the hand-built five |
| P3 | a `desync` on the century slot is proposed — `xx_mismatch` re-derived | ❌ **wrong** | nothing was tied, so nothing was proposed. **Held under the post-hoc repair** |
| P4 | the selected primary is the `YY` slot | ❌ **wrong** | chose `resample_t7` on `frame_same` and `resample_END` on `frame_own`. **Held under the repair** |
| P5 | headline recovers ≥ 6/7 size-matched | ❌ **wrong** | 3/7 — the plan's `negative` band |
| P6 | precision at 0.02 is below the hand-built 0.78 | ✅ held | 0.22 against 0.78 |
| P7 | the k-curve is flat above k = 8 and k = 2 is strictly worse | ❌ **wrong** | inverted — k=2 gives 6/7 and k=32 gives 3/7, monotonically decreasing in between |
| P8 | `frame_own` recovers strictly fewer than `frame_same` | ❌ **wrong** | 3/7 against 3/7 — identical, under both inductions |

### The three post-hoc predictions, from the amendment

All three held, which is what hindsight-informed predictions are supposed to do and is
the reason they are reported separately rather than added to the count above.

| # | prediction | outcome | measured |
|---|---|---|---|
| A1 | the repair keeps 30, finds 3 slots with the centuries tied, proposes 5 schemes including `desync_t7` and `desync_END` | ✅ held | kept 30, 3 slots, 1 tied, proposals ['resample_t2', 'resample_t7', 'resample_t8', 'desync_t7', 'desync_END'] |
| A2 | the repaired rule picks a different primary from `resample_t7` | ✅ held | picked `resample_t8` — `resample_t7`'s divergence fell from 0.755 to 0.084 once both centuries moved together |
| A3 | the repaired run recovers strictly more published heads | ✅ held | 5/7 against 3/7 |

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
3/7 pre-registered and 5/7 repaired, against a hand-built
6/7, and at the inherited cutoff the repaired version recovers
the entire published circuit at precision 0.50 against 0.78.
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
