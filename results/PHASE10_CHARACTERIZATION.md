# Phase 10 step 1 — what the pre-registered induction actually produces

**Committed before any repair is designed, and before any head sweep is run.** This is
the ordering Phase 9 used and the reason it is used again: the measurement below is what
a fix would be designed against, so it goes on record first, on its own.

What ran is section 3 of [PHASE10_PLAN.md](PHASE10_PLAN.md) exactly as committed —
`scripts/phase10_characterize.py`, no repair, no tuning, and no edit to either fixture.
Raw output in `results/phase10_characterization.json`. No answer key is opened anywhere
in this document.

---

## 1. Both fixtures survive the length filter, and that is the problem

| fixture | lines | kept by the modal-length filter | dropped |
|---|---|---|---|
| `frame_same` | 32 | **32** | 0 |
| `frame_own` | 32 | **32** | 0 |

Prediction **P1** said at least one line from each fixture would be dropped. It is
**wrong**, and the way it is wrong is the finding. The tokenizer trap it was aimed at is
real and is present in the data — it just does not change the row *length*, so a
length filter cannot see it:

```
The rebellion lasted from the year 1509 to the year 15
      -> [ ... " year", " 150", "9", " to", " the", " year", " 15"]
The dynasty   lasted from the year 1124 to the year 11
      -> [ ... " year", " 112", "4", " to", " the", " year", " 11"]
```

GPT-2 splits `" 1509"` as `[" 150", "9"]`, not `[" 15", "09"]`. Two tokens either way,
so the row is still 13 long and the filter passes it. `greater_than.py` handles this by
hand: `_valid_years` keeps only the years that split as `[" XX", "YY"]`, and the plan's
table listed that as row 4, "mechanize — a round-trip check". **The round-trip check
does not catch it either**, because `[" 150", "9"]` *is* what the tokenizer produces
from that string. The row is canonical. It is just canonical in a different shape from
the other thirty.

## 2. The consequence: the tie dissolves

Because column 7 holds `" 150"` while column 12 holds `" 15"` in one example, and
`" 112"` against `" 11"` in another, the two century columns do not agree in *every*
kept example — and the plan's tie rule requires every one.

| fixture | slots found | tied slots | schemes proposed |
|---|---|---|---|
| `frame_same` | 4 — `t2`, `t7`, `t8`, `END` | **0** | 4 `resample`, **0 `desync`** |
| `frame_own` | 4 — `t5`, `t8`, `t9`, `END` | **0** | 4 `resample`, **0 `desync`** |

So prediction **P2** is wrong (3 slots with the centuries tied — found 4, untied) and
**P3** is wrong (`desync` on the century slot, the mechanized `xx_mismatch` — nothing
was proposed, because nothing was tied).

Two lines out of thirty-two decided both.

**How close it was, measured.** Grouping the examples by *column shape* — which
positions repeat each other, ignoring what the tokens are — separates them cleanly:

| fixture | distinct shapes | largest group | minority |
|---|---|---|---|
| `frame_same` | 2 | **30 / 32** | `1509`, `1124` |
| `frame_own` | 2 | **31 / 32** | `1108` |

This is a diagnostic, not a rule. Nothing in the plan consults it and nothing in this
document proposes that anything should.

## 3. What the generated task looks like

With the century columns treated as two independent slots, generation samples them
independently. The first five clean prompts of `frame_same`:

```
The pilgrimage lasted from the year 11245 to the year 14
The migration  lasted from the year 1527  to the year 11
The pilgrimage lasted from the year 1161  to the year 16
The revival    lasted from the year 1253  to the year 13
The truce      lasted from the year 1161  to the year 16
```

Row 1 is a five-digit year, from `" 112"` + `"45"`. Rows 1, 2 and 4 have a start century
that disagrees with the final one. **This is not the greater-than task.** It is a
mixture of the task and its own `xx_mismatch` counterfactual, presented as the clean
condition.

The round-trip filter is working and is not sufficient: 16 of 144 attempts were rejected
(11%), which is a real rate and the mechanized replacement for `_valid_years` doing real
work — but `" 112"` + `"45"` decodes to `" 11245"` and re-encodes to the same two
tokens, so it survives.

## 4. Per-scheme measurements

`KL` is the plan's selection statistic. `span` is the clean-to-corrupted difference under
`clean_argmax_logprob`, which every normalized number downstream divides by. `agree` is
how often the corrupted run still predicts the clean run's token.

### `frame_same`

| scheme | KL | span | agree | corrupted rows the tokenizer would not produce |
|---|---|---|---|---|
| `resample_t2` (noun) | 0.035 | +0.048 | 72% | 0% |
| **`resample_t7`** (start century) | **0.755** | +0.846 | 37% | 6% |
| `resample_t8` (start year) | 0.259 | +0.420 | 49% | 11% |
| `resample_END` (final century) | 0.409 | +0.700 | 32% | 0% |
| `random_vocab_any` (generic, not a candidate) | 1.805 | +2.038 | 38% | n/a |

**Selected primary: `resample_t7`.** Prediction **P4** said the `YY` slot — `t8` — and
is **wrong**. Redrawing the start century moves the output distribution three times as
much as redrawing the start year does, which is what the rule ranks on.

### `frame_own`

| scheme | KL | span | agree | corrupted round-trip failures |
|---|---|---|---|---|
| `resample_t5` (noun) | 0.087 | +0.139 | 55% | 0% |
| `resample_t8` (start century) | 0.314 | +0.596 | 40% | 6% |
| `resample_t9` (start year) | 0.344 | +0.659 | 38% | 9% |
| **`resample_END`** (final century) | **0.573** | +1.351 | 10% | 0% |
| `random_vocab_any` | 0.898 | +1.057 | 54% | n/a |

**Selected primary: `resample_END`** — changing the token the model is being asked to
continue from. A different answer again, on a fixture differing only in wording.

**The generic scheme has the highest divergence in both fixtures.** It is excluded from
candidacy by section 3.4 of the plan, which is the only reason it is not primary. Had
the plan not excluded it, the rule would have selected the counterfactual that supplies
no task knowledge at all — which Phase 5 measured as the worst of the four pairings.

**Every span here is positive.** The known-answer suite showed that
`clean_argmax_logprob` does not guarantee that — on a synthetic frame with no behaviour
in it, all four induced schemes came out negative. It did not bite on these fixtures.
It is recorded because a task where it does bite would produce sign-flipped normalized
recoveries and nothing would raise.

## 5. Cross-task induction (run E) — weaker by construction

These prompts come out of hand-built generators, so the induction is handed an
already-clean, already-aligned sample. The only question asked is whether the induced
structure matches what those modules hand-code.

### IOI, all eight templates — **fails, as it must**

| | |
|---|---|
| examples kept | **12 / 32** |
| induced slots | 11, at columns that mix names, commas, verbs and places across templates |

The eight published templates have different token lengths, so the modal-length filter
keeps only the plurality template and the induction reads the survivors as one frame.
Slot `t2` has values `[',', ' Nicole', ' David', ' Dennis', ...]` — a punctuation mark
and four names in one slot, which is what column alignment across incompatible frames
looks like. **The induction requires one template, and nothing in it detects that it
was given more than one.**

### IOI, one template — recovers the name and content positions

| | |
|---|---|
| examples kept | 32 / 32 |
| induced positions | `t3`, `t5`, `t9`, `t11`, `t14`, `END` |
| hand-built positions | `IO`, `IO+1`, `S1`, `S1+1`, `S2`, `S2+1`, `END` |

`t3`, `t5` and `t11` are the three name slots; `t9` is the place and `t14` the object.
The three `+1` positions are frame columns — constant across examples — so they are not
induced, which is correct behaviour and a real loss: Phase 1 measured effects at `S2+1`.

**It does not tie `t5` and `t11`.** That is right, and worth stating because it looks
like the same miss as section 2: in IOI the repeated subject is the first name in the
BABA order and the second in the ABBA order, so those columns genuinely disagree in half
the examples. The tie rule is not being defeated here — there is no tie.

### Docstring — the tie rule works, on a task it was not designed against

| | |
|---|---|
| examples kept | 32 / 32 |
| induced slots | 16, of which **two are tied**: `t11`↔`t27` and `t13`↔`t34` |

Those two are the argument names that appear once in the function signature and again in
the docstring — the `A_def`/`A_doc` and `B_def`/`B_doc` pairs that `docstring.py` hand-codes
as separate named positions. **The mechanism found a real structural feature of a
published task from 32 example prompts, with nothing task-specific in the code.**

It also proposes 18 schemes, which is what "one per slot plus one per tied column" gives
on a 41-token prompt with 16 slots. Nothing prunes that.

---

## 6. What this measurement establishes, before anything is designed on it

1. **The mechanized pieces that work**: slot detection, vocabulary extraction, position
   induction, round-trip filtering (an 11% real rejection rate), scheme proposal, and
   the tie rule *when the examples are homogeneous* — demonstrated on docstring, which
   this code has never been pointed at before.
2. **The mechanized piece that fails**: the tie rule is unanimity-based, and one
   tokenizer-heterogeneous line in thirty-two dissolves a tie the other thirty-one
   support. Downstream this costs the `desync` schemes entirely and makes the generated
   clean set a mixture of the task and its own counterfactual.
3. **The selection rule picks a different counterfactual on each fixture**, and neither
   is the published one. Whether that costs recall is not knowable from this document.
4. **Four of eight predictions are already wrong** — P1, P2, P3, P4 — and were wrong
   before a single head was patched.

What is *not* established here is the only thing the plan's scoring table cares about:
whether the task this induction builds recovers the published circuit. That needs the
head sweep, and it is run next, on the algorithm exactly as pre-registered.
