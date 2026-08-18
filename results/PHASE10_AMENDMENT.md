# Phase 10 amendment — one repair, and why it does not get to be the headline

**Committed after [PHASE10_CHARACTERIZATION.md](PHASE10_CHARACTERIZATION.md) and before
the repaired code exists.** Both orderings are checkable in git. This document is
hindsight-informed by construction and says so in its title.

## 1. The tension, stated before it is resolved

Section 7 of [PHASE10_PLAN.md](PHASE10_PLAN.md) says:

> A negative result is reported as the phase's finding, in the README ladder as well as
> the report, and is not retried with a different induction rule. The rule in section 3
> is the one that gets tested.

Step 1 then found a specific, diagnosable defect: the tie rule requires unanimity, and
two tokenizer-heterogeneous lines out of thirty-two dissolve a tie the other thirty
support. There is an obvious repair. Applying it and reporting the improved number as
the phase's result would be exactly what that clause exists to forbid.

Not applying it at all is also wrong, for a different reason: it leaves a reader unable
to tell *"mechanized task construction does not work"* from *"mechanized task
construction has one fixable bug"*, and those imply completely different next phases.

**The resolution, fixed here:**

- The **headline** — the number in the README ladder, the row in the scoring table, the
  eight predictions — is the **pre-registered** algorithm, run to a head list. It is
  scored under section 7 of the plan whatever it comes out as, and the repair does not
  replace it.
- The repaired run is reported **beside** it, labelled post-hoc everywhere it appears,
  and carries no pre-registered weight. It answers one question — *is the failure the
  mechanism or the bug* — and nothing else.
- Both runs appear in the README. If they disagree, the README says so and reports the
  pre-registered one as the phase's result.

This is Phase 2's arrangement, not a new one: Phase 2 published the receiver-side
measurement it had stumbled into, refused to adopt it, and made a *later* phase
pre-register a rule for it. The repair below is a candidate for that treatment.

## 2. The repair

Exactly one change, to section 3.1 of the plan. Nothing else in section 3 is touched.

> **Was:** keep the rows whose token length is modal.
>
> **Is:** keep the largest group of rows sharing a **shape**, where a row's shape is
>
> ```
> shape(row) = ( len(row),  tuple(first column holding the same token as column c
>                                 for c in range(len(row))) )
> ```
>
> Group-size ties are broken toward the group containing the earliest example.

**It is a strict generalization.** Rows of different lengths necessarily have different
shapes, so the modal-length filter is the special case that compares only the first
element of the pair. Nothing that survived before is newly rejected on length grounds.

**It is parameter-free.** "The largest group" has no threshold in it, no fraction and
nothing to tune, which is the property that made the modal-length rule acceptable in the
plan and is the reason this is the repair rather than a similarity cutoff on pairwise
column agreement. That alternative was considered and rejected here, before running
either: it would need a number, and a number chosen after seeing which value ties the
century columns is not a rule, it is the answer written backwards.

**It costs the human input.** The filter can now drop examples the old one kept — 2 of
32 on `frame_same`, 1 of 32 on `frame_own`, measured in step 1. A person's examples buy
less than they appear to, and the amount is reported rather than absorbed.

## 3. What is deliberately not repaired

Each of these is a defect step 1 exposed. None is fixed, and the reason is the same in
every case: fixing it would require a choice this phase has no independent way to make.

- **The selection rule picked the century, not `YY`.** Left alone. Section 3.4's argmax
  over measured divergence is what it is; replacing it after seeing which counterfactual
  it should have chosen is fitting the rule to the answer key. The repaired run re-runs
  the *same* rule over the *repaired* structure, and whether that changes the answer is
  a measurement, not a design.
- **`clean_argmax_logprob` has no guaranteed positive span.** Left alone, reported per
  scheme. A scheme with a non-positive span is flagged in the output and not dropped —
  dropping would be a free parameter.
- **Nothing prunes the proposal set.** Docstring's 16 slots produce 18 schemes. Left
  alone; a pruning rule is a cutoff.
- **The induction cannot tell it was given more than one template.** IOI over eight
  templates fails and does not know it. Left alone, and named in the report as the
  clearest single obstacle to running this on prompts a human did not curate.
- **Both fixtures are untouched.** No line is edited, removed or added.

## 4. Predictions for the repaired run

Hindsight-informed and labelled as such. They are recorded because a prediction made
after a diagnosis is still falsifiable, not because they carry the weight of section 8's.

| # | post-hoc prediction |
|---|---|
| A1 | On `frame_same` the shape filter keeps **30** examples, finds **3 slots** with the century columns tied, and proposes **5** schemes including `desync_t7` and `desync_END` |
| A2 | The repaired selection rule picks a **different** primary from `resample_t7` — the tied `resample_t7` now moves both centuries together and preserves the greater-than relation, so it should lose divergence |
| A3 | The repaired run recovers **strictly more** of the 7 published heads, size-matched, than the pre-registered run |

A3 is the one that matters. If it fails, the repair was not the problem and the phase's
negative result is about the approach rather than the bug.

## 5. What neither run can establish

Unchanged from section 9 of the plan, and restated because the repair does not touch it:
rediscovery on a published circuit validates task *construction*, never task
*invention*. The behavioural hunch and the decision of where to cut the prompt remain
human in both runs, and n is still 1 circuit, 1 model, 2 fixtures.
