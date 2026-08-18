# Phase 10 plan — from a hunch to a runnable task, and what is left over

**Committed before any Phase 10 code exists**, together with the human-authored
prompts in [`../fixtures/`](../fixtures/). The order is checkable in git rather than
asserted: this commit contains the question, the algorithm, the scoring table, eight
predictions and the human input, and no results and no implementation.

## 0. The dependency this phase inherits

Every task in this project — IOI, greater-than, docstring — was hand-built by someone
who already knew which behaviour to probe and what a correct answer looks like.
[PHASE5_AUDIT.md](PHASE5_AUDIT.md) itemised what that costs, and section 3 of it put
task construction **explicitly out of scope**, on the grounds that a weak version
"would produce something that looked like progress without being any". Five phases
later the top two rows of the README ladder are still `supplied`.

Phase 9 closed out the agreement-flagging line with a negative result. That line is
finished; this phase does not revisit it.

## 1. The question, narrowed

**Not attempted.** Autonomous task discovery on an unfamiliar model — pointing the
pipeline at a model nobody has characterized and having it come back with a behaviour
worth studying. Phase 5 judged that intractable in one step and nothing since has
changed that judgement. Forcing a weak version would produce a report section with no
evidence, which is the trap Phase 5 declined and this phase declines again.

**Attempted.** One rung below it:

> Given a model, a **one-sentence behavioural hunch**, and a handful of **example
> prompts written by a person who has only that hunch** — how much of the remaining
> task construction can be mechanized, and what is irreducibly left for the human?

The hunch here is: *this model seems to know that the end of a date range comes after
its start.* That sentence, and the prompt files it produced, are the entire human
contribution this phase permits itself. Everything downstream of them is either
mechanized or reported as a failure to mechanize.

This is a **scoping-plus-prototype** phase in the shape of Phase 5, not a new causal
method. What is new is a piece of machinery — an induction step — that sits *upstream*
of everything Phases 1–9 built, and the test of it is the one every prior phase used:
does the pipeline still recover a circuit whose answer is already published?

## 2. Working backward from a hand-built task

`causal_interp/greater_than.py` is 457 lines. Read as a list of *things a human
supplied*, it decomposes as follows. The right-hand column is this plan's proposal,
fixed before any of it is built.

| # | what the task module supplies | proposal |
|---|---|---|
| 1 | that this model does numeric comparison at all | **human** — the hunch |
| 2 | `TEMPLATE`, a format string with named slots | **mechanize** — induce slots from the examples |
| 3 | `NOUNS`, `CENTURIES`, `MIN_YY`/`MAX_YY` — the slot vocabularies | **mechanize** — the observed values in the examples *are* the vocabularies |
| 4 | `_single_token_nouns`, `_valid_years`, `_splits_as_two_tokens` — tokenizer filtering | **mechanize** — a round-trip check, needing no notion of a noun or a year |
| 5 | that the two century slots must hold the same value | **mechanize** — columns that co-vary across every example are tied |
| 6 | `POSITIONS` and `_locate_positions` | **mechanize** — the varying columns are the positions (Phase 4 showed bare indices suffice) |
| 7 | `CORRUPT_YY = 1`, and that setting `YY` to `01` is the counterfactual | **partly** — propose one counterfactual per slot; the *content* becomes a resample, not a chosen value |
| 8 | `XX_MISMATCH`, the alternate authored in Phase 8 | **mechanize** — falls out of row 5, as "break the tie" |
| 9 | which scheme is `primary=True` | **mechanize** — pick by measured divergence, no answer key |
| 10 | `logit_diff` — the probability difference over year tokens above `YY` | **partly** — Phase 5 already removed the answer key; what remains is proposing *a* scalar |
| 11 | `year_rank_stats` — does the model actually do this | **mechanize** — agreement with the model's own clean prediction |
| 12 | that the prompt is cut immediately before the answer | **human** — the examples encode it, and nothing recovers it from them |

Rows 1 and 12 are the honest floor and are not attacked. Row 10 is half-inherited:
Phase 5 established that a distributional metric locates the circuit as well as the
answer-keyed one, so the open part is only *which scalar an auto-built task offers to
the intervention code*, not whether an answer key is needed.

## 3. The algorithm, fixed here

Written out completely, so that the implementation is a transcription and not a design
step. Phase 4 fixed its search space this way for the same reason.

### 3.1 Induction — from example strings to slots

```
INDUCE(examples E, tokenizer):
    rows  <- tokenize(e) for e in E, with BOS
    L     <- the modal row length
    keep  <- rows of length L                        # the dropped count is REPORTED
    M     <- the (k x L) token matrix of `keep`
    frame <- columns c where M[:,c] is constant
    slotc <- the remaining columns
    slots <- partition `slotc` by the value vector M[:,c]:
             two columns belong to the SAME slot iff they hold the same token in
             every example.  A slot is (its columns, its observed values).
    label(i) = "END" if i == L-1 else "t{i}"
    positions <- tuple(label(i) for i in sorted(slotc)), with "END" always present
```

No step consults what a token means. The tie rule in `slots` is the mechanized form of
"both century slots hold the same century", and it is discovered from co-variation
rather than declared.

### 3.2 Generation — from slots to a dataset

```
GENERATE(slots, frame row, n, seed):
    repeat up to 50n times, until n distinct rows are accepted:
        row <- copy of the frame row
        for each slot s:  v <- rng.choice(s.values);  write v to every column of s
        accept row iff  encode(decode(row minus BOS)) == row minus BOS   # round-trip
                  and   row is new
    report: attempts, round-trip rejections, duplicates, final count
```

The round-trip filter is the mechanized replacement for row 4 of the table above. It
asks only "would the tokenizer produce these tokens from this string" — a property of
the tokenizer, not of the task. **It is not a free parameter**: it is on, and its
rejection rate is reported.

Sampling slots independently is what makes rows 3 and 5 mechanizable and is also the
step most likely to produce nonsense. That is why the rejection rate is reported, and
why row 5's tie rule exists at all.

### 3.3 Counterfactual proposal — from slots to schemes

```
PROPOSE(slots):
    for each slot s:                       resample_{label(s)}
        redraw s's value (constrained to differ) and write it to ALL of s's columns
    for each slot s with more than one column, for each column c in s:
                                           desync_{label(c)}
        redraw only column c, leaving its partners at the clean value
    always:                                random_vocab_any
        the existing task-agnostic scheme from corruption.py, unchanged
```

Every proposed scheme changes exactly one slot or one column, so clean and corrupted
rows stay token-aligned by construction — the property `interventions.py` requires and
that all three hand-built tasks arrange by hand.

`resample_*` is the mechanized analogue of `yy01`: the same shape (one position, one
substitution), different content — a value drawn from the task's own observed
distribution instead of a value a human picked because it makes the constraint vacuous.
`desync_*` exists only for tied slots and is the mechanized analogue of `xx_mismatch`.

Provenance labels, for the Phase 8 registry: `resample_*` and `desync_*` register as
**`authored`**, not `generic`. They encode no knowledge of what the task means, but
they do encode the human's examples, and calling them generic would overstate the
result. `random_vocab_any` stays `generic`.

`preserves_answer` is **not** knowable without the answer key, so every induced scheme
declares `preserves_answer=False` and the field is reported as unavailable for
auto-built tasks. That is a real loss against Phase 8's registry and is recorded as one
rather than papered over.

### 3.4 Choosing the primary — no answer key

```
SELECT_PRIMARY(model, induced schemes):
    for each induced (non-generic) scheme s:
        build its dataset; run clean and corrupted
        D(s) <- mean_i KL( P_clean,i( . | END )  ||  P_corrupted,i( . | END ) )
    primary <- argmax D(s);  ties broken by lowest column index
```

Two forward passes per scheme. The generic scheme is excluded from candidacy for the
same reason no hand-built task makes `random_vocab_any` primary — it is Phase 5's
baseline, not a proposal.

### 3.5 The metric an auto-built task offers

The dataset contract requires a method the intervention code calls `logit_diff`. The
auto-built task fills it with, in full:

```
    at construction, one clean forward pass:   t*_i = argmax_v  P_clean,i( v | END )
    logit_diff(logits)  =  mean_i  log P_i( t*_i | END )
```

**`clean_argmax_logprob`.** It needs no answer key: the target token is whatever the
model itself predicts on the clean prompt. It is not identical to `kl` or `tv`, so the
pipeline's three-metric grid still reports three distinct quantities.

What it measures is *restoration of the model's own clean behaviour*, which is the
quantity circuit discovery actually wants and is not the same as *restoration of the
correct answer*. The gap between the two is measurable once, at the end, against the
published task, and that measurement is reported.

## 4. The test case, and the ambiguity in choosing it

The brief asks for "whichever wasn't the original target for its circuit's paper". Read
literally that selects none of the three — IOI, greater-than and docstring were each
their own paper's target. Read as the constraint it is plainly protecting against —
*keep distance from the case where this project already knew the answer* — it selects
against **IOI**, whose answer key was in view while every threshold, cutoff and rule in
Phases 1–5 was chosen.

That leaves greater-than and docstring, and this plan picks **greater-than**:

- Its hand-built baseline is clean and near-maximal — Phase 6 recovered 6/7
  size-matched and 7/7 at the 0.02 cutoff — so any degradation caused by mechanization
  is visible and attributable.
- Docstring's baseline is 3/6, and Phases 7–9 established that the number is a property
  of the counterfactual rather than of the circuit. An auto-built docstring task
  scoring 3/6 would be uninterpretable: the result would be entangled with the open
  problem Phase 9 has just declared a ceiling.
- Its published counterfactual is a **single-token substitution at one position**,
  which is the shape section 3.3 generates. The comparison is like-for-like.
- Its Phase 8 alternate `xx_mismatch` was authored by this project, which gives section
  3.3 a human-designed target to be checked against.

Docstring and IOI still appear, in section 6, as induction-only checks.

## 5. The human input, and how it is counted

Two files, both committed with this plan, both written before any code existed:

| file | lines | frame |
|---|---|---|
| `fixtures/greater_than_frame_same.txt` | 32 | the published sentence frame |
| `fixtures/greater_than_frame_own.txt` | 32 | a frame written for this phase |

`frame_same` isolates the mechanization: the same surface form as the published task,
so a difference in recovery is attributable to auto-construction alone. `frame_own`
adds surface generalization on top, and exists so that a pass on `frame_same` cannot be
read as more than it is.

**Neither file was filtered.** No line was checked against the tokenizer, against this
repo's `NOUNS` list, or against whether GPT-2 small actually performs the behaviour on
it. That is deliberate: the filtering *is* one of the things being mechanized, and a
pre-filtered fixture would hand the induction a result it is supposed to earn. Neither
file is edited after this commit; if a line turns out to be unusable, the induction
drops it and the drop is reported.

The behavioural hunch, and the decision to cut each line immediately before the answer,
are the two things rows 1 and 12 of section 2 mark as human. They are visible in these
files and are not recovered from them by anything.

## 6. What runs

Model `gpt2-small`, `n = 128` prompts, `seed = 0`, the activation-patching channel, the
0.02 cutoff inherited from Phase 1 — every one of them the setting Phases 6 and 8 used,
so the only thing that differs from Phase 6's greater-than run is how the task was
built.

| run | what it does |
|---|---|
| **A. induction report** | run section 3.1–3.4 on both fixtures and print what comes out: dropped lines, slots, ties, positions, schemes, rejection rates, the selected primary. No GPU sweep. |
| **B. rediscovery, `frame_same`** | `pipeline.discover()` on the auto-built `TaskSpec`, every proposed scheme, then score against `ground_truth_greater_than` |
| **C. rediscovery, `frame_own`** | the same, on the second fixture |
| **D. the k-sweep** | for k in {2, 4, 8, 16, 32}, take the **first k lines** of `frame_same`, re-run 3.1–3.4, sweep the selected primary only, and score. Answers "how many examples does the human have to write" with a curve instead of an opinion. |
| **E. cross-task induction** | run 3.1–3.3 on prompts taken from `IOIDataset` and `DocstringDataset`, and report what the induction produces against each module's hand-written `POSITIONS` |

Run D sweeps one scheme rather than every scheme, and that is a deliberate exception to
Phase 8's structure, declared here: D reports a *recovery curve against sample size*,
not a circuit claim, and the multi-scheme requirement exists to stop a head list being
published from a single counterfactual. B and C — which do publish head lists — sweep
everything.

Run E is **weaker than A by construction and is labelled so wherever it appears**: its
prompts come out of a hand-built generator, so the induction is handed an already-clean,
already-aligned sample. The only question asked of it is whether the induced slot
structure matches what those two modules hand-code.

`induction.py` and `autotask.py` must never import a `ground_truth` module, for the
reason `search.py` and `agreement.py` carry the same prohibition. The runner asserts it
at startup.

## 7. Scoring, fixed before the run

The headline is run **B**: the auto-built `frame_same` task, its auto-selected primary
scheme, the `clean_argmax_logprob` metric, scored **size-matched at top 7** — the
published circuit's size, which has no free parameter and therefore cannot be tuned.

The baseline it is measured against is Phase 6's stored number for the same circuit
under the hand-built task: **6/7 size-matched**, 7/7 at the 0.02 cutoff, precision 0.78.

| outcome | criterion |
|---|---|
| **clean pass** | the auto-built task recovers **≥ 6/7** size-matched — it matches the hand-built baseline |
| **partial** | **4–5/7** — mechanization costs recall but the circuit is still substantially located |
| **negative** | **≤ 3/7** — auto-construction does not produce a task the pipeline can work with, and the phase reports that |

Reported alongside, and **not** part of the criterion: the 0.02-cutoff set, precision,
the per-scheme table, the union across schemes, and the Phase 8 agreement output. Phase
9 showed the shared 0.02 cutoff is indefensible for cross-scheme comparison, so the
agreement block is carried as description and no Phase 10 claim rests on it.

A negative result is reported as the phase's finding, in the README ladder as well as
the report, and is not retried with a different induction rule. The rule in section 3
is the one that gets tested.

## 8. Predictions

Scored in `PHASE10_REPORT.md` whichever way they come out.

| # | prediction |
|---|---|
| P1 | The modal-length filter drops **at least one line from each fixture** — a person writing natural examples cannot see the tokenizer, and this is what row 4 costs |
| P2 | On `frame_same` the induction returns **exactly 3 slots**, with the two century columns **tied**, and the induced position set is a **subset** of the hand-built `POSITIONS` — no position is invented that the task module does not have |
| P3 | The proposed scheme set contains a **`desync` on the century slot** — Phase 8's authored `xx_mismatch`, re-derived from the examples without being told |
| P4 | The auto-selected primary is the **`YY` slot** — the position the published counterfactual acts on |
| P5 | **Headline:** run B recovers **≥ 6/7** size-matched |
| P6 | Run B's precision at the 0.02 cutoff is **lower** than the hand-built 0.78 — a resampled counterfactual should be noisier than a chosen one |
| P7 | The k-curve is **flat above k = 8**: k = 8, 16 and 32 give the same size-matched recovery, and k = 2 gives strictly fewer |
| P8 | Run C (`frame_own`) recovers **strictly fewer** heads size-matched than run B |

## 9. What this phase cannot establish, stated in advance

- **It does not choose a behaviour.** Row 1 of section 2 is untouched, and a phase that
  starts from a hunch has not attacked the question of where hunches come from. The top
  row of the README ladder stays `supplied` whatever happens here.
- **Rediscovery on a published circuit cannot validate task *invention*.** It can only
  validate task *construction* — given a behaviour someone has already published a
  circuit for, does mechanized construction reach the same circuit. The case that
  matters for oversight is the one where nobody has published anything, and there is no
  version of this experiment that covers it. This is the same limit Phase 5 named when
  it declined the problem, and narrowing the question does not remove it.
- **n = 1 on the expensive measurement.** One circuit, one model, two fixtures. Runs D
  and E add sample size on the cheap measurements only.
- **The examples are written by the same person who knows the answer.** Section 10.

## 10. Disclosure — this plan is not blind

Stated in the form Phases 3 and 9 stated theirs. The greater-than circuit's head list
has been in this repository since Phase 6, and the person writing this plan has seen
Phase 6's and Phase 8's numbers for it. Three specific consequences:

1. **The fixtures were written by someone who knows what the task is.** They were not
   filtered and their frame was not tuned, but the choice to write date-range sentences
   at all comes from the hunch, and the hunch here was formed by reading a paper. On an
   unfamiliar model the hunch would come from somewhere less reliable, and nothing here
   measures how much worse that is.
2. **The algorithm in section 3 was designed while knowing greater-than's structure.**
   The tie rule in 3.1 exists because a person knew a template can repeat a slot. It is
   fixed here, before any run, which is what makes it checkable; it is not blind.
3. **What is genuinely unknown at commit time**: how many fixture lines survive the
   length filter, what the round-trip rejection rate is, which scheme the divergence
   rule picks, whether the auto-built task recovers the circuit, where the k-curve
   turns over, and whether `frame_own` works at all. Those are what runs A–E measure,
   and every one of them can come out against this plan.
