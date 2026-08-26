# Phase 5, what the task, corruption and metric still require knowing

Phase 4 closed the receiver-specification gap: the search no longer has to be told
which head input, at which position, to interrogate. Its own conclusion named what
remains, and this document takes that seriously enough to itemise it before
testing any of it.

three pieces are still built by hand. this is an audit of what each one encodes,
written before the experiments in `PHASE5_REPORT.md` were run, so that the
experiments couldnt quietly redefine what counted as a dependency.

the unit throughout is **what would be unavailable on a circuit nobody has
published**, not "what a human typed", since a human types the code either way.

---

## 1. the task template

```
"Then, {a} and {b} went to the {place}. {s} gave a {obj} to"   -> " Mary"
```

| what it encodes | available on an unfamiliar circuit? |
|---|---|
| that this model performs indirect object identification at all | **no.** choosing which behaviour to study is prior to everything else here. |
| that the behaviour has two name slots, one of which repeats | **no.** this is the mechanism's shape, stated in advance. |
| that the prompt should stop immediately before the answer token | partly, "predict the next token" is generic, but *where to cut* isnt. |
| that names must be single tokens under this tokenizer | yes, mechanical, checkable without any knowledge of the task. |
| that both name orders must be balanced to control a positional confound | **no.** requires knowing that a "second name" heuristic is a competing explanation. |

The first row is the load-bearing one. everything else in this project is
downstream of a decision that a specific, nameable behaviour exists and is worth
isolating. nothing in phases 1-4 discovers behaviours. they all analyse one that
was handed to them.

## 2. the corruption schemes

Both schemes are counterfactuals designed by someone who already knew which token
carried the answer.

**`s2_swap`**, replace the repeated subject with the indirect object, flipping
which name is correct.

| what it encodes | available? |
|---|---|
| which token position is S2 | partly, phase 4 showed positions can be *searched* rather than supplied. |
| that changing that one token flips the answer | **no.** this is knowledge of the task's semantics. |
| that changing exactly one token is desirable, to isolate a single variable | generic experimental-design knowledge, not task knowledge. |

**`abc`**, replace all three name slots with fresh names.

| what it encodes | available? |
|---|---|
| which three slots are names | **no.** requires parsing the task's structure. |
| that replacing all three destroys the duplication the circuit relies on | **no.** this is the mechanism, again stated in advance. |

the honest summary: the *position* a corruption acts on is now searchable, but
*what to replace it with, and why that constitutes a meaningful counterfactual*,
isnt.

## 3. the metric

```
logit_diff = logits[END, IO_token] - logits[END, S_token]
```

this is the piece with the most specific knowledge baked in, and the easiest to
state precisely:

| what it encodes | available? |
|---|---|
| which two vocabulary tokens are the candidate answers | **no.** per-prompt, and derived from the template. |
| which of the two is correct | **no.** this is the answer key for the behaviour. |
| that the comparison should be read at the final token | partly, generic for next-token prediction. |
| that a *difference* of two logits is the right functional form | **no.** presupposes a two-alternative forced choice. |

every normalized number in phases 1-4, every "0.0 = corrupted, 1.0 = clean", is defined against this quantity. if it cant be constructed without the answer
key, then neither can any of those numbers.

---

## What this phase tests, and what it doesnt

ranked by how tractable each dependency looks:

1. **the metric** looks most tractable. A divergence between the clean and
   corrupted *output distributions* needs no knowledge of which token is correct, only the two runs, which the corruption already provides. phase 5 tests two
   such metrics against the hand-built one on IOI, where the answer is known and
   can check the substitution.
2. **the corruption** looks partly tractable. Its position is now searchable. Its
   content isnt. phase 5 tests whether replacing the semantically-chosen
   substitution with a generic random one still yields usable signal. This is
   expected to be worse, and the experiment is worth running mainly to find out
   *how much* worse and in what way.
3. **the task template** doesnt look tractable and is **explicitly out of scope
   for this phase.** constructing a task means selecting a behaviour to study,
   which is a different kind of problem from anything attempted here: the previous
   phases all take a behaviour as given and analyse it, and no amount of better
   patching turns that into behaviour discovery. Attempting a weak version would
   produce something that looked like progress without being any.

A reasonable outcome for this phase is that the metric generalizes, the corruption
partly generalizes, and task construction remains open. that is a finding about
where the remaining difficulty actually sits, which is more useful than a fourth
consecutive recall number.

## Pre-committed definitions

both replacement metrics are defined here, before being run, so neither can be
adjusted after seeing which performs better. both are normalized the same way as
the existing metric, 0 at the corrupted run, 1 at the clean run:

```
recovery = 1 - D(clean, patched) / D(clean, corrupted)
```

With `D` one of:

- **KL divergence** `D_KL(P_clean || P_patched)` over the full next-token
  distribution at the final position.
- **total variation** `½ Σ |P_clean − P_patched|` over the same distribution.

both are computed over the entire 50257-token vocabulary. neither is given the
identity of any answer token.

corruption variants, also fixed here:

- **`random_vocab_s2`**, the token at the S2 position is replaced by a token
  drawn uniformly from the vocabulary. position supplied (and searchable per Phase
  4), content generic.
- **`random_vocab_any`**, a uniformly chosen position is replaced by a uniformly
  drawn token. Nothing supplied.

the comparison to report is against `s2_swap` scored by the hand-built metric,
which is the configuration every earlier phase used.
