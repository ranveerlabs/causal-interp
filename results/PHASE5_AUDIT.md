# Phase 5 — what the task, corruption and metric still require knowing

Phase 4 closed the receiver-specification gap: the search no longer has to be told
which head input, at which position, to interrogate. Its own conclusion named what
remains, and this document takes that seriously enough to itemise it before
testing any of it.

Three pieces are still built by hand. This is an audit of what each one encodes,
written before the experiments in `PHASE5_REPORT.md` were run, so that the
experiments could not quietly redefine what counted as a dependency.

The unit throughout is **what would be unavailable on a circuit nobody has
published** — not "what a human typed", since a human types the code either way.

---

## 1. The task template

```
"Then, {a} and {b} went to the {place}. {s} gave a {obj} to"   -> " Mary"
```

| what it encodes | available on an unfamiliar circuit? |
|---|---|
| That this model performs indirect object identification at all | **No.** Choosing which behaviour to study is prior to everything else here. |
| That the behaviour has two name slots, one of which repeats | **No.** This is the mechanism's shape, stated in advance. |
| That the prompt should stop immediately before the answer token | Partly — "predict the next token" is generic, but *where to cut* is not. |
| That names must be single tokens under this tokenizer | Yes — mechanical, checkable without any knowledge of the task. |
| That both name orders must be balanced to control a positional confound | **No.** Requires knowing that a "second name" heuristic is a competing explanation. |

The first row is the load-bearing one. Everything else in this project is
downstream of a decision that a specific, nameable behaviour exists and is worth
isolating. Nothing in Phases 1–4 discovers behaviours; they all analyse one that
was handed to them.

## 2. The corruption schemes

Both schemes are counterfactuals designed by someone who already knew which token
carried the answer.

**`s2_swap`** — replace the repeated subject with the indirect object, flipping
which name is correct.

| what it encodes | available? |
|---|---|
| Which token position is S2 | Partly — Phase 4 showed positions can be *searched* rather than supplied. |
| That changing that one token flips the answer | **No.** This is knowledge of the task's semantics. |
| That changing exactly one token is desirable, to isolate a single variable | Generic experimental-design knowledge, not task knowledge. |

**`abc`** — replace all three name slots with fresh names.

| what it encodes | available? |
|---|---|
| Which three slots are names | **No.** Requires parsing the task's structure. |
| That replacing all three destroys the duplication the circuit relies on | **No.** This is the mechanism, again stated in advance. |

The honest summary: the *position* a corruption acts on is now searchable, but
*what to replace it with, and why that constitutes a meaningful counterfactual*,
is not.

## 3. The metric

```
logit_diff = logits[END, IO_token] - logits[END, S_token]
```

This is the piece with the most specific knowledge baked in, and the easiest to
state precisely:

| what it encodes | available? |
|---|---|
| Which two vocabulary tokens are the candidate answers | **No.** Per-prompt, and derived from the template. |
| Which of the two is correct | **No.** This is the answer key for the behaviour. |
| That the comparison should be read at the final token | Partly — generic for next-token prediction. |
| That a *difference* of two logits is the right functional form | **No.** Presupposes a two-alternative forced choice. |

Every normalized number in Phases 1–4 — every "0.0 = corrupted, 1.0 = clean" —
is defined against this quantity. If it cannot be constructed without the answer
key, then neither can any of those numbers.

---

## What this phase tests, and what it does not

Ranked by how tractable each dependency looks:

1. **The metric** looks most tractable. A divergence between the clean and
   corrupted *output distributions* needs no knowledge of which token is correct —
   only the two runs, which the corruption already provides. Phase 5 tests two
   such metrics against the hand-built one on IOI, where the answer is known and
   can check the substitution.
2. **The corruption** looks partly tractable. Its position is now searchable; its
   content is not. Phase 5 tests whether replacing the semantically-chosen
   substitution with a generic random one still yields usable signal. This is
   expected to be worse, and the experiment is worth running mainly to find out
   *how much* worse and in what way.
3. **The task template** does not look tractable and is **explicitly out of scope
   for this phase.** Constructing a task means selecting a behaviour to study,
   which is a different kind of problem from anything attempted here: the previous
   phases all take a behaviour as given and analyse it, and no amount of better
   patching turns that into behaviour discovery. Attempting a weak version would
   produce something that looked like progress without being any.

A reasonable outcome for this phase is that the metric generalizes, the corruption
partly generalizes, and task construction remains open. That is a finding about
where the remaining difficulty actually sits, which is more useful than a fourth
consecutive recall number.

## Pre-committed definitions

Both replacement metrics are defined here, before being run, so neither can be
adjusted after seeing which performs better. Both are normalized the same way as
the existing metric — 0 at the corrupted run, 1 at the clean run:

```
recovery = 1 - D(clean, patched) / D(clean, corrupted)
```

with `D` one of:

- **KL divergence** `D_KL(P_clean || P_patched)` over the full next-token
  distribution at the final position.
- **Total variation** `½ Σ |P_clean − P_patched|` over the same distribution.

Both are computed over the entire 50257-token vocabulary. Neither is given the
identity of any answer token.

Corruption variants, also fixed here:

- **`random_vocab_s2`** — the token at the S2 position is replaced by a token
  drawn uniformly from the vocabulary. Position supplied (and searchable per Phase
  4), content generic.
- **`random_vocab_any`** — a uniformly chosen position is replaced by a uniformly
  drawn token. Nothing supplied.

The comparison to report is against `s2_swap` scored by the hand-built metric,
which is the configuration every earlier phase used.
