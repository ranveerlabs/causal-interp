# Phase 8 plan — a second counterfactual as part of the method, not as a rescue

**Committed before any Phase 8 code exists.** Everything below — the registry design, the
scheme definitions, the decision rules, the flag rule, and the predictions — is fixed
here so that none of it can be chosen after seeing how it scores.

## What Phase 7 left behind

Phase 7 recovered 3 of the 6 published docstring heads and found the reason: the primary
counterfactual (`random_random`) replaces the answer token, so a metric read off that
token cannot see the heads whose job is to *route* attention to the right argument.
A different published counterfactual, `random_def`, keeps the answer and breaks the
pointer, and recovers 5 of 6 — both routing heads included.

That was reported as an explanation rather than a rescue, because `random_def` was fixed
in `PHASE7_PLAN.md` before the run. But it was still a **one-off diagnostic**, invoked
because a human looked at a 50% recall figure and went looking for a reason. Nothing in
the pipeline's output said "this counterfactual may be blind to part of this circuit".

Phase 8 builds that into the method: a task registers **several named counterfactual
schemes**, the pipeline runs discovery under **every** one of them by default, and the
report surfaces **per-head disagreement between schemes** without consulting any answer
key.

## Disclosure — this pre-registration is not blind

Phase 3 made the same declaration and the same narrow claim applies here. Before writing
this plan I inspected results **already committed and already published** in this repo:

- `results/phase7_sweep.json` — per-scheme size-matched and threshold-based head sets for
  all five docstring schemes. The headline of this table (3/6 under `random_random`, 5/6
  under `random_def`) is in `PHASE7_REPORT.md` and in the README.
- `results/phase6_sweep.json` — the same per-scheme sets for greater-than's three
  existing schemes.

So I know, before running anything, that a scheme-disagreement structure applied to the
stored docstring numbers will flag *something*. What I do not know, and what this plan
fixes rules for in advance, is: how large the flagged set is, what it costs in precision,
whether the same structure raises a false alarm on greater-than, whether the newly
authored greater-than scheme has any measurable power at all, and whether the
disagreement structure reproduces one level down at the receiver-specification search.

Predictions P3–P8 below are the ones carrying information. P1 and P2 are marked as
low-information and are recorded anyway so the scoring is complete rather than selective.

## 1. The registry

A new module, `causal_interp/schemes.py`, holds two inert dataclasses and no measurement:

```python
Scheme(name, provenance, breaks, preserves_answer, primary)
TaskSpec(name, dataset, positions, schemes, discovery_schemes, primary_scheme, metric_label)
```

- `provenance` is one of `published` (the paper or its code release defines it),
  `authored` (invented for this project — a hand-built ingredient, and labelled as one),
  `generic` (the Phase 5 vocabulary-substitution code path, which uses no task knowledge).
- `breaks` is one sentence naming which aspect of the prompt the scheme destroys.
- `preserves_answer` records whether the correct answer survives in the corrupted prompt.
  This is the axis Phase 7 showed matters, and it is recorded as *data about the scheme*,
  not inferred from any run.

**The enforcement that makes this non-optional**: `TaskSpec.__post_init__` raises if
`len(discovery_schemes) < 2`. A task cannot be registered with a single counterfactual, so
a future phase cannot get single-scheme discovery by leaving an option unset. It has to
delete the check.

`causal_interp/pipeline.py` exposes `discover(model, task, ...)`, which sweeps every head
at every position under **every** scheme in `discovery_schemes` and returns the per-scheme
effects together with the agreement analysis. There is no single-scheme entry point.

## 2. Scheme registrations, fixed here

### docstring (Phase 7's circuit) — five schemes, none new

| scheme | provenance | breaks | preserves answer |
|---|---|---|---|
| `random_random` (**primary**) | published (benchmark default) | replaces the definition arguments and the docstring arguments | no |
| `random_def` | published | replaces the non-answer definition arguments, so the induction match that selects the answer fails | **yes** |
| `random_answer` | published | replaces the answer argument in the definition | no |
| `random_vocab_cdef` | generic | uniform vocabulary token at the `C_def` anchor | no |
| `random_vocab_any` | generic | uniform vocabulary token anywhere | incidental |

All five already exist in `causal_interp/docstring.py` and Phase 7 already swept all five.
What changes is that the *comparison between them* becomes pipeline output.

### greater-than (Phase 6's circuit) — one scheme authored for this phase

Phase 6 recorded that greater-than has exactly one published counterfactual and that
inventing a second would be hand-tuning of the kind Phase 6 existed to detect. Phase 8's
whole premise is that one counterfactual is not enough, so a second has to be authored —
and it is labelled `authored`, not `published`, everywhere it appears.

`random_def` does **not** transfer as-is. It is defined in terms of "the argument name the
docstring points at", and greater-than has no pointer and no answer *token*: its answer is
a set of year completions defined by the start year `YY`. The analogue has to be built
from the same principle rather than copied:

> the primary scheme changes the value that gets *moved*; the alternate must leave that
> value intact and break the structure that makes the task well posed.

**`xx_mismatch`** (authored):

```
clean       The war lasted from the year 1732 to the year 17
yy01        The war lasted from the year 1701 to the year 17     (published primary)
xx_mismatch The war lasted from the year 1432 to the year 17     (authored alternate)
```

The start year's **century** is replaced with a different century; `YY` is untouched. So:

- the value the year-mover heads carry (`32`) is **identical** between clean and corrupted,
  which is what should make those heads harder to see — the mirror image of `random_def`;
- the range structure that says the completion continues *this* century's span is broken,
  which is what should make the heads carrying that structure visible;
- the metric's `YY` still comes from the clean prompt in every run, exactly as under `yy01`,
  so the three numbers stay comparable;
- token alignment is preserved: one token changes, and the century is drawn only from
  values for which `" {century}{yy}"` still tokenizes as two tokens.

Two other candidates are recorded and rejected here rather than after the fact:
**noun replacement** (breaks nothing about the numeric task — a near-null, not an
alternate), and **inserting a decoy year** (changes the token count, so clean and corrupted
activations no longer live at the same index and patching is not defined).

| scheme | provenance | breaks | preserves answer |
|---|---|---|---|
| `yy01` (**primary**) | published | sets the start year to 01, making the constraint vacuous | no |
| `xx_mismatch` | **authored** | breaks the century correspondence between the two years | **yes** |
| `random_vocab_yy` | generic | uniform vocabulary token at the `YY` anchor | no |
| `random_vocab_any` | generic | uniform vocabulary token anywhere | incidental |

### IOI (Phases 1–5) — registered, not re-run

IOI already had two hand-built schemes (`s2_swap` primary, `abc`) plus the two generic
ones, and Phases 1 and 2 reported them side by side by hand. It is registered so the
registry covers every task in the repo, and it is **not** re-run in this phase; no Phase 8
claim rests on it.

## 3. Decision rules, all inherited

- **"found under scheme s"**: `abs(normalized effect) >= 0.02` at the head's best position,
  under the hand-built metric — Phase 1's headline threshold, unchanged. The KL variant is
  computed in the same sweep and reported beside it, as Phase 5 established.
- Size-matched top-*k* sets are **not** used for the agreement analysis. *k* is the
  published circuit's size, so a top-*k* rule would need the answer key, and the entire
  point is that this analysis runs before the answer key is consulted.
- The agreement module must not import any `ground_truth` module. The runner asserts this
  at startup, the same check `search.py` has had since Phase 4.

## 4. The verdicts — what the pipeline emits when schemes disagree

For every head in the union of all schemes' discovered sets:

| status | meaning |
|---|---|
| `robust` | found under every scheme |
| `scheme-dependent` | found under at least one scheme and missed under at least one |
| — | heads found under no scheme are not in the union and are not listed |

Reported per head with the full presence vector and the per-scheme effect, **never
averaged across schemes and never reported under one scheme's numbers alone.**

Two asymmetric quantities are derived from the same table:

- **blind spot of scheme s** = heads found by some other scheme and missed by `s`. Computed
  for every scheme, not just the primary.
- **the flag**: fires when the *primary* scheme's blind spot is non-empty, with the text
  *"the head list under the primary counterfactual is not the circuit; it is what this
  counterfactual can see"*. **The flag rule is a bare non-emptiness test — no cutoff, no
  tuning.** A false alarm is possible by construction and P4 predicts one.

The blind spot is the deliverable, not the union. The union of all schemes' sets is
reported too, because it is the natural thing a reader will compute, and its precision cost
is reported with it (P3).

## 5. Power is an annotation, never a gate

A scheme with a small clean-vs-corrupted span produces normalized recoveries with a small
denominator and therefore noisy per-head effects. That is measurable:

```
power(s) = span(s) / span(primary)          span = clean metric - corrupted metric
```

`power` is reported for every scheme, and schemes below **0.10** are labelled *low-power*
in every table they appear in. **No scheme is ever excluded on this basis, and the flag
does not depend on it.** Making power a gate would be a free parameter that could be tuned
until the flag fired only where wanted; making it an annotation cannot be.

## 6. Level 2 — the same question at the receiver-specification search

Phase 7 found the blindness repeats one level down: the search ranks `v@C_def` above the
published `q@END` for both argument movers, because the wire carrying the answer outranks
the wire choosing it. So the search is run under every discovery scheme too, and for each
head the report records the **top-ranked specification under each scheme** and flags heads
whose argmax specification differs between schemes as `spec-scheme-dependent`. Again: no
answer key, no threshold, a bare inequality between argmaxes.

## 7. Level 3 — the path-patching chain, per scheme

The iterative chain is run under every discovery scheme, and each scheme's union of
discovered senders enters the same agreement table as a second discovery channel. If a
scheme's chain halts early (Phase 7's shallow-model failure), that is recorded as `halted`
rather than as an empty result.

## 8. Predictions

Scored in `PHASE8_REPORT.md` whichever way they come out.

| # | prediction | information |
|---|---|---|
| P1 | On docstring, the flag fires: the primary scheme's blind spot is non-empty. | **low** — implied by the disclosed Phase 7 numbers |
| P2 | The docstring blind spot contains published heads `1.4` and `2.0`. | **low** — same |
| P3 | The union across docstring schemes has **lower precision** than the primary scheme alone: recall is bought at a precision cost, which is why the flag rather than the union is the deliverable. | high |
| P4 | The flag **also fires on greater-than**, where Phase 6 found no such pathology — so a fired flag is not by itself evidence of a real blind spot, and the per-head detail is load-bearing. | high |
| P5 | No *published* greater-than head appears in the primary scheme's blind spot: the flag fires there on non-circuit heads only. | high |
| P6 | `xx_mismatch` has power between **0.05 and 0.60** of `yy01` — measurably weaker than the published counterfactual, but not degenerate. | high |
| P7 | The receiver-spec search flags the docstring argument movers `3.0` and `3.6` as spec-scheme-dependent, with `q@END` winning under `random_def` and `v@C_def` winning under `random_random`. | high |
| P8 | On docstring, at least one head that is **not** in the published circuit is also flagged scheme-dependent — the flag does not partition cleanly into "real circuit" and "artifact". | high |

## 9. What this phase will not have solved, stated in advance

The schemes still have to be **authored per task**. `xx_mismatch` was designed by a person
reasoning about what greater-than's mechanism must contain — the same category of
hand-built dependency Phase 5 named for the corruption and the metric, and Phase 7 showed
decides which heads exist at all. Phase 8 changes *when* that dependency shows up: instead
of one hand-built counterfactual silently defining the circuit, several hand-built
counterfactuals are forced to disagree in public. It does not remove the dependency, and
the generic schemes — which need no task knowledge and could be registered for any task —
are exactly the ones expected to be low-power.

Nor does a fired flag say **which** scheme is right. It says the answer depends on the
experiment. Choosing between them still needs either the answer key or a further
argument that this phase does not supply.
