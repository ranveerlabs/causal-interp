# Phase 7 — the same pipeline, a different model

**Target**: the Python docstring circuit in `attn-only-4l`, from Heimersheim and
Janiak (2023),
[*A circuit for Python docstrings in a 4-layer attention-only transformer*](https://www.lesswrong.com/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only).
Published ground truth: **6 attention heads** in
4 classes, cross-checked against the 37-edge manual graph in the
ACDC release.

Every phase before this one ran inside GPT-2 small. The target, the ground truth,
the scoring rules, an advance audit of what in the code was GPT-2-shaped, and seven
predictions were fixed in [PHASE7_PLAN.md](PHASE7_PLAN.md), committed before any
Phase 7 code existed.

| | |
|---|---|
| model | `attn-only-4l` (`Attn_Only_4L512W_C4_Code`) |
| shape | 4 layers × 8 heads = 32 heads, `d_model` 512, **`attn_only=True`** |
| tokenizer | `NeelNanda/gpt-neox-tokenizer-digits`, `d_vocab` 48262 |
| prompts | 128 per corruption scheme, seed 0 |
| activation-patching cutoff | 0.02 — **inherited from Phase 1** |
| size-matched set | top 6 — the published head count |
| receiver-side threshold | 0.03 — **Phase 3's rule, recalibrated null** |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| runtime | 103s sweep + 310s search |

For comparison, GPT-2 small is 12 × 12 = 144 heads, `d_model` 768, with an MLP in
every block and a 50257-token vocabulary. Nothing about the token indices, the
position layout or the depth of the model carries over from Phases 1–6.


## Headline

| method | at 0.02 | size-matched | precision |
|---|---|---|---|
| activation patching, hand-built metric | 3/6 | 3/6 | 0.33 |
| activation patching, KL (no answer key) | 4/6 | 3/6 | 0.50 |
| path patching, all rounds | 3/6 | — | 0.50 |

**Recovery is worse here than on either GPT-2 small circuit**: 50% of the published heads, against IOI 69%, greater-than 100%. That is a real drop and the sections below locate it — three of the six heads are missed, and they are missed for a reason this phase can name.

For the hand-built metric the threshold-based and size-matched columns agree
exactly, so nothing turns on which is called the headline. They differ by one head
for KL, which the cutoff finds and the size-matched cut does not.


## The seven predictions, scored

Fixed in [PHASE7_PLAN.md](PHASE7_PLAN.md) before the run. **3 held,
3 were wrong**, and one is split — the largest share of failed predictions of
any phase so far, which is the most useful thing this phase produced.

| prediction | outcome | measured |
|---|---|---|
| 1. no blindness under the primary scheme, full blocks under the anchored one | ✅ held | `random_random`: no position is a full block (largest 8/32); `random_vocab_cdef`: `A_def` 32/32, `B_def` 32/32, `comma_B` 32/32 |
| 2. activation patching recovers ≥ 4 of 6 | ✗ **wrong** | recovered 3/6; missed `1.2`, `1.4`, `2.0` |
| 3. the chain recovers the published wiring order | ✗ **wrong** | 2 of 3 rounds ran; 3/6 across the chain — the chain halted after reaching layer 0 |
| 4. precision worse than both earlier circuits | ◐ **split** | precision 0.33 against IOI 0.78, greater-than 0.78; but only 9 of 32 heads (28%) clear the cutoff, not the "more than half" the plan predicted |
| 5. fully generic recovers less | ✗ **wrong** | size-matched 4/6 generic vs 3/6 hand-built |
| 6. `mlp_out` sweep returns exact zeros rather than raising | ✅ held | all 28 cells exactly 0.0 |
| 7. the auxiliary heads `0.2`, `0.4` are not recovered | ✅ held | `0.2` +0.0030, `0.4` +0.0011; neither in the top set |

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


## Why three heads are missed: what a counterfactual can and cannot see

The three published counterfactuals disagree about which heads exist, and the
disagreement is systematic rather than noisy.

| corruption | what it breaks | corrupted | recovered | which |
|---|---|---|---|---|
| `random_random` | the benchmark's default counterfactual | -5.814 | 3/6 | `0.5`, `3.0`, `3.6` |
| `random_def` | published counterfactual: break the induction match, keep the answer | -1.947 | 5/6 | `0.5`, `1.4`, `2.0`, `3.0`, `3.6` |
| `random_answer` | published counterfactual: replace the answer itself | -6.837 | 3/6 | `0.5`, `3.0`, `3.6` |

`random_def` recovers **5/6**, adding
`1.4`, `2.0` — and those are exactly the heads
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


## Components, and a silent failure mode found on purpose

`sweep_component` was inherited with `COMPONENT_KINDS` unchanged, `mlp_out`
included, even though this model has none. The plan predicted what would happen and
it happened: **all 28 `mlp_out` cells are exactly 0.0**. The hook exists,
`run_with_hooks` accepts it, the hook never fires, the patch never happens, and the
sweep returns a clean grid that looks like a measurement of an absence rather than
an absence of measurement.

Nothing raised. On a model whose architecture was not already known, that grid
would have been reported as "the MLPs carry no causal signal".

| component | layer | largest effect | at position |
|---|---|---|---|
| `resid_pre` | 0 | +0.595 | `C_def` |
| `resid_pre` | 1 | +0.597 | `C_def` |
| `resid_pre` | 2 | +0.592 | `C_def` |
| `resid_pre` | 3 | +0.507 | `C_def` |
| `attn_out` | 0 | +0.187 | `C_def` |
| `attn_out` | 1 | -0.018 | `B_doc` |
| `attn_out` | 2 | +0.132 | `END` |
| `attn_out` | 3 | +0.986 | `END` |
| `mlp_out` | 0 | +0.000 | — |
| `mlp_out` | 1 | +0.000 | — |
| `mlp_out` | 2 | +0.000 | — |
| `mlp_out` | 3 | +0.000 | — |

`attn_out[3]` at `END` recovers **+0.986** on its own — in
a four-layer attention-only model, almost the entire behaviour is written by the
final attention layer at the final token.


## Path patching — the chain, and where it stopped

Phase 2's iterative chain. Each round's receivers are the heads discovered in the
round before; the answer key is never consulted to choose them. The receiver input
and position come from the published account of the mechanism, exactly as Phase 2's
came from the IOI paper's — this is guided rediscovery, and *which* heads turn up is
not constrained.

| round | question | top senders | expected | published class |
|---|---|---|---|---|
| 0 | which heads move the prediction without another head relaying it? | `3.0` +0.588, `3.6` +0.551, `2.3` +0.115, `3.5` +0.032 | argument mover (3.0, 3.6) | argument mover |
| 1 | which heads feed those heads' keys at C_def? | `0.5` -0.029, `1.2` -0.005, `1.5` +0.003, `0.3` +0.002 | fuzzy previous token (2.0), duplicate token (1.2) | duplicate token |
| 2 | which heads feed *those* heads' values at the comma before C_def? | *chain halted* | induction + fuzzy previous token (1.4) | — |

Union across rounds: **3/6**, precision
0.50. Round 0 returned the two argument movers as its top two out
of all 32 heads, unprompted.

Round 1 asked what feeds their keys at `C_def`. Its top senders are `0.5` and `1.2`:
`1.2` is one of the two heads the released graph names as feeding those keys, and
`0.5` is the head the graph names as feeding `1.2`'s own query and key. **`2.0`, the
other named key-input, does not appear** — the same miss activation patching made,
for the same reason the next-but-one section gives. The magnitudes are also small
(`0.5` at −0.029, `1.2` at −0.005 against round 0's +0.588), and only one sender
clears the inherited cutoff.

### The chain ran out of model

Round 2 could not run. Its receivers would have been round
1's top senders — `0.5`, `1.2`, `1.5`, `0.3` — and a
sender must sit strictly below its receiver, so with a layer-0 head among them
there are no eligible senders left.

This is not a defect of the rule and not a tuning problem: **the chain descended to
layer 0 in one round and there is nothing below layer 0**. Phase 2's chain had
twelve layers to walk down and used four rounds; here it exhausted a four-layer
model in two. The pre-registered four-round ladder
(`3.x ← {2.0, 1.2} ← 1.4 ← 0.5`) was therefore never fully testable, and
prediction 3 is scored as failed rather than excused.

The honest reading: **an iterative chain's reach is bounded by the model's depth**,
and that bound is invisible until the pipeline meets a shallow model. It is the
clearest instance in seven phases of a method assumption that was silently a GPT-2
small assumption.


## The receiver-side criterion, at a threshold recalibrated a third time

Phase 3's rule, unchanged:

> threshold = 99th percentile of `|path_signal|` under a shuffled-source null,
> rounded up to two significant figures

Recalibrated on this model's null and committed in
`results/phase7_preregistration.json` before this comparison existed. Three
recalibrations of one rule now exist: **0.11** on IOI, **0.046** on greater-than,
**0.03** here. No two agree, which is the whole argument for
inheriting the rule and not the number.

| group | position | receivers | senders scored | cleared | which |
|---|---|---|---|---|---|
| round 1 | C_def | 4 | 16 | 1 | `0.5` |

Scored against the published circuit: **1/6**,
precision 1.00. As in Phases 3 and 6 this is a *different
definition of found*, reported beside the logit-based numbers rather than merged
into them.

### The null is thin, and that is a finding about the rule

The threshold was pooled from **16 null measurements**,
against Phase 3's hundreds. The chain halts after one usable receiver group, and
that group's eligible senders are the 16 heads below
it — so a 99th-percentile rule is being asked to pick a tail from a sample that
barely has one. The rule was applied unchanged because changing it after seeing the
sample size would be exactly the move the pre-registration exists to prevent, but
**a percentile null does not survive contact with a shallow model**, and that is
worth more than the number it produced.

Only 3 of the 6 published heads
(`0.5`, `1.2`, `1.4`) were ever eligible as senders; the
other 3 (`2.0`, `3.0`, `3.6`)
occupy the *receiver* slot and are unmeasured, not measured-and-failed — the
distinction Phase 3 drew for `9.0` and `11.9`. Of the in-scope ones,
1 cleared the bar: `0.5`.


## Searching for the receiver specifications

Phase 4's exhaustive screen, unchanged. `search.py` does not import any
ground-truth module, and the run asserts that before starting — the check widened
in Phase 6 to cover any `ground_truth*` module needed no further change for a third
circuit, which is what widening it was for.

Four of the six heads have a published `(input, position)`; `0.5` and `1.2` do not,
and were declared unscoreable in the plan rather than scored against a guess.

| outcome | count |  |
|---|---|---|
| disagreement | 4 | ✗ |
| no published spec | 2 | — |

| head | class | published | rank | search's own top spec | outcome |
|---|---|---|---|---|---|
| `0.5` | duplicate token | — | — | — | no published spec |
| `1.2` | duplicate token | — | — | — | no published spec |
| `1.4` | induction + fuzzy previous token | `v@B_def` | 13 | `v@B_doc` -0.018 | disagreement |
| `2.0` | fuzzy previous token | `v@comma_B` | 16 | `q@B_def` +0.024 | disagreement |
| `3.0` | argument mover | `q@END` | 4 | `v@C_def` +0.210 | disagreement |
| `3.6` | argument mover | `q@END` | 5 | `v@C_def` +0.248 | disagreement |

**The search recovered none of the 4 primary specifications it could weigh** —
the worst rediscovery result in the project, against 16 of 17 on IOI and 7 of 7 on
greater-than.

It is the same failure as the head-level one, one level down. For both argument
movers the search's top pick is `v@C_def` — the *value* they read at the definition
argument — where the plan's primary spec was `q@END`, the query that decides which
argument to read. Under `random_random` the token at `C_def` is replaced, so
splicing the clean value there restores the behaviour and splicing the clean query
does not. The screen is a logit-effect screen; it ranked the wire that carries the
answer above the wire that chooses it, because under this counterfactual that is
the true causal ordering.

**And `v@C_def` is itself a published input for those heads** — one of the
alternatives the plan fixed in advance precisely so this could be reported without
it being a rescue. On the alternative specs the search ranks the published wire
**first of all 21** for both movers. The headline stays 0; the alternative line is
the informative one, and it says the search found a real published edge rather than
nothing.

### The alternative published specs, declared in advance

This circuit has more than one published input per class, so the plan fixed both
the primary spec and the alternatives before the run — precisely so a poor primary
result could not be rescued afterwards by adopting a better one. It is reported
here as a separate line and is **not** merged into the count above.

| head | primary spec | rank | alternative | rank | score |
|---|---|---|---|---|---|
| `1.4` | `v@B_def` | 13 | `k@B_doc` | 2 | +0.006 |
| `3.0` | `q@END` | 4 | `k@C_def` | 12 | +0.000 |
| `3.0` | `q@END` | 4 | `v@C_def` | 1 | +0.210 |
| `3.6` | `q@END` | 5 | `k@C_def` | 6 | +0.005 |
| `3.6` | `q@END` | 5 | `v@C_def` | 1 | +0.248 |

### Do the position labels matter?

The unlabelled screen scores bare token indices `t0…t40` with no semantic meaning
attached. Labels are attached *after* the search, purely to read its output.

| screen | top positions within its own top 50 |
|---|---|
| semantic | `A_def` ×14, `C_def` ×13, `B_def` ×13, `B_doc` ×5, `END` ×3, `A_doc` ×2 |
| absolute | `t11` ×14 (= A_def), `t13` ×13 (= B_def), `t15` ×12 (= C_def), `t34` ×4 (= B_doc), `t40` ×3 (= END), `t27` ×2 (= A_doc) |


## All three circuits side by side

| circuit | model | heads | published | recovered | recall | precision | chance recall |
|---|---|---|---|---|---|---|---|
| IOI (Wang et al.) | GPT-2 small | 12 × 12 = 144 | 26 | 18/26 | 69% | 0.78 | 18% |
| greater-than (Hanna et al.) | GPT-2 small | 12 × 12 = 144 | 7 | 7/7 | 100% | 0.78 | 5% |
| **docstring (Heimersheim & Janiak)** | **attn-only-4l** | **4 × 8 = 32** | **6** | **3/6** | **50%** | **0.33** | **19%** |

The last column is the deflation the plan fixed in advance, and it turns out to cut
less than the plan expected. Chance recall — the recall a size-matched set drawn at
random would get — is 19% here, 18% for IOI and
5% for greater-than. So the plan's warning applies only against
greater-than, whose 100% sits on a denominator four times harder than this one.
**Against IOI the denominators are effectively identical**, which means the drop
from 69% to
50% is a real drop and not an artifact of circuit size. Every
measurement here clears chance comfortably; none of them clears it by as much as
either earlier circuit did.


## What actually transferred — the measure of generality

Phase 6 asked this question of a second task in the same model and answered "the
causal core, untouched; one shared module needed a parameter". Phase 7 asks it of a
different model, and the answer is measured rather than asserted — this report runs
the command itself and pastes what it got:

```
$ git diff --stat 90a77c9..HEAD -- causal_interp/*.py   # the ten modules that predate Phase 7
(no output)
```

**No existing file was modified at all.** Not the causal core, not the scoring module, not the loader, not either earlier task. Phase 7 is entirely additive.

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


## What this phase does and does not show

**The code transferred; the results got worse.** Those are two findings and
collapsing them into one would misreport both.

The pipeline ran against a model with a quarter the depth, a different tokenizer, a
different training corpus and **no MLP blocks**, with not one line of the existing
library changed. Round 0 of the path chain returned the two published argument
movers as its top two of 32 heads, unprompted; greedy narrowing reached 0.995
recovery with four nodes, three of them published; and the answer-key-free KL metric
held up a third time, finding
4/6 at the inherited cutoff
against the hand-built metric's
3/6 (they tie at
3/6 size-matched), on a
third task whose hand-built metric resembles neither of the first two.

And recovery is **50%** — the lowest of the three circuits, on the most
forgiving denominator of the three. Three of the seven pre-registered predictions
were wrong.

**What went wrong is legible, which is the part that matters.** The three missed
heads are the ones that route attention rather than move content, and the primary
counterfactual replaces the answer token, which makes routing causally invisible to
a metric read off the output. A different published counterfactual — `random_def`,
which breaks the pointer and leaves the answer in place — recovers
5/6 including both routing heads. The
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
