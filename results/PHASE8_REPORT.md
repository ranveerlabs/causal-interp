# Phase 8 — a second counterfactual as part of the method, not as a rescue

**What changed is the pipeline's output, not its recall.** The design, the scheme registrations, the decision rules and eight predictions were fixed in [PHASE8_PLAN.md](PHASE8_PLAN.md), committed before any Phase 8 code existed.

Phase 7 recovered 3 of the 6 published docstring heads under the benchmark's default counterfactual and 5 of 6 under a different published one, and the reason was structural: the primary counterfactual replaces the answer token, so the heads whose job is to *route* attention to the right argument cannot move a metric read off that token. That diagnosis was correct and it was made by a human who saw a low recall number and knew which alternative to try. Nothing in the pipeline's own output said the head list was incomplete.

Phase 8 makes multi-scheme discovery structural:

- `causal_interp/schemes.py` — a task registers **named** counterfactual schemes, each declaring what it breaks and whether the answer survives it. `TaskSpec` **raises** if a task registers fewer than two: single-counterfactual discovery is not an option a later phase can leave unset, it is a check someone would have to delete.
- `causal_interp/pipeline.py` — `discover()` sweeps every registered scheme and returns the head list and the cross-scheme comparison *in the same object*. There is no single-scheme entry point.
- `causal_interp/agreement.py` — the per-head verdicts, the per-scheme blind spots and the flag. It imports no `ground_truth` module and the runner asserts that at startup, alongside the same check `search.py` has carried since Phase 4.

The flag is a bare non-emptiness test with no cutoff, so this phase adds no free parameter:

> **the primary scheme's blind spot** — heads some other registered counterfactual finds and the primary one does not.

Both earlier circuits were re-run through it. Phase 6's and Phase 7's own results files are untouched; these are new runs under the new structure.

| circuit re-run | schemes | prompts | runtime | flag |
|---|---|---|---|---|
| docstring (Phase 7's circuit, `attn-only-4l`) | 5 | 128 | 5 min | **fires** |
| greater-than (Phase 6's circuit, GPT-2 small) | 4 | 128 | 17 min | **fires** |

## docstring (Phase 7's circuit, `attn-only-4l`)

4 layers x 8 heads, 128 prompts, threshold 0.02 (Phase 1's, inherited). Every scheme below was swept in full; none was chosen after the fact.

| scheme | provenance | what it breaks | answer preserved | span | power | heads found |
|---|---|---|---|---|---|---|
| `random_random` **primary** | published | replaces the definition arguments and the docstring arguments | no | +6.447 | 1.00 | 9 |
| `random_def` | published | replaces the non-answer definition arguments, breaking the induction match that selects the answer | **yes** | +2.580 | 0.40 | 17 |
| `random_answer` | published | replaces the answer argument in the definition | no | +7.470 | 1.16 | 11 |
| `random_vocab_cdef` | generic | substitutes a uniformly drawn vocabulary token at the C_def anchor | no | +6.360 | 0.99 | 10 |
| `random_vocab_any` | generic | substitutes a uniformly drawn vocabulary token anywhere in the prompt | no | +0.366 | 0.06 *(low-power)* | 25 |

`power` is the scheme's clean-vs-corrupted span relative to the primary's. It is **an annotation and never a gate** — no code path drops a scheme for being low-power, because a power cutoff would be exactly the kind of free parameter that could be tuned until the flag fired only where it was wanted.

### What the pipeline says before the answer key is opened

- heads found by **every** scheme (`robust`): **6**
- heads found by some scheme and missed by others (`scheme-dependent`): **20**
- union across schemes: 26   ·   intersection: 6

> COUNTERFACTUAL-SCHEME-DEPENDENT: 17 head(s) [0.2, 0.4, 0.6, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0, 2.1, 2.4, 2.5, 2.6, 2.7, 3.2, 3.4] are found under another counterfactual and missed under the primary one (random_random). The head list under the primary counterfactual is not the circuit; it is what this counterfactual can see.

Every scheme's blind spot is computed, not just the primary's — the question "what can this experiment not see" is asked of each of them:

| scheme | found | blind spot (found by others, not by it) | found by it alone |
|---|---|---|---|
| `random_random` | 9 | 17 | 0 |
| `random_def` | 17 | 9 | 0 |
| `random_answer` | 11 | 15 | 0 |
| `random_vocab_cdef` | 10 | 16 | 0 |
| `random_vocab_any` | 25 | 1 | 7 |

### And now the answer key

| scheme | discovered | published heads found | recall | precision |
|---|---|---|---|---|
| `random_random` | 9 | 3/6 | 0.50 | 0.33 |
| `random_def` | 17 | 5/6 | 0.83 | 0.29 |
| `random_answer` | 11 | 3/6 | 0.50 | 0.27 |
| `random_vocab_cdef` | 10 | 4/6 | 0.67 | 0.40 |
| `random_vocab_any` | 25 | 6/6 | 1.00 | 0.24 |
| **union of all schemes** | 26 | **6/6** | 1.00 | **0.23** |

Per published head, the verdict the pipeline had already reached before this section was reachable:

| head | published class | verdict | found under |
|---|---|---|---|
| `0.5` | duplicate token | robust | `random_random`, `random_def`, `random_answer`, `random_vocab_cdef`, `random_vocab_any` |
| `1.2` | duplicate token | scheme-dependent | `random_vocab_cdef`, `random_vocab_any` |
| `1.4` | induction + fuzzy previous token | scheme-dependent | `random_def`, `random_vocab_any` |
| `2.0` | fuzzy previous token | scheme-dependent | `random_def`, `random_vocab_any` |
| `3.0` | argument mover | robust | `random_random`, `random_def`, `random_answer`, `random_vocab_cdef`, `random_vocab_any` |
| `3.6` | argument mover | robust | `random_random`, `random_def`, `random_answer`, `random_vocab_cdef`, `random_vocab_any` |

Of the 17 heads in the primary scheme's blind spot, **3 are published circuit members** (`1.2`, `1.4`, `2.0`) and 14 are not. Recall goes from 3/6 under the primary scheme to 6/6 across the union, at a precision cost of 0.33 → 0.23.

### The same comparison at the other two channels

**Path-patching chain.** Run once per scheme, each round's receivers taken from the round before, exactly as in Phases 2, 6 and 7.

| scheme | senders found | blind spot | chain |
|---|---|---|---|
| `random_random` | 6 | 12 | halted early |
| `random_def` | 7 | 11 | halted early |
| `random_answer` | 6 | 12 | halted early |
| `random_vocab_cdef` | 6 | 12 | halted early |
| `random_vocab_any` | 14 | 4 | halted early |

> COUNTERFACTUAL-SCHEME-DEPENDENT: 12 head(s) [0.1, 0.4, 0.7, 1.1, 1.2, 2.1, 2.2, 2.4, 2.6, 2.7, 3.2, 3.4] are found under another counterfactual and missed under the primary one (random_random). The head list under the primary counterfactual is not the circuit; it is what this counterfactual can see.

One caveat the chain's numbers carry and the head sweep's do not: a scheme whose chain **halts early** never measures the senders the later rounds would have reached, so those heads count as "not found" here rather than "measured and below threshold". Where a chain halted it is marked above, and its blind spot is inflated by that alone.

Scored against the published circuit, the chain recovers 3/6 under the primary scheme and 4/6 across the union.

**Receiver-specification search.** Phase 4's screen, run once per scheme. A head is `spec-scheme-dependent` when the specification that wins its own ranking changes with the counterfactual — a bare inequality between argmaxes, fixed in the plan.

- heads whose winning specification changes across schemes: **32 of 32**
- of the 4 published heads with a published receiver specification, the primary scheme's search ranks it first for **0**, and *some* scheme ranks it first for **2**

**The argmax rule flags 32 of 32 heads, which is close to everything, and that is a finding about the rule rather than about the circuit.** A head's best receiver specification is not a stable quantity across counterfactuals — most heads have no strong specification at all, so their argmax is decided by noise. The pre-registered rule was a bare inequality with no threshold, it is reported as it was written, and it is much weaker than the head-level flag. The informative part of this channel is the line above it: which *published* specifications any scheme recovers.

| head | class | published spec | `random_random` | `random_def` | `random_answer` | `random_vocab_cdef` | `random_vocab_any` |
|---|---|---|---|---|---|---|---|
| `1.4` | induction + fuzzy previous token | `v@B_def` | `v@B_doc` | **`v@B_def`** | `v@C_def` | `k@C_def` | `k@B_doc` |
| `2.0` | fuzzy previous token | `v@comma_B` | `q@B_def` | **`v@comma_B`** | `q@C_def` | `k@C_def` | **`v@comma_B`** |
| `3.0` | argument mover | `q@END` | `v@C_def` | `k@C_def` | `v@C_def` | `v@C_def` | `k@C_def` |
| `3.6` | argument mover | `q@END` | `v@C_def` | `k@C_def` | `v@C_def` | `v@C_def` | `k@C_def` |

Bold marks a scheme whose search puts the published specification first.

For the 3 heads whose sources also publish *alternative* inputs, the winning specification under at least one scheme is one of those published alternatives in **3** case(s). The search is not landing on unrelated wires; across the schemes it moves between published inputs of the same head, and which one wins is decided by the counterfactual.

## greater-than (Phase 6's circuit, GPT-2 small)

12 layers x 12 heads, 128 prompts, threshold 0.02 (Phase 1's, inherited). Every scheme below was swept in full; none was chosen after the fact.

| scheme | provenance | what it breaks | answer preserved | span | power | heads found |
|---|---|---|---|---|---|---|
| `yy01` **primary** | published | sets the start year to 01, making the greater-than constraint vacuous | no | +1.377 | 1.00 | 9 |
| `xx_mismatch` | authored | replaces the start year's century, breaking the correspondence between the two years while leaving YY untouched | **yes** | +1.019 | 0.74 | 15 |
| `random_vocab_yy` | generic | substitutes a uniformly drawn vocabulary token at the YY anchor | no | +0.834 | 0.61 | 3 |
| `random_vocab_any` | generic | substitutes a uniformly drawn vocabulary token anywhere in the prompt | no | +0.371 | 0.27 | 10 |

`power` is the scheme's clean-vs-corrupted span relative to the primary's. It is **an annotation and never a gate** — no code path drops a scheme for being low-power, because a power cutoff would be exactly the kind of free parameter that could be tuned until the flag fired only where it was wanted.

### What the pipeline says before the answer key is opened

- heads found by **every** scheme (`robust`): **0**
- heads found by some scheme and missed by others (`scheme-dependent`): **25**
- union across schemes: 25   ·   intersection: 0

> COUNTERFACTUAL-SCHEME-DEPENDENT: 16 head(s) [0.10, 1.5, 3.0, 3.3, 4.4, 4.7, 4.10, 4.11, 5.0, 5.6, 6.8, 6.11, 7.0, 7.6, 8.5, 8.6] are found under another counterfactual and missed under the primary one (yy01). The head list under the primary counterfactual is not the circuit; it is what this counterfactual can see.

Every scheme's blind spot is computed, not just the primary's — the question "what can this experiment not see" is asked of each of them:

| scheme | found | blind spot (found by others, not by it) | found by it alone |
|---|---|---|---|
| `yy01` | 9 | 16 | 2 |
| `xx_mismatch` | 15 | 10 | 6 |
| `random_vocab_yy` | 3 | 22 | 1 |
| `random_vocab_any` | 10 | 15 | 6 |

### And now the answer key

| scheme | discovered | published heads found | recall | precision |
|---|---|---|---|---|
| `yy01` | 9 | 7/7 | 1.00 | 0.78 |
| `xx_mismatch` | 15 | 5/7 | 0.71 | 0.33 |
| `random_vocab_yy` | 3 | 1/7 | 0.14 | 0.33 |
| `random_vocab_any` | 10 | 1/7 | 0.14 | 0.10 |
| **union of all schemes** | 25 | **7/7** | 1.00 | **0.28** |

Per published head, the verdict the pipeline had already reached before this section was reachable:

| head | published class | verdict | found under |
|---|---|---|---|
| `5.1` | year head -> MLP 8 | scheme-dependent | `yy01` |
| `5.5` | year head -> MLP 8 | scheme-dependent | `yy01`, `xx_mismatch` |
| `6.9` | year head -> MLP 8 | scheme-dependent | `yy01`, `xx_mismatch`, `random_vocab_yy` |
| `7.10` | year head -> MLP 8 | scheme-dependent | `yy01`, `xx_mismatch` |
| `8.8` | year head -> MLP 8 | scheme-dependent | `yy01` |
| `8.11` | year head -> MLP 8 | scheme-dependent | `yy01`, `xx_mismatch` |
| `9.1` | year head -> MLP 9 | scheme-dependent | `yy01`, `xx_mismatch`, `random_vocab_any` |

Of the 16 heads in the primary scheme's blind spot, **0 are published circuit members** (—) and 16 are not. Recall goes from 7/7 under the primary scheme to 7/7 across the union, at a precision cost of 0.78 → 0.28.

### The same comparison at the other two channels

**Path-patching chain.** Run once per scheme, each round's receivers taken from the round before, exactly as in Phases 2, 6 and 7.

| scheme | senders found | blind spot | chain |
|---|---|---|---|
| `yy01` | 9 | 11 | halted early |
| `xx_mismatch` | 12 | 8 | halted early |
| `random_vocab_yy` | 3 | 17 | ran to the last round |
| `random_vocab_any` | 7 | 13 | halted early |

> COUNTERFACTUAL-SCHEME-DEPENDENT: 11 head(s) [0.10, 1.5, 4.4, 4.10, 4.11, 5.0, 6.8, 6.11, 7.6, 8.5, 8.6] are found under another counterfactual and missed under the primary one (yy01). The head list under the primary counterfactual is not the circuit; it is what this counterfactual can see.

One caveat the chain's numbers carry and the head sweep's do not: a scheme whose chain **halts early** never measures the senders the later rounds would have reached, so those heads count as "not found" here rather than "measured and below threshold". Where a chain halted it is marked above, and its blind spot is inflated by that alone.

Scored against the published circuit, the chain recovers 7/7 under the primary scheme and 7/7 across the union.

**Receiver-specification search.** Phase 4's screen, run once per scheme. A head is `spec-scheme-dependent` when the specification that wins its own ranking changes with the counterfactual — a bare inequality between argmaxes, fixed in the plan.

- heads whose winning specification changes across schemes: **137 of 144**
- of the 7 published heads with a published receiver specification, the primary scheme's search ranks it first for **7**, and *some* scheme ranks it first for **7**

**The argmax rule flags 137 of 144 heads, which is close to everything, and that is a finding about the rule rather than about the circuit.** A head's best receiver specification is not a stable quantity across counterfactuals — most heads have no strong specification at all, so their argmax is decided by noise. The pre-registered rule was a bare inequality with no threshold, it is reported as it was written, and it is much weaker than the head-level flag. The informative part of this channel is the line above it: which *published* specifications any scheme recovers.

| head | class | published spec | `yy01` | `xx_mismatch` | `random_vocab_yy` | `random_vocab_any` |
|---|---|---|---|---|---|---|
| `5.1` | year head -> MLP 8 | `v@YY` | **`v@YY`** | `k@YY` | **`v@YY`** | `k@YY` |
| `5.5` | year head -> MLP 8 | `v@YY` | **`v@YY`** | `k@YY` | **`v@YY`** | `k@YY` |
| `6.9` | year head -> MLP 8 | `v@YY` | **`v@YY`** | `k@YY` | **`v@YY`** | `k@YY` |
| `7.10` | year head -> MLP 8 | `v@YY` | **`v@YY`** | `q@END` | **`v@YY`** | `q@END` |
| `8.8` | year head -> MLP 8 | `v@YY` | **`v@YY`** | `q@END` | **`v@YY`** | `q@END` |
| `8.11` | year head -> MLP 8 | `v@YY` | **`v@YY`** | `q@END` | `q@END` | **`v@YY`** |
| `9.1` | year head -> MLP 9 | `v@YY` | **`v@YY`** | `q@END` | `k@YY` | `k@YY` |

Bold marks a scheme whose search puts the published specification first.

## Reproduction check

The primary scheme's sweep is the same experiment Phases 6 and 7 ran. Its result is read back out of their committed files and compared, so a change in the machinery cannot silently move the baseline this phase is measured against:

| circuit | earlier phase | this phase | primary scheme's head set |
|---|---|---|---|
| docstring (Phase 7's circuit, `attn-only-4l`) | Phase 7: 3 published, 9 discovered | Phase 8: 3 published, 9 discovered | **identical** |
| greater-than (Phase 6's circuit, GPT-2 small) | Phase 6: 7 published, 9 discovered | Phase 8: 7 published, 9 discovered | **identical** |

## Does the structure catch Phase 7's blindness without knowing to look?

This is the phase's actual question. Phase 7's diagnosis was correct and was reached by consulting the answer key. The test here is whether the same conclusion is available from the pipeline's output alone.

**It is — and the flag on its own is not enough to act on.** Running the registered schemes and comparing them — no answer key, no human noticing anything — the pipeline emits:

```
COUNTERFACTUAL-SCHEME-DEPENDENT: 17 head(s) [0.2, 0.4, 0.6, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0, 2.1, 2.4, 2.5, 2.6, 2.7, 3.2, 3.4] are found under another counterfactual and missed under the primary one (random_random). The head list under the primary counterfactual is not the circuit; it is what this counterfactual can see.
```

Opening the published circuit afterwards, 3 of the 17 flagged heads are published circuit members (`1.2`, `1.4`, `2.0`), and the primary scheme's recall of 3/6 rises to 6/6 across the union. Phase 7 reported the first number as the result and needed the answer key to learn it was not the circuit.

**Three things stop this from being a clean success**, and they matter more than the headline:

1. **The flag is loud.** 17 heads are flagged and only 3 of them are published. The flag says *the answer depends on the experiment*; it does not say which heads are real, and a reader who treated the flagged set as a circuit would be badly wrong. Precision across the union is 0.23 against 0.33 for the primary scheme alone.
2. **Most of the noise comes from the schemes that need no task knowledge.** Restricting to the hand-built schemes (`random_random`, `random_def`, `random_answer`) the blind spot is 9 heads, of which 2 are published. That is a derived cut of the same table, reported after the fact and **not** part of the flag rule — the pre-registered rule is the one quoted above, and it is what the number in the headline uses.
3. **A fired flag is not evidence of a real blind spot.** It fired on greater-than too, where it was pointing at nothing — the next section is the comparison that makes that concrete, and it is the more useful half of the phase.

## The two circuits side by side — and why the flag is only half an answer

The flag fired on **both** circuits. What differs is entirely in the detail underneath it, and none of that detail is available without the answer key:

| circuit | flag | heads flagged | of which published | recall, primary → union | precision, primary → union |
|---|---|---|---|---|---|
| docstring (Phase 7's circuit, `attn-only-4l`) | **fires** | 17 | **3** | 3/6 → 6/6 | 0.33 → 0.23 |
| greater-than (Phase 6's circuit, GPT-2 small) | **fires** | 16 | **0** | 7/7 → 7/7 | 0.78 → 0.28 |

On docstring the flagged set contains the exact heads Phase 7 missed and recall rises from 3/6 to 6/6. On greater-than it contains no circuit member at all: the primary counterfactual already recovered 7/7 and the other schemes add nothing but false positives, dropping precision from 0.78 to 0.28. Phase 6's finding that this circuit has no counterfactual-blindness problem survives the check, which is the outcome that would have falsified the structure had it come out otherwise.

**The uncomfortable half.** The two flags are identical in form — same wording, same kind of head list, 17 heads against 16. Nothing the pipeline computes before the answer key opens separates *"your counterfactual is hiding half the circuit"* from *"your counterfactual is fine and the others are noisier"*. Phase 7's failure was reporting one number as the circuit; Phase 8's remaining failure is reporting a warning that cannot be graded. It moves the question from **"is this the circuit?"**, which the pipeline used to answer wrongly and confidently, to **"is this head list an artifact of the experiment?"**, which it now answers loudly and imprecisely — better, and not the same as solved.

One further measurement worth stating on its own: across the four greater-than schemes, **0 heads were found by every scheme**, against 6 of the docstring circuit's. Agreement between counterfactuals is not the normal case that disagreement interrupts. On GPT-2 small, with four schemes of genuinely different design, it is close to empty.

## The eight pre-registered predictions

**6 of 8 scored as hits.** Two were marked low-information in the plan, because the Phase 7 results they follow from are already published in this repo; they are scored anyway so the tally is complete rather than selective.

| # | prediction | outcome | measured |
|---|---|---|---|
| P1 | the flag fires on docstring *(low information — implied by Phase 7's published numbers)* | **hit** | blind spot: 17 heads |
| P2 | the docstring blind spot contains `1.4` and `2.0` *(low information)* | **hit** | published heads in blind spot: `1.2`, `1.4`, `2.0` |
| P3 | the union across schemes has **lower precision** than the primary alone | **hit** | 0.33 → 0.23 |
| P4 | the flag **also** fires on greater-than, where Phase 6 found no such pathology | **hit** | blind spot: 16 heads |
| P5 | no *published* greater-than head is in the primary's blind spot | **hit** | published heads flagged: — |
| P6 | `xx_mismatch` has power between 0.05 and 0.60 of `yy01` | **missed** | power 0.74 (span +1.019) |
| P7 | the search flags the argument movers as spec-scheme-dependent, with `q@END` winning under `random_def` and `v@C_def` under `random_random` | **missed** | 3.0: random_random v@C_def, random_def k@C_def; 3.6: random_random v@C_def, random_def k@C_def |
| P8 | at least one head outside the published circuit is also flagged scheme-dependent | **hit** | 14 unpublished heads flagged |

## What this does not solve

**The schemes are still authored per task, and that is the same dependency Phase 5 named.** The docstring circuit got its second counterfactual for free because the authors published three. Greater-than did not: `xx_mismatch` was designed for this phase by a person reasoning about what the task's mechanism must contain — which value is moved, which structure makes the question well posed — and it is labelled `authored` everywhere it appears rather than being quietly counted as published. Its power came out at 0.74 of the published scheme's, so the alternate is measurably the weaker experiment.

So Phase 8 does not remove the hand-built ingredient. It changes when the ingredient shows its hand: instead of one counterfactual silently defining what the circuit is, several are forced to disagree in public, and the disagreement is printed whether or not anyone was looking for it.

**Three further limits, stated rather than discovered later:**

- **The flag does not say which scheme is right.** It says the answer depends on the experiment. Choosing between the schemes still needs the answer key or an argument this phase does not supply — and on an unfamiliar circuit there is no answer key.
- **The generic schemes are nearly free and nearly useless.** They need no knowledge of the task, so any task can register them and clear the two-scheme bar without a human thinking about the mechanism at all. They are also the lowest-power schemes here and the largest source of flagged heads that are not in any published circuit. A task that registers only generics satisfies the letter of the check and gets very little from it.
- **Blindness that *every* registered scheme shares is still invisible.** The comparison can only report disagreement between the counterfactuals it was given. A property no registered scheme perturbs produces unanimous agreement, which this structure reports as `robust`.

## What the change cost the existing code

`git diff --stat a015ecb -- causal_interp/ scripts/`:

```
causal_interp/agreement.py    | 352 ++++++++++++++++++++++++++++++++++++++++++
 causal_interp/docstring.py    |  54 +++++++
 causal_interp/greater_than.py | 128 ++++++++++++++-
 causal_interp/ioi.py          |  48 ++++++
 causal_interp/pipeline.py     | 255 ++++++++++++++++++++++++++++++
 causal_interp/schemes.py      | 124 +++++++++++++++
 scripts/check_schemes.py      | 288 ++++++++++++++++++++++++++++++++++
 7 files changed, 1244 insertions(+), 5 deletions(-)
```

The causal core — `interventions.py`, `search.py`, `metrics.py`, `comparison.py`, `model.py`, `corruption.py` — is **untouched**, as it was in Phases 6 and 7. The three task modules gained a registration block each; `greater_than.py` additionally gained the authored scheme. Phase 6's own sweep list is frozen in place, and `scripts/check_schemes.py` verifies against git that greater-than's clean and corrupted token tensors under all three pre-existing schemes still hash identically to the module as it stood before this phase.

```bash
python scripts/check_schemes.py     # expect: SCHEMES OK
```

That script is a known-answer test in the sense `check_patching.py` is: it checks that a single-scheme `TaskSpec` is refused, that the agreement analysis returns the verdicts that follow by construction from a synthetic table nobody measured, that the flag stays silent when the schemes agree, and that `xx_mismatch` changes exactly one token and leaves `YY` bit-identical.
