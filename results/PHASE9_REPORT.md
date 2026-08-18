# Phase 9 — separating a real blind spot from noisy disagreement

**The rule, the scoring table, the holdout and eight predictions were fixed in [PHASE9_PLAN.md](PHASE9_PLAN.md)** before anything here was measured, and the measurements the rule was chosen from were committed before *that*, in [PHASE9_CHARACTERIZATION.md](PHASE9_CHARACTERIZATION.md). The order is checkable in git history rather than asserted.

Phase 8's flag fired on both circuits it saw. On docstring it named the three published heads Phase 7 had missed; on greater-than it named sixteen heads that are in no published circuit and where the primary counterfactual had already recovered everything. The flags were identical in form, so acting on one meant re-checking every one by hand.

### What step 1 found, including what it ruled out

Ten candidate signals were measured over all 33 flagged heads. **Nine do not separate the two cases**, and several separate in the wrong direction — anything that normalizes by a scheme's median or its strongest head is dominated by how many dead heads the model has, so GPT-2 small's noise outranks `attn-only-4l`'s real finds. The one axis with visible separation was raw normalized recovery in the scheme that found the head.

### The rule

Phase 3's rule, applied to a new channel, per scheme:

> **θ(s) = the 99th percentile of |normalized recovery| under a shuffled-source null sweep, rounded up to two significant figures** — the clean activation spliced in is drawn from a *deranged* prompt order, so it is a real activation whose prompt-correspondence has been destroyed. θ(s) replaces Phase 8's shared 0.02 as that scheme's discovery criterion.

The diagnosis behind it: normalized recovery divides by each scheme's own clean-vs-corrupted span, so 0.02 does not mean the same thing under two counterfactuals. Nothing else changed — the flag is still a bare non-emptiness test on the primary's blind spot — so the two runs differ in the criterion alone. For docstring and greater-than the real sweeps are read back from Phase 8's committed payloads rather than repeated, and only the null sweeps are new.

## What the null floors came out at

One row per registered scheme, across all three circuits. `power` is Phase 8's measurement — the scheme's span relative to its primary's — and θ is what this phase measured:

| circuit | scheme | provenance | power | null median | null max | θ | heads found: 0.02 → θ |
|---|---|---|---|---|---|---|---|
| docstring | `random_random` *(primary)* | published | 1.00 | 0.0005 | 0.240 | **0.07** | 9 → 4 |
| docstring | `random_def` | published | 0.40 | 0.0016 | 0.235 | **0.12** | 17 → 7 |
| docstring | `random_answer` | published | 1.16 | 0.0005 | 0.295 | **0.08** | 11 → 5 |
| docstring | `random_vocab_cdef` | generic | 0.99 | 0.0007 | 0.227 | **0.091** | 10 → 3 |
| docstring | `random_vocab_any` | generic | 0.06 | 0.0209 | 6.410 | **3.3** | 25 → 0 |
| greater_than | `yy01` *(primary)* | published | 1.00 | 0.0001 | 0.136 | **0.019** | 9 → 9 |
| greater_than | `xx_mismatch` | authored | 0.74 | 0.0001 | 0.225 | **0.033** | 15 → 10 |
| greater_than | `random_vocab_yy` | generic | 0.61 | 0.0000 | 0.045 | **0.013** | 3 → 6 |
| greater_than | `random_vocab_any` | generic | 0.27 | 0.0004 | 0.358 | **0.041** | 10 → 2 |
| ioi | `s2_swap` *(primary)* | published | 1.00 | 0.0004 | 0.210 | **0.058** | 23 → 15 |
| ioi | `abc` | published | 0.53 | 0.0001 | 0.048 | **0.0077** | 18 → 27 |
| ioi | `random_vocab_s2` | generic | 0.50 | 0.0005 | 0.069 | **0.011** | 19 → 24 |
| ioi | `random_vocab_any` | generic | 0.21 | 0.0019 | 0.429 | **0.18** | 25 → 3 |

**θ tracks power inversely, and the effect is enormous at the bottom.** Over the 9 docstring and greater-than schemes the rank correlation between power and θ is **-0.30**. The extreme case is docstring's `random_vocab_any`: power 0.06, and its shuffled-source null manufactures apparent recoveries whose 99th percentile is 3.3 — that is, patching a head with an activation belonging to a *different prompt* routinely moves the metric by many times the entire clean-to-corrupted span. Under a cutoff of 0.02 that scheme contributed 8 of Phase 8's 17 docstring flags. Under its own null it discovers nothing at all.

That is the mechanism this phase is testing, stated in the plan before the numbers existed: a scheme with a small span turns tiny absolute changes into large normalized ones, and Phase 8 compared those numbers against the same cutoff it applied to a scheme with a full-sized span.

## docstring — the known **real blind spot**

attn-only-4l, 4×8 heads, 128 prompts. Real effects: reused from Phase 8.

|  | Phase 8 (shared 0.02) | Phase 9 (per-scheme θ) |
|---|---|---|
| heads flagged (primary's blind spot) | 17 | **4** |
| — of which published | `1.2`, `1.4`, `2.0` | **`1.4`, `2.0`** |
| — of which not published | 14 | **2** |
| primary scheme's own recall | 3/6 | 3/6 |
| primary scheme's own precision | 0.33 | 0.75 |
| union recall across schemes | 6/6 | 5/6 |
| union precision | 0.23 | 0.62 |

> COUNTERFACTUAL-SCHEME-DEPENDENT: 4 head(s) [0.0, 1.0, 1.4, 2.0] are found under another counterfactual and missed under the primary one (random_random). The head list under the primary counterfactual is not the circuit; it is what this counterfactual can see.

Flags kept: `0.0`, `1.0`, `1.4`, `2.0`.

Flags dropped by calibration: `0.2`, `0.4`, `0.6`, `1.1`, `1.2`, `1.5`, `1.7`, `2.1`, `2.4`, `2.5`, `2.6`, `2.7`, `3.2`, `3.4`.

Flags **added** by calibration: `0.0` — a head the primary used to clear at 0.02 and no longer clears at its own θ, while another scheme still does. Raising the primary's floor moves heads *into* its blind spot, which is a cost of the rule and not a bug in it.

## greater-than — the known **non-problem**

gpt2-small, 12×12 heads, 128 prompts. Real effects: reused from Phase 8.

|  | Phase 8 (shared 0.02) | Phase 9 (per-scheme θ) |
|---|---|---|
| heads flagged (primary's blind spot) | 16 | **8** |
| — of which published | — | **—** |
| — of which not published | 16 | **8** |
| primary scheme's own recall | 7/7 | 7/7 |
| primary scheme's own precision | 0.78 | 0.78 |
| union recall across schemes | 7/7 | 7/7 |
| union precision | 0.28 | 0.41 |

> COUNTERFACTUAL-SCHEME-DEPENDENT: 8 head(s) [0.10, 3.0, 4.10, 4.11, 5.0, 7.11, 8.5, 8.6] are found under another counterfactual and missed under the primary one (yy01). The head list under the primary counterfactual is not the circuit; it is what this counterfactual can see.

Flags kept: `0.10`, `3.0`, `4.10`, `4.11`, `5.0`, `7.11`, `8.5`, `8.6`.

Flags dropped by calibration: `1.5`, `3.3`, `4.4`, `4.7`, `5.6`, `6.11`, `6.8`, `7.0`, `7.6`.

Flags **added** by calibration: `7.11` — a head the primary used to clear at 0.02 and no longer clears at its own θ, while another scheme still does. Raising the primary's floor moves heads *into* its blind spot, which is a cost of the rule and not a bug in it.

## Which schemes the removed flags came from

A flag belongs to whichever non-primary scheme found the head. Splitting the blind spots that way shows the rule is not removing noise evenly — it is removing whole schemes:

| circuit | scheme | provenance | power | θ | flags contributed: 0.02 → θ |
|---|---|---|---|---|---|
| docstring | `random_def` | published | 0.40 | 0.12 | 9 → **3** |
| docstring | `random_answer` | published | 1.16 | 0.08 | 2 → **1** |
| docstring | `random_vocab_cdef` | generic | 0.99 | 0.091 | 2 → **0** |
| docstring | `random_vocab_any` | generic | 0.06 | 3.3 | 17 → **0** |
| greater_than | `xx_mismatch` | authored | 0.74 | 0.033 | 8 → **5** |
| greater_than | `random_vocab_yy` | generic | 0.61 | 0.013 | 2 → **3** |
| greater_than | `random_vocab_any` | generic | 0.27 | 0.041 | 9 → **2** |
| ioi | `abc` | published | 0.53 | 0.0077 | 5 → **15** |
| ioi | `random_vocab_s2` | generic | 0.50 | 0.011 | 1 → **9** |
| ioi | `random_vocab_any` | generic | 0.21 | 0.18 | 10 → **0** |

**Two regimes, and the rule only handles one of them.** Where a flag came from a scheme too weak to measure anything — docstring's `random_vocab_any`, power 0.06, whose null manufactures recoveries of 3.3 — calibration removes it completely and the case is closed. Where a flag came from a scheme with healthy power that simply disagrees — greater-than's `xx_mismatch`, power 0.74, θ 0.033 — calibration thins it and cannot dismiss it, because there is nothing statistically wrong with those measurements. They are real effects under a real counterfactual that happen not to correspond to published circuit members.

That distinction is the honest shape of this phase's result: **the null floor separates *this scheme is too weak to be believed* from *this scheme measured something*. It does not separate *this scheme measured something the primary is blind to* from *this scheme measured something that is not part of the circuit*** — and the second distinction is the one a reader actually wants.

## The pre-registered verdict

The scoring table was fixed in the plan; this is it applied to the measurements:

| outcome | criterion | measured |
|---|---|---|
| clean pass | greater-than's blind spot empty **and** docstring keeps `1.4`, `2.0` | empty? no — 8 left |
| partial | greater-than shrinks ≥ 50 % **and** docstring keeps both | shrank 50%; docstring keeps both? yes |
| negative | shrinks < 50 %, or docstring loses either head | — |

### Verdict: **PARTIAL**

greater-than's blind spot shrank by 50% but is not empty, and docstring kept both routing heads.

## The holdout, and what it is worth

The rule was chosen while looking at docstring and greater-than, so neither can test it. **IOI had never been run through the multi-scheme pipeline** — Phase 8 registered its four schemes and deliberately did not run it — and its expected answer comes from Phase 1 independently of anything here: the primary scheme `s2_swap` is provably blind before the S2 token (Phase 1 measured 576/576 head-position cells there as exact zeros) while `abc` is not. So IOI should behave like docstring, not like greater-than.

| IOI | Phase 8 criterion | Phase 9 criterion |
|---|---|---|
| heads flagged | 15 | **19** |
| — of which published | `0.1`, `10.2`, `4.11` | **`0.1`, `10.1`, `10.10`, `10.2`, `9.0`** |
| — of which not published | 12 | **14** |
| primary recall | 18/26 | 15/26 |
| union recall | 21/26 | 20/26 |

> COUNTERFACTUAL-SCHEME-DEPENDENT: 19 head(s) [0.1, 0.3, 0.4, 0.5, 4.3, 6.0, 8.2, 9.0, 9.2, 9.3, 9.4, 9.5, 9.8, 10.1, 10.2, 10.10, 11.3, 11.6, 11.8] are found under another counterfactual and missed under the primary one (s2_swap). The head list under the primary counterfactual is not the circuit; it is what this counterfactual can see.

**IOI does have a real blind spot, and the calibrated flag points at it.** The surviving flags include `0.1` (duplicate token), `10.1` (backup name mover), `10.10` (backup name mover), `10.2` (backup name mover), `9.0` (backup name mover) — published heads that `s2_swap` misses and another scheme finds, which is the pattern Phase 1 would predict and the prediction was recorded before the run.

**And the rule did not do what it was supposed to do here.** IOI's blind spot **grew**, 15 → 19, where the whole point of calibration was to shrink it. The cause is visible in the floor table: `abc` and `random_vocab_s2` came out with θ **below** 0.02 (0.0077 and 0.011), so calibration made them *more* sensitive and they discovered more heads, while the primary's floor rose to 0.058 and it lost three published heads of its own (18/26 → 15/26). Calibration cuts both ways, and on this circuit it cut the wrong way.

The flag did get *sharper* even as it got louder — the share of flagged heads that are published circuit members went 20% → 26% — but a detector that fires on 19 heads of which 14 are not in the circuit is not one a reader can act on without the answer key, which is exactly the complaint Phase 9 set out to fix.

## The eight pre-registered predictions

**4 of 8 scored as hits.**

| # | prediction | outcome | measured |
|---|---|---|---|
| P1 | θ is inversely related to a scheme's power (negative rank correlation) | **hit** | rank correlation -0.30 over 9 schemes |
| P2 | greater-than's calibrated blind spot is **empty** | **missed** | 8 heads remain |
| P3 | docstring's calibrated blind spot is non-empty and contains `1.4`, `2.0` | **hit** | 4 flagged: `0.0`, `1.0`, `1.4`, `2.0` |
| P4 | `random_vocab_any` contributes at most 2 heads to docstring's calibrated blind spot | **hit** | 0 heads (θ = 3.3) |
| P5 | for both hand-built primaries θ lands within a factor of 3 of 0.02 | **missed** | docstring: θ=0.07, greater_than: θ=0.019 |
| P6 | **holdout:** IOI's calibrated flag fires and its blind spot contains at least one published IOI head | **hit** | flag fires; published in blind spot: `0.1`, `10.1`, `10.10`, `10.2`, `9.0` |
| P7 | **holdout:** IOI's calibrated blind spot is smaller than its uncalibrated one | **missed** | 15 → 19 |
| P8 | no scheme's θ exceeds 0.5 | **missed** | largest is docstring/random_vocab_any at θ = 3.3 |

## Does a pipeline-internal signal separate the two cases?

The whole phase in one table — the flag's *precision*, meaning the share of flagged heads that turn out to be published circuit members. A discriminator would drive this up on the circuits with a real blind spot and to zero flags on the one without:

| circuit | known case | heads flagged | published among them | flag precision |
|---|---|---|---|---|
| docstring | real blind spot | 17 → **4** | 3 → **2** | 18% → **50%** |
| greater_than | no blind spot | 16 → **8** | 0 → **0** | 0% → **0%** |
| ioi | real blind spot (holdout) | 15 → **19** | 3 → **5** | 20% → **26%** |

### The answer is: partly, and not enough to act on

**What the rule does do, decisively.** It removes flags produced by schemes that are too weak to measure anything. Docstring's `random_vocab_any` — power 0.06, θ 3.3 — contributed 17 of that circuit's flags and now contributes none, and docstring's flag went from 17 heads with 3 published to 4 heads with 2. On that circuit the flag became something a reader could act on. Every scheme's null floor is a real measurement of how much apparent recovery that experiment manufactures from a mismatched activation, and the numbers span a factor of 400 across the ten schemes measured — Phase 8's single shared cutoff was not defensible, and that much is now established.

**What it does not do.** It cannot tell a scheme that disagrees *because the primary is blind* from a scheme that disagrees *because it is measuring something outside the circuit*. Both are statistically sound measurements under a real counterfactual. Greater-than's flag still names 8 heads, none published, and its precision is still 0 %. IOI's flag got **larger**, not smaller. Two of three circuits end the phase with a flag that still needs a human and an answer key.

**So the honest statement of the result is the one the plan required in advance:** a per-scheme null floor is a real improvement to *what counts as a measurement*, and it is **not** a discriminator between a real blind spot and ordinary disagreement. The signal that separates the two known cases in step 1's tables — raw effect magnitude — turns out to separate them because docstring's noise came from one pathological scheme, not because magnitude means what a discriminator would need it to mean. Nothing else measured here separates them at all.

The Phase 8 ceiling therefore stands, in a sharper form. Disagreement detection can say *this head list depends on the experiment*, and it can now also say *this scheme was too weak to be believed*. It still cannot say *and therefore the primary is missing part of the circuit* — that step needed the published head list on all three circuits here.

## What this does not establish

- **A surviving flag is still not a claim that a head is real.** The criterion asks whether a scheme's measurement exceeds what that scheme manufactures from a mismatched activation. Whether the head is in the circuit is a different question, and nothing here answers it.
- **Calibration moves every scheme's floor, in both directions.** It applies to the primary as well, and a scheme whose null is quiet gets a floor *below* 0.02 and discovers more, not less — `abc` on IOI came out at 0.0077 and went from 18 discovered heads to 27. On docstring the primary's recall held at 3/6 while its floor rose to 0.07; on IOI the primary's floor rose to 0.058 and its recall fell 18/26 → 15/26. The rule is not a noise filter; it is a recalibration, and on one of three circuits it made the flag louder.
- **The null has its own assumptions.** Splicing an activation from another prompt destroys prompt-correspondence but preserves whatever a head does that is prompt-independent, so a positional or structural head inflates its scheme's floor. Recorded in the plan in advance; measured here; not corrected.
- **n = 3.** Three circuits, two models, one architecture family, and the rule was chosen while looking at two of the three. One holdout is one holdout.
- **Nothing here touches which behaviour to study or the task template.** The top two rows of the README ladder are where they were in Phase 5.
