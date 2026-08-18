# Phase 9 plan — one rule, fixed before it is tested

**Committed before any Phase 9 measurement exists.** The characterization it rests on is
in [PHASE9_CHARACTERIZATION.md](PHASE9_CHARACTERIZATION.md), committed first and
separately, so the order is checkable in git rather than asserted.

## The problem

Phase 8's flag fired on both circuits it was run against. On docstring the flagged set
contained the three published heads Phase 7 had missed; on greater-than it contained
sixteen heads that are in no published circuit and the primary counterfactual had
already recovered everything. The two flags were identical in form, and nothing the
pipeline computed separated them. A detector that fires the same way on a real problem
and on a non-problem cannot be acted on without a human re-checking every flag, which
is the thing Phase 8 was built to avoid.

## What step 1 found, and what it did not

Ten candidate signals were measured over all 33 flagged heads. **Nine of them do not
separate the two cases**, and several separate in the wrong direction; the full table is
in the characterization. Ratios that normalize by a scheme's median or its strongest
head are dominated by how many dead heads the model has — GPT-2 small's 144 heads give a
far smaller median than `attn-only-4l`'s 32 — so they rank greater-than's noise *above*
docstring's real finds.

The one axis with visible separation is the **raw normalized recovery in the scheme that
found the head**: docstring's flagged set reaches 0.60 and 1.09, and nothing in
greater-than's exceeds 0.09. The separation is in the upper tail rather than between the
medians (0.07 against 0.02).

**This plan is therefore not blind, and the disclosure is the same one Phase 3 made.**
The rule below was chosen after seeing that magnitude was the only axis with visible
separation. What is *not* known when this is committed: where each scheme's floor
actually lands, whether those floors separate the two flagged sets, and what happens on
a third circuit. The holdout in section 5 is the part of this phase carrying independent
evidence.

## 1. The diagnosis the rule is built on

Phase 8 compared every scheme against one shared cutoff of **0.02**, inherited from
Phase 1. But normalized recovery divides by *that scheme's own* clean-vs-corrupted span:

```
recovery = (patched − corrupted) / (clean − corrupted)
```

A scheme whose corrupted run sits close to its clean run has a small denominator, so an
absolutely tiny change becomes a large normalized number. Phase 8 already measured the
spans and reported them as `power` — docstring's `random_vocab_any` has power 0.06 and
produced 8 of the 17 flagged heads; greater-than's alternates have power 0.74, 0.61 and
0.27 and produced all 16. **0.02 does not mean the same thing under two schemes, and
Phase 8's flag treated it as if it did.**

## 2. The rule, fixed here

Phase 3's rule, applied verbatim to a new channel. For each registered scheme `s`:

> Run the identical head sweep with the spliced clean activation drawn from a
> **deranged** prompt order — a real activation whose prompt-correspondence has been
> destroyed. Then
>
> **θ(s) = the 99th percentile of |normalized recovery| over that null sweep, rounded up
> to two significant figures**,
>
> and θ(s) replaces the shared 0.02 as scheme `s`'s discovery criterion. Everything else
> in Phase 8 — the per-head verdicts, the per-scheme blind spots, the flag — is
> recomputed from the new discovered sets and is otherwise unchanged.

Notes fixed here so they cannot be adjusted later:

- **The null unit is one (head, position) cell**, the unit the sweep measures, exactly as
  Phase 3 pooled per-measurement. The real per-head statistic is a *maximum* over
  positions, so comparing it against a per-cell null is **permissive** — it keeps more
  flags than a like-for-like comparison would. That bias runs against this phase's
  hypothesis, which is why it is the version chosen.
- **θ(s) is used as computed, including when it is below 0.02.** Flooring it at Phase 1's
  number would be a free parameter, and a scheme whose null floor is genuinely lower
  deserves the more sensitive criterion.
- **The null seed is 20260815** and the derangement is `interventions.derangement`, both
  Phase 3's.
- **Rounded up to two significant figures**, Phase 3's `_round_up_sigfigs`, unchanged.
- The metric is `logit_diff`, the channel is activation patching, `n=128`, `seed=0` —
  every one of them the setting Phase 8 used, so the only difference between the two
  runs is the criterion.

**No other change is made.** In particular the flag stays a bare non-emptiness test on
the primary's blind spot; this phase changes what "found" means, not what "flagged"
means.

## 3. What is reported but is not a criterion

The descriptive signals from step 1 — prominence, number of finding schemes, detection
ratio, provenance of the finder — are printed beside every surviving flag. They are
*context for a reader*, and no rule consults them. Phase 8 made power an annotation for
the same reason: a second criterion is a second thing to tune.

## 4. Scoring, fixed before the run

| outcome | criterion |
|---|---|
| **clean pass** | greater-than's calibrated blind spot is **empty**, and docstring's still contains `1.4` and `2.0` |
| **partial** | greater-than's blind spot shrinks by **≥ 50 %** and docstring keeps `1.4` and `2.0` |
| **negative** | greater-than's shrinks by < 50 %, **or** docstring loses `1.4` or `2.0` |

A negative result is reported as the phase's finding, in the README ladder as well as
the report — that agreement-flagging can say *something is inconsistent* and cannot say
whether the inconsistency means *wrong* or *incomplete*. It is not retried with a
different statistic.

## 5. The holdout — IOI, and why it is a real one

The rule is tuned to nothing, but it was *chosen* while looking at two circuits, so those
two cannot test it. **IOI is available and has never been run through the multi-scheme
pipeline.** Phase 8 registered its four schemes and deliberately did not run it.

It is a genuine third case, and its expected answer is known from Phase 1 in a way that
does not depend on this phase:

- the primary scheme `s2_swap` is **provably blind before the S2 token** — Phase 1
  measured 576/576 head-position cells there as exact floating-point zeros;
- the second published scheme `abc` replaces all three names, so the earlier positions
  are not bit-identical and *are* measurable under it;
- so IOI should behave like docstring rather than greater-than: a real, structural blind
  spot in the primary counterfactual.

The holdout is specified completely here, before it runs: IOI, GPT-2 small, `n=128`,
`seed=0`, the four registered schemes, the activation-patching channel only, scored
against `ground_truth.py` **after** the calibrated verdicts are computed.

**Even with it, this is n = 3.** Three circuits, two models, one architecture family.
That is stated in the report and in the README rather than left for a reader to work out.

## 6. Predictions

Scored in `PHASE9_REPORT.md` whichever way they come out.

| # | prediction |
|---|---|
| P1 | θ(s) is **inversely related to a scheme's power** — over the nine docstring and greater-than schemes, the rank correlation between power and θ is negative |
| P2 | On greater-than the calibrated blind spot is **empty** and the flag goes silent |
| P3 | On docstring the calibrated blind spot is **non-empty** and contains `1.4` and `2.0` |
| P4 | On docstring, `random_vocab_any` — power 0.06, source of 8 of the 17 flags — contributes **at most 2** heads to the calibrated blind spot |
| P5 | For the two hand-built primaries, θ lands **within a factor of 3 of 0.02**, so calibration mostly moves the weak schemes rather than redefining the primary |
| P6 | **Holdout:** IOI's calibrated flag fires, and its blind spot contains **at least one published IOI head** |
| P7 | **Holdout:** IOI's calibrated blind spot is **smaller** than its uncalibrated one |
| P8 | Across all three circuits, no scheme's θ exceeds **0.5** — if a null sweep manufactures apparent recovery above half the clean-corrupted span, the scheme is not measuring anything and that would be a different finding from the one this phase expects |

## 7. What this phase cannot establish, stated in advance

- **A surviving flag is still not a claim that the head is real.** The criterion asks
  whether a scheme's measurement is above what that scheme manufactures from a mismatched
  activation. It does not ask whether the head is in the circuit, and nothing here can.
- **The null is itself an experiment with assumptions.** Drawing the source activation
  from another prompt destroys prompt-correspondence but preserves whatever a head does
  that is prompt-independent — a positional or structural head may restore behaviour from
  a mismatched value, and its null contribution will be large. That makes the floor
  conservative for such heads, in a way this phase measures but does not correct.
- **Nothing here addresses which behaviour to study, or the task template.** The top two
  rows of the README ladder are untouched, as they have been since Phase 5.
