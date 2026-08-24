# Phase 11 — pre-registration: does stability under resampling separate real findings from noise?

**Written and committed before `scripts/run_phase11_stability.py` exists.** Nothing below
was chosen after seeing a Phase 11 number. The order is checkable in git history, as in
every phase here.

## Why this is not a retry

Phases 9 and 10 both failed the same task — judging whether a finding is real without an
answer key — and [`SYNTHESIS.md` §2](../SYNTHESIS.md) diagnoses why: both answered a
question about **how far an intervention moved something**. Phase 9 asked a *validity*
question (is this bigger than what the experiment manufactures from noise?), Phase 10 a
*magnitude* question (which counterfactual moves the output most?). The
[scheme-level re-analysis](SCHEME_LEVEL_NOTE.md) then moved all twenty of those signals to
the unit that actually matters and found nothing there either.

Every signal tried so far is a functional of **one** measurement. This phase tests a class
that is not available from one measurement at all:

> **Replication.** Re-run the identical experiment under independent resampling of the
> experimental surface and ask whether the finding comes back.

That is a different kind of evidence from magnitude, from null-comparison and from ranking.
It is also the closest thing in this repository to the untested lever `SYNTHESIS.md` §5.2
names — "whether a counterfactual's effect is consistent" — though §5.2 meant consistency
*across prompts within one run*, and this phase measures consistency *across independent
runs*. The two are related and are not the same; the difference is recorded here rather
than blurred.

The hypothesis, stated so it can fail:

> A real circuit component's causal effect is a property of the mechanism, so it should
> survive redrawing the prompts and the corruption instances. A finding driven by noise or
> by an artifact of one sample should be idiosyncratic to that sample.

Steps 1–3 below construct the signal with **no reference to ground truth**. The answer key
is opened only in step 4, and the comparison method is fixed here first.

---

## 1. The resampling axis

**Chosen: the dataset seed.** All three task modules build their prompts from a single
`random.Random(seed)` stream that generates the clean prompt *and* its corrupted partner
together (`docstring.py:240`, `greater_than.py:196`, `ioi.py:176`). Changing the seed
therefore redraws, jointly and independently:

- which 128 prompts are sampled from the template's slot vocabularies,
- which corruption instance each scheme draws for each of those prompts.

**Why this axis and not the other two the brief offers.**

- *Different corruption instances, prompts held fixed* cannot be varied independently
  without editing the task constructors to split their RNG streams. That would change the
  causal core in the same phase that measures it, and the two effects could not afterwards
  be told apart. `greater_than.py` already keeps one `_alt_rng` for exactly this reason and
  the comment there explains the hazard.
- *Subsets of one example set* is a strictly weaker perturbation of the same quantity:
  subsets of one draw share prompts with each other, independent seeds share none. If the
  weaker axis showed something the stronger one did not, that would be a finding about
  subsampling, not about replication.
- The seed axis requires **no change to any existing module.** `pipeline.discover` already
  takes `seed`. Phase 11 adds a script and nothing else.

**What is held fixed across resamples**, so that a difference is the sample and nothing
else: the model, the task template, the position vocabulary, the scheme set, the metric
(`logit_diff`), `n = 128`, the collapse rule, and the discovery criterion θ.

**θ is inherited, not recomputed.** Each scheme's null floor is read from the committed
Phase 9 payload (`results/phase9_<circuit>.json` → `floors`), computed at seed 0. Holding
the bar still is what makes "did this head clear the bar again" a question about the head
rather than about the bar. The θ values in force are therefore, and are frozen here:

| circuit | scheme | θ |
|---|---|---|
| docstring | `random_random` *(primary)* | 0.070 |
| docstring | `random_def` | 0.120 |
| docstring | `random_answer` | 0.080 |
| docstring | `random_vocab_cdef` | 0.091 |
| docstring | `random_vocab_any` | 3.300 |
| greater_than | `yy01` *(primary)* | 0.019 |
| greater_than | `xx_mismatch` | 0.033 |
| greater_than | `random_vocab_yy` | 0.013 |
| greater_than | `random_vocab_any` | 0.041 |

## 2. How many resamples, and which circuits

**R = 10, seeds 0, 1, …, 9.**

*Lower bound, from the statistics.* The hit-fraction statistic below has resolution 1/R, so
R = 10 is the smallest R that distinguishes "never clears", "sometimes clears" and "always
clears" on a tenth-part grid. The standard-deviation estimate behind the SNR statistic has
relative standard error 1/sqrt(2(R−1)) ≈ 0.236 at R = 10. The scheme-level pairwise
statistics average over R(R−1)/2 = 45 pairs.

*Upper bound, from compute.* Phase 8 measured 291 s for docstring's five-scheme sweep and
1039 s for greater-than's four, on this repository's one GPU. R = 10 is 90 scheme-sweeps and
about **3.7 GPU-hours** — already the largest single measurement in the project, against
0.37 h for Phase 8's two runs combined. R = 20 would be 7.4 h and buys a 1/sqrt(2)
improvement in the noisiest quantity.

*Seed 0 is included deliberately.* It reproduces the exact configuration of the committed
Phase 8 runs, so resample 0 doubles as an integrity check (prediction 1).

**Circuits: docstring and greater-than.** Docstring carries the known real blind spot
(Phase 7) and the known-bad scheme `random_vocab_any` as a negative control; greater-than is
the clean-recovery comparison where Phase 6's primary already found everything. **IOI is
omitted**, for compute (another ~2.9 h) and because the scheme-level note established that
IOI's four rows carry almost no label variance — 0.906 / 0.889 / 0.884 / 0.880 — so they
would add rows without adding information. This leaves **9 `(circuit, scheme)` rows**,
against the note's 13, and that reduction is a cost recorded here in advance.

## 3. The stability statistics — constructed with no answer key

For circuit *c*, scheme *s*, head *h*, resample *r* in {0…9}, let

    e(h, s, r) = the pipeline's own collapsed normalized recovery under `logit_diff`
                 — `collapse_positions`, i.e. the value at the position of largest
                 absolute effect for that head in that resample.

The collapse position may differ between resamples. That is part of the experimental
surface and is left alone; the primary statistics use the pipeline's rule unchanged. A
variant holding each head's position fixed at its seed-0 choice is computed and reported as
a footnote only — **declared here as ineligible to be the result**, so it cannot become one.

Write m(h,s) = mean over r of e, and sd(h,s) = sample standard deviation over r (ddof = 1).

**Head-level candidates.**

| | definition | reads |
|---|---|---|
| **S1** `hit_fraction` | (1/R) × #{ r : abs e(h,s,r) ≥ θ(s) } | how often this head clears its scheme's own bar |
| **S2** `snr` | abs m(h,s) ÷ sd(h,s) | effect against its own replication spread |
| **S3** `sign_consistency` | abs of the mean of sign e(h,s,r) over r | does the effect even point the same way |

**Magnitude baselines**, so stability is never credited for something averaging alone buys:

| | definition | reads |
|---|---|---|
| **B0** | abs e(h,s,0) | what the current pipeline reports from one run |
| **Bm** | abs m(h,s) | the same magnitude, estimated from all ten |

**Bm, not B0, is the baseline any stability statistic must beat.** Comparing S against B0
would credit stability for the extra data it was given.

**The crux diagnostic.**

| | definition |
|---|---|
| **S4** `rel_sd` | sd(h,s) ÷ median over heads of sd(·,s) — per-head noise against the scheme's typical head |
| | Spearman(S2, Bm) over the heads of each row |

These are not discriminator candidates. They decide *what a null result means*, and are
fixed here so that step 6 of the brief cannot be answered post hoc — see §6.

**Scheme-level candidates**, since the note established the scheme is the unit a reader
actually judges. D0(s) = heads discovered at seed 0, i.e. abs e(h,s,0) ≥ θ(s).

| | definition |
|---|---|
| **T1** `rank_reproducibility` | mean over the 45 resample pairs of Spearman(e(·,s,r), e(·,s,r′)) across all heads |
| **T2** `set_reproducibility` | mean over the 45 pairs of Jaccard of the two discovered head sets at θ |
| **T3** `median_hit_fraction` | median over h in D0(s) of S1(h,s) |
| **T4** `median_snr` | median over h in D0(s) of S2(h,s) |
| **T5** `span_cv` | sd over r of span ÷ abs mean over r of span, span = clean − corrupted |

Five, fixed. No sixth is added later.

## 4. The comparisons against the answer key — method fixed before any score is seen

### P1 — head level, primary. Does stability beat magnitude at finding published heads?

For each of the 9 rows, compute the AUC of published-vs-unpublished head separation
(probability a random published head outranks a random unpublished one, ties a half — the
same `auc` the scheme-level note used) under S1, S2, S3, B0 and Bm.

**Declared an improvement** only if, for at least one of S1/S2/S3:

1. median over the 9 rows of [AUC(S) − AUC(Bm)] ≥ **+0.05**, and
2. two-sided exact Wilcoxon signed-rank over the 9 paired rows, p < **0.05 / 3 = 0.0167**
   (Bonferroni over the three candidates).

Both bars, or it is not an improvement. AUC against B0 is reported alongside and is **not**
a route to the claim.

### P2 — scheme level, secondary. Does scheme stability predict scheme aim?

Label A is taken **unchanged** from [`SCHEME_LEVEL_NOTE.md`](SCHEME_LEVEL_NOTE.md): each
scheme's `aim_auc`, computed on the seed-0 effects, so the label is identical to the one
already committed and is not re-derived to suit this phase.

Spearman between each of T1–T5 and Label A across the 9 rows; max-statistic permutation
null, 20 000 shuffles of the labels within the whole set, taking the largest absolute ρ
across all five signals each time. **Declared separating** requires all three of the note's
bars: family-wise p < 0.05, abs ρ ≥ 0.7, and sign consistent within **both** circuits.

**Registered in advance as underpowered.** n = 9 is smaller than the n = 13 that was already
inconclusive. This test is run for continuity with the note, and a null here carries little
information beyond what the note established. Saying so now is the point of saying it now.

### P3 — the operational question. Is a flagged disagreement reproducible?

This is the most on-target test in the phase and it gets exactly one statistic, named now.

Phase 8's flag put **17 heads** in docstring's primary blind spot, of which the published
head list contains **3** (`1.2`, `1.4`, `2.0`). For each flagged head let s\*(h) be the
non-primary scheme with the largest absolute effect at seed 0 — Phase 9's "best other
scheme", already tabulated in [`PHASE9_CHARACTERIZATION.md`](PHASE9_CHARACTERIZATION.md).
The statistic is

    S2( h, s*(h) )   — is the disagreement that flagged this head reproducible?

AUC of the 3 published against the other 14, with an **exact** permutation null over all
C(17,3) = 680 labellings. Bar: p < 0.05, uncorrected, because exactly one statistic is
being tested and it is named before the run.

**Greater-than cannot supply this test**: its flagged set of 16 contains zero published
heads, so the AUC is undefined. Stated now, not discovered later.

## 5. Predictions, to be scored

Recorded so this phase can fail its own expectations the way Phases 7–10 did.

1. The seed-0 re-run reproduces the committed Phase 8 effects to within 1e-4 for every head
   under every scheme on both circuits.
2. **No stability statistic clears the P1 bar against Bm**, on either circuit.
3. Bm beats B0 — median AUC gain across the 9 rows ≥ +0.01. Averaging ten runs helps; that
   is more data, not the hypothesis.
4. Median across rows of Spearman(S2, Bm) ≥ **0.85** — SNR is close to a monotone transform
   of magnitude.
5. `rel_sd` is near-homoscedastic on both circuits: p75/p25 of sd across heads < **3**.
6. P2 inconclusive, with the best absolute ρ below the median of its own permutation null.
7. P3 not significant (p ≥ 0.05).
8. Docstring's `random_vocab_any` — power 0.06, θ 3.3, the scheme Phase 9 established as
   pathological — has the **lowest** T1 of docstring's five schemes. It also had the
   *second-highest* Label A AUC of all thirteen rows, so if this holds, T1 disagrees with
   aim in the same direction θ does, and is another scale-integrity measure rather than an
   aim measure.
9. Greater-than's `yy01` (Label A 0.998) has the **highest** T1 of greater-than's four.

## 6. What a null result will be taken to mean — fixed now, not afterwards

The brief's step 6 asks for an explicit reading if stability fails. That reading depends on
the crux diagnostic, so both branches are written here before the numbers exist:

- **If predictions 4 and 5 hold** (SNR ≈ monotone in magnitude; per-head noise spread narrow),
  then the resampling axis carries almost **no head-specific information**: every head is
  measured with about the same replication error, so abs m ÷ sd is abs m divided by a
  near-constant, and no statistic built on this axis can add to magnitude. The conclusion to
  draw is that **the noise here is not "high-variance, low-sample-size" in character.**
  Findings do not fail to replicate; they replicate perfectly well and are still
  uninformative about aim. That narrows what a fix could be: not more prompts, not more
  seeds, not a variance-aware threshold.
- **If prediction 4 or 5 fails** but P1/P2/P3 still find nothing, the reading is weaker —
  there *is* head-specific replication structure and it simply does not track the published
  head list. That would leave open the possibility that a different functional of the same
  axis works, and this phase would say so.

## 7. Rules

- Steps 1–3 import no `ground_truth` module. The script re-runs `assert_analysis_is_blind`
  over the same module list Phases 4, 6, 8 and 9 use.
- The statistics S1–S4, T1–T5, the baselines, the three tests and their bars are frozen by
  this document. **Nothing is redefined, no threshold is moved, and no second analysis is
  run after seeing the scores.** If nothing clears, that is the phase.
- The position-fixed variant of e is a footnote and is ineligible to be the result.
- Whatever the outcome, the README and SYNTHESIS.md are updated to match, and the nine
  predictions above are scored in public.
