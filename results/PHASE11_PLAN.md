# phase 11, pre-registration: does stability under resampling separate real findings from noise?

Written and committed before `scripts/run_phase11_stability.py` exists. nothing below
was chosen after seeing a phase 11 number. the order is checkable in git history, as in
every phase here.

## why this isnt a retry

phases 9 and 10 both failed the same task, judging whether a finding is real without an
answer key, and [`SYNTHESIS.md` §2](../SYNTHESIS.md) diagnoses why: both answered a
question about how far an intervention moved something. phase 9 asked a *validity*
question (is this bigger than what the experiment manufactures from noise?), phase 10 a
*magnitude* question (which counterfactual moves the output most?). the
[scheme-level re-analysis](SCHEME_LEVEL_NOTE.md) then moved all twenty of those signals to
the unit that actually matters and found nothing there either.

Every signal tried so far is a functional of **one** measurement. this phase tests a class
that isnt available from one measurement at all:

> **replication.** re-run the identical experiment under independent resampling of the
> experimental surface and ask whether the finding comes back.

that is a different kind of evidence from magnitude, from null-comparison and from ranking.
It is also the closest thing in this repository to the untested lever `SYNTHESIS.md` §5.2
names, "whether a counterfactual's effect is consistent", though §5.2 meant consistency
*across prompts within one run*, and this phase measures consistency *across independent
runs*. the two are related and arent the same. the difference is recorded here rather
than blurred.

the hypothesis, stated so it can fail:

> A real circuit component's causal effect is a property of the mechanism, so it should
> survive redrawing the prompts and the corruption instances. A finding driven by noise or
> by an artifact of one sample should be idiosyncratic to that sample.

Steps 1-3 below construct the signal with **no reference to ground truth**. the answer key
is opened only in step 4, and the comparison method is fixed here first.

---

## 1. the resampling axis

Chosen: the dataset seed. all three task modules build their prompts from a single
`random.Random(seed)` stream that generates the clean prompt *and* its corrupted partner
together (`docstring.py:240`, `greater_than.py:196`, `ioi.py:176`). changing the seed
therefore redraws, jointly and independently:

- which 128 prompts are sampled from the template's slot vocabularies,
- Which corruption instance each scheme draws for each of those prompts.

Why this axis and not the other two the brief offers.

- *Different corruption instances, prompts held fixed* cant be varied independently
  without editing the task constructors to split their RNG streams. That would change the
  causal core in the same phase that measures it, and the two effects couldnt afterwards
  be told apart. `greater_than.py` already keeps one `_alt_rng` for exactly this reason and
  the comment there explains the hazard.
- *Subsets of one example set* is a strictly weaker perturbation of the same quantity:
  subsets of one draw share prompts with each other, independent seeds share none. If the
  weaker axis showed something the stronger one didnt, that would be a finding about
  subsampling, not about replication.
- the seed axis requires no change to any existing module. `pipeline.discover` already
  takes `seed`. phase 11 adds a script and nothing else.

what is held fixed across resamples, so that a difference is the sample and nothing
else: the model, the task template, the position vocabulary, the scheme set, the metric
(`logit_diff`), `n = 128`, the collapse rule, and the discovery criterion theta.

theta is inherited, not recomputed. Each scheme's null floor is read from the committed
phase 9 payload (`results/phase9_<circuit>.json` -> `floors`), computed at seed 0. holding
the bar still is what makes "did this head clear the bar again" a question about the head
rather than about the bar. the theta values in force are therefore, and are frozen here:

| circuit | scheme | theta |
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

R = 10, seeds 0, 1..., 9.

*Lower bound, from the statistics.* the hit-fraction statistic below has resolution 1/R, so
R = 10 is the smallest R that distinguishes "never clears", "sometimes clears" and "always
clears" on a tenth-part grid. the standard-deviation estimate behind the SNR statistic has
relative standard error 1/sqrt(2(R-1)) ≈ 0.236 at R = 10. the scheme-level pairwise
statistics average over R(R-1)/2 = 45 pairs.

*Upper bound, from compute.* phase 8 measured 291 s for docstring's five-scheme sweep and
1039 s for greater-than's four, on this repository's one GPU. R = 10 is 90 scheme-sweeps and
about **3.7 GPU-hours**, already the largest single measurement in the project, against
0.37 h for phase 8's two runs combined. R = 20 would be 7.4 h and buys a 1/sqrt(2)
improvement in the noisiest quantity.

*Seed 0 is included deliberately.* it reproduces the exact configuration of the committed
phase 8 runs, so resample 0 doubles as an integrity check (prediction 1).

Circuits: docstring and greater-than. docstring carries the known real blind spot
(phase 7) and the known-bad scheme `random_vocab_any` as a negative control. greater-than is
the clean-recovery comparison where phase 6's primary already found everything. **IOI is
omitted**, for compute (another ~2.9 h) and cuz the scheme-level note established that
IOI's four rows carry almost no label variance, 0.906 / 0.889 / 0.884 / 0.880, so they
would add rows without adding information. this leaves 9 `(circuit, scheme)` rows,
against the note's 13, and that reduction is a cost recorded here in advance.

## 3. the stability statistics, constructed with no answer key

For circuit *c*, scheme *s*, head *h*, resample *r* in {0...9}, let

    e(h, s, r) = the pipeline's own collapsed normalized recovery under `logit_diff`
                 `collapse_positions`, i.e. the value at the position of largest
                 absolute effect for that head in that resample.

the collapse position may differ between resamples. That is part of the experimental
surface and is left alone. the primary statistics use the pipeline's rule unchanged. A
variant holding each head's position fixed at its seed-0 choice is computed and reported as
a footnote only, declared here as ineligible to be the result, so it cant become one.

write m(h,s) = mean over r of e, and sd(h,s) = sample standard deviation over r (ddof = 1).

head-level candidates.

| | definition | reads |
|---|---|---|
| **S1** `hit_fraction` | (1/R) x #{ r : abs e(h,s,r) ≥ theta(s) } | how often this head clears its scheme's own bar |
| **S2** `snr` | abs m(h,s) ÷ sd(h,s) | effect against its own replication spread |
| **S3** `sign_consistency` | abs of the mean of sign e(h,s,r) over r | does the effect even point the same way |

**magnitude baselines**, so stability is never credited for something averaging alone buys:

| | definition | reads |
|---|---|---|
| **B0** | abs e(h,s,0) | what the current pipeline reports from one run |
| **Bm** | abs m(h,s) | the same magnitude, estimated from all ten |

Bm, not B0, is the baseline any stability statistic must beat. comparing S against B0
would credit stability for the extra data it was given.

the crux diagnostic.

| | definition |
|---|---|
| **S4** `rel_sd` | sd(h,s) ÷ median over heads of sd(·,s), per-head noise against the scheme's typical head |
| | Spearman(S2, Bm) over the heads of each row |

these arent discriminator candidates. they decide *what a null result means*, and are
fixed here so that step 6 of the brief cant be answered post hoc, see §6.

**scheme-level candidates**, since the note established the scheme is the unit a reader
actually judges. D0(s) = heads discovered at seed 0, i.e. abs e(h,s,0) ≥ theta(s).

| | definition |
|---|---|
| **T1** `rank_reproducibility` | mean over the 45 resample pairs of Spearman(e(·,s,r), e(·,s,r′)) across all heads |
| **T2** `set_reproducibility` | mean over the 45 pairs of Jaccard of the two discovered head sets at theta |
| **T3** `median_hit_fraction` | median over h in D0(s) of S1(h,s) |
| **T4** `median_snr` | median over h in D0(s) of S2(h,s) |
| **T5** `span_cv` | sd over r of span ÷ abs mean over r of span, span = clean - corrupted |

five, fixed. No sixth is added later.

## 4. the comparisons against the answer key, method fixed before any score is seen

### P1, head level, primary. does stability beat magnitude at finding published heads?

for each of the 9 rows, compute the AUC of published-vs-unpublished head separation
(probability a random published head outranks a random unpublished one, ties a half, the
same `auc` the scheme-level note used) under S1, S2, S3, B0 and Bm.

Declared an improvement only if, for at least one of S1/S2/S3:

1. median over the 9 rows of [AUC(S) - AUC(Bm)] ≥ **+0.05**, and
2. two-sided exact Wilcoxon signed-rank over the 9 paired rows, p < **0.05 / 3 = 0.0167**
   (Bonferroni over the three candidates).

Both bars, or it isnt an improvement. AUC against B0 is reported alongside and is **not**
a route to the claim.

### P2, scheme level, secondary. does scheme stability predict scheme aim?

label A is taken **unchanged** from [`SCHEME_LEVEL_NOTE.md`](SCHEME_LEVEL_NOTE.md): each
scheme's `aim_auc`, computed on the seed-0 effects, so the label is identical to the one
already committed and isnt re-derived to suit this phase.

Spearman between each of T1-T5 and Label A across the 9 rows. max-statistic permutation
null, 20 000 shuffles of the labels within the whole set, taking the largest absolute rho
across all five signals each time. **declared separating** requires all three of the note's
bars: family-wise p < 0.05, abs rho ≥ 0.7, and sign consistent within **both** circuits.

Registered in advance as underpowered. n = 9 is smaller than the n = 13 that was already
inconclusive. this test is run for continuity with the note, and a null here carries little
information beyond what the note established. Saying so now is the point of saying it now.

### P3, the operational question. is a flagged disagreement reproducible?

this is the most on-target test in the phase and it gets exactly one statistic, named now.

Phase 8's flag put **17 heads** in docstring's primary blind spot, of which the published
head list contains **3** (`1.2`, `1.4`, `2.0`). for each flagged head let s\*(h) be the
non-primary scheme with the largest absolute effect at seed 0, phase 9's "best other
scheme", already tabulated in [`PHASE9_CHARACTERIZATION.md`](PHASE9_CHARACTERIZATION.md).
the statistic is

    S2( h, s*(h) )   is the disagreement that flagged this head reproducible?

AUC of the 3 published against the other 14, with an **exact** permutation null over all
C(17,3) = 680 labellings. bar: p < 0.05, uncorrected, cuz exactly one statistic is
being tested and it is named before the run.

Greater-than cant supply this test: its flagged set of 16 contains zero published
heads, so the AUC is undefined. stated now, not discovered later.

## 5. predictions, to be scored

recorded so this phase can fail its own expectations the way phases 7-10 did.

1. the seed-0 re-run reproduces the committed phase 8 effects to within 1e-4 for every head
   under every scheme on both circuits.
2. **no stability statistic clears the P1 bar against Bm**, on either circuit.
3. Bm beats B0, median AUC gain across the 9 rows ≥ +0.01. averaging ten runs helps. That
   is more data, not the hypothesis.
4. median across rows of Spearman(S2, Bm) ≥ **0.85**, SNR is close to a monotone transform
   of magnitude.
5. `rel_sd` is near-homoscedastic on both circuits: p75/p25 of sd across heads < **3**.
6. P2 inconclusive, with the best absolute rho below the median of its own permutation null.
7. P3 not significant (p ≥ 0.05).
8. docstring's `random_vocab_any`, power 0.06, theta 3.3, the scheme phase 9 established as
   pathological, has the **lowest** T1 of docstring's five schemes. it also had the
   *second-highest* Label A AUC of all thirteen rows, so if this holds, T1 disagrees with
   aim in the same direction theta does, and is another scale-integrity measure rather than an
   aim measure.
9. greater-than's `yy01` (label A 0.998) has the **highest** T1 of greater-than's four.

## 6. what a null result will be taken to mean, fixed now, not afterwards

The brief's step 6 asks for an explicit reading if stability fails. That reading depends on
the crux diagnostic, so both branches are written here before the numbers exist:

- If predictions 4 and 5 hold (SNR ≈ monotone in magnitude. per-head noise spread narrow),
  then the resampling axis carries almost **no head-specific information**: every head is
  measured with about the same replication error, so abs m ÷ sd is abs m divided by a
  near-constant, and no statistic built on this axis can add to magnitude. the conclusion to
  draw is that the noise here isnt "high-variance, low-sample-size" in character.
  Findings dont fail to replicate. they replicate perfectly well and are still
  uninformative about aim. That narrows what a fix could be: not more prompts, not more
  seeds, not a variance-aware threshold.
- **if prediction 4 or 5 fails** but P1/P2/P3 still find nothing, the reading is weaker, there *is* head-specific replication structure and it simply doesnt track the published
  head list. that would leave open the possibility that a different functional of the same
  axis works, and this phase would say so.

## 7. rules

- steps 1-3 import no `ground_truth` module. the script re-runs `assert_analysis_is_blind`
  over the same module list phases 4, 6, 8 and 9 use.
- The statistics S1-S4, T1-T5, the baselines, the three tests and their bars are frozen by
  this document. **nothing is redefined, no threshold is moved, and no second analysis is
  run after seeing the scores.** if nothing clears, that is the phase.
- The position-fixed variant of e is a footnote and is ineligible to be the result.
- whatever the outcome, the README and SYNTHESIS.md are updated to match, and the nine
  predictions above are scored in public.
