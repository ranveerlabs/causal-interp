# Phase 11 — replication under resampling. Negative, and for a reason.

**Pre-registered in [`PHASE11_PLAN.md`](PHASE11_PLAN.md)**, committed before
`scripts/run_phase11_resample.py` existed. The analysis script was committed while the
sweeps were still running, so no statistic here can have been chosen to fit a number.
Everything below the plan is measurement.

| | |
|---|---|
| resampling axis | dataset seed — redraws prompts and corruption instances jointly |
| R | 10 (seeds 0–9), n = 128 per resample, metric `logit_diff` |
| circuits | docstring (`attn-only-4l`), greater-than (`gpt2-small`) — 9 `(circuit, scheme)` rows |
| sweeps | 90 scheme-sweeps, 1.26 GPU-hours |
| code changed in `causal_interp/` | **none** |

## The one-paragraph answer

**Stability under resampling does not separate real findings from noise. It is measurably
*worse* than the magnitude it was supposed to improve on, and the reason is mechanical
rather than statistical.** Across the 9 rows, replacing the effect's magnitude with its
replication signal-to-noise ratio *lowers* published-head AUC by a median of 0.045
(exact Wilcoxon p = 0.008, against a pre-registered bar of 0.017). The mechanism is that
replication noise **scales with the effect** — Spearman(sd, |effect|) runs +0.58 to +0.86
in every one of the 9 rows, and published heads' median replication sd is 2.4× to 17.5×
that of the rest — so dividing by it is a compressive transform that discards exactly the
resolution at the top of the ranking where the circuit lives. The deeper finding is the one
that matters for what comes next: **the pipeline's findings already replicate almost
perfectly.** In 7 of the 8 scheme rows where the question is defined, *every* head
discovered at seed 0 clears its bar again in all ten resamples. Whatever is wrong with
these findings, it is not that they wobble.

---

## The registered tests

### P1 — head level, primary. **Stability loses to magnitude, significantly.**

AUC of published-vs-unpublished head separation, per `(circuit, scheme)` row.
**Bm** = magnitude averaged over all ten resamples, the registered baseline;
**B0** = the single seed-0 magnitude, i.e. what the pipeline reports today.

| circuit | scheme | S1 hit-frac | S2 snr | S3 sign | **Bm** | B0 |
|---|---|---|---|---|---|---|
| docstring | `random_random` *(primary)* | 0.731 | 0.795 | 0.699 | **0.872** | 0.840 |
| docstring | `random_def` | 0.894 | 0.878 | 0.712 | **0.923** | 0.859 |
| docstring | `random_answer` | 0.721 | 0.577 | 0.426 | **0.615** | 0.660 |
| docstring | `random_vocab_cdef` | 0.740 | 0.756 | 0.635 | **0.750** | 0.712 |
| docstring | `random_vocab_any` | 0.500 | 0.897 | 0.840 | **0.929** | 0.962 |
| greater_than | `yy01` *(primary)* | 0.993 | 0.981 | 0.569 | **0.998** | 0.998 |
| greater_than | `xx_mismatch` | 0.761 | 0.764 | 0.628 | **0.940** | 0.941 |
| greater_than | `random_vocab_yy` | 0.769 | 0.662 | 0.664 | **0.857** | 0.932 |
| greater_than | `random_vocab_any` | 0.614 | 0.442 | 0.525 | **0.851** | 0.593 |

| candidate | median AUC gain vs Bm | exact Wilcoxon p | clears bar (+0.05 and p < 0.0167) |
|---|---|---|---|
| S1 `hit_fraction` | **−0.088** | 0.039 | no |
| S2 `snr` | **−0.045** | **0.008** | no |
| S3 `sign_consistency` | **−0.193** | **0.004** | no |

Not merely "no improvement". Two of the three are significantly *worse* than magnitude at
the phase's own alpha, and the third is worse at p = 0.039. **Prediction 2 held.**

**Prediction 3 was wrong, and the way it was wrong is informative.** Averaging ten
independent resamples does not improve head-level AUC over one run: median gain
**+0.0000**, Wilcoxon p = 0.77, with four rows going down. Ten times the data buys nothing
here, because the single-run ranking is already at this axis's information ceiling.

### P2 — scheme level, secondary. Inconclusive, as registered.

Label A is the `aim_auc` from [`SCHEME_LEVEL_NOTE.md`](SCHEME_LEVEL_NOTE.md), read back
unchanged rather than re-derived.

| circuit | scheme | Label A | T1 rank-repro | T2 set-repro | T3 med hit | T4 med snr | T5 span CV | discovered at seed 0 |
|---|---|---|---|---|---|---|---|---|
| docstring | `random_random` | 0.840 | 0.857 | 1.000 | 1.000 | 20.9 | 0.016 | 4 |
| docstring | `random_def` | 0.859 | 0.876 | 0.949 | 1.000 | 16.3 | 0.062 | 7 |
| docstring | `random_answer` | 0.660 | 0.873 | 0.929 | 1.000 | 25.2 | 0.019 | 5 |
| docstring | `random_vocab_cdef` | 0.712 | 0.950 | 0.950 | 1.000 | 20.2 | 0.017 | 3 |
| docstring | `random_vocab_any` | **0.962** | **0.718** | — | — | — | 0.223 | **0** |
| greater_than | `yy01` | 0.998 | **0.967** | 0.918 | 1.000 | 16.2 | 0.037 | 9 |
| greater_than | `xx_mismatch` | 0.941 | 0.926 | 0.788 | 1.000 | 6.1 | 0.062 | 10 |
| greater_than | `random_vocab_yy` | 0.932 | 0.454 | 0.379 | 0.500 | 1.2 | 0.018 | 6 |
| greater_than | `random_vocab_any` | 0.593 | 0.842 | 0.620 | 1.000 | 6.6 | 0.063 | 2 |

| signal | ρ vs Label A | n | family-wise p | within-circuit signs agree |
|---|---|---|---|---|
| `t4_median_snr` | −0.476 | 8 | 0.651 | no |
| `t5_span_cv` | +0.217 | 9 | 0.986 | no |
| `t3_median_hit_fraction` | −0.247 | 8 | 0.978 | yes |
| `t2_set_reproducibility` | −0.190 | 8 | 0.992 | yes |
| `t1_rank_reproducibility` | +0.167 | 9 | 0.996 | no |

Nothing declared separating. The best of the five reaches |ρ| = 0.476 where shuffled
labels reach **0.571 half the time** and 0.810 one time in twenty. **Prediction 6 held.**
This was registered in advance as underpowered at n = 9 and it is; the result adds little
beyond the note's n = 13.

Two details worth recording. `t2_set_reproducibility` correlates *positively* with aim
inside each circuit (+0.40, +0.80) and *negatively* across the pooled nine — a textbook
circuit effect, caught by the sign-consistency rule the note fixed for exactly this.
And `t3` is constant at 1.000 across all of docstring's scoreable rows, so its
within-docstring correlation is undefined.

**Predictions 8 and 9 both held**, and together they say what T1 actually is. Docstring's
`random_vocab_any` — power 0.06, θ 3.3, the scheme Phase 9 established as pathological —
has the **lowest** T1 of its five (0.718) while carrying the **second-highest Label A of
all thirteen rows** (0.962). Greater-than's `yy01` has the **highest** T1 of its four
(0.967). T1 ranks schemes the way θ does and disagrees with aim in the same place θ
disagrees with it. **Rank reproducibility is a third measurement of scale integrity, not a
measurement of aim** — which is the scheme-level note's conclusion arrived at from a new
direction.

### P3 — the operational question. **Passed as registered, and beaten by magnitude.**

Among the 17 heads Phase 8's flag put in docstring's primary blind spot, 3 are published
(`1.2`, `1.4`, `2.0`). The single registered statistic was S2 at each head's best other
scheme — is the disagreement that flagged it reproducible?

**AUC = 0.833, exact p = 0.046 over all 680 labellings.** That clears the registered bar.
**Prediction 7 was wrong**, in the direction of the hypothesis working.

It does not survive contact with its comparator. The plan named one statistic in advance,
which is correct pre-registration and leaves the obvious question unasked; reporting a pass
without answering it would be misleading. Computed **post-hoc** by
`scripts/phase11_posthoc.py`, on the identical 17 heads:

| ranking of the same 17 flagged heads | AUC | exact p |
|---|---|---|
| plain magnitude, seed 0 (`b0`) | **0.976** | **0.0029** |
| plain magnitude, 10-resample mean (`bm`) | 0.929 | 0.0103 |
| **S2 snr — the registered statistic** | **0.833** | **0.0456** |
| S1 hit fraction | 0.798 | 0.0632 |
| scheme identity (is the flagger `random_def`?) | 0.726 | 0.1912 |

Stability passes a bar that magnitude clears far more convincingly, on the same data. That
is the P1 result again at the unit where it was supposed to matter most. Two further
reasons it cannot be banked: greater-than's flagged set of 16 contains **zero** published
heads, so this test cannot be run there at all — and that is precisely the case where a
useful rule would have to stay silent; and n = 17 with 3 positives on one circuit is a
single bit of evidence however it lands.

The one thing here worth carrying forward is **not** about stability. *Magnitude at the
disagreeing scheme separates published from unpublished inside docstring's flagged set at
AUC 0.976, p = 0.0029* — a question Phase 9 never asked, because its head-level analysis
compared docstring's flagged heads against greater-than's, not published against
unpublished within one. It is post-hoc, single-circuit, and untestable on the other
circuit. It is recorded so a future analysis with more circuits can check it, not because
this one found anything.

---

## Why stability lost — the mechanism

Registered as a diagnostic, not as a candidate, so that the reading of a null could not be
invented afterwards.

| circuit | scheme | Spearman(S2, Bm) | sd p25 | sd p50 | sd p75 | p75/p25 |
|---|---|---|---|---|---|---|
| docstring | `random_random` | 0.898 | 0.00243 | 0.00347 | 0.00700 | 2.88 |
| docstring | `random_def` | 0.865 | 0.00472 | 0.00695 | 0.01591 | 3.37 |
| docstring | `random_answer` | 0.924 | 0.00142 | 0.00243 | 0.00351 | 2.47 |
| docstring | `random_vocab_cdef` | 0.931 | 0.00156 | 0.00244 | 0.00513 | 3.28 |
| docstring | `random_vocab_any` | 0.907 | 0.01997 | 0.02774 | 0.04880 | 2.44 |
| greater_than | `yy01` | 0.673 | 0.00015 | 0.00031 | 0.00075 | 4.98 |
| greater_than | `xx_mismatch` | 0.625 | 0.00017 | 0.00046 | 0.00133 | 7.92 |
| greater_than | `random_vocab_yy` | 0.734 | 0.00036 | 0.00057 | 0.00120 | 3.36 |
| greater_than | `random_vocab_any` | 0.744 | 0.00109 | 0.00221 | 0.00441 | 4.06 |

**Prediction 4 held** — median Spearman(S2, Bm) = 0.865, so SNR is largely a monotone
re-expression of magnitude. **Prediction 5 was wrong**: greater-than is not homoscedastic
(median p75/p25 = 4.52, and 0 of its 4 rows under the bar), though docstring nearly is
(median 2.88, 3 of 5 under).

That combination sent the reading to the plan's **second** branch: there *is* head-specific
replication structure, and it does not track the published head list. The post-hoc
diagnostic says why, and says something stronger than "does not track":

| circuit | scheme | Spearman(sd, \|effect\|) | median sd, published heads | median sd, the rest | ratio |
|---|---|---|---|---|---|
| docstring | `random_random` | +0.585 | 0.00991 | 0.00316 | 3.1× |
| docstring | `random_def` | +0.577 | 0.03203 | 0.00656 | 4.9× |
| docstring | `random_answer` | +0.738 | 0.00558 | 0.00236 | 2.4× |
| docstring | `random_vocab_cdef` | +0.863 | 0.00889 | 0.00226 | 3.9× |
| docstring | `random_vocab_any` | +0.663 | 0.06252 | 0.02511 | 2.5× |
| greater_than | `yy01` | +0.780 | 0.00503 | 0.00029 | **17.5×** |
| greater_than | `xx_mismatch` | +0.759 | 0.00455 | 0.00037 | **12.3×** |
| greater_than | `random_vocab_yy` | +0.710 | 0.00544 | 0.00054 | 10.2× |
| greater_than | `random_vocab_any` | +0.663 | 0.01139 | 0.00211 | 5.4× |

**The noise is multiplicative.** A head with a real causal effect is measured with a
replication sd several times larger, in absolute terms, than a head with no effect — up to
17× on greater-than. So sd ≈ a + b·|effect|, and S2 = |effect| ÷ (a + b·|effect|) is a
*saturating* function of magnitude: it preserves the ordering at the bottom, compresses it
at the top, and adds the estimation error of a 10-sample sd on the way. Compression at the
top is precisely where the published circuit sits. That is the whole of P1's negative, and
it also explains why S2's AUC lands consistently between 0.5 and Bm rather than near either.

**The hypothesis this phase tested was empirically backwards.** Real components are not
*more* stable in absolute terms; they are *noisier*, because their variance scales with
their effect. In relative terms they are about as stable as everything else, which is why
the ratio carries so little.

## The finding that constrains what comes next

Set aside the discriminator question. The resampling data says something about the pipeline
that no previous phase could:

> **Discovery is already reproducible.** In 7 of the 8 scheme rows where the question is
> defined, the median head discovered at seed 0 clears its bar in **10 of 10** independent
> resamples (T3 = 1.000). Set reproducibility across resamples runs 0.62–1.00 for 7 of the
> 8. Ten-fold averaging changes head-level AUC by a median of exactly zero.

The single exception, `random_vocab_yy` on greater-than (T2 = 0.379, T3 = 0.500), is a
generic scheme with a small span, and it is the one place a variance-aware criterion would
have had something to fix.

So **the noise this project has been fighting is not sampling noise.** It is not "we drew
128 prompts and got unlucky", and it will not yield to more prompts, more seeds, a
variance-aware threshold, or a confidence interval. The findings are stable, valid
(Phase 9), correctly scaled (Phase 9's θ), and reproducible (here) — and still no
answer-key-free quantity says whether they are *about the behaviour under study*. That
narrows the space of possible fixes considerably, by ruling out the entire family that
treats the problem as statistical.

## Predictions — 6 of 9 held

| # | prediction | outcome | |
|---|---|---|---|
| 1 | seed 0 reproduces Phase 8's effects to < 1e-4 | max \|Δ\| = **0.00e+00**, bit-identical on both circuits | **hit** |
| 2 | no stability statistic clears the P1 bar | none did; two were significantly *worse* | **hit** |
| 3 | Bm beats B0 by median AUC ≥ +0.01 | median **+0.0000**, Wilcoxon p = 0.77 | **miss** |
| 4 | median Spearman(S2, Bm) ≥ 0.85 | **0.865** | **hit** |
| 5 | per-head sd near-homoscedastic, p75/p25 < 3 | docstring 3/5 rows (median 2.88); **greater-than 0/4** (median 4.52) | **miss** |
| 6 | P2 inconclusive, best \|ρ\| below its null median | best 0.476 vs null median 0.571 | **hit** |
| 7 | P3 not significant | **AUC 0.833, p = 0.046 — significant** | **miss** |
| 8 | docstring `random_vocab_any` has the lowest T1 | lowest, 0.718 | **hit** |
| 9 | greater-than `yy01` has the highest T1 | highest, 0.967 | **hit** |

Three misses, and the interesting one is 7 — the phase's own most on-target test came out
*against* the pre-registration's expectation, and then failed to survive its comparator.
Registering the expectation is what made that visible as a two-step story rather than a
headline.

## Deviations and disclosures

- **Two plan gaps, resolved in code and marked `PLAN GAP`.** Jaccard of two empty
  discovered sets is treated as undefined and the pair skipped, rather than 1.0: docstring's
  `random_vocab_any` discovers nothing at θ = 3.3 in any resample, and calling that perfect
  set reproducibility would be the most misleading available answer. Its T2/T3/T4 are
  therefore blank and P2 runs at n = 8 for those three signals. An infinite SNR (sd exactly
  zero) was given a large finite stand-in; **no head on either circuit had sd = 0**, so the
  branch never fired.
- **P3's permutation test is one-sided** (does the statistic rank published heads *higher*).
  The plan did not say; one-sided is the direction the hypothesis names.
- **Prediction 5 is scored on the median p75/p25 across each circuit's scheme rows.** The
  plan wrote the bar per circuit without saying how to pool; the full per-row table is above
  so the pooling choice can be checked rather than trusted.
- **The position-fixed variant of e was computed and is ineligible to be the result**, as
  registered. It is stored in `phase11_stability.json` under `heads_position_fixed`.
- **The two post-hoc analyses are in a separate script** (`scripts/phase11_posthoc.py`) and
  a separate payload, both labelled, and neither changes a verdict.
- **The measurement ran contended for its first 15 minutes** — docstring and greater-than
  shared one 8 GiB GPU, memory reached 7.8 GiB, and the docstring job was killed and
  restarted from seed 1 after the two processes had slowed each other roughly threefold.
  Nothing measured was affected: each seed's payload is written whole or not at all, seed 0
  was verified bit-identical to Phase 8, and per-seed runtimes are stored in
  `phase11_stability.json`.

## Artefacts

| file | what it holds |
|---|---|
| `phase11_{circuit}_seed{0..9}.json` | the 20 raw resampled sweeps — `logit_diff` grids and spans |
| `phase11_stability.json` | Part A: S1–S4, T1–T5, the crux diagnostic. Written before any answer key was read |
| `phase11_tests.json` | Part B: P1, P2, P3, and the nine scored predictions |
| `phase11_head_stability.csv` | every (scheme, head): mean, sd, S1–S4, both baselines, published flag |
| `phase11_posthoc.json` | the two post-hoc diagnostics, labelled as such |

## Verdict

Replication is a genuinely different class of signal from magnitude, from null-comparison
and from ranking, and it was the last untested one this pipeline can compute about itself.
It does not work, it does not work for a reason that is now measured rather than guessed,
and the reason rules out the family it belongs to.

Four attempts have now failed at the same question — Phase 9 on validity, Phase 10 on
magnitude, the scheme-level note on all twenty of those signals at the unit that matters,
and Phase 11 on replication. The first three were searches that came up empty. This one
returns something more useful than another empty search: **the measurements are not
unreliable.** They are reproducible to the point where ten-fold averaging changes nothing.
The gap between "this head has a large, valid, reproducible causal effect under this
counterfactual" and "this head is part of the circuit implementing the behaviour" is not a
gap that any amount of measurement quality closes, because it is not a measurement gap.

The recommendation is to stop looking for the fix inside this pipeline's paradigm and treat
it as an open problem. What would change that is stated in `SYNTHESIS.md` §5 and is
unchanged by this phase: either more published circuits, so that aim becomes a supervised
problem at a sample size where a signal of moderate strength could be detected at all — or
a criterion that references the behaviour under study independently of the counterfactual
being graded, for which the current architecture has no slot.
