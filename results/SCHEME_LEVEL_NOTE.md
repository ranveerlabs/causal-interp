# A scheme-level re-analysis of Phase 9's discriminator signals

**Not a phase.** A single cheap re-analysis of measurements that are already committed, run
because [`SYNTHESIS.md`](../SYNTHESIS.md) §5 noticed that Phase 9 had tested its ten
candidate signals at the wrong level of description. No model was run for this note.

**Everything above the line marked `--- results ---` was written and committed before the
analysis script existed.** The order is checkable in git history rather than asserted, the
same way every phase in this repository handles it.

## Why the level matters

Phase 9 asked whether any pipeline-internal signal separates *a real blind spot* from
*ordinary disagreement*, and measured ten candidate signals over **33 flagged heads**. It
found one axis with visible separation and, in the report's own words, concluded that the
separation held "because docstring's noise came from one pathological scheme, not because
magnitude means what a discriminator would need it to mean."

That sentence locates the problem. The unit that was pathological was a **scheme**, not a
head. The judgement a reader actually has to make is not *do I believe this head* but *do I
believe this counterfactual* — and Phase 9 never ran its signals on that unit.

Stated as the supervised question SYNTHESIS.md §5 arrived at:

> Given a `(task, counterfactual)` pair and no answer key, predict whether that
> counterfactual is *aimed at* the behaviour or merely *damaging* the prompt.

## The 13 rows, and where every field comes from

Phase 9's floor table is thirteen `(circuit, scheme)` rows across three circuits. Nothing
here is recomputed from the model; every number is read back from a committed payload:

| field | source |
|---|---|
| scheme, provenance, primary, preserves_answer | `results/phase9_<circuit>.json` &rarr; `schemes` |
| θ, null median, null max, null mean, per-head null max, span | `results/phase9_<circuit>.json` &rarr; `floors` |
| power (= span ÷ primary's span) | `results/phase8_<circuit>.json` &rarr; `discovery/agreement/logit_diff/power`; for IOI, `results/phase9_ioi.json` (Phase 8 registered IOI's schemes and deliberately did not run it) |
| per-head effects, all three metrics | `results/phase8_<circuit>.json` &rarr; `discovery/runs/<scheme>/effects/<metric>`; for IOI, the same path inside `results/phase9_ioi.json` |
| published head list | `results/phase9_<circuit>.json` &rarr; `scored_before/per_scheme/<s>/matches` ∪ `misses`, which is identical across all schemes of a circuit and matches `meta.published_head_count` |

Phase 10's four frames also produce `(circuit, scheme)` rows, and SYNTHESIS.md cites them
for the labelling *principle* — `resample_t8` well-aimed, `resample_t7` not. They are **not
among the 13 and cannot be**: Phase 10 ran no null-calibration sweep, so its scheme rows
have no θ, no null median and no null max. Phase 10 informs how the label is defined below;
it contributes no rows to the analysis.

## What "label" means at the scheme level — stated before it is measured

This is the step that invalidates everything downstream if it is wrong, so it is fixed here
in full, with its known defects.

SYNTHESIS.md's informal labels are **not one rule**. Checked against the committed
per-scheme scores, they turn out to be three different rules wearing one name:

- greater-than's `yy01` is called well-aimed on **recall** (7/7).
- docstring's `random_def` is called well-aimed, and its primary `random_random` badly
  aimed, on **recall** (5/6 vs 3/6).
- `random_vocab_any` is called badly aimed everywhere — but on docstring its recall at the
  shared 0.02 criterion is **1.000**, because it discovers 25 of 32 heads. That label is
  really coming from its **precision** (0.240), and partly from Phase 9's θ.

Two consequences are fixed here in advance:

1. **Recall alone cannot be the label.** A scheme that fires on everything scores 1.000. The
   label must penalise the shotgun.
2. **θ must not enter the label.** θ is one of the signals under test. Labelling
   `random_vocab_any` badly-aimed *because* θ = 3.3 and then reporting that θ predicts the
   label would be circular. The label is built from the published head list and the raw
   effect arrays only.

### Label A (primary) — `aim_auc`, threshold-free

For each `(circuit, scheme)`: rank all of that circuit's heads by `|normalized recovery|`
under that scheme, and compute the AUC — the probability that a randomly chosen **published**
head outranks a randomly chosen unpublished one, ties counting a half.

This asks exactly what "aimed at the behaviour" should mean: does this counterfactual make
the circuit's heads stand out from the rest of the model? It is invariant to the scheme's
overall scale, which is what `power` confounds, and it never mentions a threshold, so it
cannot be contaminated by θ or by Phase 8's shared 0.02.

Reported continuously. A binary split at **AUC ≥ 0.80** is also reported; 0.80 is fixed here,
before any AUC has been computed, as a conventional "clearly better than chance" line.

### Label B (secondary) — `aim_f1`, at Phase 8's shared criterion

F1 of the scheme's discovered head set against the published head list at the shared 0.02
cutoff, read from `scored_before`. It inherits the flaw Phase 9 established — 0.02 does not
mean the same thing under two schemes — and is reported as a check on whether Label A's
answer is an artefact of the AUC construction, not as a competing ground truth.

**Disclosure:** per-scheme precision and recall at 0.02 were printed and read while locating
the data, *before* this labelling rule was written. Label B is therefore **not blind**, and
no binary cut is applied to it — it is used only as a continuous rank. Label A's AUC values
had not been computed when this section was committed.

### The defect both labels share

The published head list is the answer key for the **whole circuit**. A counterfactual aimed
at one sub-behaviour — docstring's `random_def`, which targets the induction match — is
legitimately aimed while ranking only part of the circuit highly. Phase 10 found the same
thing from the other direction. Neither label can tell *aimed at a sub-behaviour* from
*badly aimed*, and no result below should be read as if it could.

## The signals to be tested

Phase 9's ten first, each mapped to the closest scheme-level analogue of the same quantity.
Where a head-level signal asks "where does this head sit inside its scheme's distribution",
the scheme-level analogue asks "what shape is that distribution":

| # | Phase 9 head-level signal | scheme-level analogue tested here |
|---|---|---|
| 1 | primary's abs effect ÷ 0.02 | scheme's median abs effect ÷ 0.02 |
| 2 | primary's abs effect ÷ its median | scheme's max ÷ its own median |
| 3 | best other scheme's abs effect ÷ primary's | scheme's max ÷ the primary's max |
| 4 | best scheme's abs effect ÷ that scheme's median | scheme's p90 ÷ its own median |
| 5 | best scheme's abs effect ÷ that scheme's p90 | scheme's max ÷ its own p90 |
| 6 | rank of the head inside the scheme that found it | how far down its own list the scheme reaches: heads over 0.02 ÷ heads swept |
| 7 | how many non-primary schemes found it | heads the scheme discovers at 0.02 |
| 8 | abs effect ÷ the strongest head in that scheme | scheme's median ÷ its max (how concentrated it is) |
| 9 | raw normalized recovery in the scheme that found it | scheme's max abs normalized recovery |
| 10 | how many of three metrics put it over 0.02 | share of the scheme's discovered heads that clear 0.02 under all three metrics |

Then the scheme-level fields Phase 9 stored but never tested as discriminators, including
the θ-against-its-own-null family SYNTHESIS.md and this note's brief both name:

| signal | definition |
|---|---|
| `power` | span ÷ primary's span — Phase 8's measure |
| `theta` | the per-scheme null floor, Phase 9's one working criterion |
| `null_median`, `null_max` | the shuffled-source null's centre and tail |
| `theta_over_null_median` | how far the 99th percentile of the null sits above its centre |
| `null_max_over_median` | how heavy the null's own tail is |
| `theta_over_own_median` | θ against the scheme's **real** median effect — the scheme-level form of "raw score against its own null spread" |
| `span` | the raw clean-vs-corrupted distance |
| `spearman_with_primary`, `jaccard_with_primary` | Phase 9's concordance quantities, computed for all three circuits |

Provenance (`published` / `authored` / `generic`) and `preserves_answer` are recorded but are
not pipeline-internal signals — they are the experimenter's own labels — so they are reported
separately and are not eligible to be the answer.

## What will count as a result — fixed before the numbers exist

n = 13, in three strata of 5, 4 and 4, and roughly twenty signals are being tested. That
combination will manufacture a strong-looking correlation by chance, so the bar is set here
rather than after:

1. **Primary statistic:** Spearman rank correlation between each signal and Label A across
   all 13 rows.
2. **Multiplicity:** a max-statistic permutation null — shuffle the labels within the whole
   set 20,000 times, take the largest `|ρ|` across *all* signals each time, and read the
   observed best signal against that distribution. This is the family-wise correction and it
   is what will be quoted, not the per-signal p-value.
3. **Stratification:** because circuit sizes differ enormously (6 published heads of 32 vs
   26 of 144), a signal only counts if its sign is **consistent within all three circuits**.
   A cross-circuit correlation with inconsistent within-circuit signs is a circuit effect
   wearing a signal's clothes.
4. **Declared separating** requires all three: family-wise p < 0.05, `|ρ| ≥ 0.7`, and
   consistent within-circuit sign.
5. Anything short of that is reported as **inconclusive at n = 13**, with the plot, and no
   interpretation of which signal came closest.
6. **Whatever the outcome**, no signal is redefined, no threshold is moved, and no second
   analysis is run after seeing these results. If nothing clears the bar, that is the note.

Expected outcome, recorded so it can be scored: **inconclusive**. Thirteen rows against
twenty signals is not enough to establish a discriminator, and the honest prior is that the
best-looking signal will fail the permutation test.

--- results ---
