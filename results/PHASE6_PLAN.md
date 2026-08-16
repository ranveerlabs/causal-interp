# Phase 6 plan — applying the pipeline to a second published circuit

**Written and committed before any Phase 6 code was written, and before any Phase 6
measurement existed.** Its purpose is to fix the target, the ground truth, the
scoring rules and the predictions in advance, so that a poor result cannot be
converted into a good one by adjusting any of them afterwards.

Everything in Phases 1–5 was built, tuned and validated against a single circuit:
GPT-2 small's IOI circuit. Every threshold, every corruption scheme, every metric
and every criterion was chosen while looking at that one answer key. Nothing so
far distinguishes *a method that works* from *a method fitted to IOI*. This phase
runs the existing pipeline, unmodified, against a circuit it was not built around.

---

## 1. Target: the greater-than circuit

**Chosen: the "greater-than" circuit in GPT-2 small**, from Hanna, Liu, Variengien
(2023), [*How does GPT-2 compute greater-than?*](https://arxiv.org/abs/2305.00586),
NeurIPS 2023.

The brief allowed induction heads as an alternative and asked for greater-than
*if* its published circuit has head-level ground truth as specific as IOI's. It
does: the paper names seven attention heads individually, and the authors' own
code release lists exactly those seven as a literal Python list. That is the
property that makes head-by-head recovery comparable with Phases 1–5. Induction
heads would have given a broader, fuzzier target — "heads that do induction" is a
behavioural description, not a fixed component list — so recovery against it
could not be scored the same way.

Three further reasons it is the right second target:

- **Different task structure.** Numeric comparison rather than name tracking. The
  information being moved is an ordinal quantity, not an identity.
- **Different circuit shape.** IOI's published circuit is 26 attention heads and
  no MLPs. Greater-than's is 7 attention heads and 4 MLPs, and the paper's account
  puts the *MLPs* at the centre of the computation with the heads feeding them.
  A head-centric pipeline meeting an MLP-centric circuit is a genuine stress test.
- **Independently published**, by a different group, with its own counterfactual
  and its own metric — neither of which this project has ever seen.

**A limitation this phase does not remove, stated now rather than discovered
later:** greater-than lives in the *same model* as IOI, GPT-2 small. So this tests
whether the pipeline generalises across **tasks and circuits**, not across
**models**. Any claim of model-generality would need a third target and is not
made here.

## 2. The published ground truth

**Primary — the circuit, 7 attention heads:**

| head | the paper's account |
|---|---|
| a9.h1 | feeds MLP 9 |
| a8.h11, a8.h8, a7.h10, a6.h9, a5.h5, a5.h1 | feed MLP 8 |

> "MLP 9 relies on a9.h1, while MLP 8 relies on a8.h11, a8.h8, a7.h10, a6.h9,
> a5.h5, and a5.h1"

Cross-checked, exactly as `ground_truth.py` was cross-checked for IOI, against the
authors' code release ([hannamw/gpt2-greater-than](https://github.com/hannamw/gpt2-greater-than)),
which lists the circuit heads as `[(9, 1), (8, 11), (7, 10), (6, 9), (5, 5), (8, 8), (5, 1)]`
— the same seven.

**Also published, and measured separately: MLPs 8, 9, 10, 11.** These are not
attention heads and are not scored in the head-level comparison. They are swept
with the existing `sweep_component` and reported beside it, because a circuit
whose published centre is its MLPs cannot be honestly summarised by a head count
alone.

**Secondary — the extended set, declared now so it cannot be adopted later.**
Appendix B path-patches what the seven heads themselves depend on and reports
"mostly MLPs 0-3, as well as a0.h5, a0.h3, and a0.h1". The paper treats these as
upstream dependencies discussed in an appendix, not as members of the circuit it
reports, and the authors' code excludes them. **The primary comparison therefore
uses the 7 heads.** A secondary line additionally counts a0.h1, a0.h3, a0.h5 as
members (10 heads). Both are fixed here. Whichever scores better afterwards, the
7-head number is the headline.

**Receiver specification.** Unlike IOI — where only three of seven classes had a
published receiver spec — the paper states one for all seven heads:

> "The most important influences on these heads are the influences on their values
> at the YY position."

So the published receiver spec is **`v@YY`** for every circuit head, and the
Phase 4 rediscovery check is scoreable on all seven.

## 3. Task construction

Task construction is **supplied**, exactly as it was for IOI — the honest ladder in
the README puts "task template" above the line this project has crossed, and Phase 6
does not attempt to cross it. What is supplied is the paper's own construction, not
a new one tuned here.

- **Template**: `The {noun} lasted from the year {XX}{YY} to the year {XX}`
- **Sampling**: century `XX` from `{11…17}`, start year `YY` from `{02…98}` — the
  paper's ranges.
- **Nouns**: filtered to single-token, and re-verified in the constructor rather
  than trusted, mirroring `IOIDataset._single_token_names`.
- **Positions**: `NOUN`, `XX1`, `YY`, `YY+1`, `END`. Verified by tokenizer probe:
  `" 1732"` → `[" 17", "32"]`, all of `"00"…"99"` are single tokens, and every
  century `" 11"…" 17"` is a single token, so every prompt has identical length and
  clean/corrupted pairs align token-for-token.

**Corruption schemes.** Four, mirroring Phase 5's four:

| scheme | knowledge used | source |
|---|---|---|
| `yy01` | the task's own counterfactual | the paper's: replace `YY` with `01` |
| `random_vocab_yy` | position only | Phase 5's generic substitution, anchored at `YY` |
| `random_vocab_any` | **none** | Phase 5's generic substitution, position drawn uniformly |

Only **three**, not four: IOI had two hand-built schemes (`s2_swap`, `abc`), but
greater-than has exactly one published counterfactual. Inventing a second would be
new hand-tuning of precisely the kind this phase exists to detect, so it is not
done. The generic pair is the *existing Phase 5 code path*, not a reimplementation —
it is extracted to a shared module so both tasks call the same function, and the
extraction is verified bit-identical for IOI before it is used here.

**Metric.** Both options Phase 5 compared:

- hand-built — the paper's probability difference, `Σ_{y>YY} p_y − Σ_{y≤YY} p_y`,
  matching the authors' `prob_diff` implementation;
- generic — the existing `metrics.py` KL and total variation, **unmodified**.

## 4. Thresholds — inherited, not chosen

| threshold | treatment |
|---|---|
| activation-patching cutoff `0.02` | **inherited unchanged** from Phase 1, as Phase 5 did |
| size-matched set | top-**7**, the published head count, as Phase 1 used top-26 |
| receiver-side `path_signal` cutoff | **rule inherited, number recalibrated** |

The receiver-side rule is Phase 3's, verbatim:

> threshold = 99th percentile of `|path_signal|` under a shuffled-source null,
> rounded up to two significant figures

The *number* must be recalibrated because the null is task-specific — Phase 3
itself showed the null is heavy-tailed and that inheriting Phase 1's `0.02` would
have carried a large false-positive rate. Recalibrating the null under the same
rule is not choosing a new threshold; **picking a different rule, or adjusting the
quantile, would be**, and neither is permitted here. The recalibration runs in a
`--preregister` step that computes no real measurement and is committed before the
comparison exists, exactly as Phase 3's was.

## 5. Predictions

Recorded so they can be scored, in the same spirit as Phase 1's prediction about
path patching, which turned out to be **wrong** and was reported as such.

1. **Structural blindness will reappear.** `yy01` changes only the `YY` token, so
   `NOUN` and `XX1` should be bit-identical between the two runs and every
   head-position cell there should be an *exact* floating-point zero. Measured,
   not assumed.
2. **Activation patching will recover ≥ 5 of the 7 heads.** Phase 1's failure mode
   was heads acting *through another head*; here all seven act through **MLPs**,
   which ordinary activation patching propagates through rather than blocking. If
   this fails, Phase 1's structural story does not transfer.
3. **Precision will be worse than IOI's.** 7 targets among 144 heads is a far
   harsher denominator than 26, and the published centre of this circuit is its
   MLPs, so heads outside the circuit may carry real effect.
4. **Phase 5's corruption result will reproduce**: fully generic (generic
   corruption + generic metric) will recover fewer heads than the hand-built pair.

## 6. What counts as a failure of this phase

Stated in advance so the result cannot be re-framed afterwards:

- Recovery near chance (top-7 containing ~0–1 published heads) would say the
  pipeline was IOI-specific.
- Needing to change `interventions.py`, `search.py` or `metrics.py` to make the run
  work at all would say the machinery encodes task assumptions it does not admit to.
- Needing to change a threshold, a rule, or the metric *after seeing a score* would
  invalidate the phase regardless of the number it produced.

Adjustments to imports, prompt formatting, and a ground-truth module are **expected
and do not count as failures** — they are the task-specific surface any second task
would need. Section 6 of the report will itemise exactly which files were touched
and which transferred as pure reuse, because that itemisation, not the recovery
number, is the direct measure of generality.
