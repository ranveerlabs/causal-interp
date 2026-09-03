# phase 12, causal scrubbing. it passes, and a pass is worth almost nothing.

Pre-registered in [`PHASE12_PLAN.md`](PHASE12_PLAN.md), committed before
`causal_interp/scrubbing.py` existed. the analysis script went in two minutes into the run,
well before the null stage started at 682 s. Everything below the plan is measurement.

| | |
|---|---|
| method | resample ablation on clean prompts, Redwood's causal scrubbing with a degenerate `I` |
| task | IOI, `gpt2-small`, n = 128, seed 0 |
| resample sources | 8 row-aligned IOIDatasets at seeds 1000-1007, independent draw per head per row |
| candidates scored | 9 named, 260 drop/add sets, 4200 random sets, 200 layer-matched, 26 loo, 144 aoi |
| forward passes | ~31,000, 4825.7 s |
| counterfactual pairs built | **zero** |

## the one-paragraph answer

**Causal scrubbing is a genuinely different measurement and it hits a ceiling of its own,
in a different place and for a different reason than everything before it.** the published
26-head circuit passes its scrub, recovering **1.022 ± 0.025** of the logit difference and
0.870 of the KL, so P1 clears both bars. but the pass carries almost no information.
**25 of 200 uniformly random 26-head sets clear the same bar**, and 8 of them beat the
published circuit outright. **dropping 12 of the 26 published heads still scores 0.911**,
and 30% of those 12-head deletions land at or above the intact circuit. a **3-head** set
from phase 1's greedy trace scores 0.968. So sufficiency, as this method measures it, is
satisfied by a very large family of head sets and does not distinguish the published circuit
from most of them. what no prior phase could do, it does at head level. its per-head
importance ranking is **near-orthogonal to activation patching** (Spearman 0.240) and puts
the previous-token heads `4.11` and `2.2` at rank 1 and 3, heads phase 1's counterfactual
scores at |effect| 0.0028 and 0.0023 and structurally cannot see.

---

## implementation gates

registered in §4 of the plan and run before any result was read.

| gate | expected | measured |
|---|---|---|
| `H` = all 144 heads | recovered 1.0, logits identical to clean | 1.0, `torch.equal` **True** |
| `H` = {} | recovered 0.0, behaviour destroyed | -0.0031, raw logit diff **3.697 -> 0.175** |
| sources = the clean dataset itself | recovered 1.0 for any `H` | raw logit diff 3.696538 for both `H` = {} and a 70-head set, KL exactly 0 |

all three pass. gate 2's -0.0031 is the difference between two independent 20-draw
estimates of the same floor, one taken in the anchor and one in the gate, and sits inside
the draw noise.

anchors: `E_model` = 3.6965 logit diff, top-1-is-IO 0.9297. `E_0` = 0.1751 logit diff,
KL 3.408, top-1-is-IO 0.0008.

## P1, the sanity check. passed.

| | recovered | sd over 20 draws | bar | |
|---|---|---|---|---|
| `logit_diff` **primary** | **1.022** | 0.025 | ≥ 0.70 | pass |
| `kl` | **0.870** | 0.004 | ≥ 0.50 | pass |
| `top1_is_io` | 0.604 | 0.030 | none registered | |

the raw scrubbed logit difference is 3.773 against the clean model's 3.697. resample-ablating
the 118 heads outside the published circuit leaves the IO-vs-S margin very slightly *larger*
than leaving the model alone. Chan et al. say the % recovered isnt really a fraction and can
exceed 100%, and here it does.

**prediction 2 was wrong and the direction matters.** the plan expected below 0.87, reasoning
that resample ablation is harsher than the mean ablation behind Wang et al.'s 87%. it isnt,
not here. the reason is visible in `top1_is_io`, which recovers only 0.604 while the logit
difference recovers fully. the scrub preserves the *ordering* of the two name tokens and
does considerable damage to the rest of the distribution, and logit difference is blind to
that by construction. KL at 0.870 sits between the two, which is what a distributional
metric should do. first place in twelve phases where the choice of metric moves a headline
this much. the primary was registered as `logit_diff` for comparability with the published
number and not because its the better measure of "behaviour preserved".

the `bypos` variant, where each published head survives only at its class's expected
position, gives 1.012 against `allpos`'s 1.022. the position restriction the paper's account
implies costs essentially nothing here.

## P2, the pipeline's own circuits

`residual` is the score in sd units of 200 random head sets of the same size, which is the
only version of these numbers worth comparing across rows.

| candidate | size | published | `logit_diff` | sd | `kl` | `top1` | residual |
|---|---|---|---|---|---|---|---|
| `published_26` | 26 | 26 | **1.022** | 0.025 | 0.870 | 0.604 | +2.26 |
| `phase1_agree` | 13 | 13 | **1.124** | 0.026 | 0.820 | 0.440 | **+3.68** |
| `phase1_abc` | 18 | 15 | **1.031** | 0.016 | 0.852 | 0.534 | +2.41 |
| `greedy_abc` | 3 | 2 | **0.968** | 0.037 | 0.649 | 0.188 | **+6.49** |
| `phase1_s2_swap` | 23 | 18 | 0.575 | 0.020 | 0.757 | 0.328 | +0.74 |
| `phase1_matches` | 20 | 20 | 0.533 | 0.019 | 0.789 | 0.374 | +0.76 |
| `phase1_union` | 28 | 20 | 0.528 | 0.019 | 0.799 | 0.388 | +0.34 |
| `greedy_s2_swap` | 6 | 6 | 0.143 | 0.047 | -0.011 | 0.002 | +0.41 |
| `phase1_extras` | 8 | 0 | 0.014 | 0.028 | 0.001 | 0.001 | -0.52 |

four rows clear P1's own 0.70 bar and only one of them is the published circuit. the
13-head `phase1_agree` beats it. so does an 18-head set with 3 wrong heads in it. so does a
**3-head** set, `9.9 10.0 0.5`, of which one head isnt in the published circuit at all.

`phase1_union` is the one that says most about the method. it is a strict superset of
`phase1_abc`, 28 heads against 18, and it scores **0.528 against 1.031**. adding ten more
heads to a hypothesis, five of them published, halves its score. so recovered is **not
monotone in set inclusion**, which kills the simplest version of the "it just counts heads"
worry and replaces it with something more awkward: a head left unablated while its inputs
are resampled can hurt more than resampling it too. The ten heads that do the damage are
`3.0 3.4 5.5 5.9 6.0 6.6 6.9 8.6 9.3 9.4`, mid-layer, and the loo table below shows the
same sign, `5.5`, `6.9`, `8.6` and `9.6` all have *negative* leave-one-out drops.

`phase1_extras`, the 8 heads phase 1 found that arent published, scores 0.014. that is the
one clean result in the table, they carry nothing, and both the scrub and the answer key
agree about it.

### the incompleteness gradient, which is flat

`drop_k` is the published circuit minus k heads, 20 random deletions each, 20 draws each.

| k | size | mean | sd across sets | fraction at or above the intact circuit |
|---|---|---|---|---|
| 1 | 25 | 1.023 | 0.150 | 0.25 |
| 2 | 24 | 0.967 | 0.115 | 0.20 |
| 3 | 23 | 0.979 | 0.177 | 0.35 |
| 4 | 22 | 1.064 | 0.255 | 0.45 |
| 6 | 20 | 0.929 | 0.328 | 0.35 |
| 8 | 18 | 0.900 | 0.246 | 0.20 |
| 12 | 14 | 0.911 | 0.246 | 0.30 |
| 18 | 8 | 0.500 | 0.259 | 0.05 |

throw away **twelve of twenty-six** heads and the mean score is 0.911 against the intact
1.022, a gap of 0.111 where the spread across those 20 deletions is 0.246, with 30% of them
scoring *higher* than the full circuit. the curve doesnt move until k = 18. the
pre-registered hope was that a circuit short a few heads would fail in proportion to how
incomplete it is. it doesnt fail at all until its missing two thirds of itself.

adding heads goes the other way and is boring, `add_k` runs 1.026, 1.027, 1.033, 1.050,
1.077 for k = 2, 4, 8, 16, 32. Random junk added to the published circuit never hurts and
slightly helps.

### where phase 1's 20/26 actually loses

**prediction 5 half-failed and the failure is the most interesting number in the phase.**
`phase1_matches` is exactly the 20 published heads phase 1 recovered, so it is one specific
`drop_6`. it scores 0.533 against `drop_6`'s mean of 0.929, at the **5th percentile** of
random 6-head deletions, outside the registered 10-90 window. the six phase 1 missed are
not six average heads.

| missed head | class | loo drop | phase 1 \|effect\| |
|---|---|---|---|
| `4.11` | previous token | **+0.316** | 0.0028 |
| `2.2` | previous token | **+0.177** | 0.0023 |
| `5.8` | induction | +0.009 | 0.0121 |
| `9.0` | backup name mover | +0.002 | 0.0182 |
| `0.10` | duplicate token | +0.001 | 0.0021 |
| `11.9` | backup name mover | -0.001 | 0.0027 |

the individual leave-one-out drops sum to 0.505 and the joint drop is 0.489, so the six act
almost additively and two of them carry all of it. `4.11` and `2.2` are the previous-token
heads. under `s2_swap` every cell before S2 is a floating-point exact zero, which the
README has listed as a gotcha since phase 1, and under `abc` they score 0.0028 and 0.0023.
Activation patching in this repo cannot see them under either counterfactual. causal
scrubbing ranks `4.11` first of all 26 and `2.2` third.

## P3, the null. passes on random, fails on layer-matched.

| null | n | median | p95 | max | published percentile | one-sided p | bar 0.95 |
|---|---|---|---|---|---|---|---|
| `random_26` | 200 | 0.269 | 0.981 | 1.404 | **0.960** | 0.045 | **pass** |
| `layer_matched_26` | 200 | 0.596 | 1.229 | 1.594 | 0.880 | 0.124 | **fail** |

**prediction 6 held and prediction 7 held.** the published circuit beats a uniformly random
26-head set at the 96th percentile, and does not beat a random set drawn to match its own
per-layer head counts. a layer-matched null carries a mean of 7.17 published heads for free
and reaches a median of 0.596 where the uniform null reaches 0.269, so most of the published
circuit's advantage over a random set of the same size is its layer profile.

P3 passes as registered. the registered bar is also, on this evidence, too generous to be
useful, and the post-hoc number says why: **25 of the 200 random 26-head sets, 12.5%, clear
P1's 0.70 bar**, and 8 of them score at or above the published circuit. a user with a
hypothesis and no answer key who runs this scrub and sees 0.85 has learned that they are
somewhere in the top half of random guesses.

## P4, the ceiling. it isnt size, quite, but it isnt correctness either.

| statistic | value | registered expectation |
|---|---|---|
| Spearman(recovered, \|H\|) over 4200 random sets | **0.679** | ≥ 0.9 would mean "size counter" |
| Spearman(residual, published-overlap fraction) over 22 named/family points | **0.375** | none |
| *post-hoc*: Spearman(overlap, recovered) among the 200 random sets at m = 26 | **0.489** | not registered |
| *post-hoc*: same at m = 18 / m = 34 | 0.550 / 0.405 | not registered |

**prediction 8 was wrong.** size explains a lot but not everything, rho 0.679 over the full
sweep. The size-free residual does correlate with correctness, at 0.375 across candidates
and around 0.5 within a fixed size. so there is real circuit information in the scrub score
and it is not just a head count.

it is also weak, and the null curve says how weak. at m = 26 the null mean is 0.352 with
sd 0.296 and a maximum of 1.404, so the spread of random sets at one size is larger than
the whole gap between a random set and the published circuit. the signal is there and it
sits under the noise it has to beat.

| m | null mean | sd | median | p95 | max |
|---|---|---|---|---|---|
| 3 | 0.052 | 0.141 | 0.013 | 0.272 | 0.952 |
| 8 | 0.113 | 0.188 | 0.051 | 0.458 | 1.046 |
| 13 | 0.200 | 0.251 | 0.109 | 0.807 | 1.220 |
| 18 | 0.288 | 0.308 | 0.177 | 0.954 | 1.228 |
| 26 | 0.352 | 0.296 | 0.269 | 0.981 | 1.404 |
| 34 | 0.424 | 0.287 | 0.346 | 0.955 | 1.820 |
| 60 | 0.719 | 0.321 | 0.628 | 1.342 | 1.654 |
| 118 | 0.988 | 0.213 | 0.945 | 1.401 | 1.699 |

(full 21-size curve in `phase12_tests.json` under `null_curve`.)

## P5, head level, against eleven phases of patching

`aoi` is the empty hypothesis plus one head, 144 of them, and is the only head-level signal
here computable with no answer key and no counterfactual.

| ranking | needs a counterfactual | AUC, published vs not, 144 heads |
|---|---|---|
| phase 1 patching, max of both schemes | yes | **0.936** |
| phase 1 `s2_swap` alone | yes | 0.906 |
| phase 1 `abc` alone | yes | 0.889 |
| **`aoi` scrub gain** | **no** | **0.799** (permutation p = 5.0e-5) |
| *post-hoc*: rank-average of `aoi` and patching | yes | 0.914 |

**prediction 9 held.** the scrub's head ranking is significantly better than chance and
clearly worse than activation patching, delta -0.137. Averaging the two rankings does not
beat patching alone either.

the number that isnt about winning: **Spearman(`aoi`, patching) = 0.240**. these are close
to independent rankings of the same 144 heads. phase 11's stability statistic correlated
with magnitude at 0.865 and was a monotone re-expression of it. This isnt. It reaches
AUC 0.799 by looking somewhere else.

leave-one-out from the full published circuit, all 26, sorted:

| head | loo drop | patch \|effect\| | phase 1 found it | class |
|---|---|---|---|---|
| `4.11` | **+0.316** | 0.0028 | **no** | previous token |
| `9.9` | +0.205 | 0.7820 | yes | name mover |
| `2.2` | **+0.177** | 0.0023 | **no** | previous token |
| `8.10` | +0.126 | 0.2533 | yes | s-inhibition |
| `10.0` | +0.087 | 0.1853 | yes | name mover |
| `7.9` | +0.083 | 0.1942 | yes | s-inhibition |
| `10.6` | +0.079 | 0.0918 | yes | backup name mover |
| `10.10` | +0.069 | 0.1350 | yes | backup name mover |
| `9.7` | +0.064 | 0.0981 | yes | backup name mover |
| `10.1` | +0.049 | 0.0758 | yes | backup name mover |
| `7.3` | +0.024 | 0.1171 | yes | s-inhibition |
| `10.2` | +0.018 | 0.0252 | yes | backup name mover |
| `3.0` | +0.010 | 0.1190 | yes | duplicate token |
| `5.8` | +0.009 | 0.0121 | no | induction |
| `9.0` | +0.002 | 0.0182 | no | backup name mover |
| `0.10` | +0.001 | 0.0021 | no | duplicate token |
| `11.9` | -0.001 | 0.0027 | no | backup name mover |
| `5.9` | -0.006 | 0.0486 | yes | induction |
| `0.1` | -0.008 | 0.0691 | yes | duplicate token |
| `8.6` | -0.011 | 0.2687 | yes | s-inhibition |
| `9.6` | -0.038 | 0.2922 | yes | name mover |
| `5.5` | -0.040 | 0.3178 | yes | induction |
| `6.9` | -0.044 | 0.0913 | yes | induction |
| `11.2` | -0.141 | 0.1149 | yes | backup name mover |
| `11.10` | -0.226 | 0.2689 | yes | negative name mover |
| `10.7` | **-0.375** | 0.5114 | yes | negative name mover |

by class:

| class | n | median loo drop |
|---|---|---|
| previous token | 2 | +0.247 |
| name mover | 3 | +0.087 |
| s-inhibition | 4 | +0.053 |
| backup name mover | 8 | +0.033 |
| duplicate token | 3 | +0.001 |
| induction | 4 | -0.023 |
| negative name mover | 2 | **-0.300** |

ten of the twenty-six published heads have a negative leave-one-out drop, meaning the scrub
scores *better* with them removed from the hypothesis. Both negative name movers, three of
the four induction heads, `9.6`, `8.6`, `11.2`, `0.1`, `11.9`. For the negative name movers
thats mechanically sensible,
they suppress the IO token and keeping them intact while their inputs are resampled leaves
the suppression running unopposed. For the induction heads and `9.6` its less obvious and
this phase doesnt establish why.

as a completeness signal this table is bad news, a scrub cannot tell you to add `10.7` to
your hypothesis, it tells you to take it out. as a second view of the circuit its the most
useful thing here, the two heads it puts on top being the two the existing pipeline is
structurally blind to.

## P6, variance. the phase 11 trap doesnt apply.

Spearman(sd across draws, recovered) over the 4200 random sets = **-0.191**. median sd
0.031, and the published circuit's sd is 0.025.

phase 11's whole negative was that replication noise scales with effect size, so any
statistic dividing by it compresses the ranking exactly where the circuit lives. that does
not happen here. scrub noise is flat to slightly decreasing in the score, so a
variance-aware version of these numbers wouldnt hit that defect. nothing in this phase
needed one, its recorded in case a later one does.

## registered variants

| candidate | `allpos` (primary) | `bypos` | `shared` sources | `mlp_scrub` |
|---|---|---|---|---|
| `published_26` | 1.022 | 1.012 | 0.952 | **0.050** |
| `phase1_agree` | 1.124 | 1.128 | 1.078 | 0.030 |
| `phase1_abc` | 1.031 | n/a | 0.983 | 0.245 |
| `phase1_matches` | 0.533 | 0.550 | 0.466 | 0.036 |
| `phase1_union` | 0.528 | n/a | 0.429 | 0.039 |
| `greedy_abc` | 0.968 | n/a | 0.947 | 0.038 |

`shared` (one source draw per row, shared by every ablated head, what the node-ablation
faithfulness literature does) runs uniformly a little below the independent-per-head primary
and preserves every ordering. the choice between the two readings of treeified sampling
doesnt matter here.

`mlp_scrub` matters a great deal. resample the MLPs as well as the heads outside the
hypothesis and the published circuit recovers **0.050**, against a floor of 0.174 raw logit
diff and a clean 3.697. best of any candidate is `phase1_abc` at 0.245. On this variant
nothing in the repo is remotely sufficient.

that is registered as secondary and stays secondary, because the published IOI circuit is a
list of attention heads and makes no MLP claim, so ablating MLPs scrubs a hypothesis Wang
et al. didnt state. it is still the sharpest caveat on P1's pass. **the published head list
passes its scrub only under the convention that MLPs are part of the scaffolding rather than
part of the hypothesis**, and under the other convention it fails by a mile.
[Transformer Circuit Faithfulness Metrics Are Not Robust](https://arxiv.org/abs/2407.08734)
finds the IOI circuit's faithfulness swinging on choices the original treated as
incidental. this is one of those, measured here.

## predictions, 5 of 9 held

| # | prediction | outcome | |
|---|---|---|---|
| 1 | P1 passes, `published_26` clears 0.70 | **1.022** | **hit** |
| 2 | `published_26` lands below 0.87 | 1.022, above the clean model | **miss** |
| 3 | `kl` recovered below `logit_diff` recovered | 0.870 vs 1.022 | **hit** |
| 4 | `phase1_union` within 0.10 of `published_26` | gap **0.494** | **miss** |
| 5 | `phase1_matches` worse than published, inside `drop_6`'s 10-90 | worse, but at the **5th** percentile | **miss** |
| 6 | P3 passes on `random_26` | 96th percentile, p = 0.045 | **hit** |
| 7 | P3 fails on `layer_matched_26` | 88th percentile, p = 0.124 | **hit** |
| 8 | Spearman(recovered, \|H\|) ≥ 0.9 | **0.679** | **miss** |
| 9 | `aoi` AUC doesnt beat patching by > +0.02 | **-0.137** | **hit** |

Four misses. 4 and 5 are the two that changed what this phase found. 4 expected the score to
be insensitive to which heads you pick and it turned out to be violently sensitive in the
wrong direction, non-monotone in set inclusion. 5 expected phase 1's 20/26 to be an average
20-head subset and it is the 5th percentile, which is how the previous-token heads got found.

## does this change phase 11's verdict

phase 11 closed by naming two things that would change the recommendation. this phase tests
the second and leaves the first, more published circuits, exactly where it was. IOI is one
circuit and every conclusion below is one circuit's worth:

> a criterion that references the behaviour under study independently of the counterfactual
> being graded.

**it exists, it works, and it isnt strong enough to change the verdict.** stating that
precisely, because the phase was set up so the answer could go either way:

- the slot is real. causal scrubbing computes a relation between a hypothesis and the
  model's behaviour on real inputs, with no counterfactual pair anywhere in the pipeline.
  Every quantity in phases 1-11 was a property of an intervention. this one isnt.
- the signal is real and answer-key-free. `aoi` separates published from unpublished heads
  at AUC 0.799, p = 5.0e-5, and Spearman 0.240 against patching says it is looking somewhere
  else rather than restating magnitude. that is the first time in this repo that a second,
  largely independent view of the same 144 heads has been available.
- the signal is weaker than the one it was meant to supplement, 0.799 against patching's
  0.936, and the combination beats neither.
- at the level the method is actually for, judging a whole hypothesis, its close to
  uninformative. 12.5% of random 26-head sets pass P1's bar. 30% of 12-head deletions from
  the published circuit score above the intact circuit. Adding random heads never hurts. a
  3-head set scores 0.968. a pass tells you the hypothesis is sufficient in a sense so
  permissive that most hypotheses satisfy it.

so this is a different ceiling from phase 11's and shouldnt get filed as the same failure.
phase 11's was statistical, replication noise scaled with the effect and every ratio built
on it compressed the top of the ranking. the trouble here is with the question. sufficiency
under resampling is a weak property, the model has enough redundancy and the ablation
distribution is close enough to the real one that many different head sets satisfy it, and
"is this set sufficient" then doesnt narrow the space of hypotheses much. Chan et al. say as much about permissive interpretation graphs, and
[Practical Pitfalls](https://www.lesswrong.com/posts/DFarDnQjMnjsKvW8s/practical-pitfalls-of-causal-scrubbing)
works through the mechanism. this phase is a measurement of how permissive it gets with the
one published circuit in this repo that has a real head list behind it.

what would make it bite, in order of how much of a design change each is:

1. a real interpretation graph. the whole simplification here is that `I` has no
   internal structure, so no equivalence class is ever conditioned on and every ablated node
   gets an unconditioned draw. Wang et al.'s account does have the structure, s-inhibition
   heads write to name-mover queries at END, induction and duplicate-token heads write to
   s-inhibition values at S2. Encoding that as a correspondence and scrubbing edges rather
   than nodes is the version of this experiment worth running next, and it is a real
   implementation, not a parameter change.
2. score the whole distribution instead of the margin. `logit_diff` recovered 1.022 while
   `top1_is_io` recovered 0.604 on the identical runs. the primary was registered for
   comparability with a published number and it is the more permissive of the two.
3. decide whether MLPs are hypothesis or scaffolding, in advance and out loud. the
   published circuit goes from 1.022 to 0.050 on that one convention.

none of those is a statistic over existing outputs, which is consistent with what phase 11
concluded. the recommendation in `SYNTHESIS.md` §5 stands: this is an open problem. it now
has one more measured boundary on it, and one concrete next experiment that isnt a
reweighting of anything already computed.

## deviations and disclosures

- the plan registered R = 20 draws for every candidate and the 4200 random sets and 200
  layer-matched sets ran at R = 5 instead. measured before deciding, not after. across ten
  random 26-head sets the set-to-set sd of the mean is 0.227 and the median within-set draw
  sd is 0.036, so a 5-draw mean carries 0.016 of noise against a 0.227 signal. Named
  candidates, `drop_k`, `add_k`, `loo` and `aoi` all ran at the registered 20.
- three of the post-hoc numbers are labelled *post-hoc* in the tables above and live in
  `scripts/phase12_posthoc.py` and `phase12_posthoc.json`. the within-fixed-size
  overlap correlation, the combined ranking, and the count of random sets clearing the bar.
  none changes a registered verdict, and the third sharpens a pass rather than rescuing a
  fail.
- `bypos` is undefined for candidates containing unpublished heads, as the plan said. it ran
  on the three all-published candidates only.
- the resample source holds template and ABB/BAB order fixed and varies names, place and
  object. the plan records this in advance as biasing every recovered number **up**. it
  does, and the size of that bias isnt measured here.
- `scripts/phase12_report.py` was committed two minutes after the run launched and 11
  minutes before the null stage began, so no null number existed when it was written.
- `pyarrow` will not load on this machine, Smart App Control blocks the unsigned
  `lib.cp312-win_amd64.pyd`, and `transformer_lens` imports it transitively through
  `datasets`. the run used a three-function stub `datasets` module on `PYTHONPATH`, outside
  the repo. `transformer_lens` touches `datasets` only in `evals.py` and its tokenize
  helpers and this phase calls none of them. `scripts/check_env.py` reports
  `ENVIRONMENT NOT READY` here for that reason, a real failure and not this phase's.

## artefacts

| file | what it holds |
|---|---|
| `phase12_scrub.json` | every scored hypothesis, per-draw scores included, 4825.7 s of measurement |
| `phase12_tests.json` | P1-P6, the null curve at all 21 sizes, the nine scored predictions |
| `phase12_posthoc.json` | the three post-hoc diagnostics, labelled |

## verdict

The published IOI circuit passes its own causal scrub, so the implementation is sound.
nothing else here is reassuring. sufficiency under resample ablation is satisfied by a
3-head set, by 12.5% of random 26-head sets, and by the published circuit with twelve of
its heads thrown away, and it gets destroyed by adding ten mid-layer heads that are mostly
published. as a way to grade a hypothesis with no answer key its close to uninformative at
this granularity.

it does see heads activation patching cannot. `4.11` and `2.2` are the two most load-bearing
heads in the published circuit under a scrub and score 0.0028 and 0.0023 under both of phase
1's counterfactuals. twelve phases in, thats the first signal here that disagrees with
magnitude in a direction the answer key confirms. one circuit and six heads, and it wants
checking on greater-than and docstring before anyone leans on it.
