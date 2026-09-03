# phase 12, pre-registration: causal scrubbing, does sufficiency against real behaviour say anything the counterfactual pair couldnt?

Written and committed before `causal_interp/scrubbing.py` and `scripts/run_phase12_scrub.py`
exist. nothing below was chosen after seeing a phase 12 number, the order is checkable in
git history like every phase here.

## what this is and why its not another statistic

eleven phases ran one shape of experiment. build a clean prompt and a counterfactual
partner, splice activations from one run into the other, score how far the metric moved,
then compare that score against a null or against another score. phase 11 ended by ruling
out the whole family that treats the leftover as noise, and `SYNTHESIS.md` §5 named the two
things that would change the verdict. the second one:

> a criterion that references the behaviour under study independently of the counterfactual
> being graded. every quantity the pipeline computes is a property of the intervention.
> the needed quantity is a relation between the intervention and the behaviour and the
> architecture has no slot holding the second term.

causal scrubbing has that slot. it never builds a counterfactual pair at all. the input is
a real prompt, the model runs on it, and the question is whether a claimed circuit is
*sufficient*, if you resample everything outside the claim, does the model still do the
thing.

the method is Redwood's, [Chan et al., Dec 2022](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing),
with the formal metric in the [appendix post](https://www.lesswrong.com/posts/kcZZAsEjwrbczxN2i/causal-scrubbing-appendix)
and the two worked examples in [induction heads](https://www.alignmentforum.org/posts/j6s9H9SHrEhEfuJnq/causal-scrubbing-results-on-induction-heads)
and [paren balance](https://www.alignmentforum.org/posts/kjudfaQazMmC74SbF/causal-scrubbing-results-on-a-paren-balance-checker).
this plan is written off those, not off the name.

### the actual definition, and where this phase simplifies it

a full hypothesis in Chan et al. is `h = (G, I, c)`. `G` is the model as a computational
graph, `I` is a human-readable interpretation graph, `c` an injective homomorphism from
`I` into `G` such that every edge `(u, v)` in `I` has `(c(u), c(v))` in `G`. the scrub is
recursive over `I`, treeified. for each node, parents that `I` says matter get an input
sampled to *agree with the real input on that parent's value*, parents `I` says dont matter
get an input drawn at random from the dataset, and all unimportant parents of one node share
the same random draw.

resample ablation is the bit that makes it work, and it is not what this repo has been
doing:

| | replaces the activation with | on-distribution |
|---|---|---|
| zero ablation | `0` | no |
| mean ablation | the empirical mean over some distribution | no, means arent activations |
| activation patching (phases 1-11) | the same module's activation on a *paired* counterfactual prompt | yes, but the pairing is the hypothesis under test |
| resample ablation (here) | the same module's activation on a *randomly drawn other input from the same distribution* | yes, and no pairing is assumed |

phase 1-11 patching and resample ablation both splice a real activation from a real forward
pass. the difference is what the source is and which direction the claim runs. patching
takes a clean activation into a corrupted run and asks how much of the metric came back,
so the corruption is a premise. scrubbing takes a *random other prompt's* activation into a
clean run and asks whether behaviour survived, so nothing is a premise except the head list.

the simplification this phase makes, stated up front because it is the main threat to the
result: the published IOI circuit is a set of attention heads with class labels, not an
interpretation graph. there is no `I` to recurse over, so there are no equivalence classes
to agree on, and the scrub collapses to its degenerate case, every node outside `c(I)` gets
an unconditioned random draw and every node inside is left alone. that is causal scrubbing
with a maximally permissive `I`, which Chan et al. flag as scoring high for cheap, and
[Practical Pitfalls](https://www.lesswrong.com/posts/DFarDnQjMnjsKvW8s/practical-pitfalls-of-causal-scrubbing)
works through why. a permissive hypothesis passing is weak evidence. a permissive
hypothesis *failing* is still strong evidence, so the sanity check in P1 is informative in
one direction regardless.

## 0. what is held fixed

model `gpt2-small`, task IOI, `n = 128` clean prompts, `seed = 0`, the existing
`IOIDataset` construction untouched. no corrupted dataset is built anywhere in this phase.
`causal_interp/ioi.py`, `interventions.py`, `ground_truth.py` are not modified. the scrub
lives in a new module and reads `ground_truth` only in the reporting script, same split the
last four phases used.

## 1. the resample source

a source activation has to land at the right token index or the splice is meaningless.
GPT-2's tokenizer gives every place and object in `ioi.py` a single token, and
`IOIDataset` assigns template `i % 8`, so within one row index every dataset built from
this constructor has the same template, the same length, and the same semantic position
map. checked, per-template lengths are `[15, 16, 16, 16, 17, 18, 19, 20]` and each
template's IO/S1/S2/END indices are constant across its rows.

so: build `S = 8` source datasets at seeds `1000 + s`. source dataset `s` row `r` shares
template, order, length and position layout with clean row `r`, and differs in names, place
and object. token alignment is exact by construction, not by luck.

for one scrub draw, each head `(l, h)` and each row `r` independently draws
`s ~ Uniform{0..S-1}`, and the head's activation at every position of that row comes from
source `s`. independent per node, which is the closer of the two available readings of
treeified sampling. the alternative, one shared source per row across all ablated heads,
is registered as a secondary variant `shared`, since that is what the node-ablation
faithfulness literature does and the comparison is worth having.

what varies in the source and what doesnt: names, place, object vary. template and ABB/BAB
order dont, because they set the token geometry. this makes the ablation *weaker* than a
free draw from the IOI distribution would be, so it biases every recovered number *up*.
recorded here so the direction of the bias is known before the numbers are.

`S = 8`, `R = 20` independent draws per candidate, mean and sd across draws reported for
everything.

## 2. what gets ablated

heads only, at the `z` hook, every layer. MLPs, embeddings, layernorm and the residual
stream are never touched. the published circuit is a head list and makes no MLP claim, so
ablating MLPs would be scrubbing a hypothesis Wang et al. didnt state. a variant that also
resamples MLPs is registered as secondary `mlp_scrub` and reported separately, never as
the headline.

position handling, two registered variants:

`allpos` is primary. a head in the hypothesis is untouched at every position, a head
outside it is resampled at every position, the plain reading of "these 26 heads are the
circuit".

`bypos` is secondary. a head in the hypothesis is untouched only at its published class's
expected position (`ground_truth.CLASS_EXPECTED_POSITION`) and resampled everywhere else.
strictly harder, and only definable for candidate sets whose heads carry published class
labels, so it cant run on random nulls or on unpublished extras. that restriction keeps it
secondary.

## 3. the metric, and the two anchors

Chan et al. score a hypothesis as

```
% recovered = (E_scrubbed - E_randomized) / (E_model - E_randomized)
```

with `E_model` the unablated model and `E_randomized` a null where the association between
input and output is destroyed. they note it can exceed 100% or go negative and is an
arithmetic aid, not a fraction. same form here, with both anchors computed rather than
assumed:

- `E_model` = the clean forward pass, no hooks. the hypothesis `H = all 144 heads`.
- `E_0` = the empty hypothesis `H = {}`, every head resampled. this is the natural
  randomized anchor for a *sufficiency* claim, it is the same experiment with the claim
  removed.

three measures, all at the END position, all against the clean prompt's own answer:

| | `E_model` | `E_0` | recovered |
|---|---|---|---|
| `logit_diff` **primary** | clean IO−S logit diff | fully scrubbed IO−S logit diff | `(x − E_0)/(E_model − E_0)` |
| `kl` **registered secondary with its own bar** | 0 by construction | KL(clean ‖ fully scrubbed) | `1 − KL(clean ‖ x)/KL(clean ‖ scrub-all)` |
| `top1_is_io` reported, no bar | clean top-1 rate | fully scrubbed top-1 rate | same form as `logit_diff` |

`logit_diff` is primary because every prior phase used it and because the one published
number this can be held against, Wang et al.'s 87% circuit faithfulness, is in that unit.
`kl` is the stricter "did behaviour survive" reading and gets a separate pre-registered
bar, not a separate chance to pass.

## 4. implementation gates, which run before any result is looked at

three exact identities. if any fails the implementation is wrong and no number below gets
read.

1. `H = all 144 heads` gives recovered `= 1.0` and logits bit-identical to the clean run.
2. `H = {}` gives recovered `= 0.0` by definition of the anchor, and its raw logit diff
   must be measurably below clean, if scrubbing every head leaves behaviour intact the
   whole design is void.
3. source datasets replaced by the clean dataset itself gives recovered `= 1.0` for every
   `H`, since every splice is then a no-op.

gate results go in the payload and in the report whether they pass or not.

## 5. the candidate circuits

everything already in this repo that claims to be an IOI circuit, plus nulls.

| name | size | published heads | where from |
|---|---|---|---|
| `published_26` | 26 | 26 | `ground_truth.ALL_HEADS` |
| `phase1_union` | 28 | 20 | s2_swap ∪ abc at the headline cutoff, the pipeline's own answer |
| `phase1_s2_swap` | 23 | 18 | phase 1 primary scheme alone |
| `phase1_abc` | 18 | 15 | phase 1 secondary scheme alone |
| `phase1_agree` | 13 | 13 | heads both schemes found, phase 8's "robust" construction applied to IOI |
| `phase1_matches` | 20 | 20 | the 20/26 the README headline reports, published-minus-6 |
| `phase1_extras` | 8 | 0 | the 8 false positives alone |
| `greedy_s2_swap` | 6 | 6 | phase 1's greedy trace, `5.5 8.10 7.9 8.6 7.3 3.0` |
| `greedy_abc` | 3 | 2 | `9.9 10.0 0.5` |

plus, constructed here:

- **`drop_k`**, published minus `k` heads, `k ∈ {1,2,3,4,6,8,12,18}`, 20 random draws each.
  the incompleteness gradient. `k = 6` is size-matched to `phase1_matches`.
- **`add_k`**, published plus `k` random non-published heads, `k ∈ {2,4,8,16,32}`, 20 draws.
- **`random_m`**, uniform random head sets, `m ∈ {3,6,8,13,18,20,23,26,28,34,44,60,88,118}`,
  **200 draws each**. this is the null.
- **`layer_matched_26`**, random 26-head sets drawn to match the published circuit's
  per-layer head counts exactly, 200 draws. the harder null.
- **`loo`**, published minus one head, all 26.
- **`aoi`**, empty plus one head, all 144.

## 6. the registered tests

### P1, the sanity check. does the published circuit pass its own scrub?

`published_26`, `allpos`, independent sources, R = 20.

pre-registered bar, both must clear:

- `logit_diff` recovered **≥ 0.70**
- `kl` recovered **≥ 0.50**

where 0.70 comes from: Wang et al. report the circuit recovering 87% of logit difference
under node-level mean ablation at specified positions.
[Transformer Circuit Faithfulness Metrics Are Not Robust](https://arxiv.org/abs/2407.08734)
re-runs that measurement across ablation methodologies and finds resample ablation gives
systematically lower numbers than mean ablation, with the IOI circuit's score swinging hard
on choices the original paper treated as incidental. so 87% under mean ablation should not
be expected to reproduce under resample ablation, and 0.70 is a deliberately loose bar for
"the published circuit is basically sufficient". `kl` has no published anchor at all, it is
a strictly harder measure, and 0.50 is a judgement call, stated as one.

**if P1 fails, the phase stops and debugs.** every result after this depends on the scrub
being correctly implemented, and the published 26 heads are the only hypothesis in the repo
with independent evidence behind them. a failing P1 is reported as an implementation
finding, not as a finding about the circuit, unless the gates in §4 all pass and the failure
survives them, in which case it is reported as both and the ambiguity is stated.

### P2, the pipeline's own circuits. do they pass, and does incompleteness show?

`phase1_union`, `phase1_s2_swap`, `phase1_abc`, `phase1_agree`, `phase1_matches`,
`phase1_extras`, both greedy sets, all at `allpos`. no bar, this is descriptive. the two
registered questions:

- does `phase1_union` (28 heads, 20 published) clear the P1 bar? it is larger than the
  published circuit and less correct, and those pull opposite ways.
- the ordering test. `phase1_matches` is exactly published-minus-6, so its recovered score
  gets compared against the `drop_6` distribution. prediction registered in §7.

### P3, the null, and this is the phase's real test

the whole promise is that someone with no answer key can scrub a hypothesis and learn
something. so: where does `published_26` sit in the `random_26` distribution, and in the
`layer_matched_26` distribution?

reported as a percentile over 200 draws and as an exact one-sided permutation p. the bar,
registered: **`published_26` above the 95th percentile of `random_26`** for the scrub to
count as carrying circuit information at all, and above the 95th percentile of
`layer_matched_26` for it to be carrying more than layer depth.

### P4, the ceiling test. is scrubbing a relevance signal or a head counter?

the failure mode this phase most expects, and the one that would make causal scrubbing land
in the same place as everything else for a *different* reason than phase 11's:

> recovered score is essentially a function of `|H|`, so any hypothesis passes by adding
> heads and the method ranks nothing.

three registered statistics over the `random_m` sweep and the candidate pool:

- Spearman(recovered, `|H|`) across all 2800 random sets. if this is ≥ 0.9 the score is
  mostly a size counter.
- the *residual* score, recovered minus the mean recovered of `random_m` at the same `m`,
  in units of that null's sd. every named candidate gets one. this is the size-free version
  of the signal and is what P2 and P3 should really be read through.
- Spearman(residual, published-overlap fraction) over the named candidates plus the `drop_k`
  and `add_k` families. does the size-free signal track correctness.

### P5, head level, the direct comparison against eleven phases of patching

`loo` gives each published head a leave-one-out drop. `aoi` gives all 144 heads an
add-one-in gain. `aoi` is the one that can be computed with no answer key, so it is the
primary head-level signal.

AUC of published-vs-unpublished separation over all 144 heads, three rankings:

| ranking | needs an answer key | needs a counterfactual |
|---|---|---|
| `aoi` scrub gain | no | **no** |
| phase 1 `s2_swap` max-over-positions \|effect\|, read from `results/phase1_results.json` | no | yes |
| phase 1 `abc` same | no | yes |

registered comparison: scrub AUC minus the better of the two patching AUCs, with an exact
permutation test over head labels. no bar, this is the number the phase exists to produce
and pre-committing a threshold on it would be pretending the answer matters less than the
direction.

### P6, variance

sd across the R = 20 draws, per candidate, reported everywhere. and one specific check,
because phase 11 found replication noise scaling with effect size and that would matter
here too: Spearman(sd, recovered) over the `random_m` pool. if scrub scores are noisier
where they are larger, the same compressive trap applies to any ratio built on them, and
§7 prediction 8 says so in advance.

## 7. predictions, to be scored

1. P1 passes. `published_26` clears 0.70 on `logit_diff`.
2. `published_26` `logit_diff` recovered lands **below 0.87**, since resample ablation is
   harsher than the mean ablation that produced that number.
3. `kl` recovered for `published_26` lands **below** its `logit_diff` recovered. KL sees the
   whole distribution, logit diff sees two tokens.
4. `phase1_union` (28 heads, 20 published) scores **within 0.10** of `published_26` on
   `logit_diff`. it has 8 wrong heads and 2 more heads total, and if scrubbing cant tell
   those apart that is P4's failure mode arriving early.
5. `phase1_matches` (published-minus-6) scores **worse than** `published_26` and lands
   **inside** the `drop_6` distribution, between its 10th and 90th percentile.
6. P3 passes on `random_26`, `published_26` above the 95th percentile.
7. P3 **fails** on `layer_matched_26`. late-layer heads carry most of the IOI logit
   difference and a layer-matched null gets that for free.
8. Spearman(recovered, `|H|`) over the random sweep is **≥ 0.9**. size dominates.
9. P5, `aoi` scrub AUC does **not** beat the better patching AUC by more than +0.02. said
   plainly, the expectation is that this doesnt work either, registered here so that it being
   wrong stays visible.

## 8. what a null means, fixed now

if P1 passes and P3 passes on `random_26` but P4 shows the score is mostly `|H|`, the
finding is that causal scrubbing gives a real sufficiency check and a poor *ranking* signal,
and the useful output is the residual, not the raw recovered score. that is a different
ceiling from phase 11's and gets reported as one.

if P1 passes and P3 fails, causal scrubbing on this task at this granularity carries no
circuit information beyond what a random head set of the same size carries, and the honest
report is that it degrades to the same place from a different direction.

if P1 fails after the §4 gates pass, that is the most interesting outcome available and
gets reported as-is. the published IOI circuit failing its own scrub under resample ablation
would be consistent with the faithfulness paper's headline and would say the 26-head list is
not sufficient in the sense this phase tests.

no bar in this document moves after the first number is read. if a bar turns out to have
been badly chosen that gets written down next to the result, and the result is scored
against the bar as written.

## 9. rules

- `causal_interp/scrubbing.py` doesnt import `ground_truth`. the runner asserts it at
  startup, same as phases 4 and 8.
- the plan is committed before the implementation. the implementation is committed before
  the run. the report is written after.
- every candidate's raw per-draw scores go in the payload, not just the means.
- deviations get a `PLAN GAP` marker in the code and a line in the report.
