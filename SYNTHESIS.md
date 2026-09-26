# causal-interp, twelve phases

I ran twelve pre-registered phases to see what this pipeline can actually find. This is
what held up, what didnt, and where it kept failing. Numbers below come from the committed
reports in [`results/`](results/), with the report linked next to each result.

The reports have the protocols and raw details if you want to check any of it.

---

## the short answer

I started out trying to build a system that discovers and causally tests mechanisms in
neural networks, eventually including models too capable to check by hand. Twelve phases
later, the split is pretty clear:

If a question comes down to measuring a magnitude, the pipeline can do it. If it needs a
judgment about relevance, it still needs a person or an answer key.

It can find a circuit, rank components, locate where a head reads its input, use a metric
that doesnt need the answer key, and transfer to another task or model. I also tried
building a task from example sentences. Those steps ran without giving the pipeline the
answer, across three published circuits in two models.

What it cant do is tell me which experiment to believe. I tried four ways to get at that
in phases 9 and 10, the scheme-level re-analysis, and phase 11. They all hit the same
problem. That matters for the scalable-oversight idea because it only helps when there
isnt an answer key to check against.

In phase 11 I resampled both circuits ten times. The findings already replicate almost
perfectly. Dividing an effect by its own replication spread made discovery worse because
that spread grows with the effect. So far, this doesnt look like a sampling-noise or
confidence-interval problem. I updated sections 2 and 5 to match, and stopped looking for
a fix inside this pipeline.

Phase 12 tried the remaining option, causal scrubbing. The published IOI circuit passes
its scrub, and the per-head signal is almost unrelated to activation patching while still
separating the published heads at AUC 0.799. The broader conclusion didnt change: most
head sets pass the resample-ablation sufficiency test. Sections 2 and 5 have the details.

---

## 1. what was validated, and how solidly

### the ladder, phase by phase

| phase | question | headline | how solid |
|---|---|---|---|
| [1](results/PHASE1_REPORT.md) | does activation patching recover a published circuit? | 20/26 IOI heads (union of two counterfactuals, 18/26 under the primary alone) | strong, and it named its own blind spot |
| [2](results/PHASE2_REPORT.md) | does path patching close the gap? | 19/26 alone, 20/26 combined. the gap did not close | strong, and a failed prediction |
| [3](results/PHASE3_REPORT.md) | does a pre-registered receiver-side criterion find the rest? | 2 of the 6 missing heads, precision 0.64 | medium. threshold fixed in advance, criterion noisier |
| [4](results/PHASE4_REPORT.md) | can the search find where a head reads, without being told? | 16 of 17 scoreable specs | strong. exhaustive grid, 9,936 forward passes |
| [5](results/PHASE5_REPORT.md) | which hand-built pieces actually carry the result? | answer key not needed, corruption content is | strong, and the result inverted the prediction |
| [6](results/PHASE6_REPORT.md) | does any of it transfer to a second circuit? | 7/7 greater-than heads, causal core untouched | strong on transfer, deflated on difficulty |
| [7](results/PHASE7_REPORT.md) | does it transfer to a different model? | 3/6. code transferred perfectly, results got worse | strong as a negative. 3 of 7 predictions wrong |
| [8](results/PHASE8_REPORT.md) | can the pipeline flag its own blind spot? | yes, loudly. 3/6 -> 6/6 on docstring, flagged before the answer key | real, and half an answer |
| [9](results/PHASE9_REPORT.md) | can it tell a real blind spot from noise? | partial. a better criterion, not a discriminator | strong negative, w a real holdout |
| [10](results/PHASE10_REPORT.md) | can the task be induced instead of written? | 3/7 pre-registered (5/7 repaired) vs hand-built 6/7 | honest negative. 1 of 8 predictions held |
| [11](results/PHASE11_REPORT.md) | do findings survive resampling, and does that separate them? | no, and stability scores below magnitude. but discovery replicates 10/10 | strong negative with a measured mechanism. 6 of 9 predictions held |
| [12](results/PHASE12_REPORT.md) | does causal scrubbing grade a circuit without an answer key? | published 26 passes at 1.02, so do 12.5% of random 26-head sets | strong negative at circuit level, one real positive at head level. 5 of 9 predictions held |

### the four things that held up

phase 1 got 20 of Wang et al.'s 26 IOI heads and explained the six misses before anyone
asked. they act through other heads, which total-effect patching cant resolve. It also
measured its own blindness instead of asserting it. under the `s2_swap` counterfactual 576
of 576 head-position cells before the S2 token are exact floating-point zeros. every
single one, cuz the two runs are computing on bit-identical inputs there. took me a while
to believe that wasnt a bug. any published head whose role sits at those positions is
undiscoverable by that scheme in principle.

phase 2 then tested phase 1's own prediction that path patching would close the gap. It
didnt. 19/26 alone, still 20/26 combined, same six heads missing. what did improve was
precision (0.71 -> 0.90, 1.00 under the `abc` scheme alone) and, more interesting, the
causal ordering. each round's receivers were the previous round's discoveries, the answer
key never got consulted to pick them, and round 1 returned all four published
S-inhibition heads as its top four out of 144, which i did not expect

phase 3 fixed the rule before measuring, 99th percentile of the path signal under a
shuffled-source null, rounded up to two significant figures, and got 0.11. Recovered `2.2`
and `4.11`, both previous-token heads neither earlier phase could see, at precision 0.64
against the logit criterion's 0.90.

Two details make this more credible than the number alone. the pre-registration is a
separate commit with the threshold and the code and no results, so the ordering is
checkable in git instead of asserted. and the phase flatly refused to merge the two criteria
into one recall figure, cuz they disagree abt which heads count. 5 heads found by both,
16 by the logit criterion only, 6 by the receiver-side one only. merging would have reported
a bigger number while destroying the only new information the phase produced.

the search finds where a head reads its input without being told, probably the cleanest
positive result here. phase 4 scored every `(layer, head, input, position)` spec by
splicing that one input from the clean run into the corrupted one, over an exhaustive
grid. `search.py` doesnt import a ground-truth module and the run asserts that before
starting.

Of the 21 published heads w a published receiver spec: 16 agreement, 0 ambiguous, 4
unmeasurable, 1 disagreement. So 16 of the 17 it could actually weigh. all four
S-inhibition heads came back `v@S2`, every name mover, backup and negative name mover came
back `q@END`

the 4 "unmeasurable" are the phase 1 blind spot showing up again. published spec for the
induction heads is `k@S1+1` and under `s2_swap` all 432 specs at that position score
exactly zero. the search never weighed the published answer and preferred something else,
it got handed a counterfactual that cant see that position. counting those as search
failures blames the search for a defect that belongs to the experiment.

And the position labels turned out to be unnecessary. an unlabelled search over bare token
indices piled its top 50 onto `t11` (x33), `t15` (x15) and `t12` (x2), which turn out to be
S2, END and S2+1, labels attached after the search purely to read its output.

phase 6 pointed the whole pipeline at a second published circuit in the same model and got
7/7 greater-than heads and 7/7 receiver specs. `interventions.py`, `search.py` and
`metrics.py`, the causal core, were not touched. `comparison.py` gained a `circuit`
parameter it should have had all along. the phase deflated its own headline, seven targets
is easier than twenty-six and this circuit has no analogue of IOI's previous-token heads

phase 7 then pointed it at a different model. `attn-only-4l`, 4 layers, 8 heads, different
tokenizer, no MLP blocks at all. code transfer was total, the report runs
`git diff --stat` over the ten pre-existing modules and pastes the empty output. The results
were the worst of the three circuits:

| circuit | model | published | recovered | recall | precision | chance recall |
|---|---|---|---|---|---|---|
| IOI | GPT-2 small | 26 | 18/26 | 69% | 0.78 | 18% |
| greater-than | GPT-2 small | 7 | 7/7 | 100% | 0.78 | 5% |
| docstring | `attn-only-4l` | 6 | 3/6 | 50% | 0.33 | 19% |

The plan had pre-registered a deflation, 6 heads among 32 is an easier denominator, and it
cut less than expected. chance recall is 19% here against IOI's 18%, so the drop from 69%
to 50% is real and not an artifact of circuit size.

phase 7 probably matters more than any of the others, and not for the number. it produced
a clean, plausible, internally consistent circuit claim that was missing half the
mechanism, and only the published head list caught it. Cause is well understood now. the
benchmark's default counterfactual replaces the answer token, which makes the circuit's
routing heads causally invisible to a metric read off the output. a different published
counterfactual finds 5 of 6. which parts of a circuit come out discoverable is a property
of the experiment

two silent failure modes surfaced there too, neither showing up in a diff. the component
sweep patched `mlp_out` on a model w no MLPs and returned 28 exact zeros that read as
"the MLPs carry no causal signal". and the iterative path chain assumed another layer
existed below, halted after one round, and left the receiver-side null pooled from 16
measurements instead of hundreds.

### the metric doesnt need the answer key

Phase 5 swapped the hand-built logit difference, which needs to know which two tokens are
the candidates and which is correct, for divergences over the whole next-token
distribution. size-matched to the published circuit's 26 heads: logit difference 18/26, KL
19/26, total variation 19/26, per-head rankings correlating at +0.98.

but the same phase found the limit right next to it. making the corruption generic too is
what costs:

| what is supplied | size-matched recovery |
|---|---|
| hand-built corruption + hand-built metric | 18/26 |
| hand-built corruption + general metric | 19/26 |
| generic corruption + hand-built metric | 18/26 |
| generic corruption + general metric | 16/26 |
| nothing supplied at all | 13/26 |

the two pieces dont add up, either one alone carries enough task knowledge and its
removing both that kills it. `s2_swap` was built to reverse
the behaviour so the corrupted run sits as far from clean as the task allows, a logit span
of 7.30. a random token just damages the prompt, the corrupted run still partly performs
the task and the span collapses to 1.54.

that difference between reversing and damaging, most of section 2 is abt it

---

## 2. five investigations, two walls

phases 9 and 10 went at different problems w different machinery and produced the same
shape of failure. the convergence is sort of the finding on its own. it reads more like
evidence abt where the difficulty actually sits than like two unrelated disappointments.
Two later efforts, both written after this section originally said "two investigations",
tested the two most obvious escape routes and closed them. the [scheme-level
re-analysis](results/SCHEME_LEVEL_NOTE.md) moved all twenty of phase 9's signals to the unit
that actually matters, and phase 11 tested the one class of evidence none of them could
compute, replication. Both are below, after the two original ones. Phase 12 is the fifth and
sits apart from the other four. it changed method rather than statistic and ran into a
different obstacle, hence two walls in the heading

### phase 9, given several counterfactuals that disagree, which disagreement matters?

Phase 8 had made multi-counterfactual discovery structural. a task has to register at least
two schemes, `pipeline.discover()` sweeps all of them, and the pipeline prints which heads
change status between them. pointed back at phase 7's circuit it named `1.2`, `1.4` and
`2.0`, exactly the three heads phase 7 missed, before any answer key got opened.

And the same flag fired on greater-than, where the primary counterfactual had already got
7/7 and there was nothing to find:

| circuit | flag | heads flagged | of which published | recall, primary -> union | precision |
|---|---|---|---|---|---|
| docstring | fires | 17 | 3 | 3/6 -> 6/6 | 0.33 -> 0.23 |
| greater-than | fires | 16 | 0 | 7/7 -> 7/7 | 0.78 -> 0.28 |

the two flags are identical in form. phase 9 tried to tell them apart.

it measured ten candidate signals first, in a separate commit, before designing anything.
Nine dont separate the two cases and several separate in the wrong direction. any
statistic normalized by a scheme's median is dominated by how many dead heads a model has,
so GPT-2 small's noise outranks `attn-only-4l`'s real finds.

The one rule worth pre-registering was a real improvement. calibrate each scheme against its
own shuffled-source null instead of a shared cutoff. that much is settled decisively. the
ten measured floors span a factor of 400, theta = 0.0077 up to theta = 3.3. docstring's
`random_vocab_any` has a null whose 99th percentile is 3.3, meaning patching a head w an
activation from a different prompt routinely moves that metric by several times the entire
clean-to-corrupted span. it had contributed 17 of docstring's flags under the shared cutoff
and contributes none under its own floor. phase 8's shared 0.02 wasnt defensible and thats
done.

still not a discriminator tho:

| circuit | known case | heads flagged | flag precision |
|---|---|---|---|
| docstring | real blind spot | 17 -> 4 | 18% -> 50% |
| greater-than | no blind spot | 16 -> 8 | 0% -> 0% |
| IOI (holdout) | real blind spot | 15 -> 19 | 20% -> 26% |

IOI is a real holdout. phase 8 registered its four schemes and deliberately never ran it and
its expected behaviour got fixed in the plan from phase 1's finding. Its calibrated blind
spot grew, cuz `abc` and `random_vocab_s2` came out w floors below 0.02 and got more
sensitive, while the primary's floor rose to 0.058 and cost it three published heads of its
own (18/26 -> 15/26). calibration cuts both ways.

phase 9's own summary of what it got: the null floor separates "this scheme is too weak to
believe" from "this scheme measured something". it does not separate "this scheme measured
something the primary is blind to" from "this scheme measured something outside the
circuit", cuz both of those are statistically sound measurements under a real counterfactual

### Phase 10, given several auto-proposed counterfactuals, which one should be primary?

phase 10 stopped writing the task by hand. from 32 example sentences a person typed off a
one-sentence hunch it induces the template (constant token columns), the slots (varying
ones), the tied slots (columns that co-vary), the vocabularies (observed values), the
positions (bare indices, which phase 4 showed are enough), one counterfactual per slot, and
an answer-key-free metric. all of it reaches `pipeline.discover()` as an ordinary `TaskSpec`
w no pre-existing module changed.

pre-registered it got 3 of 7 published greater-than heads against the hand-built task's
6, a negative by its own scoring table. and the failure came apart into two separable
halves.

half one was a bug and its fixed post-hoc. two of the thirty-two human lines contain a
year GPT-2 splits as `[" 150", "9"]` instead of `[" 15", "09"]`. same row length so the
filter kept them. the tie rule needs unanimity so the century columns stopped being one
slot. generation then sampled them independently and produced clean prompts like `The
pilgrimage lasted from the year 11245 to the year 14`. the model's top prediction exceeded
the start year on 48% of them, a coin flip, which is the published task's own
`xx_mismatch` counterfactual served up as the control. nothing said so.

half two isnt a bug:

| scheme | size-matched | precision |
|---|---|---|
| `resample_t2` (the noun) | 2/7 | 0.09 |
| `resample_t7` (start century), chosen | 3/7 | 0.22 |
| `resample_t8` (start year) | 5/7 | 0.50 |
| `resample_END` (final century) | 4/7 | 0.46 |
| `random_vocab_any` (generic) | 1/7 | 0.12 |

`resample_t8` redraws the start year, exactly the position the published `yy01`
counterfactual acts on. it was sitting right there in the proposed set scoring 5/7 at
precision 0.50. the answer-key-free rule picked `resample_t7` instead, on an output
divergence three times larger. on the second fixture the same rule picked `resample_END`, a
third position again.

the content of the counterfactual mechanized, picking between them didnt

The repair confirms the diagnosis rather than rescuing the headline. drop the odd lines, the
tie comes back, `desync_t7` gets proposed, which is phase 8's hand-authored `xx_mismatch`
re-derived from example sentences. `resample_t7`'s divergence collapses from 0.755 to 0.084
cuz both centuries move together now, and the rule picks the right scheme on its own,
reaching 5/7 size-matched and 7/7 at the inherited 0.02 cutoff. labelled post-hoc
everywhere and it doesnt replace the 3/7.

### Why these are the same failure

The two rules arent the same rule. Phase 9's sets a per-scheme threshold, how large an
effect has to be in this experiment's own units to count as measured at all. Phase 10's
ranks whole schemes against each other by raw KL divergence. different objects, different
statistics.

what they share is the substitution they both make. each needed to know whether an
experiment says something abt the mechanism under study, and each answered a different
question that happens to be computable without an answer key.

phase 9 answered a validity question, "is this measurement bigger than what this
experiment manufactures from noise", and got a decisively good answer to it. phase 10
answered a magnitude question, "which counterfactual moves the output most", and got a
correct answer to that too.

Neither is the question that was needed. a counterfactual can be perfectly valid, move the
output a huge amount, and still be telling you abt something other than the behaviour youre
studying, and neither validity nor magnitude measures aim

Phase 9 says this abt itself in as many words. the null floor cant separate a scheme that
disagrees cuz the primary is blind from one that disagrees cuz it is measuring
something outside the circuit, "cuz both are statistically sound measurements under a
real counterfactual." phase 10's version is the same sentence w the objects swapped. a badly
aimed counterfactual and a well-aimed one are both real interventions, and the badly aimed
one moved the output further.

the evidence that magnitude and aim come apart is scattered across the whole project and
it points one way every time:

- in phase 5 a generic corruption damages the prompt where a semantic one reverses the
  behaviour. fully generic recovers 13/26 against 18/26, and the reason is that a
  distribution-wide metric "shares the shrunken span out over every irrelevant way the
  prompts differ"
- phase 8's generic schemes "need no knowledge of the task, so any task can register them",
  and theyre simultaneously the lowest-power schemes and the biggest source of flagged heads
  in no published circuit
- in phase 9 the scheme that manufactures the most apparent recovery from a mismatched
  activation (theta = 3.3) is the one that knows least about the task
- phase 10's `random_vocab_any` had the highest KL divergence of any scheme on both fixtures
  (1.805 and 0.898) and got excluded from primary candidacy only bcuz the plan excluded it
  by name. If it hadnt, the rule would have picked the counterfactual that supplies no task
  knowledge at all

So the wall isnt "thresholds need more tuning". its that every answer-key-free criterion
tried here is a proxy for how much an intervention moved something, and the quantity
actually needed is whether it moved the right thing, and "the right thing" is exactly
what the answer key encodes.

Phase 9 and phase 10 came at this from opposite ends. phase 9 had a fixed set of
human-authored counterfactuals and tried to grade them after running them. phase 10
generated its own and tried to pick between them before running them. neither had a
published head list. each reached for the best answer-key-free statistic it had. Both landed
on a quantity that measures how hard an intervention pushed rather than what it pushed on

worth being clear abt what that convergence does and doesnt prove. two failures arent a
theorem, and both got designed by the same person, which is a shared-cause risk worth naming
instead of waving off. What makes it more than coincidence is that the two phases had
different objects, different statistics, different validation strategies (phase 9 had a real
holdout circuit, phase 10 had a rediscovery target) and the diagnosis that explains one
explains the other w no adjustment.

### The two follow-ups, and what they close

both ran after the section above was first written and both tested an obvious escape route
from it.

the scheme-level re-analysis ([note](results/SCHEME_LEVEL_NOTE.md)) closed the
level-of-description escape. §5 below had noticed phase 9 ran its ten candidate signals
over 33 heads when the thing a reader has to judge is a counterfactual. twenty signals
over the 13 `(circuit, scheme)` rows, against a threshold-free label, under a family-wise
permutation correction. best reaches |rho| = 0.495 where shuffled labels reach 0.538 half
the time. inconclusive at n = 13, and theta turns out to be orthogonal to aim rather
than a partial explanation of it. docstring's `random_vocab_any`, the scheme theta correctly
kills, ranks the published heads second-best of all thirteen. ranking quality and scale
integrity are separate axes and theta measures only the second.

phase 11 ([report](results/PHASE11_REPORT.md)) closed the more-data escape and did
something the other three didnt. every signal tried up to that point is a functional of
one measurement. replication isnt. Ten independent resamples of docstring and greater-than
under every registered scheme, 90 sweeps, 9 scheme rows, asking whether a finding comes back
when you redraw the prompts and corruption instances.

all three stability statistics score below the magnitude they were meant to improve on.
median AUC change -0.045 for signal-to-noise ratio (exact Wilcoxon
p = 0.008) and -0.193 for sign consistency (p = 0.004), against a pre-registered bar of
0.017. the most on-target test, "do the flagged docstring heads that are published show
more reproducible disagreement", passed as registered at AUC 0.833, p = 0.046, and then lost
to its own comparator. plain magnitude on the identical 17 heads scores 0.976 at p = 0.0029.

the mechanism got registered as a diagnostic before the run so the reading couldnt be
invented afterwards. replication noise is multiplicative. Spearman(sd, |effect|) runs
+0.58 to +0.86 in all nine rows and published heads' median replication sd is 2.4x to 17.5x
that of the rest, so |effect| / sd is a saturating transform of magnitude that compresses the
ranking exactly at the top, where the circuit lives. the hypothesis was backwards, real
components come out noisier in absolute terms cuz their variance scales w their effect

### what phase 11 changes about the diagnosis

underneath the negative, the pipeline's findings already replicate. in 7 of the 8 scheme
rows where the question is even defined, every head discovered at seed 0 clears its bar
again in 10 of 10 independent resamples, and ten-fold averaging changes head-level AUC by a
median of exactly zero.

Up to this point the wall could still have been read as a data problem. 128 prompts isnt
many, effects are small, maybe a variance-aware criterion or a bigger sample resolves what a
point estimate cant. apparently not. the measurements are valid (phase 9), correctly scaled
(phase 9's theta) and stable (phase 11), and the question of whether theyre about the
behaviour under study is untouched by all three. that rules out the whole family of fixes that
treat the problem as statistical, which is a bit stronger than what three empty searches
could say on their own.

so §2's title went from two investigations to four and its conclusion got sharper. the wall
isnt "thresholds need more tuning", it never was, and its now also not "the estimates are
too noisy to act on". every answer-key-free criterion this pipeline can compute is a
property of the intervention, and the quantity needed is a relation between the
intervention and the behaviour. nothing in the pipeline's outputs encodes the second term of
that relation

### what phase 12 adds, from outside the paradigm

[phase 12](results/PHASE12_REPORT.md) is the only one here that isnt an activation-patching
experiment. causal scrubbing (Redwood, [Chan et al. 2022](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing))
resample-ablates everything outside a claimed circuit and runs the model on real prompts, so
it never builds a counterfactual pair at all. that is exactly the missing slot named at the
end of the paragraph above, a quantity relating an intervention to the behaviour rather than
describing the intervention.

the method does fill it. the published IOI circuit passes its own scrub, recovering 1.022 of
the logit difference and 0.870 of the KL against a floor where every head is resampled. its
per-head signal, one head added to an empty hypothesis, separates published from unpublished
at AUC 0.799 with permutation p = 5.0e-5, computed with no answer key and no counterfactual.

discriminating is where it stops. 25 of 200 uniformly random 26-head sets clear the same bar
the published circuit clears and 8 beat it outright, dropping 12 of the 26 published heads
still scores 0.911 with 30% of those deletions landing above the intact circuit, adding
random heads never hurts, a 3-head set scores 0.968. a very large family of head sets is
sufficient under resample ablation, so a pass narrows the space of hypotheses hardly at all.

this is a different ceiling from the four before it and shouldnt get filed with them. the
other four failed on the statistic, and phase 11 measured the mechanism, replication noise
scaling with effect size. here the question itself is too weak. what would sharpen it is a
hypothesis with internal structure to condition on, causal scrubbing's interpretation graph,
which phase 12 deliberately left out.

one result did come back positive. leave-one-out from the published circuit ranks the
previous-token heads `4.11` and `2.2` first and third of all 26, and both score about 0.0025
under either of phase 1's counterfactuals, one of which is the `s2_swap` scheme that returns
a floating-point exact zero everywhere before S2. Spearman between the scrub's head ranking
and patching magnitude is 0.240 across all 144 heads. Twelve phases in thats the first
answer-key-free signal here that disagrees with magnitude in a direction the answer key
confirms, and its untested on greater-than and docstring.

---

## 3. whats still human, and looks like staying that way

these ones specifically, across all twelve phases.

### Still human, untouched

| what | where it shows up | why nothing recovers it |
|---|---|---|
| the behavioural hunch | every phase | picking which behaviour to study was never attempted. phase 5 passed on it bcuz a weak version "would produce something that looked like progress without being any", and phase 10 deliberately attacked the rung below this one. |
| where to cut the prompt | phase 10 | the 64 fixture lines all stop right before the answer token. nothing in the induction recovers that decision from them, its baked into the examples a person typed. |
| which counterfactual to trust | phases 5, 7, 8, 9, 10 | section 2. this is the load-bearing one. phase 12 sidesteps it by using a method with no counterfactual and hits a different limit instead. |
| whether a flag matters | phases 8, 9, 11 | the pipeline flags counterfactual-dependent heads unprompted and on two of three circuits the flag still needs a human w an answer key. flag precision runs 50% / 0% / 26%. phase 11 adds that this isnt a noise problem. |

### was human, now mechanized, and what it cost

| what | closed by | cost |
|---|---|---|
| receiver input and position | phase 4 | none measurable. 16 of 17, and bare indices work as well as semantic labels |
| the answer key in the metric | phase 5 | none. 19/26 vs 18/26, rankings correlate +0.98 |
| corruption position | phase 4 | none |
| corruption content | phase 5 (generic), phase 10 (proposed) | real. fully generic costs a third of recall (13/26 vs 18/26) |
| task template and slot vocabularies | phase 10 | real. 3/7 pre-registered, 5/7 repaired, against a hand-built 6/7 |
| a second counterfactual per task | phase 8 (forced), phase 10 (derived) | phase 8's greater-than alternate was authored by this project and is labelled `authored` not `published` everywhere it shows up |
| knowing the measurements replicate | phase 11 | 1.26 GPU-hours, and the answer was yes, so no fix lives here |
| grading a circuit against behaviour, no answer key | phase 12 | 1.34 GPU-hours. the grade exists and passes 12.5% of random head sets, so its not usable as a filter yet |

### human in a way thats easy to miss

not in any ladder:

- which published circuit to target, and what counts as a valid answer key. phase 7
  rejected GPT-2 medium bcuz its published IOI circuit is defined as "the 2% most important
  heads" instead of a named list, and rejected Pythia for having no head-level published
  circuit at all. both rejections are in the plan. deciding a fuzzy ground truth doesnt count
  is a human judgment and its the judgment that keeps the validation meaningful.
- what "found" means. phase 3 produced two criteria that disagree about which heads count
  and reported them as two columns, refusing to merge. choosing between "it explains the
  behaviour" and "it delivers content to the next stage" isnt a measurement.
- every diagnosis of every failure. Phase 7's missing routing heads, phase 8's
  false-positive flags, phase 9's which-flags-mattered, phase 10's which-scheme-was-right,
  phase 11's which-heads-were-real. All diagnosed by reading a published head list. The
  pipeline generated the measurements. a person holding the answer key generated the
  interpretations.

---

## 4. the founding goal, more or less answered

stated aim is scalable oversight. Point causal interpretability at models more capable than
the people and systems investigating them, where no human has the ground truth to check
against.

the measurement layer works and it doesnt need an answer key. give it a behaviour, a task
and a well-aimed counterfactual and it finds the circuit that implements it, recovers the
causal ordering between the parts, works out where each component reads its input without
being told, scores the lot with a metric that never sees the correct token, and does all
of that on three published circuits across two model architectures with the causal core
unmodified. builds most of the task itself from example sentences now, too. and phase 11
adds that the readings reproduce 10 out of 10. thats a real instrument, and its been
checked against published answers rather than asserted.

The judgment layer doesnt work without an answer key and every attempt to build one has
failed the same way. the system cant tell you whether the circuit it just handed you is
the circuit or is whatever your experiment happened to be able to see. Phase 7 is the
proof by example, a clean plausible internally consistent result missing half the
mechanism, caught only by a published head list. phase 8 turned that from "a human notices
afterwards" into "the pipeline prints a warning", which is real progress. Phase 9 tried to
make the warning gradeable and couldnt. phase 10 showed the same gap shows up one level
up, in choosing the experiment at all. Phase 11 showed the gap isnt a noise problem, so it
wont close by measuring harder.

and that ordering is the worst one for the stated aim, the pieces that work are the ones
you can validate against a known answer and the one that doesnt is the one whose whole job
is to substitute for a known answer. on an unfamiliar model theres no
published head list to notice a 50% recall against, and phase 7 measured exactly that
scenario in miniature and got a confidently wrong answer.

so, roughly: built and validated an instrument. didnt build the thing that would let you
trust its readings on a target where nobody can check them. the instrument is worth having,
several of its components were genuinely uncertain and theyre demonstrated now. but the gap
between it and the stated aim isnt a matter of more phases of the same kind, and nothing in
1-12 should be read as evidence its closing.

the prediction record is bad and its public, which is worth something on its own. phase
1's central hypothesis, that path patching
would recover the classes activation patching missed, got tested by phase 2 and refuted.
Of the six phases that scored a full prediction table:

| phase | predictions | held | wrong |
|---|---|---|---|
| 6 | 4 | 3 | 0 (one scored a tie, not a hit) |
| 7 | 7 | 3 | 3 (one split) |
| 8 | 8 | 6 | 2 |
| 9 | 8 | 4 | 4 |
| 10 | 8 | 1 | 7 |
| 11 | 9 | 6 | 3 |
| total | 44 | 23 | |

23 of 44. coin flip, basically, from someone who designed the experiments and in several
cases already had the answer key sitting right there in the repo. every one got committed
before the run and scored whichever way it came out, and several phases (2, 7, 9, 10, 11)
lead with the prediction they got wrong. alongside the recovery numbers the negative results
are what actually got demonstrated, and the prediction record is the main reason to believe
the positive numbers at all

---

## 5. what a real next step would have to look like

not a plan exactly. more the shape of the problem, for whoever decides later.

after phase 11 the accumulated evidence supports calling this an open problem instead of
continuing to hunt for a fix inside the current pipeline's paradigm. four attempts, four
different statistics, four different objects, one failure, and the fourth returned a
reason instead of another empty search. the two things that would change this are at the
end of this section and neither is a cleverer statistic. phase 12 has since tested the
second of them, and its annotated in place below.

more threshold tuning inside the current framework is ruled out. phase 9 tuned a scalar
criterion over head effects and produced a defensible rule thats not a discriminator. Phase
10 tuned a scalar criterion over counterfactuals and picked the wrong one twice. different
statistics on different objects, same failure, and abt the best evidence available that the
lever isnt there

also ruled out and this one is new, anything that treats the residual as noise.
phase 11 resampled both circuits ten times and found the findings already reproducible. 7 of
8 scoreable schemes rediscover every seed-0 head in 10 of 10 resamples and ten-fold
averaging moves head-level AUC by a median of exactly zero. more prompts, more seeds,
bootstrap confidence intervals, variance-aware thresholds, all of these fix a problem this
pipeline doesnt have. worse, the one such statistic that got tried, effect over its own
replication spread, is significantly worse than the raw effect, cuz replication noise
scales w effect size and dividing by it compresses the ranking exactly where the circuit
sits.

turn it around and state it as a supervised question, cuz thats what it quietly became:

> Given a `(task, counterfactual)` pair and no answer key, predict whether that
> counterfactual is aimed at the behaviour or just damaging the prompt.

Every phase so far answered a version of "how big is this effect". this is a different
question and its never been asked directly.

three things make it tractable enough to be worth scoping and one makes it hard.

1. Labelled data already exists in this repo. phase 9's floor table is 13
   `(circuit, scheme)` rows w power, null median, null max and theta measured for each.
   phases 8 and 10 supply the labels. `yy01` is well-aimed for greater-than (7/7),
   `random_random` is badly aimed for docstring's routing heads (3/6 where `random_def` gets
   5/6), `random_vocab_any` is badly aimed everywhere it shows up, phase 10's `resample_t8`
   is well-aimed and `resample_t7` isnt. Thats a small labelled set of experiments, not of
   heads, and phase 9 only ever ran its ten candidate signals at the head level over 33
   flagged heads. the scheme-level version hasnt been tried.

   > Since run, and it doesnt work.
   > [`results/SCHEME_LEVEL_NOTE.md`](results/SCHEME_LEVEL_NOTE.md) put all ten signals plus
   > ten stored scheme-level fields against a threshold-free label over these 13 rows. Under
   > a family-wise permutation correction the best of the twenty reaches |rho| = 0.495 where
   > shuffled labels reach 0.538 half the time. inconclusive at n = 13, and theta turns
   > out to be orthogonal to aim rather than a partial explanation of it. docstring's
   > `random_vocab_any`, the scheme theta correctly kills, ranks the published heads
   > second-best of all thirteen. the note also corrects this paragraph. the labels listed
   > above arent one rule but three, since `random_vocab_any`'s recall on docstring is 1.000
   > and its "badly aimed everywhere" label is really coming from precision.

2. There are untested candidate signals that arent magnitude. two examples, offered as
   illustrations of the kind of thing and not as proposals. whether a counterfactual's
   effect is consistent across prompts (a well-aimed intervention should perturb the
   same computation every time, a damaging one should perturb something idiosyncratic per
   prompt), and whether its effect is low-dimensional in the output distribution instead
   of diffuse. both computable from two forward passes and neither reduces to "how far did
   it move".

   > the first has now been tested in its across-runs form and it fails.
   > [phase 11](results/PHASE11_REPORT.md) measured consistency across ten independent
   > resamples rather than across prompts within one run, a related but distinct quantity,
   > and every statistic built on it came in below plain magnitude, two of them
   > significantly. the reason generalizes to the within-prompt version and should get
   > checked before anyone runs it. Consistency statistics divide by a spread thats itself
   > proportional to the effect, so they systematically penalise the heads w real effects.
   > The low-dimensionality idea is untouched and isnt obviously subject to the same
   > defect, since its a shape statistic and not a ratio.

3. phase 8 measured something that points the opposite way from its own flag. across the
   four greater-than schemes 0 heads were found by every scheme. on docstring, 6 were.
   Agreement between counterfactuals of genuinely different design is rare, and where it
   happens it might carry more information than the disagreement does. nothing has followed
   that up

4. And the hard part, n is very small. Thirteen scheme-circuit pairs, three circuits, two
   models, one architecture family. A criterion fitted to that will be fitted to it. Honest
   possibility is the prerequisite isnt a cleverer statistic but more published circuits,
   and phase 7's rejection of GPT-2 medium and Pythia is a record of how scarce checkable
   ground truth actually is at this scale.

   > Both later efforts ran straight into this and neither escaped. the scheme-level note
   > was inconclusive at n = 13 against a null whose 95th percentile is 0.764. phase 11
   > dropped to n = 9 and got registered as underpowered before it ran. At these sizes a real
   > signal of moderate strength would be invisible and no analysis here can tell "there is
   > nothing" from "there is something we cannot see". this is the single most load-bearing
   > constraint in the whole document now.

so the two things that would change the recommendation, neither of them a new statistic
over the existing outputs:

- more published circuits, enough that predicting a counterfactual's aim becomes a
  supervised problem at a sample size where a moderate signal is detectable. n = 9 and
  n = 13 arent that and no reweighting of them will be.
- a criterion that references the behaviour under study independently of the counterfactual
  being graded. every quantity the pipeline computes is a property of the intervention. how
  large, how valid, how reproducible, how it ranks. the needed quantity is a relation
  between the intervention and the behaviour and the architecture has no slot holding the
  second term. adding one is a design change

  > Built, in [phase 12](results/PHASE12_REPORT.md), and it doesnt change this
  > recommendation. causal scrubbing has the slot, it grades a hypothesis against real
  > behaviour on real prompts with no counterfactual anywhere, and the published IOI circuit
  > passes at 1.022 of the logit difference. the trouble is that so does most of the space.
  > 12.5% of random 26-head sets clear the same bar, dropping 12 of the 26 published heads
  > still scores 0.911, and a 3-head set scores 0.968. Sufficiency under resample ablation
  > is too weak a property to grade with. what the phase does buy is a head-level signal at
  > AUC 0.799 that correlates with patching magnitude at only 0.240, so the second term of
  > that relation is now measurable even if it isnt yet decisive. the next version of this
  > experiment is a real interpretation graph, scrubbing edges instead of nodes so
  > equivalence classes get conditioned on. thats an implementation, not another statistic.

the validation problem is the real obstacle and it should get stated before any of the
above is attempted. every phase so far validated against a published head list. what needs
validating now is a ranking rule over experiments, and a published circuit checks that only
indirectly. you learn the rule picked the scheme that recovers more published heads, which is
one bit per circuit. if a future phase cant say in advance what would falsify its ranking rule
on a circuit nobody has published, it produces another defensible criterion nobody can grade,
which is what phase 9 produced and said so.

---

## reading further

evidence for everything above:

- per-phase reports: [`results/PHASE1_REPORT.md`](results/PHASE1_REPORT.md) thru
  [`results/PHASE12_REPORT.md`](results/PHASE12_REPORT.md)
- the one re-analysis thats not a phase:
  [`results/SCHEME_LEVEL_NOTE.md`](results/SCHEME_LEVEL_NOTE.md)
- Pre-registrations, committed before the code they judge:
  [`PHASE4_SEARCH_SPACE.md`](results/PHASE4_SEARCH_SPACE.md),
  [`PHASE5_AUDIT.md`](results/PHASE5_AUDIT.md),
  [`PHASE6_PLAN.md`](results/PHASE6_PLAN.md), [`PHASE7_PLAN.md`](results/PHASE7_PLAN.md),
  [`PHASE8_PLAN.md`](results/PHASE8_PLAN.md),
  [`PHASE9_CHARACTERIZATION.md`](results/PHASE9_CHARACTERIZATION.md) +
  [`PHASE9_PLAN.md`](results/PHASE9_PLAN.md),
  [`PHASE10_PLAN.md`](results/PHASE10_PLAN.md) +
  [`PHASE10_CHARACTERIZATION.md`](results/PHASE10_CHARACTERIZATION.md) +
  [`PHASE10_AMENDMENT.md`](results/PHASE10_AMENDMENT.md),
  [`PHASE11_PLAN.md`](results/PHASE11_PLAN.md),
  [`PHASE12_PLAN.md`](results/PHASE12_PLAN.md)
- the full narrative, phase by phase, w setup and run instructions:
  [`README.md`](README.md)

six code separations are what make the central claims checkable rather than promised.
`search.py`, `agreement.py`, `pipeline.py`, `schemes.py`,
`induction.py` and `autotask.py` must never import a `ground_truth` module and the runners
assert it at startup. The three published circuits live in separate modules so a run cant get
scored against their union. `comparison.py` is pure set arithmetic over whichever circuit it
gets handed. Phase 10's human input lives in [`fixtures/`](fixtures/) as plain text so the
human contribution to that phase can be counted instead of described. and phase 11
separates measurement from analysis into different files, `run_phase11_resample.py` asserts
it imports no answer key of itself as well as of the causal core, and `phase11_analysis.py`
writes its blind half to disk before its scoring half is entered, so the ordering is a
property of the file layout and the commit history. `scrubbing.py` joins that list in phase
12 and `run_phase12_scrub.py` checks its source text for the string at startup
