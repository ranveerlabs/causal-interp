# causal-interp — ten phases, one retrospective

This is the entry point. It pulls together what ten pre-registered phases actually
established, what they failed to establish, and what the failures have in common. Every
number here is quoted from a committed phase report in [`results/`](results/); nothing is
re-derived or rounded from memory, and the report each figure comes from is named.

If you read one thing about this project, read this. If you then want the evidence for
any single claim, the phase reports are where it lives.

---

## The short answer

The project set out to build a system that discovers and **causally validates**
mechanisms in neural networks, aimed eventually at models more capable than the people
checking them. Ten phases in, the honest position is a clean split:

> **Everything that can be settled by measuring a magnitude has been mechanized.
> Everything that requires judging relevance still needs a human or an answer key.**

Locating a circuit, ranking components, finding where a head reads its input, replacing
the answer key in the metric, transferring to a new task, transferring to a new model,
building the task itself out of example sentences — all of that now runs without being
told the answer, and has been checked against three published circuits in two models.

Deciding **which experiment to believe** does not. Two independent investigations
(Phases 9 and 10) attacked that from different directions and hit the same wall. That
wall is the project's central open problem, and it is precisely the problem that matters
for the scalable-oversight framing, because "no answer key exists" is the defining
condition of that setting.

---

## 1. What was validated, and how solidly

### The ladder, phase by phase

| phase | question | headline | how solid |
|---|---|---|---|
| [1](results/PHASE1_REPORT.md) | does activation patching recover a published circuit? | **20/26** IOI heads (union of two counterfactuals; 18/26 under the primary alone) | strong — and it named its own blind spot |
| [2](results/PHASE2_REPORT.md) | does path patching close the gap? | **19/26** alone, **20/26** combined — the gap did **not** close | strong, and a failed prediction |
| [3](results/PHASE3_REPORT.md) | does a pre-registered receiver-side criterion find the rest? | **2 of the 6** missing heads, at precision 0.64 | medium — threshold fixed in advance, criterion noisier |
| [4](results/PHASE4_REPORT.md) | can the search find where a head reads, without being told? | **16 of 17** scoreable specifications | strong — exhaustive grid, 9,936 forward passes |
| [5](results/PHASE5_REPORT.md) | which hand-built pieces actually carry the result? | answer key **not needed**; corruption content **is** | strong, and the result inverted the prediction |
| [6](results/PHASE6_REPORT.md) | does any of it transfer to a second circuit? | **7/7** greater-than heads, causal core untouched | strong on transfer, deflated on difficulty |
| [7](results/PHASE7_REPORT.md) | does it transfer to a different model? | **3/6** — code transferred perfectly, results got worse | strong as a negative; 3 of 7 predictions wrong |
| [8](results/PHASE8_REPORT.md) | can the pipeline flag its own blind spot? | **yes, loudly** — 3/6 → 6/6 on docstring, flagged before the answer key | real, and half an answer |
| [9](results/PHASE9_REPORT.md) | can it tell a real blind spot from noise? | **partial** — a better criterion, **not** a discriminator | strong negative, with a genuine holdout |
| [10](results/PHASE10_REPORT.md) | can the task be induced instead of written? | **3/7** pre-registered (5/7 repaired) vs hand-built 6/7 | honest negative; 1 of 8 predictions held |

### The four things that are genuinely established

**(a) Activation and path patching recover published circuits, and the misses are
structural rather than sloppy.** Phase 1 recovered 20 of Wang et al.'s 26 IOI heads and
explained the six misses before anyone asked: they act *through* other heads, which
total-effect patching cannot resolve. It also measured its own blindness instead of
asserting it — under the `s2_swap` counterfactual, **576 of 576** head-position cells
before the S2 token are exact floating-point zeros, because the two runs are computing on
bit-identical inputs there. Any published head whose role sits at those positions is
undiscoverable by that scheme *in principle*.

Phase 2 then tested Phase 1's own prediction that path patching would close the gap. It
did not — 19/26 alone, still 20/26 combined, the same six heads missing. What did improve
was precision (0.71 → 0.90; **1.00** under the `abc` scheme alone) and, more interestingly,
the *causal ordering*: each round's receivers were the previous round's discoveries, the
answer key was never consulted to choose them, and round 1 returned **all four** published
S-inhibition heads as its top four out of 144.

The failed prediction is reported as a failed prediction. That is the pattern for the
whole project.

**(b) A pre-registered threshold recovers two more heads, and it is honestly noisier.**
Phase 3 fixed the rule before measuring — *99th percentile of the path signal under a
shuffled-source null, rounded up to two significant figures* — and got **0.11**. It
recovered `2.2` and `4.11`, both previous-token heads that neither earlier phase could
see, at precision **0.64** against the logit criterion's 0.90.

Two details make this more credible than the number alone. First, the pre-registration is
a separate commit containing the threshold and the code and **no results**, so the
ordering is checkable in git rather than asserted. Second, the phase explicitly refused to
merge the two criteria into one recall figure, because they disagree about *which* heads
count — 5 heads found by both, 16 by the logit criterion only, 6 by the receiver-side one
only. Merging would have reported a bigger number while destroying the only new
information the phase produced.

**(c) The search finds where a head reads its input, blind.** This is the cleanest
positive result in the project. Phase 4 scored every `(layer, head, input, position)`
specification by splicing that one input from the clean run into the corrupted one, over
an exhaustive grid. `search.py` does not import a ground-truth module and the run asserts
that before starting.

Of the 21 published heads with a published receiver specification: **16 agreement, 0
ambiguous, 4 unmeasurable, 1 disagreement** — so **16 of the 17 it could actually weigh**.
All four S-inhibition heads returned `v@S2`; every name mover, backup and negative name
mover returned `q@END`.

The 4 "unmeasurable" are the Phase 1 blind spot arriving again: the published spec for the
induction heads is `k@S1+1`, and under `s2_swap` *all 432 specifications at that position
score exactly zero*. The search never weighed the published answer and preferred something
else — it was handed a counterfactual that cannot see that position. Counting those as
search failures would blame the search for a defect belonging to the experiment.

And the position labels turned out to be unnecessary. An unlabelled search over bare token
indices concentrated its top 50 on `t11` (×33), `t15` (×15) and `t12` (×2) — which turn
out to be S2, END and S2+1, labels attached *after* the search purely to read its output.

**(d) The machinery transfers; the results deflate honestly.** Phase 6 pointed the whole
pipeline at a second published circuit in the same model and recovered **7/7** greater-than
heads and **7/7** receiver specifications. `interventions.py`, `search.py` and `metrics.py`
— the causal core — were **not touched**; `comparison.py` gained a `circuit` parameter it
should always have had. The phase deflated its own headline: seven targets is an easier set
than twenty-six, and this circuit has no analogue of IOI's previous-token heads.

Phase 7 then pointed it at a different model — `attn-only-4l`, 4 layers, 8 heads, a
different tokenizer, **no MLP blocks at all**. The code transfer was total: the report runs
`git diff --stat` over the ten pre-existing modules and pastes the empty output. The
results were the worst of the three circuits:

| circuit | model | published | recovered | recall | precision | chance recall |
|---|---|---|---|---|---|---|
| IOI | GPT-2 small | 26 | 18/26 | 69% | 0.78 | 18% |
| greater-than | GPT-2 small | 7 | 7/7 | 100% | 0.78 | 5% |
| **docstring** | **`attn-only-4l`** | **6** | **3/6** | **50%** | **0.33** | **19%** |

The plan had pre-registered a deflation — 6 heads among 32 is an easier denominator — and
it cut *less* than expected: chance recall is 19% here against IOI's 18%, so the drop from
69% to 50% is real and not an artifact of circuit size.

**Phase 7 is the most important phase in the project**, and not because of the number. It
produced a clean, plausible, internally consistent circuit claim that was **missing half
the mechanism**, and only the published head list revealed that. The cause is now
well-understood: the benchmark's default counterfactual replaces the *answer token*, which
makes the circuit's routing heads causally invisible to a metric read off the output. A
different published counterfactual finds **5 of 6**. Which parts of a circuit are
discoverable is a property of the experiment, not of the circuit.

Two silent failure modes also surfaced there, neither of which appears in a diff: the
component sweep patched `mlp_out` on a model with no MLPs and returned **28 exact zeros**
that read as "the MLPs carry no causal signal", and the iterative path chain assumed
another layer existed below and halted after one round, leaving the receiver-side null
pooled from **16** measurements instead of hundreds.

### One more established result, easy to under-rate

**The answer key in the metric is not needed.** Phase 5 replaced the hand-built logit
difference — which requires knowing which two tokens are the candidates and which is
correct — with divergences over the *whole* next-token distribution. Size-matched to the
published circuit's 26 heads: logit difference **18/26**, KL **19/26**, total variation
**19/26**, with per-head rankings correlating at **+0.98**.

But the same phase found the limit right next to it. Making the *corruption* generic as
well is what costs:

| what is supplied | size-matched recovery |
|---|---|
| hand-built corruption + hand-built metric | 18/26 |
| hand-built corruption + general metric | **19/26** |
| generic corruption + hand-built metric | **18/26** |
| generic corruption + general metric | 16/26 |
| nothing supplied at all | **13/26** |

The two pieces are not independently load-bearing and not additive. Either one alone
carries enough task knowledge; what fails is removing both. `s2_swap` was built to
*reverse* the behaviour, so the corrupted run sits as far from clean as the task allows
(a logit span of 7.30). A random token merely *damages* the prompt — the corrupted run
still partly performs the task and the span collapses to 1.54.

Hold onto that distinction between **reversing** and **damaging**. It is the whole of
section 2.

---

## 2. Two investigations, one wall

Phases 9 and 10 are the project's two most recent efforts. They attacked different
problems with different machinery and produced the same shape of failure. **That
convergence is itself the finding** — it is evidence about where the real difficulty
sits, not two unrelated disappointments.

### Phase 9: given several counterfactuals that disagree, which disagreement matters?

Phase 8 had made multi-counterfactual discovery structural: a task must register at least
two schemes, `pipeline.discover()` sweeps all of them, and the pipeline prints which heads
change status between them. Pointed back at Phase 7's circuit it named `1.2`, `1.4` and
`2.0` — exactly the three heads Phase 7 missed — **before any answer key was opened.**

And the same flag fired on greater-than, where the primary counterfactual had already
recovered 7/7 and there was nothing to find:

| circuit | flag | heads flagged | of which published | recall, primary → union | precision |
|---|---|---|---|---|---|
| docstring | fires | 17 | **3** | 3/6 → 6/6 | 0.33 → 0.23 |
| greater-than | fires | 16 | **0** | 7/7 → 7/7 | 0.78 → 0.28 |

The two flags are identical in form. Phase 9 tried to tell them apart.

It measured ten candidate signals first, in a separate commit, before designing anything.
**Nine do not separate the two cases**, and several separate in the wrong direction: any
statistic normalized by a scheme's median is dominated by how many dead heads a model has,
so GPT-2 small's noise outranks `attn-only-4l`'s real finds.

The one rule worth pre-registering was a real improvement: calibrate each scheme against
its *own* shuffled-source null instead of a shared cutoff. That much is established
decisively. The ten measured floors span a factor of **400** — from θ = 0.0077 to θ = 3.3.
Docstring's `random_vocab_any` has a null whose 99th percentile is **3.3**, meaning that
patching a head with an activation from a *different prompt* routinely moves that metric by
several times the entire clean-to-corrupted span. It had contributed 17 of docstring's
flags under the shared cutoff and contributes none under its own floor. Phase 8's shared
0.02 was not defensible, and that is now settled.

It is still not a discriminator:

| circuit | known case | heads flagged | flag precision |
|---|---|---|---|
| docstring | real blind spot | 17 → **4** | 18% → **50%** |
| greater-than | no blind spot | 16 → **8** | 0% → **0%** |
| **IOI** (holdout) | real blind spot | 15 → **19** | 20% → **26%** |

IOI is a genuine holdout — Phase 8 registered its four schemes and deliberately never ran
it, and its expected behaviour was fixed in the plan from Phase 1's finding. Its calibrated
blind spot **grew**, because `abc` and `random_vocab_s2` came out with floors *below* 0.02
and became more sensitive, while the primary's floor rose to 0.058 and cost it three
published heads of its own (18/26 → 15/26). Calibration cuts both ways.

**Phase 9's own summary of what it achieved:** the null floor separates *this scheme is too
weak to be believed* from *this scheme measured something*. It does **not** separate *this
scheme measured something the primary is blind to* from *this scheme measured something
outside the circuit* — because both of those are statistically sound measurements under a
real counterfactual.

### Phase 10: given several auto-proposed counterfactuals, which one should be primary?

Phase 10 stopped writing the task by hand. From 32 example sentences a person typed from a
one-sentence hunch, it induces the template (constant token columns), the slots (varying
ones), the tied slots (columns that co-vary), the vocabularies (the observed values), the
positions (bare indices, which Phase 4 showed suffice), one counterfactual per slot, and an
answer-key-free metric. All of it reaches `pipeline.discover()` as an ordinary `TaskSpec`
with **no pre-existing module changed**.

Pre-registered, it recovered **3 of 7** published greater-than heads against the hand-built
task's 6 — a negative by its own scoring table. **And the failure came apart into two
separable halves**, which is what makes it useful.

*Half one* was a bug, and it is fixed post-hoc: two of the thirty-two human lines contain a
year GPT-2 splits as `[" 150", "9"]` rather than `[" 15", "09"]`. Same row length, so the
filter kept them; the tie rule needs unanimity, so the century columns stopped being one
slot; generation then sampled them independently and produced clean prompts like `The
pilgrimage lasted from the year 11245 to the year 14`. The model's top prediction exceeded
the start year on **48%** of them — a coin flip, which is the published task's own
`xx_mismatch` counterfactual served as the *control*. Nothing said so.

*Half two* is not a bug:

| scheme | size-matched | precision |
|---|---|---|
| `resample_t2` (the noun) | 2/7 | 0.09 |
| **`resample_t7`** (start century) — **chosen** | **3/7** | **0.22** |
| `resample_t8` (start year) | **5/7** | **0.50** |
| `resample_END` (final century) | 4/7 | 0.46 |
| `random_vocab_any` (generic) | 1/7 | 0.12 |

`resample_t8` redraws the **start year** — exactly the position the published `yy01`
counterfactual acts on. It was sitting in the proposed set, scoring 5/7 at precision 0.50.
The answer-key-free rule picked `resample_t7` instead, on an output divergence three times
larger. On the second fixture the same rule picked `resample_END`, a third position again.

**The counterfactual *content* mechanized. The counterfactual *ranking* did not.**

The repair confirms the diagnosis rather than rescuing the headline: once the odd lines are
dropped the tie returns, `desync_t7` is proposed — Phase 8's hand-authored `xx_mismatch`,
re-derived from example sentences — `resample_t7`'s divergence collapses from 0.755 to
0.084 because both centuries now move together, and the rule picks the right scheme on its
own, reaching 5/7 size-matched and **7/7 at the inherited 0.02 cutoff**. That is labelled
post-hoc everywhere and does not replace the 3/7.

### Why these are the same failure

The two rules are not the same rule. Phase 9's sets a per-scheme *threshold* — how large
must an effect be, in this experiment's own units, to count as measured at all. Phase 10's
*ranks* whole schemes against each other by raw KL divergence. Different objects,
different statistics.

What they share is the substitution they both make. Each needed to know whether an
experiment says something about **the mechanism under study**, and each answered a
different question that happens to be computable without an answer key:

- Phase 9 answered a **validity** question — *is this measurement bigger than what this
  experiment manufactures from noise?* — and got a decisively good answer to it.
- Phase 10 answered a **magnitude** question — *which counterfactual moves the output
  most?* — and got a correct answer to that too.

Neither question is the one that was needed, which is a question about **relevance**:

> **A counterfactual can be perfectly valid, move the output a great deal, and still be
> telling you about something other than the behaviour you are studying. Neither validity
> nor magnitude measures aim.**

Phase 9 says this about itself in as many words: the null floor cannot separate a scheme
that disagrees *because the primary is blind* from one that disagrees *because it is
measuring something outside the circuit*, "because both are statistically sound
measurements under a real counterfactual." Phase 10's version is the same sentence with
the objects swapped — a badly aimed counterfactual and a well-aimed one are both real
interventions, and the badly aimed one moved the output further.

**The evidence that magnitude and aim come apart is scattered across the whole project**,
and it points one way every time:

- **Phase 5**: a generic corruption *damages* the prompt where a semantic one *reverses*
  the behaviour. Fully generic recovers 13/26 against 18/26 — and the reason is that a
  distribution-wide metric "shares the shrunken span out over every irrelevant way the
  prompts differ."
- **Phase 8**: the generic schemes "need no knowledge of the task, so any task can register
  them" — and they are simultaneously the lowest-power schemes and the largest source of
  flagged heads in no published circuit.
- **Phase 9**: the scheme that manufactures the *most* apparent recovery from a mismatched
  activation (θ = 3.3) is the one that knows least about the task.
- **Phase 10**: `random_vocab_any` had the **highest** KL divergence of any scheme on both
  fixtures (1.805 and 0.898) and was excluded from primary candidacy only because the plan
  excluded it by name. Had it not been, the rule would have selected the counterfactual
  that supplies no task knowledge at all.

So the wall is not "thresholds need more tuning". It is that every answer-key-free
criterion this project has tried is a proxy for **how much an intervention moved
something**, and the quantity actually needed is **whether it moved the right thing** —
and "the right thing" is exactly what the answer key encodes.

Phase 9 and Phase 10 approached this from opposite ends. Phase 9 had a fixed set of
human-authored counterfactuals and tried to grade them *after* running them. Phase 10
generated its own and tried to choose between them *before* running them. Neither had a
published head list; each reached for the best answer-key-free statistic available to it;
both landed on a quantity that measures how hard an intervention pushed rather than what it
pushed on. **Two methods, two directions, one wall.**

It is worth being clear about what this convergence does and does not prove. Two failures
are not a theorem, and both were designed by the same person, which is a shared-cause risk
worth naming rather than waving away. What makes it more than a coincidence is that the
two phases had different objects, different statistics, different validation strategies —
Phase 9 had a genuine holdout circuit, Phase 10 had a rediscovery target — and the
diagnosis that explains one explains the other without adjustment.

---

## 3. What is still irreducibly human

Not "some things remain". These, concretely, across all ten phases.

### Still human, and untouched

| what | where it shows up | why nothing recovers it |
|---|---|---|
| **The behavioural hunch** | every phase | Choosing *which behaviour to study* was never attempted. Phase 5 declined it on the grounds that a weak version "would produce something that looked like progress without being any", and Phase 10 explicitly attacked the rung below rather than this one. |
| **Where to cut the prompt** | Phase 10 | The 64 fixture lines all stop immediately before the answer token. Nothing in the induction recovers that decision from them; it is baked into the examples a person typed. |
| **Which counterfactual to trust** | Phases 5, 7, 8, 9, 10 | Section 2. This is the load-bearing one. |
| **Whether a flag matters** | Phases 8, 9 | The pipeline flags counterfactual-dependent heads unprompted, and on two of three circuits the flag still needs a human with an answer key. Flag precision runs 50% / 0% / 26%. |

### Was human, is now mechanized — and what it cost

| what | closed by | cost |
|---|---|---|
| Receiver input and position | Phase 4 | none measurable — 16 of 17, and bare indices work as well as semantic labels |
| The answer key in the metric | Phase 5 | none — 19/26 vs 18/26, rankings correlate +0.98 |
| Corruption *position* | Phase 4 | none |
| Corruption *content* | Phase 5 (generic), Phase 10 (proposed) | real — fully generic costs a third of recall (13/26 vs 18/26) |
| The task template and slot vocabularies | Phase 10 | real — 3/7 pre-registered, 5/7 repaired, against a hand-built 6/7 |
| A second counterfactual per task | Phase 8 (forced), Phase 10 (derived) | Phase 8's greater-than alternate was **authored by this project**, and is labelled `authored` rather than `published` everywhere it appears |

### Human in a way that is easy to miss

These are not in any ladder and they matter:

- **Which published circuit to target, and what counts as a valid answer key.** Phase 7
  rejected GPT-2 medium because its published IOI circuit is defined as "the 2% most
  important heads" rather than a named list, and rejected Pythia for having no head-level
  published circuit at all. Both rejections are recorded in the plan. Deciding that a
  fuzzy ground truth does not count is a human judgment, and it is the judgment that keeps
  the validation meaningful.
- **What "found" means.** Phase 3 produced two criteria that disagree about which heads
  count and reported them as two columns, refusing to merge them. Choosing between "it
  explains the behaviour" and "it delivers content to the next stage" is not a measurement.
- **Every diagnosis of every failure.** Phase 7's missing routing heads, Phase 8's
  false-positive flags, Phase 9's which-flags-mattered, Phase 10's which-scheme-was-right —
  all four were diagnosed by consulting a published head list. The pipeline generated the
  measurements; a person holding the answer key generated the interpretations.

---

## 4. The founding goal, answered plainly

The stated aim is scalable oversight: applying causal interpretability to models *more
capable than the people and systems investigating them*, where no human has the ground
truth to check the answer against.

**What ten phases support.** The measurement layer works and does not need an answer key.
Given a behaviour, a task and a well-aimed counterfactual, this system locates the circuit
that implements it, recovers the causal ordering between its parts, finds where each
component reads its input without being told, scores everything with a metric that never
sees the correct token, and does all of that on three published circuits across two model
architectures with the causal core unmodified. It also now builds most of the task itself
from example sentences. That is a real instrument, and it has been checked against
published answers rather than asserted.

**What ten phases do not support.** The judgment layer does not work without an answer key,
and every attempt to build it has failed in the same way. The system cannot currently tell
you whether the circuit it just handed you is the circuit or is what your experiment
happened to be able to see. Phase 7 is the proof by example: a clean, plausible, internally
consistent result that was missing half the mechanism, caught only by a published head
list. Phase 8 turned that from "a human notices afterwards" into "the pipeline prints a
warning", which is genuine progress. Phase 9 tried to make the warning gradeable and could
not. Phase 10 showed the same gap appears one level up, in choosing the experiment in the
first place.

**Why that ordering is the worst possible one for the stated aim.** The pieces that work
are the pieces you can validate against a known answer. The piece that does not work is the
piece whose entire purpose is to substitute for a known answer. On an unfamiliar model
there is no published head list to notice a 50% recall against — and Phase 7 measured
exactly that scenario in miniature and got a confidently wrong answer.

So the accurate summary is: **this project has built and validated an instrument, and has
not built the thing that would let you trust its readings on a target where no one can
check them.** The instrument is worth having; several of its components were genuinely
uncertain and are now demonstrated. But the gap between it and the stated aim is not a
matter of more phases of the same kind, and nothing in Phases 1–10 should be read as
evidence that it is closing.

**One thing that should count in the project's favour, and is not a recovery number.** The
prediction record is bad, and it is public. Phase 1's central hypothesis — that path
patching would recover the classes activation patching missed — was tested by Phase 2 and
**refuted**. Of the five phases that scored a full prediction table:

| phase | predictions | held | wrong |
|---|---|---|---|
| 6 | 4 | 3 | 0 (one scored a **tie**, not a hit) |
| 7 | 7 | 3 | 3 (one split) |
| 8 | 8 | 6 | 2 |
| 9 | 8 | 4 | 4 |
| 10 | 8 | **1** | 7 |
| **total** | **35** | **17** | |

**17 of 35** — roughly a coin flip, from someone who designed the experiments and in
several cases already had the answer key in the repository. Every one was committed before
the run and scored whichever way it came out; several phases (2, 7, 9, 10) lead with the
prediction they got wrong. The negative results are not a caveat attached to the project.
Alongside the recovery numbers they are what the project actually demonstrated, and the
prediction record is the main reason to believe the positive numbers at all.

---

## 5. What a real next step would have to look like

Not a plan. The shape of the problem, for a future decision.

**What is ruled out.** More threshold tuning inside the current framework. Phase 9 tuned a
scalar criterion over head effects and produced a defensible rule that is not a
discriminator. Phase 10 tuned a scalar criterion over counterfactuals and picked the wrong
one twice. These were different statistics on different objects and they failed the same
way, which is the strongest available evidence that the lever is not there.

**What the problem actually is.** Turn it around and state it as a supervised question,
because that is what it has quietly become:

> Given a `(task, counterfactual)` pair and no answer key, predict whether that
> counterfactual is *aimed at* the behaviour or merely *damaging* the prompt.

Every phase so far has answered a version of "how big is this effect". This is a different
question and it has never been asked directly.

**Three things make it tractable enough to be worth scoping, and one makes it hard.**

1. **Labelled data already exists in this repository.** Phase 9's floor table is 13
   `(circuit, scheme)` rows with power, null median, null max and θ measured for each.
   Phases 8 and 10 supply the labels: `yy01` is well-aimed for greater-than (7/7),
   `random_random` is badly aimed for docstring's routing heads (3/6 where `random_def`
   gets 5/6), `random_vocab_any` is badly aimed everywhere it appears, Phase 10's
   `resample_t8` is well-aimed and `resample_t7` is not. That is a small labelled set of
   experiments, not of heads — and Phase 9 only ever ran its ten candidate signals at the
   **head** level, over 33 flagged heads. The scheme-level version has not been tried.

2. **There are untested candidate signals that are not magnitude.** Two examples, offered
   as illustrations of the *kind* of thing rather than as proposals: whether a
   counterfactual's effect is **consistent across prompts** (a well-aimed intervention
   should perturb the same computation every time; a damaging one should perturb something
   idiosyncratic per prompt), and whether its effect is **low-dimensional** in the output
   distribution rather than diffuse. Both are computable from two forward passes and
   neither reduces to "how far did it move".

3. **Phase 8 already measured something that points the opposite way from its own flag.**
   Across the four greater-than schemes, **0 heads were found by every scheme**; on
   docstring, 6 were. Agreement between counterfactuals of genuinely different design is
   not the normal case that disagreement interrupts — it is rare, and where it happens it
   may carry more information than the disagreement does. Nothing has followed that up.

4. **And the hard part: n is very small.** Thirteen scheme-circuit pairs, three circuits,
   two models, one architecture family. A criterion fitted to that will be fitted to it.
   The honest possibility is that the prerequisite is not a cleverer statistic but **more
   published circuits** — and Phase 7's rejection of GPT-2 medium and Pythia is a record of
   how scarce checkable ground truth actually is at this scale.

**The validation problem is the real obstacle, and it should be stated before any of the
above is attempted.** Every phase so far validated against a published head list. What
needs validating now is a *ranking rule over experiments*, and a published circuit checks
that only indirectly — you learn that the rule picked the scheme that recovers more
published heads, which is one bit per circuit. If a future phase cannot say in advance what
would falsify its ranking rule on a circuit nobody has published, it will produce another
defensible criterion that no one can grade, which is what Phase 9 produced and said so.

---

## Reading further

The evidence for everything above:

- **Per-phase reports**: [`results/PHASE1_REPORT.md`](results/PHASE1_REPORT.md) …
  [`results/PHASE10_REPORT.md`](results/PHASE10_REPORT.md)
- **Pre-registrations, committed before the code they judge**:
  [`PHASE4_SEARCH_SPACE.md`](results/PHASE4_SEARCH_SPACE.md),
  [`PHASE5_AUDIT.md`](results/PHASE5_AUDIT.md),
  [`PHASE6_PLAN.md`](results/PHASE6_PLAN.md), [`PHASE7_PLAN.md`](results/PHASE7_PLAN.md),
  [`PHASE8_PLAN.md`](results/PHASE8_PLAN.md),
  [`PHASE9_CHARACTERIZATION.md`](results/PHASE9_CHARACTERIZATION.md) +
  [`PHASE9_PLAN.md`](results/PHASE9_PLAN.md),
  [`PHASE10_PLAN.md`](results/PHASE10_PLAN.md) +
  [`PHASE10_CHARACTERIZATION.md`](results/PHASE10_CHARACTERIZATION.md) +
  [`PHASE10_AMENDMENT.md`](results/PHASE10_AMENDMENT.md)
- **The full narrative, phase by phase, with setup and run instructions**:
  [`README.md`](README.md)

Five code separations make the central claims checkable rather than promised, and are worth
knowing about before reading any result: `search.py`, `agreement.py`, `pipeline.py`,
`schemes.py`, `induction.py` and `autotask.py` must never import a `ground_truth` module,
and the runners assert it at startup; the three published circuits live in separate modules
so a run cannot be scored against their union; `comparison.py` is pure set arithmetic over
whichever circuit it is handed; and Phase 10's human input lives in
[`fixtures/`](fixtures/) as plain text, so the human contribution to that phase can be
counted rather than described.
