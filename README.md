# causal-interp

An autonomous system for discovering and **causally validating** computational mechanisms in neural networks.

## Where this stands

Phases 1–5 validated the method against GPT-2 small's IOI circuit, published in Wang et al. (2022), [*Interpretability in the Wild*](https://arxiv.org/abs/2211.00593) — 26 attention heads in 7 classes, so there is a known answer to check against. **Phase 6 ran the same pipeline against a second published circuit** in that model, **Phase 7 ran it against a circuit in a different model**, **Phase 8 changed the method itself** so the pipeline reports what its counterfactual cannot see, **Phase 9 tried and failed to make that report trustworthy enough to act on**, and **Phase 10 stopped writing the task by hand** and induced one from example prompts instead.

| phase | method | result |
|---|---|---|
| [1](#phase-1--activation-patching) | activation patching | **20/26** — recovers every class acting directly on the output, misses every class acting through another head |
| [2](#phase-2--path-patching) | path patching | **20/26** — Phase 1's prediction that this would close the gap was **wrong**; precision 0.71 → 0.90, and the paper's causal *ordering* recovered |
| [3](#phase-3--a-pre-registered-receiver-side-criterion) | pre-registered receiver-side criterion | recovers **2 of the 6** still-missing heads (both previous-token), at precision 0.64 — a *different* definition of "found", reported beside the first rather than merged |
| [4](#phase-4--searching-for-receiver-specifications) | search for receiver specifications | **16 of 17** scoreable specifications recovered without being told them — and the unlabelled search finds the same token positions the labelled one uses |
| [5](#phase-5--scoping-what-is-still-hand-built) | scope the remaining hand-built pieces | the metric's **answer key is not needed** (19/26 vs 18/26); removing the corruption's knowledge *as well* costs a third of recall; task construction left open |
| [6](#phase-6--a-second-published-circuit) | the same pipeline on a second circuit | **7/7** heads and **7/7** receiver specifications on the greater-than circuit, with the causal core **unmodified** — one shared module needed a parameter, and nothing else changed |
| [7](#phase-7--a-different-model) | **the same pipeline on a different model** | **3/6** — the code transferred with **no existing file changed at all**, and the results are the **worst** of the three circuits. Three of seven pre-registered predictions were wrong |
| [8](#phase-8--counterfactual-disagreement-as-standard-output) | **several counterfactuals per task, compared automatically** | the pipeline **flags Phase 7's blind spot on its own** — 3/6 → 6/6 on docstring, with the three missed heads named before any answer key is opened. It flags greater-than too, where there was nothing to find |
| [9](#phase-9--trying-to-tell-a-real-blind-spot-from-noise) | calibrate each counterfactual against its own null | **partial, and the holdout went the wrong way**. Phase 8's shared cutoff is shown indefensible — the ten measured floors span ×400 — but the recalibration is **not a discriminator**: docstring's flag sharpens 18% → 50%, greater-than's stays at 0%, and IOI's *grows* |
| [10](#phase-10--inducing-the-task-instead-of-writing-it) | **induce the task from 32 example prompts** instead of hand-writing it | **negative as pre-registered — 3/7 against a hand-built 6/7 — and the diagnosis splits in two.** The template, slot vocabularies, positions, counterfactual *content* and metric all mechanize; **choosing which counterfactual to trust does not**, and a 5/7 scheme sat unchosen in the candidate set both times. A post-hoc repair reaches 5/7 and **7/7 at the inherited cutoff**. One of eight predictions held |

Across both definitions of "found", 22 of the 26 published IOI heads have been recovered by something. That figure spans two criteria that disagree about which heads count, and [Phase 3 explains why they are not added together](#why-the-scores-are-not-added-together) — it is not one method's recall.

### What ten phases do and do not demonstrate

**Demonstrated.** Given a behaviour to study, this method locates the circuit that implements it, recovers the causal ordering between its parts, and does so without being told where to look. Phase 4 searched for receiver specifications and recovered 16 of the 17 it could score. Phase 5 showed the metric does not need the answer key. **Phase 6 showed none of that was fitted to IOI**, and **Phase 7 showed none of the code was fitted to GPT-2 small** — pointed at a 4-layer attention-only model with a different tokenizer and no MLP blocks, the whole library ran with not one line changed.

**Not demonstrated, and the gap is not incremental.** Nothing here chooses *which behaviour to study*. Every phase takes its behaviour as given and asks how the model implements it; no improvement to patching, searching or scoring turns that into a method for finding behaviours worth studying. The machinery can test a hypothesis and cannot propose one. **Phase 10 moved the rung below it and not this one**: given a one-sentence hunch and 32 example sentences, most of the task around that hunch can now be built without a person — but the hunch, and the decision of where to cut the prompt, were typed by hand in every run.

**And Phase 7 added a second gap, of a different kind.** Recovery fell to 3 of 6 on a denominator no harder than IOI's, and the reason is that the published counterfactual replaces the answer token, which makes the circuit's *routing* heads causally invisible to a metric read off the output. A different published counterfactual finds 5 of 6. **Which parts of a circuit are discoverable is a property of the experiment, not of the circuit** — and nothing in the pipeline's own output distinguished "the circuit is these three heads" from "this counterfactual can only see three of them". Every diagnosis in that phase was made by consulting the answer key.

**Phase 8 closed the second half of that sentence, and only that half.** A task now registers several counterfactual schemes — it cannot register fewer than two — the pipeline runs discovery under all of them, and it prints which heads change status between them. Pointed back at Phase 7's circuit it names `1.2`, `1.4` and `2.0` as counterfactual-dependent **before any answer key is opened**: the three heads Phase 7 missed and diagnosed afterwards. But the same flag fires on the greater-than circuit, where the primary counterfactual was already finding everything, and nothing the pipeline computes tells the two cases apart. It has learned to say *this head list may be an artifact of the experiment*, not *this head list is wrong*.

**Phase 9 attacked exactly that gap and did not close it.** Ten candidate signals were measured before any fix was designed; nine of them do not separate the two cases and several separate in the wrong direction. The one rule worth pre-registering — calibrate each scheme against its own shuffled-source null instead of a shared cutoff — is a real improvement to *what counts as a measurement*, and on a third circuit held out for the purpose it made the flag **louder rather than quieter**. The honest conclusion is a ceiling, not a fix: disagreement detection can now also say *this counterfactual was too weak to be believed*, and it still cannot say *therefore the primary is missing part of the circuit*. Every diagnosis of which flags mattered was made by consulting a published head list.

**Phase 10 went after the oldest gap on the list — the one Phase 5 named and declined.** Every task in Phases 1–9 was a hand-written template with hand-written slot vocabularies and a hand-designed counterfactual. Phase 10 replaces that with an induction over 32 example sentences a person typed from a one-sentence hunch: constant token columns become the frame, varying ones become slots, columns that co-vary become one slot appearing twice, the observed values become the vocabulary, and one counterfactual is proposed per slot. It runs through `pipeline.discover()` with **no pre-existing module changed**. Pre-registered, it recovers **3 of the 7** published greater-than heads against the hand-built task's 6 — a negative by its own scoring table — and **the two halves of that failure come apart cleanly.** The counterfactual *content* mechanized: a scheme scoring 5/7 at precision 0.50, acting on exactly the position the published counterfactual acts on, was in the proposed set on both fixtures. The *ranking* did not: the answer-key-free rule picked a 3/7 scheme both times. Separately, two lines in thirty-two that GPT-2 splits as `[" 150", "9"]` rather than `[" 15", "09"]` dissolved a structural constraint and turned the clean condition into a coin flip, **silently**. A post-hoc repair fixes that half — reaching 5/7 size-matched and **7/7 at the inherited cutoff** — and is reported as post-hoc throughout.

**The honest ladder**, from what is still supplied to what is now discovered:

| ingredient | status |
|---|---|
| which behaviour to study | **supplied** — no method attempted |
| where to cut the prompt | **supplied** — encoded in the human's examples, recovered from nothing (Phase 10) |
| task template and slot vocabularies | **induced** (Phase 10) — from 32 example sentences, and it costs: 3/7 pre-registered, 5/7 repaired, against a hand-built 6/7 |
| detecting that the examples disagree with each other | **not achieved** (Phase 10) — two tokenizer-odd lines in thirty-two broke the task and nothing said so; the post-hoc repair turns that into a drop or a refusal |
| corruption content | **supplied**, and now also **proposable** (Phase 10) — one counterfactual per induced slot, including a re-derivation of Phase 8's authored `xx_mismatch`. Phase 7 shows the content decides *which heads exist*, not just how well they score; Phase 8 forces **several** to be supplied and disagree in public |
| **which counterfactual to trust** | **not achieved** (Phase 10) — a 5/7 scheme was in the proposed set on both fixtures and the answer-key-free ranking took a 3/7 one both times |
| corruption position | searchable (Phase 4) |
| receiver input and position | searchable (Phases 4, 6); **not recovered in Phase 7**, for the same reason |
| answer key in the metric | **not needed** (Phases 5, 6, 7) |
| circuit components and wiring | discovered (Phases 1–3, 6); **partially** (Phase 7) |
| generality beyond one circuit | **tested** (Phase 6) — transfers to a second circuit in the same model |
| **generality beyond one model** | **tested** (Phase 7) — the *code* transfers unchanged; **two method assumptions did not**, and recovery degraded |
| **knowing what the experiment cannot see** | **partially surfaced** (Phase 8) — the pipeline flags counterfactual-dependent heads with no answer key |
| **knowing whether a flag matters** | **not achieved** (Phase 9) — a per-scheme null floor makes the criterion defensible and does **not** separate a real blind spot from ordinary disagreement; flag precision runs 50 % / 0 % / 26 % across three circuits |

**The line has moved up by exactly one row, and it is now the first row rather than the second.** Until Phase 10 the top two rows were both untouched and both looked like the same problem. They are not. *Constructing* a task around a stated behaviour turns out to be substantially mechanizable — the template, the vocabularies, the positions, the counterfactual content and the metric all come out of example sentences, and the resulting task locates the whole published circuit at the inherited cutoff once one repair is applied. *Choosing* the behaviour is untouched, and a next phase attacking it would still need a validation strategy that does not exist here, because a published circuit cannot check behaviour selection when the published behaviour *is* what was supplied.

What Phase 10 also did was split one old row into two and add a row nobody had noticed was missing. "Corruption content is supplied" is now half-answered — schemes can be *proposed* from the task's own structure — while **which of the proposed schemes to believe** is a new row reading *not achieved*, and it is the single largest thing standing between this and an unsupervised run. What Phase 7 changed is that "corruption content is supplied" used to be a statement about *precision* and is now a statement about *what the method can see at all*. What **Phase 8** changed is the blind-spot row, which did not exist before it: detecting that blindness moved from something a human catches after the fact to something the pipeline prints unprompted. **Phase 9 tried to add the row below it and failed**, which is why that row reads *not achieved* rather than *partial* — the warning the pipeline prints still cannot be graded without the answer key it is supposed to stand in for.

## Goal

Most interpretability tooling traces and visualizes: it shows which components light up when a model does something. That produces suggestive pictures, but not claims you can be wrong about.

This project aims at the stronger thing. The system forms hypotheses about the mechanism behind a behaviour, then tests them by intervening on the model and observing whether the behaviour actually changes:

- **Activation patching** — splice activations from a clean run into a corrupted one to isolate which nodes carry the causal signal.
- **Path patching** — hold every other route shut, so a node's effect on one specific downstream node can be measured on its own.
- **Iterative circuit pruning** — strip away components that carry no causal weight, until what remains is a minimal subgraph that still reproduces the behaviour.

The output is a **falsifiable causal claim** — "this circuit implements this behaviour, and here is the intervention that would break it" — not an attention heatmap. The loop from hypothesis to experiment to revised hypothesis is meant to run autonomously.

## Long-term aim

The eventual target is scalable oversight: applying this to models *more capable than the people and systems investigating them*, where no human has the ground truth to check the answer against.

That ambition is exactly why the work does not start there. Pointing an unvalidated system at an unfamiliar model produces conclusions nobody can check — the system could be confidently wrong and there would be no way to tell.

So every phase so far runs against a small open model with an **already-published circuit**, where the question is narrow and concrete:

> Do the system's conclusions match the published ground truth?

Only once the method demonstrably rediscovers what is already known does it earn the right to be pointed at anything unfamiliar. Six phases have now failed a prediction they made about themselves — Phase 7 failed three of seven, Phase 8 two of eight, Phase 9 four of eight, **Phase 10 seven of eight** — which is the argument for keeping the answer key in reach a while longer. Phase 7 is also the sharpest evidence for the policy itself: it produced a clean, plausible, internally consistent circuit claim that was **missing half the mechanism**, and only the published head list revealed that.

---

## Phase 1 — activation patching

**Partial reproduction. Not a pass.** Full numbers: **[results/PHASE1_REPORT.md](results/PHASE1_REPORT.md)**.

Every one of the 144 attention heads patched at each of 7 semantic token positions, over 128 prompts from 8 templates, under two corruption schemes.

| published class | acts on | recovered | |
|---|---|---|---|
| name mover | output directly | **3/3** | ✅ |
| negative name mover | output directly | **2/2** | ✅ |
| S-inhibition | output directly | **4/4** | ✅ |
| induction | other heads | 3/4 | ◐ |
| backup name mover | output, redundantly | 6/8 | ◐ |
| duplicate token | other heads | 2/3 | ◐ |
| previous token | other heads | **0/2** | ✗ |
| **total** | | **20/26** | |

The split is the whole result: **every class acting directly on the output logits is fully recovered, and every class that falls short acts indirectly.** Under the primary scheme at a slightly stricter cutoff, precision reaches **1.00** — the 18 highest-effect heads in the model are all published circuit members.

Narrowing to a circuit that works *jointly* rather than head-by-head, six nodes restore clean behaviour completely, and all six are published members:

```
5.5@S2 (induction) → 8.10@END → 7.9@END → 8.6@END → 7.3@END (S-inhibition) → 3.0@S2 (duplicate token)
```

**Why the misses happen** — three structural causes, not tuning failures: activation patching measures total effect where the paper used path patching; the primary corruption scheme is provably blind before the S2 token (**576/576** head-position cells there are exact floating-point zeros, measured not assumed); and marginal single-head patching is the condition under which redundant backup heads look least important.

## Phase 2 — path patching

**The head count did not move. The mechanism and the precision did.** Full numbers: **[results/PHASE2_REPORT.md](results/PHASE2_REPORT.md)**.

Phase 1 predicted path patching would recover the classes it missed. Tested directly, that prediction was **wrong**: 19/26 alone, and combined with Phase 1 still **20/26** — the same six heads missing. Reported as a failed prediction rather than folded away.

Two things did improve, neither visible in a head count.

**Precision**, from pinning every other head to its corrupted value so only the chosen route carries signal:

| | discovered | in circuit | precision |
|---|---|---|---|
| Phase 1 | 28 | 20 | 0.71 |
| Phase 2 | 21 | 19 | **0.90** |
| Phase 2, `abc` scheme alone | 14 | 14 | **1.00** |

**The wiring, not just the parts.** Each round's receivers are the heads discovered in the round before — the answer key is never consulted to choose them — and the chain recovered the paper's causal order unprompted:

| round | question | top senders found | published class |
|---|---|---|---|
| 0 | what moves the logits directly? | 9.9, 10.7, 9.6, 11.10 | name mover / negative name mover |
| 1 | what feeds *those* heads' queries at END? | **8.6, 8.10, 7.9, 7.3** | S-inhibition — all four |
| 2 | what feeds *those* heads' values at S2? | **5.5, 3.0, 6.9, 5.9** | induction + duplicate token |

Round 1 returned all four published S-inhibition heads as its top four, from a sweep of all 144.

**Previous token heads: measured, not dropped.** A dedicated probe — receivers from the `s2_swap` chain, measured on `abc`, run at every achievable sender ceiling — gives:

| head | effect on logits | signal delivered to receiver |
|---|---|---|
| 2.2 | +0.0003 | **+0.197** |
| 4.11 | +0.0003 | **+0.361** |

The two measurements disagree, and the disagreement is the finding: the path is there and carries a fifth to a third of the receiver's entire clean-vs-corrupted difference, while moving the output by essentially nothing. **They are still counted as misses in every table here** — adopting the more favourable metric after seeing it scores better is how a validation exercise stops validating anything.

## Phase 3 — a pre-registered receiver-side criterion

**Two definitions of "found", side by side. Not merged, not ranked.** Full numbers: **[results/PHASE3_REPORT.md](results/PHASE3_REPORT.md)**.

Phase 2 named the receiver-side measure as the obvious next step — and warned it had been *observed* to score the missing heads well, which is what makes adopting it dangerous. So the rule, not the number, was committed first:

> threshold = 99th percentile of `|path_signal|` under a shuffled-source null, rounded up to two significant figures

The null runs the identical procedure but draws the sender's clean value from a *different prompt*: a real activation with its prompt-correspondence destroyed, so any surviving projection is what the method manufactures from nothing. That fixes the false-positive rate at ~1% in advance.

It produced **0.11**. The null is heavy-tailed — median 0.0006, 99th percentile 0.105 — so inheriting Phase 1's 0.02 would have carried a large false-positive rate here.

The pre-registration is committed in [`b039915`](../../commit/b039915), containing the threshold and all the code and **no results**. The ordering is checkable in git history rather than asserted. **It is not a blind pre-registration**: Phase 2 published real `path_signal` values for the previous-token heads before this rule was written. The narrower claim is that the number came from a fixed rule rather than selection, and was not adjusted afterwards.

| published class | logit (all rounds) | logit (rounds 1+) | receiver-side (≥ 0.11) |
|---|---|---|---|
| name mover | 3/3 | 0/3 | 0/3 |
| backup name mover | 6/8 | 0/8 | 0/8 |
| negative name mover | 2/2 | 0/2 | 0/2 |
| S-inhibition | 4/4 | 4/4 | 3/4 |
| induction | 3/4 | 3/4 | 1/4 |
| duplicate token | 1/3 | 1/3 | 1/3 |
| previous token | **0/2** | **0/2** | **2/2** |
| **total** | 19/26 | 8/26 | 7/26 |
| precision | 0.90 | 0.80 | **0.64** |

The middle column is like-for-like. Round 0 is out of the receiver-side criterion's scope by construction: its receiver *is* the output, where the two measures are the same quantity.

Of the six heads neither earlier phase found, it recovers **two — 2.2 and 4.11, both previous-token heads**. It does not rescue `0.10` (+0.042) or `5.8` (+0.008); `9.0`/`11.9` are outside its scope entirely — unmeasured, not measured-and-failed. **The criterion is noisier**: precision 0.64 against 0.90.

**Robustness.** Per-group thresholds were fixed by the same rule at the same time. Only `3.7` and `4.3` depend on the more lenient pooled bar, and neither is a published head — pooling produced false positives and none of the recoveries.

### Why the scores are not added together

The two criteria disagree about *which* heads, not merely how many — previous-token heads appear only in the second, several induction and name-mover heads only in the first. Merging would report a larger number while destroying the only new information the phase produced.

A head can deliver its content to the next stage of the circuit and still leave the prediction unmoved. The criteria take opposite views on whether that counts, and neither is wrong: explaining a behaviour argues for the output criterion, mapping a mechanism argues for the receiver-side one. **Phases 1 and 2 answered only the first while appearing to answer both.**

### The boundary this project has not crossed

Every round in Phases 2 and 3 was told *where to look* — that S-inhibition acts on name movers' queries, that duplicate-token information arrives as a value at S2, that induction keys live at S1+1. Those come from the paper's account of the mechanism. Which heads turned up was never constrained; the question asked of them was.

So everything up to here is **guided rediscovery**: given the right question, the method finds the right components, in the right causal order. The autonomous loop described above has to generate the questions too. On a circuit nobody has published there is no paper to supply the receiver inputs, so a method that needs them supplied does not yet transfer — and unlike every phase so far, there would be no answer key to check the search against.

That is the boundary Phase 4 set out to cross.

## Phase 4 — searching for receiver specifications

**The search does not need to be told where to look.** Full numbers: **[results/PHASE4_REPORT.md](results/PHASE4_REPORT.md)**; the space and budget were fixed in **[results/PHASE4_SEARCH_SPACE.md](results/PHASE4_SEARCH_SPACE.md)**, committed before the search code was written.

Every receiver specification `(layer, head, input, position)` is scored by splicing that one input, at that one position, from the clean run into the corrupted one — one forward pass each, so the grid is searched exhaustively. `causal_interp/search.py` does not import `ground_truth`, and the run asserts that before starting.

Two screens were run, and only the second tests autonomy:

- **semantic** — 3024 specs over positions labelled IO / S1 / S2 / END. Comparable with earlier phases, but those labels already encode which name is the indirect object.
- **absolute** — 6912 specs over bare token indices `t0…t15`, single template, single name order, no semantic labels at all.

### Does it find the published specification?

| outcome | count | |
|---|---|---|
| agreement — published spec ranks first | **16** | ✅ |
| ambiguous | 0 | |
| unmeasurable under this corruption scheme | 4 | ⊘ |
| genuine disagreement | 1 | ✗ |

Of the **17 specifications the search could actually weigh, it recovered 16.** All four S-inhibition heads returned `v@S2`, and every name mover, backup and negative name mover returned `q@END` — the paper's answer, arrived at without the paper.

**The 4 "unmeasurable" are not failures of the search.** For the induction heads the published spec is `k@S1+1`, and under `s2_swap` *all 432 specifications at S1+1 score exactly zero* — the position is bit-identical between the clean and corrupted runs. The search never weighed `k@S1+1` and preferred something else; it was handed a counterfactual that cannot see that position. This is Phase 1's structural blindness (576/576 exact zeros before S2) arriving again in a new guise. Counting them as search failures would blame the search for a defect belonging to the corruption scheme.

### The position labels turned out to be unnecessary

The unlabelled search concentrates on the same two token indices the labelled search uses:

| search | top positions in its own top 50 |
|---|---|
| semantic | `S2` ×32, `END` ×16, `S2+1` ×2 |
| absolute | `t11` ×33, `t15` ×15, `t12` ×2 |

`t11` turns out to be S2 and `t15` turns out to be END — labels attached *after* the search, purely to read its output. Given only bare indices, the search independently located the two positions where the task's information lives.

### Stage B: the wiring falls out again

Sweeping senders into the specifications the *search* chose — not ones supplied — reproduces the paper's structure a second time:

```
9.9.q@END  ←  8.6 (+0.41), 8.10 (+0.25), 7.9 (+0.20)   — S-inhibition into name mover queries
8.6.v@S2   ←  5.5 (+0.39), 6.9 (+0.11), 3.0 (+0.11)    — induction + duplicate token into S-inhibition values
```

### Is autonomous discovery realistic with this approach?

**Partly, and the remaining gap is now a different one.** Phase 4 closed the receiver-specification gap: that part of the mechanism no longer has to be supplied. Three things still stand between this and autonomy, and they are named plainly in the report:

1. **The screen is a logit-effect screen**, so it inherits the exact blind spot Phase 3 documented — links that carry signal without moving the output are invisible to it by construction.
2. **The task, the counterfactual and the metric are still hand-built** from knowledge of what the model does. This is now the largest gap, and it is larger than the one this phase closed.
3. **There is no answer key on an unfamiliar circuit.** The search emits a ranking either way; nothing in the ranking distinguishes the case where it is right from the case where it is wrong. No published specification landed in the ambiguous band here, which is a better outcome than the alternative — but that is a property of this task at this sample size, not a guarantee, and the method still has no calibrated notion of when its own ranking is uninformative.

## Phase 5 — scoping what is still hand-built

**Diagnostic, not a build.** Full numbers: **[results/PHASE5_REPORT.md](results/PHASE5_REPORT.md)**; what each remaining piece encodes is itemised in **[results/PHASE5_AUDIT.md](results/PHASE5_AUDIT.md)**, written and committed before any experiment ran.

Phase 4 named three things still built by hand — the task, the corruption schemes, and the metric. This phase measures how much of the project's result actually depended on them, using IOI because the answer is still known and can check the substitution.

### The metric does not need the answer key

`logit_diff` requires knowing which two tokens are the candidates and which is correct. Replacing it with a divergence over the *whole* next-token distribution needs neither — only the two runs the corruption already provides. Size-matched to the published circuit's 26 heads:

| metric | knows the answer? | recovered |
|---|---|---|
| logit difference | yes | 18/26 |
| KL divergence | **no** | **19/26** |
| total variation | **no** | **19/26** |

Per-head rankings from the hand-built metric and KL correlate at **+0.98**. The piece of knowledge that looked most load-bearing turns out not to be needed to locate the circuit.

### Removing the corruption's knowledge *too* is what costs

Generic corruptions substitute a uniformly drawn vocabulary token instead of a semantically chosen name. The result inverted my expectation:

| what is supplied | recovered |
|---|---|
| hand-built corruption + hand-built metric | 18/26 |
| hand-built corruption + general metric | **19/26** |
| generic corruption + hand-built metric | **18/26** |
| generic corruption + general metric | 16/26 |
| nothing supplied at all | **13/26** |

**The two pieces are not independently load-bearing and not additive.** Either one alone carries enough task knowledge to locate the circuit; what fails is removing both. A corruption that damages the prompt arbitrarily still gives usable signal *if the metric knows what to look at* — and a metric that reads everything still works *if the corruption was aimed at the right thing*.

The mechanism: `s2_swap` was built to *reverse* the behaviour, so the corrupted run sits as far from clean as the task allows. A random token merely damages the prompt, the corrupted run still partly performs the task, and the measurable span collapses. A two-token metric still points that shrunken span at the right quantity; a distribution-wide metric shares it out over every irrelevant way the prompts differ.

The negative result stands in the form that matters: **fully generic recovers 13/26 against 18/26**, and on an unfamiliar circuit there would be no published answer to notice that degradation against.

### Task construction: not attempted, and not on budget grounds

Choosing which behaviour to study is a different kind of problem from anything in Phases 1–5. A weak version was available — sweeping templates, or mining a corpus for predictable completions — and was deliberately not built, because it would have produced a section in the report and no evidence that the resulting tasks isolate anything mechanistically interesting.

## Phase 6 — a second published circuit

**The pipeline was not fitted to IOI.** Full numbers: **[results/PHASE6_REPORT.md](results/PHASE6_REPORT.md)**; the target, ground truth, scoring rules and four predictions were fixed in **[results/PHASE6_PLAN.md](results/PHASE6_PLAN.md)**, committed before any Phase 6 code existed.

Every threshold, criterion and design choice in Phases 1–5 was made while looking at one answer key. That is the standing reason to distrust all of it: nothing so far separated *a method that works* from *a method fitted to IOI*. So the whole pipeline was pointed at the **greater-than circuit** (Hanna, Liu, Variengien 2023, [*How does GPT-2 compute greater-than?*](https://arxiv.org/abs/2305.00586)) — 7 attention heads plus MLPs 8–11 — with no retuning.

Chosen over induction heads because its published ground truth is as specific as IOI's: the paper names seven heads individually, and the authors' code release lists exactly those seven.

| measurement | greater-than | IOI, same measurement |
|---|---|---|
| activation patching, hand-built metric | **7/7** | 18/26 |
| activation patching, KL (no answer key) | **7/7** | 19/26 |
| path patching, all rounds | **7/7** | 19/26 |
| receiver specifications recovered by search | **7/7** | 16/17 scoreable |

**Recovery is better than IOI's, not worse** — which is the opposite of the failure this phase was built to detect, and worth deflating: seven targets is an easier set than twenty-six, and this circuit has no analogue of IOI's previous-token heads, the class activation patching structurally cannot see.

### What transferred, which is the real measure

| module | status |
|---|---|
| `interventions.py`, `search.py`, `metrics.py`, `model.py` | **untouched** — the causal core ran as-is |
| `comparison.py` | one added `circuit` parameter, defaulting to IOI |
| `ioi.py` | generic-corruption body moved to a shared module, call site left |
| `greater_than.py`, `ground_truth_greater_than.py` | new — the task and the answer key |

No existing call site was edited. Backward compatibility is verified rather than claimed: `check_patching.py` passes unchanged, IOI's corrupted token tensors hash identically under all four schemes before and after the corruption extraction, and Phase 1's stored headline is reproduced exactly by the modified `comparison.py`.

### Three results that complicate the headline

Reported because they cut against it, not despite that:

- **The two circuits are not disjoint.** `5.5` and `6.9` belong to both published circuits, and Phase 1 had already found both. Five of the seven are genuinely new to this project.
- **The receiver-side criterion found none of the in-scope published heads.** Only 2 of the 7 were ever eligible as senders — the rest occupy the *receiver* slot and are unmeasured, not measured-and-failed. It did surface `0.1` and `0.3`, which the paper's appendix names as the circuit's upstream dependencies, and which the chain was never told about.
- **Phase 3's threshold could not be inherited as a number.** Its *rule* was, and recalibrating the null gave **0.046** against IOI's 0.11 — so reusing the number would have been far too strict here, erring in the opposite direction from the one Phase 3 warned about.

One prediction was scored a **tie rather than a hit**: precision was predicted to degrade, and came out 0.778 against IOI's 0.783, a gap of 0.005 that confirms nothing.

### What it still does not show

Both circuits live in **GPT-2 small**. This tests generality across tasks and circuits, not across models. And two circuits is two — the honest reading is that one specific failure mode was tested for and not found, not that the method transfers to circuits unlike both.

That is what Phase 7 set out to test.

## Phase 7 — a different model

**The code transferred. The results got worse.** Full numbers: **[results/PHASE7_REPORT.md](results/PHASE7_REPORT.md)**; the target, ground truth, thresholds, an advance audit of what in the code was GPT-2-shaped, and seven predictions were fixed in **[results/PHASE7_PLAN.md](results/PHASE7_PLAN.md)**, committed before any Phase 7 code existed.

Every number in Phases 1–6 came from one 12-layer, 12-head, MLP-bearing model with one tokenizer. So the pipeline was pointed at the **Python docstring circuit** (Heimersheim & Janiak 2023, [*A circuit for Python docstrings in a 4-layer attention-only transformer*](https://www.lesswrong.com/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only)) in **`attn-only-4l`** — 4 layers × 8 heads, `d_model` 512, a different tokenizer, a different corpus, and **no MLP blocks at all**. Ground truth: 6 heads named in the post and recurring as a literal 37-edge graph in the ACDC benchmark's released code.

GPT-2 medium was the obvious scale test and was **rejected**: its published IOI circuit ([Merullo et al. 2024](https://arxiv.org/abs/2310.08744)) is defined by "the 2% most important heads" rather than a named list, and the authors' code release contains no ground-truth head list — building one would have meant choosing which heads count. Pythia was rejected for having no head-level published circuit at all. Both rejections are for lack of an answer key, and both are recorded in the plan.

### The code transferred completely

| | |
|---|---|
| pre-existing modules changed | **none** — the report runs `git diff --stat` and pastes the empty output |
| `interventions.py`, `search.py`, `metrics.py`, `comparison.py`, `model.py`, `corruption.py` | imported and called as-is |
| new | the task, the answer key, a runner, a report, and known-answer tests for the new model |

`load("attn-only-4l")` was the entire model change. `comparison.py`'s `circuit` parameter — added in Phase 6 — took a third answer key without modification.

### The results are the worst of the three circuits

| circuit | model | published | recovered | recall | chance recall |
|---|---|---|---|---|---|
| IOI | GPT-2 small | 26 | 18/26 | 69% | 18% |
| greater-than | GPT-2 small | 7 | 7/7 | 100% | 5% |
| **docstring** | **`attn-only-4l`** | **6** | **3/6** | **50%** | **19%** |

The plan pre-registered a deflation — 6 heads among 32 is an easier denominator — and it turned out to cut **less** than expected: chance recall here is 19% against IOI's 18%. Against IOI the denominators are effectively identical, so the drop from 69% to 50% is real and not an artifact of circuit size.

### One mechanism explains almost all of it

The circuit has two jobs: **route** attention to the right definition argument (heads `1.4`, `2.0`) and **move** whatever is attended to (`0.5`, `3.0`, `3.6`). The three missed heads are routing heads.

| published counterfactual | what it breaks | recovered |
|---|---|---|
| `random_random` (the benchmark default, **pre-registered as primary**) | replaces the answer token | **3/6** |
| `random_answer` | replaces the answer token | 3/6 |
| `random_def` | breaks the pointer, keeps the answer | **5/6** — both routing heads appear |

When the counterfactual replaces the answer itself, restoring a routing head buys nothing: the movers point at the right position and the token sitting there is still wrong. **The same failure repeats one level down**: the receiver-spec search recovered **0 of 4** primary specifications, because for both argument movers it ranks `v@C_def` above the pre-registered `q@END` — the wire carrying the answer above the wire choosing it. `v@C_def` is itself a published input, and it was fixed in the plan as an alternative *before* the run, which is the only reason that can be reported without it being a rescue.

### Two method assumptions were GPT-2 small assumptions

Neither is in the causal core, and neither shows up in a diff:

- **The component sweep patches `mlp_out` on a model with no MLPs.** Nothing raises. The hook exists, never fires, and the sweep returns **28 exact zeros** that read as "the MLPs carry no causal signal". Predicted in advance and run deliberately to demonstrate it.
- **The iterative path chain assumes there is another layer below.** It descended to layer 0 in one round and halted, leaving the pre-registered four-round ladder untestable and the receiver-side null pooled from **16** measurements instead of hundreds. A 99th-percentile rule does not survive contact with a shallow model.

### Does this support or complicate the long-term aim?

**Both, and the complication is the more useful half.** It supports the aim in that the machinery is demonstrably not architecture-specific. It complicates it in the way that matters for oversight: on an unfamiliar model there would be no published head list to notice a 50% recall against, and every diagnosis above was made by consulting one. Nothing in the pipeline's own output separates *"the circuit is these three heads"* from *"this experiment can only see three of them"* — and both of the assumptions that broke produced plausible-looking output rather than errors.

Scale remains untested in the direction that matters: `attn-only-4l` is *smaller* than GPT-2 small, and testing upward is waiting on a larger model with a published circuit specific enough to check against.

The half of that finding Phase 8 could attack is the sentence about the pipeline's own output.

## Phase 8 — counterfactual disagreement as standard output

**The method changed, not the recall.** Full numbers: **[results/PHASE8_REPORT.md](results/PHASE8_REPORT.md)**; the registry design, the authored greater-than scheme, the decision rules and eight predictions were fixed in **[results/PHASE8_PLAN.md](results/PHASE8_PLAN.md)**, committed before any Phase 8 code existed.

Phase 7's diagnosis — that the primary counterfactual hides the routing heads — was correct, pre-registered, and *run as a one-off because a human saw a low recall number*. Phase 8 makes it structural:

- a task registers **named** counterfactual schemes, each declaring what it breaks and whether the answer survives it, and `TaskSpec` **raises** if it registers fewer than two;
- `pipeline.discover()` sweeps every registered scheme and returns the head list and the cross-scheme comparison in the same object — there is no single-scheme entry point;
- `agreement.py` emits per-head verdicts (`robust` / `scheme-dependent`), a blind spot for **every** scheme rather than only the primary, and one flag: *heads some other counterfactual finds and the primary does not*. It is a bare non-emptiness test, so the phase adds no free parameter, and it imports no answer key.

Both earlier circuits were re-run through it, with Phase 6's and Phase 7's own results left untouched. The primary scheme reproduces each earlier phase's head set exactly, checked against their committed files rather than asserted.

| circuit | flag | heads flagged | of which published | recall, primary → union | precision |
|---|---|---|---|---|---|
| docstring (Phase 7) | **fires** | 17 | **3** — `1.2`, `1.4`, `2.0` | **3/6 → 6/6** | 0.33 → 0.23 |
| greater-than (Phase 6) | **fires** | 16 | **0** | 7/7 → 7/7 | 0.78 → 0.28 |

**The structure catches Phase 7's blindness without the foreknowledge that produced it.** Of the 17 heads it flags, the three that turn out to be published circuit members are exactly the three Phase 7 missed — and the flag is emitted before the published circuit is read.

**And the same flag fires where there is nothing to find.** On greater-than the primary counterfactual was already recovering 7/7; the other schemes contribute 16 flagged heads, none of them circuit members, and union precision falls from 0.78 to 0.28. The two flags are identical in form. **Nothing the pipeline computes before the answer key opens separates "your counterfactual is hiding half the circuit" from "your counterfactual is fine and the others are noisier."**

Six of eight predictions held. Both misses are in the report as they came out: the authored greater-than counterfactual turned out **stronger** than predicted (power 0.74 against a predicted 0.05–0.60), and the receiver-spec argmax rule, run as written, flags 32 of 32 and 137 of 144 heads — a result about the rule, left unretuned.

### What it does not fix

The schemes are still **authored per task**. Docstring got its alternates free from the paper; greater-than's `xx_mismatch` was designed for this phase by a person reasoning about the task's mechanism, and is labelled `authored` rather than counted as published wherever it appears. The generic vocabulary-substitution schemes need no task knowledge and can be registered anywhere — and they are the lowest-power schemes here and the largest source of flagged heads in no published circuit. Blindness that *every* registered scheme shares is still reported as agreement.

Which leaves the flag itself unusable without a human. That is what Phase 9 went after.

## Phase 9 — trying to tell a real blind spot from noise

**A better criterion, and not the discriminator it was meant to be.** Full numbers: **[results/PHASE9_REPORT.md](results/PHASE9_REPORT.md)**. The order of this phase is its main methodological claim and is checkable in git: the measurements came first, in **[results/PHASE9_CHARACTERIZATION.md](results/PHASE9_CHARACTERIZATION.md)**; then the rule, the scoring table, the holdout and eight predictions, in **[results/PHASE9_PLAN.md](results/PHASE9_PLAN.md)**; then the runs.

### Measure first

Ten candidate signals were computed over all 33 heads Phase 8 flagged, from stored results, before anything was designed. **Nine do not separate the two cases.** Several separate in the wrong direction: any statistic normalized by a scheme's median or its strongest head is dominated by how many dead heads a model has, so GPT-2 small's noise outranks `attn-only-4l`'s real finds. Only raw effect magnitude showed visible separation, and only in the upper tail.

### The rule

Phase 3's rule on a new channel, per scheme: **θ(s) = the 99th percentile of |normalized recovery| under a shuffled-source null sweep**, replacing Phase 8's shared 0.02. The diagnosis is that normalized recovery divides by each scheme's own clean-vs-corrupted span, so one cutoff cannot mean the same thing under two counterfactuals. Nothing else changed, so the two runs differ in the criterion alone.

**That much is established.** The ten measured floors span a factor of 400 — from 0.0077 to 3.3. Docstring's `random_vocab_any` has a null whose 99th percentile is **3.3**: patching a head with an activation from a *different prompt* routinely moves that metric by several times the entire clean-to-corrupted span. It contributed 17 of docstring's flags under Phase 8's cutoff and contributes none under its own floor. Phase 8's shared cutoff was not defensible.

### And it is still not a discriminator

| circuit | known case | heads flagged | flag precision |
|---|---|---|---|
| docstring | real blind spot | 17 → **4** | 18 % → **50 %** |
| greater-than | no blind spot | 16 → **8** | 0 % → **0 %** |
| **IOI** (holdout) | real blind spot | 15 → **19** | 20 % → **26 %** |

IOI is a genuine holdout: Phase 8 registered its four schemes and deliberately never ran it, and its expected behaviour was fixed in the plan from Phase 1's finding that `s2_swap` is provably blind before S2 while `abc` is not. Its calibrated blind spot **grew**, because `abc` and `random_vocab_s2` came out with floors *below* 0.02 and became more sensitive while the primary's floor rose to 0.058 and cost it three published heads of its own (18/26 → 15/26). Calibration cuts both ways.

**The result, stated as the plan required if it came out this way:** a per-scheme null floor separates *this counterfactual was too weak to be believed* from *this counterfactual measured something*. It does **not** separate *measured something the primary is blind to* from *measured something outside the circuit* — both are sound measurements under a real counterfactual — and that second distinction is the one a reader needs. Four of eight predictions held.

n = 3: three circuits, two models, one architecture family, and the rule was chosen while looking at two of the three.

## Phase 10 — inducing the task instead of writing it

**Negative as pre-registered, and the failure comes apart into two separable halves.** Full numbers: **[results/PHASE10_REPORT.md](results/PHASE10_REPORT.md)**. Four documents, committed in this order and checkable in git: **[PHASE10_PLAN.md](results/PHASE10_PLAN.md)** with the question, the algorithm in full pseudocode, the scoring table and eight predictions — committed alongside the human input in **[`fixtures/`](fixtures/)** and before any code existed; then **[PHASE10_CHARACTERIZATION.md](results/PHASE10_CHARACTERIZATION.md)**, measuring what that algorithm builds, before any repair was designed; then **[PHASE10_AMENDMENT.md](results/PHASE10_AMENDMENT.md)**, holding one repair and an explicit refusal to let it become the headline; then the runs.

### The question, narrowed on purpose

Phase 5 put task construction out of scope because a weak version "would produce something that looked like progress without being any". This phase does not attempt autonomous task *discovery* either. It attempts the rung below:

> Given a model, a one-sentence hunch, and example prompts written by a person who has only that hunch — how much of the rest of task construction can be mechanized?

The hunch was *this model seems to know that the end of a date range comes after its start*. It produced 64 lines across two sentence frames, committed unfiltered — not checked against the tokenizer, against this repo's word lists, or against whether the model performs the behaviour on them, because that filtering is itself one of the things being mechanized.

### What the induction does

Constant token columns are the frame; varying ones are slots; columns that hold the same token in **every** example are one slot appearing twice; the observed values are the vocabulary; the varying columns are the position vocabulary; one counterfactual is proposed per slot, plus one per tied column. The metric is `clean_argmax_logprob` — the log-probability of whatever the model itself predicted on the clean prompt, which needs no answer key. The primary scheme is chosen by measured output divergence.

All of it reaches `pipeline.discover()` as an ordinary `TaskSpec`. **No pre-existing module was changed**, the same measure Phases 6 and 7 used.

### The headline, and the two halves of it

| | size-matched | at the 0.02 cutoff | precision | is the generated task actually greater-than? |
|---|---|---|---|---|
| hand-built (Phase 6) | **6/7** | 7/7 of 9 discovered | 0.78 | by construction |
| induced, **pre-registered** | **3/7** | 6/7 of 27 | 0.22 | **48 %** |
| induced, post-hoc repair | 5/7 | **7/7** of 14 | 0.50 | 100 % |

**Half one — the clean condition was not the task.** Two of the thirty-two lines contain a year GPT-2 splits as `[" 150", "9"]` rather than `[" 15", "09"]`. Same row length, so the pre-registered filter keeps them; the tie rule needs unanimity, so the two century columns stop being one slot; generation then samples them independently and produces `The pilgrimage lasted from the year 11245 to the year 14`. The model's top prediction exceeds the start year on 48 % of the clean prompts — a coin flip, which is the published task's own `xx_mismatch` counterfactual served as the control. **Nothing in the pipeline said so.**

**Half two — the ranking chose badly, and this half is not a bug.** `resample_t8`, which redraws the start year and therefore acts on exactly the position the published `yy01` counterfactual acts on, scores **5/7 at precision 0.50** and was sitting in the proposed set. The answer-key-free rule picked a different scheme on both fixtures, each time on larger output divergence. The counterfactual *content* mechanized; the counterfactual *ranking* did not, and this phase does not offer a better rule — writing one after seeing which scheme it should have chosen is fitting to the answer key.

### Two results that ran hard against expectation

**The k-curve is inverted.** Two example prompts recover 6/7 and thirty-two recover 3/7, monotonically. Every additional natural example is another chance to poison a unanimity-based rule. A seven-pair robustness check says it is not a fluke: five of seven contiguous line-pairs reach 5–6/7, and the two that do not are exactly the two containing a tokenizer-odd line — which under the repair **refuse to build** rather than building something broken. The caveat is real: a two-line task generates at most 8 distinct prompts.

**The two sentence frames are indistinguishable.** 3/7 against 3/7 pre-registered, 5/7 against 5/7 repaired. Prediction P8 said the unfamiliar frame would do worse and it did not.

### The repair, and why it is not the headline

One change, fixed in the amendment before it ran: keep the largest group of examples sharing a column **shape** rather than a token **length** — a strict generalization with no threshold in it. It drops exactly the two odd lines, the tie returns, `desync_t7` and `desync_END` are proposed — **Phase 8's authored `xx_mismatch`, re-derived from example sentences with nothing task-specific in the code** — `resample_t7`'s divergence collapses from 0.755 to 0.084 because both centuries now move together, and the rule picks the right scheme on its own.

Section 7 of the plan said a negative result is not retried with a different induction rule. It is not: the headline stays 3/7, and the repair is reported beside it, labelled post-hoc, answering one question only — whether the failure was the mechanism or the bug. All three of its hindsight-informed predictions held, which is what hindsight-informed predictions do.

### Run E — the induction on the other two tasks

Weaker by construction, since these prompts come from hand-built generators. **On docstring — a different model, a different tokenizer, a task this code was never pointed at — the induction finds both tied argument-name slots**, which are the `A_def`/`A_doc` and `B_def`/`B_doc` pairs `docstring.py` hand-codes as separate named positions. **On IOI it fails outright and does not notice**: the eight published templates have different token lengths, the filter keeps the plurality one, and it reports a confident structure over 12 of 32 examples with one slot whose values are a comma and four names.

### What it does not establish

Rediscovery on a published circuit validates task *construction*, never task *invention* — the case that matters for oversight has no published anything, and no version of this experiment covers it. n = 1 circuit, 1 model, 2 frames. And `clean_argmax_logprob` has no guaranteed positive span: it did not bite here, and the known-answer suite contains a synthetic frame where all four induced schemes come out negative.

---

## Stack

- **Python** 3.12
- **[transformer_lens](https://github.com/TransformerLensOrg/TransformerLens)** — hooked model internals, activation caching, intervention hooks
- **PyTorch** with CUDA (see setup — a CPU-only build will work but is painfully slow)

## Setup

Requires Python 3.12 and, for GPU acceleration, an NVIDIA GPU with a recent driver.

```bash
git clone https://github.com/ranveerlabs/causal-interp.git
cd causal-interp

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### CUDA note

`requirements.txt` pulls PyTorch from the **CUDA 12.8** wheel index. This matters: the default PyPI `torch` on Windows is a CPU-only build, and newer NVIDIA cards (Blackwell, `sm_120` — the RTX 50-series) have no compiled kernels in pre-12.8 builds, so they fall back to CPU silently.

If `pip install -r requirements.txt` does not pick up the index URL, install torch explicitly first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then verify — this checks for a CUDA build, a visible GPU, matching kernels for its compute capability, and a real matmul on the device:

```bash
python scripts/check_env.py
```

Expected tail of output:

```
ENVIRONMENT OK
```

If it reports `CPU-ONLY BUILD` or `no kernels for sm_XX`, the GPU is not being used regardless of what `nvidia-smi` shows.

## Running the phases

Patching is easy to get subtly wrong in ways that still produce reasonable-looking numbers, so verify the machinery first. These are known-answer tests — cases where the correct result follows from how the experiment is built, not from the model:

```bash
python scripts/check_patching.py             # GPT-2 small + IOI      — expect: PATCHING OK
python scripts/check_patching_docstring.py   # attn-only-4l + docstring — expect: PATCHING OK
python scripts/check_schemes.py              # Phase 8's registry + Phase 9's null — expect: SCHEMES OK
python scripts/check_induction.py            # Phase 10's induction and auto-task — expect: INDUCTION OK
```

`check_induction.py` runs 46 checks against a synthetic frame — `Then {name} went {place} and {name} slept` — whose right answers are readable off the template string: the repeated slot ties across two columns, an over-long example is dropped and counted, a non-canonical token pair is rejected, every scheme sees a bit-identical clean sample, each counterfactual changes exactly the columns it declares, and `clean_argmax_logprob` equals the model's own clean maximum log-probability per prompt exactly.

The second is Phase 7's counterpart, not a replacement: two of its checks only exist on an attention-only model. With no MLPs, patching every head at every position must reproduce the clean run at the final token *exactly* (it does, to `1e-5`, against `0.05` on GPT-2 small), and patching `hook_mlp_out` must be a silent no-op rather than an error (it is, exactly `0.0` at every layer).

Then the pipeline. It downloads GPT-2 small on first run; timings are for a laptop RTX 5060:

```bash
python scripts/run_phase1_ioi.py                        # ~6 min, activation patching
python scripts/run_phase2_paths.py                      # ~4 min, path patching
python scripts/run_phase3_receiver.py --preregister     # ~2 min, fix the threshold
python scripts/run_phase3_receiver.py                   # ~2 min, apply it
python scripts/run_phase4_search.py                     # ~31 min, receiver-spec search
python scripts/run_phase5_scoping.py                    # ~9 min, metric/corruption scoping
```

Phase 6 targets a different circuit and runs in three stages, for the same reason Phase 3 splits its pre-registration out — the recalibrated threshold has to be on record before the comparison it will be judged by:

```bash
python scripts/run_phase6_greater_than.py --stage sweep   # ~4 min, patching + path chain
python scripts/run_phase6_greater_than.py --preregister    # ~1 min, recalibrate the null
python scripts/run_phase6_greater_than.py                  # ~15 min, search + comparison
```

Phase 7 has the same three stages and downloads `attn-only-4l` on first run. It is much faster than Phase 6 despite sweeping five corruption schemes, because the model is a fifth the size:

```bash
python scripts/run_phase7_docstring.py --stage sweep      # ~2 min, patching + path chain
python scripts/run_phase7_docstring.py --preregister       # <1 min, recalibrate the null
python scripts/run_phase7_docstring.py                     # ~5 min, search + comparison
```

Phase 8 re-runs both of the earlier circuits under the multi-scheme structure, one command each. It downloads nothing new:

```bash
python scripts/run_phase8_multischeme.py --circuit docstring      # ~5 min, 5 schemes
python scripts/run_phase8_multischeme.py --circuit greater_than   # ~17 min, 4 schemes
python scripts/run_phase8_multischeme.py --report-only            # rebuild the report
```

Each sweeps every scheme its task registers, across activation patching, the path chain and the receiver-spec search, and prints the cross-scheme comparison before it opens any answer key.

Phase 9 recalibrates that comparison and tests the recalibration on a third circuit. The first two reuse Phase 8's committed sweeps and measure only the nulls; IOI has no stored multi-scheme run and measures both:

```bash
python scripts/phase9_characterize.py                             # <1 min, no GPU
python scripts/run_phase9_calibration.py --circuit docstring      # ~1 min
python scripts/run_phase9_calibration.py --circuit greater_than   # ~4 min
python scripts/run_phase9_calibration.py --circuit ioi            # ~18 min, the holdout
python scripts/run_phase9_calibration.py --report-only
```

Phase 10 induces its task from `fixtures/` rather than importing one. The characterization runs first and on its own, because it is what the amendment was designed against:

```bash
python scripts/phase10_characterize.py                                      # ~2 min
python scripts/run_phase10_autotask.py --fixture frame_same --induction plan   # ~4 min, THE HEADLINE
python scripts/run_phase10_autotask.py --fixture frame_same --induction shape  # ~5 min, post-hoc
python scripts/run_phase10_autotask.py --fixture frame_own  --induction plan   # ~5 min
python scripts/run_phase10_autotask.py --fixture frame_own  --induction shape  # ~6 min
python scripts/run_phase10_autotask.py --stage ksweep                          # ~9 min
python scripts/run_phase10_autotask.py --stage pairs                           # ~12 min
python scripts/phase10_report.py
```

`--induction plan` is the pre-registered algorithm and is what the phase reports; `--induction shape` is the amendment's repair and is labelled post-hoc in every table it appears in. Neither reads a `ground_truth` module before the `ANSWER KEY OPENS HERE` banner, and the runner asserts that `induction.py` and `autotask.py` do not import one at all.

Phase 4 is the longest single sweep — two exhaustive grids, 9,936 forward passes. Phases 4, 5, 6 and 7 all support `--report-only`, which rebuilds their reports from the stored results without repeating the sweep.

Each regenerates its own report, the JSON behind it, and per-head CSVs in `results/`. All runs are seeded, so they reproduce exactly. The phases chain — Phase 2 reads `phase1_results.json`, Phase 3 reads `phase2_results.json`, Phases 6 and 7's later stages read their earlier ones — so run them in order on a clean checkout.

Phases 3, 6 and 7 refuse to run without a recorded threshold, and reject one calibrated at a different `n` or seed. That is deliberate: the point of the pre-registration is that the threshold cannot be adjusted once results exist.

## Layout

```
causal_interp/
  model.py           # model loading, device selection — takes any transformer_lens name
  ioi.py             # IOI task: prompt pairs, corruption schemes, position indices
  greater_than.py    # greater-than task, same interface — Phase 6's second target
  docstring.py       # docstring task, same interface — Phase 7's target, a different model
  corruption.py      # task-agnostic generic corruption, shared by all three tasks
  interventions.py   # activation patching, path patching, sweeps, circuit narrowing
  schemes.py         # counterfactual registry — a task cannot register fewer than two
  pipeline.py        # discovery under every registered scheme; no single-scheme path
  agreement.py       # per-head cross-scheme verdicts and the blind-spot flag
  induction.py       # Phase 10: slots, ties and counterfactuals induced from example prompts
  autotask.py        # Phase 10: an induced structure wrapped as a TaskSpec the pipeline runs
  comparison.py      # scoring a discovered head set against a ground truth
  metrics.py         # answer-key-free recovery metrics (KL, total variation)
  search.py          # receiver-spec search — must never import a ground_truth module
  ground_truth.py                # the published IOI circuit — inert data
  ground_truth_greater_than.py   # the published greater-than circuit — inert data
  ground_truth_docstring.py      # the published docstring circuit — inert data
scripts/
  check_env.py            # environment + CUDA verification
  check_patching.py       # known-answer tests, GPT-2 small + IOI
  check_patching_docstring.py  # known-answer tests, attn-only-4l + docstring
  check_schemes.py        # known-answer tests, Phase 8's registry and agreement analysis
  check_induction.py      # known-answer tests, Phase 10's induction and auto-built task
  run_phase1_ioi.py       # Phase 1: activation patching -> results/
  run_phase2_paths.py     # Phase 2: iterative path patching -> results/
  run_phase3_receiver.py  # Phase 3: --preregister fixes the threshold; main run applies it
  phase3_analysis.py      # Phase 3 comparison + report, imported only by the main run
  run_phase4_search.py    # Phase 4: exhaustive receiver-spec search -> results/
  phase4_report.py        # Phase 4 report, kept out of the search module
  run_phase5_scoping.py   # Phase 5: metric and corruption scoping -> results/
  phase5_report.py        # Phase 5 report
  run_phase6_greater_than.py  # Phase 6: the whole pipeline on the second circuit
  phase6_report.py            # Phase 6 report
  run_phase7_docstring.py     # Phase 7: the whole pipeline on a different model
  phase7_report.py            # Phase 7 report
  run_phase8_multischeme.py   # Phase 8: both circuits under every registered scheme
  phase8_report.py            # Phase 8 report
  phase9_characterize.py      # Phase 9 step 1: the measurement that precedes the fix
  run_phase9_calibration.py   # Phase 9: per-scheme null floors, and the IOI holdout
  phase9_report.py            # Phase 9 report
  phase10_characterize.py     # Phase 10 step 1: what the pre-registered induction builds
  run_phase10_autotask.py     # Phase 10: the induced task, the k-sweep, the pair check
  phase10_report.py           # Phase 10 report
fixtures/
  greater_than_frame_same.txt  # 32 hand-written prompts, the published sentence frame
  greater_than_frame_own.txt   # 32 more, a frame written for Phase 10
  README.md                    # what counts as human input, and why it is unfiltered
results/
  PHASE1_REPORT.md … PHASE10_REPORT.md
  PHASE4_SEARCH_SPACE.md       # the space and budget, committed before the search code
  PHASE5_AUDIT.md              # what each hand-built piece encodes, committed before the tests
  PHASE6_PLAN.md               # target, ground truth and predictions, committed before the code
  PHASE7_PLAN.md               # the same, for a different model, plus an advance code audit
  PHASE8_PLAN.md               # the registry design and the authored scheme, before the code
  PHASE9_CHARACTERIZATION.md   # ten candidate signals, committed before the rule existed
  PHASE9_PLAN.md               # the calibration rule, the scoring table and the holdout
  PHASE10_PLAN.md              # the induction algorithm in full, committed with the fixtures
  PHASE10_CHARACTERIZATION.md  # what that algorithm builds, before any repair was designed
  PHASE10_AMENDMENT.md         # one repair, and the refusal to let it be the headline
  phase3_preregistration.json  # the threshold, committed before the results existed
  phase6_preregistration.json  # the recalibrated threshold, same rule, same ordering
  phase7_preregistration.json  # recalibrated a third time — 0.11, 0.046, 0.03
  phase1_results.json … phase7_results.json
  phase8_docstring.json / phase8_greater_than.json   # per-scheme effects + agreement
  phase9_*.json / phase9_*_calibration.csv           # null floors, before/after verdicts
  phase10_characterization.json                      # the induction's own output, unscored
  phase10_<fixture>_<induction>.json / _schemes.csv  # per-fixture runs, both inductions
  phase10_ksweep.json / phase10_pairs.json           # the k-curve and the pair robustness check
  head_effects_*.csv / component_effects_*.csv / path_effects_*.csv
  receiver_signals.csv / receiver_search_*.csv / phase5_metric_effects.csv
  phase6_*.csv / phase7_*.csv / phase8_*_agreement.csv
```

Five separations are deliberate, and each one makes a claim checkable rather than promised. The `ground_truth*` modules hold published circuits as inert data while `comparison.py` scores against whichever it is handed, with pure set arithmetic, so nothing measured in a run can influence what counts as a match. `phase3_analysis.py` is a separate module so the `--preregister` path cannot import it. `search.py` must never import a `ground_truth` module — Phases 4, 6 and 7 assert this at startup, because a search that can see the answer key is not a search. And the three circuits live in **separate** modules rather than one registry, so a run cannot accidentally be scored against their union. Phase 8 extends the second of those: `agreement.py`, `pipeline.py` and `schemes.py` must not import a `ground_truth` module either, because a disagreement flag computed with the answer key in reach would prove nothing about what the pipeline can see on its own. **Phase 10 extends it again to `induction.py` and `autotask.py`** — a task *built* with the published circuit in reach would prove nothing about what can be built without one — and adds the fifth separation: the human input lives in `fixtures/` as plain text rather than inside a Python module, so the human contribution to that phase can be counted rather than described.

## License

Apache 2.0 — see [LICENSE](LICENSE).
