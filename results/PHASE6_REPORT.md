# Phase 6 — the same pipeline, a second published circuit

**Target**: the greater-than circuit in GPT-2 small, from Hanna, Liu, Variengien
(2023), [*How does GPT-2 compute greater-than?*](https://arxiv.org/abs/2305.00586).
Published ground truth: **7 attention heads** in
2 classes, plus MLPs 8, 9, 10, 11.

The target, the ground truth, the scoring rules and four predictions were fixed in
[PHASE6_PLAN.md](PHASE6_PLAN.md), committed before any Phase 6 code existed.

| | |
|---|---|
| model | `gpt2` |
| prompts | 128 per corruption scheme, seed 0 |
| activation-patching cutoff | 0.02 — **inherited from Phase 1** |
| size-matched set | top 7 — the published head count |
| receiver-side threshold | 0.046 — **Phase 3's rule, recalibrated null** |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| runtime | 266s sweep + 889s search |

Every cutoff, chain width, confirmation depth and ambiguity margin was inherited
verbatim from the phase that introduced it. The only number recomputed is the
receiver-side threshold, whose rule is unchanged and whose value must be
recalibrated because the null is task-specific — the plan fixed that in advance.


## Headline

| method | recovered | precision |
|---|---|---|
| activation patching, hand-built metric | 7/7 | 0.78 |
| activation patching, KL metric | 7/7 | 0.37 |
| path patching, all rounds | 7/7 | 0.78 |

Recovery on this circuit is **higher than** IOI's: 100% of the published heads here against 69% of IOI's 26 under the same cutoff and the same metric.


## The four predictions, scored

Fixed in [PHASE6_PLAN.md](PHASE6_PLAN.md) before the run.

| prediction | outcome | measured |
|---|---|---|
| 1. blindness before `YY` reappears | ✅ held | exact zeros at `NOUN` 144/144, `XX1` 144/144; at `YY` itself 12/144 |
| 2. activation patching recovers ≥ 5 of 7 | ✅ held | recovered 7/7 |
| 3. precision worse than IOI's | ◐ **tie — not confirmed** | precision 0.778 against IOI's 0.783 under the same cutoff — a gap of 0.005, which is a tie, not a confirmation. Read as: precision did **not** degrade |
| 4. fully generic recovers less | ✅ held | size-matched 4/7 generic vs 6/7 hand-built (7 discovered at the cutoff) |


## Activation patching, every corruption against every metric

The Phase 5 grid, rerun on this task. One forward pass yields all three metrics, so
any difference between them is the metric and not the run. **Size-matched** is the
top 7 heads by absolute effect — no free parameter, so it
cannot be tuned; the IOI column beside it is the same measurement from Phase 5,
size-matched to that circuit's 26.

| corruption | metric | size-matched | at 0.02 | precision | discovered | IOI (Phase 5) |
|---|---|---|---|---|---|---|
| `yy01` | hand-built | 6/7 | 7/7 | 0.78 | 9 | 18/26 |
| `yy01` | KL divergence | 6/7 | 7/7 | 0.37 | 19 | 19/26 |
| `yy01` | total variation | 6/7 | 7/7 | 0.78 | 9 | 19/26 |
| `random_vocab_yy` | hand-built | 3/7 | 1/7 | 0.33 | 3 | 18/26 |
| `random_vocab_yy` | KL divergence | 4/7 | 7/7 | 0.18 | 38 | 15/26 |
| `random_vocab_yy` | total variation | 4/7 | 5/7 | 0.62 | 8 | 16/26 |
| `random_vocab_any` | hand-built | 1/7 | 1/7 | 0.10 | 10 | 17/26 |
| `random_vocab_any` | KL divergence | 4/7 | 7/7 | 0.25 | 28 | 13/26 |
| `random_vocab_any` | total variation | 3/7 | 6/7 | 0.50 | 12 | 12/26 |

The counterfactuals themselves:

| corruption | what it supplies | clean | corrupted | span | corrupted still valid |
|---|---|---|---|---|---|
| `yy01` | the task's published counterfactual | +0.822 | -0.556 | +1.377 | 11% |
| `random_vocab_yy` | generic substitution at the task's pivot | +0.822 | -0.012 | +0.834 | 22% |
| `random_vocab_any` | generic substitution, position drawn uniformly | +0.822 | +0.451 | +0.371 | 73% |


## The MLPs — the part of this circuit a head count cannot describe

IOI's published circuit is 26 attention heads and no MLPs. This one is
7 heads and **four MLPs**, and the paper puts the MLPs at
the centre: "MLPs 9, 10, and 11 appear to compute the greater-than operation in
tandem, and in steps". Reporting only a head count would omit the published claim
this circuit mostly consists of.

`sweep_component` already existed — Phase 1 used it to localize depth before
attributing effect to heads — and it needed no change to answer this.

| component | published? | largest effect | at position |
|---|---|---|---|
| MLP 0 |  | +0.862 | `YY` |
| MLP 1 |  | +0.065 | `YY` |
| MLP 2 |  | +0.034 | `YY` |
| MLP 3 |  | +0.021 | `YY` |
| MLP 4 |  | -0.009 | `END` |
| MLP 5 |  | -0.033 | `YY` |
| MLP 6 |  | +0.034 | `END` |
| MLP 7 |  | +0.028 | `END` |
| MLP 8 | **published** | +0.186 | `END` |
| MLP 9 | **published** | +0.486 | `END` |
| MLP 10 | **published** | +0.535 | `END` |
| MLP 11 | **published** | +0.131 | `END` |

**3 of the top 4 MLPs by absolute effect are published circuit members**
(MLP 0, MLP 8, MLP 9, MLP 10 recovered against a published
MLP 8, MLP 9, MLP 10, MLP 11).

The fourth is **MLP 0**, and it is not a false positive so much as a different
kind of object. Its effect is the largest of any component here and it sits at
`YY`, not `END` — the paper looks for what MLP 0 depends on and finds nothing
upstream of it, concluding "it depends primarily on the token embeddings". An
MLP 0 that behaves as an extended embedding of the year token *should* dominate a
patch at the year position, and reading that as a discovered circuit component
would be a mistake the published account already warns against.

MLP 11 is the published member the top-4 cut misses: its effect (+0.131) is real
but smaller than MLP 8's, so a size-matched top-4 ranks it fifth.


## Path patching — the chain, and what it found on its own

Phase 2's iterative chain. Each round's receivers are the heads discovered in the
round before; the answer key is never consulted to choose them. The receiver input
and position come from the paper's account of the mechanism, exactly as Phase 2's
came from the IOI paper's — this is guided rediscovery, and *which* heads turn up
is not constrained.

| round | question | top senders | published class |
|---|---|---|---|
| 0 | which heads move the prediction without another head relaying it? | `9.1` +0.246, `7.10` +0.163, `8.11` +0.098, `6.9` +0.091 | year head -> MLP 8, year head -> MLP 9 |
| 1 | which heads feed those heads' values at YY? | `0.1` +0.036, `0.3` +0.018, `0.5` +0.012, `0.10` -0.010 | appendix upstream |
| 2 | which heads feed *those* heads' values at YY? | *chain halted* | — |

Union across rounds: **7/7**,
precision 0.78.

Round 1 asked which heads feed the round-0 heads' values at `YY` — the receiver
spec the paper states for all seven circuit heads. What it returned is the set the
paper's **Appendix B** names as those heads' upstream dependencies
(`0.1`, `0.3`, `0.5`), which the plan
recorded in advance as the secondary comparison and which the chain was not told
about.


## The receiver-side criterion, at a recalibrated threshold

Phase 3's rule, unchanged:

> threshold = 99th percentile of `|path_signal|` under a shuffled-source null,
> rounded up to two significant figures

Recalibrated on this task's null and committed in
`results/phase6_preregistration.json` before this comparison existed. It produced
**0.046**, against Phase 3's 0.11 on IOI — the null here is much
tighter (median 0.0012, 99th percentile
0.0452, max 0.0615 over
72 measurements).

Inheriting IOI's 0.11 would have been the wrong call in the other direction from
Phase 3's warning: here it would have been far too *strict*, not too lenient. That
is the argument for recalibrating the null under a fixed rule rather than reusing
a number.

| group | position | receivers | senders scored | cleared | which |
|---|---|---|---|---|---|
| round 1 | YY | 4 | 72 | 2 | `0.1`, `0.3` |

Scored against the published circuit: **0/7**,
precision 0.00. As in Phase 3 this is a *different definition of
found* and is reported beside the logit-based numbers rather than merged into them.

### 0 of 7 — but only 2 of the 7 were ever in scope

A sender must sit strictly below its receiver. The chain's only surviving receiver
group sits at layers 6+, so of the 7 published heads
only **2** (`5.1`, `5.5`) were
eligible to be scored as senders at all. The other 5
(`6.9`, `7.10`, `8.8`, `8.11`, `9.1`) occupy the *receiver* slot in
this measurement — they are unmeasured, not measured-and-failed, exactly the
distinction Phase 3 drew for `9.0` and `11.9`.

So the honest reading is **0 of the 2 in-scope heads cleared the bar**,
not 0 of 7. That is still a miss, and it is the clearest negative result in this
phase.

What the criterion *did* find is the appendix set: `0.1`, `0.3`.
Against the **extended 10-head circuit** the plan fixed in advance — the seven plus
Appendix B's `0.1`, `0.3`, `0.5` — that is
**2/10**.
The criterion is doing on this task what it did on IOI: finding early heads that
deliver signal to a receiver without moving the output much, and finding nothing
among the heads that move the output directly.


## Searching for the receiver specifications

Phase 4's exhaustive screen, unchanged. `search.py` does not import either
ground-truth module, and the run asserts that before starting — the check was
widened in this phase from "does not import `ground_truth`" to "does not import
any `ground_truth*` module", because a second answer key would otherwise have
opened a hole in the guarantee.

This circuit is better served by the check than IOI was. For IOI only three of
seven classes had a published receiver spec, so four were unscoreable by
construction. Here the paper states one covering all
7 heads at once — "the most important influences on these
heads are the influences on their **values at the YY position**" — so every head
has a published `v@YY` to check the search against.

| outcome | count |  |
|---|---|---|
| agreement | 7 | ✅ |

**Of the 7 specifications the search could weigh, it recovered 7.**

| head | class | published | rank | search's own top spec | outcome |
|---|---|---|---|---|---|
| `5.1` | year head -> MLP 8 | `v@YY` | 1 | `v@YY` +0.055 | agreement |
| `5.5` | year head -> MLP 8 | `v@YY` | 1 | `v@YY` +0.052 | agreement |
| `6.9` | year head -> MLP 8 | `v@YY` | 1 | `v@YY` +0.080 | agreement |
| `7.10` | year head -> MLP 8 | `v@YY` | 1 | `v@YY` +0.107 | agreement |
| `8.11` | year head -> MLP 8 | `v@YY` | 1 | `v@YY` +0.081 | agreement |
| `8.8` | year head -> MLP 8 | `v@YY` | 1 | `v@YY` +0.032 | agreement |
| `9.1` | year head -> MLP 9 | `v@YY` | 1 | `v@YY` +0.218 | agreement |

### Do the position labels matter?

The unlabelled screen scores bare token indices `t0…tN` with no semantic meaning
attached. Labels are attached *after* the search, purely to read its output.

| screen | top positions within its own top 50 |
|---|---|
| semantic | `YY` x41, `END` x9 |
| absolute | `t8` x41 (= YY), `t12` x9 (= END) |

Given only bare token indices, the search concentrated on the same positions the
labelled screen used, and its top five specifications are identical up to the
position's name. The labels were not carrying the result.

Unlike IOI, this task needed no restriction to a single template or ordering for
the absolute screen: the published task is one sentence frame with single-token
substitutions, so every prompt already has the same length and index *k* means the
same thing in every row.


## A complication: the two circuits are not disjoint

2 of the 7 published greater-than heads are also
members of the published IOI circuit.

| head | IOI class | greater-than class |
|---|---|---|
| `5.5` | induction | year head -> MLP 8 |
| `6.9` | induction | year head -> MLP 8 |

This phase recovered 2 of those 2
(`5.5`, `6.9`), and Phase 1 had already
recovered both on IOI. So of the
7/7 headline,
**5 heads are ones no
earlier phase had ever found**, and 2 were already known to this
pipeline from the other task.

The overlap extends to the appendix set the path chain surfaced: `0.1` is an IOI duplicate token head.

That does not make the transfer result circular — the search and the chain were
given no IOI information, and the heads were rediscovered from this task's own
counterfactual. But "a second, independent circuit" is not quite the right phrase
for a target that shares 2 components with the first, and the number
above is the honest version of it.


## What actually transferred — the measure of generality

Recovery numbers say how well the method did. This says how much of it was the
*same method*. Every file in the repository falls into exactly one row.

### Pure reuse — imported and called, not one line changed

| module | what it does | used here for |
|---|---|---|
| `interventions.py` | activation patching, path patching, sweeps, greedy narrowing | every measurement in this phase |
| `search.py` | receiver-specification screen and confirmation | both screens, stage B |
| `metrics.py` | answer-key-free KL and total variation | the generic-metric columns |
| `model.py` | model loading | — |
| `corruption.py` | generic vocabulary substitution | both generic schemes |

`interventions.py`, `search.py` and `metrics.py` are the causal core, and **none of
them was touched**. They type-annotate against `IOIDataset` but never depend on it
at runtime, so a dataset exposing the same five members drops straight in. That
contract was implicit before this phase and is now written down at the top of
`greater_than.py`.

### Changed, and exactly how much

| module | change | why |
|---|---|---|
| `comparison.py` | added a `circuit` parameter, defaulting to IOI | it hard-imported the IOI answer key at module level, so it could only ever score one circuit |
| `ground_truth.py` | added a `CIRCUIT` alias | so `comparison.py` need not know which circuit it holds |
| `ioi.py` | generic corruption body moved out, call site left | so both tasks call one function instead of two lookalikes |

No existing call site was edited. Every `compare(...)` written in Phases 1–5 still
means what it meant, because the new parameter defaults to IOI — verified by
`check_patching.py` passing unchanged, and by IOI's corrupted token tensors hashing
identically under all four schemes before and after the corruption extraction.

### New, and necessarily task-specific

| module | why it has to be new |
|---|---|
| `greater_than.py` | the task: template, positions, counterfactual, metric |
| `ground_truth_greater_than.py` | the second published circuit, as inert data |
| `run_phase6_greater_than.py` | the runner, wiring the above into the existing library |
| `phase6_report.py` | this report |

**The honest summary**: the causal machinery transferred untouched. The scoring
module needed a parameter it should always have had. Everything else that is new
is either the task or the answer key — the two things the README's ladder already
lists as *supplied*.


## What this phase does and does not show

**Recovery is better here than on IOI**, not worse: 7/7 (100%) against 69% of IOI's 26 under the same cutoff and metric. That is the opposite of the failure mode this phase was built to detect, and it is worth being precise about why it is not a stronger result than it looks: seven targets is a smaller and easier set than twenty-six, and this circuit has no analogue of IOI's previous-token heads — the class that acted only through other heads and that activation patching structurally could not see.

The pipeline was pointed at a circuit built by a different group, on a different
task, with a different counterfactual and a different metric, and the causal core
ran against it without modification. Path patching reached
7/7 at precision
0.78, and its second round independently produced the heads the
paper's appendix names as upstream dependencies — a set fixed in the plan before
the run and never shown to the chain.

Phase 5's asymmetry reproduced: the fully generic pairing recovers 4/7 size-matched against 6/7 for the hand-built pairing. The answer-key-free metric held up again: KL recovered
7/7 against the hand-built
metric's 7/7, on a task
whose hand-built metric — a probability difference over a hundred year tokens —
looks nothing like IOI's two-token logit difference.

**What it does not show.** Three limits, none of them incremental:

1. **Same model.** Greater-than lives in GPT-2 small, as IOI does. This tests
   generality across *tasks and circuits*, not across models. Nothing here licenses
   a claim about a model the pipeline has not seen.
2. **Still supplied: which behaviour to study.** The plan named the task, the
   template and the counterfactual, all taken from the paper. The README's ladder
   put task construction above the line this project has crossed, and this phase
   does not move it — it only shows that everything *below* the line transfers.
3. **Two circuits is two.** A method that fits one circuit and transfers to a
   second is better evidence than one that fits one circuit. It is not evidence
   that it transfers to circuits unlike both, and the honest reading of this phase
   is that one specific failure mode — being silently fitted to IOI — was tested
   for and not found.
