# Phase 4 — searching for receiver specifications
Phases 1-3 were told which head input, at which position, to interrogate. Those choices came from the paper's account of the mechanism, which is why nothing so far transfers to a circuit nobody has published. This phase searches for them instead, on the same task, and then checks the search against the answer it was not allowed to see.

The space and budget were fixed in [PHASE4_SEARCH_SPACE.md](PHASE4_SEARCH_SPACE.md), committed before the search code was written. `causal_interp/search.py` does not import `ground_truth`, and the run asserts that before starting.

## Run configuration
| setting | value |
|---|---|
| model | `gpt2` |
| gpu | `NVIDIA GeForce RTX 5060 Laptop GPU` |
| torch | `2.11.0+cu128` |
| transformer_lens | `3.7.1` |
| python | `3.12.10` |
| prompts | `128` |
| seed | `0` |
| signal_threshold | `0.11` |
| top_k_confirmed | `20` |
| ambiguity_rule | `published spec in top 3 and within 20% of the best score` |
| runtime_seconds | `1865.1` |

## 1. What the search found, before checking
Stage A scores every receiver specification by splicing that one input, at that one position, from the clean run into the corrupted one. Exhaustive over the grid.

**Semantic positions** — top 12 of 3024 specifications:

| head | input | position | screen score | published class |
|---|---|---|---|---|
| **10.7** | q | END | -0.520 | negative name mover |
| **11.10** | q | END | -0.336 | negative name mover |
| **5.5** | q | S2 | +0.318 | induction |
| **8.6** | v | S2 | +0.272 | s-inhibition |
| **8.10** | v | S2 | +0.257 | s-inhibition |
| **7.9** | v | S2 | +0.210 | s-inhibition |
| **9.9** | q | END | +0.199 | name mover |
| **10.7** | k | S2 | -0.142 | negative name mover |
| **10.0** | q | END | +0.124 | name mover |
| **3.0** | q | S2 | +0.122 | duplicate token |
| **9.7** | v | S2 | +0.110 | backup name mover |
| **11.10** | k | S2 | -0.100 | negative name mover |

**Absolute positions (single template, no semantic labels)** — top 12 of 6912 specifications:

| head | input | position | screen score | published class |
|---|---|---|---|---|
| **10.7** | q | `t15` (= END) | -0.536 | negative name mover |
| **11.10** | q | `t15` (= END) | -0.345 | negative name mover |
| **5.5** | q | `t11` (= S2) | +0.224 | induction |
| **8.10** | v | `t11` (= S2) | +0.209 | s-inhibition |
| **8.6** | v | `t11` (= S2) | +0.185 | s-inhibition |
| **9.9** | q | `t15` (= END) | +0.159 | name mover |
| **7.9** | v | `t11` (= S2) | +0.139 | s-inhibition |
| **10.7** | k | `t11` (= S2) | -0.133 | negative name mover |
| **10.0** | q | `t15` (= END) | +0.106 | name mover |
| **9.6** | q | `t15` (= END) | -0.099 | name mover |
| **11.10** | k | `t11` (= S2) | -0.095 | negative name mover |
| **11.2** | q | `t15` (= END) | -0.089 | backup name mover |

### Which positions the search prefers overall
Counting how often each position appears in the top 50 specifications — the search's own view of where the task's information lives, with no labels supplied in the absolute case.

- **semantic**: `S2` ×32, `END` ×16, `S2+1` ×2
- **absolute**: `t11` (= S2) ×33, `t15` (= END) ×15, `t12` (= S2+1) ×2

## 2. The rediscovery check
Only now is the published circuit consulted. For each published head that has a published *receiver* specification, its 21 candidate specifications are ranked by the search's own score, and the published one is located in that ranking.

Outcome rule, stated openly and fixed at analysis time rather than pre-registered: **agreement** if the published spec ranks first, **ambiguous** if it ranks in the top 3 and scores within 20% of the best, **disagreement** otherwise. Raw ranks and scores are shown so the labels can be second-guessed.

| head | class | published spec | its rank | search's top | top score | outcome |
|---|---|---|---|---|---|---|
| **5.5** | induction | `k@S1+1` | 17 | `q@S2` | +0.318 | ⊘ unmeasurable |
| **5.8** | induction | `k@S1+1` | 17 | `q@S2` | -0.012 | ⊘ unmeasurable |
| **5.9** | induction | `k@S1+1` | 17 | `q@S2` | +0.059 | ⊘ unmeasurable |
| **6.9** | induction | `k@S1+1` | 17 | `q@S2` | +0.091 | ⊘ unmeasurable |
| **7.3** | s-inhibition | `v@S2` | 1 | `v@S2` | +0.069 | ✅ agreement |
| **7.9** | s-inhibition | `v@S2` | 1 | `v@S2` | +0.210 | ✅ agreement |
| **8.6** | s-inhibition | `v@S2` | 1 | `v@S2` | +0.272 | ✅ agreement |
| **8.10** | s-inhibition | `v@S2` | 1 | `v@S2` | +0.257 | ✅ agreement |
| **9.0** | backup name mover | `q@END` | 1 | `q@END` | +0.003 | ✅ agreement |
| **9.6** | name mover | `q@END` | 1 | `q@END` | -0.062 | ✅ agreement |
| **9.7** | backup name mover | `q@END` | 2 | `v@S2` | +0.110 | ✗ disagreement |
| **9.9** | name mover | `q@END` | 1 | `q@END` | +0.199 | ✅ agreement |
| **10.0** | name mover | `q@END` | 1 | `q@END` | +0.124 | ✅ agreement |
| **10.1** | backup name mover | `q@END` | 1 | `q@END` | +0.038 | ✅ agreement |
| **10.2** | backup name mover | `q@END` | 1 | `q@END` | +0.029 | ✅ agreement |
| **10.6** | backup name mover | `q@END` | 1 | `q@END` | +0.070 | ✅ agreement |
| **10.7** | negative name mover | `q@END` | 1 | `q@END` | -0.520 | ✅ agreement |
| **10.10** | backup name mover | `q@END` | 1 | `q@END` | +0.059 | ✅ agreement |
| **11.2** | backup name mover | `q@END` | 1 | `q@END` | -0.071 | ✅ agreement |
| **11.9** | backup name mover | `q@END` | 1 | `q@END` | +0.009 | ✅ agreement |
| **11.10** | negative name mover | `q@END` | 1 | `q@END` | -0.336 | ✅ agreement |

**16/21 agreement, 0/21 ambiguous, 4/21 unmeasurable, 1/21 disagreement.**

**Unmeasurable is not disagreement.** For `5.5`, `5.8`, `5.9`, `6.9` the published specification scores *exactly* zero — not a small number, an exact floating-point zero. Under the `s2_swap` corruption the S1+1 position is bit-identical between the clean and corrupted runs, so every one of the 432 specifications at that position is unscoreable, the published one included. The search did not weigh `k@S1+1` against the alternatives and prefer something else; it was handed a counterfactual that cannot see that position at all. This is the same structural blindness Phase 1 measured (576/576 exact zeros before S2) arriving again, one phase later, in a new guise.

Counting these as search failures would credit the search with a defect belonging to the corruption scheme. Counting them as successes would be worse. They are reported as their own category and excluded from both.

5 published heads have no published *receiver* specification to check against — `0.1`, `0.10`, `2.2`, `3.0`, `4.11`. The paper describes them by what they send, not what they receive. A search result for them is unfalsifiable here rather than correct, so they are excluded from the tally rather than counted as successes.

## 3. Did the search need the position labels?
The semantic search uses positions named IO, S1, S2 and END — labels that already encode which name is the indirect object and where the subject repeats. The absolute search has only bare token indices on a single template, so it is the one that tests whether the method can find the structure rather than be handed it.

Of the top 50 specifications the absolute search returned, **50** sit at token indices that turn out to carry a semantic label, and 0 do not.

| what that index turns out to be | count in top 50 |
|---|---|
| S2 | 33 |
| END | 15 |
| S2+1 | 2 |

The labels in that table were attached *after* the search, purely to interpret its output. The search itself ranked bare indices.

## 4. Stage B — what feeds the specifications the search chose
For the top surviving specifications, every head below them was swept as a sender and scored with both of the project's criteria: delivery to the receiver (`path_signal`, against Phase 3's recorded threshold of 0.11) and effect on the output logits.

| specification | screen score | senders clearing threshold | top senders |
|---|---|---|---|
| `10.7.q@END` | -0.520 | 2 | 9.9 (+0.57, name mover), 9.6 (+0.22, name mover), 8.10 (+0.07, s-inhibition) |
| `11.10.q@END` | -0.336 | 4 | 9.9 (+0.58, name mover), 10.7 (-0.35, negative name mover), 9.6 (+0.18, name mover) |
| `5.5.q@S2` | +0.318 | 0 | 0.1 (+0.08, duplicate token), 3.0 (+0.07, duplicate token), 3.4 (+0.03) |
| `8.6.v@S2` | +0.272 | 3 | 5.5 (+0.39, induction), 6.9 (+0.11, induction), 3.0 (+0.11, duplicate token) |
| `8.10.v@S2` | +0.257 | 1 | 5.5 (+0.17, induction), 3.0 (+0.10, duplicate token), 6.9 (+0.04, induction) |
| `7.9.v@S2` | +0.210 | 1 | 5.5 (+0.20, induction), 3.0 (+0.09, duplicate token), 6.9 (+0.05, induction) |
| `9.9.q@END` | +0.199 | 4 | 8.6 (+0.41, s-inhibition), 8.10 (+0.25, s-inhibition), 7.9 (+0.20, s-inhibition) |
| `10.7.k@S2` | -0.142 | 0 | 0.1 (+0.06, duplicate token), 0.3 (+0.03), 3.0 (+0.03, duplicate token) |
| `10.0.q@END` | +0.124 | 4 | 8.6 (+0.27, s-inhibition), 8.10 (+0.24, s-inhibition), 7.9 (+0.17, s-inhibition) |
| `3.0.q@S2` | +0.122 | 0 | 0.1 (+0.08, duplicate token), 0.4 (+0.03), 0.3 (+0.03) |

## 5. Assessment
On the 21 heads where the paper names a receiver specification, the search agrees with it 16 times, is ambiguous 0 times, and disagrees 1 time. A further 4 are unscoreable under this corruption scheme, so of the 17 the search could actually weigh, it recovered the published specification **16**.

**What this does and does not license.** The search recovered receiver specifications without being told them, on a task where the answer happens to be known. That is the step Phases 1-3 could not take. It is not the same as autonomous discovery, and three things still stand between the two.

1. **The screen is a logit-effect screen.** A receiver only reaches stage B if splicing its input moves the output. Phase 3 established that some genuine circuit links do not move the output — the previous-token heads are exactly that case — so this search inherits that blind spot by construction, not by accident.

2. **The task, the counterfactual and the metric are still supplied.** The IOI templates, both corruption schemes and the logit-difference metric were all designed by hand from knowledge of what the model is doing. A system pointed at an unfamiliar circuit would have to construct its own counterfactual, and nothing here does that. It is the largest remaining gap and it is larger than the one this phase closed.

3. **There is no answer key on an unfamiliar circuit.** Every outcome above was legible only because a published specification existed to compare against. The search itself emits a ranking either way, and nothing in that ranking distinguishes the case where it is right from the case where it is wrong.

**The 1 genuine disagreement matters more than the count suggests.** There the search weighed the published specification against the alternatives and preferred a different one. On this task that is catchable. On an unfamiliar circuit the same result would be indistinguishable from a correct answer, since the only thing marking it wrong is a published account to compare against.

Worth noting what did *not* happen: no published specification landed in the ambiguous band, where the search ranks it near the top but cannot separate it from a rival. Rankings here were decisive rather than marginal. That is a better outcome than the alternative, but it is a property of this task at this sample size and should not be assumed to hold elsewhere — an autonomous use of the method still needs a calibrated notion of when its own ranking is uninformative, and this phase does not provide one.

