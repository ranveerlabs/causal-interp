# Phase 3 — a pre-registered receiver-side criterion
Phases 1 and 2 both scored a head as *found* by what it does to the output logit difference, and both arrived at 20/26. Phase 2 also computed `path_signal` — how much of a receiver's clean-vs-corrupted difference a path actually delivers — and noticed that this diagnostic scored several of the missing heads well.

That observation is exactly what makes it dangerous to adopt. This phase fixes the threshold first, by a rule, in a step that computes no real measurement, and then applies it without adjustment.

## 1. The pre-registration

> threshold = 99% percentile of |path_signal| under a shuffled-source null, rounded up to 2 significant figures

The null runs the identical procedure — same freezing, same path, same receiver, same projection — but draws the sender's clean value from a *different* prompt in the batch. The value carried is a real activation of the right kind; only its correspondence to the prompt is destroyed. Whatever projection survives that is what the method manufactures from nothing, so the 99th percentile of it fixes the false-positive rate at about one in a hundred, in advance, the same role Phase 1's 0.02 cutoff played.

| null statistic | value |
|---|---|
| null measurements pooled | 552 |
| null median |signal| | 0.0006 |
| null mean |signal| | 0.0062 |
| null max |signal| | 0.1846 |
| 99th percentile (raw) | 0.1052 |
| **threshold (rounded up, 2 s.f.)** | **0.11** |

Recorded in `results/phase3_preregistration.json` and committed before the comparison below was run, so the ordering is visible in git history rather than merely claimed.

### What this pre-registration is not
It is not blind. Phase 2 printed a handful of real `path_signal` values — the previous-token heads among them — and those numbers were known before this threshold was derived. A genuinely blind pre-registration was no longer available once Phase 2 was published.

What is claimed instead is narrower and checkable: the number was produced by a fixed rule applied to a null distribution, not selected; the rule and its two free parameters were written into the code before the null was run; and the number was not adjusted after the comparison. A reader who suspects the rule itself was reverse-engineered should weigh the sensitivity table in section 5, which shows what a stricter per-group bar would do.

## 2. What the criterion can and cannot see
Two limits are structural and worth stating before any number.

- **The output round is out of scope.** Round 0 asks what a head does to the logits directly. Its receiver *is* the output, where a receiver-side measure and the logit measure are the same quantity — there is no independent second opinion to take. Heads found only by direct effect (the name movers) therefore cannot be found by this criterion, and the like-for-like comparison in section 4 restricts the logit criterion to the same rounds to keep that from reading as a failure.

- **Under `s2_swap`, S1+1 is undefined rather than zero.** Clean and corrupted coincide there, so the quantity this criterion normalizes by is exactly zero and the ratio does not exist. Those cells are dropped, not scored as misses.

## 3. Results by receiver group

**s2_swap round 1** — receivers `9.9.q@END`, `10.7.q@END`, `9.6.q@END`, `11.10.q@END`, senders at END.

| sender | path signal | clears 0.11 | published class |
|---|---|---|---|
| **8.6** | +0.243 | yes | s-inhibition |
| **8.10** | +0.180 | yes | s-inhibition |
| **7.9** | +0.134 | yes | s-inhibition |
| **7.3** | +0.088 | no | s-inhibition |
| **6.1** | +0.010 | no | — *not in published circuit* |
| **8.3** | +0.006 | no | — *not in published circuit* |

**s2_swap round 2** — receivers `8.6.v@S2`, `8.10.v@S2`, `7.9.v@S2`, `7.3.v@S2`, senders at S2.

| sender | path signal | clears 0.11 | published class |
|---|---|---|---|
| **5.5** | +0.227 | yes | induction |
| **3.0** | +0.110 | yes | duplicate token |
| **6.9** | +0.058 | no | induction |
| **5.9** | +0.042 | no | induction |
| **0.1** | +0.034 | no | duplicate token |
| **3.4** | +0.028 | no | — *not in published circuit* |

**s2_swap round 3** — receivers `5.5.k@S1+1`, `3.0.k@S1+1`, `6.9.k@S1+1`, `5.9.k@S1+1`, senders at S1+1.

*No eligible sender with a defined measurement.*

**abc round 1** — receivers `9.9.q@END`, `10.7.q@END`, `9.6.q@END`, `11.10.q@END`, senders at END.

| sender | path signal | clears 0.11 | published class |
|---|---|---|---|
| **8.6** | +0.160 | yes | s-inhibition |
| **7.9** | +0.079 | no | s-inhibition |
| **8.10** | +0.071 | no | s-inhibition |
| **7.3** | +0.043 | no | s-inhibition |
| **8.3** | +0.017 | no | — *not in published circuit* |
| **8.5** | +0.010 | no | — *not in published circuit* |

**abc round 2** — receivers `8.10.v@S2`, `7.9.v@S2`, `8.6.v@S2`, `7.3.v@S2`, senders at S2.

| sender | path signal | clears 0.11 | published class |
|---|---|---|---|
| **5.5** | +0.194 | yes | induction |
| **0.1** | +0.100 | no | duplicate token |
| **3.0** | +0.083 | no | duplicate token |
| **0.10** | +0.042 | no | duplicate token |
| **6.9** | +0.041 | no | induction |
| **5.10** | +0.032 | no | — *not in published circuit* |

**abc round 3** — receivers `0.1.k@S1+1`, `3.0.k@S1+1`, `5.5.k@S1+1`, `4.4.k@S1+1`, senders at S1+1.

*No eligible sender with a defined measurement.*

**previous-token probe (receivers layer >= 3)** — receivers `5.5.k@S1+1`, `3.0.k@S1+1`, `6.9.k@S1+1`, `5.9.k@S1+1`, senders at S1+1.

| sender | path signal | clears 0.11 | published class |
|---|---|---|---|
| **2.2** | +0.197 | yes | previous token |
| **2.9** | +0.101 | no | — *not in published circuit* |
| **1.0** | +0.060 | no | — *not in published circuit* |
| **0.7** | +0.017 | no | — *not in published circuit* |
| **2.4** | +0.017 | no | — *not in published circuit* |
| **2.3** | +0.008 | no | — *not in published circuit* |

**previous-token probe (receivers layer >= 5)** — receivers `5.5.k@S1+1`, `6.9.k@S1+1`, `5.9.k@S1+1`, senders at S1+1.

| sender | path signal | clears 0.11 | published class |
|---|---|---|---|
| **4.11** | +0.361 | yes | previous token |
| **2.2** | +0.195 | yes | previous token |
| **4.7** | -0.185 | yes | — *not in published circuit* |
| **3.7** | +0.121 | yes | — *not in published circuit* |
| **4.3** | +0.119 | yes | — *not in published circuit* |
| **2.9** | +0.101 | no | — *not in published circuit* |

**previous-token probe (receivers layer >= 6)** — receivers `6.9.k@S1+1`, senders at S1+1.

| sender | path signal | clears 0.11 | published class |
|---|---|---|---|
| **4.11** | +0.344 | yes | previous token |
| **2.2** | +0.187 | yes | previous token |
| **4.7** | -0.174 | yes | — *not in published circuit* |
| **5.6** | +0.130 | yes | — *not in published circuit* |
| **3.7** | +0.107 | no | — *not in published circuit* |
| **4.3** | +0.096 | no | — *not in published circuit* |

## 4. The two criteria, side by side
These are two definitions of *found*, reported as two columns rather than one merged score. A head that clears one and not the other is a genuine disagreement about what counts as being part of the circuit, not noise to be averaged away.

| published class | logit (all rounds) | logit (rounds 1+) | receiver-side (>= 0.11) |
|---|---|---|---|
| name mover | 3/3 | 0/3 | 0/3 |
| backup name mover | 6/8 | 0/8 | 0/8 |
| negative name mover | 2/2 | 0/2 | 0/2 |
| s-inhibition | 4/4 | 4/4 | 3/4 |
| induction | 3/4 | 3/4 | 1/4 |
| duplicate token | 1/3 | 1/3 | 1/3 |
| previous token | 0/2 | 0/2 | 2/2 |
| **total** | **19/26** | **8/26** | **7/26** |
| precision | 0.90 | 0.80 | 0.64 |

The middle column is the like-for-like one: same rounds, same receivers, same paths, scored by effect on the output instead of delivery to the receiver.

### The six heads neither Phase 1 nor Phase 2 found

| head | published class | best path signal | clears 0.11 |
|---|---|---|---|
| 0.10 | duplicate token | +0.042 | no |
| 2.2 | previous token | +0.197 | **yes** |
| 4.11 | previous token | +0.361 | **yes** |
| 5.8 | induction | +0.008 | no |
| 9.0 | backup name mover | — | not measurable |
| 11.9 | backup name mover | — | not measurable |

**2 of 6** are recovered by the pre-registered receiver-side criterion: `2.2` (previous token), `4.11` (previous token).

### Where the criteria disagree

| agreement | count | heads |
|---|---|---|
| found by both | 5 | `3.0`, `5.5`, `7.9`, `8.6`, `8.10` |
| logit only | 16 | `3.4`, `5.9`, `6.6`, `6.9`, `7.3`, `9.6`, `9.7`, `9.9`, `10.0`, `10.1`, `10.2`, `10.6`, `10.7`, `10.10`, `11.2`, `11.10` |
| receiver-side only | 6 | `2.2`, `3.7`, `4.3`, `4.7`, `4.11`, `5.6` |

The *logit only* set is dominated by heads whose receiver is the output, which the receiver-side criterion cannot evaluate at all. The *receiver-side only* set is the interesting one: paths that demonstrably deliver their content and still do not move the prediction.

## 5. Sensitivity: a stricter per-group bar
Pooling the null across receiver groups controls the overall false-positive rate, which is the right target for a single criterion. It also means a group with a wide null gets a bar that is lenient relative to its own noise. Both thresholds were fixed by the same rule at the same time, so the comparison below is not a second bite at the cherry.

| receiver group | null max | per-group threshold | clears it |
|---|---|---|---|
| s2_swap round 1 | 0.134 | 0.11 | `7.9`, `8.6`, `8.10` |
| s2_swap round 2 | 0.135 | 0.074 | `3.0`, `5.5` |
| s2_swap round 3 | — | — | — |
| abc round 1 | 0.081 | 0.037 | `7.3`, `7.9`, `8.6`, `8.10` |
| abc round 2 | 0.145 | 0.09 | `0.1`, `5.5` |
| abc round 3 | — | — | — |
| previous-token probe (receivers layer >= 3) | 0.102 | 0.085 | `2.2`, `2.9` |
| previous-token probe (receivers layer >= 5) | 0.185 | 0.14 | `2.2`, `4.7`, `4.11` |
| previous-token probe (receivers layer >= 6) | 0.178 | 0.12 | `2.2`, `4.7`, `4.11`, `5.6` |

**Depends on pooling:** `3.7`, `4.3` clears the pooled threshold but not its own group's. Those discoveries should be read as weaker than the rest.

None of them is a published head. The lenience pooling introduces produced false positives and none of the recoveries, so every published head found above also clears its own group's stricter bar.

## 6. Two limitations carried forward, not fixed

### (a) Why the `abc` chain dies at round 1

The chain dies on the *logit* criterion, and the receiver-side numbers show it is not because the paths are absent:

| sender | logit effect (Phase 2) | path signal | published class |
|---|---|---|---|
| `8.6` | +0.0020 | +0.160 | s-inhibition |
| `7.9` | -0.0024 | +0.079 | s-inhibition |
| `8.10` | -0.0043 | +0.071 | s-inhibition |
| `7.3` | -0.0015 | +0.043 | s-inhibition |
| `8.3` | -0.0005 | +0.017 | — *not in published circuit* |

The paths deliver. The prediction does not move because `abc` replaces all three names, so the tokens the logit difference is defined on — the clean prompt's IO and S — are not in the corrupted prompt at all. Restoring a name mover's query makes it attend to the right *position*, but the token sitting there is a different name, so there is no logit difference to restore. That is a property of the counterfactual, not of the circuit, and it is why `abc` contributes little past round 0 where the answer tokens still carry the measurement.

### (b) Receiver inputs are still supplied, not searched

Every round in Phases 2 and 3 was told *where to look*: that S-inhibition heads act on name movers' queries, that duplicate-token information arrives as a value at S2, that induction keys live at S1+1. Those choices come from the paper's account of the mechanism. Which heads turn up was never constrained — all 144 are swept every round — but the question asked of them was.

That is the line this project has not yet crossed. Everything so far is **guided rediscovery**: given the right question, the method finds the right components, and finds them in the right causal order. The autonomous loop the README describes has to generate the questions too — searching over receiver inputs, positions and depths without being handed the mechanism first — and nothing here demonstrates that.

The distinction matters most exactly where the project is aimed. On a circuit nobody has published there is no paper to supply the receiver inputs, so a method that needs them supplied does not yet transfer. Search over receiver specifications is the concrete next problem, and it is a larger one than either patching primitive: the space is the product of receiver head, input, and position, and unlike this phase there is no answer key to check the search against.

## 7. What this phase settled
The pre-registered criterion recovers 2 of the 6 heads neither earlier phase found: `2.2` (previous token), `4.11` (previous token). The threshold was fixed before the measurement and not touched afterwards.

Below the bar and not rescued: `0.10` (duplicate token, +0.042), `5.8` (induction, +0.008).

Outside the criterion's scope entirely: `9.0`, `11.9` — these sit above every receiver layer available in rounds 1 and later, so no path into the swept receivers reaches them. Not measured and found wanting; not measured at all.

**The criterion is noisier than the one it sits beside.** Precision 0.64 against 0.90 for the logit criterion. It finds the previous-token heads and it also admits several heads with no published role. Both facts are properties of the same fixed threshold and neither is reported without the other.

**The two criteria are not ranked, and the scores are not merged.** On the like-for-like comparison — same rounds, same receivers, same paths — the logit criterion finds 8/26 and the receiver-side criterion 7/26, but they disagree about *which* heads, not merely how many: previous-token heads appear only in the second, several induction and name-mover heads only in the first. Adding them into a single recall number would report a larger figure while destroying the only genuinely new information this phase produced.

**What the disagreement means.** A head can deliver its content to the next stage of the circuit and still leave the prediction unmoved, and the two criteria simply take opposite views of whether that counts as being part of the circuit. Neither view is wrong. Which one is appropriate depends on the question: explaining a behaviour argues for the output criterion, mapping a mechanism argues for the receiver-side one. Phases 1 and 2 answered only the first while appearing to answer both, and making that visible — rather than raising a number — is what this phase was for.

