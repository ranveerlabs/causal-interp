# Phase 5 — scoping the remaining hand-built dependency
Phase 4 removed the need to be told which head input, at which position, to interrogate, and named what was left: the task, the corruption schemes and the metric are all still built from knowledge of what the model does and what a correct answer looks like.

This phase does not try to build those automatically. It measures how much of the project's result actually depended on them, using IOI — where the answer is still known and can check the substitution. What each piece encodes is itemised in [PHASE5_AUDIT.md](PHASE5_AUDIT.md), written before any of this ran.

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
| threshold | `0.02` |
| runtime_seconds | `535.0` |

## 1. How much room each counterfactual leaves to measure in
Before any circuit result: a corruption is only useful if the clean and corrupted runs differ enough that a patched run can sit somewhere informative between them. That range is what every normalized recovery in this project is divided by.

| corruption | logit diff, clean → corrupted | logit span | KL span | TV span |
|---|---|---|---|---|
| `s2_swap` — semantic, one token | +3.70 → -3.60 | 7.30 | 1.84 | 0.578 |
| `abc` — semantic, three tokens | +3.70 → -0.19 | 3.88 | 3.65 | 0.641 |
| `random_vocab_s2` — generic token, position supplied | +3.70 → +0.08 | 3.61 | 0.76 | 0.451 |
| `random_vocab_any` — generic token, generic position | +3.70 → +2.16 | 1.54 | 1.15 | 0.327 |

The semantic corruptions were designed so the answer flips, which is why `s2_swap` spans a signed range twice the size of anything the generic ones produce. A generic token substitution damages the prompt without reversing the behaviour, so the corrupted run still partly does the task — and the measurement has correspondingly less room.

## 2. Can the metric be replaced?
Every combination of corruption and metric, scored against the same published circuit at the same cutoff (0.02) Phase 1 used. All three metrics come from the *same* forward passes, so any difference between columns is the metric and nothing else.

| corruption | logit difference (hand-built) | KL divergence (general) | total variation (general) |
|---|---|---|---|
| `s2_swap` — semantic, one token | 18/26 · p0.78 | 21/26 · p0.64 | 17/26 · p1.00 |
| `abc` — semantic, three tokens | 15/26 · p0.83 | 16/26 · p0.57 | 8/26 · p0.80 |
| `random_vocab_s2` — generic token, position supplied | 15/26 · p0.79 | 19/26 · p0.47 | 15/26 · p0.60 |
| `random_vocab_any` — generic token, generic position | 17/26 · p0.68 | 14/26 · p0.36 | 9/26 · p0.50 |

Each cell is *heads recovered / 26* and the precision behind it.

### The same comparison with the scale removed
A fixed cutoff is not neutral between metrics: one whose effects simply run larger clears it more often without discriminating any better. Taking each metric's **top 26** — the size of the published circuit — removes the scale and leaves only the ranking, so precision and recall coincide.

| corruption | logit difference (hand-built) | KL divergence (general) | total variation (general) |
|---|---|---|---|
| `s2_swap` — semantic, one token | 18/26 | 19/26 | 19/26 |
| `abc` — semantic, three tokens | 17/26 | 15/26 | 17/26 |
| `random_vocab_s2` — generic token, position supplied | 18/26 | 15/26 | 16/26 |
| `random_vocab_any` — generic token, generic position | 17/26 | 13/26 | 12/26 |
### Do the metrics rank heads the same way?
Spearman correlation between the per-head rankings each metric produces. A high value means the metrics disagree about magnitudes but not about which heads matter.

| corruption | logit ~ KL | logit ~ TV | KL ~ TV |
|---|---|---|---|
| `s2_swap` — semantic, one token | +0.978 | +0.925 | +0.938 |
| `abc` — semantic, three tokens | +0.842 | +0.775 | +0.887 |
| `random_vocab_s2` — generic token, position supplied | +0.711 | +0.705 | +0.971 |
| `random_vocab_any` — generic token, generic position | +0.705 | +0.761 | +0.833 |

### By class, on the primary corruption
| published class | logit difference (hand-built) | KL divergence (general) | total variation (general) |
|---|---|---|---|
| name mover | 3/3 | 3/3 | 3/3 |
| backup name mover | 5/8 | 6/8 | 4/8 |
| negative name mover | 2/2 | 2/2 | 2/2 |
| s-inhibition | 4/4 | 4/4 | 4/4 |
| induction | 3/4 | 4/4 | 3/4 |
| duplicate token | 1/3 | 2/3 | 1/3 |
| previous token | 0/2 | 0/2 | 0/2 |
| **total** | **18/26** | **21/26** | **17/26** |

## 3. Can the corruption be made generic?
The bottom two rows of the grid in section 2 answer this. Reading them against the top two is the whole experiment: the generic schemes substitute a uniformly drawn vocabulary token rather than a semantically chosen name, and `random_vocab_any` does not even choose the position.

| corruption | best recovery | under which metric | precision |
|---|---|---|---|
| `s2_swap` — semantic, one token | 21/26 | KL divergence (general) | 0.64 |
| `random_vocab_s2` — generic token, position supplied | 19/26 | KL divergence (general) | 0.47 |
| `random_vocab_any` — generic token, generic position | 17/26 | logit difference (hand-built) | 0.68 |

## 4. What this phase settled
### The metric: substitutable

On the corruption every earlier phase used, the hand-built logit difference recovers 18/26 heads at the fixed cutoff. Replacing it with a KL divergence over the full next-token distribution — which needs no knowledge of which token is correct — recovers 21/26, and total variation 17/26.

The fixed cutoff flatters whichever metric runs larger, so the size-matched numbers are the ones to read: top-26 gives **18** for the hand-built metric, **19** for KL and **19** for total variation. The per-head rankings of the hand-built metric and KL correlate at +0.98.

That is the substantive positive result of this phase. The single piece of knowledge that looked most load-bearing — knowing which token is the right answer — turns out not to be needed to *locate the circuit*. The two runs the corruption already provides are enough, because the difference between their output distributions is dominated by the behaviour under study.

Worth being precise about what this does not show. The distributional metrics still require a *clean and a corrupted run* to compare, so they inherit whatever knowledge built the corruption. They remove the answer key, not the counterfactual.

### The corruption: partly, and the cost is not where it was expected

| what is supplied | size-matched recovery | configuration |
|---|---|---|
| hand-built corruption, hand-built metric | **18/26** | `s2_swap` + logit difference |
| hand-built corruption, general metric | **19/26** | `s2_swap` + KL / TV |
| generic corruption, hand-built metric | **18/26** | `random_vocab_s2` + logit difference |
| generic corruption, general metric | **16/26** | `random_vocab_s2` + KL / TV |
| nothing supplied at all | **13/26** | `random_vocab_any` + KL / TV |

The expectation going in — stated in the audit and in the brief that commissioned this phase — was that a generic corruption would degrade badly. It does, but not where that prediction pointed. Replacing the semantic swap with a uniformly drawn token while keeping the hand-built metric costs almost nothing (18/26 against 18/26). The loss appears only when *both* pieces are generic: 16/26 with the position still supplied, and 13/26 with nothing supplied.

That pattern is worth stating carefully, because it inverts the natural reading. The two hand-built pieces are not independently load-bearing and they are not additive. Either one alone carries enough task knowledge to locate the circuit; what fails is removing both. A corruption that damages the prompt arbitrarily still produces a usable signal *if the metric knows what to look at*, and a metric that looks at everything still works *if the corruption was aimed at the right thing*. Neither survives the other being taken away.

The mechanism is visible in section 1. `s2_swap` was built to *reverse* the behaviour, so the corrupted run sits as far from clean as the task allows. A random token damages the prompt instead: the corrupted run still partly performs the task, the span collapses, and every normalized recovery is divided by a smaller number. With a metric that reads only the two answer tokens, that shrunken span is still pointed at the right quantity. With a metric that reads the whole distribution, the shrunken span is shared out over every way a corrupted prompt differs, most of which have nothing to do with the circuit.

So the negative result stands, in the form that matters: the fully generic configuration recovers 13/26 against 18/26, a loss of about a third, and on an unfamiliar circuit there would be no published answer to notice that degradation against.

### The task: not attempted, and not because of budget

Constructing a task means choosing which behaviour to study, and that is a different kind of problem from anything in Phases 1-5. Every phase so far takes a behaviour as given and asks how the model implements it. No improvement to patching, searching or scoring turns that into a method for finding behaviours worth studying in the first place — the machinery here has no way to propose a hypothesis, only to test one.

A weak attempt was available and was deliberately not made. Sweeping templates, or mining a corpus for prompts with a predictable completion, would have produced a section in this report and no evidence that the resulting tasks isolate anything mechanistically interesting. The honest position is that this is open.

### Where that leaves the autonomy claim

Stacking what each phase has established, from the outside in:

| ingredient | status | evidence |
|---|---|---|
| behaviour to study | **supplied** | no method attempted; a different kind of problem |
| task template | **supplied** | out of scope for this phase, explicitly |
| corruption content | **supplied** | generic substitute costs little alone, 16/26 once the metric is generic too |
| corruption position | searchable | Phase 4 |
| receiver input and position | searchable | Phase 4, 16 of 17 recovered |
| answer key in the metric | **not needed** | KL over the output distribution: 19/26 vs 18/26 |
| circuit components and wiring | discovered | Phases 1-3 |

The line between the top two rows and the rest is the real boundary. Everything below it is a question about a behaviour that has already been chosen, and the project now answers those with progressively less help. Everything above it is the question of which behaviour to look at, and this project has never addressed it.

A next phase pursuing autonomy on task construction would have to start there, and it would need a validation strategy that does not exist yet: IOI cannot check it, because IOI *is* the supplied task. Checking whether an automatically constructed task is a good one requires either a second published circuit to rediscover, or a criterion for task quality that does not reduce to 'it found the thing we already knew about'.

