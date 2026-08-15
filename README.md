# causal-interp

An autonomous system for discovering and **causally validating** computational mechanisms in neural networks.

## Goal

Most interpretability tooling traces and visualizes: it shows which components light up when a model does something. That produces suggestive pictures, but not claims you can be wrong about.

This project aims at the stronger thing. The system forms hypotheses about the mechanism behind a behaviour, then tests them by intervening on the model and observing whether the behaviour actually changes:

- **Activation patching** — splice activations from a clean run into a corrupted one to isolate which nodes carry the causal signal.
- **Iterative circuit pruning** — strip away components that carry no causal weight, until what remains is a minimal subgraph that still reproduces the behaviour.

The output is a **falsifiable causal claim** — "this circuit implements this behaviour, and here is the intervention that would break it" — not an attention heatmap. The loop from hypothesis to experiment to revised hypothesis is meant to run autonomously.

## Long-term aim

The eventual target is scalable oversight: applying this to models *more capable than the people and systems investigating them*, where no human has the ground truth to check the answer against.

That ambition is exactly why Phase 1 does not start there. Pointing an unvalidated system at an unfamiliar model produces conclusions nobody can check — the system could be confidently wrong and there would be no way to tell.

So Phase 1 runs the system against a small open model with an **already-published circuit** — GPT-2 small's IOI (indirect object identification) circuit, documented in Wang et al. (2022), [*Interpretability in the Wild*](https://arxiv.org/abs/2211.00593), which means there is a known answer. The question Phase 1 has to settle is narrow and concrete:

> Do the system's autonomous conclusions match the published ground truth?

Only once the method demonstrably rediscovers what is already known does it earn the right to be pointed at anything unfamiliar.

## Phase 1 — result

**Partial reproduction. Not a pass.** Activation patching recovers the parts of the published circuit that act directly on the output, and misses the parts that act indirectly. The miss is systematic and has an identifiable cause, which is the useful finding.

Full numbers: **[results/PHASE1_REPORT.md](results/PHASE1_REPORT.md)**.

The run patches every one of GPT-2 small's 144 attention heads at each of 7 semantic token positions, over 128 prompts from 8 templates, under two independent corruption schemes, and scores the result against the paper's 26 heads.

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

Recovery is the union of both corruption schemes; the primary scheme alone gives 18/26. The cutoff was fixed before the results were looked at, and the report shows the full sensitivity sweep around it rather than one flattering choice. Under the primary scheme at a slightly stricter cutoff, **precision reaches 1.00** — the 18 highest-effect heads in the model are all published circuit members, with no false positives.

The split in that table is the whole result: **every class that acts directly on the output logits is fully recovered, and every class that falls short acts indirectly**, feeding other heads rather than the prediction.

Narrowing to a circuit that works *jointly* rather than head-by-head, six nodes restore clean behaviour completely (recovery 1.000), and all six are published circuit members:

```
5.5@S2 (induction) → 8.10@END → 7.9@END → 8.6@END → 7.3@END (S-inhibition) → 3.0@S2 (duplicate token)
```

### Why the misses happen

That the shortfall lands entirely on the indirect classes is not a tuning failure. Three causes, each structural:

1. **Activation patching is not path patching.** The paper derived its circuit with path patching, which measures a component's effect along a specific downstream route. Plain activation patching measures total effect on the output, so a head whose entire contribution is routed through another head can be invisible here while genuinely belonging to the circuit.
2. **The primary corruption scheme is provably blind before the S2 token.** It changes exactly one token, so activations at every earlier position are bit-identical between the clean and corrupted runs. Measured, not assumed: **576/576** head-position cells before S2 come out as exact floating-point zeros. Previous-token heads act at S1+1 and cannot be detected by this scheme even in principle.
3. **Marginal effects understate redundant components.** Backup name movers exist precisely to activate when the primary name movers are removed — patching one head at a time, with everything else intact, is the condition under which they look least important.

The second corruption scheme was added specifically to see past limit 2, and it does surface early-layer structure — but its extra detections are layer-0 heads sitting on name tokens that the corruption itself replaced, which restores *which name is written there* rather than any IOI mechanism. The report flags these as artifacts rather than counting them as discoveries.

**What this earns:** the method demonstrably finds real, published circuit components rather than plausible-looking noise, and its failures are predictable from its construction. It has not earned the right to be pointed at an unknown circuit yet — path patching is the prerequisite, and that is Phase 2.

## Phase 2 — result

**The head count did not move. The mechanism and the precision did.**

Phase 1 predicted that path patching would recover the classes it missed. Tested directly, that prediction was **wrong**: path patching recovers 19/26 on its own, and combined with Phase 1 the total stays at **20/26** — the same six heads are still missing. That is reported as a failed prediction rather than folded away.

Full numbers: **[results/PHASE2_REPORT.md](results/PHASE2_REPORT.md)**.

Two things did improve, and neither shows up in a head count.

**Precision.** Path patching pins every other attention head to its corrupted value, so only the chosen route can carry signal. That strips out most of what total-effect patching swept up incidentally:

| | heads discovered | in circuit | precision |
|---|---|---|---|
| Phase 1 (activation patching) | 28 | 20 | 0.71 |
| Phase 2 (path patching) | 21 | 19 | **0.90** |
| Phase 2, `abc` scheme alone | 14 | 14 | **1.00** |

**The wiring, not just the parts.** Phase 1 produced a ranked list of heads. Phase 2 recovered the paper's causal *order*, without being given it. Each round's receivers are the heads discovered in the round before — the answer key is never consulted to choose them:

| round | question | top senders found | published class |
|---|---|---|---|
| 0 | what moves the logits directly? | 9.9, 10.7, 9.6, 11.10 | name mover / negative name mover |
| 1 | what feeds *those* heads' queries at END? | **8.6, 8.10, 7.9, 7.3** | S-inhibition — all four |
| 2 | what feeds *those* heads' values at S2? | **5.5, 3.0, 6.9, 5.9** | induction + duplicate token |

Round 1 returned all four published S-inhibition heads as its top four senders, from a sweep of all 144. Recovering the wiring is a stronger claim than recovering the component list, and it is the claim Phase 1 could not make at all.

### Previous token heads: measured, not dropped

Phase 1 recorded 0/2 and blamed the corruption scheme. That excuse is now tested. Neither scheme can settle it alone — under `s2_swap` the S1+1 position is bit-identical between runs, and under `abc` the chain dies before producing receivers — so the probe takes each half from where it is sound: receivers discovered by the `s2_swap` chain, measured on `abc`.

| head | effect on logits | signal delivered to receiver |
|---|---|---|
| 2.2 | +0.0003 | **+0.197** |
| 4.11 | +0.0003 | **+0.361** |

**The two measurements disagree, and the disagreement is the finding.** Both heads deliver a fifth to a third of the receiver's entire clean-vs-corrupted difference — the path is there and carries signal — while moving the output logit difference by essentially nothing. The deeper a link sits in the chain, the more of its effect is absorbed before reaching the output.

So logit-difference path patching is the wrong instrument for these heads, rather than the heads being absent. Fixing that needs a metric defined at the receiver instead of at the output — a change of measurement, not of method. It is left for a later phase, and **the previous-token heads are counted as misses in every table above**: adopting the more favourable metric after seeing that it scores better is how a validation exercise stops validating anything.

## Phase 3 — result

**Two definitions of "found", reported side by side. Not merged, and not ranked.**

Phase 2 ended by naming the receiver-side measure as the obvious next step — and by warning that it had been *observed* to score the missing heads well, which is exactly what makes adopting it dangerous. Phase 3 does it under pre-registration.

Full numbers: **[results/PHASE3_REPORT.md](results/PHASE3_REPORT.md)**.

### The threshold was fixed before the measurement

The rule, not the number, was committed:

> threshold = 99th percentile of `|path_signal|` under a shuffled-source null, rounded up to two significant figures

The null runs the identical procedure but draws the sender's clean value from a *different prompt*. The value carried is a real activation; only its correspondence to the prompt is destroyed. Whatever projection survives is what the method manufactures from nothing. That fixes the false-positive rate at ~1% in advance — the role Phase 1's 0.02 played.

It produced **0.11**. Worth noting the null is heavy-tailed — median 0.0006 but 99th percentile 0.105 — so simply inheriting 0.02 would have carried a large false-positive rate on this quantity.

The pre-registration is committed in [`b039915`](../../commit/b039915), a commit containing the threshold and all the code and **no results**. The ordering is checkable in git history rather than asserted here.

**It is not a blind pre-registration, and the report says so.** Phase 2 published real `path_signal` values for the previous-token heads before this rule was written. The narrower claim is that the number was produced by a fixed rule rather than selected, and was not adjusted afterwards.

### What the criterion found

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

The middle column is the like-for-like one — same rounds, same receivers, same paths, scored by effect on the output instead of delivery to the receiver. Round 0 is out of the receiver-side criterion's scope by construction: its receiver *is* the output, where the two measures are the same quantity.

Of the six heads neither earlier phase found, the criterion recovers **two — 2.2 and 4.11, both previous-token heads**, the class that scored 0/2 in both prior phases. It does not rescue `0.10` (+0.042) or `5.8` (+0.008), and `9.0`/`11.9` are outside its scope entirely — not measured and found wanting, not measured at all.

**The criterion is noisier than the one it sits beside**: precision 0.64 against 0.90. It finds the previous-token heads and also admits several heads with no published role. Both are properties of the same fixed threshold.

**Robustness.** Per-group thresholds were fixed by the same rule at the same time. Only `3.7` and `4.3` depend on the more lenient pooled bar — and neither is a published head. Every published head found also clears its own group's stricter bar, so pooling produced false positives and none of the recoveries.

### Why the scores are not added together

On the like-for-like comparison the two criteria find 8/26 and 7/26, but they disagree about *which* heads, not merely how many — previous-token heads appear only in the second, several induction and name-mover heads only in the first. Merging them would report a larger number while destroying the only new information the phase produced.

A head can deliver its content to the next stage of the circuit and still leave the prediction unmoved. The two criteria take opposite views on whether that counts as being part of the circuit, and neither is wrong: explaining a behaviour argues for the output criterion, mapping a mechanism argues for the receiver-side one. **Phases 1 and 2 answered only the first while appearing to answer both.** Making that visible, rather than raising a number, is what this phase was for.

### The boundary this project has not crossed

Every round in Phases 2 and 3 was told *where to look* — that S-inhibition acts on name movers' queries, that duplicate-token information arrives as a value at S2, that induction keys live at S1+1. Those come from the paper's account of the mechanism. Which heads turned up was never constrained; the question asked of them was.

So everything so far is **guided rediscovery**: given the right question, the method finds the right components, in the right causal order. The autonomous loop this README describes has to generate the questions too. On a circuit nobody has published there is no paper to supply the receiver inputs, so a method that needs them supplied does not yet transfer — and unlike every phase so far, there would be no answer key to check the search against.

## Stack

- **Python** 3.12
- **[transformer_lens](https://github.com/TransformerLensOrg/TransformerLens)** — hooked model internals, activation caching, intervention hooks
- **PyTorch** with CUDA (see setup — a CPU-only build will work but is painfully slow)

## Status

**Phases 1–3 built and run.** Activation patching, path patching, and a pre-registered receiver-side criterion, all validated against the published IOI circuit. On the output criterion the project stands at 20/26 published heads with the paper's causal ordering reproduced; the receiver-side criterion finds a partly different set, including the two previous-token heads the output criterion cannot reach.

Not implemented: ablation, iterative pruning, and — the real barrier — any search over *receiver specifications*. Every result so far depends on being told which head input, at which position, to ask about. That is the line between guided rediscovery and the autonomous loop described above, and it is the concrete next problem.

No autonomous discovery until then. Phase 2 exists because a confident prediction from Phase 1 turned out to be wrong when tested, and Phase 3 exists because the fix Phase 2 proposed had already been seen to flatter the result. Both are the failure modes an unvalidated system pointed at an unknown circuit would produce silently.

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

## Running Phase 1

Patching is easy to get subtly wrong in ways that still produce reasonable-looking numbers, so verify the machinery first. These are known-answer tests — cases where the correct result follows from how the experiment is built, not from the model:

```bash
python scripts/check_patching.py     # expect: PATCHING OK
```

Then the full pipeline. It downloads GPT-2 small on first run and takes about 6 minutes on a laptop RTX 5060:

```bash
python scripts/run_phase1_ioi.py                        # ~6 min, activation patching
python scripts/run_phase2_paths.py                      # ~4 min, path patching
python scripts/run_phase3_receiver.py --preregister     # ~2 min, fix the threshold
python scripts/run_phase3_receiver.py                   # ~2 min, apply it
```

Each regenerates its own report, the JSON behind it, and per-head CSVs in `results/`. All runs are seeded, so they reproduce exactly. The phases chain — Phase 2 reads `phase1_results.json`, Phase 3 reads `phase2_results.json` — so run them in order on a clean checkout.

Phase 3 refuses to run without a recorded threshold, and rejects one calibrated at a different `n` or seed. That is deliberate: the point of the pre-registration is that the threshold cannot be adjusted once results exist.

## Layout

```
causal_interp/
  model.py           # model loading, device selection
  ioi.py             # IOI task: prompt pairs, corruption schemes, position indices
  interventions.py   # activation patching, path patching, sweeps, circuit narrowing
  comparison.py      # scoring a discovered head set against ground truth
  ground_truth.py    # the published IOI circuit — inert data, never derived from a run
scripts/
  check_env.py       # environment + CUDA verification
  check_patching.py  # known-answer tests for both patching methods
  run_phase1_ioi.py  # Phase 1: activation patching -> results/
  run_phase2_paths.py # Phase 2: iterative path patching -> results/
  run_phase3_receiver.py # Phase 3: --preregister fixes the threshold; main run applies it
  phase3_analysis.py # Phase 3 comparison + report, imported only by the main run
results/
  PHASE1_REPORT.md   # activation-patching comparison against Wang et al.
  PHASE2_REPORT.md   # path-patching comparison, and the combined result
  PHASE3_REPORT.md   # the two criteria side by side
  phase3_preregistration.json  # the threshold, committed before the results existed
  phase1_results.json / phase2_results.json / phase3_results.json
  head_effects_*.csv / component_effects_*.csv / path_effects_*.csv / receiver_signals.csv
```

`phase3_analysis.py` is a separate module so the `--preregister` path cannot import it: the step that fixes the threshold has no access to the code that computes a real measurement.

The separation between `ground_truth.py` and `comparison.py` is deliberate: the published circuit is hard-coded and the scoring is pure set arithmetic over it, so nothing measured in a run can influence what counts as a match.

## License

Apache 2.0 — see [LICENSE](LICENSE).
