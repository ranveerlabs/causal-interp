# causal-interp

An autonomous system for discovering and **causally validating** computational mechanisms in neural networks.

## Where this stands

Validating the method against GPT-2 small's IOI circuit, published in Wang et al. (2022), [*Interpretability in the Wild*](https://arxiv.org/abs/2211.00593) — 26 attention heads in 7 classes, so there is a known answer to check against.

| phase | method | result |
|---|---|---|
| [1](#phase-1--activation-patching) | activation patching | **20/26** — recovers every class acting directly on the output, misses every class acting through another head |
| [2](#phase-2--path-patching) | path patching | **20/26** — Phase 1's prediction that this would close the gap was **wrong**; precision 0.71 → 0.90, and the paper's causal *ordering* recovered |
| [3](#phase-3--a-pre-registered-receiver-side-criterion) | pre-registered receiver-side criterion | recovers **2 of the 6** still-missing heads (both previous-token), at precision 0.64 — a *different* definition of "found", reported beside the first rather than merged |
| [4](#phase-4--searching-for-receiver-specifications) | search for receiver specifications | **16 of 17** scoreable specifications recovered without being told them — and the unlabelled search finds the same token positions the labelled one uses |

Across both definitions of "found", 22 of the 26 published heads have been recovered by something. That figure spans two criteria that disagree about which heads count, and [Phase 3 explains why they are not added together](#why-the-scores-are-not-added-together) — it is not one method's recall.

**Not implemented:** ablation, iterative pruning, and — now the largest remaining gap — any construction of the *task and its counterfactual*. Phase 4 removed the need to be told which head input at which position to interrogate; the templates, the corruption schemes and the metric are all still designed by hand from knowledge of what the model is doing.

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

Only once the method demonstrably rediscovers what is already known does it earn the right to be pointed at anything unfamiliar. Two phases have already failed a prediction they made about themselves, which is the argument for keeping the answer key in reach a while longer.

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
python scripts/check_patching.py     # expect: PATCHING OK
```

Then the pipeline. It downloads GPT-2 small on first run; timings are for a laptop RTX 5060:

```bash
python scripts/run_phase1_ioi.py                        # ~6 min, activation patching
python scripts/run_phase2_paths.py                      # ~4 min, path patching
python scripts/run_phase3_receiver.py --preregister     # ~2 min, fix the threshold
python scripts/run_phase3_receiver.py                   # ~2 min, apply it
python scripts/run_phase4_search.py                     # ~31 min, receiver-spec search
```

Phase 4 is by far the longest — it is two exhaustive grids, 9,936 forward passes. `--report-only` rebuilds its report from the stored CSVs without repeating the sweep.

Each regenerates its own report, the JSON behind it, and per-head CSVs in `results/`. All runs are seeded, so they reproduce exactly. The phases chain — Phase 2 reads `phase1_results.json`, Phase 3 reads `phase2_results.json` — so run them in order on a clean checkout.

Phase 3 refuses to run without a recorded threshold, and rejects one calibrated at a different `n` or seed. That is deliberate: the point of the pre-registration is that the threshold cannot be adjusted once results exist.

## Layout

```
causal_interp/
  model.py           # model loading, device selection
  ioi.py             # IOI task: prompt pairs, corruption schemes, position indices
  interventions.py   # activation patching, path patching, sweeps, circuit narrowing
  comparison.py      # scoring a discovered head set against ground truth
  search.py          # Phase 4 receiver-spec search — must never import ground_truth
  ground_truth.py    # the published IOI circuit — inert data, never derived from a run
scripts/
  check_env.py            # environment + CUDA verification
  check_patching.py       # known-answer tests for both patching methods
  run_phase1_ioi.py       # Phase 1: activation patching -> results/
  run_phase2_paths.py     # Phase 2: iterative path patching -> results/
  run_phase3_receiver.py  # Phase 3: --preregister fixes the threshold; main run applies it
  phase3_analysis.py      # Phase 3 comparison + report, imported only by the main run
  run_phase4_search.py    # Phase 4: exhaustive receiver-spec search -> results/
  phase4_report.py        # Phase 4 report, kept out of the search module
results/
  PHASE1_REPORT.md / PHASE2_REPORT.md / PHASE3_REPORT.md / PHASE4_REPORT.md
  PHASE4_SEARCH_SPACE.md       # the space and budget, committed before the search code
  phase3_preregistration.json  # the threshold, committed before the results existed
  phase1_results.json / phase2_results.json / phase3_results.json / phase4_results.json
  head_effects_*.csv / component_effects_*.csv / path_effects_*.csv
  receiver_signals.csv / receiver_search_*.csv
```

Three separations are deliberate, and each one makes a claim checkable rather than promised. `ground_truth.py` holds the published circuit as inert data while `comparison.py` scores against it with pure set arithmetic, so nothing measured in a run can influence what counts as a match. `phase3_analysis.py` is a separate module so the `--preregister` path cannot import it. And `search.py` must never import `ground_truth` — Phase 4 asserts this at startup, because a search that can see the answer key is not a search.

## License

Apache 2.0 — see [LICENSE](LICENSE).
