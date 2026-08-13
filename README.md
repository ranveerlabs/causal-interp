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

## Stack

- **Python** 3.12
- **[transformer_lens](https://github.com/TransformerLensOrg/TransformerLens)** — hooked model internals, activation caching, intervention hooks
- **PyTorch** with CUDA (see setup — a CPU-only build will work but is painfully slow)

## Status

**Phase 1 built and run.** Activation patching is implemented and validated against the published IOI circuit; see the result above. Ablation and iterative pruning are not implemented, and neither is path patching — that is the next phase, and the Phase 1 result is the argument for why it comes before any autonomous discovery.

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
python scripts/run_phase1_ioi.py             # ~6 min, writes results/
python scripts/run_phase1_ioi.py --quick     # ~2 min smoke test
```

It regenerates everything in `results/`: the report, the JSON behind it, and per-head CSVs. The run is seeded, so it reproduces exactly.

## Layout

```
causal_interp/
  model.py           # model loading, device selection
  ioi.py             # IOI task: prompt pairs, corruption schemes, position indices
  interventions.py   # activation patching, sweeps, greedy circuit narrowing
  comparison.py      # scoring a discovered head set against ground truth
  ground_truth.py    # the published IOI circuit — inert data, never derived from a run
scripts/
  check_env.py       # environment + CUDA verification
  check_patching.py  # known-answer tests for the patching machinery
  run_phase1_ioi.py  # Phase 1 end to end -> results/
results/
  PHASE1_REPORT.md   # the comparison against Wang et al. (the deliverable)
  phase1_results.json
  head_effects_*.csv
  component_effects_*.csv
```

The separation between `ground_truth.py` and `comparison.py` is deliberate: the published circuit is hard-coded and the scoring is pure set arithmetic over it, so nothing measured in a run can influence what counts as a match.

## License

Apache 2.0 — see [LICENSE](LICENSE).
