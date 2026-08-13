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

So Phase 1 runs the system against a small open model with an **already-published circuit** — GPT-2 small's IOI (indirect object identification) circuit, or its induction heads. Both are documented in the literature, which means there is a known answer. The question Phase 1 has to settle is narrow and concrete:

> Do the system's autonomous conclusions match the published ground truth?

Only once the method demonstrably rediscovers what is already known does it earn the right to be pointed at anything unfamiliar.

## Stack

- **Python** 3.12
- **[transformer_lens](https://github.com/TransformerLensOrg/TransformerLens)** — hooked model internals, activation caching, intervention hooks
- **PyTorch** with CUDA (see setup — a CPU-only build will work but is painfully slow)

## Status

**Fresh restart.** Phase 1 is not yet built — the repository currently contains the environment setup and a scaffold marking the intended entry points. The intervention primitives in `causal_interp/interventions.py` raise `NotImplementedError`.

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

## Layout

```
causal_interp/
  model.py           # model loading, device selection
  interventions.py   # activation patching, ablation, circuit pruning
scripts/
  check_env.py       # environment + CUDA verification
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
