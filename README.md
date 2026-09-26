# causal-interp

causal interpretability experiments that are easy to reproduce, modify, and break.

## why

I reproduced a published circuit, then changed the counterfactual used in the intervention.

**3/6 → 5/6 active heads.**

Same circuit, different counterfactual, substantially different result. That made me wonder how robust these findings are—and how easy it should be to test them.

## result

The reproduced result is the Python docstring circuit from Heimersheim and Janiak's [*A circuit for Python docstrings in a 4-layer attention-only transformer*](https://www.lesswrong.com/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only). The paper identifies six attention heads, checked here against the released ACDC manual graph.

| Counterfactual | Change | Heads recovered |
|---|---|---:|
| `random_random` (primary) | Replaces definition and docstring arguments | 3/6 |
| `random_def` | Breaks the induction match, keeps the answer | 5/6 |
| `random_answer` | Replaces the answer | 3/6 |

“Active” here means recovered by activation patching with an absolute change of at least **0.02** in the hand-built logit-difference metric. With `random_def`, heads `1.4` and `2.0` also cross that cutoff. The primary counterfactual remains `random_random`, as specified before the run; the alternate is a comparison, not a replacement headline.

Why it changes: `random_random` and `random_answer` change the answer token, so restoring routing heads does not fix the output. `random_def` keeps the answer but breaks the pointer to it, making those routing heads visible to the metric. This is evidence that measured recovery depends on the counterfactual—not evidence that one corruption is universally better.

## reproduce

The measured run used 128 prompts per corruption, seed 0, and `attn-only-4l` (`Attn_Only_4L512W_C4_Code`): 4 layers, 8 heads per layer, width 512, attention-only. The cutoff is 0.02. See the [full Phase 7 report](results/PHASE7_REPORT.md) and [run plan](results/PHASE7_PLAN.md) for the protocol, other metrics, and detailed findings.

Requires Python 3.12 and a CUDA 12.8-compatible PyTorch install; dependencies are pinned in `requirements.txt`.

```sh
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_phase7_docstring.py --stage sweep
python scripts/run_phase7_docstring.py --preregister
python scripts/run_phase7_docstring.py
```

The last command runs the experiment. `--quick` uses 16 prompts for a smoke test, not for comparing with the reported result. `--report-only` rebuilds the report from saved output. GPU runtime depends on hardware; the recorded setup and timing are in the report.

## limits

This is one small model, one task, and one seeded prompt set—not evidence that the same effect generalizes. “Recovered” is an operational threshold, not a claim that a head is necessary or sufficient. The paper's circuit labels guide where to inspect; see the report for the rest of the pipeline's findings and failure modes.

## more

- [Experiments and synthesis](SYNTHESIS.md)
- [Other reproduced circuits](results/)
- [Apache 2.0 license](LICENSE)
