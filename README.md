# causal-interp

A set of experiments on whether circuit findings hold up when the model, task or
counterfactual changes.

## why

I reproduced the published docstring circuit, then switched the counterfactual. Same model and code, just a different thing getting corrupted. I went back to the paper to work out why.

**3/6 -> 5/6 active heads.**

`random_random` and `random_answer` replace the answer token, so restoring routing heads doesn't fix the output. `random_def` keeps the answer and breaks the pointer to it. Now those heads can move the metric.

## result

This is the Python docstring circuit from Heimersheim and Janiak's [*A circuit for Python docstrings in a 4-layer attention-only transformer*](https://www.lesswrong.com/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only). The paper names six heads. I checked those against the released ACDC manual graph.

| Counterfactual | Change | Heads recovered |
|---|---|---:|
| `random_random` (primary) | Replaces definition and docstring arguments | 3/6 |
| `random_def` | Breaks the induction match, keeps the answer | 5/6 |
| `random_answer` | Replaces the answer | 3/6 |

“Active” is my cutoff: activation patching has to change the hand-built logit-difference metric by at least **0.02**. Heads `1.4` and `2.0` cross it too with `random_def`. `random_random` was the planned run. `random_def` is the comparison.

The active count changes with the counterfactual. That doesn't mean `random_def` is the better corruption for every test.

## reproduce

I ran 128 prompts per corruption with seed 0 on `attn-only-4l` (`Attn_Only_4L512W_C4_Code`): 4 layers, 8 heads per layer, width 512, attention-only. Cutoff is 0.02. The [Phase 7 report](results/PHASE7_REPORT.md) has the protocol and other metrics. The [run plan](results/PHASE7_PLAN.md) has the setup.

You need Python 3.12 and a CUDA 12.8-compatible PyTorch install. Dependencies are pinned in `requirements.txt`.

```sh
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_phase7_docstring.py --stage sweep
python scripts/run_phase7_docstring.py --preregister
python scripts/run_phase7_docstring.py
```

The last command runs the experiment. `--quick` uses 16 prompts to see if it runs, not to compare with the result. `--report-only` rebuilds the report from saved output. GPU time depends on the card. I put the setup and timing in the report.

## limits

It's one task on one small model with one seeded prompt set. The 0.02 cutoff is mine. “Recovered” just means the patch crossed it, not that a head is necessary or sufficient, or that the result carries over to another model. The report has the other results and the parts that didn't work.

## more

- [Experiments and synthesis](SYNTHESIS.md)
- [Other reproduced circuits](results/)
- [Apache 2.0 license](LICENSE)
