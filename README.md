# causal-interp

patch heads in small transformers, check whether the published circuits are actually
there

| circuit | model | published | found |
|---|---|---|---|
| IOI | GPT-2 small | 26 | 20 |
| greater-than | GPT-2 small | 7 | 7 |
| docstring | attn-only-4l | 6 | 3 |

- [Interpretability in the Wild](https://arxiv.org/abs/2211.00593)
- [How does GPT-2 compute greater-than?](https://arxiv.org/abs/2305.00586)
- [a circuit for Python docstrings](https://www.lesswrong.com/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only)

docstring sits at 3/6. seems to be the counterfactual. The published one swaps out the
answer token, so heads whose whole job is routing attention have nothing left to move. a
different counterfactual gets 5/6, same code, same model. only worked that out from reading
the paper though, which is kind of the problem.

three goes at getting the pipeline to spot that on its own. none of them worked. last one
resampled the whole thing ten times:

```
$ python scripts/phase11_posthoc.py

  circuit/scheme                     rho(sd,|mean|)  sd pub / sd rest
  docstring/random_random                     0.585              3.14
  docstring/random_def                        0.577              4.88
  greater_than/yy01                           0.780             17.51
  greater_than/xx_mismatch                    0.759             12.30
```

replication sd tracks effect size in all nine rows and published heads sit somewhere
between 2.4x and 17.5x above the rest, so dividing by it squashes the top of the ranking.
long version in [SYNTHESIS.md](SYNTHESIS.md), numbers in [results/](results/)

so phase 12 stopped building statistics on top of patching scores and tried a different
method entirely. [causal scrubbing](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing),
resample-ablate everything outside a claimed circuit, run on real prompts, no counterfactual
pair anywhere. The published 26 pass, recovering 1.02 of the logit difference. so do 25 of
200 random 26-head sets, and 8 of those beat it. drop 12 of the 26 and it still scores 0.911

one bit of it did something new. leave-one-out from the published circuit puts the
previous-token heads `4.11` and `2.2` first and third, and both score about 0.0025 under
either of phase 1's counterfactuals:

```
head    loo drop patch |eff| p1 found  class
4.11       0.316      0.0028       NO  previous token
9.9        0.205      0.7820      yes  name mover
2.2        0.177      0.0023       NO  previous token
8.10       0.126      0.2533      yes  s-inhibition
...
10.7      -0.375      0.5114      yes  negative name mover
```

Spearman against patching magnitude is 0.240 across all 144 heads, so its looking somewhere
else. twelve phases in thats the first signal here that disagrees w magnitude in a direction
the answer key backs up. one circuit, six heads, untested on the other two

## gotchas

- all 576 head-position cells before S2 come out zero under `s2_swap`. identical inputs
  either side, so, yeah. Spent a while assuming the sweep was broken
- `mlp_out` on a model with no MLPs. 28 zeros, no error, nothing complains
- 2 of the 32 fixture prompts had years GPT-2 splits as `[" 150", "9"]` and not
  `[" 15", "09"]`. century columns quietly stopped tying, and prompts started coming out as
  `year 11245 to the year 14`
- 2 prompts recover 6/7, 32 recover 3/7. more data made it worse
- both circuits on one 8GB card slowed each other ~3x. one at a time
- Smart App Control blocks `pyarrow`'s unsigned `lib.cp312-win_amd64.pyd`, and
  `transformer_lens` reaches it through `datasets` at import. `check_env.py` says
  `transformer_lens is not installed` which is not what happened

## setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_env.py      # ENVIRONMENT OK
```

requirements pins the CUDA 12.8 index. default PyPI torch on Windows is CPU-only, and
50-series cards have no kernels in anything older.

## running

```bash
python scripts/check_patching.py
python scripts/check_patching_docstring.py
python scripts/check_schemes.py
python scripts/check_induction.py

python scripts/run_phase1_ioi.py                        # ~6 min
python scripts/run_phase2_paths.py                      # ~4 min
python scripts/run_phase3_receiver.py --preregister
python scripts/run_phase3_receiver.py
python scripts/run_phase4_search.py                     # ~31 min
python scripts/run_phase5_scoping.py

python scripts/run_phase6_greater_than.py --stage sweep
python scripts/run_phase6_greater_than.py --preregister
python scripts/run_phase6_greater_than.py               # ~15 min

python scripts/run_phase7_docstring.py --stage sweep
python scripts/run_phase7_docstring.py --preregister
python scripts/run_phase7_docstring.py

python scripts/run_phase8_multischeme.py --circuit docstring
python scripts/run_phase8_multischeme.py --circuit greater_than   # ~17 min

python scripts/phase9_characterize.py
python scripts/run_phase9_calibration.py --circuit docstring
python scripts/run_phase9_calibration.py --circuit greater_than
python scripts/run_phase9_calibration.py --circuit ioi   # ~18 min

python scripts/phase10_characterize.py
python scripts/run_phase10_autotask.py --fixture frame_same --induction plan
python scripts/run_phase10_autotask.py --stage ksweep
python scripts/run_phase10_autotask.py --stage pairs

python scripts/run_phase11_resample.py --circuit docstring       # ~12 min
python scripts/run_phase11_resample.py --circuit greater_than    # ~50 min
python scripts/phase11_analysis.py

python scripts/run_phase12_scrub.py                             # ~80 min
python scripts/phase12_report.py
python scripts/phase12_posthoc.py
```

seeded, and they chain, so on a clean checkout run them in order. `--report-only` gets you
the writeup without redoing the sweep. on 7 and 8 itll clobber a stale git diff though.

`search.py`, `pipeline.py`, `agreement.py`, `schemes.py`, `induction.py` and
`autotask.py` cant import `ground_truth`. the runners check at startup.

Apache 2.0
