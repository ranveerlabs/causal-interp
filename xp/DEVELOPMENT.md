# xp development report

Built and smoke-tested only. No registered experiment, pretrained model inference,
or historical-result reanalysis was run during development. Existing Phase 1–11
payloads and reports were left unchanged. Existing unrelated working-tree changes
are excluded from the implementation commit.

## Method

Sources read before implementing the scrub:

- [Redwood causal scrubbing](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-redwood-research)
- [Redwood appendix](https://www.lesswrong.com/posts/kcZZAsEjwrbczxN2i/causal-scrubbing-appendix)
- [Redwood reference implementation directory](https://github.com/redwoodresearch/rust_circuit_public/tree/master/python/rust_circuit/causal_scrubbing)
- [Official PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/)
- [Pinned TransformerLens dependency metadata](https://pypi.org/pypi/transformer-lens/3.7.1/json)

The implementation tests the explicit coarse output-contribution graph in PLAN.
It recursively resamples selected-head inputs within paired semantic classes and
shares an unconditional donor for omitted contributions. Non-attention remainder
and upstream internals are retained. It does not test an isolated full circuit or
claim that an ordinary mean ablation is causal scrubbing. The generic recursive
sampler has separate tests for conditional interchangeability, donor variation and
shared nuisance donors.

Few schemes, dependent heads, three tasks and a finite donor pool limit inference.
The class-level t approximation and scheme permutation exchangeability are working
assumptions. See PLAN for exact distributions and confounders.

## Frozen analyses

173 finite units. 40 confirmatory hypotheses, Bonferroni M=40 and alpha=0.05:

| Tests | Statistic and decision |
|---|---|
| 31 preservation | R=(S-F)/(C-F), anchor mean(C-F)>1e-4, R>=0.8, one-sided t on S-0.8C-0.2F across 128 semantic classes, adjusted p<0.05 |
| 1 IOI closeness | Spearman of Jaccard versus R over 15 recovered candidates, positive rho and one-sided 20,000-permutation p with +1 correction, adjusted p<0.05 |
| 3 Phase 11 P1 | Original exact paired Wilcoxon and median AUC gain>=0.05 versus Bm, with original verdict and additional xp correction |
| 5 Phase 11 P2 | Original 20,000 max-stat permutations, abs rho>=0.7 and original within-circuit sign criterion, with additional xp correction |

Preservation includes 16 IOI, 7 greater-than and 8 docstring candidates. The three
published candidates predict preservation. Other candidates predict no demonstrated
preservation. Closeness predicts positive rho. P1 and P2 predict no separating
candidate. Missing tests retain their place in M. FAILURE diagnostics do not create
an additional confirmatory rejection. Independent donor seeds 1–2 measure stability,
and 3–4 are the finite extension. No seed selection or pooled primary-test retuning.

Phase 11's 13-row extension reuses the original analysis functions and statistics.
The original nine-row result is not replaced. Correlated errors, phi, n_eff,
Cantelli and majority error frequency are descriptive and use stored inputs only.
The CPU worker rejects model-library imports before executing any unit.

The Phase 1 environment gate reproduces its clean and corrupted s2_swap mean logit
differences, n=128, seed 0, with maximum absolute discrepancy <=0.001 logits.
Failure blocks the entire workload. The original preregistration was committed as
`6547d36` before implementation. Pre-execution wording was completed to explicitly
name the non-published predictions and distinguish the descriptive FAILURE bound.
No hypotheses, test count, thresholds or results were changed by those clarifications.

## Checks completed

`xp/smoke.py` passed all 13 tests in the temporary CPU environment and all 13 with
`XP_SMOKE_GPU` selecting the actual RTX 5060. The GPU mode uses a 32-float fp32 CUDA
driver kernel inside synthetic queue units. It needs no pretrained model weights.
The model test uses a random two-layer, width-16 transformer on CPU.

| Check | Evidence |
|---|---|
| End-to-end queue | Synthetic units claimed, executed, validated and published, real fp32 CUDA kernels in GPU mode |
| Concurrency | Two simulated GPU workers share a queue, each successful unit has one claim |
| Kill/resume | Launcher killed during a unit, Linux parent-death signal ends own worker, incomplete result absent, resume completes |
| Failure/retry | Injected one-failure unit succeeds on retry, two-failure unit becomes FAILED, dependent BLOCKED, independent work completes |
| STOP | Stops claims, safe active work completes, partial digest stays useful, removing STOP permits resume |
| Time budget | Leaves pending work and no active claims after a graceful stop |
| Plan refusal | Untracked, staged-but-uncommitted and modified plans rejected, committed plan accepted in a temporary git repository |
| Atomic output | Injected rename interruption leaves no final payload, stale temporary output recovered, lost claim cannot overwrite a completed payload |
| Resume validity | Completed payload timestamps unchanged, altered-checksum payload requeued and repaired |
| Reproduction paths | Actual launcher control flow tested with simulated gate PASS and failure, failure starts no experimental supervisor |
| Occupancy/draining | Own PIDs ignored, simulated foreign PID drains our worker and returns its unit to pending |
| CPU-only errors | Attempted torch import fails under worker guard, tiny stored fixture verifies phi=1, n_eff=1 and majority error=0.5 |
| Scrub math | Tiny random transformer reconstructs original scores within 0.001 and produces identical results for repeated donor seeds |

Python compilation, shell syntax and whitespace checks also passed. Test-only
fixture errors found and fixed: temporary git identity missing, and an incomplete
mock tokenizer missing padding_side. These were not experiment failures.

The sandbox initially hid the NVIDIA driver. An unsandboxed check detected the RTX
5060 Laptop GPU, 8151 MiB, driver 615.71.09. The initial Chrome compute context was
excluded. After the user closed it, the compute-process query was empty and the
small CUDA tests ran. No foreign process was signalled or changed.

Test environment: Python 3.14, torch 2.11.0+cpu, TransformerLens 3.7.1, scipy 1.16.3.
Actual CUDA checks used the installed NVIDIA driver directly. This is separate from
the planned server Python 3.10–3.12 / torch 2.7.1+cu126 / scipy 1.15.3 environment.

## Still to verify on LambdaV

- Installation of the constrained cu126 environment on the server and actual sm_60 execution.
- Detected P100 count, memory and driver compatibility. The code expects four 16 GB P100s but accepts a compatible free subset and records differences.
- Frozen pretrained weights/tokenizers, real Phase 1 reproduction, and all published-circuit gates.
- Actual task variant coverage, real GPU memory requirements, throughput and runtime estimates.
- Server-local filesystem locking, configured git identity and real experiment checkpoint commits.
- Real foreign-workload contention. Its control logic was simulated without launching or disturbing another user's job.

The full experiment suite has deliberately not been used as a smoke test. No P100
compatibility or scientific preservation result is claimed from the RTX checks.

## Files

Added under xp: setup/run/status shell entry points, queue/state helpers, supervisor,
worker, hardware checks, experiment adapters, recursive sampler, digest generator,
smoke suite and CUDA smoke helper, PLAN, DIGEST, RUNBOOK, this report, torch
constraints, runtime ignore rules and the logs directory. Changed only
`scripts/check_env.py` outside xp, to validate requested and detected architectures.
The project requirements file and historical outputs were not edited.

## Server startup

```sh
ssh YOUR_USER@LambdaV
git clone https://github.com/ranveerlabs/causal-interp.git
cd causal-interp
./xp/setup.sh
tmux new -s xp './xp/run.sh'
```

Use the actual configured git identity for local checkpoints. Detach with Ctrl-b d.
Reconnect with `./xp/status.sh`, read `xp/DIGEST.md`, and use `touch xp/STOP` to stop.
Remove STOP manually, then rerun `./xp/run.sh` to resume. Review and transfer server
commits before pushing from your development machine. The runner never pushes.
