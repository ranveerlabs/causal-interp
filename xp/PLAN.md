# xp preregistration

Few schemes per circuit, dependent heads, and only three tasks limit generalization.
Scrub conclusions depend on the specification and donor distribution. These tests
are conditional on the finite prompt population below. They do not validate a full
algorithmic account of IOI, greater-than, or docstrings. Historical Phase 1–11 files
remain unchanged. No real experiment has been run for this implementation.

## Method and scope

[Redwood's original description](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-redwood-research)
defines a hypothesis using a model graph G, interpretation graph I, and correspondence c.
Interpretation values define interchangeability classes. Scrubbing recursively samples
matching inputs for important parents and unrestricted inputs for unimportant parents,
sharing one unrestricted donor across unimportant parents of a node. The comparison is
expected behavior before versus after these interventions. Preservation supports the
specified equivalences, failure challenges them. Neither establishes uniqueness or truth.

Resample ablation uses a sampled activation, preserving its marginal variation. Mean
ablation removes that variation. The repository's recovery patching moves clean
activations into corrupted runs. Neither alone tests recursive interchangeability.
[Redwood's reference implementation](https://github.com/redwoodresearch/rust_circuit_public/tree/master/python/rust_circuit/causal_scrubbing)
provides the correspondence and conditional-sampler vocabulary used here.

[Redwood’s appendix](https://www.lesswrong.com/posts/kcZZAsEjwrbczxN2i/causal-scrubbing-appendix)
permits exact graph rewrites and warns that coarse explanations can be uninformative.
The specification below uses that freedom and states what it leaves untested.

### Exact graph tested here

This is a coarse output-contribution hypothesis, chosen to make the head-set question
explicit and computationally finite. G has input nodes, one whole-prefix computation
node for each head's END contribution z @ W_O, a remainder node, an addition node,
and the original final layer norm and unembedding. The remainder is final residual
minus the sum of these head contributions. It includes embeddings, biases and MLPs.
Each contribution is computed by the original model on its input, including all its
upstream attention. This graph is an exact re-expression of the original output.

I maps selected heads, remainder, addition, normalization and unembedding to G.
A selected head's interpretation value is its semantic class below. Its input child
has the identity feature, so recursively evaluating that child uses the chosen donor
unchanged. Remainder has the identity feature and stays on the reference input.
All omitted heads are unimportant parents of addition and share one unconditional
donor. Selected heads independently sample within the reference semantic class.
An explicit recursive sampler implements these rules, including singleton identity
classes. It never replaces an activation with a mean.

This coarse graph deliberately makes no claim about paths inside a head computation.
Information through upstream excluded heads and MLPs can survive. A pass means
preservation of this conditional output decomposition, not sufficiency of the listed
heads as an isolated circuit. The published 26-head test is a sanity gate for this
specification, not a reproduction of every mechanistic claim in the IOI paper.

### Frozen donor population

Generate 128 clean references with the repository constructors, seed 1701, primary
scheme, model revisions below. Each reference has one second, length-preserving
semantic variant. IOI changes place and object to the next tokenizable entry in their
existing lists, retaining names, order and template. Greater-than changes only its
noun to the next tokenizable noun. Docstring swaps two distinct description-only
noun token IDs, excluding all argument IDs. Failure to find a valid variant is an
error, never a fallback to another class. The class is the pair of reference and
variant. This defines a finite distribution of 256 prompts, uniform over classes
and the two variants. Identity means the complete token sequence.

Evaluate both variants of each class, average within class and over 8 independent
donor draws. The observational unit is the semantic class, not a head or draw.
Unconditional donors are uniform over all 256 prompts, including the reference.
Head contributions always come from the donor's own END, so variable prompt lengths
are not padded into a fictitious matching position. All calculations use fp32,
eval mode, no autocast, TF32 disabled, fixed batches of 8 during cache construction.

## Confirmatory family

There are 40 confirmatory tests: 31 candidate preservation tests, one IOI
closeness test, three Phase 11 P1 candidates and five Phase 11 P2 candidates.
Bonferroni uses M = 40, alpha = 0.05, so each raw p must be < 0.00125.
Missing, failed, gated and inconclusive tests retain their places in the denominator.
No repetition, extension or descriptive diagnostic adds a hypothesis or a new decision.

### Preservation, 31 tests

Let C, F, S be per-class task scores for intact, no-selected-head, and candidate
scrubs. F keeps the same intact remainder and shares its donor across every head.
Use the repository's task logit_diff, including the probability-difference definition
for greater-than. Report R = mean(S-F) / mean(C-F), without clipping.
The anchor is invalid unless mean(C-F) > 1e-4. Invalid anchors are inconclusive.

The operational tolerance is at most 20% loss of the clean-to-floor span, R >= 0.8.
This is a declared practical preservation policy, not a universal Redwood threshold
or an estimated property of these tasks. Test H0: E[S - 0.8 C - 0.2 F] <= 0 with a
one-sided Student t test across 128 class means, df 127. PASS requires a valid anchor,
R >= 0.8, and Bonferroni p < 0.05. An upper one-sided simultaneous t bound below zero
is FAILURE. All other cases are INCONCLUSIVE. Zero variance gives p 0 only for a
strictly positive margin, p 1 for a negative margin, and p 0.5 for zero.
Only PASS rejects a confirmatory null. The FAILURE bound is a descriptive
non-preservation diagnostic, not an additional family-controlled negative test.
Approximate t inference depends on class sampling and distributional regularity.
The finite donor pool couples classes, so coverage is approximate, not guaranteed.

The three published sets predict PASS directionally. For each of the 28 other
candidates, the prediction is the operational null: no demonstrated preservation
under the registered criterion. This predicts a non-PASS decision, not proof that
its population effect is zero. These predictions remain fixed even if conservative. IOI recovered comparisons
require published IOI PASS. Cross-task candidates likewise require their own
published PASS. Any non-PASS gate blocks dependents, with no criterion adjustment.
The IOI robust and dependent sets use Phase 9's stored shared-0.02 reconstruction
of the Phase 8 scheme comparison: no Phase 8 IOI payload exists. This provenance
limitation is reported rather than fabricating a historical IOI Phase 8 run.
Recovered Phase 1 headline sets, Phase 2 discovered sets, all three Phase 3 criteria,
and Phase 4 semantic receiver and confirmed sender sets are enumerated below.
Phase 4 uses the existing 0.11 signal threshold. All head IDs are frozen below.

### Closeness, one test

Across the 15 non-published IOI candidates, Spearman rho between Jaccard overlap
with published heads and R. Prediction: positive. Use 20,000 permutations of R,
seed 4242, one-sided p = (1 + count(null >= observed)) / 20001, then Bonferroni.
Require every candidate and valid anchor. Report raw pass/fail alongside R.
Repeated sets and overlapping schemes are dependent, so exchangeability is only
a working assumption. No claim of IID heads or broad population generalization.

### Phase 11 table, eight tests

Run IOI discovery for seeds 0 through 9, n=128, all four registered schemes,
logit_diff, max-absolute position collapse, each Phase 9 theta unchanged. Combine
with the nine stored rows into 13. Reuse the committed phase11_analysis.py functions
for S1-S4, B0/Bm, T1-T5, P1 exact Wilcoxon and P2 20,000 max-stat permutations,
seed 20260823. Preserve the P1 median gain >= 0.05 and P2 abs rho >= 0.7 plus
within-circuit sign criterion. Preserve the original verdicts and p values in the
payload, with an additional xp Bonferroni verdict. The original nine-row results
are not overwritten. P3 adds no IOI data and is not rerun or recounted.
Predictions: no P1 candidate beats Bm, and no P2 candidate clears all bars.
These are a newly registered extension to 13 rows, not a retroactive claim that
Phase 11 originally preregistered IOI. Within-circuit dependence still limits tests.

## Descriptive work, no additional tests

Scrub stability repeats identical statistics with donor seeds 1 and 2 after core
measurements, reporting R range, sample SD and fraction matching seed-0 classification.
These repeat classifications are diagnostics, not independent confirmatory verdicts.
The finite extension repeats donor seeds 3 and 4 only after all core units terminate.
It does not pool them into the primary seed-0 test or select the best seed.

Correlated errors read committed Phase 8–11 payloads only. For each task and threshold
variant (per-scheme Phase 9 theta, shared 0.02), E[s,h] is the XOR of discovery
(abs effect >= theta) and published membership. Report every scheme-pair phi,
mean phi rho and n_eff = n / (1 + (n-1)rho). Constant error columns make phi undefined:
report null and exclude that pair from the mean, listing the number excluded. If no
pairs remain or the denominator is nonpositive, n_eff is undefined. Negative rho
may produce n_eff > n: retain it with the caveat that unequal marginals prevent an
exact independent-sample interpretation.

For a uniformly selected head, X is the mean of scheme error indicators. Compute
its exact finite-population mean mu and variance v including cross-covariances.
A strict majority requires X >= a = (floor(n/2)+1)/n. Cantelli gives v/(v+(a-mu)^2)
when mu < a, otherwise the trivial bound 1. For v=0 and mu<a it is 0. This is a bound
on this finite head population, not an uncertainty interval or an independence claim.
Also report the observed strict-majority error fraction. Do not substitute n_eff
into Cantelli. Predict positive rho directionally, descriptive only.
CPU workers prohibit importing torch, transformer_lens, tensorflow, jax and the
model package and assert zero model imports at exit. Inputs are frozen git blobs.

## Reproducibility, resources and stopping

Before any work unit, reproduce the committed Phase 1 s2_swap clean and corrupted
mean logit differences for all 128 prompts, seed 0. Compare maximum absolute error
to 1e-3 logits. This absolute tolerance permits small fp32 kernel/reduction drift
across CUDA builds while remaining far below the historical ~7.3-logit span.
It does not guarantee effect-level agreement. No historical value is replaced.
Failure stops the entire queue, records expected and observed values, and requires
investigation without automatic tolerance changes. Gate runs every launcher start.

A separate short synthetic fp32 matrix probe estimates throughput before the gate.
It is not a task dataset or a real work unit. Estimates use conservative initial
costs, scaled by probe time, then replace estimates with observed unit times.
Model download time and disk use are not known precisely. Cache size is reported
separately. At least one compatible free GPU is required before the gate.

Target: four 16 GB Tesla P100 cards, dynamically detected. Setup pins
[torch 2.7.1 from the official cu126 index](https://pytorch.org/get-started/previous-versions/)
and requires its reported architecture list to contain sm_60. This check and an
actual allocation are required on LambdaV, local RTX tests cannot establish it.
requirements.txt's conflicting torch line and wheel-index directive are explicitly
filtered in an xp-local requirements copy, with torch constrained throughout.
TransformerLens 3.7.1 is retained, its published dependency allows torch >=2.6.

Only xp-owned workers receive signals. Any foreign compute process, or loss of
process-query visibility, drains and terminates our unit, removes temporary output,
and returns it to pending. No GPU is reclaimed in that invocation. Own workers are
identified by PID plus Linux process start time, not usernames or VRAM usage.
No MPS or MIG sharing is assumed. Monitoring occurs every second. A race with a
foreign launch cannot be eliminated, but our allocation is yielded promptly.

Stop on STOP, time budget, SIGINT, SIGTERM, exhausted queue or failed reproduction.
Graceful stops finish in-flight units unless contention requires yielding. SIGKILL
leaves claims recoverable. Failures retry once with identical settings then become
FAILED. Failed or non-PASS gate dependencies become BLOCKED. Extension units wait
for all core units to reach DONE, FAILED or BLOCKED. Local server checkpoints occur
only when all core units of an experiment are terminal and its digest is current.
There is no server push. STOP must be removed manually before resume.

## Frozen manifest

The following JSON is the executable specification. Source files and stored inputs
are read from source_commit into an isolated xp-local snapshot, avoiding unrelated
working-tree edits. PLAN must match HEAD and the index before experimental execution.
Runner code must also be committed and unchanged. A changed plan or executable
fingerprint cannot reuse an existing queue. Every output records plan and code hashes.
Models are pinned to the listed upstream revisions. No dynamic hypotheses are added.

```json
{
  "version": 1,
  "source_commit": "2b4d048e609cc1b9428335db264544a41ceef3b8",
  "prompts": 128,
  "draws": 8,
  "preservation": 0.8,
  "alpha": 0.05,
  "tests": 40,
  "sets": {
    "ioi": {
      "published": [
        "0.1",
        "0.10",
        "10.0",
        "10.1",
        "10.10",
        "10.2",
        "10.6",
        "10.7",
        "11.10",
        "11.2",
        "11.9",
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "5.8",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.0",
        "9.6",
        "9.7",
        "9.9"
      ],
      "robust": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "dependent": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "2.8",
        "3.0",
        "3.4",
        "4.0",
        "4.3",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "5.9",
        "6.0",
        "6.6",
        "6.8",
        "6.9",
        "8.6",
        "9.3",
        "9.4",
        "9.7",
        "9.8",
        "10.1",
        "10.2"
      ],
      "scheme_s2_swap": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.0",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "scheme_abc": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "scheme_random_vocab_s2": [
        "3.0",
        "4.3",
        "5.5",
        "6.0",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "scheme_random_vocab_any": [
        "2.8",
        "3.0",
        "4.0",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "6.8",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.8",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "recovered1_s2_swap": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.0",
        "6.6",
        "9.3",
        "9.4"
      ],
      "recovered1_abc": [
        "0.1",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "0.3",
        "0.4",
        "0.5"
      ],
      "recovered2_s2_swap": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "recovered2_abc": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "recovered3_logit_all_rounds": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.6"
      ],
      "recovered3_logit_rounds_1plus": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "3.4",
        "6.6"
      ],
      "recovered3_receiver_side": [
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "7.9",
        "8.6",
        "8.10",
        "3.7",
        "4.3",
        "4.7",
        "5.6"
      ],
      "recovered4_semantic": [
        "10.0",
        "10.7",
        "11.10",
        "3.0",
        "5.5",
        "7.9",
        "8.10",
        "8.6",
        "9.7",
        "9.9"
      ],
      "recovered4_senders": [
        "10.0",
        "10.10",
        "10.7",
        "3.0",
        "5.5",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.6",
        "9.9"
      ]
    },
    "greater_than": {
      "published": [
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.11",
        "8.8",
        "9.1"
      ],
      "robust": [],
      "dependent": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.7",
        "4.10",
        "4.11",
        "5.0",
        "5.1",
        "5.5",
        "5.6",
        "6.8",
        "6.9",
        "6.11",
        "7.0",
        "7.6",
        "7.10",
        "8.5",
        "8.6",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "scheme_yy01": [
        "0.1",
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "scheme_xx_mismatch": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.10",
        "4.11",
        "5.5",
        "6.9",
        "7.10",
        "8.5",
        "8.11",
        "9.1",
        "10.7"
      ],
      "scheme_random_vocab_yy": [
        "5.0",
        "6.9",
        "8.6"
      ],
      "scheme_random_vocab_any": [
        "4.7",
        "4.11",
        "5.6",
        "6.8",
        "6.11",
        "7.0",
        "7.6",
        "8.5",
        "8.6",
        "9.1"
      ]
    },
    "docstring": {
      "published": [
        "0.5",
        "1.2",
        "1.4",
        "2.0",
        "3.0",
        "3.6"
      ],
      "robust": [
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.6"
      ],
      "dependent": [
        "0.0",
        "0.2",
        "0.4",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.2",
        "3.3",
        "3.4",
        "3.5"
      ],
      "scheme_random_random": [
        "0.0",
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "scheme_random_def": [
        "0.1",
        "0.2",
        "0.5",
        "1.0",
        "1.1",
        "1.4",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.5",
        "2.6",
        "3.0",
        "3.3",
        "3.4",
        "3.5",
        "3.6"
      ],
      "scheme_random_answer": [
        "0.0",
        "0.1",
        "0.5",
        "1.0",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "scheme_random_vocab_cdef": [
        "0.0",
        "0.1",
        "0.5",
        "1.2",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.5",
        "3.6"
      ],
      "scheme_random_vocab_any": [
        "0.0",
        "0.1",
        "0.2",
        "0.4",
        "0.5",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.0",
        "3.2",
        "3.3",
        "3.4",
        "3.6"
      ]
    }
  },
  "units": [
    {
      "id": "ioi.published.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 0,
      "deps": [],
      "core": true,
      "output": "results/ioi.published.s0.json",
      "task": "ioi",
      "candidate": "published",
      "heads": [
        "0.1",
        "0.10",
        "10.0",
        "10.1",
        "10.10",
        "10.2",
        "10.6",
        "10.7",
        "11.10",
        "11.2",
        "11.9",
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "5.8",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.0",
        "9.6",
        "9.7",
        "9.9"
      ],
      "seed": 0,
      "require_pass": false
    },
    {
      "id": "ioi.robust.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 1,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.robust.s0.json",
      "task": "ioi",
      "candidate": "robust",
      "heads": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.dependent.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 2,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.dependent.s0.json",
      "task": "ioi",
      "candidate": "dependent",
      "heads": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "2.8",
        "3.0",
        "3.4",
        "4.0",
        "4.3",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "5.9",
        "6.0",
        "6.6",
        "6.8",
        "6.9",
        "8.6",
        "9.3",
        "9.4",
        "9.7",
        "9.8",
        "10.1",
        "10.2"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.scheme_s2_swap.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 3,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_s2_swap.s0.json",
      "task": "ioi",
      "candidate": "scheme_s2_swap",
      "heads": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.0",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.scheme_abc.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 4,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_abc.s0.json",
      "task": "ioi",
      "candidate": "scheme_abc",
      "heads": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.scheme_random_vocab_s2.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 5,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_random_vocab_s2.s0.json",
      "task": "ioi",
      "candidate": "scheme_random_vocab_s2",
      "heads": [
        "3.0",
        "4.3",
        "5.5",
        "6.0",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.scheme_random_vocab_any.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 6,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_random_vocab_any.s0.json",
      "task": "ioi",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "2.8",
        "3.0",
        "4.0",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "6.8",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.8",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.recovered1_s2_swap.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 7,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered1_s2_swap.s0.json",
      "task": "ioi",
      "candidate": "recovered1_s2_swap",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.0",
        "6.6",
        "9.3",
        "9.4"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.recovered1_abc.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 8,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered1_abc.s0.json",
      "task": "ioi",
      "candidate": "recovered1_abc",
      "heads": [
        "0.1",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "0.3",
        "0.4",
        "0.5"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.recovered2_s2_swap.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 9,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered2_s2_swap.s0.json",
      "task": "ioi",
      "candidate": "recovered2_s2_swap",
      "heads": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.recovered2_abc.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 10,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered2_abc.s0.json",
      "task": "ioi",
      "candidate": "recovered2_abc",
      "heads": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.recovered3_logit_all_rounds.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 11,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered3_logit_all_rounds.s0.json",
      "task": "ioi",
      "candidate": "recovered3_logit_all_rounds",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.6"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.recovered3_logit_rounds_1plus.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 12,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered3_logit_rounds_1plus.s0.json",
      "task": "ioi",
      "candidate": "recovered3_logit_rounds_1plus",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "3.4",
        "6.6"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.recovered3_receiver_side.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 13,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered3_receiver_side.s0.json",
      "task": "ioi",
      "candidate": "recovered3_receiver_side",
      "heads": [
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "7.9",
        "8.6",
        "8.10",
        "3.7",
        "4.3",
        "4.7",
        "5.6"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.recovered4_semantic.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 14,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered4_semantic.s0.json",
      "task": "ioi",
      "candidate": "recovered4_semantic",
      "heads": [
        "10.0",
        "10.7",
        "11.10",
        "3.0",
        "5.5",
        "7.9",
        "8.10",
        "8.6",
        "9.7",
        "9.9"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.recovered4_senders.s0",
      "experiment": 1,
      "kind": "scrub",
      "device": "gpu",
      "priority": 15,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered4_senders.s0.json",
      "task": "ioi",
      "candidate": "recovered4_senders",
      "heads": [
        "10.0",
        "10.10",
        "10.7",
        "3.0",
        "5.5",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.6",
        "9.9"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "greater_than.published.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 16,
      "deps": [],
      "core": true,
      "output": "results/greater_than.published.s0.json",
      "task": "greater_than",
      "candidate": "published",
      "heads": [
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.11",
        "8.8",
        "9.1"
      ],
      "seed": 0,
      "require_pass": false
    },
    {
      "id": "greater_than.robust.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 17,
      "deps": [
        "greater_than.published.s0"
      ],
      "core": true,
      "output": "results/greater_than.robust.s0.json",
      "task": "greater_than",
      "candidate": "robust",
      "heads": [],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "greater_than.dependent.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 18,
      "deps": [
        "greater_than.published.s0"
      ],
      "core": true,
      "output": "results/greater_than.dependent.s0.json",
      "task": "greater_than",
      "candidate": "dependent",
      "heads": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.7",
        "4.10",
        "4.11",
        "5.0",
        "5.1",
        "5.5",
        "5.6",
        "6.8",
        "6.9",
        "6.11",
        "7.0",
        "7.6",
        "7.10",
        "8.5",
        "8.6",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "greater_than.scheme_yy01.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 19,
      "deps": [
        "greater_than.published.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_yy01.s0.json",
      "task": "greater_than",
      "candidate": "scheme_yy01",
      "heads": [
        "0.1",
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "greater_than.scheme_xx_mismatch.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 20,
      "deps": [
        "greater_than.published.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_xx_mismatch.s0.json",
      "task": "greater_than",
      "candidate": "scheme_xx_mismatch",
      "heads": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.10",
        "4.11",
        "5.5",
        "6.9",
        "7.10",
        "8.5",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "greater_than.scheme_random_vocab_yy.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 21,
      "deps": [
        "greater_than.published.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_random_vocab_yy.s0.json",
      "task": "greater_than",
      "candidate": "scheme_random_vocab_yy",
      "heads": [
        "5.0",
        "6.9",
        "8.6"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "greater_than.scheme_random_vocab_any.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 22,
      "deps": [
        "greater_than.published.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_random_vocab_any.s0.json",
      "task": "greater_than",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "4.7",
        "4.11",
        "5.6",
        "6.8",
        "6.11",
        "7.0",
        "7.6",
        "8.5",
        "8.6",
        "9.1"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "docstring.published.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 23,
      "deps": [],
      "core": true,
      "output": "results/docstring.published.s0.json",
      "task": "docstring",
      "candidate": "published",
      "heads": [
        "0.5",
        "1.2",
        "1.4",
        "2.0",
        "3.0",
        "3.6"
      ],
      "seed": 0,
      "require_pass": false
    },
    {
      "id": "docstring.robust.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 24,
      "deps": [
        "docstring.published.s0"
      ],
      "core": true,
      "output": "results/docstring.robust.s0.json",
      "task": "docstring",
      "candidate": "robust",
      "heads": [
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.6"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "docstring.dependent.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 25,
      "deps": [
        "docstring.published.s0"
      ],
      "core": true,
      "output": "results/docstring.dependent.s0.json",
      "task": "docstring",
      "candidate": "dependent",
      "heads": [
        "0.0",
        "0.2",
        "0.4",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.2",
        "3.3",
        "3.4",
        "3.5"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "docstring.scheme_random_random.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 26,
      "deps": [
        "docstring.published.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_random.s0.json",
      "task": "docstring",
      "candidate": "scheme_random_random",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "docstring.scheme_random_def.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 27,
      "deps": [
        "docstring.published.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_def.s0.json",
      "task": "docstring",
      "candidate": "scheme_random_def",
      "heads": [
        "0.1",
        "0.2",
        "0.5",
        "1.0",
        "1.1",
        "1.4",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.5",
        "2.6",
        "3.0",
        "3.3",
        "3.4",
        "3.5",
        "3.6"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "docstring.scheme_random_answer.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 28,
      "deps": [
        "docstring.published.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_answer.s0.json",
      "task": "docstring",
      "candidate": "scheme_random_answer",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "1.0",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "docstring.scheme_random_vocab_cdef.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 29,
      "deps": [
        "docstring.published.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_vocab_cdef.s0.json",
      "task": "docstring",
      "candidate": "scheme_random_vocab_cdef",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "1.2",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.5",
        "3.6"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "docstring.scheme_random_vocab_any.s0",
      "experiment": 2,
      "kind": "scrub",
      "device": "gpu",
      "priority": 30,
      "deps": [
        "docstring.published.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_vocab_any.s0.json",
      "task": "docstring",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "0.0",
        "0.1",
        "0.2",
        "0.4",
        "0.5",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.0",
        "3.2",
        "3.3",
        "3.4",
        "3.6"
      ],
      "seed": 0,
      "require_pass": true
    },
    {
      "id": "ioi.relationship",
      "experiment": 1,
      "kind": "relationship",
      "device": "cpu",
      "priority": 31,
      "deps": [
        "ioi.published.s0",
        "ioi.robust.s0",
        "ioi.dependent.s0",
        "ioi.scheme_s2_swap.s0",
        "ioi.scheme_abc.s0",
        "ioi.scheme_random_vocab_s2.s0",
        "ioi.scheme_random_vocab_any.s0",
        "ioi.recovered1_s2_swap.s0",
        "ioi.recovered1_abc.s0",
        "ioi.recovered2_s2_swap.s0",
        "ioi.recovered2_abc.s0",
        "ioi.recovered3_logit_all_rounds.s0",
        "ioi.recovered3_logit_rounds_1plus.s0",
        "ioi.recovered3_receiver_side.s0",
        "ioi.recovered4_semantic.s0",
        "ioi.recovered4_senders.s0"
      ],
      "core": true,
      "output": "results/ioi.relationship.json",
      "require_pass": false
    },
    {
      "id": "ioi.discovery.s0",
      "experiment": 3,
      "kind": "discovery",
      "device": "gpu",
      "priority": 32,
      "deps": [],
      "core": true,
      "output": "results/ioi.discovery.s0.json",
      "seed": 0,
      "task": "ioi"
    },
    {
      "id": "ioi.discovery.s1",
      "experiment": 3,
      "kind": "discovery",
      "device": "gpu",
      "priority": 33,
      "deps": [],
      "core": true,
      "output": "results/ioi.discovery.s1.json",
      "seed": 1,
      "task": "ioi"
    },
    {
      "id": "ioi.discovery.s2",
      "experiment": 3,
      "kind": "discovery",
      "device": "gpu",
      "priority": 34,
      "deps": [],
      "core": true,
      "output": "results/ioi.discovery.s2.json",
      "seed": 2,
      "task": "ioi"
    },
    {
      "id": "ioi.discovery.s3",
      "experiment": 3,
      "kind": "discovery",
      "device": "gpu",
      "priority": 35,
      "deps": [],
      "core": true,
      "output": "results/ioi.discovery.s3.json",
      "seed": 3,
      "task": "ioi"
    },
    {
      "id": "ioi.discovery.s4",
      "experiment": 3,
      "kind": "discovery",
      "device": "gpu",
      "priority": 36,
      "deps": [],
      "core": true,
      "output": "results/ioi.discovery.s4.json",
      "seed": 4,
      "task": "ioi"
    },
    {
      "id": "ioi.discovery.s5",
      "experiment": 3,
      "kind": "discovery",
      "device": "gpu",
      "priority": 37,
      "deps": [],
      "core": true,
      "output": "results/ioi.discovery.s5.json",
      "seed": 5,
      "task": "ioi"
    },
    {
      "id": "ioi.discovery.s6",
      "experiment": 3,
      "kind": "discovery",
      "device": "gpu",
      "priority": 38,
      "deps": [],
      "core": true,
      "output": "results/ioi.discovery.s6.json",
      "seed": 6,
      "task": "ioi"
    },
    {
      "id": "ioi.discovery.s7",
      "experiment": 3,
      "kind": "discovery",
      "device": "gpu",
      "priority": 39,
      "deps": [],
      "core": true,
      "output": "results/ioi.discovery.s7.json",
      "seed": 7,
      "task": "ioi"
    },
    {
      "id": "ioi.discovery.s8",
      "experiment": 3,
      "kind": "discovery",
      "device": "gpu",
      "priority": 40,
      "deps": [],
      "core": true,
      "output": "results/ioi.discovery.s8.json",
      "seed": 8,
      "task": "ioi"
    },
    {
      "id": "ioi.discovery.s9",
      "experiment": 3,
      "kind": "discovery",
      "device": "gpu",
      "priority": 41,
      "deps": [],
      "core": true,
      "output": "results/ioi.discovery.s9.json",
      "seed": 9,
      "task": "ioi"
    },
    {
      "id": "ioi.table",
      "experiment": 3,
      "kind": "table",
      "device": "cpu",
      "priority": 42,
      "deps": [
        "ioi.discovery.s0",
        "ioi.discovery.s1",
        "ioi.discovery.s2",
        "ioi.discovery.s3",
        "ioi.discovery.s4",
        "ioi.discovery.s5",
        "ioi.discovery.s6",
        "ioi.discovery.s7",
        "ioi.discovery.s8",
        "ioi.discovery.s9"
      ],
      "core": true,
      "output": "results/ioi.table.json"
    },
    {
      "id": "ioi.errors",
      "experiment": 5,
      "kind": "errors",
      "device": "cpu",
      "priority": 43,
      "deps": [],
      "core": true,
      "output": "results/ioi.errors.json",
      "task": "ioi"
    },
    {
      "id": "greater_than.errors",
      "experiment": 5,
      "kind": "errors",
      "device": "cpu",
      "priority": 44,
      "deps": [],
      "core": true,
      "output": "results/greater_than.errors.json",
      "task": "greater_than"
    },
    {
      "id": "docstring.errors",
      "experiment": 5,
      "kind": "errors",
      "device": "cpu",
      "priority": 45,
      "deps": [],
      "core": true,
      "output": "results/docstring.errors.json",
      "task": "docstring"
    },
    {
      "id": "ioi.published.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 46,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.published.s1.json",
      "task": "ioi",
      "candidate": "published",
      "heads": [
        "0.1",
        "0.10",
        "10.0",
        "10.1",
        "10.10",
        "10.2",
        "10.6",
        "10.7",
        "11.10",
        "11.2",
        "11.9",
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "5.8",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.0",
        "9.6",
        "9.7",
        "9.9"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.robust.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 47,
      "deps": [
        "ioi.robust.s0"
      ],
      "core": true,
      "output": "results/ioi.robust.s1.json",
      "task": "ioi",
      "candidate": "robust",
      "heads": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.dependent.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 48,
      "deps": [
        "ioi.dependent.s0"
      ],
      "core": true,
      "output": "results/ioi.dependent.s1.json",
      "task": "ioi",
      "candidate": "dependent",
      "heads": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "2.8",
        "3.0",
        "3.4",
        "4.0",
        "4.3",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "5.9",
        "6.0",
        "6.6",
        "6.8",
        "6.9",
        "8.6",
        "9.3",
        "9.4",
        "9.7",
        "9.8",
        "10.1",
        "10.2"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_s2_swap.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 49,
      "deps": [
        "ioi.scheme_s2_swap.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_s2_swap.s1.json",
      "task": "ioi",
      "candidate": "scheme_s2_swap",
      "heads": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.0",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_abc.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 50,
      "deps": [
        "ioi.scheme_abc.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_abc.s1.json",
      "task": "ioi",
      "candidate": "scheme_abc",
      "heads": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_random_vocab_s2.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 51,
      "deps": [
        "ioi.scheme_random_vocab_s2.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_random_vocab_s2.s1.json",
      "task": "ioi",
      "candidate": "scheme_random_vocab_s2",
      "heads": [
        "3.0",
        "4.3",
        "5.5",
        "6.0",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_random_vocab_any.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 52,
      "deps": [
        "ioi.scheme_random_vocab_any.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_random_vocab_any.s1.json",
      "task": "ioi",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "2.8",
        "3.0",
        "4.0",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "6.8",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.8",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.recovered1_s2_swap.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 53,
      "deps": [
        "ioi.recovered1_s2_swap.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered1_s2_swap.s1.json",
      "task": "ioi",
      "candidate": "recovered1_s2_swap",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.0",
        "6.6",
        "9.3",
        "9.4"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.recovered1_abc.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 54,
      "deps": [
        "ioi.recovered1_abc.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered1_abc.s1.json",
      "task": "ioi",
      "candidate": "recovered1_abc",
      "heads": [
        "0.1",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "0.3",
        "0.4",
        "0.5"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.recovered2_s2_swap.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 55,
      "deps": [
        "ioi.recovered2_s2_swap.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered2_s2_swap.s1.json",
      "task": "ioi",
      "candidate": "recovered2_s2_swap",
      "heads": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.recovered2_abc.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 56,
      "deps": [
        "ioi.recovered2_abc.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered2_abc.s1.json",
      "task": "ioi",
      "candidate": "recovered2_abc",
      "heads": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_logit_all_rounds.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 57,
      "deps": [
        "ioi.recovered3_logit_all_rounds.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered3_logit_all_rounds.s1.json",
      "task": "ioi",
      "candidate": "recovered3_logit_all_rounds",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.6"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_logit_rounds_1plus.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 58,
      "deps": [
        "ioi.recovered3_logit_rounds_1plus.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered3_logit_rounds_1plus.s1.json",
      "task": "ioi",
      "candidate": "recovered3_logit_rounds_1plus",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "3.4",
        "6.6"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_receiver_side.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 59,
      "deps": [
        "ioi.recovered3_receiver_side.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered3_receiver_side.s1.json",
      "task": "ioi",
      "candidate": "recovered3_receiver_side",
      "heads": [
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "7.9",
        "8.6",
        "8.10",
        "3.7",
        "4.3",
        "4.7",
        "5.6"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.recovered4_semantic.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 60,
      "deps": [
        "ioi.recovered4_semantic.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered4_semantic.s1.json",
      "task": "ioi",
      "candidate": "recovered4_semantic",
      "heads": [
        "10.0",
        "10.7",
        "11.10",
        "3.0",
        "5.5",
        "7.9",
        "8.10",
        "8.6",
        "9.7",
        "9.9"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.recovered4_senders.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 61,
      "deps": [
        "ioi.recovered4_senders.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered4_senders.s1.json",
      "task": "ioi",
      "candidate": "recovered4_senders",
      "heads": [
        "10.0",
        "10.10",
        "10.7",
        "3.0",
        "5.5",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.6",
        "9.9"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "greater_than.published.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 62,
      "deps": [
        "greater_than.published.s0"
      ],
      "core": true,
      "output": "results/greater_than.published.s1.json",
      "task": "greater_than",
      "candidate": "published",
      "heads": [
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.11",
        "8.8",
        "9.1"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "greater_than.robust.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 63,
      "deps": [
        "greater_than.robust.s0"
      ],
      "core": true,
      "output": "results/greater_than.robust.s1.json",
      "task": "greater_than",
      "candidate": "robust",
      "heads": [],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "greater_than.dependent.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 64,
      "deps": [
        "greater_than.dependent.s0"
      ],
      "core": true,
      "output": "results/greater_than.dependent.s1.json",
      "task": "greater_than",
      "candidate": "dependent",
      "heads": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.7",
        "4.10",
        "4.11",
        "5.0",
        "5.1",
        "5.5",
        "5.6",
        "6.8",
        "6.9",
        "6.11",
        "7.0",
        "7.6",
        "7.10",
        "8.5",
        "8.6",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_yy01.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 65,
      "deps": [
        "greater_than.scheme_yy01.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_yy01.s1.json",
      "task": "greater_than",
      "candidate": "scheme_yy01",
      "heads": [
        "0.1",
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_xx_mismatch.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 66,
      "deps": [
        "greater_than.scheme_xx_mismatch.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_xx_mismatch.s1.json",
      "task": "greater_than",
      "candidate": "scheme_xx_mismatch",
      "heads": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.10",
        "4.11",
        "5.5",
        "6.9",
        "7.10",
        "8.5",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_random_vocab_yy.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 67,
      "deps": [
        "greater_than.scheme_random_vocab_yy.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_random_vocab_yy.s1.json",
      "task": "greater_than",
      "candidate": "scheme_random_vocab_yy",
      "heads": [
        "5.0",
        "6.9",
        "8.6"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_random_vocab_any.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 68,
      "deps": [
        "greater_than.scheme_random_vocab_any.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_random_vocab_any.s1.json",
      "task": "greater_than",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "4.7",
        "4.11",
        "5.6",
        "6.8",
        "6.11",
        "7.0",
        "7.6",
        "8.5",
        "8.6",
        "9.1"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "docstring.published.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 69,
      "deps": [
        "docstring.published.s0"
      ],
      "core": true,
      "output": "results/docstring.published.s1.json",
      "task": "docstring",
      "candidate": "published",
      "heads": [
        "0.5",
        "1.2",
        "1.4",
        "2.0",
        "3.0",
        "3.6"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "docstring.robust.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 70,
      "deps": [
        "docstring.robust.s0"
      ],
      "core": true,
      "output": "results/docstring.robust.s1.json",
      "task": "docstring",
      "candidate": "robust",
      "heads": [
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.6"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "docstring.dependent.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 71,
      "deps": [
        "docstring.dependent.s0"
      ],
      "core": true,
      "output": "results/docstring.dependent.s1.json",
      "task": "docstring",
      "candidate": "dependent",
      "heads": [
        "0.0",
        "0.2",
        "0.4",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.2",
        "3.3",
        "3.4",
        "3.5"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_random.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 72,
      "deps": [
        "docstring.scheme_random_random.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_random.s1.json",
      "task": "docstring",
      "candidate": "scheme_random_random",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_def.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 73,
      "deps": [
        "docstring.scheme_random_def.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_def.s1.json",
      "task": "docstring",
      "candidate": "scheme_random_def",
      "heads": [
        "0.1",
        "0.2",
        "0.5",
        "1.0",
        "1.1",
        "1.4",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.5",
        "2.6",
        "3.0",
        "3.3",
        "3.4",
        "3.5",
        "3.6"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_answer.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 74,
      "deps": [
        "docstring.scheme_random_answer.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_answer.s1.json",
      "task": "docstring",
      "candidate": "scheme_random_answer",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "1.0",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_vocab_cdef.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 75,
      "deps": [
        "docstring.scheme_random_vocab_cdef.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_vocab_cdef.s1.json",
      "task": "docstring",
      "candidate": "scheme_random_vocab_cdef",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "1.2",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.5",
        "3.6"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_vocab_any.s1",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 76,
      "deps": [
        "docstring.scheme_random_vocab_any.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_vocab_any.s1.json",
      "task": "docstring",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "0.0",
        "0.1",
        "0.2",
        "0.4",
        "0.5",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.0",
        "3.2",
        "3.3",
        "3.4",
        "3.6"
      ],
      "seed": 1,
      "require_pass": false
    },
    {
      "id": "ioi.published.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 77,
      "deps": [
        "ioi.published.s0"
      ],
      "core": true,
      "output": "results/ioi.published.s2.json",
      "task": "ioi",
      "candidate": "published",
      "heads": [
        "0.1",
        "0.10",
        "10.0",
        "10.1",
        "10.10",
        "10.2",
        "10.6",
        "10.7",
        "11.10",
        "11.2",
        "11.9",
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "5.8",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.0",
        "9.6",
        "9.7",
        "9.9"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.robust.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 78,
      "deps": [
        "ioi.robust.s0"
      ],
      "core": true,
      "output": "results/ioi.robust.s2.json",
      "task": "ioi",
      "candidate": "robust",
      "heads": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.dependent.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 79,
      "deps": [
        "ioi.dependent.s0"
      ],
      "core": true,
      "output": "results/ioi.dependent.s2.json",
      "task": "ioi",
      "candidate": "dependent",
      "heads": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "2.8",
        "3.0",
        "3.4",
        "4.0",
        "4.3",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "5.9",
        "6.0",
        "6.6",
        "6.8",
        "6.9",
        "8.6",
        "9.3",
        "9.4",
        "9.7",
        "9.8",
        "10.1",
        "10.2"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_s2_swap.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 80,
      "deps": [
        "ioi.scheme_s2_swap.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_s2_swap.s2.json",
      "task": "ioi",
      "candidate": "scheme_s2_swap",
      "heads": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.0",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_abc.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 81,
      "deps": [
        "ioi.scheme_abc.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_abc.s2.json",
      "task": "ioi",
      "candidate": "scheme_abc",
      "heads": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_random_vocab_s2.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 82,
      "deps": [
        "ioi.scheme_random_vocab_s2.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_random_vocab_s2.s2.json",
      "task": "ioi",
      "candidate": "scheme_random_vocab_s2",
      "heads": [
        "3.0",
        "4.3",
        "5.5",
        "6.0",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_random_vocab_any.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 83,
      "deps": [
        "ioi.scheme_random_vocab_any.s0"
      ],
      "core": true,
      "output": "results/ioi.scheme_random_vocab_any.s2.json",
      "task": "ioi",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "2.8",
        "3.0",
        "4.0",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "6.8",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.8",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.recovered1_s2_swap.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 84,
      "deps": [
        "ioi.recovered1_s2_swap.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered1_s2_swap.s2.json",
      "task": "ioi",
      "candidate": "recovered1_s2_swap",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.0",
        "6.6",
        "9.3",
        "9.4"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.recovered1_abc.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 85,
      "deps": [
        "ioi.recovered1_abc.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered1_abc.s2.json",
      "task": "ioi",
      "candidate": "recovered1_abc",
      "heads": [
        "0.1",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "0.3",
        "0.4",
        "0.5"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.recovered2_s2_swap.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 86,
      "deps": [
        "ioi.recovered2_s2_swap.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered2_s2_swap.s2.json",
      "task": "ioi",
      "candidate": "recovered2_s2_swap",
      "heads": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.recovered2_abc.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 87,
      "deps": [
        "ioi.recovered2_abc.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered2_abc.s2.json",
      "task": "ioi",
      "candidate": "recovered2_abc",
      "heads": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_logit_all_rounds.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 88,
      "deps": [
        "ioi.recovered3_logit_all_rounds.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered3_logit_all_rounds.s2.json",
      "task": "ioi",
      "candidate": "recovered3_logit_all_rounds",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.6"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_logit_rounds_1plus.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 89,
      "deps": [
        "ioi.recovered3_logit_rounds_1plus.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered3_logit_rounds_1plus.s2.json",
      "task": "ioi",
      "candidate": "recovered3_logit_rounds_1plus",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "3.4",
        "6.6"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_receiver_side.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 90,
      "deps": [
        "ioi.recovered3_receiver_side.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered3_receiver_side.s2.json",
      "task": "ioi",
      "candidate": "recovered3_receiver_side",
      "heads": [
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "7.9",
        "8.6",
        "8.10",
        "3.7",
        "4.3",
        "4.7",
        "5.6"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.recovered4_semantic.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 91,
      "deps": [
        "ioi.recovered4_semantic.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered4_semantic.s2.json",
      "task": "ioi",
      "candidate": "recovered4_semantic",
      "heads": [
        "10.0",
        "10.7",
        "11.10",
        "3.0",
        "5.5",
        "7.9",
        "8.10",
        "8.6",
        "9.7",
        "9.9"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.recovered4_senders.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 92,
      "deps": [
        "ioi.recovered4_senders.s0"
      ],
      "core": true,
      "output": "results/ioi.recovered4_senders.s2.json",
      "task": "ioi",
      "candidate": "recovered4_senders",
      "heads": [
        "10.0",
        "10.10",
        "10.7",
        "3.0",
        "5.5",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.6",
        "9.9"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "greater_than.published.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 93,
      "deps": [
        "greater_than.published.s0"
      ],
      "core": true,
      "output": "results/greater_than.published.s2.json",
      "task": "greater_than",
      "candidate": "published",
      "heads": [
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.11",
        "8.8",
        "9.1"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "greater_than.robust.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 94,
      "deps": [
        "greater_than.robust.s0"
      ],
      "core": true,
      "output": "results/greater_than.robust.s2.json",
      "task": "greater_than",
      "candidate": "robust",
      "heads": [],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "greater_than.dependent.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 95,
      "deps": [
        "greater_than.dependent.s0"
      ],
      "core": true,
      "output": "results/greater_than.dependent.s2.json",
      "task": "greater_than",
      "candidate": "dependent",
      "heads": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.7",
        "4.10",
        "4.11",
        "5.0",
        "5.1",
        "5.5",
        "5.6",
        "6.8",
        "6.9",
        "6.11",
        "7.0",
        "7.6",
        "7.10",
        "8.5",
        "8.6",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_yy01.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 96,
      "deps": [
        "greater_than.scheme_yy01.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_yy01.s2.json",
      "task": "greater_than",
      "candidate": "scheme_yy01",
      "heads": [
        "0.1",
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_xx_mismatch.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 97,
      "deps": [
        "greater_than.scheme_xx_mismatch.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_xx_mismatch.s2.json",
      "task": "greater_than",
      "candidate": "scheme_xx_mismatch",
      "heads": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.10",
        "4.11",
        "5.5",
        "6.9",
        "7.10",
        "8.5",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_random_vocab_yy.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 98,
      "deps": [
        "greater_than.scheme_random_vocab_yy.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_random_vocab_yy.s2.json",
      "task": "greater_than",
      "candidate": "scheme_random_vocab_yy",
      "heads": [
        "5.0",
        "6.9",
        "8.6"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_random_vocab_any.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 99,
      "deps": [
        "greater_than.scheme_random_vocab_any.s0"
      ],
      "core": true,
      "output": "results/greater_than.scheme_random_vocab_any.s2.json",
      "task": "greater_than",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "4.7",
        "4.11",
        "5.6",
        "6.8",
        "6.11",
        "7.0",
        "7.6",
        "8.5",
        "8.6",
        "9.1"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "docstring.published.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 100,
      "deps": [
        "docstring.published.s0"
      ],
      "core": true,
      "output": "results/docstring.published.s2.json",
      "task": "docstring",
      "candidate": "published",
      "heads": [
        "0.5",
        "1.2",
        "1.4",
        "2.0",
        "3.0",
        "3.6"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "docstring.robust.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 101,
      "deps": [
        "docstring.robust.s0"
      ],
      "core": true,
      "output": "results/docstring.robust.s2.json",
      "task": "docstring",
      "candidate": "robust",
      "heads": [
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.6"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "docstring.dependent.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 102,
      "deps": [
        "docstring.dependent.s0"
      ],
      "core": true,
      "output": "results/docstring.dependent.s2.json",
      "task": "docstring",
      "candidate": "dependent",
      "heads": [
        "0.0",
        "0.2",
        "0.4",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.2",
        "3.3",
        "3.4",
        "3.5"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_random.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 103,
      "deps": [
        "docstring.scheme_random_random.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_random.s2.json",
      "task": "docstring",
      "candidate": "scheme_random_random",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_def.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 104,
      "deps": [
        "docstring.scheme_random_def.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_def.s2.json",
      "task": "docstring",
      "candidate": "scheme_random_def",
      "heads": [
        "0.1",
        "0.2",
        "0.5",
        "1.0",
        "1.1",
        "1.4",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.5",
        "2.6",
        "3.0",
        "3.3",
        "3.4",
        "3.5",
        "3.6"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_answer.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 105,
      "deps": [
        "docstring.scheme_random_answer.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_answer.s2.json",
      "task": "docstring",
      "candidate": "scheme_random_answer",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "1.0",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_vocab_cdef.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 106,
      "deps": [
        "docstring.scheme_random_vocab_cdef.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_vocab_cdef.s2.json",
      "task": "docstring",
      "candidate": "scheme_random_vocab_cdef",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "1.2",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.5",
        "3.6"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_vocab_any.s2",
      "experiment": 4,
      "kind": "scrub",
      "device": "gpu",
      "priority": 107,
      "deps": [
        "docstring.scheme_random_vocab_any.s0"
      ],
      "core": true,
      "output": "results/docstring.scheme_random_vocab_any.s2.json",
      "task": "docstring",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "0.0",
        "0.1",
        "0.2",
        "0.4",
        "0.5",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.0",
        "3.2",
        "3.3",
        "3.4",
        "3.6"
      ],
      "seed": 2,
      "require_pass": false
    },
    {
      "id": "ioi.published.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 108,
      "deps": [
        "ioi.published.s0"
      ],
      "core": false,
      "output": "results/ioi.published.s3.json",
      "task": "ioi",
      "candidate": "published",
      "heads": [
        "0.1",
        "0.10",
        "10.0",
        "10.1",
        "10.10",
        "10.2",
        "10.6",
        "10.7",
        "11.10",
        "11.2",
        "11.9",
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "5.8",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.0",
        "9.6",
        "9.7",
        "9.9"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.robust.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 109,
      "deps": [
        "ioi.robust.s0"
      ],
      "core": false,
      "output": "results/ioi.robust.s3.json",
      "task": "ioi",
      "candidate": "robust",
      "heads": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.dependent.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 110,
      "deps": [
        "ioi.dependent.s0"
      ],
      "core": false,
      "output": "results/ioi.dependent.s3.json",
      "task": "ioi",
      "candidate": "dependent",
      "heads": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "2.8",
        "3.0",
        "3.4",
        "4.0",
        "4.3",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "5.9",
        "6.0",
        "6.6",
        "6.8",
        "6.9",
        "8.6",
        "9.3",
        "9.4",
        "9.7",
        "9.8",
        "10.1",
        "10.2"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_s2_swap.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 111,
      "deps": [
        "ioi.scheme_s2_swap.s0"
      ],
      "core": false,
      "output": "results/ioi.scheme_s2_swap.s3.json",
      "task": "ioi",
      "candidate": "scheme_s2_swap",
      "heads": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.0",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_abc.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 112,
      "deps": [
        "ioi.scheme_abc.s0"
      ],
      "core": false,
      "output": "results/ioi.scheme_abc.s3.json",
      "task": "ioi",
      "candidate": "scheme_abc",
      "heads": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_random_vocab_s2.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 113,
      "deps": [
        "ioi.scheme_random_vocab_s2.s0"
      ],
      "core": false,
      "output": "results/ioi.scheme_random_vocab_s2.s3.json",
      "task": "ioi",
      "candidate": "scheme_random_vocab_s2",
      "heads": [
        "3.0",
        "4.3",
        "5.5",
        "6.0",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_random_vocab_any.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 114,
      "deps": [
        "ioi.scheme_random_vocab_any.s0"
      ],
      "core": false,
      "output": "results/ioi.scheme_random_vocab_any.s3.json",
      "task": "ioi",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "2.8",
        "3.0",
        "4.0",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "6.8",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.8",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.recovered1_s2_swap.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 115,
      "deps": [
        "ioi.recovered1_s2_swap.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered1_s2_swap.s3.json",
      "task": "ioi",
      "candidate": "recovered1_s2_swap",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.0",
        "6.6",
        "9.3",
        "9.4"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.recovered1_abc.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 116,
      "deps": [
        "ioi.recovered1_abc.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered1_abc.s3.json",
      "task": "ioi",
      "candidate": "recovered1_abc",
      "heads": [
        "0.1",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "0.3",
        "0.4",
        "0.5"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.recovered2_s2_swap.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 117,
      "deps": [
        "ioi.recovered2_s2_swap.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered2_s2_swap.s3.json",
      "task": "ioi",
      "candidate": "recovered2_s2_swap",
      "heads": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.recovered2_abc.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 118,
      "deps": [
        "ioi.recovered2_abc.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered2_abc.s3.json",
      "task": "ioi",
      "candidate": "recovered2_abc",
      "heads": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_logit_all_rounds.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 119,
      "deps": [
        "ioi.recovered3_logit_all_rounds.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered3_logit_all_rounds.s3.json",
      "task": "ioi",
      "candidate": "recovered3_logit_all_rounds",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.6"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_logit_rounds_1plus.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 120,
      "deps": [
        "ioi.recovered3_logit_rounds_1plus.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered3_logit_rounds_1plus.s3.json",
      "task": "ioi",
      "candidate": "recovered3_logit_rounds_1plus",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "3.4",
        "6.6"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_receiver_side.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 121,
      "deps": [
        "ioi.recovered3_receiver_side.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered3_receiver_side.s3.json",
      "task": "ioi",
      "candidate": "recovered3_receiver_side",
      "heads": [
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "7.9",
        "8.6",
        "8.10",
        "3.7",
        "4.3",
        "4.7",
        "5.6"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.recovered4_semantic.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 122,
      "deps": [
        "ioi.recovered4_semantic.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered4_semantic.s3.json",
      "task": "ioi",
      "candidate": "recovered4_semantic",
      "heads": [
        "10.0",
        "10.7",
        "11.10",
        "3.0",
        "5.5",
        "7.9",
        "8.10",
        "8.6",
        "9.7",
        "9.9"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.recovered4_senders.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 123,
      "deps": [
        "ioi.recovered4_senders.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered4_senders.s3.json",
      "task": "ioi",
      "candidate": "recovered4_senders",
      "heads": [
        "10.0",
        "10.10",
        "10.7",
        "3.0",
        "5.5",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.6",
        "9.9"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "greater_than.published.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 124,
      "deps": [
        "greater_than.published.s0"
      ],
      "core": false,
      "output": "results/greater_than.published.s3.json",
      "task": "greater_than",
      "candidate": "published",
      "heads": [
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.11",
        "8.8",
        "9.1"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "greater_than.robust.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 125,
      "deps": [
        "greater_than.robust.s0"
      ],
      "core": false,
      "output": "results/greater_than.robust.s3.json",
      "task": "greater_than",
      "candidate": "robust",
      "heads": [],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "greater_than.dependent.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 126,
      "deps": [
        "greater_than.dependent.s0"
      ],
      "core": false,
      "output": "results/greater_than.dependent.s3.json",
      "task": "greater_than",
      "candidate": "dependent",
      "heads": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.7",
        "4.10",
        "4.11",
        "5.0",
        "5.1",
        "5.5",
        "5.6",
        "6.8",
        "6.9",
        "6.11",
        "7.0",
        "7.6",
        "7.10",
        "8.5",
        "8.6",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_yy01.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 127,
      "deps": [
        "greater_than.scheme_yy01.s0"
      ],
      "core": false,
      "output": "results/greater_than.scheme_yy01.s3.json",
      "task": "greater_than",
      "candidate": "scheme_yy01",
      "heads": [
        "0.1",
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_xx_mismatch.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 128,
      "deps": [
        "greater_than.scheme_xx_mismatch.s0"
      ],
      "core": false,
      "output": "results/greater_than.scheme_xx_mismatch.s3.json",
      "task": "greater_than",
      "candidate": "scheme_xx_mismatch",
      "heads": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.10",
        "4.11",
        "5.5",
        "6.9",
        "7.10",
        "8.5",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_random_vocab_yy.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 129,
      "deps": [
        "greater_than.scheme_random_vocab_yy.s0"
      ],
      "core": false,
      "output": "results/greater_than.scheme_random_vocab_yy.s3.json",
      "task": "greater_than",
      "candidate": "scheme_random_vocab_yy",
      "heads": [
        "5.0",
        "6.9",
        "8.6"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_random_vocab_any.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 130,
      "deps": [
        "greater_than.scheme_random_vocab_any.s0"
      ],
      "core": false,
      "output": "results/greater_than.scheme_random_vocab_any.s3.json",
      "task": "greater_than",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "4.7",
        "4.11",
        "5.6",
        "6.8",
        "6.11",
        "7.0",
        "7.6",
        "8.5",
        "8.6",
        "9.1"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "docstring.published.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 131,
      "deps": [
        "docstring.published.s0"
      ],
      "core": false,
      "output": "results/docstring.published.s3.json",
      "task": "docstring",
      "candidate": "published",
      "heads": [
        "0.5",
        "1.2",
        "1.4",
        "2.0",
        "3.0",
        "3.6"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "docstring.robust.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 132,
      "deps": [
        "docstring.robust.s0"
      ],
      "core": false,
      "output": "results/docstring.robust.s3.json",
      "task": "docstring",
      "candidate": "robust",
      "heads": [
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.6"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "docstring.dependent.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 133,
      "deps": [
        "docstring.dependent.s0"
      ],
      "core": false,
      "output": "results/docstring.dependent.s3.json",
      "task": "docstring",
      "candidate": "dependent",
      "heads": [
        "0.0",
        "0.2",
        "0.4",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.2",
        "3.3",
        "3.4",
        "3.5"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_random.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 134,
      "deps": [
        "docstring.scheme_random_random.s0"
      ],
      "core": false,
      "output": "results/docstring.scheme_random_random.s3.json",
      "task": "docstring",
      "candidate": "scheme_random_random",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_def.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 135,
      "deps": [
        "docstring.scheme_random_def.s0"
      ],
      "core": false,
      "output": "results/docstring.scheme_random_def.s3.json",
      "task": "docstring",
      "candidate": "scheme_random_def",
      "heads": [
        "0.1",
        "0.2",
        "0.5",
        "1.0",
        "1.1",
        "1.4",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.5",
        "2.6",
        "3.0",
        "3.3",
        "3.4",
        "3.5",
        "3.6"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_answer.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 136,
      "deps": [
        "docstring.scheme_random_answer.s0"
      ],
      "core": false,
      "output": "results/docstring.scheme_random_answer.s3.json",
      "task": "docstring",
      "candidate": "scheme_random_answer",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "1.0",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_vocab_cdef.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 137,
      "deps": [
        "docstring.scheme_random_vocab_cdef.s0"
      ],
      "core": false,
      "output": "results/docstring.scheme_random_vocab_cdef.s3.json",
      "task": "docstring",
      "candidate": "scheme_random_vocab_cdef",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "1.2",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.5",
        "3.6"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_vocab_any.s3",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 138,
      "deps": [
        "docstring.scheme_random_vocab_any.s0"
      ],
      "core": false,
      "output": "results/docstring.scheme_random_vocab_any.s3.json",
      "task": "docstring",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "0.0",
        "0.1",
        "0.2",
        "0.4",
        "0.5",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.0",
        "3.2",
        "3.3",
        "3.4",
        "3.6"
      ],
      "seed": 3,
      "require_pass": false
    },
    {
      "id": "ioi.published.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 139,
      "deps": [
        "ioi.published.s0"
      ],
      "core": false,
      "output": "results/ioi.published.s4.json",
      "task": "ioi",
      "candidate": "published",
      "heads": [
        "0.1",
        "0.10",
        "10.0",
        "10.1",
        "10.10",
        "10.2",
        "10.6",
        "10.7",
        "11.10",
        "11.2",
        "11.9",
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "5.8",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.0",
        "9.6",
        "9.7",
        "9.9"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.robust.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 140,
      "deps": [
        "ioi.robust.s0"
      ],
      "core": false,
      "output": "results/ioi.robust.s4.json",
      "task": "ioi",
      "candidate": "robust",
      "heads": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.dependent.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 141,
      "deps": [
        "ioi.dependent.s0"
      ],
      "core": false,
      "output": "results/ioi.dependent.s4.json",
      "task": "ioi",
      "candidate": "dependent",
      "heads": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "2.8",
        "3.0",
        "3.4",
        "4.0",
        "4.3",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "5.9",
        "6.0",
        "6.6",
        "6.8",
        "6.9",
        "8.6",
        "9.3",
        "9.4",
        "9.7",
        "9.8",
        "10.1",
        "10.2"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_s2_swap.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 142,
      "deps": [
        "ioi.scheme_s2_swap.s0"
      ],
      "core": false,
      "output": "results/ioi.scheme_s2_swap.s4.json",
      "task": "ioi",
      "candidate": "scheme_s2_swap",
      "heads": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.0",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_abc.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 143,
      "deps": [
        "ioi.scheme_abc.s0"
      ],
      "core": false,
      "output": "results/ioi.scheme_abc.s4.json",
      "task": "ioi",
      "candidate": "scheme_abc",
      "heads": [
        "0.1",
        "0.3",
        "0.4",
        "0.5",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_random_vocab_s2.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 144,
      "deps": [
        "ioi.scheme_random_vocab_s2.s0"
      ],
      "core": false,
      "output": "results/ioi.scheme_random_vocab_s2.s4.json",
      "task": "ioi",
      "candidate": "scheme_random_vocab_s2",
      "heads": [
        "3.0",
        "4.3",
        "5.5",
        "6.0",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.3",
        "9.4",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.scheme_random_vocab_any.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 145,
      "deps": [
        "ioi.scheme_random_vocab_any.s0"
      ],
      "core": false,
      "output": "results/ioi.scheme_random_vocab_any.s4.json",
      "task": "ioi",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "2.8",
        "3.0",
        "4.0",
        "4.6",
        "4.7",
        "4.11",
        "5.2",
        "5.5",
        "5.7",
        "6.8",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.8",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.recovered1_s2_swap.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 146,
      "deps": [
        "ioi.recovered1_s2_swap.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered1_s2_swap.s4.json",
      "task": "ioi",
      "candidate": "recovered1_s2_swap",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.0",
        "6.6",
        "9.3",
        "9.4"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.recovered1_abc.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 147,
      "deps": [
        "ioi.recovered1_abc.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered1_abc.s4.json",
      "task": "ioi",
      "candidate": "recovered1_abc",
      "heads": [
        "0.1",
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "0.3",
        "0.4",
        "0.5"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.recovered2_s2_swap.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 148,
      "deps": [
        "ioi.recovered2_s2_swap.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered2_s2_swap.s4.json",
      "task": "ioi",
      "candidate": "recovered2_s2_swap",
      "heads": [
        "3.0",
        "3.4",
        "5.5",
        "5.9",
        "6.6",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.recovered2_abc.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 149,
      "deps": [
        "ioi.recovered2_abc.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered2_abc.s4.json",
      "task": "ioi",
      "candidate": "recovered2_abc",
      "heads": [
        "7.3",
        "7.9",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_logit_all_rounds.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 150,
      "deps": [
        "ioi.recovered3_logit_all_rounds.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered3_logit_all_rounds.s4.json",
      "task": "ioi",
      "candidate": "recovered3_logit_all_rounds",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "9.6",
        "9.7",
        "9.9",
        "10.0",
        "10.1",
        "10.2",
        "10.6",
        "10.7",
        "10.10",
        "11.2",
        "11.10",
        "3.4",
        "6.6"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_logit_rounds_1plus.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 151,
      "deps": [
        "ioi.recovered3_logit_rounds_1plus.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered3_logit_rounds_1plus.s4.json",
      "task": "ioi",
      "candidate": "recovered3_logit_rounds_1plus",
      "heads": [
        "3.0",
        "5.5",
        "5.9",
        "6.9",
        "7.3",
        "7.9",
        "8.6",
        "8.10",
        "3.4",
        "6.6"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.recovered3_receiver_side.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 152,
      "deps": [
        "ioi.recovered3_receiver_side.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered3_receiver_side.s4.json",
      "task": "ioi",
      "candidate": "recovered3_receiver_side",
      "heads": [
        "2.2",
        "3.0",
        "4.11",
        "5.5",
        "7.9",
        "8.6",
        "8.10",
        "3.7",
        "4.3",
        "4.7",
        "5.6"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.recovered4_semantic.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 153,
      "deps": [
        "ioi.recovered4_semantic.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered4_semantic.s4.json",
      "task": "ioi",
      "candidate": "recovered4_semantic",
      "heads": [
        "10.0",
        "10.7",
        "11.10",
        "3.0",
        "5.5",
        "7.9",
        "8.10",
        "8.6",
        "9.7",
        "9.9"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.recovered4_senders.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 154,
      "deps": [
        "ioi.recovered4_senders.s0"
      ],
      "core": false,
      "output": "results/ioi.recovered4_senders.s4.json",
      "task": "ioi",
      "candidate": "recovered4_senders",
      "heads": [
        "10.0",
        "10.10",
        "10.7",
        "3.0",
        "5.5",
        "6.9",
        "7.3",
        "7.9",
        "8.10",
        "8.6",
        "9.6",
        "9.9"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "greater_than.published.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 155,
      "deps": [
        "greater_than.published.s0"
      ],
      "core": false,
      "output": "results/greater_than.published.s4.json",
      "task": "greater_than",
      "candidate": "published",
      "heads": [
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.11",
        "8.8",
        "9.1"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "greater_than.robust.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 156,
      "deps": [
        "greater_than.robust.s0"
      ],
      "core": false,
      "output": "results/greater_than.robust.s4.json",
      "task": "greater_than",
      "candidate": "robust",
      "heads": [],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "greater_than.dependent.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 157,
      "deps": [
        "greater_than.dependent.s0"
      ],
      "core": false,
      "output": "results/greater_than.dependent.s4.json",
      "task": "greater_than",
      "candidate": "dependent",
      "heads": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.7",
        "4.10",
        "4.11",
        "5.0",
        "5.1",
        "5.5",
        "5.6",
        "6.8",
        "6.9",
        "6.11",
        "7.0",
        "7.6",
        "7.10",
        "8.5",
        "8.6",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_yy01.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 158,
      "deps": [
        "greater_than.scheme_yy01.s0"
      ],
      "core": false,
      "output": "results/greater_than.scheme_yy01.s4.json",
      "task": "greater_than",
      "candidate": "scheme_yy01",
      "heads": [
        "0.1",
        "5.1",
        "5.5",
        "6.9",
        "7.10",
        "8.8",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_xx_mismatch.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 159,
      "deps": [
        "greater_than.scheme_xx_mismatch.s0"
      ],
      "core": false,
      "output": "results/greater_than.scheme_xx_mismatch.s4.json",
      "task": "greater_than",
      "candidate": "scheme_xx_mismatch",
      "heads": [
        "0.1",
        "0.10",
        "1.5",
        "3.0",
        "3.3",
        "4.4",
        "4.10",
        "4.11",
        "5.5",
        "6.9",
        "7.10",
        "8.5",
        "8.11",
        "9.1",
        "10.7"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_random_vocab_yy.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 160,
      "deps": [
        "greater_than.scheme_random_vocab_yy.s0"
      ],
      "core": false,
      "output": "results/greater_than.scheme_random_vocab_yy.s4.json",
      "task": "greater_than",
      "candidate": "scheme_random_vocab_yy",
      "heads": [
        "5.0",
        "6.9",
        "8.6"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "greater_than.scheme_random_vocab_any.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 161,
      "deps": [
        "greater_than.scheme_random_vocab_any.s0"
      ],
      "core": false,
      "output": "results/greater_than.scheme_random_vocab_any.s4.json",
      "task": "greater_than",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "4.7",
        "4.11",
        "5.6",
        "6.8",
        "6.11",
        "7.0",
        "7.6",
        "8.5",
        "8.6",
        "9.1"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "docstring.published.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 162,
      "deps": [
        "docstring.published.s0"
      ],
      "core": false,
      "output": "results/docstring.published.s4.json",
      "task": "docstring",
      "candidate": "published",
      "heads": [
        "0.5",
        "1.2",
        "1.4",
        "2.0",
        "3.0",
        "3.6"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "docstring.robust.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 163,
      "deps": [
        "docstring.robust.s0"
      ],
      "core": false,
      "output": "results/docstring.robust.s4.json",
      "task": "docstring",
      "candidate": "robust",
      "heads": [
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.6"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "docstring.dependent.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 164,
      "deps": [
        "docstring.dependent.s0"
      ],
      "core": false,
      "output": "results/docstring.dependent.s4.json",
      "task": "docstring",
      "candidate": "dependent",
      "heads": [
        "0.0",
        "0.2",
        "0.4",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.2",
        "3.3",
        "3.4",
        "3.5"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_random.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 165,
      "deps": [
        "docstring.scheme_random_random.s0"
      ],
      "core": false,
      "output": "results/docstring.scheme_random_random.s4.json",
      "task": "docstring",
      "candidate": "scheme_random_random",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "2.2",
        "2.3",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_def.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 166,
      "deps": [
        "docstring.scheme_random_def.s0"
      ],
      "core": false,
      "output": "results/docstring.scheme_random_def.s4.json",
      "task": "docstring",
      "candidate": "scheme_random_def",
      "heads": [
        "0.1",
        "0.2",
        "0.5",
        "1.0",
        "1.1",
        "1.4",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.5",
        "2.6",
        "3.0",
        "3.3",
        "3.4",
        "3.5",
        "3.6"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_answer.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 167,
      "deps": [
        "docstring.scheme_random_answer.s0"
      ],
      "core": false,
      "output": "results/docstring.scheme_random_answer.s4.json",
      "task": "docstring",
      "candidate": "scheme_random_answer",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "1.0",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.3",
        "3.5",
        "3.6"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_vocab_cdef.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 168,
      "deps": [
        "docstring.scheme_random_vocab_cdef.s0"
      ],
      "core": false,
      "output": "results/docstring.scheme_random_vocab_cdef.s4.json",
      "task": "docstring",
      "candidate": "scheme_random_vocab_cdef",
      "heads": [
        "0.0",
        "0.1",
        "0.5",
        "1.2",
        "2.2",
        "2.3",
        "2.5",
        "3.0",
        "3.5",
        "3.6"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "docstring.scheme_random_vocab_any.s4",
      "experiment": 6,
      "kind": "scrub",
      "device": "gpu",
      "priority": 169,
      "deps": [
        "docstring.scheme_random_vocab_any.s0"
      ],
      "core": false,
      "output": "results/docstring.scheme_random_vocab_any.s4.json",
      "task": "docstring",
      "candidate": "scheme_random_vocab_any",
      "heads": [
        "0.0",
        "0.1",
        "0.2",
        "0.4",
        "0.5",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.4",
        "1.5",
        "1.7",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "3.0",
        "3.2",
        "3.3",
        "3.4",
        "3.6"
      ],
      "seed": 4,
      "require_pass": false
    },
    {
      "id": "ioi.stability",
      "experiment": 4,
      "kind": "stability",
      "device": "cpu",
      "priority": 170,
      "deps": [
        "ioi.published.s0",
        "ioi.robust.s0",
        "ioi.dependent.s0",
        "ioi.scheme_s2_swap.s0",
        "ioi.scheme_abc.s0",
        "ioi.scheme_random_vocab_s2.s0",
        "ioi.scheme_random_vocab_any.s0",
        "ioi.recovered1_s2_swap.s0",
        "ioi.recovered1_abc.s0",
        "ioi.recovered2_s2_swap.s0",
        "ioi.recovered2_abc.s0",
        "ioi.recovered3_logit_all_rounds.s0",
        "ioi.recovered3_logit_rounds_1plus.s0",
        "ioi.recovered3_receiver_side.s0",
        "ioi.recovered4_semantic.s0",
        "ioi.recovered4_senders.s0",
        "ioi.published.s1",
        "ioi.robust.s1",
        "ioi.dependent.s1",
        "ioi.scheme_s2_swap.s1",
        "ioi.scheme_abc.s1",
        "ioi.scheme_random_vocab_s2.s1",
        "ioi.scheme_random_vocab_any.s1",
        "ioi.recovered1_s2_swap.s1",
        "ioi.recovered1_abc.s1",
        "ioi.recovered2_s2_swap.s1",
        "ioi.recovered2_abc.s1",
        "ioi.recovered3_logit_all_rounds.s1",
        "ioi.recovered3_logit_rounds_1plus.s1",
        "ioi.recovered3_receiver_side.s1",
        "ioi.recovered4_semantic.s1",
        "ioi.recovered4_senders.s1",
        "ioi.published.s2",
        "ioi.robust.s2",
        "ioi.dependent.s2",
        "ioi.scheme_s2_swap.s2",
        "ioi.scheme_abc.s2",
        "ioi.scheme_random_vocab_s2.s2",
        "ioi.scheme_random_vocab_any.s2",
        "ioi.recovered1_s2_swap.s2",
        "ioi.recovered1_abc.s2",
        "ioi.recovered2_s2_swap.s2",
        "ioi.recovered2_abc.s2",
        "ioi.recovered3_logit_all_rounds.s2",
        "ioi.recovered3_logit_rounds_1plus.s2",
        "ioi.recovered3_receiver_side.s2",
        "ioi.recovered4_semantic.s2",
        "ioi.recovered4_senders.s2"
      ],
      "core": true,
      "output": "results/ioi.stability.json",
      "task": "ioi"
    },
    {
      "id": "greater_than.stability",
      "experiment": 4,
      "kind": "stability",
      "device": "cpu",
      "priority": 171,
      "deps": [
        "greater_than.published.s0",
        "greater_than.robust.s0",
        "greater_than.dependent.s0",
        "greater_than.scheme_yy01.s0",
        "greater_than.scheme_xx_mismatch.s0",
        "greater_than.scheme_random_vocab_yy.s0",
        "greater_than.scheme_random_vocab_any.s0",
        "greater_than.published.s1",
        "greater_than.robust.s1",
        "greater_than.dependent.s1",
        "greater_than.scheme_yy01.s1",
        "greater_than.scheme_xx_mismatch.s1",
        "greater_than.scheme_random_vocab_yy.s1",
        "greater_than.scheme_random_vocab_any.s1",
        "greater_than.published.s2",
        "greater_than.robust.s2",
        "greater_than.dependent.s2",
        "greater_than.scheme_yy01.s2",
        "greater_than.scheme_xx_mismatch.s2",
        "greater_than.scheme_random_vocab_yy.s2",
        "greater_than.scheme_random_vocab_any.s2"
      ],
      "core": true,
      "output": "results/greater_than.stability.json",
      "task": "greater_than"
    },
    {
      "id": "docstring.stability",
      "experiment": 4,
      "kind": "stability",
      "device": "cpu",
      "priority": 172,
      "deps": [
        "docstring.published.s0",
        "docstring.robust.s0",
        "docstring.dependent.s0",
        "docstring.scheme_random_random.s0",
        "docstring.scheme_random_def.s0",
        "docstring.scheme_random_answer.s0",
        "docstring.scheme_random_vocab_cdef.s0",
        "docstring.scheme_random_vocab_any.s0",
        "docstring.published.s1",
        "docstring.robust.s1",
        "docstring.dependent.s1",
        "docstring.scheme_random_random.s1",
        "docstring.scheme_random_def.s1",
        "docstring.scheme_random_answer.s1",
        "docstring.scheme_random_vocab_cdef.s1",
        "docstring.scheme_random_vocab_any.s1",
        "docstring.published.s2",
        "docstring.robust.s2",
        "docstring.dependent.s2",
        "docstring.scheme_random_random.s2",
        "docstring.scheme_random_def.s2",
        "docstring.scheme_random_answer.s2",
        "docstring.scheme_random_vocab_cdef.s2",
        "docstring.scheme_random_vocab_any.s2"
      ],
      "core": true,
      "output": "results/docstring.stability.json",
      "task": "docstring"
    }
  ],
  "model_revisions": {
    "gpt2-small": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "attn-only-4l": "f92b3883ed8e6f34d2e15e7e69d2c361b020be42"
  }
}
```
