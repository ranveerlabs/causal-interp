# Phase 7 plan — applying the pipeline to a circuit in a different model

**Written and committed before any Phase 7 code was written, and before any Phase 7
measurement existed.** Its purpose is the same as [PHASE6_PLAN.md](PHASE6_PLAN.md)'s:
fix the target, the ground truth, the thresholds and the predictions in advance, so
that a poor result cannot be turned into a good one afterwards.

Phase 6 pointed the pipeline at a second published circuit and it transferred with
the causal core untouched. But both circuits lived in **GPT-2 small**. Every number
this project has ever produced comes from one 12-layer, 12-head, MLP-bearing model
with one tokenizer. Nothing so far distinguishes *a method that works on
transformers* from *a method fitted to GPT-2 small*. This phase changes the model.

---

## 1. Target: the Python docstring circuit in `attn-only-4l`

**Chosen: the docstring circuit** in **`attn-only-4l`** (`NeelNanda/Attn_Only_4L512W_C4_Code`),
from Heimersheim and Janiak (2023),
[*A circuit for Python docstrings in a 4-layer attention-only transformer*](https://www.lesswrong.com/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only).

### Why this one

The bar is Phase 6's, unchanged: **the published circuit must name its components
individually**, so that recovery can be scored head by head against a fixed list
that no run of ours can influence. The docstring circuit clears it twice over —
the post names its heads in prose, and the heads recur as a literal 37-edge graph
in released benchmark code (§2).

Four properties make it a real change of model rather than a change of label:

- **Different architecture.** `attn_only=True`: the model has **no MLP layers at
  all**. Every published circuit this project has touched so far lives in a model
  where attention and MLPs alternate, and the greater-than circuit's published
  *centre* was its MLPs. A pipeline that has only ever run on MLP-bearing models
  is about to run on one that has none.
- **Different everything else, too**: 4 layers × 8 heads (32 heads, against
  GPT-2 small's 144), `d_model` 512 against 768, `LNPre` normalization, and a
  **different tokenizer** — `NeelNanda/gpt-neox-tokenizer-digits`, `d_vocab`
  48262, not GPT-2's 50257. Nothing about the token indices, vocabulary or
  position layout carries over from Phases 1–6.
- **Different training run and corpus** (C4 + Python code), by a different author,
  so "the same model happened to be studied twice" cannot explain any transfer.
- **Independently benchmarked.** The circuit is one of the canonical ground-truth
  circuits in Conmy et al. (2023), [*Towards Automated Circuit Discovery*](https://arxiv.org/abs/2304.14997),
  whose released code contains the manual graph as data.

### Why the alternatives were rejected

Both candidates the brief named were checked against real sources, not memory.

**GPT-2 medium / large — rejected on ground truth, not on principle.** The natural
candidate was the IOI circuit reproduced in a larger GPT-2 by Merullo, Eickhoff and
Pavlick (2024), [*Circuit Component Reuse Across Tasks*](https://arxiv.org/abs/2310.08744)
(ICLR 2024), which studies **GPT-2 medium** (24 layers, 16 heads). It would have
been an attractive target — same task as Phases 1–4, so model and task could be
varied one at a time. It fails the head-level bar:

- The paper defines its circuit by **thresholding at "the 2% most important heads
  for each circuit"**, giving *approximately* 25–32 heads, and never enumerates
  them in a table. The heads it names in prose (movers 14.14, 15.14, 16.15, 17.4,
  18.5, 19.15; negative mover 19.1; inhibition 12.3, 13.4, 13.14; duplicate/induction
  6.4, 9.3) are given as examples, not as the circuit.
- The authors' code release, [`jmerullo/circuit_reuse`](https://github.com/jmerullo/circuit_reuse),
  contains the patching scripts that *produce* that thresholded set and **no
  ground-truth head list** to cross-check against.

Building `ground_truth_gpt2_medium.py` from that would mean choosing which heads
count — which is exactly what the `ground_truth*` modules exist to prevent. It is
the same reason Phase 6 rejected induction heads: a behavioural description, or a
threshold on someone else's importance score, is not a fixed component list.

**Pythia — rejected on availability of a published circuit.** Pythia is genuinely
a different architecture family (rotary embeddings, parallel attention/MLP) and
would have been the strongest target on architecture grounds. Searching for a
head-level published circuit in any Pythia model turned up feature-level work
(sparse-autoencoder circuits in Pythia-70m), single-head phenomena (successor
heads), and automated-discovery papers that *produce* circuits rather than
publishing a manually verified one. No paper was found that names a set of Pythia
attention heads as a circuit the way Wang et al. name 26 and Hanna et al. name 7.
**Rejected for lack of an answer key, not for lack of interest** — this is the
target to revisit if one is published.

### The limitation this target carries, stated now

`attn-only-4l` is **smaller and simpler** than GPT-2 small, not larger. So this
phase tests generality across **architecture and model identity**, and does *not*
test generality across **scale**. A result here cannot be read as evidence that the
method survives going up. That limitation is a direct consequence of the rejection
above: the scale candidate had no usable ground truth, and inventing one would have
destroyed the property that makes any of these comparisons meaningful.

## 2. The published ground truth

**Primary — 6 attention heads.** The post's own summary of the circuit:

> "mainly heads **1.4**, **2.0**, **3.0**, and **3.6** for moving the information",
> with "head **0.5** for token transformation and some additional support, as well
> as **1.2** for additional support."

Cross-checked against the released benchmark code exactly as `ground_truth.py` and
`ground_truth_greater_than.py` were. `acdc/docstring/utils.py` in
[ArthurConmy/Automatic-Circuit-Discovery](https://github.com/ArthurConmy/Automatic-Circuit-Discovery)
contains `get_docstring_subgraph_true_edges()`, commented "the manual graph, from
Stefan", and ending

```python
assert len(edges_to_keep) == 37, len(edges_to_keep)  # reflects the value in the
                                                     # docstring appendix of the
                                                     # manual circuit
```

The heads appearing in those 37 edges are **0.5, 1.2, 1.4, 2.0, 3.0, 3.6** — the
same six the post names, and no others.

| head | published role |
|---|---|
| 3.0, 3.6 | argument mover — copy the argument name to the output |
| 2.0 | fuzzy previous-token head — moves `B_def` information from `,_B` to `C_def` |
| 1.4 | fuzzy previous-token head *and* induction head, depending on position |
| 0.5 | duplicate-token head, acting mostly as a token-embedding transform |
| 1.2 | duplicate-token head, suppressing the movers' attention to docstring args |

**Secondary — the 8-head set, declared now so it cannot be adopted later.** The
post names two further heads and says plainly that patching will not see them:

> "At least Previous Token Head **0.2** and Positional Information Head **0.4**
> (and potentially more) are also essential parts of the circuit, but won't show
> differ in our patching experiments."

They are excluded from the released 37-edge graph. **The primary comparison is
therefore the 6 heads**, and a secondary line additionally counts 0.2 and 0.4
(8 heads). Both are fixed here; whichever scores better afterwards, the 6-head
number is the headline. This mirrors Phase 6's primary-7 / secondary-10 split.

**Receiver specifications.** Read off the released edge graph (which input each
head receives on) crossed with the post's account (at which token position). Where
the two do not pin down a single position, the head is declared **unscoreable now**
rather than scored against a guess:

| class | published receiver spec | source |
|---|---|---|
| argument mover (3.0, 3.6) | `q@END` | query sits at the final `:param`; `3.hook_q_input ← 1.attn.hook_result H4` |
| fuzzy previous token (2.0) | `v@comma_B` | attends from `C_def` to `,_B`; `2.hook_v_input H0 ← 1.attn.hook_result H4` |
| induction + fuzzy prev token (1.4) | `v@B_def` | attends from `,_B` to `B_def`; `1.hook_v_input H4 ← 0.attn.hook_result H5` |
| duplicate token (0.5, 1.2) | **none** | 0.5 acts at several positions at once; 1.2 is defined by its effect on *other* heads' keys |

So **4 of the 6 heads are scoreable** on receiver-spec rediscovery, and 2 are not
measured rather than measured-and-failed. Two alternatives are recorded here so
they cannot be introduced afterwards: the movers also have published `k@C_def`
(from 2.0 and 1.2) and `v@C_def` inputs, and 1.4 has a docstring-side `k@B_doc`.
**`q@END` and `v@B_def` are the primary specs**; the alternatives are reported as
a separate line, never merged into the headline.

## 3. Task construction

Supplied, as in every phase — and supplied from the authors' own generator, not a
new one. `causal_interp/docstring.py` reproduces `docstring_prompt_templ` and
`docstring_induction_prompt_generator` from `acdc/docstring/prompts.py` (itself
credited to Heimersheim and Kajetan Janiak), with the same word lists and the same
argument counts the benchmark uses:

```python
n_matching_args=3, n_def_prefix_args=2, n_def_suffix_args=1,
n_doc_prefix_args=0, met_desc_len=3, arg_desc_len=2
```

which produces prompts of the form

```
def old(self, first, files, page, names, size, read):
    """sector gap population

    :param page: message tree
    :param names: detail mine
    :param
```

with the correct completion ` size`. Every argument and description word is a
single token under this tokenizer (**verified by probe: 0 of 111 argument names and
0 of 687 description nouns are multi-token**), so every prompt is exactly 41 tokens
and clean/corrupted pairs align token-for-token.

**Positions** — seven, the same count IOI used, named after the post's own labels:

| name | token | index |
|---|---|---|
| `A_def`, `B_def`, `C_def` | the three matching arguments in the `def` line | 11, 13, 15 |
| `comma_B` | the post's `,_B`, the comma between `B_def` and `C_def` | 14 |
| `A_doc`, `B_doc` | the two arguments repeated in `:param` lines | 27, 34 |
| `END` | the final `param` token, where the answer is read off | 40 |

Indices are constant because every prompt has the same shape; the dataset locates
them by matching the argument tokens and asserts, rather than hard-coding.

**Corruption schemes.** Five: three published, two generic.

| scheme | knowledge used | source |
|---|---|---|
| `random_random` | the task's own counterfactual | authors' generator; **primary** |
| `random_def` | the task's own counterfactual | authors' generator |
| `random_answer` | the task's own counterfactual | authors' generator |
| `random_vocab_cdef` | position only | Phase 5's generic substitution, anchored at `C_def` |
| `random_vocab_any` | **none** | Phase 5's generic substitution, position drawn uniformly |

The generic pair is the existing `causal_interp/corruption.py` code path, called
unchanged, as Phase 6 called it.

**Why `random_random` is primary, decided on a stated rule.** It is the default in
the authors' released benchmark harness — `get_all_docstring_things(...,
dataset_version="random_random")` — so the choice rests on a fact about the code
release rather than on anything measured here. **This is not a blind choice and is
not presented as one:** clean and corrupted logit differences for all four
published schemes were measured while checking the task runs at all, and are
recorded here before any patching:

| scheme | corrupted logit diff | tokens changed |
|---|---|---|
| clean | **+0.78** | — |
| `random_random` | −5.74 | 5 |
| `random_answer` | −6.57 | 1 |
| `random_def` | −1.51 | 2 |
| `random_doc` | −1.58 | 2 |

`random_answer` has the **larger** span and is deliberately *not* primary: it
changes only `C_def`, so everything before token 15 is bit-identical and three of
the seven positions would be structurally unmeasurable. Choosing the scheme with
the smaller span, for a structural reason, is the opposite of tuning.

**Metric.** Both options Phase 5 compared:

- hand-built — the authors' `raw_docstring_metric`: the correct argument's logit
  minus the **largest** wrong-argument logit, at the final position, with the
  wrong-answer list taken from their generator;
- generic — the existing `metrics.py` KL and total variation, **unmodified**.

## 4. Thresholds — inherited, not chosen

| threshold | treatment |
|---|---|
| activation-patching cutoff `0.02` | **inherited unchanged** from Phase 1, as Phases 5 and 6 did |
| size-matched set | top-**6**, the published head count — **the headline**, as top-7 was in Phase 6 |
| receiver-side `path_signal` cutoff | **rule inherited, number recalibrated** |

The receiver-side rule is Phase 3's, verbatim:

> threshold = 99th percentile of `|path_signal|` under a shuffled-source null,
> rounded up to two significant figures

Phase 6 established why the *number* cannot be inherited: recalibrating gave 0.046
against IOI's 0.11, so reusing IOI's value would have been more than twice too
strict. A different model, a different null width again. Recalibrating under the
same rule is not choosing a threshold; changing the rule or the quantile would be,
and neither is permitted. As in Phases 3 and 6 the recalibration runs in a
`--preregister` step that computes no real measurement and is committed before the
comparison exists.

## 5. What has to change to point the pipeline at a new model — audited in advance

Written before the run, from reading the code and probing the model, so that §6 of
the report can be checked against it rather than written to match it.

| assumption | status on `attn-only-4l` |
|---|---|
| `model.py` loads by name | **already parameterised**; `load("attn-only-4l")` needs no edit |
| `n_layers` / `n_heads` | read from `model.cfg` everywhere in `interventions.py` and `search.py` — no literal 12 or 144 anywhere in the core |
| head activations shaped `(batch, pos, head, d_head)` | unchanged; `q`/`k`/`v`/`z` hooks all present |
| residual stream conventions | `positional_embedding_type="standard"`, so `resid_pre` patching means what it meant before (this model is *not* the shortformer variant some of Neel Nanda's toy models use) |
| **MLPs exist** | **false.** `hook_mlp_out` is registered but **never fires** |

That last row is the one real architecture dependency, and it fails *silently*:
`run_with_hooks` accepts a hook on `blocks.N.hook_mlp_out`, the hook is never
called, the patch never happens, and `sweep_component("mlp_out")` returns a clean
grid of zeros that looks like a measurement. **Prediction (§6.6): it returns exact
zeros at every layer and position rather than raising.** The MLP sweep is run
deliberately to demonstrate this, then excluded from the analysis.

`freeze_mlps=True` is affected the same way, and is moot here for a second reason:
with no MLPs, pinning every attention head already blocks every route, so the
strict and lenient notions of "direct" coincide by construction.

A new `scripts/check_patching_docstring.py` provides known-answer tests for the new
model, because a new model is exactly where the machinery could be silently wrong.
One of its checks is stronger than anything available on GPT-2 small: with no MLPs,
patching **every head at every position** must reproduce the clean run at the final
token *exactly* — the END token is unchanged by the corruption, so its residual
stream is its own embedding plus the head outputs, and every one of those has been
replaced. Tolerance `1e-5`, not GPT-2 small's `0.05`.

## 6. Predictions

Recorded so they can be scored. Phase 1's prediction about path patching was
**wrong** and was reported as such; Phase 6 scored one of its four a tie.

1. **Structural blindness will *not* appear under the primary scheme, and *will*
   under the anchored generic one.** `random_random` changes tokens at 11, 13, 15,
   27 and 34, and the earliest semantic position is `A_def` at 11, so no position
   should be an exact-zero block. `random_vocab_cdef` anchors at `C_def` (15), so
   `A_def`, `B_def` and `comma_B` must be **exact** floating-point zeros in every
   cell. Measured, not assumed.
2. **Activation patching recovers ≥ 4 of the 6 heads** in the size-matched top-6.
   3.0 and 3.6 act directly on the logits; 2.0 and 1.4 act through them but this
   model is four layers deep, so total-effect patching propagates. **0.5 and 1.2
   are the ones at risk**: 0.5 is a token-embedding transform feeding other heads,
   and 1.2 acts only by modulating another head's keys — Phase 1's exact failure
   class.
3. **The path chain recovers the published wiring order** `3.x ← {2.0, 1.2} ←
   1.4 ← 0.5`, sweeping all 32 heads as senders at each round and never being told
   which heads to expect.
4. **Precision at the inherited `0.02` cutoff will be worse than on both earlier
   circuits** (IOI 0.90, greater-than 0.778). With 32 heads, no MLPs, and a span
   normalized to a 4-layer model, a 2% cutoff is a very low bar; more than half of
   all heads are expected to clear it.
5. **Phase 5's corruption result reproduces**: generic corruption *and* generic
   metric together recover fewer heads than the hand-built pair.
6. **`sweep_component("mlp_out")` returns exact zeros rather than raising** — the
   silent-no-op failure mode described in §5.
7. **0.2 and 0.4 are not recovered by activation patching**, because the post says
   they will not be. If they *are*, that is a result against the published account
   and will be reported as one.

**A deflation fixed in advance, before any score exists.** 6 published heads among
32 is 19% of the model, against 26 of 144 (18%) for IOI and 7 of 144 (5%) for
greater-than. A random top-6 draw here expects **1.1** hits, against 4.7 for IOI's
top-26 and 0.3 for greater-than's top-7. **A given recovery fraction on this
circuit is therefore worth less than the same fraction on greater-than**, and any
headline must be read against that, not against Phase 6's 7/7.

## 7. What counts as a failure of this phase

- Recovery near chance — a top-6 containing 0 or 1 published heads — would say the
  pipeline was GPT-2-shaped.
- Needing to change `interventions.py`, `search.py`, `metrics.py` or
  `comparison.py` to make the run work at all would say the causal core encodes
  model assumptions it does not admit to. The MLP dependency in §5 is *already
  known* and lives in a script-level constant, not in those modules; if it turns
  out to reach into them, that is a failure and will be reported as one.
- Changing a threshold, a rule, a metric or the primary corruption *after seeing a
  score* invalidates the phase whatever number it produced.

Adjustments to imports, a task module, a ground-truth module and a report module
are **expected and are not failures** — they are the surface any new target needs.
Section 6 of the report will itemise exactly which files were touched, and that
itemisation, not the recovery number, is what this phase actually measures.
