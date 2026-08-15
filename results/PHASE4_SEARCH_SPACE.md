# Phase 4 — search space, defined before the search was written

Phases 1–3 were told *where to look*. Every path measurement asked about a
receiver input the paper had already named: S-inhibition acting on name movers'
queries, duplicate-token information arriving as a value at S2, induction keys at
S1+1. Which heads turned up was never constrained; the question asked of them
was. That is guided rediscovery, and it is the reason nothing so far transfers to
a circuit nobody has published.

This phase replaces the supplied receiver specification with a searched one, on
the same task, where the answer is already known and can be checked afterwards.

This document fixes the search space and the compute budget **before** the search
code was written, and is committed before any search result exists. It records
what is searched exhaustively, what is constrained, and what is excluded — the
exclusions are limits on how much this validates, not footnotes.

## What a receiver specification is

A single point in the space is `(layer, head, input, position)`:

| axis | values | count |
|---|---|---|
| layer | 0–11 | 12 |
| head | 0–11 | 12 |
| input | `q`, `k`, `v` | 3 |
| position | see below | 7 or 20 |

Two axes are *not* free, for architectural reasons rather than budgetary ones:

- **Sender position equals receiver position.** A head's `q`, `k` and `v` at token
  position *p* are computed from the residual stream at *p* alone, so only writes
  at *p* can reach them. Nothing else is reachable, so nothing else is searched.
- **Sender layer is below receiver layer.** A head cannot write to a residual
  stream that an earlier layer already read.

## Measured costs

Timed on the actual model and batch (GPT-2 small, 128 prompts, RTX 5060), not
estimated:

| operation | cost |
|---|---|
| score one receiver spec (stage A) | 138 ms |
| sweep all 144 senders into one spec (stage B) | 18 s |

## Stage A — exhaustive

Score every receiver specification by splicing that one input, at that one
position, from the clean run into the corrupted run, and measuring the normalized
recovery of the logit difference. This is Phase 1's metric applied to a head's
*input* rather than its output: it asks whether the information arriving on that
wire is enough to move the behaviour.

One forward pass per specification, so the whole grid is affordable:

| grid | specs | cost | exhaustive? |
|---|---|---|---|
| 12 × 12 × 3 inputs × 7 semantic positions | 3024 | 6.9 min | **yes** |
| 12 × 12 × 3 inputs × 20 absolute positions | 8640 | 19.9 min | **yes** |

Both are run. They ask different questions, and only the second is a real test of
autonomy:

- **Semantic-position search** uses the position vocabulary the earlier phases
  used (IO, IO+1, S1, S1+1, S2, S2+1, END). It is directly comparable with Phases
  2 and 3 — but those position labels are derived from knowing which name is the
  indirect object and where the subject repeats, which is knowledge about the task
  that a search on an unfamiliar circuit would not have.
- **Absolute-position search** runs over every token index with no semantic labels
  at all, on a single fixed-length template so that index *k* means the same thing
  in every prompt. Nothing about the task's structure is supplied. Whether this
  search lands on the indices that happen to be S2 and END is the question the
  phase exists to answer.

## Stage B — constrained, and this is the real budget limit

For a specification that survives stage A, sweep all 144 heads as senders into it
and score them with the two criteria already built: `path_signal` against Phase
3's pre-registered threshold of 0.11, and path effect on the output logits.

At 18 s per specification, the full grid is **about 15 hours**. That is not a
budget this project has, so stage B runs only on the top **K = 20** specifications
by stage-A score. K is fixed here, in advance, and is not tuned afterwards.

This is the sharpest limitation in the phase. A specification that scores poorly
on stage A never reaches stage B, so a receiver whose incoming information matters
to the mechanism but not to the logit difference is invisible to this search —
and Phase 3 established that such receivers exist, since that is exactly what the
previous-token heads turned out to be. The screen is a logit-effect screen, and it
inherits the blind spot Phase 3 documented.

## Excluded from the search entirely

Each of these is a real restriction on what the phase can claim:

- **MLPs as receivers.** Only attention-head inputs are searched. MLPs are
  recomputed during path patching and can carry a path, but are never themselves
  the endpoint of one here.
- **Anything other than `q`, `k`, `v`.** A head's output `z` is treated only as a
  sender. Residual-stream and attention-pattern receivers are not searched.
- **Compositions.** Only single sender → single receiver paths. A pair of heads
  that matters jointly and neither of which matters alone will not be found, which
  is the same limitation Phase 1 documented for marginal effects.
- **The corruption scheme.** Both schemes were designed by hand from the task's
  structure. A fully autonomous system would have to construct its own
  counterfactual, and nothing here does that.
- **The task.** The IOI dataset, its templates and its metric are all supplied.

## Scoring, and what the search may not see

The search uses only measurements already built and validated in earlier phases:
normalized logit recovery for stage A, and `path_signal` plus path effect for
stage B. No new criterion is introduced, and Phase 3's threshold is reused as
recorded rather than recalibrated.

`causal_interp/ground_truth.py` is not imported by any search code. The published
circuit is consulted only in the comparison that runs after the search has
produced its output.

## The check this phase runs afterwards

Once the search has produced rankings, and not before, they are compared against
the paper's account:

> For heads the paper places in the circuit, does the search independently rank
> the receiver specification the paper names — name movers at `q@END`,
> S-inhibition at `v@S2`, induction at `k@S1+1` — above the alternatives?

Three outcomes are possible and all three are reportable: **agreement**, where the
search's top specification matches the published one; **disagreement**, where it
prefers a different input or position; and **ambiguity**, where several
specifications score close enough that the ranking does not distinguish them.
Ambiguity is a finding about how reliable such a search can be, not noise to be
resolved away — a search that cannot separate `q@END` from `v@IO` for a name mover
is telling us something about the method's resolution, and on an unfamiliar
circuit there would be no answer key to notice it.
