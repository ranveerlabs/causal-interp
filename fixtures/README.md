# phase 10 fixtures

I wrote the prompts in this directory by hand. Phase 10 tests how much task setup can be
automated from a one-sentence hunch and a few examples, so these are the human part of the
input.

I committed them with [`../results/PHASE10_PLAN.md`](../results/PHASE10_PLAN.md), before
the phase 10 code existed. That makes the amount of human input countable: its the lines
in these files.

Each file has one prompt per line. No answers, annotations or slot markup. The prompt
ends right before the token the model is supposed to produce.

I didnt filter the examples. I didnt check them against the tokenizer, the word lists in
`causal_interp/greater_than.py`, or whether the model gets them right. Doing that would
mean hand-building the task this phase is supposed to automate. The plan instead counts
how many examples the induction step has to drop.

| file | hunch it came from | frame |
|---|---|---|
| `greater_than_frame_same.txt` | "this model seems to know that the end of a date range comes after its start" | the published sentence frame |
| `greater_than_frame_own.txt` | the same hunch | a frame written for this phase, sharing no wording with the published one except the numerals |
