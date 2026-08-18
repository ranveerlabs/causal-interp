# fixtures — the human-authored input to Phase 10

Everything in this directory was typed by a person. That is the point of it.

Phase 10 asks how much of task construction can be mechanized starting from a
one-sentence behavioural hunch plus a handful of example prompts. These files are
that handful. They are committed **before** any Phase 10 code exists, in the same
commit as [`../results/PHASE10_PLAN.md`](../results/PHASE10_PLAN.md), so that the
human contribution to the phase can be counted rather than described — it is
literally the number of lines below.

Each file is one prompt per line, no answers, no annotations, no slot markup. A
line is a complete prompt: the text the model sees, cut immediately before the
token the behaviour is supposed to produce.

**They were written naturally and are not filtered.** No line was checked against
the tokenizer, against `causal_interp/greater_than.py`'s word lists, or against
whether the model actually performs the behaviour on it. Doing any of that would
be hand-construction of exactly the kind this phase is trying to measure, and the
plan pre-registers that the induction reports how many lines it had to drop.

| file | hunch it came from | frame |
|---|---|---|
| `greater_than_frame_same.txt` | "this model seems to know that the end of a date range comes after its start" | the published sentence frame |
| `greater_than_frame_own.txt` | the same hunch | a frame written for this phase, sharing no wording with the published one except the numerals |
