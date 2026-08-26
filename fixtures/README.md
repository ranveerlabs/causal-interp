# fixtures, the human-authored input to phase 10

everything in this directory was typed by a person, which is more or less the point
of it.

phase 10 asks how much of task construction can be mechanized starting from a
one-sentence behavioural hunch plus a handful of example prompts. these files are
that handful. They get committed **before** any phase 10 code exists, in the same
commit as [`../results/PHASE10_PLAN.md`](../results/PHASE10_PLAN.md), so the human
contribution to the phase can be counted rather than described. it's the number of
lines below, basically.

each file is one prompt per line, no answers, no annotations, no slot markup. a
line is a complete prompt: the text the model sees, cut immediately before the
token the behaviour is supposed to produce.

they were written naturally and arent filtered. No line was checked against the
tokenizer, or against `causal_interp/greater_than.py`'s word lists, or against
whether the model actually performs the behaviour on it. doing any of that would be
hand-construction of the exact kind this phase is trying to measure, so the plan
pre-registers that the induction reports how many lines it had to drop instead.

| file | hunch it came from | frame |
|---|---|---|
| `greater_than_frame_same.txt` | "this model seems to know that the end of a date range comes after its start" | the published sentence frame |
| `greater_than_frame_own.txt` | the same hunch | a frame written for this phase, sharing no wording with the published one except the numerals |
