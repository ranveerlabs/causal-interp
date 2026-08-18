"""Inducing a task's structure from a handful of example prompts — Phase 10.

Every task module in this project is a hand-written template plus hand-written slot
vocabularies plus a hand-written counterfactual. `greater_than.py` is 457 lines of it,
and `results/PHASE5_AUDIT.md` marked the whole of it `supplied` five phases ago.

This module is the attempt to derive most of that from a person's example prompts
instead. It takes a list of strings — prompts a human wrote from a one-sentence hunch,
cut immediately before the token the behaviour should produce — and returns:

- which token columns are **frame** (constant across every example) and which are
  **slots** (they vary);
- which slot columns are **tied** (they co-vary exactly, so they are one slot appearing
  twice — greater-than's two century positions are the case this exists for);
- the observed value set of each slot, which serves as its vocabulary;
- a **position vocabulary** of bare token indices, which Phase 4 showed is as good as
  semantic labels;
- a set of proposed **counterfactual schemes**, one per slot plus one per tied column.

Nothing here knows what a token means. There is no notion of a noun, a year, a name or
an argument; the only operations are "is this column constant", "do these two columns
hold the same token in every example", and "would the tokenizer reproduce this row from
its own decoding". The last of those is the mechanized replacement for the hand-written
`_single_token_nouns` / `_valid_years` filters, and it is the only quality gate.

The algorithm is a transcription of section 3 of `results/PHASE10_PLAN.md`, committed
before this file existed. Where the implementation had to decide something the plan did
not spell out, the decision is commented and named as such.

**It must never import a `ground_truth` module**, and the Phase 10 runner asserts that
at startup, for the same reason `search.py` and `agreement.py` carry the prohibition: a
task built with the answer key in reach would prove nothing about what can be built
without one.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor

# How many draws the generator is allowed per prompt it is asked for, before it gives
# up and returns however many distinct rows it managed. Fixed here rather than exposed:
# a caller who could raise it would be tuning the dataset size against the round-trip
# rejection rate, and the rejection rate is one of the phase's measurements.
ATTEMPT_BUDGET = 50

# Label for the final token position. Everything downstream — `metrics.final_log_probs`
# in particular — reads the output distribution at `positions["END"]`, so this key has
# to exist whether or not the last column happens to vary.
END = "END"


def label_for(index: int, length: int) -> str:
    """Position label for a token index: bare `t{i}`, except the last, which is END.

    Phase 4 established that the receiver-spec search finds the same positions from
    bare indices as from semantic labels, so an induced task labels positions by index
    and loses nothing. The one exception is the final position, which every metric in
    the repository reads by name.
    """
    return END if index == length - 1 else f"t{index}"


@dataclass(frozen=True)
class Slot:
    """One varying part of the prompt: where it appears, and what was observed in it.

    `columns` holds every token index this slot occupies. More than one means the slot
    is **tied** — the same value appeared at all of those indices in every example, so
    they are treated as one thing that must be written together. `values` is the
    distinct tokens seen there, in the order the examples introduced them; it is the
    slot's whole vocabulary, and it is exactly as large as the human's input allows.
    """

    columns: tuple[int, ...]
    values: tuple[int, ...]
    length: int

    @property
    def anchor(self) -> int:
        """The column this slot is named after: its first."""
        return self.columns[0]

    @property
    def label(self) -> str:
        return label_for(self.anchor, self.length)

    @property
    def is_tied(self) -> bool:
        return len(self.columns) > 1

    def as_dict(self, decode: Any = None) -> dict:
        block = {
            "label": self.label,
            "columns": list(self.columns),
            "tied": self.is_tied,
            "n_values": len(self.values),
            "values": list(self.values),
        }
        if decode is not None:
            block["value_strings"] = [decode(v) for v in self.values]
        return block


@dataclass(frozen=True)
class Proposal:
    """One counterfactual the induction proposes, before anything has been measured.

    `kind` is `resample` — redraw the slot's value and write it to every column the
    slot occupies — or `desync`, which exists only for tied slots and rewrites **one**
    of their columns, leaving its partners at the clean value. The second is the
    mechanized form of Phase 8's authored `xx_mismatch`, and it is proposed here
    because the slot was found to be tied, not because anyone said so.
    """

    name: str
    kind: str
    slot_index: int
    columns: tuple[int, ...]
    label: str
    breaks: str


@dataclass
class Structure:
    """What the induction found: the frame, the slots, and what it had to throw away.

    `dropped` is not an error path. A person writing natural examples cannot see the
    tokenizer, so some lines will not tokenize to the same length as the rest; those
    are dropped and counted, and the count is one of the phase's measurements.
    """

    length: int
    base_row: tuple[int, ...]
    frame_columns: tuple[int, ...]
    slots: tuple[Slot, ...]
    positions: tuple[str, ...]
    n_examples_given: int
    n_examples_kept: int
    dropped: tuple[dict, ...] = ()

    @property
    def slot_columns(self) -> tuple[int, ...]:
        return tuple(sorted(c for slot in self.slots for c in slot.columns))

    def position_index(self, label: str) -> int:
        """The token index a position label refers to."""
        if label == END:
            return self.length - 1
        return int(label[1:])

    def as_dict(self, decode: Any = None) -> dict:
        return {
            "length": self.length,
            "n_examples_given": self.n_examples_given,
            "n_examples_kept": self.n_examples_kept,
            "dropped": list(self.dropped),
            "n_frame_columns": len(self.frame_columns),
            "frame_columns": list(self.frame_columns),
            "n_slots": len(self.slots),
            "slots": [s.as_dict(decode) for s in self.slots],
            "positions": list(self.positions),
        }


def _tokenize(model: Any, text: str) -> tuple[int, ...]:
    """One example's tokens, with BOS, as a plain tuple.

    Tokenized one string at a time on purpose. `to_tokens` on a *list* pads to the
    longest entry, which would make every example the same length and destroy the one
    signal this module needs — that some of the human's lines do not fit the others.
    """
    return tuple(int(t) for t in model.to_tokens(text)[0])


def induce(model: Any, examples: Sequence[str]) -> Structure:
    """Section 3.1 of the plan: example strings in, slot structure out.

    Raises only when there is nothing to induce from — fewer than two usable examples
    means no column can be observed to vary, and a `Structure` with no slots is not a
    task. Everything else that goes wrong is reported rather than raised.
    """
    if len(examples) < 2:
        raise ValueError(f"induction needs at least 2 examples, got {len(examples)}")

    rows = [_tokenize(model, text) for text in examples]
    lengths = Counter(len(row) for row in rows)
    # Modal length, ties broken towards the longer row. The tie-break is a choice the
    # plan did not make; it is arbitrary and is recorded so it is not mistaken for a
    # finding. With the fixtures used here no tie occurs.
    modal = max(lengths, key=lambda L: (lengths[L], L))

    keep, dropped = [], []
    for i, (text, row) in enumerate(zip(examples, rows)):
        if len(row) == modal:
            keep.append(row)
        else:
            dropped.append({"index": i, "length": len(row), "modal": modal, "text": text})

    if len(keep) < 2:
        raise ValueError(
            f"only {len(keep)} of {len(examples)} examples tokenize to the modal length "
            f"{modal}; at least 2 are needed for any column to be seen to vary"
        )

    matrix = list(zip(*keep))  # column-major: matrix[c] is the tuple of values at c
    frame_columns = tuple(c for c, col in enumerate(matrix) if len(set(col)) == 1)
    slot_columns = [c for c in range(modal) if c not in set(frame_columns)]

    # Tie columns by their *value vector*: two columns are one slot exactly when they
    # hold the same token in every kept example. Grouped by the vector itself so the
    # relation is transitive by construction rather than by a pairwise sweep.
    groups: dict[tuple[int, ...], list[int]] = {}
    for c in slot_columns:
        groups.setdefault(matrix[c], []).append(c)

    slots = tuple(
        Slot(
            columns=tuple(columns),
            values=tuple(dict.fromkeys(vector)),  # distinct, in order of appearance
            length=modal,
        )
        for vector, columns in sorted(groups.items(), key=lambda kv: kv[1][0])
    )

    positions = tuple(label_for(c, modal) for c in slot_columns)
    if END not in positions:
        positions = positions + (END,)

    return Structure(
        length=modal,
        base_row=keep[0],
        frame_columns=frame_columns,
        slots=slots,
        positions=positions,
        n_examples_given=len(examples),
        n_examples_kept=len(keep),
        dropped=tuple(dropped),
    )


def round_trips(model: Any, row: Sequence[int]) -> bool:
    """Would this tokenizer produce exactly this row from its own decoding of it?

    The mechanized stand-in for every hand-written tokenizer filter in the three task
    modules. `greater_than.py` checks that `" {century}{yy}"` splits as two tokens and
    that nouns are single tokens; both are special cases of this one question, and
    neither of them needs to be asked in terms of centuries or nouns.

    BOS is stripped before decoding and expected back after re-encoding, because that
    is what `to_tokens` does to any string it is handed.
    """
    text = model.to_string(torch.tensor(list(row[1:])))
    return _tokenize(model, text) == tuple(int(t) for t in row)


@dataclass
class Generated:
    """A generated clean batch and an account of what was thrown away making it."""

    rows: tuple[tuple[int, ...], ...]
    attempts: int
    rejected_round_trip: int
    rejected_duplicate: int
    requested: int

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def round_trip_rate(self) -> float:
        return self.rejected_round_trip / self.attempts if self.attempts else 0.0

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "produced": self.count,
            "attempts": self.attempts,
            "rejected_round_trip": self.rejected_round_trip,
            "rejected_duplicate": self.rejected_duplicate,
            "round_trip_rejection_rate": self.round_trip_rate,
        }


def generate(
    model: Any, structure: Structure, n: int, seed: int, budget: int = ATTEMPT_BUDGET
) -> Generated:
    """Section 3.2 of the plan: slots in, a batch of distinct clean prompts out.

    Each slot is sampled independently from its own observed values, which is what
    makes the vocabularies mechanizable and is also the step that can produce nonsense:
    nothing here knows that a century and a start year have to tokenize together. The
    round-trip filter catches the cases where that goes wrong at the token level, and
    the rejection rate it reports is the measurement of how often it does.

    Returns however many distinct rows it managed. Falling short of `n` is a finding
    about how small the human's input was, not an error, and the caller reports it.
    """
    rng = random.Random(f"{seed}:generate")
    seen: set[tuple[int, ...]] = set()
    rows: list[tuple[int, ...]] = []
    attempts = rejected_rt = rejected_dup = 0

    while len(rows) < n and attempts < budget * n:
        attempts += 1
        row = list(structure.base_row)
        for slot in structure.slots:
            value = rng.choice(slot.values)
            for column in slot.columns:
                row[column] = value
        candidate = tuple(row)
        if candidate in seen:
            rejected_dup += 1
            continue
        if not round_trips(model, candidate):
            rejected_rt += 1
            seen.add(candidate)  # do not pay to re-check a row already rejected
            continue
        seen.add(candidate)
        rows.append(candidate)

    return Generated(
        rows=tuple(rows),
        attempts=attempts,
        rejected_round_trip=rejected_rt,
        rejected_duplicate=rejected_dup,
        requested=n,
    )


def propose(structure: Structure) -> tuple[Proposal, ...]:
    """Section 3.3 of the plan: one counterfactual per slot, plus one per tied column.

    The set is fully determined by the induced structure — there is no cutoff, no
    ranking and nothing to tune. Which of them becomes primary is decided later, by
    measurement, in `autotask.select_primary`.
    """
    proposals: list[Proposal] = []
    for index, slot in enumerate(structure.slots):
        proposals.append(
            Proposal(
                name=f"resample_{slot.label}",
                kind="resample",
                slot_index=index,
                columns=slot.columns,
                label=slot.label,
                breaks=(
                    f"redraws the value at {'/'.join(str(c) for c in slot.columns)} "
                    f"from the {len(slot.values)} values observed there"
                ),
            )
        )
    for index, slot in enumerate(structure.slots):
        if not slot.is_tied:
            continue
        for column in slot.columns:
            label = label_for(column, structure.length)
            proposals.append(
                Proposal(
                    name=f"desync_{label}",
                    kind="desync",
                    slot_index=index,
                    columns=(column,),
                    label=label,
                    breaks=(
                        f"redraws column {column} alone, breaking its agreement with "
                        f"{'/'.join(str(c) for c in slot.columns if c != column)}"
                    ),
                )
            )
    return tuple(proposals)


def apply_proposal(
    proposal: Proposal, structure: Structure, rows: Sequence[Sequence[int]], seed: int
) -> tuple[list[tuple[int, ...]], list[int]]:
    """Build the corrupted counterpart of every clean row under one proposal.

    The redrawn value is constrained to differ from the clean one, so no prompt is
    silently left uncorrupted — the same guarantee `corruption.random_vocab_corruption`
    makes for its own draws, and for the same reason: a no-op draw would quietly weaken
    the measured effect of every head.

    The RNG is seeded from the scheme's *name*, so each scheme draws its own stream and
    none of them disturbs the clean sample. `greater_than.py` arranges the same thing
    by hand with a second `random.Random`, and Phase 8's `check_schemes.py` exists
    because getting it wrong makes a cross-scheme comparison read a different prompt
    set as well as a different counterfactual.
    """
    rng = random.Random(f"{seed}:{proposal.name}")
    slot = structure.slots[proposal.slot_index]
    out: list[tuple[int, ...]] = []
    changed: list[int] = []

    for row in rows:
        current = row[proposal.columns[0]]
        options = [v for v in slot.values if v != current]
        if not options:
            raise ValueError(
                f"{proposal.name}: slot {slot.label} has no value other than {current}; "
                "a counterfactual cannot be built from a single observed value"
            )
        value = rng.choice(options)
        corrupted = list(row)
        for column in proposal.columns:
            corrupted[column] = value
        out.append(tuple(corrupted))
        changed.append(proposal.columns[0])

    return out, changed


def corrupted_round_trip_rate(model: Any, rows: Sequence[Sequence[int]]) -> float:
    """What fraction of a corrupted batch the tokenizer would not itself produce.

    Reported, never filtered. A corrupted row has to stay token-aligned with the clean
    row it is paired with, so it cannot be rejected and redrawn without breaking the
    pairing every patch depends on. `greater_than.py` avoids the problem by hand — its
    `_alt_century` only draws centuries that tokenize with the given start year — and
    the induction has no way to know that, so it measures the cost instead of hiding
    it.
    """
    if not rows:
        return 0.0
    bad = sum(0 if round_trips(model, row) else 1 for row in rows)
    return bad / len(rows)
