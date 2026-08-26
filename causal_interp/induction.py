"""inducing a task's structure from a handful of example prompts, phase 10."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor

ATTEMPT_BUDGET = 50

END = "END"

FILTER_LENGTH = "length"
FILTER_SHAPE = "shape"
FILTER_MODES = (FILTER_LENGTH, FILTER_SHAPE)


def shape_signature(row: Sequence[int]) -> tuple[int, ...]:
    first: dict[int, int] = {}
    return tuple(first.setdefault(token, column) for column, token in enumerate(row))


def label_for(index: int, length: int) -> str:
    return END if index == length - 1 else f"t{index}"


@dataclass(frozen=True)
class Slot:
    """one varying part of the prompt: where it appears, and what was observed in it."""

    columns: tuple[int, ...]
    values: tuple[int, ...]
    length: int

    @property
    def anchor(self) -> int:
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
    """one counterfactual the induction proposes, before anything has been measured."""

    name: str
    kind: str
    slot_index: int
    columns: tuple[int, ...]
    label: str
    breaks: str


@dataclass
class Structure:
    """What the induction found: the frame, the slots, and what it had to throw away."""

    length: int
    base_row: tuple[int, ...]
    frame_columns: tuple[int, ...]
    slots: tuple[Slot, ...]
    positions: tuple[str, ...]
    n_examples_given: int
    n_examples_kept: int
    dropped: tuple[dict, ...] = ()
    filter_mode: str = FILTER_LENGTH

    @property
    def slot_columns(self) -> tuple[int, ...]:
        return tuple(sorted(c for slot in self.slots for c in slot.columns))

    def position_index(self, label: str) -> int:
        if label == END:
            return self.length - 1
        return int(label[1:])

    def as_dict(self, decode: Any = None) -> dict:
        return {
            "length": self.length,
            "filter_mode": self.filter_mode,
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
    return tuple(int(t) for t in model.to_tokens(text)[0])


def _keep_by_length(
    examples: Sequence[str], rows: Sequence[tuple[int, ...]]
) -> tuple[list[tuple[int, ...]], list[dict]]:
    """section 3.1 as pre-registered: keep the rows whose token length is modal."""
    lengths = Counter(len(row) for row in rows)

    modal = max(lengths, key=lambda L: (lengths[L], L))

    keep, dropped = [], []
    for i, (text, row) in enumerate(zip(examples, rows)):
        if len(row) == modal:
            keep.append(row)
        else:
            dropped.append(
                {"index": i, "length": len(row), "modal": modal, "reason": "length", "text": text}
            )
    return keep, dropped


def _keep_by_shape(
    examples: Sequence[str], rows: Sequence[tuple[int, ...]]
) -> tuple[list[tuple[int, ...]], list[dict]]:
    """The amendment's repair: keep the largest group of rows sharing a column shape."""
    groups: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault((len(row), shape_signature(row)), []).append(i)
    best = max(groups.values(), key=lambda members: (len(members), -members[0]))
    chosen = set(best)

    keep = [rows[i] for i in best]
    dropped = [
        {
            "index": i,
            "length": len(rows[i]),
            "modal": len(keep[0]),
            "reason": "shape",
            "text": examples[i],
        }
        for i in range(len(rows))
        if i not in chosen
    ]
    return keep, dropped


def induce(
    model: Any, examples: Sequence[str], *, filter_mode: str = FILTER_LENGTH
) -> Structure:
    """Section 3.1 of the plan: example strings in, slot structure out."""
    if filter_mode not in FILTER_MODES:
        raise ValueError(f"filter_mode must be one of {FILTER_MODES}, got {filter_mode!r}")
    if len(examples) < 2:
        raise ValueError(f"induction needs at least 2 examples, got {len(examples)}")

    rows = [_tokenize(model, text) for text in examples]
    if filter_mode == FILTER_LENGTH:
        keep, dropped = _keep_by_length(examples, rows)
    else:
        keep, dropped = _keep_by_shape(examples, rows)

    if len(keep) < 2:
        raise ValueError(
            f"only {len(keep)} of {len(examples)} examples survive the {filter_mode} "
            "filter; at least 2 are needed for any column to be seen to vary"
        )
    modal = len(keep[0])

    matrix = list(zip(*keep))  # column-major: matrix[c] is the tuple of values at c
    frame_columns = tuple(c for c, col in enumerate(matrix) if len(set(col)) == 1)
    slot_columns = [c for c in range(modal) if c not in set(frame_columns)]

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
        filter_mode=filter_mode,
    )


def round_trips(model: Any, row: Sequence[int]) -> bool:
    text = model.to_string(torch.tensor(list(row[1:])))
    return _tokenize(model, text) == tuple(int(t) for t in row)


@dataclass
class Generated:
    """a generated clean batch and an account of what was thrown away making it."""

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
    """section 3.2 of the plan: slots in, a batch of distinct clean prompts out."""
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
            seen.add(candidate)  # dont pay to re-check a row already rejected
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
    """Section 3.3 of the plan: one counterfactual per slot, plus one per tied column."""
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
    """build the corrupted counterpart of every clean row under one proposal."""
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
    if not rows:
        return 0.0
    bad = sum(0 if round_trips(model, row) else 1 for row in rows)
    return bad / len(rows)
