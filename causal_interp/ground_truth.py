"""the published IOI circuit, the ground truth phase 1 is validated against."""

from __future__ import annotations

Head = tuple[int, int]

# class -> heads, as (layer, head). layers and heads are 0-indexed.
IOI_CIRCUIT: dict[str, tuple[Head, ...]] = {
    "name mover": ((9, 9), (9, 6), (10, 0)),
    "backup name mover": ((10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)),
    "negative name mover": ((10, 7), (11, 10)),
    "s-inhibition": ((7, 3), (7, 9), (8, 6), (8, 10)),
    "induction": ((5, 5), (5, 8), (5, 9), (6, 9)),
    "duplicate token": ((0, 1), (0, 10), (3, 0)),
    "previous token": ((2, 2), (4, 11)),
}

CIRCUIT = IOI_CIRCUIT

# the paper's headline count
# the comparison this whole phase rests on.
PUBLISHED_HEAD_COUNT = 26

ALL_HEADS: frozenset[Head] = frozenset(h for heads in IOI_CIRCUIT.values() for h in heads)

HEAD_TO_CLASS: dict[Head, str] = {h: cls for cls, heads in IOI_CIRCUIT.items() for h in heads}

assert len(ALL_HEADS) == PUBLISHED_HEAD_COUNT, (
    f"expected {PUBLISHED_HEAD_COUNT} distinct heads, found {len(ALL_HEADS)}"
)
assert sum(len(h) for h in IOI_CIRCUIT.values()) == PUBLISHED_HEAD_COUNT, "a head is listed in two classes"

CLASS_EXPECTED_POSITION: dict[str, str] = {
    "name mover": "END",
    "backup name mover": "END",
    "negative name mover": "END",
    "s-inhibition": "END",
    "induction": "S2",
    "duplicate token": "S2",
    "previous token": "S1+1",
}

CLASS_RECEIVER_SPEC: dict[str, tuple[str, str] | None] = {
    "name mover": ("q", "END"),
    "backup name mover": ("q", "END"),
    "negative name mover": ("q", "END"),
    "s-inhibition": ("v", "S2"),
    "induction": ("k", "S1+1"),
    "duplicate token": None,
    "previous token": None,
}

assert set(CLASS_RECEIVER_SPEC) == set(IOI_CIRCUIT), "receiver specs must cover every class"


def receiver_spec(head: Head) -> tuple[str, str] | None:
    cls = classify(head)
    return CLASS_RECEIVER_SPEC.get(cls) if cls else None


def classify(head: Head) -> str | None:
    return HEAD_TO_CLASS.get(head)


def describe(head: Head) -> str:
    cls = classify(head)
    return f"{head[0]}.{head[1]} ({cls or 'not in circuit'})"
