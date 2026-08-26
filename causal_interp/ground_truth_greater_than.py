"""the published greater-than circuit, phase 6's ground truth."""

from __future__ import annotations

Head = tuple[int, int]

GREATER_THAN_CIRCUIT: dict[str, tuple[Head, ...]] = {
    "year head -> MLP 9": ((9, 1),),
    "year head -> MLP 8": ((8, 11), (8, 8), (7, 10), (6, 9), (5, 5), (5, 1)),
}

# alias under the neutral name `comparison.py` looks for
# have to know which circuit it was handed.
CIRCUIT = GREATER_THAN_CIRCUIT

PUBLISHED_HEAD_COUNT = 7

ALL_HEADS: frozenset[Head] = frozenset(h for heads in CIRCUIT.values() for h in heads)

HEAD_TO_CLASS: dict[Head, str] = {h: cls for cls, heads in CIRCUIT.items() for h in heads}

assert len(ALL_HEADS) == PUBLISHED_HEAD_COUNT, (
    f"expected {PUBLISHED_HEAD_COUNT} distinct heads, found {len(ALL_HEADS)}"
)
assert sum(len(h) for h in CIRCUIT.values()) == PUBLISHED_HEAD_COUNT, "a head is listed in two classes"

PUBLISHED_MLPS: tuple[int, ...] = (8, 9, 10, 11)

APPENDIX_UPSTREAM_HEADS: tuple[Head, ...] = ((0, 1), (0, 3), (0, 5))

EXTENDED_CIRCUIT: dict[str, tuple[Head, ...]] = {
    **CIRCUIT,
    "appendix upstream": APPENDIX_UPSTREAM_HEADS,
}

CLASS_EXPECTED_POSITION: dict[str, str] = {
    "year head -> MLP 9": "END",
    "year head -> MLP 8": "END",
}

CLASS_RECEIVER_SPEC: dict[str, tuple[str, str] | None] = {
    "year head -> MLP 9": ("v", "YY"),
    "year head -> MLP 8": ("v", "YY"),
}

assert set(CLASS_RECEIVER_SPEC) == set(CIRCUIT), "receiver specs must cover every class"
assert set(CLASS_EXPECTED_POSITION) == set(CIRCUIT), "expected positions must cover every class"


def receiver_spec(head: Head) -> tuple[str, str] | None:
    cls = classify(head)
    return CLASS_RECEIVER_SPEC.get(cls) if cls else None


def classify(head: Head) -> str | None:
    return HEAD_TO_CLASS.get(head)


def describe(head: Head) -> str:
    cls = classify(head)
    return f"{head[0]}.{head[1]} ({cls or 'not in circuit'})"
