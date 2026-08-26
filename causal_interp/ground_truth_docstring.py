"""The published docstring circuit, phase 7's ground truth."""

from __future__ import annotations

Head = tuple[int, int]

DOCSTRING_CIRCUIT: dict[str, tuple[Head, ...]] = {
    "argument mover": ((3, 0), (3, 6)),
    "fuzzy previous token": ((2, 0),),
    "induction + fuzzy previous token": ((1, 4),),
    "duplicate token": ((0, 5), (1, 2)),
}

# alias under the neutral name `comparison.py` looks for
# have to know which circuit it was handed.
CIRCUIT = DOCSTRING_CIRCUIT

PUBLISHED_HEAD_COUNT = 6

ALL_HEADS: frozenset[Head] = frozenset(h for heads in CIRCUIT.values() for h in heads)

HEAD_TO_CLASS: dict[Head, str] = {h: cls for cls, heads in CIRCUIT.items() for h in heads}

assert len(ALL_HEADS) == PUBLISHED_HEAD_COUNT, (
    f"expected {PUBLISHED_HEAD_COUNT} distinct heads, found {len(ALL_HEADS)}"
)
assert sum(len(h) for h in CIRCUIT.values()) == PUBLISHED_HEAD_COUNT, "a head is listed in two classes"

AUXILIARY_HEADS: tuple[Head, ...] = ((0, 2), (0, 4))

EXTENDED_CIRCUIT: dict[str, tuple[Head, ...]] = {
    **CIRCUIT,
    "auxiliary (published as patching-invisible)": AUXILIARY_HEADS,
}

CLASS_EXPECTED_POSITION: dict[str, str] = {
    "argument mover": "END",
    "fuzzy previous token": "C_def",
    "induction + fuzzy previous token": "comma_B",
    "duplicate token": "B_def",
}

CLASS_RECEIVER_SPEC: dict[str, tuple[str, str] | None] = {
    "argument mover": ("q", "END"),
    "fuzzy previous token": ("v", "comma_B"),
    "induction + fuzzy previous token": ("v", "B_def"),
    "duplicate token": None,
}

CLASS_ALTERNATIVE_SPECS: dict[str, tuple[tuple[str, str], ...]] = {
    "argument mover": (("k", "C_def"), ("v", "C_def")),
    "induction + fuzzy previous token": (("k", "B_doc"),),
}

assert set(CLASS_RECEIVER_SPEC) == set(CIRCUIT), "receiver specs must cover every class"
assert set(CLASS_EXPECTED_POSITION) == set(CIRCUIT), "expected positions must cover every class"
assert set(CLASS_ALTERNATIVE_SPECS) <= set(CIRCUIT), "alternative specs name an unknown class"


def receiver_spec(head: Head) -> tuple[str, str] | None:
    cls = classify(head)
    return CLASS_RECEIVER_SPEC.get(cls) if cls else None


def alternative_specs(head: Head) -> tuple[tuple[str, str], ...]:
    cls = classify(head)
    return CLASS_ALTERNATIVE_SPECS.get(cls, ()) if cls else ()


def classify(head: Head) -> str | None:
    return HEAD_TO_CLASS.get(head)


def describe(head: Head) -> str:
    cls = classify(head)
    return f"{head[0]}.{head[1]} ({cls or 'not in circuit'})"
