"""The published IOI circuit — the ground truth Phase 1 is validated against.

Source: Wang, Variengien, Conmy, Shlegeris, Steinhardt (2022),
"Interpretability in the Wild: a Circuit for Indirect Object Identification in
GPT-2 small", arXiv:2211.00593. The paper reports 26 attention heads in 7
classes (Figure 2).

The head assignments below were cross-checked against the authors' own
implementation, `easy_transformer/ioi_circuit_extraction.py` in
redwoodresearch/Easy-Transformer, whose `CIRCUIT` dict matches. One naming
difference: the authors' code merges name movers and backup name movers into a
single "name mover" list; the paper separates them, and so does this file.

This module is deliberately inert data. Nothing here is derived from our own
runs — if it were, the comparison would be circular.
"""

from __future__ import annotations

Head = tuple[int, int]

# Class -> heads, as (layer, head). Layers and heads are 0-indexed.
IOI_CIRCUIT: dict[str, tuple[Head, ...]] = {
    "name mover": ((9, 9), (9, 6), (10, 0)),
    "backup name mover": ((10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)),
    "negative name mover": ((10, 7), (11, 10)),
    "s-inhibition": ((7, 3), (7, 9), (8, 6), (8, 10)),
    "induction": ((5, 5), (5, 8), (5, 9), (6, 9)),
    "duplicate token": ((0, 1), (0, 10), (3, 0)),
    "previous token": ((2, 2), (4, 11)),
}

# The paper's headline count. Asserted so a typo above cannot silently weaken
# the comparison this whole phase rests on.
PUBLISHED_HEAD_COUNT = 26

ALL_HEADS: frozenset[Head] = frozenset(h for heads in IOI_CIRCUIT.values() for h in heads)

HEAD_TO_CLASS: dict[Head, str] = {h: cls for cls, heads in IOI_CIRCUIT.items() for h in heads}

assert len(ALL_HEADS) == PUBLISHED_HEAD_COUNT, (
    f"expected {PUBLISHED_HEAD_COUNT} distinct heads, found {len(ALL_HEADS)}"
)
assert sum(len(h) for h in IOI_CIRCUIT.values()) == PUBLISHED_HEAD_COUNT, "a head is listed in two classes"


# Where each class is expected to act, in the position vocabulary used by
# causal_interp.ioi. Used only to annotate the comparison — a head is never
# counted as a match or a miss on the basis of position.
CLASS_EXPECTED_POSITION: dict[str, str] = {
    "name mover": "END",
    "backup name mover": "END",
    "negative name mover": "END",
    "s-inhibition": "END",
    "induction": "S2",
    "duplicate token": "S2",
    "previous token": "S1+1",
}


def classify(head: Head) -> str | None:
    """Return the published class of `head`, or None if it is not in the circuit."""
    return HEAD_TO_CLASS.get(head)


def describe(head: Head) -> str:
    """Human-readable label, e.g. '9.9 (name mover)' or '1.5 (not in circuit)'."""
    cls = classify(head)
    return f"{head[0]}.{head[1]} ({cls or 'not in circuit'})"
