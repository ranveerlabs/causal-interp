"""The published greater-than circuit — Phase 6's ground truth.

Source: Hanna, Liu, Variengien (2023), "How does GPT-2 compute greater-than?:
Interpreting mathematical abilities in a pre-trained language model",
arXiv:2305.00586, NeurIPS 2023.

Deliberately kept separate from `ground_truth.py` rather than merged into it. The
two circuits are independent published claims about the same model, and a single
module holding both would make it easy to score a run against the union by
accident. `comparison.py` takes whichever one it is given.

Structured identically to `ground_truth.py`: a class -> heads mapping, a published
count asserted against it, and per-class annotations for expected position and
receiver specification. Like that module, this one is inert data. Nothing here is
derived from any run in this project — if it were, the comparison would be circular.

The head assignments were cross-checked against the authors' own code release,
`hannamw/gpt2-greater-than`, whose circuit head list is
`[(9, 1), (8, 11), (7, 10), (6, 9), (5, 5), (8, 8), (5, 1)]` — the same seven heads
the paper names, and the same seven listed below.
"""

from __future__ import annotations

Head = tuple[int, int]

# Class -> heads, as (layer, head), 0-indexed. The paper groups the circuit's
# attention heads by which MLP they feed:
#
#   "MLP 9 relies on a9.h1, while MLP 8 relies on a8.h11, a8.h8, a7.h10, a6.h9,
#    a5.h5, and a5.h1"
#
# Functionally the seven do the same job — they attend to the YY position and
# communicate the start year onward — so the split is by destination, not by role.
GREATER_THAN_CIRCUIT: dict[str, tuple[Head, ...]] = {
    "year head -> MLP 9": ((9, 1),),
    "year head -> MLP 8": ((8, 11), (8, 8), (7, 10), (6, 9), (5, 5), (5, 1)),
}

# Alias under the neutral name `comparison.py` looks for, so that module does not
# have to know which circuit it was handed.
CIRCUIT = GREATER_THAN_CIRCUIT

PUBLISHED_HEAD_COUNT = 7

ALL_HEADS: frozenset[Head] = frozenset(h for heads in CIRCUIT.values() for h in heads)

HEAD_TO_CLASS: dict[Head, str] = {h: cls for cls, heads in CIRCUIT.items() for h in heads}

assert len(ALL_HEADS) == PUBLISHED_HEAD_COUNT, (
    f"expected {PUBLISHED_HEAD_COUNT} distinct heads, found {len(ALL_HEADS)}"
)
assert sum(len(h) for h in CIRCUIT.values()) == PUBLISHED_HEAD_COUNT, "a head is listed in two classes"


# The published circuit is not only attention heads. These four MLPs are its
# centre — "MLPs 9, 10, and 11 appear to compute the greater-than operation in
# tandem, and in steps", with MLP 8 "contributing mainly indirectly, via the other
# MLPs". They are swept and reported separately; they are not attention heads and
# take no part in the head-level comparison.
PUBLISHED_MLPS: tuple[int, ...] = (8, 9, 10, 11)


# Appendix B path-patches what the seven circuit heads themselves depend on and
# reports "mostly MLPs 0-3, as well as a0.h5, a0.h3, and a0.h1". The paper presents
# these as upstream dependencies in an appendix rather than as members of the
# circuit it reports, and the authors' code excludes them.
#
# They are recorded here, and fixed in results/PHASE6_PLAN.md before any run, so
# that the secondary 10-head comparison exists in advance and cannot be adopted
# afterwards on the strength of scoring better. The primary comparison is the 7.
APPENDIX_UPSTREAM_HEADS: tuple[Head, ...] = ((0, 1), (0, 3), (0, 5))

EXTENDED_CIRCUIT: dict[str, tuple[Head, ...]] = {
    **CIRCUIT,
    "appendix upstream": APPENDIX_UPSTREAM_HEADS,
}


# Where each class is expected to act, in the position vocabulary used by
# causal_interp.greater_than. The circuit heads attend *from* the final token *to*
# the YY position, so what they write lands at END. Used only to annotate the
# comparison — a head is never counted as a match or a miss on the basis of position.
CLASS_EXPECTED_POSITION: dict[str, str] = {
    "year head -> MLP 9": "END",
    "year head -> MLP 8": "END",
}


# The receiver specification the paper names for each class.
#
# This circuit is better served here than IOI was. For IOI only three of seven
# classes had a published receiver spec, so four were unscoreable. Here Appendix B
# states one for all seven heads at once:
#
#   "The most important influences on these heads are the influences on their
#    values at the YY position."
#
# so every circuit head has a published (input, position) the search can be scored
# against.
CLASS_RECEIVER_SPEC: dict[str, tuple[str, str] | None] = {
    "year head -> MLP 9": ("v", "YY"),
    "year head -> MLP 8": ("v", "YY"),
}

assert set(CLASS_RECEIVER_SPEC) == set(CIRCUIT), "receiver specs must cover every class"
assert set(CLASS_EXPECTED_POSITION) == set(CIRCUIT), "expected positions must cover every class"


def receiver_spec(head: Head) -> tuple[str, str] | None:
    """The published (input, position) this head receives on, or None if unspecified."""
    cls = classify(head)
    return CLASS_RECEIVER_SPEC.get(cls) if cls else None


def classify(head: Head) -> str | None:
    """Return the published class of `head`, or None if it is not in the circuit."""
    return HEAD_TO_CLASS.get(head)


def describe(head: Head) -> str:
    """Human-readable label, e.g. '9.1 (year head -> MLP 9)'."""
    cls = classify(head)
    return f"{head[0]}.{head[1]} ({cls or 'not in circuit'})"
