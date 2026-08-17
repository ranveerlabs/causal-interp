"""The published docstring circuit — Phase 7's ground truth.

Source: Heimersheim and Janiak (2023), "A circuit for Python docstrings in a
4-layer attention-only transformer", https://www.lesswrong.com/posts/u6KXXmKFbXfWzoAXn

Model: `attn-only-4l` (`NeelNanda/Attn_Only_4L512W_C4_Code`) — 4 layers, 8 heads,
no MLP blocks. This is the first ground truth in this project that does not
describe GPT-2 small.

Kept separate from `ground_truth.py` and `ground_truth_greater_than.py` for the
reason Phase 6 gave: three published circuits in one module would make it easy to
score a run against a union by accident. `comparison.py` takes whichever it is
given. Like the other two, this module is inert data — nothing here is derived from
any run in this project, or the comparison would be circular.

The head assignments were cross-checked against released benchmark code, as the
other two were. `acdc/docstring/utils.py` in ArthurConmy/Automatic-Circuit-Discovery
contains `get_docstring_subgraph_true_edges()`, commented "the manual graph, from
Stefan", which builds a 37-edge graph and asserts that count against "the value in
the docstring appendix of the manual circuit". The heads appearing in those 37
edges are 0.5, 1.2, 1.4, 2.0, 3.0 and 3.6 — the same six the post names in prose,
and no others.
"""

from __future__ import annotations

Head = tuple[int, int]

# Class -> heads, as (layer, head), 0-indexed. The classes are the post's own role
# names. Its one-sentence summary of the circuit is:
#
#   "mainly heads 1.4, 2.0, 3.0, and 3.6 for moving the information [...] head 0.5
#    for token transformation and some additional support, as well as 1.2 for
#    additional support"
#
# 1.4 and 2.0 are split into separate classes despite sharing a role name, because
# they act at different token positions and so have different receiver specs.
DOCSTRING_CIRCUIT: dict[str, tuple[Head, ...]] = {
    "argument mover": ((3, 0), (3, 6)),
    "fuzzy previous token": ((2, 0),),
    "induction + fuzzy previous token": ((1, 4),),
    "duplicate token": ((0, 5), (1, 2)),
}

# Alias under the neutral name `comparison.py` looks for, so that module does not
# have to know which circuit it was handed.
CIRCUIT = DOCSTRING_CIRCUIT

PUBLISHED_HEAD_COUNT = 6

ALL_HEADS: frozenset[Head] = frozenset(h for heads in CIRCUIT.values() for h in heads)

HEAD_TO_CLASS: dict[Head, str] = {h: cls for cls, heads in CIRCUIT.items() for h in heads}

assert len(ALL_HEADS) == PUBLISHED_HEAD_COUNT, (
    f"expected {PUBLISHED_HEAD_COUNT} distinct heads, found {len(ALL_HEADS)}"
)
assert sum(len(h) for h in CIRCUIT.values()) == PUBLISHED_HEAD_COUNT, "a head is listed in two classes"


# The post names two further heads and says, in the same breath, that patching will
# not find them:
#
#   "At least Previous Token Head 0.2 and Positional Information Head 0.4 (and
#    potentially more) are also essential parts of the circuit, but won't show
#    differ in our patching experiments."
#
# They are absent from the released 37-edge graph. Recorded here, and fixed in
# results/PHASE7_PLAN.md before any run, so the secondary 8-head comparison exists
# in advance and cannot be adopted afterwards on the strength of scoring better.
# The primary comparison is the 6.
AUXILIARY_HEADS: tuple[Head, ...] = ((0, 2), (0, 4))

EXTENDED_CIRCUIT: dict[str, tuple[Head, ...]] = {
    **CIRCUIT,
    "auxiliary (published as patching-invisible)": AUXILIARY_HEADS,
}


# Where each class is expected to act, in the position vocabulary used by
# causal_interp.docstring. Used only to annotate the comparison — a head is never
# counted as a match or a miss on the basis of position.
CLASS_EXPECTED_POSITION: dict[str, str] = {
    "argument mover": "END",
    "fuzzy previous token": "C_def",
    "induction + fuzzy previous token": "comma_B",
    "duplicate token": "B_def",
}


# The receiver specification the sources name for each class: which input, at which
# token position. The input comes from the released edge graph, the position from
# the post's account of where each head's query sits and what it attends to.
#
#   argument mover   3.hook_q_input <- 1.attn.hook_result H4, query at the final
#                    `:param` token                                   -> q@END
#   2.0              2.hook_v_input H0 <- 1.attn.hook_result H4, attends from
#                    C_def to `,_B`, so the value it reads sits at `,_B`
#                                                                     -> v@comma_B
#   1.4              1.hook_v_input H4 <- 0.attn.hook_result H5, attends from
#                    `,_B` to B_def, so the value it reads sits at B_def
#                                                                     -> v@B_def
#   duplicate token  no single published position. 0.5 acts at several positions at
#                    once ("almost entirely explained as transforming the
#                    embedding"), and 1.2 is defined by its effect on *other* heads'
#                    keys rather than by an input of its own.          -> None
#
# So 4 of the 6 heads are scoreable on receiver-spec rediscovery and 2 are not
# measured, rather than measured-and-failed. `None` is the same signal
# `ground_truth.py` uses for IOI's classes with no published spec.
CLASS_RECEIVER_SPEC: dict[str, tuple[str, str] | None] = {
    "argument mover": ("q", "END"),
    "fuzzy previous token": ("v", "comma_B"),
    "induction + fuzzy previous token": ("v", "B_def"),
    "duplicate token": None,
}

# Also published, and fixed here before the run so they cannot be introduced
# afterwards if the primary specs score badly. The movers additionally receive keys
# from 2.0 and 1.2 at C_def and values from 0.5 at C_def; 1.4 has a docstring-side
# key input at B_doc. Reported as a separate line, never merged into the headline.
CLASS_ALTERNATIVE_SPECS: dict[str, tuple[tuple[str, str], ...]] = {
    "argument mover": (("k", "C_def"), ("v", "C_def")),
    "induction + fuzzy previous token": (("k", "B_doc"),),
}

assert set(CLASS_RECEIVER_SPEC) == set(CIRCUIT), "receiver specs must cover every class"
assert set(CLASS_EXPECTED_POSITION) == set(CIRCUIT), "expected positions must cover every class"
assert set(CLASS_ALTERNATIVE_SPECS) <= set(CIRCUIT), "alternative specs name an unknown class"


def receiver_spec(head: Head) -> tuple[str, str] | None:
    """The published (input, position) this head receives on, or None if unspecified."""
    cls = classify(head)
    return CLASS_RECEIVER_SPEC.get(cls) if cls else None


def alternative_specs(head: Head) -> tuple[tuple[str, str], ...]:
    """Further published (input, position) pairs for this head, possibly empty."""
    cls = classify(head)
    return CLASS_ALTERNATIVE_SPECS.get(cls, ()) if cls else ()


def classify(head: Head) -> str | None:
    """Return the published class of `head`, or None if it is not in the circuit."""
    return HEAD_TO_CLASS.get(head)


def describe(head: Head) -> str:
    """Human-readable label, e.g. '3.0 (argument mover)'."""
    cls = classify(head)
    return f"{head[0]}.{head[1]} ({cls or 'not in circuit'})"
