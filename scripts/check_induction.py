"""Known-answer tests for Phase 10's induction and auto-built task.

    python scripts/check_induction.py        # expect: INDUCTION OK

The counterpart of `check_patching.py` and `check_schemes.py`: cases where the correct
answer follows from how the experiment is built rather than from anything the model
does. Structure induction is easy to get subtly wrong in ways that still produce a
plausible-looking task — a tie missed, a slot vocabulary silently shared between two
schemes, a clean sample that shifts when the counterfactual changes — and every one of
those would corrupt a cross-scheme comparison without raising anything.

The synthetic frame used for the structural checks is

    Then {name} went {place} and {name} slept

which has one tied slot appearing at two token positions, one free slot, and a frame
column in the final position. The right answer for every structural assertion below is
readable off that string; none of it depends on GPT-2 having any particular behaviour.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_interp import autotask, induction
from causal_interp.model import load

NAMES = ("Mary", "John", "Tom", "Paul")
PLACES = ("home", "north", "south", "east")
FRAME = "Then {name} went {place} and {name} slept"

# One example whose name does not tokenize to a single token, so the row is longer than
# the rest. It must be dropped by the modal-length filter and counted.
LONG_EXAMPLE = "Then Bartholomew went home and Bartholomew slept"

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"   {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def main() -> int:
    print("loading gpt2-small ...")
    model = load("gpt2-small")

    examples = [
        FRAME.format(name=name, place=place)
        for name in NAMES
        for place in PLACES
    ]

    # -- 1. induction ------------------------------------------------------
    print("\n1. structure induction on a synthetic frame with a known answer")
    structure = induction.induce(model, examples)

    check("modal row length is 8", structure.length == 8, f"got {structure.length}")
    check("every example kept", structure.n_examples_kept == len(examples),
          f"{structure.n_examples_kept}/{len(examples)}")
    check("exactly two slots", len(structure.slots) == 2,
          f"got {[s.label for s in structure.slots]}")

    tied = structure.slots[0]
    free = structure.slots[1]
    check("the repeated slot is tied across columns 2 and 6", tied.columns == (2, 6),
          f"got {tied.columns}")
    check("the tied slot records 4 observed values", len(tied.values) == len(NAMES),
          f"got {len(tied.values)}")
    check("the free slot occupies column 4 alone", free.columns == (4,), f"got {free.columns}")
    check("the free slot records 4 observed values", len(free.values) == len(PLACES),
          f"got {len(free.values)}")
    check("frame columns are the other four", structure.frame_columns == (0, 1, 3, 5, 7),
          f"got {structure.frame_columns}")
    check("positions are the slot columns plus END",
          structure.positions == ("t2", "t4", "t6", "END"), f"got {structure.positions}")
    check("END resolves to the last index", structure.position_index("END") == 7)

    # -- 2. the modal-length filter ---------------------------------------
    print("\n2. an example that does not fit is dropped and counted")
    with_long = induction.induce(model, examples + [LONG_EXAMPLE])
    check("one example dropped", len(with_long.dropped) == 1,
          f"got {len(with_long.dropped)}")
    check("the dropped one is the long name",
          bool(with_long.dropped) and with_long.dropped[0]["text"] == LONG_EXAMPLE)
    check("the surviving structure is unchanged",
          with_long.slots == structure.slots and with_long.positions == structure.positions)

    # -- 2b. the amendment's shape filter ----------------------------------
    print("\n2b. a same-length example that breaks the tie: length keeps it, shape drops it")
    # Same token count as the rest, but the two name columns disagree — the synthetic
    # analogue of `" 1509"` splitting as `[" 150", "9"]`. Under the pre-registered rule
    # this one row dissolves a tie the other sixteen support.
    breaker = "Then Mary went home and John slept"
    mixed = examples + [breaker]
    by_length = induction.induce(model, mixed)
    by_shape = induction.induce(model, mixed, filter_mode=induction.FILTER_SHAPE)

    check("the breaker is the same length as the rest",
          len(induction._tokenize(model, breaker)) == structure.length)
    check("length filter keeps it", by_length.n_examples_kept == len(mixed))
    check("length filter loses the tie",
          len(by_length.slots) == 3 and not any(s.is_tied for s in by_length.slots),
          f"slots {[(s.label, s.columns) for s in by_length.slots]}")
    check("shape filter drops exactly the breaker",
          by_shape.n_examples_kept == len(examples)
          and [d["text"] for d in by_shape.dropped] == [breaker])
    check("shape filter recovers the tie",
          by_shape.slots == structure.slots,
          f"slots {[(s.label, s.columns) for s in by_shape.slots]}")
    check("shape filter is recorded on the structure",
          by_shape.filter_mode == induction.FILTER_SHAPE
          and by_length.filter_mode == induction.FILTER_LENGTH)
    check("with no odd example the two filters agree",
          induction.induce(model, examples, filter_mode=induction.FILTER_SHAPE).slots
          == structure.slots)
    check("the shape filter still drops a wrong-length row",
          len(induction.induce(model, examples + [LONG_EXAMPLE],
                               filter_mode=induction.FILTER_SHAPE).dropped) == 1)

    # -- 3. the round-trip filter -----------------------------------------
    print("\n3. the round-trip filter accepts canonical rows and rejects a mangled one")
    check("every input example round-trips",
          all(induction.round_trips(model, induction._tokenize(model, e)) for e in examples))
    # " 15" followed by "09" is a real pair of tokens that the tokenizer would never
    # produce from " 1509" — it emits [" 150", "9"] instead. This is exactly the case
    # that breaks the fixtures, and it is checked here as a property of the filter.
    mangled = (
        model.tokenizer.bos_token_id,
        *model.tokenizer.encode(" 15", add_special_tokens=False),
        *model.tokenizer.encode("09", add_special_tokens=False),
    )
    check("a non-canonical token pair is rejected",
          not induction.round_trips(model, mangled),
          f"tokens {[model.to_string([t]) for t in mangled[1:]]}")

    # -- 4. generation -----------------------------------------------------
    print("\n4. generation is deterministic, distinct, and inside the observed vocabulary")
    gen_a = induction.generate(model, structure, n=8, seed=0)
    gen_b = induction.generate(model, structure, n=8, seed=0)
    check("same seed gives identical rows", gen_a.rows == gen_b.rows)
    check("rows are distinct", len(set(gen_a.rows)) == len(gen_a.rows))
    check("requested count produced", gen_a.count == 8, f"got {gen_a.count}")
    check("every tied column pair agrees in every generated row",
          all(row[2] == row[6] for row in gen_a.rows))
    check("every slot value came from the observed set",
          all(row[2] in tied.values and row[4] in free.values for row in gen_a.rows))
    check("frame columns are untouched",
          all(all(row[c] == structure.base_row[c] for c in structure.frame_columns)
              for row in gen_a.rows))

    # -- 5. proposals ------------------------------------------------------
    print("\n5. the proposal set follows from the structure and nothing else")
    proposals = induction.propose(structure)
    names = [p.name for p in proposals]
    check("one resample per slot, one desync per tied column",
          names == ["resample_t2", "resample_t4", "desync_t2", "desync_t6"], f"got {names}")
    check("resample of the tied slot writes both its columns",
          proposals[0].columns == (2, 6), f"got {proposals[0].columns}")
    check("desync writes one column only",
          proposals[2].columns == (2,) and proposals[3].columns == (6,))
    check("the free slot gets no desync",
          not any(p.kind == "desync" and p.slot_index == 1 for p in proposals))

    # -- 6. the dataset contract ------------------------------------------
    print("\n6. the auto-built dataset holds the invariants a cross-scheme sweep needs")
    built = [
        autotask.AutoDataset(
            model, structure=structure, generated=gen_a, proposals=proposals,
            n=gen_a.count, corruption=name, seed=0,
        )
        for name in names + [autotask.GENERIC_SCHEME]
    ]
    clean_digests = {digest(ds.clean_tokens) for ds in built}
    check("every scheme sees the identical clean sample", len(clean_digests) == 1,
          f"{len(clean_digests)} distinct clean batches")
    check("corrupted batches all differ from each other",
          len({digest(ds.corrupted_tokens) for ds in built}) == len(built))

    for ds, proposal in zip(built, proposals):
        changed = (ds.clean_tokens != ds.corrupted_tokens).any(dim=0).nonzero().flatten().tolist()
        check(f"{proposal.name} changes exactly columns {list(proposal.columns)}",
              changed == list(proposal.columns), f"changed {changed}")
        check(f"{proposal.name} changes every prompt",
              bool((ds.clean_tokens != ds.corrupted_tokens).any(dim=1).all()))

    ds = built[0]
    check("lengths are the induced length", bool((ds.lengths == structure.length).all()))
    check("positions cover the induced vocabulary plus END",
          set(ds.positions) >= set(structure.positions) | {"END"})

    # -- 7. the metric -----------------------------------------------------
    print("\n7. clean_argmax_logprob is the model's own clean maximum, exactly")
    with torch.no_grad():
        clean_logits = model(ds.clean_tokens)
        corrupted_logits = model(ds.corrupted_tokens)
    rows = torch.arange(len(ds), device=clean_logits.device)
    expected = clean_logits[rows, ds.positions["END"]].log_softmax(dim=-1).max(dim=-1).values.mean()
    measured = ds.logit_diff(clean_logits)
    check("on the clean run the metric equals the mean maximum log-probability",
          torch.allclose(measured, expected, atol=0, rtol=0),
          f"{measured.item():.6f} vs {expected.item():.6f}")
    check("the clean run agrees with itself exactly",
          ds.auto_rank_stats(clean_logits)["agrees_with_clean"] == 1.0)
    check("per-prompt returns one value per prompt",
          ds.logit_diff(clean_logits, per_prompt=True).shape == (len(ds),))
    per_prompt = ds.logit_diff(clean_logits, per_prompt=True)
    per_prompt_max = clean_logits[rows, ds.positions["END"]].log_softmax(dim=-1).max(dim=-1).values
    check("per-prompt, the clean metric is that prompt's maximum log-probability",
          torch.equal(per_prompt, per_prompt_max))

    # NOT a known answer, and deliberately not asserted. `clean_argmax_logprob` pins the
    # target to the clean run's argmax, which bounds the *clean* value at that prompt's
    # maximum but says nothing about the corrupted run: a counterfactual that leaves the
    # behaviour alone can make the model *more* confident in the same token, and the
    # clean-to-corrupted span then goes to zero or turns negative. Every normalized
    # number in the repository divides by that span. The spans are printed here for
    # every scheme so the property is on record; what it costs is a Phase 10 result and
    # is measured in the report, not decided by a check.
    print("\n   spans under clean_argmax_logprob (printed, not asserted):")
    for other, proposal_name in zip(built, names + [autotask.GENERIC_SCHEME]):
        with torch.no_grad():
            span = float(other.logit_diff(model(other.clean_tokens))) - float(
                other.logit_diff(model(other.corrupted_tokens))
            )
        flag = "" if span > 0 else "   <-- not normalizable"
        print(f"     {proposal_name:22s} span {span:+.4f}{flag}")

    # -- 8. the firewall ---------------------------------------------------
    print("\n8. neither new module can see an answer key")
    for module in (induction, autotask):
        imported = [
            name for name in dir(module)
            if "ground_truth" in name
        ] + [
            name for name in sys.modules
            if name.startswith("causal_interp.ground_truth")
            and name in getattr(module, "__dict__", {})
        ]
        check(f"{module.__name__} imports no ground_truth module", not imported, str(imported))
    source = (Path(__file__).resolve().parents[1] / "causal_interp")
    for filename in ("induction.py", "autotask.py"):
        text = (source / filename).read_text(encoding="utf-8")
        check(f"{filename} contains no ground_truth import",
              "import ground_truth" not in text and "from causal_interp.ground_truth" not in text)

    print()
    if failures:
        print(f"INDUCTION FAILED — {len(failures)} check(s):")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("INDUCTION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
