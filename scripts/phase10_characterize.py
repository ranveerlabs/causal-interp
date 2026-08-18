"""Phase 10 step 1: run the pre-registered induction and measure what it produces.

    python scripts/phase10_characterize.py        # ~2 min, one GPU pass per scheme

Phase 9 established the ordering this script follows: measure first, in its own commit,
before anything is designed on top of the measurement. What runs here is section 3 of
`results/PHASE10_PLAN.md` exactly as committed — no repair, no tuning, no filtering of
the human's fixture — applied to the two fixtures the plan fixed, plus the two
cross-task induction checks the plan lists as run E.

It writes `results/phase10_characterization.json` and
`results/PHASE10_CHARACTERIZATION.md`. It opens no answer key and scores nothing: the
question here is only what the induction says about the prompts it was handed, and
whether the task it builds is one the pipeline could run at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_interp import autotask, induction
from causal_interp.model import load

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIXTURES = ROOT / "fixtures"

N_PROMPTS = 128
SEED = 0

FIXTURE_FILES = {
    "frame_same": FIXTURES / "greater_than_frame_same.txt",
    "frame_own": FIXTURES / "greater_than_frame_own.txt",
}


def read_fixture(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shape_signature(row: tuple[int, ...]) -> tuple[int, ...]:
    """Each column replaced by the first column holding the same token.

    A row's *shape*: which of its positions repeat each other, ignoring what the tokens
    actually are. Two examples of the same template have the same shape; an example the
    tokenizer split differently does not. Measured here as a diagnostic only — no rule
    in `PHASE10_PLAN.md` consults it, and this script designs nothing.
    """
    first: dict[int, int] = {}
    return tuple(first.setdefault(token, column) for column, token in enumerate(row))


def diagnose_shapes(model, examples: list[str]) -> dict:
    """How many of the human's lines agree on a column-repetition pattern?

    The induction ties two columns only when they agree in *every* kept example, so a
    single line the tokenizer split unusually is enough to dissolve a tie that the other
    thirty-one support. This measures how close to unanimous the fixtures actually are.
    """
    rows = [induction._tokenize(model, text) for text in examples]
    groups: dict[tuple, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((len(row), shape_signature(row)), []).append(index)
    ranked = sorted(groups.values(), key=len, reverse=True)
    minority = [i for group in ranked[1:] for i in group]
    return {
        "n_distinct_shapes": len(ranked),
        "largest_shape_group": len(ranked[0]),
        "minority_examples": [{"index": i, "text": examples[i]} for i in minority],
    }


def describe_fixture(model, name: str, examples: list[str]) -> dict:
    """Induce, generate, propose and measure — everything short of a head sweep."""
    print(f"\n{'=' * 72}\n{name}  ({len(examples)} lines)\n{'=' * 72}")
    decode = lambda t: model.to_string([t])  # noqa: E731

    structure = induction.induce(model, examples)
    print(f"  modal token length   {structure.length}")
    print(f"  examples kept        {structure.n_examples_kept} / {structure.n_examples_given}")
    for dropped in structure.dropped:
        print(f"    dropped (len {dropped['length']}): {dropped['text']!r}")
    print(f"  frame columns        {list(structure.frame_columns)}")
    print(f"  positions            {list(structure.positions)}")
    for slot in structure.slots:
        shown = ", ".join(repr(decode(v)) for v in slot.values[:6])
        print(
            f"    slot {slot.label:5s} columns={list(slot.columns)} "
            f"tied={slot.is_tied} values={len(slot.values):3d}  [{shown} ...]"
        )

    shapes = diagnose_shapes(model, examples)
    print(
        f"  column-shape groups  {shapes['n_distinct_shapes']} "
        f"(largest holds {shapes['largest_shape_group']} of {len(examples)})"
    )
    for odd in shapes["minority_examples"]:
        print(f"    minority shape: {odd['text']!r}")

    generated = induction.generate(model, structure, n=N_PROMPTS, seed=SEED)
    print(
        f"  generated            {generated.count}/{generated.requested} distinct "
        f"in {generated.attempts} attempts "
        f"(round-trip rejected {generated.rejected_round_trip}, "
        f"duplicates {generated.rejected_duplicate})"
    )

    proposals = induction.propose(structure)
    print(f"  proposed schemes     {[p.name for p in proposals]}")

    # Per-scheme measurement. The divergence is the plan's selection statistic; the span
    # is what every normalized number downstream divides by, and it is measured here
    # because the known-answer suite showed it is not guaranteed positive.
    per_scheme: dict[str, dict] = {}
    names = [p.name for p in proposals] + [autotask.GENERIC_SCHEME]
    for scheme in names:
        ds = autotask.AutoDataset(
            model, structure=structure, generated=generated, proposals=proposals,
            n=generated.count, corruption=scheme, seed=SEED,
        )
        with torch.no_grad():
            clean_logits = model(ds.clean_tokens)
            corrupted_logits = model(ds.corrupted_tokens)
        clean_value = float(ds.logit_diff(clean_logits))
        corrupted_value = float(ds.logit_diff(corrupted_logits))
        per_scheme[scheme] = {
            "kl_divergence": autotask._kl_at_end(ds, clean_logits, corrupted_logits),
            "clean": clean_value,
            "corrupted": corrupted_value,
            "span": clean_value - corrupted_value,
            "corrupted_round_trip_rejected": ds.corrupted_round_trip,
            "agrees_with_clean_on_corrupted": ds.auto_rank_stats(corrupted_logits)[
                "agrees_with_clean"
            ],
        }

    print(f"  {'scheme':22s} {'KL':>8s} {'span':>9s} {'agree':>7s} {'corrupt !round-trip':>20s}")
    for scheme, block in per_scheme.items():
        rt = block["corrupted_round_trip_rejected"]
        rt_text = "n/a" if rt != rt else f"{rt:.0%}"  # NaN check without importing math
        print(
            f"  {scheme:22s} {block['kl_divergence']:8.3f} {block['span']:+9.4f} "
            f"{block['agrees_with_clean_on_corrupted']:7.0%} {rt_text:>20s}"
        )

    induced = [p.name for p in proposals]
    primary = max(induced, key=lambda s: per_scheme[s]["kl_divergence"]) if induced else None
    print(f"  selected primary     {primary}")

    return {
        "name": name,
        "structure": structure.as_dict(decode),
        "shape_diagnostic": shapes,
        "generation": generated.as_dict(),
        "proposals": [
            {"name": p.name, "kind": p.kind, "columns": list(p.columns), "label": p.label}
            for p in proposals
        ],
        "schemes": per_scheme,
        "primary": primary,
        "sample_prompts": [
            model.to_string(torch.tensor(list(row[1:]))) for row in generated.rows[:5]
        ],
    }


def cross_task(model, label: str, prompts: list[str], hand_built: tuple[str, ...]) -> dict:
    """Run E: induction on prompts drawn from a hand-built generator.

    Weaker than the fixture cases by construction, and labelled so everywhere it
    appears: these prompts come out of a task module that already solved the problem
    the induction is being asked to solve. The only question is whether the induced
    slot structure matches what that module hand-codes.
    """
    print(f"\n{'=' * 72}\ncross-task induction: {label}   (weaker — prompts from a hand-built generator)\n{'=' * 72}")
    try:
        structure = induction.induce(model, prompts)
    except ValueError as exc:
        print(f"  induction FAILED: {exc}")
        return {"label": label, "failed": str(exc), "hand_built_positions": list(hand_built)}

    decode = lambda t: model.to_string([t])  # noqa: E731
    print(f"  modal token length   {structure.length}")
    print(f"  examples kept        {structure.n_examples_kept} / {structure.n_examples_given}")
    print(f"  induced positions    {list(structure.positions)}")
    print(f"  hand-built positions {list(hand_built)}")
    for slot in structure.slots:
        shown = ", ".join(repr(decode(v)) for v in slot.values[:5])
        print(
            f"    slot {slot.label:6s} columns={list(slot.columns)} "
            f"tied={slot.is_tied} values={len(slot.values):3d}  [{shown} ...]"
        )
    proposals = induction.propose(structure)
    print(f"  proposed schemes     {len(proposals)}: {[p.name for p in proposals][:8]}")
    return {
        "label": label,
        "structure": structure.as_dict(decode),
        "n_proposals": len(proposals),
        "proposals": [p.name for p in proposals],
        "hand_built_positions": list(hand_built),
    }


def main() -> int:
    out: dict = {"n_prompts": N_PROMPTS, "seed": SEED, "fixtures": {}, "cross_task": {}}

    print("loading gpt2-small ...")
    model = load("gpt2-small")

    for name, path in FIXTURE_FILES.items():
        out["fixtures"][name] = describe_fixture(model, name, read_fixture(path))

    # Run E, case 1: IOI, whose eight templates have different token lengths.
    from causal_interp.ioi import POSITIONS as IOI_POSITIONS
    from causal_interp.ioi import IOIDataset

    ioi = IOIDataset(model, n=32, seed=SEED)
    out["cross_task"]["ioi_all_templates"] = cross_task(
        model, "IOI (all 8 templates)", [p.clean for p in ioi.prompts], IOI_POSITIONS
    )

    # And the same task restricted to one template, which is the condition the induction
    # actually requires. Reported side by side so the failure above is attributable.
    from causal_interp.ioi import TEMPLATES

    ioi_one = IOIDataset(model, n=32, seed=SEED, templates=(TEMPLATES[0],))
    out["cross_task"]["ioi_one_template"] = cross_task(
        model, "IOI (one template)", [p.clean for p in ioi_one.prompts], IOI_POSITIONS
    )

    # Run E, case 2: docstring, on its own model.
    print("\nloading attn-only-4l ...")
    small = load("attn-only-4l")
    from causal_interp.docstring import POSITIONS as DOC_POSITIONS
    from causal_interp.docstring import DocstringDataset

    doc = DocstringDataset(small, n=32, seed=SEED)
    out["cross_task"]["docstring"] = cross_task(
        small, "docstring", [p.clean for p in doc.prompts], DOC_POSITIONS
    )

    path = RESULTS / "phase10_characterization.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
