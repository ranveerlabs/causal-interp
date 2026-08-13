"""Known-answer tests for the patching machinery.

Run before trusting a result:
    python scripts/check_patching.py

Activation patching is easy to get subtly wrong — an off-by-one in a position
index or a hook writing to the wrong head produces numbers that look entirely
reasonable and mean nothing. The checks here are cases where the correct answer
is known in advance from the construction of the experiment, not from the model:

  * patching nothing must reproduce the corrupted run exactly (0.0)
  * patching the whole residual stream at every position must reproduce the
    clean run exactly (1.0), at every layer
  * under `s2_swap` exactly one token differs, so clean and corrupted activations
    are identical everywhere before S2 and patching there must be an *exact* zero
  * the recorded semantic positions must land on the tokens they name
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_interp import ground_truth
from causal_interp.interventions import (
    ALL_POSITIONS,
    Patch,
    baseline_for,
    clean_cache_for,
    patch_effect,
)
from causal_interp.ioi import IOIDataset
from causal_interp.model import load

# Positions that necessarily precede S2 in every template, and so cannot differ
# between the clean and corrupted runs under the single-token s2_swap corruption.
PRE_S2_POSITIONS = ("IO", "IO+1", "S1", "S1+1")


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, label: str, got: float, want: float, tol: float) -> None:
        ok = abs(got - want) <= tol
        self.failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got:+.8f}, want {want:+.8f} (tol {tol:g})")

    def assert_true(self, label: str, ok: bool, detail: str = "") -> None:
        self.failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f': {detail}' if detail else ''}")


def main() -> int:
    c = Checker()
    model = load()

    print(f"\nground truth: {len(ground_truth.ALL_HEADS)} heads in {len(ground_truth.IOI_CIRCUIT)} classes")

    print("\ndataset invariants")
    for corruption, expected_diff in (("s2_swap", 1), ("abc", 3)):
        ds = IOIDataset(model, n=32, corruption=corruption, seed=0)
        differing = (ds.clean_tokens != ds.corrupted_tokens).sum(1)
        c.assert_true(
            f"{corruption}: exactly {expected_diff} token(s) differ per prompt",
            bool((differing == expected_diff).all()),
            f"observed {differing.min().item()}-{differing.max().item()}",
        )

        # Every semantic position must land on the token it claims to name.
        ok_io = all(
            ds.clean_tokens[i, ds.positions["IO"][i]] == ds.io_token_ids[i] for i in range(len(ds))
        )
        ok_s = all(
            ds.clean_tokens[i, ds.positions["S1"][i]] == ds.s_token_ids[i]
            and ds.clean_tokens[i, ds.positions["S2"][i]] == ds.s_token_ids[i]
            for i in range(len(ds))
        )
        ok_order = bool((ds.positions["S1"] < ds.positions["S2"]).all())
        c.assert_true(f"{corruption}: IO index holds the IO token", ok_io)
        c.assert_true(f"{corruption}: S1 and S2 indices hold the S token", ok_s)
        c.assert_true(f"{corruption}: S1 precedes S2", ok_order)

    ds = IOIDataset(model, n=32, corruption="s2_swap", seed=0)
    baseline, _, _ = baseline_for(model, ds)
    cache, _ = clean_cache_for(model, ds)
    print(
        f"\nbaseline: clean {baseline.clean_logit_diff:+.4f}, "
        f"corrupted {baseline.corrupted_logit_diff:+.4f}, span {baseline.span:.4f}"
    )
    c.assert_true("clean logit diff is positive (model does the task)", baseline.clean_logit_diff > 0)
    c.assert_true("corrupted logit diff is negative (corruption flips the answer)", baseline.corrupted_logit_diff < 0)

    print("\npatching identities")
    c.check("empty patch set == corrupted run", patch_effect(model, ds, cache, [], baseline), 0.0, 1e-9)

    for layer in range(model.cfg.n_layers):
        effect = patch_effect(model, ds, cache, [Patch(layer, "resid_pre", ALL_POSITIONS)], baseline)
        c.check(f"resid_pre[{layer}] @ ALL == clean run", effect, 1.0, 5e-3)

    print("\ns2_swap: positions before S2 are structurally unpatchable (exact zeros)")
    for position in PRE_S2_POSITIONS:
        effect = patch_effect(model, ds, cache, [Patch(0, "resid_pre", position)], baseline)
        c.check(f"resid_pre[0] @ {position}", effect, 0.0, 1e-9)
    contrast = patch_effect(model, ds, cache, [Patch(0, "resid_pre", "S2")], baseline)
    c.assert_true("resid_pre[0] @ S2 is nonzero (the one token that does differ)", abs(contrast) > 0.5,
                  f"{contrast:+.4f}")

    print("\nhead-level coverage")
    all_heads = [
        Patch(layer, "z", ALL_POSITIONS, head)
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
    ]
    c.check("all 144 heads @ ALL restores clean behaviour", patch_effect(model, ds, cache, all_heads, baseline), 1.0, 0.05)

    print("\n" + ("PATCHING OK" if not c.failures else f"{c.failures} CHECK(S) FAILED"))
    return 0 if not c.failures else 1


if __name__ == "__main__":
    sys.exit(main())
