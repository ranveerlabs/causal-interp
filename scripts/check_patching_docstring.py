"""Known-answer tests for the patching machinery on a *different model*.

Run before trusting a Phase 7 result:
    python scripts/check_patching_docstring.py

`check_patching.py` verifies the same primitives against GPT-2 small and the IOI
task. It is left untouched. This is its Phase 7 counterpart, and it exists because
a new model is exactly where the machinery could be silently wrong: an off-by-one
in a position index, a hook that never fires, or a head-shaped activation with a
different layout all produce numbers that look entirely reasonable.

Most checks mirror the GPT-2 small ones. Two are specific to this model:

  * with **no MLP blocks**, patching every attention head at every position must
    reproduce the clean run at the final token *exactly* — the corruption leaves
    the END token itself unchanged, so its residual stream is its own embedding
    plus the head outputs, and every one of those has been replaced. Tolerance
    1e-5, against 0.05 for the same check on GPT-2 small.
  * `hook_mlp_out` is registered on this model but never fires. Patching it is a
    **silent no-op**, not an error. That is asserted here rather than left to be
    discovered, because a sweep over it returns a clean grid of zeros that looks
    like a measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_interp import ground_truth_docstring as gt
from causal_interp.docstring import POSITIONS, DocstringDataset
from causal_interp.interventions import (
    ALL_POSITIONS,
    LOGITS,
    Patch,
    Receiver,
    baseline_for,
    clean_cache_for,
    cache_for,
    patch_effect,
    path_patch,
)
from causal_interp.model import load

MODEL = "attn-only-4l"

# The published counterfactual replaces exactly these many tokens per prompt.
TOKENS_CHANGED = {"random_random": 5, "random_def": 2, "random_answer": 1}

# Under `random_answer` only C_def differs, so clean and corrupted activations are
# identical everywhere before it and patching there must be an *exact* zero.
PRE_CDEF_POSITIONS = ("A_def", "B_def", "comma_B")

CACHE_KINDS = ("z", "q", "k", "v", "resid_pre", "attn_out")


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
    model = load(MODEL)

    print(f"\nmodel: {model.cfg.model_name}  "
          f"{model.cfg.n_layers} layers x {model.cfg.n_heads} heads, "
          f"d_model {model.cfg.d_model}, attn_only={model.cfg.attn_only}")
    print(f"ground truth: {len(gt.ALL_HEADS)} heads in {len(gt.CIRCUIT)} classes")

    print("\narchitecture")
    c.assert_true("model has no MLP blocks", bool(model.cfg.attn_only))
    c.assert_true(
        "positional embeddings are standard (not shortformer)",
        model.cfg.positional_embedding_type == "standard",
        model.cfg.positional_embedding_type,
    )

    print("\ndataset invariants")
    for corruption, expected in TOKENS_CHANGED.items():
        ds = DocstringDataset(model, n=32, corruption=corruption, seed=0)
        differing = (ds.clean_tokens != ds.corrupted_tokens).sum(1)
        c.assert_true(
            f"{corruption}: exactly {expected} token(s) differ per prompt",
            bool((differing == expected).all()),
            f"observed {differing.min().item()}-{differing.max().item()}",
        )

    ds = DocstringDataset(model, n=32, corruption="random_random", seed=0)
    c.assert_true(
        "every prompt is the same length (absolute-position search is well posed)",
        int(ds.lengths.min()) == int(ds.lengths.max()),
        f"{int(ds.lengths.min())}-{int(ds.lengths.max())} tokens",
    )
    comma_id = ds._token_id(model, ",")
    ok_args = all(
        int(ds.clean_tokens[i, ds.positions["C_def"][i]])
        == ds._token_id(model, " " + ds.prompts[i].answer)
        for i in range(len(ds))
    )
    ok_doc = all(
        ds.clean_tokens[i, ds.positions["A_doc"][i]] == ds.clean_tokens[i, ds.positions["A_def"][i]]
        and ds.clean_tokens[i, ds.positions["B_doc"][i]] == ds.clean_tokens[i, ds.positions["B_def"][i]]
        for i in range(len(ds))
    )
    ok_comma = all(
        int(ds.clean_tokens[i, ds.positions["comma_B"][i]]) == comma_id for i in range(len(ds))
    )
    c.assert_true("C_def index holds the answer token", ok_args)
    c.assert_true("A_doc / B_doc repeat the tokens at A_def / B_def", ok_doc)
    c.assert_true("comma_B index holds a comma", ok_comma)
    c.assert_true(
        "positions are strictly ordered A_def < B_def < comma_B < C_def < A_doc < B_doc < END",
        all(
            bool((ds.positions[a] < ds.positions[b]).all())
            for a, b in zip(POSITIONS, POSITIONS[1:])
        ),
    )

    baseline, _, _ = baseline_for(model, ds)
    cache, _ = clean_cache_for(model, ds, kinds=CACHE_KINDS)
    print(
        f"\nbaseline: clean {baseline.clean_logit_diff:+.4f}, "
        f"corrupted {baseline.corrupted_logit_diff:+.4f}, span {baseline.span:.4f}"
    )
    c.assert_true("clean logit diff is positive (model does the task)", baseline.clean_logit_diff > 0)
    c.assert_true(
        "corrupted logit diff is negative (corruption reverses the answer)",
        baseline.corrupted_logit_diff < 0,
    )

    print("\npatching identities")
    c.check("empty patch set == corrupted run", patch_effect(model, ds, cache, [], baseline), 0.0, 1e-9)
    for layer in range(model.cfg.n_layers):
        effect = patch_effect(model, ds, cache, [Patch(layer, "resid_pre", ALL_POSITIONS)], baseline)
        c.check(f"resid_pre[{layer}] @ ALL == clean run", effect, 1.0, 5e-3)

    print("\nattention-only identity: every head at every position must be *exactly* clean")
    print("  (no MLPs, and the END token is unchanged, so nothing else can differ)")
    all_heads = [
        Patch(layer, "z", ALL_POSITIONS, head)
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
    ]
    c.check(
        f"all {model.cfg.n_layers * model.cfg.n_heads} heads @ ALL == clean run",
        patch_effect(model, ds, cache, all_heads, baseline), 1.0, 1e-5,
    )

    print("\nmlp_out is a silent no-op on this model, not an error")
    for layer in range(model.cfg.n_layers):
        effect = patch_effect(model, ds, cache, [Patch(layer, "mlp_out", ALL_POSITIONS)], baseline)
        c.check(f"mlp_out[{layer}] @ ALL changes nothing", effect, 0.0, 0.0)

    print("\nrandom_answer: positions before C_def are structurally unpatchable (exact zeros)")
    single = DocstringDataset(model, n=32, corruption="random_answer", seed=0)
    single_baseline, _, _ = baseline_for(model, single)
    single_cache, _ = clean_cache_for(model, single, kinds=CACHE_KINDS)
    for position in PRE_CDEF_POSITIONS:
        effect = patch_effect(
            model, single, single_cache, [Patch(0, "resid_pre", position)], single_baseline
        )
        c.check(f"resid_pre[0] @ {position}", effect, 0.0, 1e-9)
    contrast = patch_effect(
        model, single, single_cache, [Patch(0, "resid_pre", "C_def")], single_baseline
    )
    c.assert_true(
        "resid_pre[0] @ C_def is nonzero (the one token that does differ)",
        abs(contrast) > 0.5, f"{contrast:+.4f}",
    )

    # -- path patching ------------------------------------------------------

    corrupted_cache, _ = cache_for(model, ds.corrupted_tokens, CACHE_KINDS)
    last = model.cfg.n_layers - 1

    print("\npath patching: a sender cannot reach a receiver at or below its own layer")
    for sender_layer, receiver_layer in ((2, 2), (3, 1)):
        effect = path_patch(
            model, ds, cache, corrupted_cache, baseline,
            sender=Patch(sender_layer, "z", "END", 0),
            receivers=[Receiver(layer=receiver_layer, head=0, position="END", input="q")],
        )
        c.check(f"path {sender_layer}.0 -> {receiver_layer}.0.q", effect, 0.0, 1e-9)

    print(f"\npath patching: a layer-{last} sender has no downstream head to relay through,")
    print("so its direct effect must equal its plain activation-patching effect exactly")
    for head in range(model.cfg.n_heads):
        direct = path_patch(
            model, ds, cache, corrupted_cache, baseline,
            sender=Patch(last, "z", "END", head), receivers=LOGITS,
        )
        total = patch_effect(model, ds, cache, [Patch(last, "z", "END", head)], baseline)
        c.check(f"{last}.{head} direct == total", direct, total, 1e-5)

    print("\n" + ("PATCHING OK" if not c.failures else f"{c.failures} CHECK(S) FAILED"))
    return 0 if not c.failures else 1


if __name__ == "__main__":
    sys.exit(main())
