"""Known-answer tests for Phase 8's scheme registry and agreement analysis.

    python scripts/check_schemes.py        # expect: SCHEMES OK

Patching is easy to get subtly wrong in ways that still produce reasonable numbers,
which is why `check_patching.py` exists. The same is true of a disagreement report: a
comparison that silently dropped a scheme, or one that flagged everything, would still
print a plausible table. These are cases where the right answer follows from how the
experiment is built rather than from the model.

Six checks:

1. `TaskSpec` refuses a single-scheme task. This is the whole structural claim of the
   phase — that multi-scheme discovery is not an option a caller can leave unset — so
   it is tested rather than asserted in prose.
2. `agreement.py` and `pipeline.py` import no `ground_truth` module. The disagreement
   report has to be available on a circuit with no published answer.
3. The agreement analysis returns the verdicts that follow by construction from a
   synthetic effect table nobody measured.
4. The flag does not fire when every scheme agrees.
5. Phase 8 did not disturb Phase 6: greater-than's clean and corrupted token tensors
   under `yy01` and both generic schemes hash identically to the module as it stood
   before this phase, read straight out of git.
6. `xx_mismatch` is what it claims to be: exactly one token differs from the clean
   prompt, it is the start year's century, `YY` is untouched, and the metric's answer
   definition is unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_interp import agreement, greater_than
from causal_interp.model import load
from causal_interp.schemes import Scheme, TaskSpec

ROOT = Path(__file__).resolve().parents[1]
PRE_PHASE8 = "a015ecb"  # the last commit before Phase 8 touched anything

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def _hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. a task cannot register a single counterfactual
# ---------------------------------------------------------------------------


def check_registry() -> None:
    print("\nregistry enforcement")
    one = {"only": Scheme("only", "published", "the one scheme", False, primary=True)}
    try:
        TaskSpec(
            name="single", dataset=object, positions=("END",), schemes=one,
            discovery_schemes=("only",), metric_label="", model_alias="",
        )
        check("single-scheme TaskSpec is rejected", False, "it was accepted")
    except ValueError as exc:
        check("single-scheme TaskSpec is rejected", "at least two" in str(exc))

    two = {
        **one,
        "other": Scheme("other", "generic", "a second scheme", True),
    }
    spec = TaskSpec(
        name="pair", dataset=object, positions=("END",), schemes=two,
        discovery_schemes=("only", "other"), metric_label="", model_alias="",
    )
    check("two-scheme TaskSpec is accepted", spec.primary_scheme == "only")

    try:
        TaskSpec(
            name="two primaries", dataset=object, positions=("END",),
            schemes={
                "a": Scheme("a", "published", "", False, primary=True),
                "b": Scheme("b", "published", "", False, primary=True),
            },
            discovery_schemes=("a", "b"), metric_label="", model_alias="",
        )
        check("two primaries are rejected", False, "they were accepted")
    except ValueError:
        check("two primaries are rejected", True)

    for module in ("ioi", "greater_than", "docstring"):
        task = __import__(f"causal_interp.{module}", fromlist=["TASK"]).TASK
        check(
            f"{module} registers >= 2 discovery schemes",
            len(task.discovery_schemes) >= 2,
            f"{len(task.discovery_schemes)}: {', '.join(task.discovery_schemes)}",
        )


# ---------------------------------------------------------------------------
# 2. the analysis cannot see an answer key
# ---------------------------------------------------------------------------


def check_blindness() -> None:
    print("\nblindness")
    for name in ("agreement.py", "pipeline.py", "schemes.py"):
        source = (ROOT / "causal_interp" / name).read_text(encoding="utf-8")
        offenders = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith(("import ", "from ")) and "ground_truth" in line
        ]
        check(f"{name} imports no ground_truth module", not offenders, "; ".join(offenders))


# ---------------------------------------------------------------------------
# 3 and 4. the agreement analysis on a table nobody measured
# ---------------------------------------------------------------------------


def check_agreement() -> None:
    print("\nagreement analysis (synthetic)")
    # Head (0,0) clears everywhere; (1,1) clears only under `alt`; (2,2) only under
    # `primary`; (3,3) clears nowhere. Threshold 0.02, Phase 1's.
    effects = {
        "primary": {(0, 0): 0.9, (1, 1): 0.001, (2, 2): 0.5, (3, 3): 0.0},
        "alt": {(0, 0): 0.4, (1, 1): 0.30, (2, 2): 0.001, (3, 3): 0.0},
        "third": {(0, 0): 0.2, (1, 1): 0.05, (2, 2): 0.001, (3, 3): 0.0},
    }
    spans = {"primary": 4.0, "alt": 2.0, "third": 0.2}
    report = agreement.compare_schemes(
        effects, threshold=0.02, primary="primary", channel="synthetic", spans=spans
    )
    status = {v.head: v.status for v in report.verdicts}

    check("head found under every scheme is robust", status.get((0, 0)) == agreement.ROBUST)
    check(
        "head found under one scheme only is scheme-dependent",
        status.get((1, 1)) == agreement.SCHEME_DEPENDENT,
    )
    check("head found under no scheme is absent from the union", (3, 3) not in status)
    check("the primary's blind spot is exactly the head it missed", report.primary_blind_spot == [(1, 1)])
    check("the flag fires", report.flag and "1.1" in report.flag_text)
    check("union is the three heads some scheme found", report.union == [(0, 0), (1, 1), (2, 2)])
    check("intersection is the head every scheme found", report.intersection == [(0, 0)])
    check(
        "only_in reports the primary's exclusive find",
        report.only_in["primary"] == [(2, 2)],
    )
    check(
        "power is relative to the primary's span",
        abs(report.power["alt"].power - 0.5) < 1e-9 and report.power["third"].low_power,
        f"alt {report.power['alt'].power:.2f}, third {report.power['third'].power:.2f} (low-power)",
    )
    check(
        "a low-power scheme is annotated, not dropped",
        "third" in report.per_scheme and (1, 1) in report.per_scheme["third"],
    )

    agreed = agreement.compare_schemes(
        {"primary": {(0, 0): 0.9}, "alt": {(0, 0): 0.4}},
        threshold=0.02, primary="primary", channel="synthetic",
    )
    check("the flag stays silent when the schemes agree", not agreed.flag)

    try:
        agreement.compare_schemes(
            {"primary": {(0, 0): 0.9}}, threshold=0.02, primary="primary", channel="synthetic"
        )
        check("single-scheme comparison is rejected", False, "it was accepted")
    except ValueError as exc:
        check("single-scheme comparison is rejected", "at least two" in str(exc))

    specs = {
        "primary": {(3, 0): ("v@C_def", 0.8), (0, 5): ("v@B_def", 0.3)},
        "alt": {(3, 0): ("q@END", 0.7), (0, 5): ("v@B_def", 0.2)},
    }
    spec_report = agreement.compare_spec_rankings(specs, primary="primary")
    check(
        "a head whose winning spec changes is spec-scheme-dependent",
        spec_report["scheme_dependent"] == ["3.0"],
        f"{spec_report['n_scheme_dependent']} of {spec_report['n_heads']} heads",
    )


# ---------------------------------------------------------------------------
# 5. Phase 6 is undisturbed
# ---------------------------------------------------------------------------


def _load_pre_phase8_module(model):
    source = subprocess.run(
        ["git", "show", f"{PRE_PHASE8}:causal_interp/greater_than.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    path = Path(tempfile.gettempdir()) / "greater_than_pre_phase8.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("greater_than_pre_phase8", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules, so the module has to be
    # registered before its body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_backward_compatibility(model) -> None:
    print(f"\nbackward compatibility against {PRE_PHASE8} (pre-Phase 8)")
    old = _load_pre_phase8_module(model)
    for scheme in ("yy01", "random_vocab_yy", "random_vocab_any"):
        before = old.GreaterThanDataset(model, n=32, corruption=scheme, seed=0)
        after = greater_than.GreaterThanDataset(model, n=32, corruption=scheme, seed=0)
        same = (
            _hash(before.clean_tokens) == _hash(after.clean_tokens)
            and _hash(before.corrupted_tokens) == _hash(after.corrupted_tokens)
            and torch.equal(before.yy_values, after.yy_values)
        )
        check(f"{scheme}: tokens hash identically", same, _hash(after.corrupted_tokens))


# ---------------------------------------------------------------------------
# 6. the authored scheme is what it says it is
# ---------------------------------------------------------------------------


def check_xx_mismatch(model) -> None:
    print("\nxx_mismatch (authored alternate for greater-than)")
    ds = greater_than.GreaterThanDataset(model, n=32, corruption="xx_mismatch", seed=0)
    clean, corrupted = ds.clean_tokens, ds.corrupted_tokens

    check("clean and corrupted have the same shape", clean.shape == corrupted.shape,
          str(tuple(clean.shape)))

    differing = (clean != corrupted)
    per_row = differing.sum(dim=1)
    check("exactly one token differs per prompt", bool((per_row == 1).all()),
          f"min {int(per_row.min())}, max {int(per_row.max())}")

    rows = torch.arange(len(ds), device=clean.device)
    where = differing.float().argmax(dim=1)
    check("the differing token is XX1, the start year's century",
          bool((where == ds.positions["XX1"]).all()))
    check("YY is bit-identical between the two runs",
          bool((clean[rows, ds.positions["YY"]] == corrupted[rows, ds.positions["YY"]]).all()))
    check("the final century token is untouched",
          bool((clean[rows, ds.positions["END"]] == corrupted[rows, ds.positions["END"]]).all()))
    check("the substituted century always differs from the clean one",
          all(p.corrupt_century is not None and p.corrupt_century != p.century for p in ds.prompts))

    yy01 = greater_than.GreaterThanDataset(model, n=32, corruption="yy01", seed=0)
    check("the answer definition (YY) is the same as under the published scheme",
          torch.equal(ds.yy_values, yy01.yy_values))
    check("the clean prompts are the same as under the published scheme",
          _hash(ds.clean_tokens) == _hash(yy01.clean_tokens))
    print(f"    example clean      {ds.prompts[0].clean!r}")
    print(f"    example xx_mismatch {ds.prompts[0].corrupted!r}")


def main() -> int:
    check_registry()
    check_blindness()
    check_agreement()

    model = load("gpt2-small")
    check_backward_compatibility(model)
    check_xx_mismatch(model)

    print()
    if failures:
        print(f"SCHEMES FAILED — {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("SCHEMES OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
