"""Phase 11 post-hoc diagnostics, run after the registered tests, and marked as such."""

from __future__ import annotations

import importlib.util
import itertools
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

_spec = importlib.util.spec_from_file_location("phase11_analysis", ROOT / "scripts" / "phase11_analysis.py")
_a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a)


def p3_comparators(blind: dict, tests: dict) -> dict:
    """every ranking of the same 17 flagged heads, so P3's pass can be read in context."""
    p3 = tests["p3_flagged"]
    flagged, positives, star = p3["flagged"], set(p3["positives"]), p3["best_other_scheme"]
    heads = blind["circuits"]["docstring"]["heads"]

    variants = {
        "s2_snr_REGISTERED": {h: heads[star[h]][h]["s2_snr"] for h in flagged},
        "b0_magnitude_seed0": {h: heads[star[h]][h]["b0_seed0"] for h in flagged},
        "bm_magnitude_mean": {h: heads[star[h]][h]["bm_mean_abs"] for h in flagged},
        "s1_hit_fraction": {h: heads[star[h]][h]["s1_hit_fraction"] for h in flagged},
        "scheme_identity_is_random_def":
            {h: 1.0 if star[h] == "random_def" else 0.0 for h in flagged},
    }
    out = {}
    for name, scores in variants.items():
        observed = _a.auc(scores, positives)
        exact = [_a.auc(scores, set(c))
                 for c in itertools.combinations(flagged, len(positives))]
        out[name] = {
            "auc": observed,
            "p": sum(1 for v in exact if v >= observed - 1e-12) / len(exact),
            "scores": scores,
        }
    return out


def noise_scaling(blind: dict) -> dict:
    """does replication sd grow with the size of the effect it is measuring?"""
    out = {}
    for circuit in _a.CIRCUITS:
        published = _a.published_heads(circuit)
        for scheme, block in blind["circuits"][circuit]["heads"].items():
            heads = sorted(block)
            sds = [block[h]["sd"] for h in heads]
            magnitudes = [block[h]["bm_mean_abs"] for h in heads]
            pub_sd = statistics.median(block[h]["sd"] for h in heads if h in published)
            rest_sd = statistics.median(block[h]["sd"] for h in heads if h not in published)
            out[f"{circuit}/{scheme}"] = {
                "spearman_sd_vs_magnitude": _a.spearman(sds, magnitudes),
                "median_sd_published": pub_sd,
                "median_sd_rest": rest_sd,
                "ratio": float("nan") if rest_sd == 0 else pub_sd / rest_sd,
            }
    return out


def main() -> int:
    blind = json.loads((RESULTS / "phase11_stability.json").read_text(encoding="utf-8"))
    tests = json.loads((RESULTS / "phase11_tests.json").read_text(encoding="utf-8"))

    comparators = p3_comparators(blind, tests)
    scaling = noise_scaling(blind)

    print("POST-HOC, and unable to change any Phase 11 verdict.\n")
    print("P3's 17 flagged docstring heads, ranked every way:")
    print(f"  {'statistic':34} {'AUC':>7} {'exact p':>9}")
    for name, block in comparators.items():
        print(f"  {name:34} {block['auc']:7.3f} {block['p']:9.4f}")

    print("\nDoes replication noise scale with effect size?")
    print(f"  {'circuit/scheme':34} {'rho(sd,|mean|)':>14} {'sd pub / sd rest':>17}")
    for name, block in scaling.items():
        print(f"  {name:34} {block['spearman_sd_vs_magnitude']:14.3f} "
              f"{block['ratio']:17.2f}")

    payload = {
        "disclaimer": "post-hoc; not registered in PHASE11_PLAN.md; cannot change a verdict",
        "p3_comparators": comparators,
        "noise_scaling": scaling,
    }
    out = RESULTS / "phase11_posthoc.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
