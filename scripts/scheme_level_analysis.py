"""Scheme-level re-analysis of Phase 9's discriminator signals.

    python scripts/scheme_level_analysis.py

Phase 9 measured ten candidate signals over 33 flagged **heads** and found none of them
was a discriminator. SYNTHESIS.md section 5 noticed that the unit was wrong: the thing a
reader has to judge is a counterfactual scheme, not a head, and Phase 9's own diagnosis
("docstring's noise came from one pathological scheme") names a scheme as the culprit.

This script runs the same ten signals, plus the scheme-level fields Phase 9 stored and
never tested, over the 13 (circuit, scheme) rows of Phase 9's floor table.

Everything is read back from committed payloads. No model is run, nothing is recomputed
from activations, and no `ground_truth` module is imported -- the published head lists are
recovered from Phase 9's own `scored_before` block, where matches u misses is the full
list and is identical across every scheme of a circuit.

The labelling rule, the signal list and the bar for declaring a result were all fixed in
results/SCHEME_LEVEL_NOTE.md and committed before this file existed.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CIRCUITS = ("docstring", "greater_than", "ioi")
METRICS = ("logit_diff", "kl", "tv")
METRIC = "logit_diff"
THRESHOLD = 0.02          # Phase 8's shared cutoff, kept only where a signal names it
N_PERMUTATIONS = 20_000
PERM_SEED = 20260820
AUC_CUT = 0.80            # fixed in the note before any AUC was computed


# --------------------------------------------------------------------------- loading


def _payloads(circuit: str) -> tuple[dict, dict]:
    """(calibration payload, discovery payload) for one circuit.

    Phase 8 registered IOI's schemes and deliberately did not run it, so IOI's discovery
    block lives inside its Phase 9 payload instead. Both have the same shape.
    """
    phase9 = json.loads((RESULTS / f"phase9_{circuit}.json").read_text(encoding="utf-8"))
    if "discovery" in phase9:
        return phase9, phase9
    phase8 = json.loads((RESULTS / f"phase8_{circuit}.json").read_text(encoding="utf-8"))
    return phase9, phase8


def _effects(discovery: dict, scheme: str, metric: str = METRIC) -> dict[str, float]:
    return discovery["discovery"]["runs"][scheme]["effects"][metric]


def _published(phase9: dict) -> list[str]:
    """The circuit's published head list, recovered from the committed scores.

    Every scheme's `matches` u `misses` is the same set; the assertion is the check.
    """
    sets = {
        frozenset(v["matches"]) | frozenset(v["misses"])
        for v in phase9["scored_before"]["per_scheme"].values()
    }
    assert len(sets) == 1, "published head list disagrees across schemes"
    published = sorted(next(iter(sets)))
    assert len(published) == phase9["meta"]["published_head_count"]
    return published


# ----------------------------------------------------------------------------- stats


def auc(scores: dict[str, float], positives: set[str]) -> float:
    """P(a random published head outranks a random unpublished one), ties at a half.

    Threshold-free by construction, which is why the note picked it: it cannot be
    contaminated by theta or by the shared 0.02, both of which are signals under test.
    """
    pos = [abs(scores[h]) for h in scores if h in positives]
    neg = [abs(scores[h]) for h in scores if h not in positives]
    if not pos or not neg:
        return float("nan")
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in pos
        for n in neg
    )
    return wins / (len(pos) * len(neg))


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, with midranks for ties.

    Phase 9's own spearman() assumed no ties. Several scheme-level signals here are
    counts and do tie, so this one handles them properly and is used throughout.
    """
    def ranks(vs: list[float]) -> list[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            mid = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = mid
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def percentiles(values: list[float]) -> dict[str, float]:
    """Same order-statistic convention Phase 9's scheme_scale() used, so the numbers
    printed here line up with the ones already in PHASE9_CHARACTERIZATION.md."""
    v = sorted(values)
    n = len(v)
    return {
        "median": v[n // 2],
        "p90": v[int(0.90 * n)],
        "max": v[-1],
        "mean": statistics.fmean(v),
    }


# ---------------------------------------------------------------------------- rows


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for circuit in CIRCUITS:
        phase9, discovery = _payloads(circuit)
        published = set(_published(phase9))
        schemes = [s["scheme"] for s in phase9["schemes"]]
        primary = phase9["before"]["primary"]
        meta = {s["scheme"]: s for s in phase9["schemes"]}

        eff = {s: _effects(discovery, s) for s in schemes}
        scale = {s: percentiles([abs(v) for v in eff[s].values()]) for s in schemes}
        primary_max = scale[primary]["max"]
        primary_span = phase9["floors"][primary]["span"]

        for s in schemes:
            floor = phase9["floors"][s]
            sc = scale[s]
            discovered = {h for h, v in eff[s].items() if abs(v) >= THRESHOLD}
            n_heads = len(eff[s])

            # Signal 10's analogue: of the heads this scheme discovers under logit_diff,
            # what share also clear 0.02 under kl and tv?
            all_three = sum(
                1
                for h in discovered
                if all(abs(_effects(discovery, s, m).get(h, 0.0)) >= THRESHOLD for m in METRICS)
            )

            scored = phase9["scored_before"]["per_scheme"][s]
            prec, rec = scored["precision"], scored["recall"]

            rows.append({
                "circuit": circuit,
                "scheme": s,
                "primary": s == primary,
                "provenance": meta[s]["provenance"],
                "preserves_answer": bool(meta[s].get("preserves_answer")),

                # ---- labels
                "aim_auc": auc(eff[s], published),
                "aim_f1": (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
                "precision": prec,
                "recall": rec,

                # ---- scheme-level analogues of Phase 9's ten head-level signals
                "s01_median_over_threshold": sc["median"] / THRESHOLD,
                "s02_max_over_median": sc["max"] / sc["median"] if sc["median"] else float("inf"),
                "s03_max_over_primary_max": sc["max"] / primary_max if primary_max else float("inf"),
                "s04_p90_over_median": sc["p90"] / sc["median"] if sc["median"] else float("inf"),
                "s05_max_over_p90": sc["max"] / sc["p90"] if sc["p90"] else float("inf"),
                "s06_discovered_fraction": len(discovered) / n_heads,
                "s07_n_discovered": float(len(discovered)),
                "s08_median_over_max": sc["median"] / sc["max"] if sc["max"] else float("inf"),
                "s09_max_effect": sc["max"],
                "s10_all_three_metrics": all_three / len(discovered) if discovered else 0.0,

                # ---- scheme-level fields Phase 9 stored and never tested
                "power": floor["span"] / primary_span,
                "theta": floor["threshold"],
                "null_median": floor["null_median"],
                "null_max": floor["null_max"],
                "theta_over_null_median": floor["threshold"] / floor["null_median"],
                "null_max_over_median": floor["null_max"] / floor["null_median"],
                "theta_over_own_median": floor["threshold"] / sc["median"] if sc["median"] else float("inf"),
                "span": floor["span"],
                "spearman_with_primary": (
                    1.0 if s == primary else spearman(
                        [abs(eff[primary][h]) for h in sorted(eff[s])],
                        [abs(eff[s][h]) for h in sorted(eff[s])],
                    )
                ),
                "jaccard_with_primary": (
                    1.0 if s == primary else (
                        len(discovered & {h for h, v in eff[primary].items() if abs(v) >= THRESHOLD})
                        / len(discovered | {h for h, v in eff[primary].items() if abs(v) >= THRESHOLD})
                        if (discovered | {h for h, v in eff[primary].items() if abs(v) >= THRESHOLD})
                        else 0.0
                    )
                ),
            })
    return rows


SIGNALS = [
    "s01_median_over_threshold", "s02_max_over_median", "s03_max_over_primary_max",
    "s04_p90_over_median", "s05_max_over_p90", "s06_discovered_fraction",
    "s07_n_discovered", "s08_median_over_max", "s09_max_effect",
    "s10_all_three_metrics",
    "power", "theta", "null_median", "null_max", "theta_over_null_median",
    "null_max_over_median", "theta_over_own_median", "span",
    "spearman_with_primary", "jaccard_with_primary",
]


# ------------------------------------------------------------------------- testing


def analyse(rows: list[dict], label: str) -> dict:
    ys = [r[label] for r in rows]

    observed = {}
    for sig in SIGNALS:
        xs = [r[sig] for r in rows]
        rho = spearman(xs, ys)
        # Sign of the same correlation computed inside each circuit on its own. The note
        # requires these to agree, so that a cross-circuit correlation driven purely by
        # circuit size cannot be reported as a signal.
        per_circuit = {}
        for c in CIRCUITS:
            sub = [r for r in rows if r["circuit"] == c]
            per_circuit[c] = spearman([r[sig] for r in sub], [r[label] for r in sub])
        signs = {(1 if v > 0 else -1 if v < 0 else 0) for v in per_circuit.values()}
        observed[sig] = {
            "rho": rho,
            "abs_rho": abs(rho),
            "per_circuit": per_circuit,
            "sign_consistent": len(signs - {0}) <= 1,
        }

    # Max-statistic permutation null: shuffle the labels, recompute every signal, keep the
    # largest |rho| of the family. This is the correction the note fixed in advance -- with
    # 20 signals at n=13 the per-signal p-value is meaningless.
    rng = random.Random(PERM_SEED)
    columns = {sig: [r[sig] for r in rows] for sig in SIGNALS}
    null_max = []
    shuffled = list(ys)
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(shuffled)
        null_max.append(max(abs(spearman(columns[sig], shuffled)) for sig in SIGNALS))

    best = max(SIGNALS, key=lambda s: observed[s]["abs_rho"])
    best_abs = observed[best]["abs_rho"]
    fwer_p = (1 + sum(1 for v in null_max if v >= best_abs)) / (N_PERMUTATIONS + 1)

    for sig in SIGNALS:
        a = observed[sig]["abs_rho"]
        observed[sig]["fwer_p"] = (1 + sum(1 for v in null_max if v >= a)) / (N_PERMUTATIONS + 1)
        observed[sig]["separating"] = bool(
            observed[sig]["fwer_p"] < 0.05
            and a >= 0.7
            and observed[sig]["sign_consistent"]
        )

    return {
        "label": label,
        "n": len(rows),
        "signals": observed,
        "best": best,
        "best_abs_rho": best_abs,
        "best_fwer_p": fwer_p,
        "null_max_p95": sorted(null_max)[int(0.95 * len(null_max))],
        "null_max_median": statistics.median(null_max),
        "any_separating": any(observed[s]["separating"] for s in SIGNALS),
        "n_permutations": N_PERMUTATIONS,
        "permutation_seed": PERM_SEED,
    }


def main() -> None:
    rows = build_rows()
    assert len(rows) == 13, f"expected 13 (circuit, scheme) rows, got {len(rows)}"

    out = {
        "meta": {
            "note": "scheme-level re-analysis of Phase 9's discriminator signals",
            "n_rows": len(rows),
            "circuits": list(CIRCUITS),
            "sources": "results/phase8_*.json, results/phase9_*.json (committed)",
            "model_runs": 0,
            "auc_cut": AUC_CUT,
        },
        "rows": rows,
        "aim_auc": analyse(rows, "aim_auc"),
        "aim_f1": analyse(rows, "aim_f1"),
    }
    (RESULTS / "scheme_level_analysis.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8"
    )

    # ---- console summary
    print(f"{'circuit':13s} {'scheme':18s} {'prov':9s} {'AUC':>6s} {'F1':>6s} "
          f"{'power':>6s} {'theta':>8s} {'th/nullmed':>11s}")
    for r in rows:
        print(f"{r['circuit']:13s} {r['scheme']:18s} {r['provenance']:9s} "
              f"{r['aim_auc']:6.3f} {r['aim_f1']:6.3f} {r['power']:6.2f} "
              f"{r['theta']:8.4f} {r['theta_over_null_median']:11.0f}")

    for label in ("aim_auc", "aim_f1"):
        a = out[label]
        print(f"\n=== {label}: n={a['n']}, {len(SIGNALS)} signals, "
              f"{a['n_permutations']} permutations ===")
        ranked = sorted(SIGNALS, key=lambda s: -a["signals"][s]["abs_rho"])
        print(f"{'signal':28s} {'rho':>7s} {'|rho|':>7s} {'fwer_p':>8s} "
              f"{'signs ok':>9s} {'sep':>5s}")
        for sig in ranked:
            v = a["signals"][sig]
            print(f"{sig:28s} {v['rho']:7.3f} {v['abs_rho']:7.3f} {v['fwer_p']:8.4f} "
                  f"{str(v['sign_consistent']):>9s} {str(v['separating']):>5s}")
        print(f"null max-|rho|: median {a['null_max_median']:.3f}, "
              f"95th pct {a['null_max_p95']:.3f}")
        print(f"best: {a['best']} |rho|={a['best_abs_rho']:.3f} "
              f"family-wise p={a['best_fwer_p']:.4f}")
        print(f"ANY SIGNAL SEPARATING: {a['any_separating']}")

    above = [r for r in rows if r["aim_auc"] >= AUC_CUT]
    print(f"\nbinary split at AUC >= {AUC_CUT}: {len(above)} of {len(rows)} well-aimed")
    for r in rows:
        side = "well-aimed" if r["aim_auc"] >= AUC_CUT else "badly-aimed"
        print(f"  {r['circuit']:13s} {r['scheme']:18s} AUC={r['aim_auc']:.3f}  {side}")


if __name__ == "__main__":
    main()
