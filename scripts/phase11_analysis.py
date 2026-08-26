"""Phase 11, steps 3-5, stability statistics, then the answer key, in that order."""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CIRCUITS = ("docstring", "greater_than")
SEEDS = tuple(range(10))
METRIC = "logit_diff"
PHASE8_THRESHOLD = 0.02        # the criterion phase 8's flag used. p3's flagged set
N_PERMUTATIONS = 20_000
PERM_SEED = 20260823
BIG = 1e12                     # stand-in for an infinite SNR (sd exactly zero)

# the plan's bars, transcribed. none of them is computed here.
P1_MEDIAN_GAIN = 0.05
P1_ALPHA = 0.05 / 3
P2_RHO = 0.7
P2_ALPHA = 0.05
P3_ALPHA = 0.05
PRED4_BAR = 0.85
PRED5_BAR = 3.0


def spearman(xs: list[float], ys: list[float]) -> float:
    """rank correlation with midranks for ties, the note's implementation."""
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

    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return float("nan") if dx == 0 or dy == 0 else num / (dx * dy)


def auc(scores: dict[str, float], positives: set[str]) -> float:
    pos = [scores[h] for h in scores if h in positives]
    neg = [scores[h] for h in scores if h not in positives]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def quantile(values: list[float], q: float) -> float:
    vs = sorted(values)
    if not vs:
        return float("nan")
    pos = q * (len(vs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (pos - lo)


def wilcoxon_signed_rank_exact(diffs: list[float]) -> dict:
    """two-sided exact signed-rank test. n <= 9 here, so the null is enumerated."""
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return {"n_eff": 0, "w_plus": float("nan"), "p": 1.0}

    magnitudes = [abs(d) for d in nonzero]
    order = sorted(range(n), key=lambda i: magnitudes[i])
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and magnitudes[order[j + 1]] == magnitudes[order[i]]:
            j += 1
        mid = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rank[order[k]] = mid
        i = j + 1

    observed = sum(r for r, d in zip(rank, nonzero) if d > 0)
    centre = sum(rank) / 2
    deviation = abs(observed - centre)
    extreme = sum(
        1
        for signs in itertools.product((0, 1), repeat=n)
        if abs(sum(r for r, s in zip(rank, signs) if s) - centre) >= deviation - 1e-12
    )
    return {"n_eff": n, "w_plus": observed, "p": extreme / (2 ** n)}


# ============================================================ PART A


def load_resamples(circuit: str) -> dict:
    """every seed's grids for one circuit, plus the metadata they must all agree on."""
    payloads = {}
    for seed in SEEDS:
        path = RESULTS / f"phase11_{circuit}_seed{seed}.json"
        if not path.exists():
            raise SystemExit(f"missing {path.name}, run scripts/run_phase11_resample.py first")
        payloads[seed] = json.loads(path.read_text(encoding="utf-8"))

    metas = {seed: p["meta"] for seed, p in payloads.items()}
    for field in ("task", "prompts", "metric", "model", "n_layers", "n_heads", "positions",
                  "primary"):
        values = {json.dumps(m[field]) for m in metas.values()}
        if len(values) != 1:
            raise SystemExit(f"{circuit}: resamples disagree about {field}: {values}")
    return payloads


def thetas(circuit: str) -> dict[str, float]:
    payload = json.loads((RESULTS / f"phase9_{circuit}.json").read_text(encoding="utf-8"))
    return {scheme: block["threshold"] for scheme, block in payload["floors"].items()}


def collapse(grid: list, mode: str, fixed: dict[str, int] | None = None) -> dict[str, float]:
    """The pipeline's own rule: each head's value at the position of largest |effect|."""
    out: dict[str, float] = {}
    for layer, rows in enumerate(grid):
        for head, row in enumerate(rows):
            name = f"{layer}.{head}"
            if mode == "fixed":
                p = fixed[name]
            else:
                p = max(range(len(row)), key=lambda i: abs(row[i]))
            out[name] = float(row[p])
    return out


def effect_series(payloads: dict, mode: str = "max") -> tuple[dict, dict]:
    """E(h, s, r) for every scheme, as {scheme: {head: [value per seed]}}."""
    schemes = list(payloads[SEEDS[0]]["runs"])
    fixed_positions = {
        scheme: {
            f"{layer}.{head}": max(range(len(row)), key=lambda i: abs(row[i]))
            for layer, rows in enumerate(payloads[SEEDS[0]]["runs"][scheme]["grid"])
            for head, row in enumerate(rows)
        }
        for scheme in schemes
    }
    series: dict[str, dict[str, list[float]]] = {}
    spans: dict[str, list[float]] = {}
    for scheme in schemes:
        per_head: dict[str, list[float]] = {}
        for seed in SEEDS:
            run = payloads[seed]["runs"][scheme]
            values = collapse(run["grid"], mode, fixed_positions[scheme])
            for head, value in values.items():
                per_head.setdefault(head, []).append(value)
        series[scheme] = per_head
        spans[scheme] = [payloads[seed]["runs"][scheme]["span"] for seed in SEEDS]
    return series, spans


def head_statistics(series: dict, theta: dict[str, float]) -> dict:
    """S1-S4 and the two magnitude baselines, per (scheme, head). no answer key."""
    out: dict[str, dict[str, dict]] = {}
    zero_sd = 0
    for scheme, per_head in series.items():
        sds = {h: statistics.stdev(v) for h, v in per_head.items()}
        median_sd = statistics.median(sds.values())
        block: dict[str, dict] = {}
        for head, values in per_head.items():
            mean = statistics.fmean(values)
            sd = sds[head]
            if sd == 0:
                zero_sd += 1
            signs = [0.0 if v == 0 else math.copysign(1.0, v) for v in values]
            block[head] = {
                "mean": mean,
                "sd": sd,
                # S1: how often it clears its scheme's own frozen bar
                "s1_hit_fraction": sum(1 for v in values if abs(v) >= theta[scheme]) / len(values),
                # S2: effect against its own replication spread
                "s2_snr": BIG if sd == 0 else abs(mean) / sd,
                # S3: does it even point the same way every time
                "s3_sign_consistency": abs(statistics.fmean(signs)),
                # S4: this head's noise against its scheme's typical head
                "s4_rel_sd": float("nan") if median_sd == 0 else sd / median_sd,
                "b0_seed0": abs(values[0]),
                "bm_mean_abs": abs(mean),
            }
        out[scheme] = block
    return out, zero_sd


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def scheme_statistics(series: dict, spans: dict, theta: dict[str, float],
                      heads_stats: dict) -> dict:
    """T1-T5, per scheme. no answer key."""
    out: dict[str, dict] = {}
    for scheme, per_head in series.items():
        heads = sorted(per_head)
        columns = [[per_head[h][r] for h in heads] for r in range(len(SEEDS))]
        discovered = [
            {h for h in heads if abs(per_head[h][r]) >= theta[scheme]}
            for r in range(len(SEEDS))
        ]
        pairs = list(itertools.combinations(range(len(SEEDS)), 2))

        rank_rhos = [spearman(columns[i], columns[j]) for i, j in pairs]
        jaccards = [jaccard(discovered[i], discovered[j]) for i, j in pairs]
        jaccards = [v for v in jaccards if not math.isnan(v)]

        d0 = sorted(discovered[0])
        span_mean = statistics.fmean(spans[scheme])
        out[scheme] = {
            "n_discovered_seed0": len(d0),
            "discovered_seed0": d0,
            "t1_rank_reproducibility": statistics.fmean(rank_rhos),
            "t2_set_reproducibility": statistics.fmean(jaccards) if jaccards else float("nan"),
            "t3_median_hit_fraction":
                statistics.median(heads_stats[scheme][h]["s1_hit_fraction"] for h in d0)
                if d0 else float("nan"),
            "t4_median_snr":
                statistics.median(heads_stats[scheme][h]["s2_snr"] for h in d0)
                if d0 else float("nan"),
            "t5_span_cv":
                float("nan") if span_mean == 0
                else statistics.stdev(spans[scheme]) / abs(span_mean),
            "span_mean": span_mean,
            "span_sd": statistics.stdev(spans[scheme]),
        }
    return out


def crux(series: dict, heads_stats: dict) -> dict:
    """the plan's diagnostic: is SNR just magnitude, and is the noise homoscedastic?"""
    out: dict[str, dict] = {}
    for scheme, block in heads_stats.items():
        heads = sorted(block)
        snr = [block[h]["s2_snr"] for h in heads]
        bm = [block[h]["bm_mean_abs"] for h in heads]
        sds = [block[h]["sd"] for h in heads]
        p25, p75 = quantile(sds, 0.25), quantile(sds, 0.75)
        out[scheme] = {
            "spearman_snr_vs_bm": spearman(snr, bm),
            "sd_p25": p25,
            "sd_p50": statistics.median(sds),
            "sd_p75": p75,
            "sd_p75_over_p25": float("nan") if p25 == 0 else p75 / p25,
            "sd_min": min(sds),
            "sd_max": max(sds),
        }
    return out


def part_a() -> dict:
    """everything the plan registered for steps 1-3, with no answer key in reach."""
    blind: dict = {"circuits": {}, "meta": {
        "seeds": list(SEEDS), "metric": METRIC,
        "note": "computed and written before any published head list was read",
    }}
    for circuit in CIRCUITS:
        payloads = load_resamples(circuit)
        theta = thetas(circuit)
        series, spans = effect_series(payloads, mode="max")
        fixed_series, _ = effect_series(payloads, mode="fixed")

        heads_stats, zero_sd = head_statistics(series, theta)
        fixed_stats, _ = head_statistics(fixed_series, theta)
        blind["circuits"][circuit] = {
            "meta": payloads[SEEDS[0]]["meta"],
            "theta": theta,
            "runtimes": {str(s): payloads[s]["meta"]["runtime_seconds"] for s in SEEDS},
            "series": series,
            "heads": heads_stats,
            "heads_position_fixed": fixed_stats,
            "schemes": scheme_statistics(series, spans, theta, heads_stats),
            "crux": crux(series, heads_stats),
            "spans": spans,
            "zero_sd_heads": zero_sd,
        }
    return blind


# ============================================================ PART B


def published_heads(circuit: str) -> set[str]:
    """The circuit's published head list, recovered from phase 9's committed scores."""
    payload = json.loads((RESULTS / f"phase9_{circuit}.json").read_text(encoding="utf-8"))
    sets = {
        frozenset(v["matches"]) | frozenset(v["misses"])
        for v in payload["scored_before"]["per_scheme"].values()
    }
    assert len(sets) == 1, f"{circuit}: published head list disagrees across schemes"
    heads = set(next(iter(sets)))
    assert len(heads) == payload["meta"]["published_head_count"]
    return heads


def label_a() -> dict[tuple[str, str], float]:
    payload = json.loads((RESULTS / "scheme_level_analysis.json").read_text(encoding="utf-8"))
    return {(r["circuit"], r["scheme"]): r["aim_auc"] for r in payload["rows"]}


def p1_head_level(blind: dict) -> dict:
    """does any stability statistic beat magnitude at ranking published heads?"""
    candidates = ("s1_hit_fraction", "s2_snr", "s3_sign_consistency")
    baselines = ("bm_mean_abs", "b0_seed0")
    rows = []
    for circuit in CIRCUITS:
        pub = published_heads(circuit)
        for scheme, block in blind["circuits"][circuit]["heads"].items():
            row = {"circuit": circuit, "scheme": scheme,
                   "n_heads": len(block), "n_published": len(pub)}
            for name in candidates + baselines:
                row[f"auc_{name}"] = auc({h: v[name] for h, v in block.items()}, pub)
            rows.append(row)

    tests = {}
    for name in candidates:
        diffs = [r[f"auc_{name}"] - r["auc_bm_mean_abs"] for r in rows]
        test = wilcoxon_signed_rank_exact(diffs)
        median_gain = statistics.median(diffs)
        tests[name] = {
            "median_gain_vs_bm": median_gain,
            "gains": diffs,
            "wilcoxon": test,
            "clears_median_bar": median_gain >= P1_MEDIAN_GAIN,
            "clears_alpha_bar": test["p"] < P1_ALPHA,
            "declared_improvement": median_gain >= P1_MEDIAN_GAIN and test["p"] < P1_ALPHA,
        }

    bm_vs_b0 = [r["auc_bm_mean_abs"] - r["auc_b0_seed0"] for r in rows]
    return {
        "rows": rows,
        "tests": tests,
        "bm_vs_b0": {"gains": bm_vs_b0, "median_gain": statistics.median(bm_vs_b0),
                     "wilcoxon": wilcoxon_signed_rank_exact(bm_vs_b0)},
        "any_declared": any(t["declared_improvement"] for t in tests.values()),
    }


def p2_scheme_level(blind: dict) -> dict:
    """does scheme-level stability predict scheme aim? n = 9, registered as underpowered."""
    signals = ("t1_rank_reproducibility", "t2_set_reproducibility",
               "t3_median_hit_fraction", "t4_median_snr", "t5_span_cv")
    labels = label_a()
    rows = []
    for circuit in CIRCUITS:
        for scheme, block in blind["circuits"][circuit]["schemes"].items():
            row = {"circuit": circuit, "scheme": scheme,
                   "label_a": labels[(circuit, scheme)]}
            row.update({s: block[s] for s in signals})
            rows.append(row)

    ys = [r["label_a"] for r in rows]
    complete = {s: [i for i, r in enumerate(rows) if not math.isnan(r[s])] for s in signals}
    observed = {
        s: spearman([rows[i][s] for i in complete[s]], [ys[i] for i in complete[s]])
        for s in signals
    }
    within = {}
    for s in signals:
        per_circuit = {}
        for circuit in CIRCUITS:
            idx = [i for i in complete[s] if rows[i]["circuit"] == circuit]
            per_circuit[circuit] = spearman([rows[i][s] for i in idx], [ys[i] for i in idx])
        signs = [v for v in per_circuit.values() if not math.isnan(v) and v != 0]
        within[s] = {"per_circuit": per_circuit,
                     "consistent": bool(signs) and len({v > 0 for v in signs}) == 1}

    rng = random.Random(PERM_SEED)
    null = []
    for _ in range(N_PERMUTATIONS):
        shuffled = ys[:]
        rng.shuffle(shuffled)
        null.append(max(
            abs(spearman([rows[i][s] for i in complete[s]], [shuffled[i] for i in complete[s]]))
            for s in signals
            if len(complete[s]) >= 3
        ))
    null.sort()

    results = {}
    for s in signals:
        rho = observed[s]
        p = float("nan") if math.isnan(rho) else \
            sum(1 for v in null if v >= abs(rho) - 1e-12) / len(null)
        results[s] = {
            "rho": rho, "n": len(complete[s]), "family_wise_p": p,
            "within_circuit": within[s]["per_circuit"],
            "sign_consistent": within[s]["consistent"],
            "declared_separating": (not math.isnan(rho)) and p < P2_ALPHA
                                   and abs(rho) >= P2_RHO and within[s]["consistent"],
        }
    best = max(signals, key=lambda s: 0.0 if math.isnan(observed[s]) else abs(observed[s]))
    return {
        "rows": rows,
        "results": results,
        "best_signal": best,
        "best_abs_rho": abs(observed[best]),
        "null_median": statistics.median(null),
        "null_p95": quantile(null, 0.95),
        "any_declared": any(r["declared_separating"] for r in results.values()),
    }


def p3_flagged(blind: dict) -> dict:
    """Is the disagreement that flagged a head reproducible? docstring only."""
    circuit = "docstring"
    phase9 = json.loads((RESULTS / f"phase9_{circuit}.json").read_text(encoding="utf-8"))
    before = phase9["before"]
    primary = before["primary"]
    flagged = before["blind_spots"][primary]
    pub = published_heads(circuit)

    heads_stats = blind["circuits"][circuit]["heads"]
    seed0 = {s: {h: v["b0_seed0"] for h, v in block.items()}
             for s, block in heads_stats.items()}

    scores, chosen = {}, {}
    for head in flagged:
        best = max((s for s in seed0 if s != primary), key=lambda s: seed0[s][head])
        chosen[head] = best
        scores[head] = heads_stats[best][head]["s2_snr"]

    positives = {h for h in flagged if h in pub}
    observed = auc(scores, positives)

    exact = [
        auc(scores, set(combo))
        for combo in itertools.combinations(flagged, len(positives))
    ]
    p = sum(1 for v in exact if v >= observed - 1e-12) / len(exact)
    return {
        "circuit": circuit,
        "n_flagged": len(flagged),
        "flagged": flagged,
        "positives": sorted(positives),
        "best_other_scheme": chosen,
        "s2_by_head": scores,
        "auc": observed,
        "n_labellings": len(exact),
        "p": p,
        "significant": p < P3_ALPHA,
    }


def reproduction_check() -> dict:
    """prediction 1: does seed 0 reproduce the committed phase 8 effects?"""
    out = {}
    for circuit in CIRCUITS:
        seed0 = json.loads(
            (RESULTS / f"phase11_{circuit}_seed0.json").read_text(encoding="utf-8"))
        phase8 = json.loads(
            (RESULTS / f"phase8_{circuit}.json").read_text(encoding="utf-8"))
        worst = 0.0
        for scheme, run in seed0["runs"].items():
            mine = collapse(run["grid"], "max")
            theirs = phase8["discovery"]["runs"][scheme]["effects"][METRIC]
            worst = max(worst, max(abs(mine[h] - theirs[h]) for h in theirs))
        out[circuit] = worst
    return {"max_abs_delta": out, "holds": all(v < 1e-4 for v in out.values())}


def score_predictions(blind, p1, p2, p3, repro) -> list[dict]:
    """The nine predictions from the plan, scored in public."""
    def rows_of(key):
        return [(c, s, blind["circuits"][c]["crux"][s][key])
                for c in CIRCUITS for s in blind["circuits"][c]["crux"]]

    snr_bm = [v for _, _, v in rows_of("spearman_snr_vs_bm")]
    ratio = rows_of("sd_p75_over_p25")
    per_circuit_ratio = {
        c: [v for cc, _, v in ratio if cc == c] for c in CIRCUITS
    }
    t1 = {c: {s: b["t1_rank_reproducibility"]
              for s, b in blind["circuits"][c]["schemes"].items()} for c in CIRCUITS}
    doc_lowest = min(t1["docstring"], key=t1["docstring"].get)
    gt_highest = max(t1["greater_than"], key=t1["greater_than"].get)

    return [
        {"n": 1, "text": "seed 0 reproduces the committed Phase 8 effects to <1e-4",
         "outcome": f"max |delta| = {max(repro['max_abs_delta'].values()):.2e}",
         "held": repro["holds"]},
        {"n": 2, "text": "no stability statistic clears the P1 bar against Bm",
         "outcome": "none cleared" if not p1["any_declared"] else
                    "at least one cleared", "held": not p1["any_declared"]},
        {"n": 3, "text": "Bm beats B0 by a median AUC gain of at least +0.01",
         "outcome": f"median gain {p1['bm_vs_b0']['median_gain']:+.4f}",
         "held": p1["bm_vs_b0"]["median_gain"] >= 0.01},
        {"n": 4, "text": f"median Spearman(S2, Bm) across rows >= {PRED4_BAR}",
         "outcome": f"median {statistics.median(snr_bm):.3f}",
         "held": statistics.median(snr_bm) >= PRED4_BAR},
        {"n": 5, "text": f"per-head sd is near-homoscedastic: p75/p25 < {PRED5_BAR}",
         "outcome": "; ".join(
             f"{c}: {sum(1 for v in vs if v < PRED5_BAR)}/{len(vs)} rows "
             f"(median {statistics.median(vs):.2f})" for c, vs in per_circuit_ratio.items()),
         "held": all(statistics.median(vs) < PRED5_BAR for vs in per_circuit_ratio.values())},
        {"n": 6, "text": "P2 inconclusive, best |rho| below its own permutation null median",
         "outcome": f"best {p2['best_signal']} |rho| = {p2['best_abs_rho']:.3f}, "
                    f"null median {p2['null_median']:.3f}",
         "held": (not p2["any_declared"]) and p2["best_abs_rho"] < p2["null_median"]},
        {"n": 7, "text": "P3 not significant (p >= 0.05)",
         "outcome": f"AUC {p3['auc']:.3f}, p = {p3['p']:.3f}",
         "held": not p3["significant"]},
        {"n": 8, "text": "docstring random_vocab_any has the lowest T1 of its five schemes",
         "outcome": f"lowest T1 is {doc_lowest} "
                    f"({t1['docstring'][doc_lowest]:+.3f}); random_vocab_any "
                    f"{t1['docstring']['random_vocab_any']:+.3f}",
         "held": doc_lowest == "random_vocab_any"},
        {"n": 9, "text": "greater-than yy01 has the highest T1 of its four schemes",
         "outcome": f"highest T1 is {gt_highest} "
                    f"({t1['greater_than'][gt_highest]:+.3f}); yy01 "
                    f"{t1['greater_than']['yy01']:+.3f}",
         "held": gt_highest == "yy01"},
    ]


# ============================================================ output


def write_head_csv(blind: dict) -> None:
    path = RESULTS / "phase11_head_stability.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["circuit", "scheme", "head", "published", "theta", "mean", "sd",
                    "s1_hit_fraction", "s2_snr", "s3_sign_consistency", "s4_rel_sd",
                    "b0_seed0", "bm_mean_abs"])
        for circuit in CIRCUITS:
            pub = published_heads(circuit)
            theta = blind["circuits"][circuit]["theta"]
            for scheme, block in blind["circuits"][circuit]["heads"].items():
                for head in sorted(block, key=lambda h: tuple(map(int, h.split(".")))):
                    v = block[head]
                    w.writerow([circuit, scheme, head, head in pub, theta[scheme],
                                f"{v['mean']:.6f}", f"{v['sd']:.6f}",
                                f"{v['s1_hit_fraction']:.2f}",
                                f"{min(v['s2_snr'], BIG):.4f}",
                                f"{v['s3_sign_consistency']:.2f}",
                                f"{v['s4_rel_sd']:.4f}",
                                f"{v['b0_seed0']:.6f}", f"{v['bm_mean_abs']:.6f}"])
    print(f"wrote {path.name}")


def main() -> int:
    print("PART A, stability statistics, no answer key")
    blind = part_a()
    blind_path = RESULTS / "phase11_stability.json"
    slim = {
        "meta": blind["meta"],
        "circuits": {
            c: {k: v for k, v in block.items() if k != "series"}
            for c, block in blind["circuits"].items()
        },
    }
    blind_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"wrote {blind_path.name}, Part A is on disk before Part B is entered\n")

    for circuit in CIRCUITS:
        block = blind["circuits"][circuit]
        print(f"  {circuit}: {len(block['heads'])} schemes, "
              f"{len(next(iter(block['heads'].values())))} heads, "
              f"{block['zero_sd_heads']} heads with sd exactly 0")

    print("\nPART B, the published head lists are opened only here")
    repro = reproduction_check()
    p1 = p1_head_level(blind)
    p2 = p2_scheme_level(blind)
    p3 = p3_flagged(blind)
    predictions = score_predictions(blind, p1, p2, p3, repro)

    payload = {
        "meta": {"seeds": list(SEEDS), "metric": METRIC,
                 "n_permutations": N_PERMUTATIONS, "perm_seed": PERM_SEED,
                 "bars": {"p1_median_gain": P1_MEDIAN_GAIN, "p1_alpha": P1_ALPHA,
                          "p2_rho": P2_RHO, "p2_alpha": P2_ALPHA, "p3_alpha": P3_ALPHA}},
        "reproduction": repro,
        "p1_head_level": p1,
        "p2_scheme_level": p2,
        "p3_flagged": p3,
        "predictions": predictions,
    }
    out = RESULTS / "phase11_tests.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_head_csv(blind)
    print(f"wrote {out.name}")

    print(f"\n  P1 declared improvement: {p1['any_declared']}")
    for name, t in p1["tests"].items():
        print(f"    {name:22} median gain vs Bm {t['median_gain_vs_bm']:+.4f}   "
              f"wilcoxon p = {t['wilcoxon']['p']:.4f}")
    print(f"  P2 declared separating : {p2['any_declared']}   "
          f"best {p2['best_signal']} |rho| {p2['best_abs_rho']:.3f} "
          f"vs null median {p2['null_median']:.3f}")
    print(f"  P3 significant         : {p3['significant']}   "
          f"AUC {p3['auc']:.3f}, p = {p3['p']:.3f}")
    print("\n  predictions:")
    for p in predictions:
        print(f"    {p['n']}. {'HELD ' if p['held'] else 'WRONG'}  {p['text']}")
        print(f"       -> {p['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
