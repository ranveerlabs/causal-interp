"""phase 12 analysis, scores PHASE12_PLAN.md's P1-P6 and the nine predictions."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_interp.ground_truth import ALL_HEADS, HEAD_TO_CLASS

RESULTS = Path(__file__).resolve().parents[1] / "results"
PAYLOAD = RESULTS / "phase12_scrub.json"
TESTS = RESULTS / "phase12_tests.json"
REPORT = RESULTS / "PHASE12_REPORT.md"

P1_LOGIT_BAR = 0.70
P1_KL_BAR = 0.50
P3_PERCENTILE_BAR = 0.95
PRED8_BAR = 0.9

PUBLISHED = {f"{l}.{h}" for l, h in ALL_HEADS}


def spearman(xs: list[float], ys: list[float]) -> float:
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


def auc_permutation_p(scores: dict[str, float], n_pos: int, observed: float, seed: int = 0) -> float:
    import random

    rng = random.Random(seed)
    keys = list(scores)
    hits = 0
    trials = 20_000
    for _ in range(trials):
        shuffled = set(rng.sample(keys, n_pos))
        if auc(scores, shuffled) >= observed:
            hits += 1
    return (hits + 1) / (trials + 1)


def percentile_of(value: float, null: list[float]) -> float:
    return sum(1 for v in null if v < value) / len(null)


def one_sided_p(value: float, null: list[float]) -> float:
    return (1 + sum(1 for v in null if v >= value)) / (1 + len(null))


def ld(row: dict) -> float:
    return row["logit_diff"]["mean"]


def fmt(x: float, n: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{n}f}"


def main() -> None:
    d = json.loads(PAYLOAD.read_text())
    cand = d["candidates"]
    null = {int(m): rows for m, rows in d["null"].items()}
    out: dict = {"bars": {"p1_logit": P1_LOGIT_BAR, "p1_kl": P1_KL_BAR}}

    curve = {}
    for m, rows in sorted(null.items()):
        vals = [ld(r) for r in rows]
        curve[m] = {
            "n": len(vals),
            "mean": statistics.fmean(vals),
            "sd": statistics.stdev(vals),
            "median": statistics.median(vals),
            "p95": sorted(vals)[int(0.95 * (len(vals) - 1))],
            "max": max(vals),
        }
    out["null_curve"] = curve

    def residual(value: float, size: int) -> float:
        c = curve.get(size)
        return float("nan") if c is None or c["sd"] == 0 else (value - c["mean"]) / c["sd"]

    pub = cand["published_26"]
    out["P1"] = {
        "logit_diff": pub["logit_diff"],
        "kl": pub["kl"],
        "top1_is_io": pub["top1_is_io"],
        "raw_logit_diff": pub["logit_diff_raw"]["mean"],
        "clean_logit_diff": d["anchors"]["clean"]["logit_diff"],
        "passes_logit": ld(pub) >= P1_LOGIT_BAR,
        "passes_kl": pub["kl"]["mean"] >= P1_KL_BAR,
    }
    out["P1"]["passes"] = out["P1"]["passes_logit"] and out["P1"]["passes_kl"]

    rows = []
    for name, row in cand.items():
        rows.append(
            {
                "name": name,
                "size": row["size"],
                "overlap": row["published_overlap"],
                "logit_diff": ld(row),
                "sd": row["logit_diff"]["sd"],
                "kl": row["kl"]["mean"],
                "top1": row["top1_is_io"]["mean"],
                "residual": residual(ld(row), row["size"]),
            }
        )
    out["P2"] = {"candidates": rows}

    drop6 = [ld(r) for r in d["families"]["drop"]["6"]]
    matches = cand["phase1_matches"]
    out["P2"]["drop6"] = {
        "mean": statistics.fmean(drop6),
        "p10": sorted(drop6)[int(0.10 * (len(drop6) - 1))],
        "p90": sorted(drop6)[int(0.90 * (len(drop6) - 1))],
        "phase1_matches": ld(matches),
        "percentile": percentile_of(ld(matches), drop6),
        "inside_10_90": None,
    }
    p2 = out["P2"]["drop6"]
    p2["inside_10_90"] = p2["p10"] <= p2["phase1_matches"] <= p2["p90"]

    out["P2"]["families"] = {
        kind: {
            k: {
                "size": rows_[0]["size"],
                "mean": statistics.fmean([ld(r) for r in rows_]),
                "sd": statistics.stdev([ld(r) for r in rows_]),
                "residual": residual(statistics.fmean([ld(r) for r in rows_]), rows_[0]["size"]),
            }
            for k, rows_ in fam.items()
        }
        for kind, fam in d["families"].items()
    }

    r26 = [ld(r) for r in null[26]]
    lm = [ld(r) for r in d["layer_matched"]]
    out["P3"] = {
        "random_26": {
            "percentile": percentile_of(ld(pub), r26),
            "p": one_sided_p(ld(pub), r26),
            "median": statistics.median(r26),
            "p95": sorted(r26)[int(0.95 * (len(r26) - 1))],
            "max": max(r26),
            "passes": percentile_of(ld(pub), r26) >= P3_PERCENTILE_BAR,
        },
        "layer_matched_26": {
            "percentile": percentile_of(ld(pub), lm),
            "p": one_sided_p(ld(pub), lm),
            "median": statistics.median(lm),
            "p95": sorted(lm)[int(0.95 * (len(lm) - 1))],
            "max": max(lm),
            "mean_overlap": statistics.fmean([r["published_overlap"] for r in d["layer_matched"]]),
            "passes": percentile_of(ld(pub), lm) >= P3_PERCENTILE_BAR,
        },
    }

    all_null = [(r["size"], ld(r), r["published_overlap"]) for rows_ in null.values() for r in rows_]
    out["P4"] = {
        "n_random_sets": len(all_null),
        "spearman_size": spearman([a for a, _, _ in all_null], [b for _, b, _ in all_null]),
    }

    named = [
        (r["residual"], r["overlap"] / max(r["size"], 1))
        for r in rows
        if r["residual"] == r["residual"]
    ]
    for kind, fam in out["P2"]["families"].items():
        for k, v in fam.items():
            size = v["size"]
            overlap = min(size, 26) if kind == "add" else size
            if v["residual"] == v["residual"]:
                named.append((v["residual"], overlap / size))
    out["P4"]["spearman_residual_overlap"] = spearman(
        [a for a, _ in named], [b for _, b in named]
    )
    out["P4"]["n_residual_points"] = len(named)

    # post-hoc
    for m in (18, 26, 34):
        if m in null:
            out["P4"][f"within_{m}_spearman_overlap"] = spearman(
                [r["published_overlap"] for r in null[m]], [ld(r) for r in null[m]]
            )

    aoi = {h: ld(r) for h, r in d["aoi"].items()}
    p1 = json.loads((RESULTS / "phase1_results.json").read_text())
    patch = {
        scheme: {h: abs(v) for h, v in p1["schemes"][scheme]["effects"].items()}
        for scheme in ("s2_swap", "abc")
    }
    patch["max_of_both"] = {
        h: max(patch["s2_swap"][h], patch["abc"][h]) for h in patch["s2_swap"]
    }
    aucs = {"aoi_scrub": auc(aoi, PUBLISHED)}
    for name, scores in patch.items():
        aucs[f"patch_{name}"] = auc(scores, PUBLISHED)
    loo = {h: ld(r) for h, r in d["loo"].items()}
    out["P5"] = {
        "auc": aucs,
        "best_patching": max(aucs[k] for k in aucs if k.startswith("patch_")),
        "delta": aucs["aoi_scrub"] - max(aucs[k] for k in aucs if k.startswith("patch_")),
        "aoi_permutation_p": auc_permutation_p(aoi, len(PUBLISHED), aucs["aoi_scrub"]),
        "spearman_aoi_vs_patch": spearman(
            [aoi[h] for h in sorted(aoi)], [patch["max_of_both"][h] for h in sorted(aoi)]
        ),
        "loo_range": {"min": min(loo.values()), "max": max(loo.values())},
        "loo_by_class": {},
    }
    by_class: dict[str, list[float]] = {}
    for head, value in loo.items():
        l, h = head.split(".")
        cls = HEAD_TO_CLASS[(int(l), int(h))]
        by_class.setdefault(cls, []).append(ld(pub) - value)
    out["P5"]["loo_by_class"] = {
        c: {"n": len(v), "median_drop": statistics.median(v), "max_drop": max(v)}
        for c, v in sorted(by_class.items(), key=lambda kv: -statistics.median(kv[1]))
    }
    out["P5"]["top_aoi"] = sorted(aoi.items(), key=lambda kv: -kv[1])[:15]

    sds = [r["logit_diff"]["sd"] for rows_ in null.values() for r in rows_]
    means = [ld(r) for rows_ in null.values() for r in rows_]
    out["P6"] = {
        "spearman_sd_vs_recovered": spearman(means, sds),
        "median_sd": statistics.median(sds),
        "published_sd": pub["logit_diff"]["sd"],
    }

    out["variants"] = {
        kind: {
            name: {"logit_diff": ld(row), "kl": row["kl"]["mean"], "size": row["size"]}
            for name, row in group.items()
            if name != "_anchors"
        }
        for kind, group in d["variants"].items()
    }

    union_gap = abs(ld(cand["phase1_union"]) - ld(pub))
    preds = [
        ("1", "P1 passes, published_26 clears 0.70 on logit_diff",
         fmt(ld(pub)), out["P1"]["passes_logit"]),
        ("2", "published_26 logit_diff recovered below 0.87",
         fmt(ld(pub)), ld(pub) < 0.87),
        ("3", "kl recovered below logit_diff recovered",
         f"{fmt(pub['kl']['mean'])} vs {fmt(ld(pub))}", pub["kl"]["mean"] < ld(pub)),
        ("4", "phase1_union within 0.10 of published_26",
         fmt(union_gap), union_gap <= 0.10),
        ("5", "phase1_matches worse than published_26 and inside drop_6's 10-90",
         f"{fmt(ld(matches))} vs {fmt(ld(pub))}, pct {fmt(p2['percentile'], 2)}",
         ld(matches) < ld(pub) and p2["inside_10_90"]),
        ("6", "P3 passes on random_26",
         fmt(out["P3"]["random_26"]["percentile"], 3), out["P3"]["random_26"]["passes"]),
        ("7", "P3 fails on layer_matched_26",
         fmt(out["P3"]["layer_matched_26"]["percentile"], 3),
         not out["P3"]["layer_matched_26"]["passes"]),
        ("8", "Spearman(recovered, |H|) over the random sweep >= 0.9",
         fmt(out["P4"]["spearman_size"]), out["P4"]["spearman_size"] >= PRED8_BAR),
        ("9", "aoi scrub AUC does not beat the best patching AUC by more than +0.02",
         fmt(out["P5"]["delta"]), out["P5"]["delta"] <= 0.02),
    ]
    out["predictions"] = [
        {"n": n, "text": t, "observed": o, "held": bool(h)} for n, t, o, h in preds
    ]
    out["predictions_held"] = sum(1 for p in out["predictions"] if p["held"])

    TESTS.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("null_curve", "P2")}, indent=2)[:6000])
    print("wrote", TESTS)


if __name__ == "__main__":
    main()
