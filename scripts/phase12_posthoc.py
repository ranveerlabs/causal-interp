"""phase 12 post-hoc, none of this is registered in PHASE12_PLAN.md."""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_interp.ground_truth import ALL_HEADS, HEAD_TO_CLASS

_spec = importlib.util.spec_from_file_location("phase12_report", ROOT / "scripts" / "phase12_report.py")
_r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r)
auc, auc_permutation_p, spearman = _r.auc, _r.auc_permutation_p, _r.spearman

RESULTS = ROOT / "results"
PUBLISHED = {f"{l}.{h}" for l, h in ALL_HEADS}


def ranks(scores: dict[str, float]) -> dict[str, float]:
    order = sorted(scores, key=lambda k: scores[k])
    return {k: i for i, k in enumerate(order)}


def main() -> None:
    d = json.loads((RESULTS / "phase12_scrub.json").read_text())
    p1 = json.loads((RESULTS / "phase1_results.json").read_text())
    out: dict = {}

    aoi = {h: r["logit_diff"]["mean"] for h, r in d["aoi"].items()}
    patch = {
        h: max(abs(p1["schemes"]["s2_swap"]["effects"][h]), abs(p1["schemes"]["abc"]["effects"][h]))
        for h in aoi
    }

    ra, rp = ranks(aoi), ranks(patch)
    combined = {h: ra[h] + rp[h] for h in aoi}
    out["combined_ranking"] = {
        "auc_aoi": auc(aoi, PUBLISHED),
        "auc_patch": auc(patch, PUBLISHED),
        "auc_combined": auc(combined, PUBLISHED),
        "p_combined": auc_permutation_p(combined, len(PUBLISHED), auc(combined, PUBLISHED)),
        "spearman_aoi_patch": spearman(
            [aoi[h] for h in sorted(aoi)], [patch[h] for h in sorted(aoi)]
        ),
    }

    pub = d["candidates"]["published_26"]
    noise = pub["logit_diff"]["sd"]
    out["drop_flatness"] = {
        "published": pub["logit_diff"]["mean"],
        "draw_sd": noise,
        "by_k": {
            k: {
                "size": rows[0]["size"],
                "mean": statistics.fmean([r["logit_diff"]["mean"] for r in rows]),
                "sd_across_sets": statistics.stdev([r["logit_diff"]["mean"] for r in rows]),
                "frac_at_or_above_published": sum(
                    1 for r in rows if r["logit_diff"]["mean"] >= pub["logit_diff"]["mean"]
                )
                / len(rows),
            }
            for k, rows in d["families"]["drop"].items()
        },
    }

    matches = set(p1["schemes"]["s2_swap"]["headline"]["matches"]) | set(
        p1["schemes"]["abc"]["headline"]["matches"]
    )
    missed = sorted(PUBLISHED - matches)
    loo = {h: r["logit_diff"]["mean"] for h, r in d["loo"].items()}
    out["phase1_misses"] = {
        "heads": missed,
        "detail": {
            h: {
                "class": HEAD_TO_CLASS[tuple(int(x) for x in h.split("."))],
                "loo_drop": pub["logit_diff"]["mean"] - loo[h],
                "aoi_gain": aoi[h],
                "patch_magnitude": patch[h],
            }
            for h in missed
        },
        "joint": d["candidates"]["phase1_matches"]["logit_diff"]["mean"],
        "sum_of_individual_loo_drops": sum(pub["logit_diff"]["mean"] - loo[h] for h in missed),
    }

    r26 = [r["logit_diff"]["mean"] for r in d["null"]["26"]]
    out["random_26"] = {
        "n": len(r26),
        "beating_published": sum(1 for v in r26 if v >= pub["logit_diff"]["mean"]),
        "above_the_P1_bar": sum(1 for v in r26 if v >= 0.70),
        "frac_above_bar": sum(1 for v in r26 if v >= 0.70) / len(r26),
    }

    out["clearing_the_bar"] = {
        name: row["logit_diff"]["mean"]
        for name, row in d["candidates"].items()
        if row["logit_diff"]["mean"] >= 0.70
    }

    out["mlp_scrub"] = {
        "anchors": d["variants"]["mlp_scrub"]["_anchors"],
        "published": d["variants"]["mlp_scrub"]["published_26"]["logit_diff"]["mean"],
        "best_candidate": max(
            (r["logit_diff"]["mean"] for n, r in d["variants"]["mlp_scrub"].items() if n != "_anchors")
        ),
    }

    (RESULTS / "phase12_posthoc.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
