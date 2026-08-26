"""phase 9, step 1: what is measurably different about the two flagged sets?"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CIRCUITS = ("docstring", "greater_than")
METRIC = "logit_diff"
THRESHOLD = 0.02  # phase 1's, the one phase 8's flag used

KNOWN_CASE = {
    "docstring": "real blind spot (phase 7: the primary counterfactual hides routing heads)",
    "greater_than": "noise (phase 6: the primary counterfactual already recovered everything)",
}


def _load(circuit: str) -> dict:
    path = RESULTS_DIR / f"phase8_{circuit}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}. run phase 8 first")
    return json.loads(path.read_text(encoding="utf-8"))


def _effects(payload: dict, scheme: str, metric: str = METRIC) -> dict[str, float]:
    return payload["discovery"]["runs"][scheme]["effects"][metric]


def scheme_scale(effects: dict[str, float]) -> dict[str, float]:
    """Descriptors of one scheme's own effect distribution, over every head it swept."""
    values = sorted(abs(v) for v in effects.values())
    n = len(values)
    return {
        "n_heads": n,
        "median": values[n // 2],
        "p75": values[int(0.75 * n)],
        "p90": values[int(0.90 * n)],
        "p95": values[int(0.95 * n)],
        "p99": values[min(int(0.99 * n), n - 1)],
        "max": values[-1],
        "mean": statistics.fmean(values),
        "n_over_threshold": sum(1 for v in values if v >= THRESHOLD),
    }


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    n = len(keys)
    ra = {k: i for i, k in enumerate(sorted(keys, key=lambda k: abs(a[k])))}
    rb = {k: i for i, k in enumerate(sorted(keys, key=lambda k: abs(b[k])))}
    d2 = sum((ra[k] - rb[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def head_row(payload: dict, head: str, schemes: list[str], primary: str,
             scales: dict[str, dict], power: dict[str, dict],
             provenance: dict[str, str]) -> dict:
    """every candidate signal for one head, from stored measurements only."""
    effects = {s: _effects(payload, s).get(head, 0.0) for s in schemes}
    primary_effect = effects[primary]
    finders = [s for s in schemes if abs(effects[s]) >= THRESHOLD and s != primary]

    best = max(finders, key=lambda s: abs(effects[s])) if finders else None
    ratios = {}
    if best:
        scale = scales[best]
        ratios = {
            "best_scheme": best,
            "best_effect": effects[best],
            "best_over_median": abs(effects[best]) / scale["median"] if scale["median"] else float("inf"),
            "best_over_p90": abs(effects[best]) / scale["p90"] if scale["p90"] else float("inf"),
            "best_rank": sorted(
                (abs(v) for v in _effects(payload, best).values()), reverse=True
            ).index(abs(effects[best])) + 1,
            "best_provenance": provenance[best],
            "best_power": power[best]["power"],
        }

    signs = {s: (1 if effects[s] > 0 else -1 if effects[s] < 0 else 0) for s in schemes}
    finder_signs = {signs[s] for s in finders} if finders else set()

    extra = {}
    if best:
        scheme_max = max(abs(v) for v in _effects(payload, best).values())
        extra["prominence"] = abs(effects[best]) / scheme_max if scheme_max else float("nan")
        per_metric = {
            m: _effects(payload, best, m).get(head, 0.0) for m in ("logit_diff", "kl", "tv")
        }
        extra["per_metric"] = per_metric
        extra["n_metrics"] = sum(1 for v in per_metric.values() if abs(v) >= THRESHOLD)

    return {
        "head": head,
        "effects": effects,
        "primary_effect": primary_effect,
        "primary_abs": abs(primary_effect),

        "primary_over_threshold": abs(primary_effect) / THRESHOLD,
        "primary_over_median": (
            abs(primary_effect) / scales[primary]["median"] if scales[primary]["median"] else float("inf")
        ),
        "n_finders": len(finders),
        "finders": finders,
        "finder_provenances": sorted({provenance[s] for s in finders}),
        "detection_ratio": (
            abs(effects[best]) / abs(primary_effect) if best and primary_effect else float("inf")
        ),
        "sign_consistent": len(finder_signs) <= 1,
        **ratios,
        **extra,
    }


def characterize(circuit: str) -> dict:
    payload = _load(circuit)
    report = payload["head_agreement"]
    schemes = list(report["schemes"])
    primary = report["primary"]
    provenance = {s["scheme"]: s["provenance"] for s in payload["schemes"]}
    scales = {s: scheme_scale(_effects(payload, s)) for s in schemes}

    flagged = report["blind_spots"][primary]
    robust = [v["head"] for v in report["verdicts"] if v["status"] == "robust"]

    rows = [
        head_row(payload, head, schemes, primary, scales, report["power"], provenance)
        for head in flagged
    ]
    robust_rows = [
        head_row(payload, head, schemes, primary, scales, report["power"], provenance)
        for head in robust
    ]

    def summary(items: list[dict], key: str) -> dict:
        values = [r[key] for r in items if key in r and r[key] == r[key]]
        finite = [v for v in values if v != float("inf")]
        if not finite:
            return {}
        return {
            "n": len(finite),
            "min": min(finite),
            "median": statistics.median(finite),
            "max": max(finite),
        }

    concordance = {
        s: {
            "spearman_with_primary": spearman(_effects(payload, primary), _effects(payload, s)),
            "jaccard_with_primary": next(
                (row["jaccard"] for row in payload["head_overlap"]
                 if {row["a"], row["b"]} == {primary, s}), float("nan"),
            ),
            "max_abs_effect": max(abs(v) for v in _effects(payload, s).values()),
        }
        for s in schemes if s != primary
    }

    return {
        "circuit": circuit,
        "known_case": KNOWN_CASE[circuit],
        "concordance": concordance,
        "primary_max_abs_effect": max(abs(v) for v in _effects(payload, primary).values()),
        "primary": primary,
        "schemes": schemes,
        "provenance": provenance,
        "power": {s: report["power"][s] for s in schemes},
        "scheme_scales": scales,
        "n_flagged": len(flagged),
        "flagged": rows,
        "robust": robust_rows,
        "summaries": {
            key: summary(rows, key)
            for key in ("primary_over_threshold", "primary_over_median", "detection_ratio",
                        "best_over_median", "best_over_p90", "best_rank", "n_finders",
                        "best_power", "prominence", "n_metrics", "best_effect")
        },
        "robust_summaries": {
            key: summary(robust_rows, key)
            for key in ("primary_over_threshold", "primary_over_median", "detection_ratio",
                        "best_over_median", "best_over_p90", "best_rank", "n_finders",
                        "prominence", "n_metrics", "best_effect")
        },
        "flagged_by_provenance": {
            p: sum(1 for r in rows if p in r["finder_provenances"])
            for p in ("published", "authored", "generic")
        },
        "flagged_by_sole_provenance": {
            p: sum(1 for r in rows if r["finder_provenances"] == [p])
            for p in ("published", "authored", "generic")
        },
        "sign_inconsistent": sum(1 for r in rows if not r["sign_consistent"]),
    }


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _fmt(value: float) -> str:
    if value != value:
        return ", "
    if value == float("inf"):
        return "∞"
    return f"{value:.2f}"


def write_markdown(path: Path, data: dict[str, dict]) -> None:
    """the characterization write-up. ground truth appears only in the last section."""
    from causal_interp import ground_truth as gt_ioi  # noqa: PLC0415
    from causal_interp import ground_truth_docstring as gt_doc  # noqa: PLC0415
    from causal_interp import ground_truth_greater_than as gt_gt  # noqa: PLC0415

    published = {
        "docstring": {f"{l}.{h}" for l, h in gt_doc.ALL_HEADS},
        "greater_than": {f"{l}.{h}" for l, h in gt_gt.ALL_HEADS},
    }

    out = [
        "# Phase 9, step 1, what is measurably different about the two flagged sets",
        "",
        "measurement before design. phase 8's flag fired on both circuits and couldnt "
        "tell them apart. before proposing any rule, this is every quantity the pipeline "
        "already computed, for every head it flagged, on both circuits. nothing here is a "
        "criterion. the candidate signals are described so that the rule fixed in "
        "[PHASE9_PLAN.md](PHASE9_PLAN.md) can be justified against measurements that "
        "already existed rather than invented to fit.",
        "",
        "generated by `scripts/phase9_characterize.py` from `results/phase8_*.json`. no new "
        "model runs, and the script reads no `ground_truth` module outside the final "
        "section.",
        "",
    ]

    for circuit, block in data.items():
        out += [
            f"## {circuit}, known to be: {block['known_case']}",
            "",
            f"primary scheme `{block['primary']}`, {block['n_flagged']} heads flagged.",
            "",
            "### each scheme's own effect distribution",
            "",
            "normalized recovery divides by that scheme's own clean-vs-corrupted span, so "
            "the schemes are **not on one scale to begin with**, a scheme with a small span "
            "turns small absolute changes into large normalized ones. phase 8's flag "
            "compared all of them against one shared cutoff of 0.02:",
            "",
        ]
        rows = []
        for scheme in block["schemes"]:
            scale = block["scheme_scales"][scheme]
            power = block["power"][scheme]
            rows.append([
                f"`{scheme}`" + (" *(primary)*" if scheme == block["primary"] else ""),
                block["provenance"][scheme],
                f"{power['power']:.2f}",
                f"{scale['median']:.4f}",
                f"{scale['p90']:.4f}",
                f"{scale['max']:.3f}",
                f"{scale['n_over_threshold']}/{scale['n_heads']}",
            ])
        out += [
            _table(rows, ["scheme", "provenance", "power", "median abs effect",
                          "p90 abs effect", "max", "heads over 0.02"]),
            "",
            "### does each scheme agree with the primary about anything else?",
            "",
            "if a scheme ranks the whole circuit roughly as the primary does, its one "
            "disagreement is a more interesting witness than a scheme that agrees about "
            "nothing:",
            "",
        ]
        rows = []
        for scheme, conc in block["concordance"].items():
            rows.append([
                f"`{scheme}`",
                block["provenance"][scheme],
                f"{conc['spearman_with_primary']:+.3f}",
                f"{conc['jaccard_with_primary']:.3f}",
                f"{conc['max_abs_effect']:.3f}",
                f"{conc['max_abs_effect'] / block['primary_max_abs_effect']:.2f}",
            ])
        out += [
            _table(rows, ["scheme", "provenance", "rank correlation with primary",
                          "Jaccard of discovered sets", "its strongest head",
                          "÷ primary's strongest"]),
            "",
            "### the flagged heads",
            "",
        ]
        rows = []
        for r in sorted(block["flagged"], key=lambda r: -r.get("best_over_median", 0)):
            rows.append([
                f"`{r['head']}`",
                f"{r['primary_effect']:+.4f}",
                _fmt(r["primary_over_threshold"]),
                f"`{r.get('best_scheme', "n/a")}`",
                f"{r.get('best_effect', 0):+.4f}",
                _fmt(r.get("prominence", float("nan"))),
                _fmt(r.get("best_over_median", float("nan"))),
                str(r.get("best_rank", "n/a")),
                str(r["n_finders"]),
                str(r.get("n_metrics", "n/a")),
                ",".join(p[:3] for p in r["finder_provenances"]),
                _fmt(r["detection_ratio"]),
            ])
        out += [
            _table(rows, ["head", "primary effect", "primary/0.02", "best other scheme",
                          "its effect", "prominence", "÷ that scheme's median", "rank in it",
                          "finders", "metrics ≥ 0.02", "provenance", "detection ratio"]),
            "",
        ]

        out += ["### summary of the candidate signals", ""]
        rows = []
        for key, label in (
            ("primary_over_threshold", "primary's own |effect| ÷ 0.02"),
            ("primary_over_median", "primary's own |effect| ÷ its median |effect|"),
            ("detection_ratio", "best other scheme's |effect| ÷ primary's"),
            ("best_over_median", "best scheme's |effect| ÷ that scheme's median"),
            ("best_over_p90", "best scheme's |effect| ÷ that scheme's p90"),
            ("best_rank", "rank of the head inside the scheme that found it"),
            ("n_finders", "how many non-primary schemes found it"),
            ("prominence", "its |effect| ÷ the strongest head in the scheme that found it"),
            ("best_effect", "its raw normalized recovery in the scheme that found it"),
            ("n_metrics", "how many of the three metrics put it over 0.02"),
        ):
            s = block["summaries"].get(key, {})
            rb = block["robust_summaries"].get(key, {})
            if not s:
                continue
            rows.append([
                label,
                f"{_fmt(s['min'])} / **{_fmt(s['median'])}** / {_fmt(s['max'])}",
                f"{_fmt(rb['min'])} / {_fmt(rb['median'])} / {_fmt(rb['max'])}" if rb else ", ",
            ])
        out += [
            _table(rows, ["signal", "flagged heads: min / **median** / max",
                          "robust heads: min / median / max"]),
            "",
            f"by provenance of the scheme that found them, flagged heads found *only* by a "
            f"scheme of that kind: published {block['flagged_by_sole_provenance']['published']}, "
            f"authored {block['flagged_by_sole_provenance']['authored']}, "
            f"generic {block['flagged_by_sole_provenance']['generic']}. "
            f"sign-inconsistent across their finders: {block['sign_inconsistent']}.",
            "",
        ]

    # -- the answer key, last and marked ------------------------------------
    out += [
        "## Which flagged heads were real, the answer key, consulted only here",
        "",
        "everything above is computable on a circuit with no published answer. this section "
        "exists to say which of those numbers a rule would need to separate.",
        "",
    ]
    rows = []
    for circuit, block in data.items():
        for r in sorted(block["flagged"], key=lambda r: -r.get("best_over_median", 0)):
            if r["head"] not in published[circuit]:
                continue
            rows.append([
                circuit,
                f"`{r['head']}`",
                f"{r['primary_effect']:+.4f}",
                _fmt(r["primary_over_threshold"]),
                f"`{r.get('best_scheme', "n/a")}`",
                _fmt(r.get("best_over_median", float("nan"))),
                _fmt(r["detection_ratio"]),
            ])
    out += [
        _table(rows, ["circuit", "published head flagged", "primary effect", "primary/0.02",
                      "found by", "effect / that scheme's median", "detection ratio"]),
        "",
        "every published head in either flagged set is a docstring head. greater-than "
        "contributed none. a rule that separates the cases has to keep the rows above and "
        "drop most of the rest.",
        "",
    ]
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    data = {circuit: characterize(circuit) for circuit in CIRCUITS}
    (RESULTS_DIR / "phase9_characterization.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    write_markdown(RESULTS_DIR / "PHASE9_CHARACTERIZATION.md", data)
    for circuit, block in data.items():
        print(f"\n{circuit}: {block['n_flagged']} flagged  [{block['known_case']}]")
        for key in ("primary_over_threshold", "detection_ratio", "best_over_median",
                    "best_rank", "n_finders"):
            s = block["summaries"].get(key, {})
            if s:
                print(f"  {key:24} min {_fmt(s['min']):>6}  median {_fmt(s['median']):>6}"
                      f"  max {_fmt(s['max']):>6}")
        print(f"  sole-provenance finders: {block['flagged_by_sole_provenance']}")
    print(f"\nwrote {RESULTS_DIR / 'PHASE9_CHARACTERIZATION.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
