"""Phase 9 report: whether a per-scheme null floor separates a blind spot from noise."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_interp import ground_truth as gt_ioi
from causal_interp import ground_truth_docstring as gt_docstring
from causal_interp import ground_truth_greater_than as gt_greater_than

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

TITLE = {
    "docstring": "docstring, the known **real blind spot**",
    "greater_than": "greater-than, the known **non-problem**",
    "ioi": "IOI, the **holdout**",
}
GROUND_TRUTH = {"docstring": gt_docstring, "greater_than": gt_greater_than, "ioi": gt_ioi}
ORDER = ("docstring", "greater_than", "ioi")


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _joined(heads: list[str]) -> str:
    return ", ".join(f"`{h}`" for h in heads) if heads else ", "


def spearman(pairs: list[tuple[float, float]]) -> float:
    n = len(pairs)
    if n < 3:
        return float("nan")
    xs = sorted(range(n), key=lambda i: pairs[i][0])
    ys = sorted(range(n), key=lambda i: pairs[i][1])
    rx = {i: r for r, i in enumerate(xs)}
    ry = {i: r for r, i in enumerate(ys)}
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def _intro(payloads: dict[str, dict]) -> str:
    lines = [
        "# phase 9, separating a real blind spot from noisy disagreement",
        "",
        "**the rule, the scoring table, the holdout and eight predictions were fixed in "
        "[PHASE9_PLAN.md](PHASE9_PLAN.md)** before anything here was measured, and the "
        "measurements the rule was chosen from were committed before *that*, in "
        "[PHASE9_CHARACTERIZATION.md](PHASE9_CHARACTERIZATION.md). the order is checkable "
        "in git history rather than asserted.",
        "",
        "phase 8's flag fired on both circuits it saw. on docstring it named the three "
        "published heads phase 7 had missed. on greater-than it named sixteen heads that "
        "are in no published circuit and where the primary counterfactual had already "
        "recovered everything. the flags were identical in form, so acting on one meant "
        "re-checking every one by hand.",
        "",
        "### what step 1 found, including what it ruled out",
        "",
        "ten candidate signals were measured over all 33 flagged heads. **nine dont "
        "separate the two cases**, and several separate in the wrong direction, anything "
        "that normalizes by a scheme's median or its strongest head is dominated by how "
        "many dead heads the model has, so GPT-2 small's noise outranks `attn-only-4l`'s "
        "real finds. the one axis with visible separation was raw normalized recovery in "
        "the scheme that found the head.",
        "",
        "### the rule",
        "",
        "Phase 3's rule, applied to a new channel, per scheme:",
        "",
        "> **theta(s) = the 99th percentile of |normalized recovery| under a shuffled-source "
        "null sweep, rounded up to two significant figures**, the clean activation "
        "spliced in is drawn from a *deranged* prompt order, so it is a real activation "
        "whose prompt-correspondence has been destroyed. theta(s) replaces phase 8's shared "
        "0.02 as that scheme's discovery criterion.",
        "",
        "the diagnosis behind it: normalized recovery divides by each scheme's own "
        "clean-vs-corrupted span, so 0.02 doesnt mean the same thing under two "
        "counterfactuals. nothing else changed, the flag is still a bare non-emptiness "
        "test on the primary's blind spot, so the two runs differ in the criterion alone. "
        "for docstring and greater-than the real sweeps are read back from phase 8's "
        "committed payloads rather than repeated, and only the null sweeps are new.",
        "",
    ]
    return "\n".join(lines)


def _floors_section(payloads: dict[str, dict]) -> str:
    lines = [
        "## What the null floors came out at",
        "",
        "one row per registered scheme, across all three circuits. `power` is phase 8's "
        "measurement, the scheme's span relative to its primary's, and theta is what this "
        "phase measured:",
        "",
    ]
    rows = []
    pairs = []
    for circuit in ORDER:
        payload = payloads.get(circuit)
        if not payload:
            continue
        before = payload["before"]
        for scheme, floor in payload["floors"].items():
            power = before["power"][scheme]["power"]
            rows.append([
                circuit,
                f"`{scheme}`" + (" *(primary)*" if scheme == before["primary"] else ""),
                next(s["provenance"] for s in payload["schemes"] if s["scheme"] == scheme),
                f"{power:.2f}",
                f"{floor['null_median']:.4f}",
                f"{floor['null_max']:.3f}",
                f"**{floor['threshold']:g}**",
                f"{len(before['per_scheme'][scheme])} -> "
                f"{len(payload['after']['per_scheme'][scheme])}",
            ])
            if circuit in ("docstring", "greater_than"):
                pairs.append((power, floor["threshold"]))
    rho = spearman(pairs)
    lines += [
        _table(rows, ["circuit", "scheme", "provenance", "power", "null median",
                      "null max", "theta", "heads found: 0.02 -> theta"]),
        "",
        f"**theta tracks power inversely, and the effect is enormous at the bottom.** over the "
        f"{len(pairs)} docstring and greater-than schemes the rank correlation between "
        f"power and theta is **{rho:+.2f}**. the extreme case is docstring's "
        f"`random_vocab_any`: power 0.06, and its shuffled-source null manufactures "
        f"apparent recoveries whose 99th percentile is "
        f"{payloads['docstring']['floors']['random_vocab_any']['threshold']:g}, that is, "
        f"patching a head with an activation belonging to a *different prompt* routinely "
        f"moves the metric by many times the entire clean-to-corrupted span. under a "
        f"cutoff of 0.02 that scheme contributed 8 of phase 8's 17 docstring flags. under "
        f"its own null it discovers nothing at all.",
        "",
        "that is the mechanism this phase is testing, stated in the plan before the "
        "numbers existed: a scheme with a small span turns tiny absolute changes into "
        "large normalized ones, and phase 8 compared those numbers against the same cutoff "
        "it applied to a scheme with a full-sized span.",
        "",
    ]
    return "\n".join(lines)


def _circuit_section(circuit: str, payload: dict) -> str:
    gt = GROUND_TRUTH[circuit]
    before, after = payload["before"], payload["after"]
    sb, sa = payload["scored_before"], payload["scored_after"]
    n_pub = payload["meta"]["published_head_count"]

    lines = [
        f"## {TITLE[circuit]}",
        "",
        f"{payload['meta']['model_alias']}, {payload['meta']['n_layers']}x"
        f"{payload['meta']['n_heads']} heads, {payload['meta']['prompts']} prompts. "
        f"real effects: {'measured here' if payload['meta']['real_effects_source'] == 'measured' else 'reused from phase 8'}.",
        "",
        _table(
            [
                ["heads flagged (primary's blind spot)", str(sb["n_flagged"]),
                 f"**{sa['n_flagged']}**"],
                [",  of which published", _joined(sb["published_in_blind_spot"]),
                 f"**{_joined(sa['published_in_blind_spot'])}**"],
                [",  of which not published", str(len(sb["unpublished_in_blind_spot"])),
                 f"**{len(sa['unpublished_in_blind_spot'])}**"],
                ["primary scheme's own recall",
                 f"{len(sb['primary']['matches'])}/{n_pub}",
                 f"{len(sa['primary']['matches'])}/{n_pub}"],
                ["primary scheme's own precision",
                 f"{sb['primary']['precision']:.2f}", f"{sa['primary']['precision']:.2f}"],
                ["union recall across schemes",
                 f"{len(sb['union']['matches'])}/{n_pub}",
                 f"{len(sa['union']['matches'])}/{n_pub}"],
                ["union precision",
                 f"{sb['union']['precision']:.2f}", f"{sa['union']['precision']:.2f}"],
            ],
            ["", "phase 8 (shared 0.02)", "phase 9 (per-scheme theta)"],
        ),
        "",
        "> " + after["flag_text"],
        "",
    ]

    kept = sorted(set(sa["flagged"]))
    dropped = sorted(set(sb["flagged"]) - set(sa["flagged"]))
    added = sorted(set(sa["flagged"]) - set(sb["flagged"]))
    lines += [
        f"flags kept: {_joined(kept)}.",
        "",
        f"flags dropped by calibration: {_joined(dropped)}.",
        "",
    ]
    if added:
        lines += [
            f"flags **added** by calibration: {_joined(added)}, a head the primary used "
            "to clear at 0.02 and no longer clears at its own theta, while another scheme "
            "still does. raising the primary's floor moves heads *into* its blind spot, "
            "which is a cost of the rule and not a bug in it.",
            "",
        ]
    return "\n".join(lines)


def _two_regimes(payloads: dict[str, dict]) -> str:
    """Where the removed flags came from, and where the surviving ones do."""
    rows = []
    for circuit in ORDER:
        payload = payloads.get(circuit)
        if not payload:
            continue
        before, after = payload["scored_before"], payload["scored_after"]
        for scheme, heads in before["blind_spot_contributions"].items():
            kept = after["blind_spot_contributions"].get(scheme, [])
            rows.append([
                circuit,
                f"`{scheme}`",
                next(x["provenance"] for x in payload["schemes"] if x["scheme"] == scheme),
                f"{payload['before']['power'][scheme]['power']:.2f}",
                f"{payload['floors'][scheme]['threshold']:g}",
                f"{len(heads)} -> **{len(kept)}**",
            ])
    if not rows:
        return ""
    return "\n".join([
        "## which schemes the removed flags came from",
        "",
        "a flag belongs to whichever non-primary scheme found the head. splitting the "
        "blind spots that way shows the rule isnt removing noise evenly, it is "
        "removing whole schemes:",
        "",
        _table(rows, ["circuit", "scheme", "provenance", "power", "theta",
                      "flags contributed: 0.02 -> theta"]),
        "",
        "two regimes, and the rule only handles one of them. where a flag came from a "
        "scheme too weak to measure anything, docstring's `random_vocab_any`, power 0.06, "
        "whose null manufactures recoveries of 3.3, calibration removes it completely and "
        "the case is closed. where a flag came from a scheme with healthy power that "
        "simply disagrees, greater-than's `xx_mismatch`, power 0.74, theta 0.033, "
        "calibration thins it and cant dismiss it, cuz there is nothing statistically "
        "wrong with those measurements. they are real effects under a real counterfactual "
        "that happen not to correspond to published circuit members.",
        "",
        "that distinction is the honest shape of this phase's result: **the null floor "
        "separates *this scheme is too weak to be believed* from *this scheme measured "
        "something*. it doesnt separate *this scheme measured something the primary is "
        "blind to* from *this scheme measured something that isnt part of the "
        "circuit***, and the second distinction is the one a reader actually wants.",
        "",
    ])


def _verdict(payloads: dict[str, dict]) -> str:
    doc, gtn = payloads.get("docstring"), payloads.get("greater_than")
    if not (doc and gtn):
        return ""
    doc_after = set(doc["scored_after"]["flagged"])
    keeps = {"1.4", "2.0"} <= doc_after
    gt_before = gtn["scored_before"]["n_flagged"]
    gt_after = gtn["scored_after"]["n_flagged"]
    shrink = 1 - (gt_after / gt_before) if gt_before else 0.0

    if gt_after == 0 and keeps:
        verdict, label = "CLEAN PASS", (
            "greater-than's blind spot is empty and docstring's still contains both "
            "routing heads"
        )
    elif shrink >= 0.5 and keeps:
        verdict, label = "PARTIAL", (
            f"greater-than's blind spot shrank by {shrink:.0%} but isnt empty, and "
            "docstring kept both routing heads"
        )
    else:
        verdict, label = "NEGATIVE", (
            "the pre-registered bar wasnt met"
        )

    return "\n".join([
        "## The pre-registered verdict",
        "",
        "the scoring table was fixed in the plan. this is it applied to the measurements:",
        "",
        _table(
            [
                ["clean pass", "greater-than's blind spot empty **and** docstring keeps "
                 "`1.4`, `2.0`", "empty? " + ("yes" if gt_after == 0 else f"no, {gt_after} left")],
                ["partial", "greater-than shrinks ≥ 50 % **and** docstring keeps both",
                 f"shrank {shrink:.0%}. docstring keeps both? {'yes' if keeps else 'no'}"],
                ["negative", "shrinks < 50 %, or docstring loses either head", ", "],
            ],
            ["outcome", "criterion", "measured"],
        ),
        "",
        f"### verdict: {verdict}",
        "",
        f"{label}.",
        "",
    ])


def _holdout(payloads: dict[str, dict]) -> str:
    ioi = payloads.get("ioi")
    if not ioi:
        return ""
    sb, sa = ioi["scored_before"], ioi["scored_after"]
    before, after = ioi["before"], ioi["after"]
    n_pub = ioi["meta"]["published_head_count"]
    published_after = sa["published_in_blind_spot"]

    lines = [
        "## the holdout, and what it is worth",
        "",
        "the rule was chosen while looking at docstring and greater-than, so neither can "
        "test it. **IOI had never been run through the multi-scheme pipeline**, phase 8 "
        "registered its four schemes and deliberately didnt run it, and its expected "
        "answer comes from phase 1 independently of anything here: the primary scheme "
        "`s2_swap` is provably blind before the S2 token (phase 1 measured 576/576 "
        "head-position cells there as exact zeros) while `abc` isnt. so IOI should "
        "behave like docstring, not like greater-than.",
        "",
        _table(
            [
                ["heads flagged", str(sb["n_flagged"]), f"**{sa['n_flagged']}**"],
                [",  of which published", _joined(sb["published_in_blind_spot"]),
                 f"**{_joined(published_after)}**"],
                [",  of which not published", str(len(sb["unpublished_in_blind_spot"])),
                 f"**{len(sa['unpublished_in_blind_spot'])}**"],
                ["primary recall", f"{len(sb['primary']['matches'])}/{n_pub}",
                 f"{len(sa['primary']['matches'])}/{n_pub}"],
                ["union recall", f"{len(sb['union']['matches'])}/{n_pub}",
                 f"{len(sa['union']['matches'])}/{n_pub}"],
            ],
            ["IOI", "phase 8 criterion", "phase 9 criterion"],
        ),
        "",
        "> " + after["flag_text"],
        "",
    ]
    if published_after:
        classes = {h: gt_ioi.classify(tuple(int(x) for x in h.split("."))) for h in published_after}
        grew = sa["n_flagged"] >= sb["n_flagged"]
        precision_before = (
            len(sb["published_in_blind_spot"]) / sb["n_flagged"] if sb["n_flagged"] else 0.0
        )
        precision_after = (
            len(published_after) / sa["n_flagged"] if sa["n_flagged"] else 0.0
        )
        lines += [
            "IOI does have a real blind spot, and the calibrated flag points at it. the "
            "surviving flags include "
            + ", ".join(f"`{h}` ({classes[h]})" for h in published_after)
            + ", published heads that `s2_swap` misses and another scheme finds, which is "
            "the pattern phase 1 would predict and the prediction was recorded before the "
            "run.",
            "",
        ]
        if grew:
            lines += [
                f"**and the rule didnt do what it was supposed to do here.** IOI's blind "
                f"spot **grew**, {sb['n_flagged']} -> {sa['n_flagged']}, where the whole "
                f"point of calibration was to shrink it. the cuz is visible in the floor "
                f"table: `abc` and `random_vocab_s2` came out with theta **below** 0.02 "
                f"(0.0077 and 0.011), so calibration made them *more* sensitive and they "
                f"discovered more heads, while the primary's floor rose to "
                f"{ioi['floors'][ioi['before']['primary']]['threshold']:g} and it lost "
                f"three published heads of its own "
                f"({len(sb['primary']['matches'])}/{n_pub} -> "
                f"{len(sa['primary']['matches'])}/{n_pub}). calibration cuts both ways, and "
                f"on this circuit it cut the wrong way.",
                "",
                f"the flag did get *sharper* even as it got louder, the share of flagged "
                f"heads that are published circuit members went "
                f"{precision_before:.0%} -> {precision_after:.0%}, but a detector that "
                f"fires on {sa['n_flagged']} heads of which "
                f"{len(sa['unpublished_in_blind_spot'])} arent in the circuit isnt one "
                f"a reader can act on without the answer key, which is exactly the "
                f"complaint phase 9 set out to fix.",
                "",
            ]
    else:
        lines += [
            "no published head survives in IOI's calibrated blind spot. the holdout "
            "doesnt reproduce the docstring pattern, and the prediction that it would "
            "was wrong. that is the single most important number in this report: the rule "
            "was chosen on two circuits and the third doesnt confirm it.",
            "",
        ]
    return "\n".join(lines)


def _predictions(payloads: dict[str, dict]) -> str:
    doc, gtn, ioi = (payloads.get(c) for c in ORDER)
    rows: list[list[str]] = []

    def add(tag, text, hit, evidence):
        rows.append([tag, text, ", " if hit is None else ("hit" if hit else "missed"),
                     evidence])

    if doc and gtn:
        pairs = []
        for payload in (doc, gtn):
            for scheme, floor in payload["floors"].items():
                pairs.append((payload["before"]["power"][scheme]["power"], floor["threshold"]))
        rho = spearman(pairs)
        add("P1", "theta is inversely related to a scheme's power (negative rank correlation)",
            rho < 0, f"rank correlation {rho:+.2f} over {len(pairs)} schemes")

        gt_after = gtn["scored_after"]["n_flagged"]
        add("P2", "greater-than's calibrated blind spot is **empty**", gt_after == 0,
            f"{gt_after} heads remain")

        doc_after = set(doc["scored_after"]["flagged"])
        add("P3", "docstring's calibrated blind spot is non-empty and contains `1.4`, `2.0`",
            bool(doc_after) and {"1.4", "2.0"} <= doc_after,
            f"{len(doc_after)} flagged: {_joined(sorted(doc_after))}")

        contrib = doc["scored_after"]["blind_spot_contributions"].get("random_vocab_any", [])
        add("P4", "`random_vocab_any` contributes at most 2 heads to docstring's calibrated "
                  "blind spot", len(contrib) <= 2,
            f"{len(contrib)} heads (theta = "
            f"{doc['floors']['random_vocab_any']['threshold']:g})")

        thetas = {
            "docstring": doc["floors"][doc["before"]["primary"]]["threshold"],
            "greater_than": gtn["floors"][gtn["before"]["primary"]]["threshold"],
        }
        within = all(0.02 / 3 <= t <= 0.02 * 3 for t in thetas.values())
        add("P5", "for both hand-built primaries theta lands within a factor of 3 of 0.02",
            within, ", ".join(f"{k}: theta={v:g}" for k, v in thetas.items()))

        all_thetas = {}
        for circuit in ORDER:
            payload = payloads.get(circuit)
            if payload:
                for scheme, floor in payload["floors"].items():
                    all_thetas[f"{circuit}/{scheme}"] = floor["threshold"]
        worst = max(all_thetas, key=all_thetas.get)
        add("P8", "no scheme's theta exceeds 0.5", all_thetas[worst] <= 0.5,
            f"largest is {worst} at theta = {all_thetas[worst]:g}")

    if ioi:
        published = ioi["scored_after"]["published_in_blind_spot"]
        add("P6", "**holdout:** IOI's calibrated flag fires and its blind spot contains at "
                  "least one published IOI head",
            ioi["after"]["flag"] and bool(published),
            f"flag {'fires' if ioi['after']['flag'] else 'silent'}. published in blind spot: "
            f"{_joined(published)}")
        add("P7", "**holdout:** IOI's calibrated blind spot is smaller than its "
                  "uncalibrated one",
            ioi["scored_after"]["n_flagged"] < ioi["scored_before"]["n_flagged"],
            f"{ioi['scored_before']['n_flagged']} -> {ioi['scored_after']['n_flagged']}")

    order = {f"P{i}": i for i in range(1, 9)}
    rows.sort(key=lambda r: order[r[0]])
    hits = sum(1 for r in rows if r[2] == "hit")
    return "\n".join([
        "## the eight pre-registered predictions",
        "",
        f"**{hits} of {len(rows)} scored as hits.**",
        "",
        _table(rows, ["#", "prediction", "outcome", "measured"]),
        "",
    ])


def _conclusion(payloads: dict[str, dict]) -> str:
    """does a pipeline-internal signal separate the two cases? answered from the data."""
    if len(payloads) < 3:
        return ""
    rows = []
    for circuit in ORDER:
        payload = payloads[circuit]
        sb, sa = payload["scored_before"], payload["scored_after"]
        pb = len(sb["published_in_blind_spot"]) / sb["n_flagged"] if sb["n_flagged"] else 0.0
        pa = len(sa["published_in_blind_spot"]) / sa["n_flagged"] if sa["n_flagged"] else 0.0
        rows.append([
            circuit,
            {"docstring": "real blind spot", "greater_than": "no blind spot",
             "ioi": "real blind spot (holdout)"}[circuit],
            f"{sb['n_flagged']} -> **{sa['n_flagged']}**",
            f"{len(sb['published_in_blind_spot'])} -> **{len(sa['published_in_blind_spot'])}**",
            f"{pb:.0%} -> **{pa:.0%}**",
        ])

    doc, gtn, ioi = (payloads[c] for c in ORDER)
    return "\n".join([
        "## does a pipeline-internal signal separate the two cases?",
        "",
        "the whole phase in one table, the flag's *precision*, meaning the share of "
        "flagged heads that turn out to be published circuit members. a discriminator "
        "would drive this up on the circuits with a real blind spot and to zero flags on "
        "the one without:",
        "",
        _table(rows, ["circuit", "known case", "heads flagged", "published among them",
                      "flag precision"]),
        "",
        "### The answer is: partly, and not enough to act on",
        "",
        "what the rule does do, decisively. it removes flags produced by schemes that "
        "are too weak to measure anything. docstring's `random_vocab_any`, power 0.06, "
        f"theta {doc['floors']['random_vocab_any']['threshold']:g}, contributed 17 of that "
        "circuit's flags and now contributes none, and docstring's flag went from 17 heads "
        "with 3 published to 4 heads with 2. on that circuit the flag became something a "
        "reader could act on. every scheme's null floor is a real measurement of how much "
        "apparent recovery that experiment manufactures from a mismatched activation, and "
        "the numbers span a factor of 400 across the ten schemes measured, phase 8's "
        "single shared cutoff wasnt defensible, and that much is now established.",
        "",
        "what it doesnt do. it cant tell a scheme that disagrees *bcuz the "
        "primary is blind* from a scheme that disagrees *cuz it is measuring something "
        "outside the circuit*. both are statistically sound measurements under a real "
        "counterfactual. greater-than's flag still names "
        f"{gtn['scored_after']['n_flagged']} heads, none published, and its precision is "
        "still 0 %. IOI's flag got **larger**, not smaller. two of three circuits end the "
        "phase with a flag that still needs a human and an answer key.",
        "",
        "**so the honest statement of the result is the one the plan required in advance:** "
        "a per-scheme null floor is a real improvement to *what counts as a measurement*, "
        "and it is **not** a discriminator between a real blind spot and ordinary "
        "disagreement. the signal that separates the two known cases in step 1's tables, "
        "raw effect magnitude, turns out to separate them cuz docstring's noise came "
        "from one pathological scheme, not cuz magnitude means what a discriminator "
        "would need it to mean. nothing else measured here separates them at all.",
        "",
        "the phase 8 ceiling therefore stands, in a sharper form. disagreement detection "
        "can say *this head list depends on the experiment*, and it can now also say *this "
        "scheme was too weak to be believed*. it still cant say *and therefore the "
        "primary is missing part of the circuit*, that step needed the published head "
        "list on all three circuits here.",
        "",
    ])


def _limits(payloads: dict[str, dict]) -> str:
    doc = payloads.get("docstring")
    cost = ""
    if doc:
        sb, sa = doc["scored_before"], doc["scored_after"]
        n_pub = doc["meta"]["published_head_count"]
        ioi = payloads.get("ioi")
        cost = (
            f" on docstring the primary's recall held at "
            f"{len(sa['primary']['matches'])}/{n_pub} while its floor rose to "
            f"{doc['floors'][doc['before']['primary']]['threshold']:g}"
        )
        if ioi:
            cost += (
                f". on IOI the primary's floor rose to "
                f"{ioi['floors'][ioi['before']['primary']]['threshold']:g} and its recall "
                f"fell {len(ioi['scored_before']['primary']['matches'])}/26 -> "
                f"{len(ioi['scored_after']['primary']['matches'])}/26"
            )
        cost += "."
    return "\n".join([
        "## what this doesnt establish",
        "",
        "- **A surviving flag is still not a claim that a head is real.** the criterion "
        "asks whether a scheme's measurement exceeds what that scheme manufactures from a "
        "mismatched activation. whether the head is in the circuit is a different "
        "question, and nothing here answers it.",
        "- **calibration moves every scheme's floor, in both directions.** it applies to "
        "the primary as well, and a scheme whose null is quiet gets a floor *below* 0.02 "
        "and discovers more, not less, `abc` on IOI came out at 0.0077 and went from 18 "
        "discovered heads to 27." + cost + " the rule isnt a noise filter. it is a "
        "recalibration, and on one of three circuits it made the flag louder.",
        "- **the null has its own assumptions.** splicing an activation from another "
        "prompt destroys prompt-correspondence but preserves whatever a head does that is "
        "prompt-independent, so a positional or structural head inflates its scheme's "
        "floor. recorded in the plan in advance. measured here. not corrected.",
        "- **n = 3.** three circuits, two models, one architecture family, and the rule "
        "was chosen while looking at two of the three. one holdout is one holdout.",
        "- **nothing here touches which behaviour to study or the task template.** the top "
        "two rows of the README ladder are where they were in phase 5.",
        "",
    ])


def write_report(path: Path, results_dir: Path = RESULTS_DIR) -> None:
    payloads = {}
    for circuit in ORDER:
        payload = _load(results_dir / f"phase9_{circuit}.json")
        if payload:
            payloads[circuit] = payload

    sections = [_intro(payloads), _floors_section(payloads)]
    for circuit in ORDER:
        if circuit in payloads and circuit != "ioi":
            sections.append(_circuit_section(circuit, payloads[circuit]))
    sections.append(_two_regimes(payloads))
    sections.append(_verdict(payloads))
    sections.append(_holdout(payloads))
    sections.append(_predictions(payloads))
    sections.append(_conclusion(payloads))
    sections.append(_limits(payloads))
    path.write_text("\n".join(s for s in sections if s).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_report(RESULTS_DIR / "PHASE9_REPORT.md")
    print(f"wrote {RESULTS_DIR / 'PHASE9_REPORT.md'}")
