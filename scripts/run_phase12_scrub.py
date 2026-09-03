"""phase 12, causal scrubbing on IOI. registered in results/PHASE12_PLAN.md."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_interp import scrubbing as sc
from causal_interp.ground_truth import ALL_HEADS, CLASS_EXPECTED_POSITION, HEAD_TO_CLASS
from causal_interp.ioi import IOIDataset
from causal_interp.model import load

RESULTS = Path(__file__).resolve().parents[1] / "results"
PAYLOAD = RESULTS / "phase12_scrub.json"

N_PROMPTS = 128
SEED = 0
N_SOURCES = 8
DRAWS = 20
NULL_DRAWS = 5
NULL_SETS = 200

DROP_K = (1, 2, 3, 4, 6, 8, 12, 18)
ADD_K = (2, 4, 8, 16, 32)
FAMILY_SETS = 20
RANDOM_SIZES = (3, 6, 8, 13, 18, 20, 23, 26, 28, 34, 44, 60, 88, 118)


def stage(label: str, since: float) -> float:
    print(f"[{time.time() - since:7.1f}s] {label}", flush=True)
    return time.time()


def parse(name: str) -> sc.Head:
    layer, head = name.split(".")
    return int(layer), int(head)


def phase1_sets() -> dict[str, list[sc.Head]]:
    p1 = json.loads((RESULTS / "phase1_results.json").read_text())
    out: dict[str, set[str]] = {}
    for scheme in ("s2_swap", "abc"):
        head = p1["schemes"][scheme]["headline"]
        out[f"phase1_{scheme}"] = set(head["matches"]) | set(head["extras"])
    out["phase1_union"] = out["phase1_s2_swap"] | out["phase1_abc"]
    out["phase1_agree"] = out["phase1_s2_swap"] & out["phase1_abc"]
    out["phase1_matches"] = set(p1["schemes"]["s2_swap"]["headline"]["matches"]) | set(
        p1["schemes"]["abc"]["headline"]["matches"]
    )
    out["phase1_extras"] = out["phase1_union"] - out["phase1_matches"]
    for scheme, key in (("s2_swap", "greedy_s2_swap"), ("abc", "greedy_abc")):
        out[key] = {step["node"].split("@")[0] for step in p1["schemes"][scheme]["greedy"]}
    return {k: sorted(parse(h) for h in v) for k, v in out.items()}


def layer_counts(heads) -> dict[int, int]:
    counts: dict[int, int] = {}
    for layer, _ in heads:
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def layer_matched(rng: random.Random, counts: dict[int, int], n_heads: int) -> list[sc.Head]:
    out = []
    for layer, k in counts.items():
        out += [(layer, h) for h in rng.sample(range(n_heads), k)]
    return out


def main() -> None:
    if "ground_truth" in Path(sc.__file__).read_text():
        raise SystemExit("scrubbing.py must not import ground_truth")

    started = time.time()
    model = load("gpt2-small")
    ds = IOIDataset(model, n=N_PROMPTS, seed=SEED)
    sources = sc.ResampleSources(model, ds, n_sources=N_SOURCES)
    scrubber = sc.Scrubber(model, ds, sources, draws=DRAWS, seed=SEED)

    pool = sc.all_heads(model)
    published = sorted(ALL_HEADS)
    payload: dict = {
        "meta": {
            "model": "gpt2-small",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "n_prompts": N_PROMPTS,
            "seed": SEED,
            "n_sources": N_SOURCES,
            "draws": DRAWS,
            "null_draws": NULL_DRAWS,
            "null_sets": NULL_SETS,
        },
        "anchors": {
            "clean": vars(scrubber.clean),
            "floor": vars(scrubber.floor),
            "floor_draws": [vars(r) for r in scrubber.floor_draws],
        },
    }

    all_kept = scrubber.score(pool, "gate_all")
    identical = torch.equal(
        sc.run_scrub(model, ds, sources, sc.keep_mask(model, ds, pool),
                     torch.Generator().manual_seed(1)),
        scrubber.clean_logits,
    )
    payload["gates"] = {
        "all_heads_recovered": all_kept["logit_diff"]["mean"],
        "all_heads_logits_identical": bool(identical),
        "empty_recovered": scrubber.score([], "gate_empty")["logit_diff"]["mean"],
        "empty_raw_logit_diff": scrubber.floor.logit_diff,
        "clean_raw_logit_diff": scrubber.clean.logit_diff,
    }
    print("gates", json.dumps(payload["gates"]), flush=True)
    mark = stage("gates", started)

    candidates: dict[str, list[sc.Head]] = {"published_26": published, **phase1_sets()}

    payload["candidates"] = {}
    for name, heads in candidates.items():
        payload["candidates"][name] = scrubber.score(heads, name)
        payload["candidates"][name]["published_overlap"] = len(set(heads) & ALL_HEADS)
        print(name, round(payload["candidates"][name]["logit_diff"]["mean"], 4), flush=True)
    mark = stage("candidates", mark)

    rng = random.Random(4242)
    unpublished = [h for h in pool if h not in ALL_HEADS]
    payload["families"] = {"drop": {}, "add": {}}
    for k in DROP_K:
        rows = []
        for i in range(FAMILY_SETS):
            heads = rng.sample(published, len(published) - k)
            rows.append(scrubber.score(heads, f"drop{k}_{i}"))
        payload["families"]["drop"][str(k)] = rows
        print("drop", k, round(sum(r["logit_diff"]["mean"] for r in rows) / len(rows), 4), flush=True)
    for k in ADD_K:
        rows = []
        for i in range(FAMILY_SETS):
            heads = published + rng.sample(unpublished, k)
            rows.append(scrubber.score(heads, f"add{k}_{i}"))
        payload["families"]["add"][str(k)] = rows
        print("add", k, round(sum(r["logit_diff"]["mean"] for r in rows) / len(rows), 4), flush=True)
    mark = stage("families", mark)

    sizes = sorted(
        set(RANDOM_SIZES)
        | {len(h) for h in candidates.values()}
        | {len(published) - k for k in DROP_K}
        | {len(published) + k for k in ADD_K}
    )
    payload["null"] = {}
    for m in sizes:
        rows = []
        for i in range(NULL_SETS):
            heads = rng.sample(pool, m)
            r = scrubber.score(heads, f"rand{m}_{i}", draws=NULL_DRAWS)
            r["published_overlap"] = len(set(heads) & ALL_HEADS)
            rows.append(r)
        payload["null"][str(m)] = rows
        vals = sorted(r["logit_diff"]["mean"] for r in rows)
        print("null", m, "med", round(vals[len(vals) // 2], 4), "p95",
              round(vals[int(0.95 * len(vals))], 4), flush=True)
    mark = stage("null", mark)

    counts = layer_counts(published)
    rows = []
    for i in range(NULL_SETS):
        heads = layer_matched(rng, counts, model.cfg.n_heads)
        r = scrubber.score(heads, f"lm{i}", draws=NULL_DRAWS)
        r["published_overlap"] = len(set(heads) & ALL_HEADS)
        rows.append(r)
    payload["layer_matched"] = rows
    mark = stage("layer matched", mark)

    payload["loo"] = {}
    for head in published:
        rest = [h for h in published if h != head]
        payload["loo"][f"{head[0]}.{head[1]}"] = scrubber.score(rest, f"loo_{head}")
    payload["aoi"] = {}
    for head in pool:
        payload["aoi"][f"{head[0]}.{head[1]}"] = scrubber.score([head], f"aoi_{head}")
    mark = stage("head level", mark)

    payload["variants"] = {"bypos": {}, "shared": {}, "mlp_scrub": {}}
    for name in ("published_26", "phase1_agree", "phase1_matches"):
        heads = candidates[name]
        labelled = [h for h in heads if h in HEAD_TO_CLASS]
        positions = {h: CLASS_EXPECTED_POSITION[HEAD_TO_CLASS[h]] for h in labelled}
        payload["variants"]["bypos"][name] = scrubber.score(
            labelled, f"bypos_{name}", positions=positions
        )
        payload["variants"]["bypos"][name]["n_labelled"] = len(labelled)

    sources.shared = True
    scrubber_shared = sc.Scrubber(model, ds, sources, draws=DRAWS, seed=SEED)
    for name, heads in candidates.items():
        payload["variants"]["shared"][name] = scrubber_shared.score(heads, f"shared_{name}")
    payload["variants"]["shared"]["_anchors"] = {
        "clean": vars(scrubber_shared.clean),
        "floor": vars(scrubber_shared.floor),
    }
    sources.shared = False
    del scrubber_shared, sources
    torch.cuda.empty_cache()

    mlp_sources = sc.ResampleSources(model, ds, n_sources=N_SOURCES, scrub_mlps=True)
    scrubber_mlp = sc.Scrubber(model, ds, mlp_sources, draws=DRAWS, seed=SEED)
    for name, heads in candidates.items():
        payload["variants"]["mlp_scrub"][name] = scrubber_mlp.score(heads, f"mlp_{name}")
    payload["variants"]["mlp_scrub"]["_anchors"] = {
        "clean": vars(scrubber_mlp.clean),
        "floor": vars(scrubber_mlp.floor),
    }

    mark = stage("variants", mark)
    payload["meta"]["runtime_seconds"] = round(time.time() - started, 1)
    PAYLOAD.write_text(json.dumps(payload, indent=2))
    print("wrote", PAYLOAD, payload["meta"]["runtime_seconds"], "s")


if __name__ == "__main__":
    main()
