"""causal scrubbing, Chan et al. 2022. resample-ablate everything outside a hypothesis and
run on clean prompts. no counterfactual pair anywhere."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.utilities import get_act_name

from causal_interp.ioi import IOIDataset

Head = tuple[int, int]

SOURCE_SEED_BASE = 1000


class ResampleSources:
    """the alternative prompts a scrub draws its replacement activations from.

    row r of every IOIDataset gets template r % 8, so a source at a different seed shares
    the clean row's template, length and position map and differs only in names, place and
    object. splicing by absolute token index is only legal because of that.
    """

    def __init__(
        self,
        model: HookedTransformer,
        clean: IOIDataset,
        n_sources: int = 8,
        seed_base: int = SOURCE_SEED_BASE,
        scrub_mlps: bool = False,
        shared: bool = False,
    ) -> None:
        self.n_sources = n_sources
        self.model = model
        self.scrub_mlps = scrub_mlps
        self.shared = shared
        self.datasets = [
            IOIDataset(model, n=len(clean), corruption=clean.corruption, seed=seed_base + s)
            for s in range(n_sources)
        ]

        for s, ds in enumerate(self.datasets):
            if not torch.equal(ds.lengths, clean.lengths):
                raise AssertionError(f"source {s} is not length-aligned with the clean set")
            for name, idx in clean.positions.items():
                if not torch.equal(ds.positions[name], idx):
                    raise AssertionError(f"source {s} misaligns position {name!r}")

        wanted = {get_act_name("z", l) for l in range(model.cfg.n_layers)}
        if scrub_mlps:
            wanted |= {get_act_name("mlp_out", l) for l in range(model.cfg.n_layers)}

        stacks: dict[str, list[Tensor]] = {name: [] for name in wanted}
        with torch.no_grad():
            for ds in self.datasets:
                _, cache = model.run_with_cache(ds.clean_tokens, names_filter=lambda n: n in wanted)
                for name in wanted:
                    stacks[name].append(cache[name].clone())
        self.acts = {name: torch.stack(v) for name, v in stacks.items()}

    def pick(self, name: str, choice: Tensor) -> Tensor:
        src = self.acts[name]
        if choice.ndim == 1:
            return src[choice, torch.arange(choice.shape[0], device=choice.device)]
        n_heads, b = choice.shape
        rows = torch.arange(b, device=choice.device)[None, :].expand(n_heads, b)
        heads = torch.arange(n_heads, device=choice.device)[:, None].expand(n_heads, b)
        picked = src[choice, rows, :, heads, :]
        return picked.permute(1, 2, 0, 3)


def keep_mask(
    model: HookedTransformer,
    ds: IOIDataset,
    heads: Iterable[Head],
    positions: dict[Head, str] | None = None,
) -> Tensor:
    """(n_layers, b, seq, n_heads) bool, True where the activation survives untouched."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    b, seq = ds.clean_tokens.shape
    device = ds.clean_tokens.device
    mask = torch.zeros(n_layers, b, seq, n_heads, dtype=torch.bool, device=device)
    rows = torch.arange(b, device=device)

    for layer, head in heads:
        if positions is None:
            mask[layer, :, :, head] = True
        else:
            pos = ds.positions[positions[(layer, head)]]
            mask[layer, rows, pos, head] = True
    return mask


@dataclass(frozen=True)
class ScrubResult:
    logit_diff: float
    kl: float
    top1_is_io: float


def run_scrub(
    model: HookedTransformer,
    ds: IOIDataset,
    sources: ResampleSources,
    mask: Tensor,
    generator: torch.Generator,
) -> Tensor:
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    device = ds.clean_tokens.device
    b = len(ds)

    if sources.shared:
        row = torch.randint(sources.n_sources, (b,), generator=generator)
        choice = row[None, None, :].expand(n_layers, n_heads, b).to(device)
    else:
        choice = torch.randint(
            sources.n_sources, (n_layers, n_heads, b), generator=generator, device="cpu"
        ).to(device)

    def make_z_hook(layer: int) -> Callable:
        def hook(activation: Tensor, hook) -> Tensor:  # noqa: ANN001
            picked = sources.pick(hook.name, choice[layer])
            return torch.where(mask[layer][..., None], activation, picked)

        return hook

    hooks = [(get_act_name("z", l), make_z_hook(l)) for l in range(n_layers)]

    if sources.scrub_mlps:
        mlp_choice = choice[:, 0, :].contiguous()

        def make_mlp_hook(layer: int) -> Callable:
            def hook(activation: Tensor, hook) -> Tensor:  # noqa: ANN001
                return sources.pick(hook.name, mlp_choice[layer])

            return hook

        hooks += [(get_act_name("mlp_out", l), make_mlp_hook(l)) for l in range(n_layers)]

    with torch.no_grad():
        return model.run_with_hooks(ds.clean_tokens, fwd_hooks=hooks)


def measure(ds: IOIDataset, clean_log_probs: Tensor, logits: Tensor) -> ScrubResult:
    rows = torch.arange(len(ds), device=logits.device)
    final = logits[rows, ds.positions["END"]]
    log_q = final.log_softmax(dim=-1)
    kl = (clean_log_probs.exp() * (clean_log_probs - log_q)).sum(dim=-1).mean().item()
    return ScrubResult(
        logit_diff=ds.logit_diff(logits).item(),
        kl=kl,
        top1_is_io=(final.argmax(dim=-1) == ds.io_token_ids).float().mean().item(),
    )


class Scrubber:
    def __init__(
        self,
        model: HookedTransformer,
        ds: IOIDataset,
        sources: ResampleSources,
        draws: int = 20,
        seed: int = 0,
    ) -> None:
        self.model, self.ds, self.sources = model, ds, sources
        self.draws, self.seed = draws, seed

        with torch.no_grad():
            clean_logits = model(ds.clean_tokens)
        rows = torch.arange(len(ds), device=clean_logits.device)
        self.clean_logits = clean_logits
        self.clean_log_probs = clean_logits[rows, ds.positions["END"]].log_softmax(dim=-1)
        self.clean = measure(ds, self.clean_log_probs, clean_logits)

        self.floor_draws = self._draws(keep_mask(model, ds, []), tag="floor")
        self.floor = ScrubResult(
            logit_diff=_mean(self.floor_draws, "logit_diff"),
            kl=_mean(self.floor_draws, "kl"),
            top1_is_io=_mean(self.floor_draws, "top1_is_io"),
        )
        if self.floor.logit_diff >= self.clean.logit_diff:
            raise AssertionError(
                "scrubbing every head left the logit difference intact, the anchor is void"
            )

    def _draws(self, mask: Tensor, tag: str, draws: int | None = None) -> list[ScrubResult]:
        n = self.draws if draws is None else draws
        gen = torch.Generator().manual_seed(self.seed + zlib.crc32(tag.encode()))
        out = []
        for _ in range(n):
            logits = run_scrub(self.model, self.ds, self.sources, mask, gen)
            out.append(measure(self.ds, self.clean_log_probs, logits))
        return out

    def recovered(self, result: ScrubResult) -> dict[str, float]:
        top1_span = self.clean.top1_is_io - self.floor.top1_is_io
        if abs(top1_span) < 1e-9:
            top1_span = float("nan")
        return {
            "logit_diff": (result.logit_diff - self.floor.logit_diff)
            / (self.clean.logit_diff - self.floor.logit_diff),
            "kl": 1.0 - result.kl / self.floor.kl,
            "top1_is_io": (result.top1_is_io - self.floor.top1_is_io) / top1_span,
        }

    def score(
        self,
        heads: Iterable[Head],
        tag: str,
        positions: dict[Head, str] | None = None,
        draws: int | None = None,
    ) -> dict:
        heads = list(heads)
        results = self._draws(keep_mask(self.model, self.ds, heads, positions), tag, draws)
        per_draw = [self.recovered(r) for r in results]
        out = {"tag": tag, "size": len(heads), "n_draws": len(results)}
        for metric in ("logit_diff", "kl", "top1_is_io"):
            out[metric] = _stats([d[metric] for d in per_draw])
            out[f"{metric}_raw"] = _stats([getattr(r, metric) for r in results])
        out["per_draw_logit_diff"] = [d["logit_diff"] for d in per_draw]
        return out


def all_heads(model: HookedTransformer) -> list[Head]:
    return [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]


def _mean(results: Sequence[ScrubResult], field: str) -> float:
    return sum(getattr(r, field) for r in results) / len(results)


def _stats(values: Sequence[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "min": min(values), "max": max(values)}
