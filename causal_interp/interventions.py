"""Intervention primitives — the causal core of the system.

Tracing and visualization show what correlates with a behaviour. These primitives
exist to answer the stronger question: if this component is changed, does the
behaviour change? Every claim the system emits must be backed by one of these.

Phase 1 implements activation patching in the *denoising* direction: run the
model on the corrupted prompt, splice in activations cached from the clean run at
one node, and measure how much of the clean behaviour comes back. A node that
restores the behaviour on its own is carrying the causal signal.

The unit of measurement is the normalized recovery of the logit difference:

    0.0   patching this node changed nothing (still behaves corrupted)
    1.0   patching this node alone fully restored clean behaviour

Ablation and iterative pruning — the removal-direction counterparts — belong to a
later phase and are deliberately not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utilities import get_act_name

from causal_interp.ioi import IOIDataset

# Position sentinel meaning "patch every token position", used to get an upper
# bound on a node's total effect regardless of where it acts.
ALL_POSITIONS = "ALL"


@dataclass(frozen=True)
class Patch:
    """One node to splice from the clean run into the corrupted run.

    `kind` is a transformer_lens activation name ("z", "resid_pre", "attn_out",
    "mlp_out"). `head` is required for "z" and must be None otherwise. `position`
    is a semantic position name from `causal_interp.ioi.POSITIONS`, or
    `ALL_POSITIONS`.
    """

    layer: int
    kind: str
    position: str
    head: int | None = None

    @property
    def hook_name(self) -> str:
        return get_act_name(self.kind, self.layer)

    def __str__(self) -> str:
        node = f"{self.layer}.{self.head}" if self.head is not None else f"{self.kind}[{self.layer}]"
        return f"{node}@{self.position}"


@dataclass(frozen=True)
class Baseline:
    """The two reference points every patched run is scored against."""

    clean_logit_diff: float
    corrupted_logit_diff: float

    @property
    def span(self) -> float:
        return self.clean_logit_diff - self.corrupted_logit_diff

    def normalize(self, patched_logit_diff: float) -> float:
        """Map a patched logit difference onto the 0 (corrupted) .. 1 (clean) scale."""
        return (patched_logit_diff - self.corrupted_logit_diff) / self.span


def clean_cache_for(
    model: HookedTransformer, ds: IOIDataset, kinds: Sequence[str] = ("z", "resid_pre", "attn_out", "mlp_out")
) -> tuple[ActivationCache, Tensor]:
    """Cache the clean-run activations that patching will draw from.

    Only the requested activation kinds are kept — a full cache of every hook is
    several times larger for no benefit.
    """
    wanted = {get_act_name(kind, layer) for kind in kinds for layer in range(model.cfg.n_layers)}
    with torch.no_grad():
        logits, cache = model.run_with_cache(ds.clean_tokens, names_filter=lambda n: n in wanted)
    return cache, logits


def baseline_for(model: HookedTransformer, ds: IOIDataset) -> tuple[Baseline, Tensor, Tensor]:
    """Compute the clean and corrupted logit differences that bracket every result."""
    with torch.no_grad():
        clean_logits = model(ds.clean_tokens)
        corrupted_logits = model(ds.corrupted_tokens)
    baseline = Baseline(
        clean_logit_diff=ds.logit_diff(clean_logits).item(),
        corrupted_logit_diff=ds.logit_diff(corrupted_logits).item(),
    )
    return baseline, clean_logits, corrupted_logits


def _make_hook(specs: Sequence[Patch], ds: IOIDataset, cache: ActivationCache) -> Callable:
    """Build a forward hook that overwrites the listed nodes with clean activations."""
    rows = torch.arange(len(ds), device=ds.clean_tokens.device)

    def hook(activation: Tensor, hook) -> Tensor:  # noqa: ANN001 - TL's hook signature
        clean = cache[hook.name]
        for spec in specs:
            if spec.position == ALL_POSITIONS:
                if spec.head is None:
                    activation[...] = clean
                else:
                    activation[:, :, spec.head] = clean[:, :, spec.head]
            else:
                # Per-prompt indices: templates differ in length, so the same
                # semantic position sits at a different column in each row.
                pos = ds.positions[spec.position]
                if spec.head is None:
                    activation[rows, pos] = clean[rows, pos]
                else:
                    activation[rows, pos, spec.head] = clean[rows, pos, spec.head]
        return activation

    return hook


def run_patched(
    model: HookedTransformer, ds: IOIDataset, cache: ActivationCache, patches: Iterable[Patch]
) -> Tensor:
    """Run the corrupted prompts with `patches` spliced in, returning logits."""
    grouped: dict[str, list[Patch]] = {}
    for patch in patches:
        if (patch.kind == "z") != (patch.head is not None):
            raise ValueError(f"'head' must be set for kind='z' and unset otherwise: {patch}")
        grouped.setdefault(patch.hook_name, []).append(patch)

    fwd_hooks = [(name, _make_hook(specs, ds, cache)) for name, specs in grouped.items()]
    with torch.no_grad():
        return model.run_with_hooks(ds.corrupted_tokens, fwd_hooks=fwd_hooks)


def patch_effect(
    model: HookedTransformer,
    ds: IOIDataset,
    cache: ActivationCache,
    patches: Iterable[Patch],
    baseline: Baseline,
) -> float:
    """Normalized recovery from patching `patches` together (0 = corrupted, 1 = clean)."""
    logits = run_patched(model, ds, cache, patches)
    return baseline.normalize(ds.logit_diff(logits).item())


def sweep_heads(
    model: HookedTransformer,
    ds: IOIDataset,
    cache: ActivationCache,
    baseline: Baseline,
    positions: Sequence[str],
    progress: Callable[[int, int], None] | None = None,
) -> Tensor:
    """Patch every attention head at every position, one at a time.

    Returns a (n_layers, n_heads, n_positions) tensor of normalized recoveries.
    This is the marginal effect of each head in isolation; heads that only matter
    in combination will not show up here, which is a real limitation of the
    method rather than of this implementation.
    """
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    out = torch.zeros(n_layers, n_heads, len(positions))
    total = n_layers * n_heads * len(positions)
    done = 0

    for layer in range(n_layers):
        for head in range(n_heads):
            for p, position in enumerate(positions):
                patch = Patch(layer=layer, kind="z", position=position, head=head)
                out[layer, head, p] = patch_effect(model, ds, cache, [patch], baseline)
                done += 1
                if progress is not None:
                    progress(done, total)
    return out


def sweep_component(
    model: HookedTransformer,
    ds: IOIDataset,
    cache: ActivationCache,
    baseline: Baseline,
    kind: str,
    positions: Sequence[str],
) -> Tensor:
    """Patch a whole-layer component (resid_pre / attn_out / mlp_out) per position.

    Returns a (n_layers, n_positions) tensor of normalized recoveries. Coarser than
    the head sweep, but it localizes *where in depth* the signal appears before
    attributing it to individual heads.
    """
    out = torch.zeros(model.cfg.n_layers, len(positions))
    for layer in range(model.cfg.n_layers):
        for p, position in enumerate(positions):
            patch = Patch(layer=layer, kind=kind, position=position)
            out[layer, p] = patch_effect(model, ds, cache, [patch], baseline)
    return out


def greedy_select(
    model: HookedTransformer,
    ds: IOIDataset,
    cache: ActivationCache,
    baseline: Baseline,
    candidates: Sequence[Patch],
    max_size: int,
    min_gain: float = 0.005,
) -> list[tuple[Patch, float]]:
    """Iteratively narrow to a small set of nodes that *jointly* restore behaviour.

    At each step, add whichever remaining candidate brings the patched run
    *closest to clean behaviour* — that is, minimizes |1 - recovery| — and stop
    when the best available addition closes less than `min_gain` of that gap.

    The objective is deliberately closeness to 1.0 rather than maximum recovery.
    Patching name movers can drive the logit difference well past its clean value,
    and a set chosen by maximization would keep adding heads to overshoot further;
    that would score well while describing something other than the behaviour
    being explained.

    This is the step that turns a ranking into a circuit claim: a head with a
    large marginal effect can still be redundant once another head is already
    patched, and only joint patching exposes that.

    Returns [(patch, cumulative recovery after adding it), ...].
    """
    remaining = list(candidates)
    chosen: list[Patch] = []
    trace: list[tuple[Patch, float]] = []
    current = 0.0
    distance = abs(1.0 - current)

    for _ in range(max_size):
        best_patch, best_distance, best_score = None, distance, current
        for patch in remaining:
            score = patch_effect(model, ds, cache, chosen + [patch], baseline)
            if abs(1.0 - score) < best_distance:
                best_patch, best_distance, best_score = patch, abs(1.0 - score), score

        if best_patch is None or distance - best_distance < min_gain:
            break

        chosen.append(best_patch)
        remaining.remove(best_patch)
        trace.append((best_patch, best_score))
        current, distance = best_score, best_distance

    return trace
