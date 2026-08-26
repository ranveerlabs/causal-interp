"""the patching primitives. everything else in here is built on these."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utilities import get_act_name

from causal_interp.ioi import IOIDataset

# position sentinel meaning "patch every token position"
# bound on a node's total effect regardless of where it acts.
ALL_POSITIONS = "ALL"

# Activation kinds stored per attention head
# a patch on one of these needs a head index
HEAD_KINDS = ("z", "q", "k", "v")


@dataclass(frozen=True)
class Patch:
    """one node to splice from the clean run into the corrupted run."""

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
        return (patched_logit_diff - self.corrupted_logit_diff) / self.span


def cache_for(
    model: HookedTransformer, tokens: Tensor, kinds: Sequence[str]
) -> tuple[ActivationCache, Tensor]:
    wanted = {get_act_name(kind, layer) for kind in kinds for layer in range(model.cfg.n_layers)}
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    return cache, logits


def clean_cache_for(
    model: HookedTransformer, ds: IOIDataset, kinds: Sequence[str] = ("z", "resid_pre", "attn_out", "mlp_out")
) -> tuple[ActivationCache, Tensor]:
    return cache_for(model, ds.clean_tokens, kinds)


def corrupted_cache_for(
    model: HookedTransformer, ds: IOIDataset, kinds: Sequence[str] = ("z", "mlp_out")
) -> tuple[ActivationCache, Tensor]:
    return cache_for(model, ds.corrupted_tokens, kinds)


def baseline_for(model: HookedTransformer, ds: IOIDataset) -> tuple[Baseline, Tensor, Tensor]:
    with torch.no_grad():
        clean_logits = model(ds.clean_tokens)
        corrupted_logits = model(ds.corrupted_tokens)
    baseline = Baseline(
        clean_logit_diff=ds.logit_diff(clean_logits).item(),
        corrupted_logit_diff=ds.logit_diff(corrupted_logits).item(),
    )
    return baseline, clean_logits, corrupted_logits


def _make_hook(specs: Sequence[Patch], ds: IOIDataset, cache: ActivationCache) -> Callable:
    """build a forward hook that overwrites the listed nodes with clean activations."""
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
                # per-prompt indices: templates differ in length, so the same
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
        if (patch.kind in HEAD_KINDS) != (patch.head is not None):
            raise ValueError(
                f"'head' must be set for head-shaped kinds {HEAD_KINDS} and unset otherwise: {patch}"
            )
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
    """patch every attention head at every position, one at a time."""
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


def _make_null_hook(
    spec: Patch, ds: IOIDataset, cache: ActivationCache, permutation: Tensor
) -> Callable:
    """like `_make_hook`, but the clean value comes from a *different* prompt."""
    rows = torch.arange(len(ds), device=ds.clean_tokens.device)
    src_rows = permutation.to(rows.device)
    pos = ds.positions[spec.position]
    src_pos = pos[src_rows]

    def hook(activation: Tensor, hook) -> Tensor:  # noqa: ANN001 - TL's hook signature
        clean = cache[hook.name]
        activation[rows, pos, spec.head] = clean[src_rows, src_pos, spec.head]
        return activation

    return hook


def sweep_heads_null(
    model: HookedTransformer,
    ds: IOIDataset,
    cache: ActivationCache,
    baseline: Baseline,
    positions: Sequence[str],
    permutation: Tensor,
    progress: Callable[[int, int], None] | None = None,
) -> Tensor:
    """`sweep_heads` with the spliced clean activation drawn from a deranged prompt order."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    out = torch.zeros(n_layers, n_heads, len(positions))
    total = n_layers * n_heads * len(positions)
    done = 0

    for layer in range(n_layers):
        for head in range(n_heads):
            for p, position in enumerate(positions):
                spec = Patch(layer=layer, kind="z", position=position, head=head)
                hook = _make_null_hook(spec, ds, cache, permutation)
                with torch.no_grad():
                    logits = model.run_with_hooks(
                        ds.corrupted_tokens, fwd_hooks=[(spec.hook_name, hook)]
                    )
                out[layer, head, p] = baseline.normalize(ds.logit_diff(logits).item())
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
    """Patch a whole-layer component (resid_pre / attn_out / mlp_out) per position."""
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
    """iteratively narrow to a small set of nodes that *jointly* restore behaviour."""
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


# path patching

RECEIVER_INPUTS = ("q", "k", "v")

LOGITS = "logits"


@dataclass(frozen=True)
class Receiver:
    """A head input that a path terminates at, head `layer.head`'s q, k or v."""

    layer: int
    head: int
    position: str
    input: str

    def __post_init__(self) -> None:
        if self.input not in RECEIVER_INPUTS:
            raise ValueError(f"input must be one of {RECEIVER_INPUTS}, got {self.input!r}")

    @property
    def hook_name(self) -> str:
        return get_act_name(self.input, self.layer)

    def __str__(self) -> str:
        return f"{self.layer}.{self.head}.{self.input}@{self.position}"


def derangement(n: int, seed: int = 0) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    # fixed points are rare but not impossible
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    return perm


def _freeze_hooks(
    model: HookedTransformer,
    ds: IOIDataset,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    sender: Patch,
    freeze_mlps: bool,
    source_permutation: Tensor | None = None,
) -> list[tuple[str, Callable]]:
    """hooks for the third pass: every head pinned to corrupted, the sender to clean."""
    rows = torch.arange(len(ds), device=ds.clean_tokens.device)
    sender_pos = ds.positions[sender.position]
    if source_permutation is None:
        src_rows, src_pos = rows, sender_pos
    else:
        src_rows = source_permutation.to(rows.device)
        src_pos = sender_pos[src_rows]
    hooks: list[tuple[str, Callable]] = []

    def make_z_hook(layer: int) -> Callable:
        def hook(activation: Tensor, hook) -> Tensor:  # noqa: ANN001
            activation[:] = corrupted_cache[hook.name]
            if layer == sender.layer:
                clean = clean_cache[hook.name]
                activation[rows, sender_pos, sender.head] = clean[src_rows, src_pos, sender.head]
            return activation

        return hook

    for layer in range(model.cfg.n_layers):
        hooks.append((get_act_name("z", layer), make_z_hook(layer)))

    if freeze_mlps:
        def mlp_hook(activation: Tensor, hook) -> Tensor:  # noqa: ANN001
            activation[:] = corrupted_cache[hook.name]
            return activation

        for layer in range(model.cfg.n_layers):
            hooks.append((get_act_name("mlp_out", layer), mlp_hook))

    return hooks


def path_patch(
    model: HookedTransformer,
    ds: IOIDataset,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    baseline: Baseline,
    sender: Patch,
    receivers: Sequence[Receiver] | str = LOGITS,
    freeze_mlps: bool = False,
) -> float:
    """normalized recovery carried by the direct path from `sender` to `receivers`."""
    if receivers != LOGITS:
        if not receivers:
            raise ValueError("receivers must be non-empty, or the LOGITS sentinel")
        if all(sender.layer >= r.layer for r in receivers):
            return 0.0  # nothing downstream to reach

    freeze = _freeze_hooks(model, ds, clean_cache, corrupted_cache, sender, freeze_mlps)

    if receivers == LOGITS:
        with torch.no_grad():
            logits = model.run_with_hooks(ds.corrupted_tokens, fwd_hooks=freeze)
        return baseline.normalize(ds.logit_diff(logits).item())

    # step 2: record what the receivers see while every other route is shut.
    grouped: dict[str, list[Receiver]] = {}
    for receiver in receivers:
        grouped.setdefault(receiver.hook_name, []).append(receiver)

    recorded: dict[str, Tensor] = {}

    def make_save_hook() -> Callable:
        def hook(activation: Tensor, hook) -> Tensor:  # noqa: ANN001
            recorded[hook.name] = activation.detach().clone()
            return activation

        return hook

    with torch.no_grad():
        model.run_with_hooks(
            ds.corrupted_tokens,
            fwd_hooks=freeze + [(name, make_save_hook()) for name in grouped],
        )

    # step 3
    rows = torch.arange(len(ds), device=ds.clean_tokens.device)

    def make_apply_hook(specs: Sequence[Receiver]) -> Callable:
        def hook(activation: Tensor, hook) -> Tensor:  # noqa: ANN001
            saved = recorded[hook.name]
            for spec in specs:
                pos = ds.positions[spec.position]
                activation[rows, pos, spec.head] = saved[rows, pos, spec.head]
            return activation

        return hook

    with torch.no_grad():
        logits = model.run_with_hooks(
            ds.corrupted_tokens,
            fwd_hooks=[(name, make_apply_hook(specs)) for name, specs in grouped.items()],
        )
    return baseline.normalize(ds.logit_diff(logits).item())


def path_signal(
    model: HookedTransformer,
    ds: IOIDataset,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    sender: Patch,
    receivers: Sequence[Receiver],
    freeze_mlps: bool = False,
    source_permutation: Tensor | None = None,
) -> float:
    """How much of the receiver's clean-vs-corrupted difference this path delivers."""
    if not receivers:
        raise ValueError("receivers must be non-empty")
    if all(sender.layer >= r.layer for r in receivers):
        return 0.0

    grouped: dict[str, list[Receiver]] = {}
    for receiver in receivers:
        grouped.setdefault(receiver.hook_name, []).append(receiver)

    recorded: dict[str, Tensor] = {}

    def save_hook(activation: Tensor, hook) -> Tensor:  # noqa: ANN001
        recorded[hook.name] = activation.detach().clone()
        return activation

    freeze = _freeze_hooks(
        model, ds, clean_cache, corrupted_cache, sender, freeze_mlps, source_permutation
    )
    with torch.no_grad():
        model.run_with_hooks(
            ds.corrupted_tokens, fwd_hooks=freeze + [(name, save_hook) for name in grouped]
        )

    rows = torch.arange(len(ds), device=ds.clean_tokens.device)
    numerator = torch.zeros((), device=rows.device)
    denominator = torch.zeros((), device=rows.device)
    for name, specs in grouped.items():
        for spec in specs:
            pos = ds.positions[spec.position]
            patched = recorded[name][rows, pos, spec.head]
            clean = clean_cache[name][rows, pos, spec.head]
            corrupted = corrupted_cache[name][rows, pos, spec.head]
            direction = clean - corrupted
            numerator += ((patched - corrupted) * direction).sum()
            denominator += (direction * direction).sum()

    if denominator.item() == 0.0:
        # clean and corrupted are identical at the receiver: the question is
        # undefined rather than answered zero.
        return float("nan")
    return (numerator / denominator).item()


def sweep_path_signal(
    model: HookedTransformer,
    ds: IOIDataset,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    receivers: Sequence[Receiver],
    sender_position: str,
    freeze_mlps: bool = False,
    source_permutation: Tensor | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Tensor:
    """`path_signal` for every head into a fixed receiver set."""
    out = torch.full((model.cfg.n_layers, model.cfg.n_heads), float("nan"))
    ceiling = min(r.layer for r in receivers)
    total = ceiling * model.cfg.n_heads
    done = 0
    for layer in range(ceiling):
        for head in range(model.cfg.n_heads):
            sender = Patch(layer=layer, kind="z", position=sender_position, head=head)
            out[layer, head] = path_signal(
                model, ds, clean_cache, corrupted_cache, sender, receivers,
                freeze_mlps, source_permutation,
            )
            done += 1
            if progress is not None:
                progress(done, total)
    return out


def sweep_path_senders(
    model: HookedTransformer,
    ds: IOIDataset,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    baseline: Baseline,
    receivers: Sequence[Receiver] | str,
    sender_position: str,
    freeze_mlps: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> Tensor:
    """path-patch every head into a fixed set of receivers."""
    out = torch.full((model.cfg.n_layers, model.cfg.n_heads), float("nan"))
    ceiling = model.cfg.n_layers if receivers == LOGITS else min(r.layer for r in receivers)
    total = ceiling * model.cfg.n_heads
    done = 0
    for layer in range(ceiling):
        for head in range(model.cfg.n_heads):
            sender = Patch(layer=layer, kind="z", position=sender_position, head=head)
            out[layer, head] = path_patch(
                model, ds, clean_cache, corrupted_cache, baseline, sender, receivers, freeze_mlps
            )
            done += 1
            if progress is not None:
                progress(done, total)
    return out
