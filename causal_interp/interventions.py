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

Phase 2 adds *path* patching. Activation patching asks what a node does to the
output through every route at once; path patching asks what it does to one
specific downstream node, with every other route held shut. That distinction is
the whole reason Phase 1 could not see components whose contribution reaches the
logits only by way of another head.

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

# Activation kinds stored per attention head, shaped (batch, pos, head, d_head).
# A patch on one of these needs a head index; anything else is whole-layer.
HEAD_KINDS = ("z", "q", "k", "v")


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


def cache_for(
    model: HookedTransformer, tokens: Tensor, kinds: Sequence[str]
) -> tuple[ActivationCache, Tensor]:
    """Cache the requested activation kinds for one run.

    Only the requested kinds are kept — a full cache of every hook is several
    times larger for no benefit.
    """
    wanted = {get_act_name(kind, layer) for kind in kinds for layer in range(model.cfg.n_layers)}
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    return cache, logits


def clean_cache_for(
    model: HookedTransformer, ds: IOIDataset, kinds: Sequence[str] = ("z", "resid_pre", "attn_out", "mlp_out")
) -> tuple[ActivationCache, Tensor]:
    """Cache the clean-run activations that patching draws from."""
    return cache_for(model, ds.clean_tokens, kinds)


def corrupted_cache_for(
    model: HookedTransformer, ds: IOIDataset, kinds: Sequence[str] = ("z", "mlp_out")
) -> tuple[ActivationCache, Tensor]:
    """Cache the corrupted-run activations that path patching freezes nodes to."""
    return cache_for(model, ds.corrupted_tokens, kinds)


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


# ---------------------------------------------------------------------------
# Path patching
# ---------------------------------------------------------------------------

RECEIVER_INPUTS = ("q", "k", "v")

# Sentinel receiver meaning "the output logits themselves". A sender's effect on
# this receiver is its *direct* effect: what it contributes to the prediction
# without any other attention head relaying it.
LOGITS = "logits"


@dataclass(frozen=True)
class Receiver:
    """A head input that a path terminates at — head `layer.head`'s q, k or v."""

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
    """A permutation of 0..n-1 with no fixed point, for the shuffled-source null.

    Every prompt must be paired with a *different* prompt, otherwise part of the
    null would be the real measurement and the noise floor it calibrates would be
    contaminated by genuine signal.
    """
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    # Fixed points are rare but not impossible; rotate each one into its neighbour.
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
    """Hooks for the third pass: every head pinned to corrupted, the sender to clean.

    Pinning all other heads is what makes the measurement a *path* measurement.
    Without it, the sender's change would propagate through downstream heads and
    we would be back to measuring total effect.

    MLPs are recomputed rather than pinned, following the paper: it treats
    attention heads as the nodes of the circuit and lets MLPs carry a path
    between them. `freeze_mlps=True` blocks that route too, which is a stricter
    notion of "direct" and is available so the choice can be measured instead of
    assumed.

    `source_permutation` draws the sender's clean value from a *different* prompt
    in the batch instead of the matching one. Everything else about the run is
    unchanged, so the result is what this path measures when the value it carries
    is real but belongs to the wrong prompt: a null, used to calibrate how much
    apparent signal the method produces from nothing.
    """
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
    """Normalized recovery carried by the direct path from `sender` to `receivers`.

    Implements the scheme from Wang et al. (2022), in the denoising direction used
    throughout this project (base run corrupted, clean values spliced in):

    1. Cache the clean run and the corrupted run (done once by the caller).
    2. Run on the corrupted prompts with every attention head pinned to its
       corrupted value except the sender, which is set to its clean value, and
       record what the receivers now see. Because everything else is pinned, the
       only thing that can have changed at a receiver is what reached it straight
       from the sender.
    3. Run the corrupted prompts again, unpinned, with the receivers set to the
       values recorded in step 2, and measure the logit difference.

    With `receivers=LOGITS` there is no step 3: pinning every other head already
    isolates the sender's direct contribution to the prediction.

    A sender at or above a receiver's layer cannot reach it, and returns 0.0
    without running anything.
    """
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

    # Step 2: record what the receivers see while every other route is shut.
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

    # Step 3: replay the corrupted run with only those receiver inputs replaced.
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
    """How much of the receiver's clean-vs-corrupted difference this path delivers.

    `path_patch` scores a path by what it does to the output logits, which asks a
    lot of an early link in a long chain: even a path that carries its signal
    perfectly may not move the prediction while every later stage is still
    running on corrupted input. This measures the path at its own endpoint
    instead.

    The receiver's activation is projected onto the corrupted -> clean direction:

        0.0   the path delivered nothing; the receiver sees its corrupted input
        1.0   the path delivered the entire difference on its own

    A path can score high here and near zero on `path_patch`. That combination is
    informative rather than contradictory — it says the connection exists but the
    logit-difference metric cannot see it — and distinguishing the two is the
    whole point of measuring both.

    Requires both caches to contain the receivers' q/k/v hooks.
    """
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
        # Clean and corrupted are identical at the receiver: the question is
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
    """`path_signal` for every head into a fixed receiver set.

    The receiver-side counterpart of `sweep_path_senders`. Same sender eligibility
    rule: senders at or above the earliest receiver's layer are left NaN, because
    their score would be an effect on a smaller receiver set than everyone else's.

    Pass `source_permutation` to sweep the shuffled-source null instead.
    """
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
    """Path-patch every head into a fixed set of receivers.

    Returns an (n_layers, n_heads) tensor of normalized recoveries.

    The sender is swept exhaustively while the receivers are fixed. That asymmetry
    is deliberate: fixing the receivers is what makes the question specific, and
    sweeping the senders is what keeps the answer a discovery rather than a
    confirmation of the heads we already expected.

    `sender_position` must equal the receivers' position. A head's q, k and v at
    token position p are computed from the residual stream at p alone, so only
    writes at p can reach them — a sender at any other position has no path here.

    Senders at or above the *earliest* receiver's layer are left as NaN rather than
    measured. Such a sender can reach some receivers and not others, so its score
    would be an effect on a smaller receiver set than everyone else's and would not
    be comparable with the rest of the sweep. NaN marks "not a well-posed question
    here", which a zero would have silently misreported as "no effect".
    """
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
