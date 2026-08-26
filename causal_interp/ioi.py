"""The IOI task: clean/corrupted prompt pairs and the logit-difference metric."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from causal_interp.corruption import random_vocab_corruption
from causal_interp.schemes import Scheme, TaskSpec

TEMPLATES: tuple[str, ...] = (
    "Then, {a} and {b} went to the {place}. {s} gave a {obj} to",
    "Then, {a} and {b} had a lot of fun at the {place}. {s} gave a {obj} to",
    "Then, {a} and {b} were working at the {place}. {s} decided to give a {obj} to",
    "After {a} and {b} went to the {place}, {s} gave a {obj} to",
    "While {a} and {b} were working at the {place}, {s} gave a {obj} to",
    "While {a} and {b} were commuting to the {place}, {s} gave a {obj} to",
    "When {a} and {b} got a {obj} at the {place}, {s} decided to give it to",
    "Afterwards, {a} and {b} went to the {place}. {s} gave a {obj} to",
)

# Filtered to single-token-with-leading-space under the GPT-2 tokenizer
# constructor re-verifies rather than trusting this list.
NAMES: tuple[str, ...] = (
    "John", "Mary", "Tom", "James", "Dan", "Sid", "Martin", "Amy", "Sarah", "Alex",
    "Michael", "Jessica", "David", "Robert", "Anna", "Kevin", "Laura", "Chris",
    "Emily", "Jack", "Paul", "Peter", "Rachel", "Susan", "Brian", "Steven",
    "Karen", "Nancy", "Linda", "Frank", "Henry", "Kate", "Lisa", "Mark",
    "Patrick", "Sean", "Julia", "Grace", "Eric", "Adam", "Ryan", "Jason",
    "Carl", "Bob", "Alice", "Carol", "Dave", "Sam", "Rose", "Victoria",
    "Charlotte", "Oliver", "Harry", "George", "Emma", "Sophie", "Andrew",
    "Matthew", "Daniel", "Joseph", "Charles", "Thomas", "Anthony", "Edward",
    "Timothy", "Larry", "Jeffrey", "Scott", "Stephen", "Gregory", "Joshua",
    "Jerry", "Dennis", "Walter", "Patricia", "Barbara", "Elizabeth", "Margaret",
    "Dorothy", "Helen", "Betty", "Ruth", "Sharon", "Michelle", "Deborah",
    "Angela", "Melissa", "Brenda", "Amanda", "Stephanie", "Nicole", "Katherine",
    "Christine", "Samantha", "Rebecca", "Kathleen", "Janet",
)

PLACES: tuple[str, ...] = (
    "store", "garden", "school", "hospital", "station", "market", "beach", "airport",
)

OBJECTS: tuple[str, ...] = (
    "drink", "kiss", "snack", "basketball", "bone", "computer", "necklace", "ring",
)

POSITIONS: tuple[str, ...] = ("IO", "IO+1", "S1", "S1+1", "S2", "S2+1", "END")

CORRUPTIONS: tuple[str, ...] = ("s2_swap", "abc", "random_vocab_s2", "random_vocab_any")

GENERIC_CORRUPTIONS: tuple[str, ...] = ("random_vocab_s2", "random_vocab_any")

SCHEMES: dict[str, Scheme] = {
    "s2_swap": Scheme(
        name="s2_swap",
        provenance="published",
        breaks="replaces the repeated subject at S2 with the indirect object's name, reversing the answer",
        preserves_answer=False,
        primary=True,
    ),
    "abc": Scheme(
        name="abc",
        provenance="published",
        breaks="replaces both subject mentions with a third name, so no name is repeated",
        preserves_answer=True,
    ),
    "random_vocab_s2": Scheme(
        name="random_vocab_s2",
        provenance="generic",
        breaks="substitutes a uniformly drawn vocabulary token at the S2 anchor",
        preserves_answer=False,
    ),
    "random_vocab_any": Scheme(
        name="random_vocab_any",
        provenance="generic",
        breaks="substitutes a uniformly drawn vocabulary token anywhere in the prompt",
        preserves_answer=False,
    ),
}

DISCOVERY_SCHEMES: tuple[str, ...] = CORRUPTIONS


@dataclass(frozen=True)
class IOIPrompt:
    """one clean/corrupted pair and the names that define its answer."""

    clean: str
    corrupted: str
    io_name: str
    s_name: str
    template: str
    order: str  # "ABB" (subject first) or "BAB" (indirect object first)


class IOIDataset:
    """a batch of IOI pairs, tokenized, with semantic position indices."""

    def __init__(
        self,
        model: HookedTransformer,
        n: int = 128,
        corruption: str = "s2_swap",
        seed: int = 0,
        templates: tuple[str, ...] | None = None,
        orders: tuple[str, ...] = ("ABB", "BAB"),
    ) -> None:
        if corruption not in CORRUPTIONS:
            raise ValueError(f"corruption must be one of {CORRUPTIONS}, got {corruption!r}")

        self.corruption = corruption
        self.seed = seed
        self.model = model

        self.templates = templates or TEMPLATES

        if not set(orders) <= {"ABB", "BAB"}:
            raise ValueError(f"orders must be drawn from ABB/BAB, got {orders}")
        self.orders = orders

        names = self._single_token_names(model)

        rng = random.Random(seed)
        corrupt_rng = random.Random(seed + 10_000)
        self.prompts = [self._make_prompt(rng, corrupt_rng, names, i) for i in range(n)]

        device = model.cfg.device
        self.clean_tokens, clean_lengths = _tokenize([p.clean for p in self.prompts], model)
        self.corrupted_tokens, corrupted_lengths = _tokenize([p.corrupted for p in self.prompts], model)

        # patching by index is only meaningful if the pair is aligned token for
        # token
        if not torch.equal(clean_lengths, corrupted_lengths):
            bad = int((clean_lengths != corrupted_lengths).nonzero()[0])
            raise AssertionError(
                f"clean/corrupted length mismatch at prompt {bad}: "
                f"{self.prompts[bad].clean!r} vs {self.prompts[bad].corrupted!r}"
            )
        self.lengths = clean_lengths

        self.io_token_ids = torch.tensor(
            [_name_token_id(model, p.io_name) for p in self.prompts], device=device
        )
        self.s_token_ids = torch.tensor(
            [_name_token_id(model, p.s_name) for p in self.prompts], device=device
        )
        self.positions = self._locate_positions()

        if self.corruption in GENERIC_CORRUPTIONS:
            self.corrupted_tokens, self.corrupted_indices = self._apply_generic_corruption(seed)

    def __len__(self) -> int:
        return len(self.prompts)

    # -- construction -------------------------------------------------------

    @staticmethod
    def _single_token_names(model: HookedTransformer) -> list[str]:
        keep = [
            name for name in NAMES
            if len(model.tokenizer.encode(" " + name, add_special_tokens=False)) == 1
        ]
        if len(keep) < 20:
            raise RuntimeError(f"only {len(keep)} single-token names survived filtering")
        return keep

    def _make_prompt(
        self, rng: random.Random, corrupt_rng: random.Random, names: list[str], i: int
    ) -> IOIPrompt:
        template = self.templates[i % len(self.templates)]
        order = self.orders[i % len(self.orders)]

        io_name, s_name = rng.sample(names, 2)
        a, b = (s_name, io_name) if order == "ABB" else (io_name, s_name)
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)

        clean = template.format(a=a, b=b, s=s_name, place=place, obj=obj)

        if self.corruption == "s2_swap":

            corrupted = template.format(a=a, b=b, s=io_name, place=place, obj=obj)
        elif self.corruption == "abc":  # three fresh distinct names: no name repeats
            pool = [x for x in names if x not in (io_name, s_name)]
            a2, b2, s2 = corrupt_rng.sample(pool, 3)
            corrupted = template.format(a=a2, b=b2, s=s2, place=place, obj=obj)
        else:

            corrupted = clean

        return IOIPrompt(clean, corrupted, io_name, s_name, template, order)

    def _apply_generic_corruption(self, seed: int) -> tuple[Tensor, Tensor]:
        anchor = self.positions["S2"] if self.corruption == "random_vocab_s2" else None
        return random_vocab_corruption(
            clean_tokens=self.clean_tokens,
            lengths=self.lengths,
            d_vocab=self.model.cfg.d_vocab,
            seed=seed,
            anchor=anchor,
        )

    def _locate_positions(self) -> dict[str, Tensor]:
        """find IO/S1/S2/END indices by searching the clean tokens for the name ids."""
        device = self.clean_tokens.device
        found: dict[str, list[int]] = {name: [] for name in POSITIONS}

        for i, prompt in enumerate(self.prompts):
            row = self.clean_tokens[i, : self.lengths[i]]
            io_hits = (row == self.io_token_ids[i]).nonzero().flatten().tolist()
            s_hits = (row == self.s_token_ids[i]).nonzero().flatten().tolist()

            # a template that repeated a name elsewhere would silently corrupt every
            # position index downstream, so this is a hard failure, not a warning.
            if len(io_hits) != 1 or len(s_hits) != 2:
                raise AssertionError(
                    f"prompt {i} has {len(io_hits)} IO and {len(s_hits)} S occurrences "
                    f"(expected 1 and 2): {prompt.clean!r}"
                )

            end = int(self.lengths[i]) - 1
            idx = {
                "IO": io_hits[0],
                "IO+1": io_hits[0] + 1,
                "S1": s_hits[0],
                "S1+1": s_hits[0] + 1,
                "S2": s_hits[1],
                "S2+1": s_hits[1] + 1,
                "END": end,
            }
            for name, value in idx.items():
                if value > end:
                    raise AssertionError(f"position {name}={value} past END={end} in {prompt.clean!r}")
                found[name].append(value)

        return {name: torch.tensor(v, device=device) for name, v in found.items()}

    # -- metric -------------------------------------------------------------

    def logit_diff(self, logits: Tensor, per_prompt: bool = False) -> Tensor:
        end = self.positions["END"]
        rows = torch.arange(len(self), device=logits.device)
        final = logits[rows, end]  # (batch, d_vocab)
        diff = final.gather(1, self.io_token_ids[:, None]) - final.gather(1, self.s_token_ids[:, None])
        diff = diff.squeeze(1)
        return diff if per_prompt else diff.mean()

    def io_rank_stats(self, logits: Tensor) -> dict[str, float]:
        end = self.positions["END"]
        rows = torch.arange(len(self), device=logits.device)
        final = logits[rows, end]
        top1 = final.argmax(dim=-1)
        return {
            "top1_is_io": (top1 == self.io_token_ids).float().mean().item(),
            "io_beats_s": (self.logit_diff(logits, per_prompt=True) > 0).float().mean().item(),
        }


def _tokenize(texts: list[str], model: HookedTransformer) -> tuple[Tensor, Tensor]:
    """right-pad a list of prompts into one batch, returning tokens and true lengths."""
    seqs = [model.to_tokens(t)[0] for t in texts]
    lengths = torch.tensor([len(s) for s in seqs], device=seqs[0].device)
    padded = torch.full(
        (len(seqs), int(lengths.max())),
        model.tokenizer.pad_token_id,
        dtype=torch.long,
        device=seqs[0].device,
    )
    for i, seq in enumerate(seqs):
        padded[i, : len(seq)] = seq
    return padded, lengths


def _name_token_id(model: HookedTransformer, name: str) -> int:
    ids = model.tokenizer.encode(" " + name, add_special_tokens=False)
    if len(ids) != 1:
        raise AssertionError(f"name {name!r} is not a single token: {ids}")
    return ids[0]


# the phase 8 registration
# Defined above. nothing else in this file depends on it.
TASK = TaskSpec(
    name="ioi",
    dataset=IOIDataset,
    positions=POSITIONS,
    schemes=SCHEMES,
    discovery_schemes=DISCOVERY_SCHEMES,
    metric_label="logit difference (indirect object vs subject)",
    model_alias="gpt2-small",
)
