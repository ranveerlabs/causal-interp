"""
the greater-than task: clean/corrupted prompt pairs and the probability-difference metric.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from causal_interp.corruption import random_vocab_corruption
from causal_interp.schemes import Scheme, TaskSpec

TEMPLATE = "The {noun} lasted from the year {century}{yy} to the year {century}"

TEMPLATE_SPLIT = "The {noun} lasted from the year {c1}{yy} to the year {c2}"

NOUNS: tuple[str, ...] = (
    "war", "drought", "famine", "siege", "journey", "voyage", "reign", "dynasty",
    "epidemic", "plague", "revolt", "uprising", "conflict", "struggle", "campaign",
    "expedition", "project", "program", "process", "occupation", "rebellion",
    "crusade", "feud", "boom", "recession", "depression", "renaissance", "revival",
    "movement", "partnership", "alliance", "truce", "ceasefire", "blockade",
    "quarrel", "dispute", "trial", "investigation", "search", "hunt", "chase",
    "festival", "tour", "exhibition", "construction", "restoration", "renovation",
    "excavation", "study", "experiment", "negotiation", "strike", "protest",
    "flood", "storm", "winter", "summer", "harvest", "migration", "voyage",
)

CENTURIES: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 17)
MIN_YY, MAX_YY = 2, 98

# the value the published counterfactual substitutes for YY.
CORRUPT_YY = 1

POSITIONS: tuple[str, ...] = ("NOUN", "XX1", "YY", "YY+1", "END")

CORRUPTIONS: tuple[str, ...] = ("yy01", "random_vocab_yy", "random_vocab_any")

GENERIC_CORRUPTIONS: tuple[str, ...] = ("random_vocab_yy", "random_vocab_any")

XX_MISMATCH = "xx_mismatch"

SCHEMES: dict[str, Scheme] = {
    "yy01": Scheme(
        name="yy01",
        provenance="published",
        breaks="sets the start year to 01, making the greater-than constraint vacuous",
        preserves_answer=False,
        primary=True,
    ),
    XX_MISMATCH: Scheme(
        name=XX_MISMATCH,
        provenance="authored",
        breaks=(
            "replaces the start year's century, breaking the correspondence between "
            "the two years while leaving YY untouched"
        ),
        preserves_answer=True,
    ),
    "random_vocab_yy": Scheme(
        name="random_vocab_yy",
        provenance="generic",
        breaks="substitutes a uniformly drawn vocabulary token at the YY anchor",
        preserves_answer=False,
    ),
    "random_vocab_any": Scheme(
        name="random_vocab_any",
        provenance="generic",
        breaks="substitutes a uniformly drawn vocabulary token anywhere in the prompt",
        preserves_answer=False,
    ),
}

DISCOVERY_SCHEMES: tuple[str, ...] = (
    "yy01", XX_MISMATCH, "random_vocab_yy", "random_vocab_any",
)


@dataclass(frozen=True)
class GreaterThanPrompt:
    """One clean/corrupted pair and the start year that defines its answer."""

    clean: str
    corrupted: str
    noun: str
    century: int
    yy: int
    corrupt_century: int | None = None   # xx_mismatch only: the century it substituted


class GreaterThanDataset:
    """A batch of greater-than pairs, tokenized, with semantic position indices."""

    def __init__(
        self,
        model: HookedTransformer,
        n: int = 128,
        corruption: str = "yy01",
        seed: int = 0,
    ) -> None:
        if corruption not in SCHEMES:
            raise ValueError(f"corruption must be one of {tuple(SCHEMES)}, got {corruption!r}")

        self.corruption = corruption
        self.seed = seed
        self.model = model

        nouns = self._single_token_nouns(model)
        self.valid_years = _valid_years(model)
        rng = random.Random(seed)

        self._alt_rng = random.Random(seed + 1)
        self.prompts = [self._make_prompt(rng, nouns) for _ in range(n)]

        device = model.cfg.device
        self.clean_tokens = model.to_tokens([p.clean for p in self.prompts])
        self.corrupted_tokens = model.to_tokens([p.corrupted for p in self.prompts])

        if self.clean_tokens.shape != self.corrupted_tokens.shape:
            raise AssertionError(
                f"clean/corrupted shape mismatch: "
                f"{tuple(self.clean_tokens.shape)} vs {tuple(self.corrupted_tokens.shape)}"
            )
        length = int(self.clean_tokens.shape[1])
        self.lengths = torch.full((n,), length, dtype=torch.long, device=device)

        # The 100 two-digit year tokens the metric reads. all of "00".."99" are
        # single tokens under GPT-2's tokenizer. verified, not assumed.
        self.year_token_ids = torch.tensor(
            [_year_token_id(model, y) for y in range(100)], device=device
        )
        self.yy_values = torch.tensor([p.yy for p in self.prompts], device=device)
        self.positions = self._locate_positions()

        if self.corruption in GENERIC_CORRUPTIONS:
            self.corrupted_tokens, self.corrupted_indices = self._apply_generic_corruption(seed)

    def __len__(self) -> int:
        return len(self.prompts)

    # -- construction -------------------------------------------------------

    @staticmethod
    def _single_token_nouns(model: HookedTransformer) -> list[str]:
        keep = []
        for noun in NOUNS:
            if len(model.tokenizer.encode(" " + noun, add_special_tokens=False)) == 1:
                if noun not in keep:
                    keep.append(noun)
        if len(keep) < 20:
            raise RuntimeError(f"only {len(keep)} single-token nouns survived filtering")
        return keep

    def _make_prompt(self, rng: random.Random, nouns: list[str]) -> GreaterThanPrompt:
        noun = rng.choice(nouns)
        century = rng.choice(sorted(self.valid_years))
        yy = rng.choice(self.valid_years[century])

        clean = _render(noun, century, yy, century)
        alt: int | None = None
        if self.corruption in GENERIC_CORRUPTIONS:

            corrupted = clean
        elif self.corruption == XX_MISMATCH:

            alt = self._alt_century(self._alt_rng, century, yy)
            corrupted = _render(noun, alt, yy, century)
        else:

            corrupted = _render(noun, century, CORRUPT_YY, century)
        return GreaterThanPrompt(clean, corrupted, noun, century, yy, alt)

    def _alt_century(self, rng: random.Random, century: int, yy: int) -> int:
        """A different century whose pairing with this `yy` still splits as two tokens."""
        options = [
            c for c in sorted(self.valid_years)
            if c != century and yy in self.valid_years[c]
        ]
        if not options:
            raise RuntimeError(
                f"no alternative century tokenizes with start year {yy:02d}; "
                f"{XX_MISMATCH} cannot be built for this prompt"
            )
        return rng.choice(options)

    def _apply_generic_corruption(self, seed: int) -> tuple[Tensor, Tensor]:
        anchor = self.positions["YY"] if self.corruption == "random_vocab_yy" else None
        return random_vocab_corruption(
            clean_tokens=self.clean_tokens,
            lengths=self.lengths,
            d_vocab=self.model.cfg.d_vocab,
            seed=seed,
            anchor=anchor,
        )

    def _locate_positions(self) -> dict[str, Tensor]:
        """Find NOUN/XX1/YY/END indices by searching the clean tokens for the century id."""
        device = self.clean_tokens.device
        found: dict[str, list[int]] = {name: [] for name in POSITIONS}

        for i, prompt in enumerate(self.prompts):
            row = self.clean_tokens[i, : self.lengths[i]]
            century_id = _century_token_id(self.model, prompt.century)
            hits = (row == century_id).nonzero().flatten().tolist()
            if len(hits) != 2:
                raise AssertionError(
                    f"prompt {i} has {len(hits)} occurrences of century token "
                    f"{prompt.century} (expected 2): {prompt.clean!r}"
                )

            end = int(self.lengths[i]) - 1
            if hits[1] != end:
                raise AssertionError(
                    f"prompt {i}: second century token at {hits[1]}, not at END={end}: "
                    f"{prompt.clean!r}"
                )
            if int(row[hits[0] + 1]) != _year_token_id(self.model, prompt.yy):
                raise AssertionError(
                    f"prompt {i}: token after the first century is not the start year "
                    f"{prompt.yy:02d}: {prompt.clean!r}"
                )

            idx = {
                "NOUN": 2,  # [BOS] "The" " <noun>" ...
                "XX1": hits[0],
                "YY": hits[0] + 1,
                "YY+1": hits[0] + 2,
                "END": end,
            }
            for name, value in idx.items():
                if value > end:
                    raise AssertionError(f"position {name}={value} past END={end} in {prompt.clean!r}")
                found[name].append(value)

        return {name: torch.tensor(v, device=device) for name, v in found.items()}

    # -- metric -------------------------------------------------------------

    def logit_diff(self, logits: Tensor, per_prompt: bool = False) -> Tensor:
        """the paper's probability difference, at the END position."""
        end = self.positions["END"]
        rows = torch.arange(len(self), device=logits.device)
        probs = logits[rows, end].softmax(dim=-1)  # (batch, d_vocab)
        year_probs = probs[:, self.year_token_ids]  # (batch, 100)

        years = torch.arange(100, device=logits.device)[None, :]
        sign = torch.where(years > self.yy_values[:, None], 1.0, -1.0)
        diff = (year_probs * sign).sum(dim=-1)
        return diff if per_prompt else diff.mean()

    def year_rank_stats(self, logits: Tensor) -> dict[str, float]:
        """how often the model actually solves the task, a precondition for the phase."""
        end = self.positions["END"]
        rows = torch.arange(len(self), device=logits.device)
        probs = logits[rows, end].softmax(dim=-1)
        year_probs = probs[:, self.year_token_ids]
        top_year = year_probs.argmax(dim=-1)
        return {
            "top_year_is_valid": (top_year > self.yy_values).float().mean().item(),
            "prob_diff_positive": (self.logit_diff(logits, per_prompt=True) > 0).float().mean().item(),
            "year_mass": year_probs.sum(dim=-1).mean().item(),
        }


def _render(noun: str, century_start: int, yy: int, century_end: int) -> str:
    return TEMPLATE_SPLIT.format(noun=noun, c1=century_start, yy=f"{yy:02d}", c2=century_end)


def _splits_as_two_tokens(model: HookedTransformer, century: int, yy: int) -> bool:
    ids = model.tokenizer.encode(f" {century}{yy:02d}", add_special_tokens=False)
    return ids == [_century_token_id(model, century), _year_token_id(model, yy)]


def _valid_years(model: HookedTransformer) -> dict[int, list[int]]:
    """per century, the start years that tokenize as [" XX", "YY"]."""
    out: dict[int, list[int]] = {}
    for century in CENTURIES:
        if not _splits_as_two_tokens(model, century, CORRUPT_YY):
            continue
        years = [
            yy for yy in range(MIN_YY, MAX_YY + 1)
            if _splits_as_two_tokens(model, century, yy)
        ]
        if years:
            out[century] = years
    if not out:
        raise RuntimeError("no century survived year-tokenization filtering")
    return out


def _year_token_id(model: HookedTransformer, year: int) -> int:
    ids = model.tokenizer.encode(f"{year:02d}", add_special_tokens=False)
    if len(ids) != 1:
        raise AssertionError(f"year {year:02d} is not a single token: {ids}")
    return ids[0]


def _century_token_id(model: HookedTransformer, century: int) -> int:
    ids = model.tokenizer.encode(f" {century}", add_special_tokens=False)
    if len(ids) != 1:
        raise AssertionError(f"century {century} is not a single token: {ids}")
    return ids[0]


# the phase 8 registration
# and the two generic ones, which need no knowledge of the task at all.
TASK = TaskSpec(
    name="greater-than",
    dataset=GreaterThanDataset,
    positions=POSITIONS,
    schemes=SCHEMES,
    discovery_schemes=DISCOVERY_SCHEMES,
    metric_label="probability difference (years > YY minus years <= YY)",
    model_alias="gpt2-small",
)
