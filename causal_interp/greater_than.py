"""The greater-than task: clean/corrupted prompt pairs and the probability-difference metric.

Numeric comparison. Given "The war lasted from the year 1732 to the year 17", the
model should predict a two-digit ending greater than 32. From Hanna, Liu,
Variengien (2023), arXiv:2305.00586.

This module is the Phase 6 counterpart of `causal_interp.ioi`, and is written to
the same interface so that everything downstream — patching, path patching, the
receiver search, the distributional metrics — runs against it unchanged. What a
dataset has to provide is small and is listed here so the contract is explicit
rather than discovered by breakage:

    clean_tokens / corrupted_tokens   aligned (batch, pos) token batches
    lengths                           true length per prompt
    positions[name]                   (batch,) token indices per semantic position
    logit_diff(logits, per_prompt)    the task's hand-built scalar metric
    __len__                           batch size

The pair is generated token-aligned, as for IOI: the corruption replaces exactly
one token, so a clean activation and its corrupted counterpart live at the same
index and patching one into the other is well defined.

    clean      The war lasted from the year 1732 to the year 17  -> " 33".." 99"
    corrupted  The war lasted from the year 1701 to the year 17  -> " 02".." 99"

One naming compromise, called out rather than hidden: the metric method is named
`logit_diff` because that is the name the intervention code calls. What it returns
is the *probability* difference the greater-than paper defines, not a difference of
logits. Renaming the interface across `interventions.py` would have meant editing
the shared causal core for a second task, which is exactly the kind of change
Phase 6 is trying to measure the absence of.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from causal_interp.corruption import random_vocab_corruption
from causal_interp.schemes import Scheme, TaskSpec

# The paper's template. A single template, unlike IOI's eight: the published task
# is this one sentence frame, and every prompt therefore has identical length,
# which is what lets the absolute-position search index tokens directly.
TEMPLATE = "The {noun} lasted from the year {century}{yy} to the year {century}"

# The same frame with the two century slots addressed separately. Phase 8's authored
# counterfactual changes the *start* year's century and leaves the final one alone, so
# it needs a renderer that can tell them apart. `TEMPLATE` is left exactly as it was:
# every prompt Phases 1-6 built is still built from the same string.
TEMPLATE_SPLIT = "The {noun} lasted from the year {c1}{yy} to the year {c2}"

# Event nouns that can sensibly last a span of years. The paper draws 120 from
# FrameNet; this list is filtered to single-token-with-leading-space under the
# GPT-2 tokenizer, and the constructor re-verifies rather than trusting it.
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

# Centuries and start years, exactly the paper's ranges: "We sample the century XX
# of the sentence from {11,...,17}, and the start year YY from {02,...,98}."
# Excluding 00, 01 and 99 also keeps the corrupted year "01" distinct from every
# clean one, so the corruption always changes something.
CENTURIES: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 17)
MIN_YY, MAX_YY = 2, 98

# Not every year in that range survives the tokenizer. GPT-2 splits " 1102" as
# [" 110", "2"] and " 1440" as one token [" 1440"], neither of which is the
# [" XX", "YY"] layout the whole position scheme depends on. The paper filters the
# same way — its `get_valid_years` keeps "only those tokenized into exactly 2
# tokens" — so this is the published construction, not a convenience of ours.
# Measured, not assumed: the filter is applied by probing the tokenizer.

# The value the published counterfactual substitutes for YY.
CORRUPT_YY = 1

# The semantic positions patching is resolved over. "YY" is the pivot — the token
# the whole task depends on, and the one the published corruption replaces, so it
# plays the role S2 plays for IOI. "END" is where the answer is read off.
POSITIONS: tuple[str, ...] = ("NOUN", "XX1", "YY", "YY+1", "END")

# `yy01` is the paper's own counterfactual and is the only hand-built scheme here.
# IOI had two, but greater-than has exactly one published counterfactual, and
# inventing a second would be new hand-tuning of the kind Phase 6 exists to detect.
# The `random_vocab_*` pair is the generic Phase 5 code path, shared verbatim.
# `CORRUPTIONS` is frozen at the three schemes Phase 6 swept, so re-running that phase
# reproduces it exactly. Phase 8's authored alternate is registered in `SCHEMES` below
# and reached through `DISCOVERY_SCHEMES`, which is what the multi-scheme pipeline uses.
CORRUPTIONS: tuple[str, ...] = ("yy01", "random_vocab_yy", "random_vocab_any")

GENERIC_CORRUPTIONS: tuple[str, ...] = ("random_vocab_yy", "random_vocab_any")

# Phase 8. Phase 6 recorded that greater-than has exactly one published counterfactual
# and that inventing a second would be hand-tuning of the kind Phase 6 existed to
# detect. Phase 7 then showed that running on one counterfactual decides which parts of
# a circuit are visible at all, so a second one has to exist -- and it is labelled
# `authored` wherever it appears, because that is exactly what it is.
#
# `random_def` does not transfer from the docstring task: it is defined in terms of the
# argument name the docstring points at, and greater-than has no pointer and no answer
# *token*. What transfers is the principle. The published scheme changes the value that
# gets moved (`YY`); the alternate leaves that value identical and breaks the structure
# that makes the task well posed:
#
#     clean        The war lasted from the year 1732 to the year 17
#     yy01         The war lasted from the year 1701 to the year 17
#     xx_mismatch  The war lasted from the year 1432 to the year 17
#
# One token changes either way, so clean and corrupted activations still live at the
# same index. The design, and two rejected candidates, are in results/PHASE8_PLAN.md.
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
    """A batch of greater-than pairs, tokenized, with semantic position indices.

    Mirrors `IOIDataset`. `positions[name]` is a (batch,) tensor of token indices,
    so every analysis can say "patch head 9.1 at YY" without knowing the layout.
    """

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
        # A separate stream for the one draw only `xx_mismatch` makes. Sharing `rng`
        # would shift every later draw, so the *clean* prompts under the authored
        # scheme would no longer be the clean prompts under the published one — and a
        # cross-scheme comparison would then be reading a different prompt sample as
        # well as a different counterfactual. Verified by scripts/check_schemes.py.
        self._alt_rng = random.Random(seed + 1)
        self.prompts = [self._make_prompt(rng, nouns) for _ in range(n)]

        device = model.cfg.device
        self.clean_tokens = model.to_tokens([p.clean for p in self.prompts])
        self.corrupted_tokens = model.to_tokens([p.corrupted for p in self.prompts])

        # Every prompt is the same template with single-token substitutions, so the
        # batch should be exactly rectangular. Verify rather than assume: a
        # ragged batch would silently misalign every position index below.
        if self.clean_tokens.shape != self.corrupted_tokens.shape:
            raise AssertionError(
                f"clean/corrupted shape mismatch: "
                f"{tuple(self.clean_tokens.shape)} vs {tuple(self.corrupted_tokens.shape)}"
            )
        length = int(self.clean_tokens.shape[1])
        self.lengths = torch.full((n,), length, dtype=torch.long, device=device)

        # The 100 two-digit year tokens the metric reads. All of "00".."99" are
        # single tokens under GPT-2's tokenizer; verified, not assumed.
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
            # Generic corruptions act on tokens, not text: a uniformly drawn
            # vocabulary entry has no spelling to substitute into a template. The
            # corrupted text is the clean text, as in IOIDataset, and the token
            # substitution happens after tokenization.
            corrupted = clean
        elif self.corruption == XX_MISMATCH:
            # Phase 8's authored alternate: a different century opens the start year,
            # so `YY` — the value the circuit's year heads move — is bit-identical
            # between the two runs, and what breaks is the relation between the two
            # years rather than the quantity being compared. The metric still reads
            # `YY` from the clean prompt, so the three numbers stay comparable.
            alt = self._alt_century(self._alt_rng, century, yy)
            corrupted = _render(noun, alt, yy, century)
        else:
            # The published counterfactual: the start year becomes 01, so the
            # greater-than constraint becomes vacuous while everything else — the
            # sentence, the century, the token count — is untouched.
            corrupted = _render(noun, century, CORRUPT_YY, century)
        return GreaterThanPrompt(clean, corrupted, noun, century, yy, alt)

    def _alt_century(self, rng: random.Random, century: int, yy: int) -> int:
        """A different century whose pairing with this `yy` still splits as two tokens.

        Drawn from the same filtered table the clean prompt is drawn from, so the
        corrupted prompt has the same token count as the clean one by construction
        rather than by hope. Raising here is correct: silently falling back to the
        clean century would produce a counterfactual that corrupts nothing.
        """
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
        """Corrupt by substituting a uniformly drawn vocabulary token.

        The same function `IOIDataset` calls, with the anchor pointed at this
        task's pivot instead of IOI's. No knowledge of what any token means is
        used: not which token is the year, not which answer is correct.
        """
        anchor = self.positions["YY"] if self.corruption == "random_vocab_yy" else None
        return random_vocab_corruption(
            clean_tokens=self.clean_tokens,
            lengths=self.lengths,
            d_vocab=self.model.cfg.d_vocab,
            seed=seed,
            anchor=anchor,
        )

    def _locate_positions(self) -> dict[str, Tensor]:
        """Find NOUN/XX1/YY/END indices by searching the clean tokens for the century id.

        The century token appears exactly twice — once opening the start year, once
        as the final token the answer follows. Anything else means the template or
        the tokenizer is not what this module assumes, so it is a hard failure.
        """
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
        """The paper's probability difference, at the END position.

            sum of p(year) over years > YY   minus   sum over years <= YY

        Named `logit_diff` to satisfy the interface `interventions.py` calls; it is
        a probability difference, not a logit difference. YY comes from the *clean*
        prompt in every run — clean, corrupted and patched — which is what makes
        the three numbers comparable. Positive means the model respects the
        greater-than constraint. Bounded in [-1, 1].

        Matches the authors' `prob_diff`: probabilities are read from the full
        next-token distribution and not renormalized over the year tokens.
        """
        end = self.positions["END"]
        rows = torch.arange(len(self), device=logits.device)
        probs = logits[rows, end].softmax(dim=-1)  # (batch, d_vocab)
        year_probs = probs[:, self.year_token_ids]  # (batch, 100)

        years = torch.arange(100, device=logits.device)[None, :]
        sign = torch.where(years > self.yy_values[:, None], 1.0, -1.0)
        diff = (year_probs * sign).sum(dim=-1)
        return diff if per_prompt else diff.mean()

    def year_rank_stats(self, logits: Tensor) -> dict[str, float]:
        """How often the model actually solves the task — a precondition for the phase.

        The counterpart of `IOIDataset.io_rank_stats`.
        """
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
    """One prompt. `century_start` and `century_end` differ only under `xx_mismatch`."""
    return TEMPLATE_SPLIT.format(noun=noun, c1=century_start, yy=f"{yy:02d}", c2=century_end)


def _splits_as_two_tokens(model: HookedTransformer, century: int, yy: int) -> bool:
    """Does " {century}{yy}" tokenize as exactly [" century", "yy"]?"""
    ids = model.tokenizer.encode(f" {century}{yy:02d}", add_special_tokens=False)
    return ids == [_century_token_id(model, century), _year_token_id(model, yy)]


def _valid_years(model: HookedTransformer) -> dict[int, list[int]]:
    """Per century, the start years that tokenize as [" XX", "YY"].

    A century is dropped entirely unless the *corrupted* year survives the same
    test: if " {century}01" does not split as two tokens, the counterfactual would
    have a different length from the prompt it is paired with and every position
    index would shift.
    """
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


# The Phase 8 registration. Four schemes: the paper's, one authored for this project,
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
