"""
the Python docstring task: clean/corrupted prompt pairs and the logit-difference metric.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from causal_interp.corruption import random_vocab_corruption
from causal_interp.schemes import Scheme, TaskSpec

# The two published word lists
VARIABLE_NAMES: tuple[str, ...] = tuple(
    "data name file value test new result line user key default request path output "
    "node item url model response text version function log string field start number "
    "values sub index error current context image message check json create size event "
    "state row obj parser end files form query fields instance label run action target "
    "array code num table source msg config first required options ret update module "
    "status group content client filename results last read command val base format "
    "color settings order found count match tag title parent call session token server "
    "host date description mode expected project info valid page header option task "
    "shape load names port old resource root".split(" ")
)

DESCRIPTION_NOUNS: tuple[str, ...] = tuple(
    "price control action cost issue process position course minute education type "
    "research subject girl age father force order value act matter health lot street "
    "patient class mind church condition paper bank century section activity table "
    "building sense sort staff team student language town plan management morning "
    "committee product practice evidence ground letter foot boy back game food union "
    "role event land art support range stage trade voice arm club field history parent "
    "account material care manager project record example training window air light "
    "wife quality rule pound story tax worker data model nature bed hospital method "
    "unit detail date wall computer amount bit president chapter property son director "
    "leader south application board king production operation share lord contract "
    "picture test security election source colour future site loss shop animal heart "
    "purpose standard page doctor factor hair love music charge pattern design piece "
    "population tree knowledge performance plant pressure fire environment size "
    "analysis rest success thought region list relation set space statement demand sea "
    "step capital choice player station film feature income individual cup technology "
    "machine cell degree energy growth treatment mile function risk sound task top "
    "resource floor science style hall horse response character user answer dog look "
    "brother argument season bill element glass duty claim fund leg park title note "
    "daughter sun box river profit division stone post client help image oil sector "
    "attack direction seat employment goal sign ability campaign fish item medium show "
    "version drug library press surface blood culture memory return bar talk access "
    "deal star text cuz mouth payment context reference second article chair earth "
    "object agency card collection communication public document weight bird rock call "
    "edge miss option quarter stock aid concept match network radio target finger "
    "forest race sex ball crime message peace review scale scene speech band expression "
    "hill reader owner trust truth turn file past start trial balance copy league "
    "length wind front move pain spirit train contact official strength cash gas shape "
    "agent pair protection rise driver master meaning vote adult play fig route speed "
    "credit impact danger flower half path track video aim bag comment content distance "
    "gold link skin boat dad prison sight wine offer writer hole package confidence "
    "generation key phone sample ship threat volume author background engine entry "
    "stuff yard bus regulation row song category consumer football meal wood bridge "
    "description existence flat lip session sheet code flight limit plate rain respect "
    "writing capacity dream selection spring while definition egg examination mark "
    "notice output will address bottom kid middle neck run acid assembly birth ear "
    "error module store transfer wave channel component cut fee lead weather block "
    "brain guide program screen connection cover map metal phase pool search sequence "
    "sky sum trip cat display gallery gate heat location reading theme drive faith hell "
    "learning opening priority spot tool castle coal flow lane motion release total "
    "wing border index pocket ring billion device engineering fruit lake leaf pub "
    "request square tone walk chain circle creation fall present shot fashion god mass "
    "panel fan iron origin ticket bone cancer chief editor fuel general height round "
    "vision warning being cross living rail working column finding gap grass plane root "
    "score shadow shock touch database formation male ref resident resolution topic "
    "break camp currency phrase thinking coat mill stress boot command depth female "
    "frame framework holder ice inch port protein rose saving string wheel bishop cycle "
    "export hat load mode setting stairs travel average camera focus green guard input "
    "layer poll sleep suit bread cake efficiency peak self angle bay bid cloud custom "
    "dollar variable zone actor chip core final frequency gain guy relative rent reply "
    "secret shirt apple instance intelligence lad pipe anger disk fight hero journal "
    "left mail pace paragraph platform print rank sand scope shift stream chemical "
    "clock delay host human count mission pack seed tail tie watch dark hold mine net "
    "tip dimension operator professional shell storage summary tube conduct "
    "determination drop import label percent pot proof unity win wire bell comfort "
    "complex draft grade lift mouse movie profile standing belt black check joy local "
    "pit red register storm album ban bench button cap chart coin dust folk margin pole "
    "stand stick tin".split(" ")
)

# the benchmark's argument counts
# With exactly these
N_MATCHING_ARGS = 3
N_DEF_PREFIX_ARGS = 2
N_DEF_SUFFIX_ARGS = 1
N_DOC_PREFIX_ARGS = 0
MET_DESC_LEN = 3
ARG_DESC_LEN = 2

POSITIONS: tuple[str, ...] = (
    "A_def", "B_def", "comma_B", "C_def", "A_doc", "B_doc", "END",
)

CORRUPTIONS: tuple[str, ...] = (
    "random_random", "random_def", "random_answer", "random_vocab_cdef", "random_vocab_any",
)

PUBLISHED_CORRUPTIONS: tuple[str, ...] = ("random_random", "random_def", "random_answer")
GENERIC_CORRUPTIONS: tuple[str, ...] = ("random_vocab_cdef", "random_vocab_any")

SCHEMES: dict[str, Scheme] = {
    "random_random": Scheme(
        name="random_random",
        provenance="published",
        breaks="replaces the definition arguments and the docstring arguments",
        preserves_answer=False,
        primary=True,
    ),
    "random_def": Scheme(
        name="random_def",
        provenance="published",
        breaks="replaces the non-answer definition arguments, breaking the induction match that selects the answer",
        preserves_answer=True,
    ),
    "random_answer": Scheme(
        name="random_answer",
        provenance="published",
        breaks="replaces the answer argument in the definition",
        preserves_answer=False,
    ),
    "random_vocab_cdef": Scheme(
        name="random_vocab_cdef",
        provenance="generic",
        breaks="substitutes a uniformly drawn vocabulary token at the C_def anchor",
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
class DocstringPrompt:
    """One clean/corrupted pair, with the argument names that define its answer."""

    clean: str
    corrupted: str
    matching: tuple[str, ...]   # the arguments repeated in the docstring, plus the answer
    answer: str                 # matching[-1]: the argument the model should predict
    wrong: tuple[str, ...]      # every other argument name drawn for this prompt
    all_args: tuple[str, ...] = field(default_factory=tuple)


class DocstringDataset:
    """a batch of docstring prompts, tokenized, with semantic position indices."""

    def __init__(
        self,
        model: HookedTransformer,
        n: int = 128,
        corruption: str = "random_random",
        seed: int = 0,
    ) -> None:
        if corruption not in CORRUPTIONS:
            raise ValueError(f"corruption must be one of {CORRUPTIONS}, got {corruption!r}")

        self.corruption = corruption
        self.seed = seed
        self.model = model

        names = self._single_token(model, VARIABLE_NAMES, "argument names")
        nouns = self._single_token(model, DESCRIPTION_NOUNS, "description nouns")
        rng = random.Random(seed)
        self.prompts = [self._make_prompt(rng, names, nouns) for _ in range(n)]

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

        # the answer and the distractors the metric reads, as token ids.
        self.answer_token_ids = torch.tensor(
            [self._token_id(model, " " + p.answer) for p in self.prompts], device=device
        )
        n_wrong = min(len(p.wrong) for p in self.prompts)
        self.wrong_token_ids = torch.stack(
            [
                torch.tensor([self._token_id(model, " " + w) for w in p.wrong[:n_wrong]])
                for p in self.prompts
            ]
        ).to(device)

        self.positions = self._locate_positions()

        if self.corruption in GENERIC_CORRUPTIONS:
            self.corrupted_tokens, self.corrupted_indices = self._apply_generic_corruption(seed)

    def __len__(self) -> int:
        return len(self.prompts)

    # -- construction -------------------------------------------------------

    @staticmethod
    def _token_id(model: HookedTransformer, text: str) -> int:
        ids = model.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise AssertionError(f"{text!r} is not a single token: {ids}")
        return ids[0]

    @staticmethod
    def _single_token(model: HookedTransformer, words: tuple[str, ...], label: str) -> list[str]:
        keep = [w for w in words if len(model.tokenizer.encode(" " + w, add_special_tokens=False)) == 1]
        if len(keep) < 40:
            raise RuntimeError(f"only {len(keep)} single-token {label} survived filtering")
        return keep

    def _make_prompt(
        self, rng: random.Random, names: list[str], nouns: list[str]
    ) -> DocstringPrompt:
        """one prompt, following the authors' `docstring_induction_prompt_generator`."""
        n_not_matching = N_MATCHING_ARGS - 1
        total = (
            2 + N_MATCHING_ARGS + n_not_matching + n_not_matching + N_MATCHING_ARGS
            + N_DEF_PREFIX_ARGS + N_DEF_SUFFIX_ARGS + N_DOC_PREFIX_ARGS
        )
        met_name, *all_args = rng.sample(names, total)

        rest = list(all_args)
        def take(k: int) -> list[str]:
            nonlocal rest
            head, rest = rest[:k], rest[k:]
            return head

        random_answer = take(1)[0]
        rand_mid_def = take(N_MATCHING_ARGS)
        rand_mid_doc = take(n_not_matching)
        not_matching = take(n_not_matching)
        matching = take(N_MATCHING_ARGS)
        def_prefix = take(N_DEF_PREFIX_ARGS)
        def_suffix = take(N_DEF_SUFFIX_ARGS)
        doc_prefix = take(N_DOC_PREFIX_ARGS)
        if rest:
            raise AssertionError(f"{len(rest)} argument names left unassigned")

        # description words must not collide with any argument name in this prompt
        # Or the token index of `A_def` / `B_doc` stops being well defined.
        reserved = {met_name, *all_args}
        pool = [w for w in nouns if w not in reserved]
        met_desc = rng.sample(pool, MET_DESC_LEN)

        clean_def = def_prefix + matching + def_suffix
        clean_doc = doc_prefix + matching[:-1]
        doc_desc = [rng.sample(pool, ARG_DESC_LEN) for _ in clean_doc]

        def render(def_args: list[str], doc_args: list[str]) -> str:
            return _template(
                met_name=met_name, met_desc_words=met_desc,
                def_args=def_args, doc_args=doc_args, doc_args_desc_words=doc_desc,
            )

        clean = render(clean_def, clean_doc)
        if self.corruption == "random_def":

            corrupted = render(def_prefix + not_matching + matching[-1:] + def_suffix, clean_doc)
        elif self.corruption == "random_answer":
            # the answer itself is replaced, so the argument the model should predict
            # isnt in the prompt at all.
            corrupted = render(def_prefix + matching[:-1] + [random_answer] + def_suffix, clean_doc)
        elif self.corruption == "random_random":
            corrupted = render(def_prefix + rand_mid_def + def_suffix, doc_prefix + rand_mid_doc)
        else:

            corrupted = clean

        return DocstringPrompt(
            clean=clean,
            corrupted=corrupted,
            matching=tuple(matching),
            answer=matching[-1],
            wrong=tuple(a for a in all_args if a != matching[-1]),
            all_args=tuple(all_args),
        )

    def _apply_generic_corruption(self, seed: int) -> tuple[Tensor, Tensor]:
        anchor = self.positions["C_def"] if self.corruption == "random_vocab_cdef" else None
        return random_vocab_corruption(
            clean_tokens=self.clean_tokens,
            lengths=self.lengths,
            d_vocab=self.model.cfg.d_vocab,
            seed=seed,
            anchor=anchor,
        )

    def _locate_positions(self) -> dict[str, Tensor]:
        """Find the seven position indices by searching the clean tokens."""
        device = self.clean_tokens.device
        comma = self._token_id(self.model, ",")
        found: dict[str, list[int]] = {name: [] for name in POSITIONS}

        for i, prompt in enumerate(self.prompts):
            row = self.clean_tokens[i, : self.lengths[i]]
            a, b, c = (self._token_id(self.model, " " + arg) for arg in prompt.matching)

            hits = {}
            for label, token_id, expected in (("A", a, 2), ("B", b, 2), ("C", c, 1)):
                where = (row == token_id).nonzero().flatten().tolist()
                if len(where) != expected:
                    raise AssertionError(
                        f"prompt {i}: argument {label} occurs {len(where)} times "
                        f"(expected {expected}) in {prompt.clean!r}"
                    )
                hits[label] = where

            end = int(self.lengths[i]) - 1
            idx = {
                "A_def": hits["A"][0],
                "B_def": hits["B"][0],
                "comma_B": hits["C"][0] - 1,
                "C_def": hits["C"][0],
                "A_doc": hits["A"][1],
                "B_doc": hits["B"][1],
                "END": end,
            }
            if int(row[idx["comma_B"]]) != comma:
                raise AssertionError(
                    f"prompt {i}: token before C_def is not a comma: {prompt.clean!r}"
                )
            if not idx["A_def"] < idx["B_def"] < idx["C_def"] < idx["A_doc"] < idx["B_doc"] < end:
                raise AssertionError(f"prompt {i}: positions out of order: {idx}")
            for name, value in idx.items():
                found[name].append(value)

        return {name: torch.tensor(v, device=device) for name, v in found.items()}

    # -- metric -------------------------------------------------------------

    def logit_diff(self, logits: Tensor, per_prompt: bool = False) -> Tensor:
        end = self.positions["END"]
        rows = torch.arange(len(self), device=logits.device)
        final = logits[rows, end]  # (batch, d_vocab)
        correct = final[rows, self.answer_token_ids]
        wrong = final[rows[:, None], self.wrong_token_ids].max(dim=-1).values
        diff = correct - wrong
        return diff if per_prompt else diff.mean()

    def answer_rank_stats(self, logits: Tensor) -> dict[str, float]:
        end = self.positions["END"]
        rows = torch.arange(len(self), device=logits.device)
        final = logits[rows, end]
        return {
            "answer_is_top_token": (final.argmax(dim=-1) == self.answer_token_ids).float().mean().item(),
            "logit_diff_positive": (self.logit_diff(logits, per_prompt=True) > 0).float().mean().item(),
        }


def _template(
    *,
    met_name: str,
    met_desc_words: list[str],
    def_args: list[str],
    doc_args: list[str],
    doc_args_desc_words: list[list[str]],
) -> str:
    """the authors' `docstring_prompt_templ` in its "rest" style, reproduced verbatim."""
    ind4 = 4 * " "
    def_args_str = ", ".join(def_args)
    met_desc_str = " ".join(met_desc_words)
    def_and_desc = f'''def {met_name}(self, {def_args_str}):
{ind4}"""{met_desc_str}
'''
    param_prefix = f"{ind4}:param"
    doc_lines = [
        f"{param_prefix} {arg}: {' '.join(desc)}"
        for arg, desc in zip(doc_args, doc_args_desc_words)
    ]
    doc_lines_str = "\n".join(doc_lines)
    return f"""
{def_and_desc}
{doc_lines_str}
{param_prefix}"""


# the phase 8 registration
# Class defined above. nothing else in this file depends on it.
TASK = TaskSpec(
    name="docstring",
    dataset=DocstringDataset,
    positions=POSITIONS,
    schemes=SCHEMES,
    discovery_schemes=DISCOVERY_SCHEMES,
    metric_label="logit difference (answer argument vs best wrong argument)",
    model_alias="attn-only-4l",
)
