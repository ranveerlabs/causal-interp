"""The Python docstring task: clean/corrupted prompt pairs and the logit-difference metric.

Argument-name prediction. Given a Python function signature followed by a
reST-style docstring that has already described some of its arguments, the model
should predict which argument name comes next. From Heimersheim and Janiak (2023),
*A circuit for Python docstrings in a 4-layer attention-only transformer*.

This module is the Phase 7 counterpart of `causal_interp.ioi` and
`causal_interp.greater_than`, written to the same interface so that everything
downstream — patching, path patching, the receiver search, the distributional
metrics — runs against it unchanged:

    clean_tokens / corrupted_tokens   aligned (batch, pos) token batches
    lengths                           true length per prompt
    positions[name]                   (batch,) token indices per semantic position
    logit_diff(logits, per_prompt)    the task's hand-built scalar metric
    __len__                           batch size

Unlike the two earlier tasks, this one does not live in GPT-2 small. The model is
`attn-only-4l` (`NeelNanda/Attn_Only_4L512W_C4_Code`) — 4 layers, 8 heads, **no MLP
blocks at all**, and a different tokenizer. Nothing in this module assumes
otherwise; the token ids, the prompt length and the position indices are all
derived from whatever tokenizer the model arrives with.

The prompt construction is the authors' own, reproduced from
`acdc/docstring/prompts.py` in the ACDC release (credited there to Stefan
Heimersheim and Kajetan Janiak), with the same word lists and the same argument
counts the benchmark uses. Two things differ, both recorded in
`results/PHASE7_PLAN.md` rather than discovered later:

- the random draws come from a local `random.Random` instead of reseeding the
  global `random` module once per prompt, so a dataset is reproducible without
  disturbing anything else in the process;
- description words are drawn from the noun list **with this prompt's argument
  names removed**. The two published word lists overlap, so a description word can
  otherwise coincide with an argument name and there is then no unambiguous token
  index for `A_def` or `B_doc`. The exclusion removes at most 15 of 687 nouns and
  is what makes the position vocabulary well defined, exactly as `greater_than.py`
  filters years that do not tokenize as two tokens.

    clean      def f(self, p, q, A, B, C, s):  ... :param A: ..  :param B: ..  :param
    corrupted  def f(self, p, q, X, Y, Z, s):  ... :param U: ..  :param V: ..  :param

The pair is generated token-aligned: every substitution swaps one single-token
argument name for another, so a clean activation and its corrupted counterpart live
at the same index and patching one into the other is well defined.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from causal_interp.corruption import random_vocab_corruption
from causal_interp.schemes import Scheme, TaskSpec

# The two published word lists, copied verbatim from `acdc/docstring/prompts.py`.
# Both are checked against the tokenizer in the constructor rather than trusted.
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
    "deal star text cause mouth payment context reference second article chair earth "
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

# The benchmark's argument counts: `get_all_docstring_things` calls the generator
# with exactly these, so the prompt shape is the released one and not a new choice.
N_MATCHING_ARGS = 3
N_DEF_PREFIX_ARGS = 2
N_DEF_SUFFIX_ARGS = 1
N_DOC_PREFIX_ARGS = 0
MET_DESC_LEN = 3
ARG_DESC_LEN = 2

# The semantic positions patching is resolved over. Seven, the same count IOI used,
# named after the post's own labels. `comma_B` is the post's `,_B` — the comma
# between `B_def` and `C_def` — spelled without the comma so it survives a markdown
# table cell. `END` is the final `param` token, where the answer is read off.
POSITIONS: tuple[str, ...] = (
    "A_def", "B_def", "comma_B", "C_def", "A_doc", "B_doc", "END",
)

# Three published counterfactuals from the authors' generator, then the two generic
# Phase 5 schemes. `random_random` is primary because it is the default in the
# released benchmark harness (`dataset_version="random_random"`) — a fact about the
# code release, not about anything measured here.
CORRUPTIONS: tuple[str, ...] = (
    "random_random", "random_def", "random_answer", "random_vocab_cdef", "random_vocab_any",
)

PUBLISHED_CORRUPTIONS: tuple[str, ...] = ("random_random", "random_def", "random_answer")
GENERIC_CORRUPTIONS: tuple[str, ...] = ("random_vocab_cdef", "random_vocab_any")

# Phase 8: the same five schemes, declared rather than listed. What each one *breaks*
# and whether it leaves the answer in the prompt were the facts Phase 7 had to
# reconstruct by hand after seeing a low recall number; here they are registered with
# the scheme, and `causal_interp.pipeline` runs discovery under every one of them.
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
    """A batch of docstring prompts, tokenized, with semantic position indices.

    Mirrors `IOIDataset` and `GreaterThanDataset`. `positions[name]` is a (batch,)
    tensor of token indices, so every analysis can say "patch head 2.0 at C_def"
    without knowing the layout.
    """

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

        # Every prompt is the same template with single-token substitutions, so the
        # batch must be exactly rectangular. Verify rather than assume: a ragged
        # batch would silently misalign every position index below.
        if self.clean_tokens.shape != self.corrupted_tokens.shape:
            raise AssertionError(
                f"clean/corrupted shape mismatch: "
                f"{tuple(self.clean_tokens.shape)} vs {tuple(self.corrupted_tokens.shape)}"
            )
        length = int(self.clean_tokens.shape[1])
        self.lengths = torch.full((n,), length, dtype=torch.long, device=device)

        # The answer and the distractors the metric reads, as token ids.
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
        """Keep only the words that are one token with a leading space.

        Both published lists were chosen for this model's tokenizer, so nothing is
        expected to be dropped — but a different tokenizer would silently break the
        whole position scheme, so it is measured rather than trusted.
        """
        keep = [w for w in words if len(model.tokenizer.encode(" " + w, add_special_tokens=False)) == 1]
        if len(keep) < 40:
            raise RuntimeError(f"only {len(keep)} single-token {label} survived filtering")
        return keep

    def _make_prompt(
        self, rng: random.Random, names: list[str], nouns: list[str]
    ) -> DocstringPrompt:
        """One prompt, following the authors' `docstring_induction_prompt_generator`."""
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

        # Description words must not collide with any argument name in this prompt,
        # or the token index of `A_def` / `B_doc` stops being well defined.
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
            # The non-answer matching arguments are replaced in the *definition*, so
            # the docstring's arguments no longer appear there and the induction
            # match that selects the answer is broken.
            corrupted = render(def_prefix + not_matching + matching[-1:] + def_suffix, clean_doc)
        elif self.corruption == "random_answer":
            # The answer itself is replaced, so the argument the model should predict
            # is not in the prompt at all.
            corrupted = render(def_prefix + matching[:-1] + [random_answer] + def_suffix, clean_doc)
        elif self.corruption == "random_random":
            corrupted = render(def_prefix + rand_mid_def + def_suffix, doc_prefix + rand_mid_doc)
        else:
            # Generic corruptions act on tokens, not text: a uniformly drawn
            # vocabulary entry has no spelling to substitute into a template. The
            # corrupted text is the clean text and the substitution happens after
            # tokenization, as it does for both earlier tasks.
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
        """Corrupt by substituting a uniformly drawn vocabulary token.

        The same function `IOIDataset` and `GreaterThanDataset` call, with the
        anchor pointed at this task's pivot instead of theirs. No knowledge of what
        any token means is used.
        """
        anchor = self.positions["C_def"] if self.corruption == "random_vocab_cdef" else None
        return random_vocab_corruption(
            clean_tokens=self.clean_tokens,
            lengths=self.lengths,
            d_vocab=self.model.cfg.d_vocab,
            seed=seed,
            anchor=anchor,
        )

    def _locate_positions(self) -> dict[str, Tensor]:
        """Find the seven position indices by searching the clean tokens.

        Each matching argument is looked up by token id. `A` and `B` must occur
        exactly twice — once in the definition, once in the docstring — and `C`
        exactly once, in the definition. Anything else means a description word
        collided with an argument name or the template is not what this module
        assumes, so it is a hard failure rather than a silently wrong index.
        """
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
        """The authors' docstring metric, at the END position.

            logit(correct argument)  minus  max over logits of the wrong arguments

        Matches `raw_docstring_metric` in the ACDC release, up to its sign: that
        implementation negates the quantity so it can be minimized, and this one
        does not, because everything in this project reads higher-is-better.

        The wrong-answer set is the authors': every other argument name drawn for
        the prompt, including the ones held back for the corruptions and never
        shown. Positive means the model prefers the argument the docstring
        convention demands.
        """
        end = self.positions["END"]
        rows = torch.arange(len(self), device=logits.device)
        final = logits[rows, end]  # (batch, d_vocab)
        correct = final[rows, self.answer_token_ids]
        wrong = final[rows[:, None], self.wrong_token_ids].max(dim=-1).values
        diff = correct - wrong
        return diff if per_prompt else diff.mean()

    def answer_rank_stats(self, logits: Tensor) -> dict[str, float]:
        """How often the model actually solves the task — a precondition for the phase.

        The counterpart of `IOIDataset.io_rank_stats` and
        `GreaterThanDataset.year_rank_stats`.
        """
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
    """The authors' `docstring_prompt_templ` in its "rest" style, reproduced verbatim."""
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


# The Phase 8 registration. Kept at the end of the module because it names the dataset
# class defined above; nothing else in this file depends on it.
TASK = TaskSpec(
    name="docstring",
    dataset=DocstringDataset,
    positions=POSITIONS,
    schemes=SCHEMES,
    discovery_schemes=DISCOVERY_SCHEMES,
    metric_label="logit difference (answer argument vs best wrong argument)",
    model_alias="attn-only-4l",
)
