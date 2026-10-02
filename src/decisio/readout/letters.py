# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Letter readout: list the options with letter markers and read one answer position.

Scoring every option string as its own continuation costs one continuation row per option (77 for a
Banking77 question), and in serving each row carries its own copy of the question's cache, including
the Gated DeltaNet state. Score levels rendered " 0" to " 9" are two tokens each in the Qwen3.6
tokenizer, with an identical first token, so every level pays for a row that adds nothing.

The letter readout asks for the option's letter instead. Each question is then one prefill and one read
at its last position, through only the K label rows of the output layer: no option rows, no per-row
state copies. It needs no training, because a base model already answers multiple choice by letter,
which is MMLU's native format.

Labels are " A" to " Z", then " AA", " AB", ... in order, keeping only strings that encode to exactly
one token. In the Qwen3.6 tokenizer all 26 single letters and 544 of the 676 two-letter codes are
single tokens, so 255 labels exist without adding to the vocabulary.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

MAX_LABELS = 255  # the wire format's cap, an 8-bit option index

LETTERS_INSTRUCTION = "Answer with the letter only."
CODES_INSTRUCTION = "Answer with the code only."

# `--labels consonant-pairs`: two-letter uppercase codes without vowels or Y. " A" .. " J" are also
# words or word-starts (" I", " A"), and a base model over-picked " I" and " J" on MMLU-Pro; codes like
# " BK" carry no such meaning. 321 of the 400 pairs are single tokens in the Qwen3.6 tokenizer.
CONSONANTS = "BCDFGHJKLMNPQRSTVWXZ"
LABEL_SCHEMES = ("letters", "consonant-pairs")


def fmt_state(state):
    """How a state is rendered into the prompt, the same for serving and evaluation."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):  # "_" keys are item metadata, never shown
        return "\n".join(f"{k}: {v}" for k, v in state.items() if not k.startswith("_"))
    return str(state)


def _candidates():
    letters = string.ascii_uppercase
    yield from letters
    for a in letters:
        for b in letters:
            yield a + b


@lru_cache(maxsize=None)
def _labels_for(tok_key, k, tok, form="spaced"):
    out, seen = [], set()
    for code in _candidates():
        ids = tok.encode((" " + code) if form == "spaced" else code, add_special_tokens=False)
        if len(ids) != 1:
            continue
        if ids[0] in seen:
            raise AssertionError(f"label {code!r} collides with an earlier label's token {ids[0]}")
        seen.add(ids[0])
        out.append(code)
        if len(out) == k:
            return tuple(out)
    raise AssertionError(f"only {len(out)} single-token letter labels exist in this tokenizer, {k} were asked for")


def letter_labels(tok, k, form="spaced"):
    """The first `k` single-token letter codes, without the leading space ("A", "B", ... "AA", ...).

    `form`: the form that must be one token, "spaced" (" AA", today's rule) or "bare" ("AA", for a format that reads
    the bare form only: the template's answer slot with one form per label). In the Qwen3.6 tokenizer the two lists
    agree up to 85 options; past that the bare list skips CJ, CZ, EJ, GK and JL.

    Fails loudly rather than silently falling back to a multi-token label: a multi-token label would
    need a continuation row, which is exactly the cost this readout exists to remove.
    """
    if not 1 <= k <= MAX_LABELS:
        raise AssertionError(f"letter readout supports 1 to {MAX_LABELS} options, got {k}")
    if form not in ("spaced", "bare"):
        raise ValueError(f"form must be 'spaced' or 'bare', not {form!r}")
    return list(_labels_for(getattr(tok, "name_or_path", id(tok)), k, tok, form))


def consonant_pair_pool(tok):
    """Every two-consonant code whose " XY" is one token, alphabetical. Order it with order_pool."""
    out, seen = [], set()
    for a in CONSONANTS:
        for b in CONSONANTS:
            ids = tok.encode(" " + a + b, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0])
                out.append(a + b)
    return out


def pool_prior_prompt():
    """The content-free prompt a label pool is ordered under: no state and no option list, so the
    score of each code is its prior as an answer, not its position in some listing."""
    return f"N/A\n\nWhich option correctly answers the question?\n{CODES_INSTRUCTION}\nAnswer:"


def order_pool(pool, logprobs):
    """Flattest first: codes sorted by distance of their content-free log-probability from the pool's
    median, so every prefix of the order - the first K codes are what a K-option question uses - is
    as close to uniform as the pool allows. Ties keep alphabetical order."""
    lp = [float(x) for x in logprobs]
    med = sorted(lp)[len(lp) // 2]
    return [c for _, c in sorted(zip([abs(x - med) for x in lp], pool), key=lambda t: t[0])]


def label_token_ids(tok, labels):
    """Token id for each candidate string (with its leading space); each must be exactly one token."""
    ids = []
    for lab in labels:
        t = tok.encode(lab, add_special_tokens=False)
        if len(t) != 1:
            raise AssertionError(f"candidate {lab!r} is {len(t)} tokens; letter readout needs one")
        ids.append(t[0])
    if len(set(ids)) != len(ids):
        raise AssertionError(f"candidate labels collide: {labels}")
    return ids


PROMPT_MODES = ("base", "chat", "chat-think")


def chat_wrap(tok, prompt, mode):
    """A completion-format prompt (ending "\nAnswer:") in the given prompt mode.

    base:       unchanged.
    chat:       the body (without "\nAnswer:") as the user turn of the chat template with thinking
                disabled, then "Answer:" prefilled in the assistant turn. Qwen3.6's template disables
                thinking by writing an empty, closed "<think>\n\n</think>" block, so the check is that no
                think block is left open.
    chat-think: the template's default, which opens "<think>" and leaves it open.

    The one implementation for serving and evaluation, so the two cannot drift apart.
    """
    if mode == "base":
        return prompt
    if mode not in PROMPT_MODES:
        raise ValueError(f"unknown prompt mode {mode!r}")
    assert prompt.endswith("\nAnswer:"), prompt[-40:]
    kw = {"enable_thinking": False} if mode == "chat" else {}
    out = tok.apply_chat_template(
        [{"role": "user", "content": prompt[: -len("\nAnswer:")]}], add_generation_prompt=True, tokenize=False, **kw
    )
    if mode == "chat" and out.count("<think>") != out.count("</think>"):
        raise AssertionError(f"thinking left open in chat mode: {out[-60:]!r}")
    return out + "Answer:"


def letters_prompt(tok, kind, body, instructions, options, pool=None, form="spaced"):
    """The letters-mode prompt and its candidate strings, for an already-formatted state `body`.

    Choice lists options as "A. <option>" lines. Score labels its levels with the same letters, so
    each is one token; the text-mode Score format (" 0" to " 9") is left alone because the trained
    adapters were trained on it. Yes/no is unchanged: " yes" and " no" are already single tokens.

    `pool` replaces the letters with an ordered list of codes (consonant_pair_pool + order_pool): the
    first K are used and the instruction asks for the code rather than the letter. `form`: as letter_labels.
    """
    if kind == "noul":
        return f"{body}\n\n{instructions}\nAnswer yes or no.\nAnswer:", [" yes", " no"]
    if pool is None:
        labs, say = letter_labels(tok, len(options), form), LETTERS_INSTRUCTION
    else:
        if len(options) > len(pool):
            raise AssertionError(f"{len(options)} options but only {len(pool)} codes in the pool")
        labs, say = list(pool[: len(options)]), CODES_INSTRUCTION
    if kind == "score":
        legend = "; ".join(f"{lab} = {o}" for lab, o in zip(labs, options))
        q = f"{body}\n\n{instructions}\nRate on this scale: {legend}.\n{say}"
    else:
        listing = "\n".join(f"{lab}. {o}" for lab, o in zip(labs, options))
        q = f"{body}\n\n{instructions}\nOptions:\n{listing}\n{say}"
    return q + "\nAnswer:", [" " + lab for lab in labs]


# ---- prompt formats: system prompt, tail, answer slot, label forms ---------------------------------------------------

SYSTEM_PROMPTS = ("none", "cygnet")
PROMPT_TAILS = ("decisio", "cygnet")
ANSWER_SLOTS = ("prefill", "template")
LABEL_VARIANTS = ("single", "summed")
# thinking blocks a chat template may write; with thinking off each must be closed
THINK_MARKERS = (("<think>", "</think>"), ("<|channel>", "<channel|>"))


@dataclass(frozen=True)
class PromptFormat:
    """How a letters question is put to the chat model; every row the engine builds goes through it.

    system   none: no system turn; cygnet: the Cygnet recipe's system prompt (decisio.readout.cygnet.SYSTEM)
    tail     decisio: "<instructions>\nOptions:\n<listing>\nAnswer with the letter only." (a score as a legend);
             cygnet: "<instructions>\n\nOptions:\n<listing>\n\n" + the recipe's last line (score levels listed)
    slot     prefill: "Answer:" after the template's generation prompt; template: the template's own first
             assistant position (Qwen3.6 with thinking off: after its empty, closed "<think>" block)
    variants single: one token per label, the form the slot reads (" A" after "Answer:", "A" in the template
             slot); summed: every single-token form of a label (" A" and "A"; " yes", "yes", " Yes", "Yes"),
             read as allowed tokens of the same request, their probabilities summed per label

    The default is the served prompt as it was before these flags, token for token."""

    system: str = "none"
    tail: str = "decisio"
    slot: str = "prefill"
    variants: str = "single"

    def __post_init__(self):
        for value, allowed, what in (
            (self.system, SYSTEM_PROMPTS, "system"),
            (self.tail, PROMPT_TAILS, "tail"),
            (self.slot, ANSWER_SLOTS, "slot"),
            (self.variants, LABEL_VARIANTS, "variants"),
        ):
            if value not in allowed:
                raise ValueError(f"PromptFormat.{what} must be one of {allowed}, not {value!r}")

    def is_default(self) -> bool:
        return self == PromptFormat()

    def label_form(self) -> str:
        """The letter-code list the format needs: "bare" when it reads the bare form only, else "spaced"."""
        return "bare" if (self.variants, self.slot) == ("single", "template") else "spaced"

    def facts(self) -> dict:
        return {"system": self.system, "tail": self.tail, "slot": self.slot, "variants": self.variants}


DEFAULT_FORMAT = PromptFormat()


def question_body(tok, kind, instructions, options, fmt: PromptFormat = DEFAULT_FORMAT):
    """A question's own text, the part after "<state>\n\n" in the user turn, and its label strings (" A", ...;
    " yes", " no"). Under the default format exactly what letters_prompt writes between the state's blank line and
    "\nAnswer:"."""
    form = fmt.label_form()
    if fmt.tail == "decisio":
        full, cands = letters_prompt(tok, kind, "", instructions, options, form=form)
        assert full.startswith("\n\n") and full.endswith("\nAnswer:")
        return full[2 : -len("\nAnswer:")], cands
    from decisio.readout.cygnet import TAIL

    if kind == "noul":
        return f"{instructions}\n\nAnswer with yes or no, and nothing else:", [" yes", " no"]
    labs = letter_labels(tok, len(options), form)
    listing = "\n".join(f"{lab}. {o}" for lab, o in zip(labs, options))
    return f"{instructions}\n\nOptions:\n{listing}\n\n{TAIL}", [" " + lab for lab in labs]


def check_thinking_closed(text: str) -> None:
    for open_, close in THINK_MARKERS:
        if text.count(open_) != text.count(close):
            raise AssertionError(f"thinking left open in chat mode: {text[-80:]!r}")


def chat_turn(tok, content, fmt: PromptFormat = DEFAULT_FORMAT) -> str:
    """One question's chat prompt: the system turn (if any), `content` as the user turn, the template's generation
    prompt with thinking disabled, then "Answer:" under the prefill slot. With the default format this is
    chat_wrap(tok, content + "\nAnswer:", "chat"), the prompt evaluation scores, by construction."""
    if fmt.is_default():
        return chat_wrap(tok, content + "\nAnswer:", "chat")
    messages = [{"role": "user", "content": content}]
    if fmt.system == "cygnet":
        from decisio.readout.cygnet import SYSTEM

        messages.insert(0, {"role": "system", "content": SYSTEM})
    out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    check_thinking_closed(out)
    return out + ("Answer:" if fmt.slot == "prefill" else "")


def _byte_token(tok, ch):
    """The id of a byte-fallback token for one ASCII character ("<0x41>" for "A"), or None (byte-level BPE
    tokenizers such as Qwen's have none)."""
    if len(ch) != 1 or ord(ch) > 127:
        return None
    name = f"<0x{ord(ch):02X}>"
    tid = tok.convert_tokens_to_ids(name)
    unk = getattr(tok, "unk_token_id", None)
    return tid if isinstance(tid, int) and tid >= 0 and tid != unk and tok.convert_ids_to_tokens(tid) == name else None


def _one_token(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 and tok.decode(ids) == text else None


def label_forms(tok, cand, fmt: PromptFormat = DEFAULT_FORMAT) -> tuple[int, ...]:
    """The token ids read for one label string (" A", " yes"), in a fixed order.

    single: one token, the form the slot reads (spaced after "Answer:", bare in the template slot), which must be
    one token. summed: every form that is exactly one token (spaced, bare, and for yes / no the capitalised forms;
    a byte-fallback token where the tokenizer has one), at least one."""
    lab = cand.strip()
    if fmt.variants == "single":
        text = (" " + lab) if fmt.slot == "prefill" else lab
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise AssertionError(f"label {text!r} is {len(ids)} tokens; the letter readout needs one")
        return (ids[0],)
    texts = [" " + lab, lab]
    if lab.isalpha() and lab.islower():  # yes / no
        texts += [" " + lab.capitalize(), lab.capitalize()]
    forms = [_one_token(tok, t) for t in texts] + [_byte_token(tok, lab)]
    out = tuple(dict.fromkeys(t for t in forms if t is not None))
    if not out:
        raise AssertionError(f"label {lab!r} has no single-token form in this tokenizer")
    return out


def label_groups(tok, cands, fmt: PromptFormat = DEFAULT_FORMAT):
    """A question's label ids: a list of ints, one token per label, under single (the prefill slot's is exactly
    label_token_ids, today's rule); a tuple of per-label tuples of ids under summed. Two labels never share a token."""
    if fmt.variants == "single" and fmt.slot == "prefill":
        return label_token_ids(tok, cands)
    groups = tuple(label_forms(tok, c, fmt) for c in cands)
    flat = [t for g in groups for t in g]
    if len(set(flat)) != len(flat):
        raise AssertionError(f"label forms collide: {cands}")
    return [g[0] for g in groups] if fmt.variants == "single" else groups


def is_grouped(lab) -> bool:
    return bool(lab) and isinstance(lab[0], tuple)


def allowed_ids(lab) -> list[int]:
    """The token ids a question's read allows (every form of every label, in label order)."""
    return [t for g in lab for t in g] if is_grouped(lab) else list(lab)


def label_logprobs(lab, lp_of):
    """Per-label log-probabilities from per-token ones (`lp_of(token id)`, a log-softmax over allowed_ids(lab)): the
    tokens themselves for one token per label; the log of each label's summed probability otherwise. float64."""
    if not is_grouped(lab):
        return np.array([lp_of(t) for t in lab], dtype=np.float64)
    out = []
    for g in lab:
        v = np.array([lp_of(t) for t in g], dtype=np.float64)
        m = v.max()
        out.append(m + np.log(np.exp(v - m).sum()))
    return np.array(out, dtype=np.float64)


def label_log_softmax(z_allowed, lab):
    """Per-label log-probabilities from the logits of allowed_ids(lab), in that order: the log-softmax over the allowed
    ids (what vLLM returns as processed log-probabilities), then label_logprobs. float64."""
    z = np.asarray(z_allowed, dtype=np.float64)
    z = z - z.max()
    lp = z - np.log(np.exp(z).sum())
    at = {t: i for i, t in enumerate(allowed_ids(lab))}
    return label_logprobs(lab, lambda t: lp[at[t]])
