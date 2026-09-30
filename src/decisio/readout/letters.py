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
from functools import lru_cache

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
def _labels_for(tok_key, k, tok):
    out, seen = [], set()
    for code in _candidates():
        ids = tok.encode(" " + code, add_special_tokens=False)
        if len(ids) != 1:
            continue
        if ids[0] in seen:
            raise AssertionError(f"label {code!r} collides with an earlier label's token {ids[0]}")
        seen.add(ids[0])
        out.append(code)
        if len(out) == k:
            return tuple(out)
    raise AssertionError(f"only {len(out)} single-token letter labels exist in this tokenizer, {k} were asked for")


def letter_labels(tok, k):
    """The first `k` single-token letter codes, without the leading space ("A", "B", ... "AA", ...).

    Fails loudly rather than silently falling back to a multi-token label: a multi-token label would
    need a continuation row, which is exactly the cost this readout exists to remove.
    """
    if not 1 <= k <= MAX_LABELS:
        raise AssertionError(f"letter readout supports 1 to {MAX_LABELS} options, got {k}")
    return list(_labels_for(getattr(tok, "name_or_path", id(tok)), k, tok))


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


def letters_prompt(tok, kind, body, instructions, options, pool=None):
    """The letters-mode prompt and its candidate strings, for an already-formatted state `body`.

    Choice lists options as "A. <option>" lines. Score labels its levels with the same letters, so
    each is one token; the text-mode Score format (" 0" to " 9") is left alone because the trained
    adapters were trained on it. Yes/no is unchanged: " yes" and " no" are already single tokens.

    `pool` replaces the letters with an ordered list of codes (consonant_pair_pool + order_pool): the
    first K are used and the instruction asks for the code rather than the letter.
    """
    if kind == "noul":
        return f"{body}\n\n{instructions}\nAnswer yes or no.\nAnswer:", [" yes", " no"]
    if pool is None:
        labs, say = letter_labels(tok, len(options)), LETTERS_INSTRUCTION
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
