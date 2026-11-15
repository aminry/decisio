# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Images at their place in a request's text, and what an image costs on the Gemma 4 bases (standard library only).

A request that carries images binds each one where its reference stands: a mark in the state's text or in a question's
own text. `bind_assets` and `extract_state_images` write the marks (`mark`), the Gemma image engine
(`decisio.serve.gemma_images`) renders the text, cuts it at the marks and puts the model's image placeholder there.
A mark is printable (a nested state is rendered with `str`, which would escape an unprintable character) and carries a
tag made fresh for each request, which only the server knows, so no text a user writes can be taken for a mark: text
that looks like one is plain text and is never refused.

The token count of an image is the Gemma 4 processor's: the picture is scaled, keeping its proportions, so that its
area fills the budget (`max_soft_tokens`), each side rounded down to a multiple of 48 pixels (16-pixel patches pooled
3 by 3), and every 48 by 48 block is one token; begin-of-image and end-of-image tokens wrap them.
"""

from __future__ import annotations

import math
import re
import secrets

# the soft-token budgets the processor accepts; 280 is Google's default
BUDGETS = (70, 140, 280, 560, 1120)
DEFAULT_BUDGET = 280
PATCH = 16
POOL = 3
UNIT = PATCH * POOL  # a token covers 48 by 48 pixels of the scaled picture

OPEN = "\u27e6image:"
CLOSE = "\u27e7"


def new_tag() -> str:
    """A fresh tag for one request's marks: 16 random hex characters, unknown to whoever wrote the request's text."""
    return secrets.token_hex(8)


def mark(n: int, tag: str) -> str:
    """The mark of image number `n` (counting from 1) in a text of the request that has `tag`."""
    return f"{OPEN}{tag}:{n}{CLOSE}"


def pattern(tag: str):
    """The regular expression of this request's marks and no other text: text that looks like a mark but carries
    another tag (or none) is plain text."""
    return re.compile(re.escape(f"{OPEN}{tag}:") + r"(\d+)" + re.escape(CLOSE))


def marks_in(text: str, tag: str) -> list[int]:
    """The numbers of the request's marks in `text`, in reading order."""
    return [int(m.group(1)) for m in pattern(tag).finditer(text)]


def marks_as_text(text: str, tag: str) -> str:
    """`text` with each of the request's marks written as "[image N]" (a question's image in the header that lists the
    questions: the picture itself stays in the question's own part of the prompt)."""
    return pattern(tag).sub(lambda m: f"[image {m.group(1)}]", text)


def without_marks(node, tag: str):
    """A state, question or text without the request's marks (what its text costs, apart from its images)."""
    if isinstance(node, str):
        return pattern(tag).sub("", node) if OPEN in node else node
    if isinstance(node, list):
        return [without_marks(x, tag) for x in node]
    if isinstance(node, dict):
        return {k: without_marks(v, tag) for k, v in node.items()}
    return node


def soft_tokens(height: int, width: int, budget: int = DEFAULT_BUDGET) -> int:
    """The image tokens the Gemma 4 processor makes of a picture of `height` by `width` pixels at `budget`, without
    the begin and end tokens: the same arithmetic as the processor's resize and vLLM's placeholder count."""
    if budget not in BUDGETS:
        raise ValueError(f"the image budget is one of {BUDGETS}, not {budget}")
    max_patches = budget * POOL**2
    scale = math.sqrt(max_patches / ((height / PATCH) * (width / PATCH)))
    new_h = max(UNIT, int(math.floor(height * scale / UNIT)) * UNIT)
    new_w = max(UNIT, int(math.floor(width * scale / UNIT)) * UNIT)
    return min(((new_h // PATCH) * (new_w // PATCH)) // POOL**2, budget)


def image_tokens(height: int, width: int, budget: int = DEFAULT_BUDGET) -> int:
    """What one image adds to a prompt: its soft tokens and the begin and end tokens around them."""
    return soft_tokens(height, width, budget) + 2


def block_tokens(budget: int) -> int:
    """The longest image block a budget can make: its most soft tokens and the begin and end tokens around them."""
    return budget + 2


def allowed_budgets(window: int | None) -> tuple[int, ...]:
    """The budgets whose largest image block fits a sliding window of `window` tokens (all of them when the window is
    not known)."""
    return BUDGETS if window is None else tuple(b for b in BUDGETS if block_tokens(b) <= window)


def check_budget_fits_window(budget: int, window: int | None) -> None:
    """Refuse a budget whose largest image block is longer than the base's sliding window. Gemma 4 lets an image's
    tokens see each other in both directions on a sliding layer only within the window, so in a longer block the
    later tokens could not see the top rows. A refusal, not a clamp: the operator asked for that budget."""
    if window is None or block_tokens(budget) <= window:
        return
    ok = allowed_budgets(window)
    use = f"use {ok[-1]} or lower" if ok else "no image budget fits this window"
    raise ValueError(
        f"--image-budget {budget} makes an image of up to {block_tokens(budget):,} tokens, longer than this model's "
        f"{window:,}-token sliding window, so the image's later tokens could not see its top rows; {use}"
    )


def sliding_window_of(config: dict) -> int | None:
    """The sliding window a checkpoint's config.json declares (`text_config.sliding_window`), None when it declares
    none."""
    w = (config.get("text_config") or config).get("sliding_window")
    return int(w) if isinstance(w, int) and w > 0 else None
