# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Images at their place in a request's text, and what an image costs on the Gemma 4 bases (standard library only).

A request that carries images binds each one where its reference stands: a mark in the state's text or in a question's
own text. `bind_assets` and `extract_state_images` write the marks (`mark`), the Gemma image engine
(`decisio.serve.gemma_images`) renders the text, cuts it at the marks and puts the model's image placeholder there.
A mark is printable (a nested state is rendered with `str`, which would escape an unprintable character) and a request
whose own text contains the mark's opening is refused, so a mark cannot come from user text.

The token count of an image is the Gemma 4 processor's: the picture is scaled, keeping its proportions, so that its
area fills the budget (`max_soft_tokens`), each side rounded down to a multiple of 48 pixels (16-pixel patches pooled
3 by 3), and every 48 by 48 block is one token; begin-of-image and end-of-image tokens wrap them.
"""

from __future__ import annotations

import math
import re

# the soft-token budgets the processor accepts; 280 is Google's default
BUDGETS = (70, 140, 280, 560, 1120)
DEFAULT_BUDGET = 280
PATCH = 16
POOL = 3
UNIT = PATCH * POOL  # a token covers 48 by 48 pixels of the scaled picture

OPEN = "⟦image:"
CLOSE = "⟧"
MARK = re.compile(re.escape(OPEN) + r"(\d+)" + re.escape(CLOSE))


def mark(n: int) -> str:
    """The mark of image number `n` (counting from 1) in a text."""
    return f"{OPEN}{n}{CLOSE}"


def refuse_marks(text: str) -> None:
    """ValueError when user text contains a mark's opening: it could be taken for an image's place."""
    if OPEN in text:
        raise ValueError(f"the text contains {OPEN!r}, which is how a request marks where an image stands")


def marks_in(text: str) -> list[int]:
    """The numbers of the marks in `text`, in reading order."""
    return [int(m.group(1)) for m in MARK.finditer(text)]


def marks_as_text(text: str) -> str:
    """`text` with each mark written as "[image N]" (a question's image in the header that lists the questions: the
    picture itself stays in the question's own part of the prompt)."""
    return MARK.sub(lambda m: f"[image {m.group(1)}]", text)


def without_marks(node):
    """A state, question or text without its marks (what the text of a request costs, apart from its images)."""
    if isinstance(node, str):
        return MARK.sub("", node) if OPEN in node else node
    if isinstance(node, list):
        return [without_marks(x) for x in node]
    if isinstance(node, dict):
        return {k: without_marks(v) for k, v in node.items()}
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
