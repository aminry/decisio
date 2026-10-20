# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A state longer than the engine's prompt ceiling: refuse it (the default) or answer from its start.

`refuse`: the engine's own refusal stands (a 422 that says the rendered prompt is over the context). `truncate`: the
state, never a question, is cut once per request to what the longest question of the request leaves, keeping the
start; the response says what was cut, and the answers are never certified (docs/api.md, docs/running.md).

What is cut: a request's questions share one state prefix (the prefix cache depends on it), so the state is cut once,
at a token, backed up to whitespace within `BACK_UP` tokens so that it does not end mid-word. No marker is inserted: a
tuned model has not seen one, and the response carries the cut. A truncated state is a prefix of the full one, so the
same state at two ceilings nests.
"""

from __future__ import annotations

MODES = ("refuse", "truncate")
PART = "start"  # the only part kept (docs/running.md says why)
BACK_UP = 64  # tokens: how far the cut may move back to land on whitespace
MARGIN = 32  # tokens kept free below the ceiling: the second order's rows, the block padding, re-tokenization


class OverLimit(ValueError):
    """The request cannot be answered even from an empty state (its questions alone are over the ceiling), or the
    engine has no known ceiling to cut to. The server answers 422, as for any input over capacity."""


def check_mode(value, where: str) -> str:
    if value not in MODES:
        raise ValueError(f"{where} must be one of {list(MODES)}, got {value!r}")
    return value


def fit_state(tok, body: str, overhead: int, limit: int, reserve: int = 0) -> tuple[str, dict | None]:
    """(state text, report). `body`: the rendered state; `overhead`: tokens of the longest row without its state (the
    template, the question's text and the closing line); `limit`: the engine's prompt ceiling in tokens; `reserve`:
    more tokens kept free (the block padding). The report is None when the whole state fits; otherwise
    `{"kept_tokens", "of_tokens", "part", "limit"}`, the state's own tokens kept and the state's tokens in all. Raises
    OverLimit when no state at all fits."""
    room = limit - overhead - MARGIN - reserve
    # a token is at least one byte, so a state of at most `room` bytes fits without being tokenized
    if len(body.encode("utf-8")) <= room:
        return body, None
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    ids = enc(body)
    if len(ids) <= room:
        return body, None
    if room < 1:
        raise OverLimit(
            f"the questions alone are over the prompt limit of {limit:,} tokens (the longest row takes {overhead:,} "
            "without a state); shorten the question or its options"
        )
    cut = room
    for j in range(room, max(0, room - BACK_UP), -1):  # land on whitespace: the next token or the last one kept
        if tok.decode(ids[j : j + 1])[:1].isspace() or tok.decode(ids[j - 1 : j])[-1:].isspace():
            cut = j
            break
    text = tok.decode(ids[:cut]).rstrip().rstrip("�").rstrip()
    while text and len(enc(text)) > room:  # re-tokenizing the text can differ at its end
        text = text[: max(0, len(text) - max(1, len(text) // 200))].rstrip()
    if not text:
        raise OverLimit(f"no part of the state fits the prompt limit of {limit:,} tokens beside the question")
    return text, {"kept_tokens": len(enc(text)), "of_tokens": len(ids), "part": PART, "limit": limit}
