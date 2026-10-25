# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A header that does not outgrow the context (docs/running.md, "The header layout", "A header over the context").

Under the header layout every row of a request lists the instructions of all the request's questions before the state.
A request with many questions whose instructions are long (a tool catalogue, a set of documents) can render a row
longer than the context while the same request fits under the plain layout. The server then lists as many of the
instruction lines as fit, in the request's order, and leaves out the lines after them (a prefix, so two limits nest, as
a cut state does). When not even one line fits the row is the plain layout's, exactly. A request whose rows fit is not
touched: its rows are byte for byte what they were.

    --header-limit auto     the default: the limit is what the context leaves beside the state and the longest question
    --header-limit 20000    auto, and the listed lines take at most 20000 tokens
    --header-limit off      no limit: a row over the context is refused (a 422 that says where its tokens are)

Standard library only; the arithmetic lives here so that it can be tested without a tokenizer.
"""

from __future__ import annotations

AUTO = "auto"
OFF = "off"


def parse(value) -> tuple[str, int | None]:
    """(mode, cap) from the `--header-limit` value: ("auto", None), ("auto", N) or ("off", None). Raises ValueError."""
    if value is None:
        return AUTO, None
    text = str(value).strip().lower()
    if text in (AUTO, OFF):
        return text, None
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise ValueError(f"--header-limit {value!r}: auto, off, or a number of tokens of at least 1")
    return AUTO, n


def describe(mode: str, cap: int | None) -> str:
    """The setting as it is written on the command line and shown in /health."""
    return OFF if mode == OFF else (str(cap) if cap else AUTO)


def choose_lines(line_tokens: list[int], head_tokens: int, allowed: int) -> int:
    """The largest k such that the heading and the first k lines take at most `allowed` tokens; 0 when not even the
    first line does (the row is then the plain layout's). A prefix, never a pick of the shorter lines: the lines
    listed are the first the request asked about."""
    used, k = head_tokens, 0
    for t in line_tokens:
        if used + t > allowed:
            break
        used += t
        k += 1
    return k


def refusal(limit: int, pieces: dict, header_off: bool) -> str:
    """The 422 for a row over the context that no header limit can bring under: where its tokens are and the ways out.
    `pieces`: {"header": tokens of the full header or 0, "state": n, "question": n of the longest question,
    "template": n, "row": the longest row's tokens}."""
    row = pieces["row"]
    parts = [f"the state {pieces['state']:,}", f"the longest question {pieces['question']:,}"]
    if pieces.get("header"):
        parts.insert(0, f"the header {pieces['header']:,}")
    head = (
        f"the longest row of the request is {row:,} tokens ({', '.join(parts)}, the template and system turn "
        f"{pieces['template']:,}) against the {limit:,} the server answers"
    )
    ways = []
    if header_off and pieces.get("header"):
        ways.append("start the server with --header-limit auto, which lists only the questions that fit")
    ways.append('send "over_limit": "truncate" to cut the state to its start')
    ways.append("shorten the state or the question")
    ways.append("or start the server with a longer --max-model-len (up to the base's ceiling)")
    return f"{head}; " + ", ".join(ways[:-1]) + ", " + ways[-1]


def report(
    *, listed: int, of: int, header_tokens: int, full_header_tokens: int, limit: int, allowed: int, reason: str
) -> dict:
    """What the response, the `x-decisio-layout` header and /health say of a request whose header was limited:
    the lines listed of those asked, the header's tokens listed and in all, the row limit and the tokens the header
    could take, and why (`context`: the row would be over the context; `cap`: the --header-limit number)."""
    return {
        "layout": "header" if listed else "plain",
        "listed": listed,
        "of": of,
        "header_tokens": header_tokens,
        "of_header_tokens": full_header_tokens,
        "row_limit": limit,
        "allowed_tokens": allowed,
        "reason": reason,
    }


def header_value(rep: dict) -> str:
    """The `x-decisio-layout` value of a limited request, e.g. "header; listed=12; of=32; reason=context"."""
    return f"{rep['layout']}; listed={rep['listed']}; of={rep['of']}; reason={rep['reason']}"
