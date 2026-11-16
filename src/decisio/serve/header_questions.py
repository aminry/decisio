# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A client's declared question list (`"header_questions"` on /v1/systemone and /v1/answer; docs/api.md, and
docs/running.md, "The header layout").

Under the header layout every row starts with the request's questions listed before the state. A client that asks one
question at a time about one state, in separate requests, would send a different header each time, share no prefix and
make the server read the state again on every request. With `"header_questions": ["<instructions>", ...]` it declares
the whole list once, on every request about the state; the header then lists exactly those lines, in that order, on
every row of the request, so every request that sends the same list has the same header and the same prefix, and the
prefix cache serves the state after the first one. The question asked is still read after the state.

The rules (each a 422 that names the line):
  - a list of one or more strings, each with some text once stripped, no two the same once stripped;
  - every question of the request has its instructions among them (the line as the server renders it for the header,
    or the question's own `instructions` as it was sent);
  - the header the list makes must fit the server's prompt limit: a declared header is never cut (the header limit
    drops lines of an undeclared one, which would give two requests of one list different headers).
Under the plain layout the field is read, checked and then has no effect, so one client request works under both.
"""

from __future__ import annotations

from decisio.readout.letters import DECLARED_KEY
from decisio.serve.layout import LayoutRefused

FIELD = DECLARED_KEY


class HeaderQuestionsRefused(LayoutRefused):
    """A declared question list that cannot be served (a 422)."""


def parse(value, where: str = f'"{FIELD}"') -> tuple[str, ...] | None:
    """The declared list as the header lists it (each line stripped, as the request's own instructions are), or None
    where none was declared (absent or null). HeaderQuestionsRefused for an empty list, a value that is not a list of
    strings, an empty line or a duplicate."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise HeaderQuestionsRefused(
            f"{where} must be a non-empty list of strings (the questions' instructions), or null"
        )
    lines: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise HeaderQuestionsRefused(f"{where}[{i}] must be a string, not {type(item).__name__}")
        text = item.strip()
        if not text:
            raise HeaderQuestionsRefused(f"{where}[{i}] is empty")
        if text in lines:
            raise HeaderQuestionsRefused(f"{where}[{i}] repeats line {lines.index(text)}: each question is listed once")
        lines.append(text)
    return tuple(lines)


def check(declared, asked, where: str = f'"{FIELD}"') -> None:
    """Refuse a request whose question is not in the declared list. `asked`: per question, (name, the instruction
    lines it may be declared as). Names the first question missing and the line it would list."""
    if declared is None:
        return
    for name, candidates in asked:
        if not any(c in declared for c in candidates):
            raise HeaderQuestionsRefused(
                f"{where}: the question {name!r} is asked, but its instructions are not in the declared list; add "
                f"{candidates[0]!r} (or send the request without {where})"
            )


def stamp(questions: list[dict], declared) -> list[dict]:
    """The engine questions carrying the declared list (copies; the engine builds the header from it). The questions
    unchanged when none was declared."""
    if declared is None:
        return questions
    return [{**q, FIELD: list(declared)} for q in questions]
