# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Which layout of the user turn a server serves (docs/design/default-layout.md; docs/running.md, "The header layout").

The Gemma bases are served with the header layout (the request's questions listed before the state) unless something the
server loads was fitted or trained on the plain layout, in which case the server holds the plain layout and says so.
The server's layout is decided once, at start, from the flags, the base, the repository and adapters, and the tasks the
stores hold. A request may ask for the other one (`"layout"` on `/v1/systemone`), and a task is registered under one.

    decide_layout(explicit=None, default="header", tuned=False, recorded=None, adapters={"n1": None},
                  tasks=["routing"], abstention=[])
        -> {"value": "plain", "why": "...", "default": "header", "held_by": ["adapter n1 (no recorded layout)"],
            "notes": []}

Precedence:
  1. `--layout` given: that layout, whatever is loaded (what does not match it is reported in `notes`).
  2. A tuned repository or an adapter was trained on a layout and measured on it: its recorded layout, plain when it
     recorded none (every tuned model that exists was trained on plain). Loaded together they must agree.
  3. Otherwise the base's default (`Family.layout`: header for the Gemma bases).
  4. A default of header is held at plain while tasks (calibration or abstention) fitted under the plain layout apply to
     this server: a task applies to the requests of the layout it was fitted under, and the requests that name no
     layout would stop being answered by it.

The layout in effect (`using`, `current`, `applied`): a request that names a layout, and a registration under one, are
served inside `using(layout)`. Inside it every prompt the engines build (`applied` gives the engine's format with
that layout) and every task and stack fingerprint is the layout's, so a row, a task and a certificate are the ones a
server of that layout makes. Outside it, the server's own. A context variable, as `decisio.serve.combine.serial_reads`:
the worker thread of a route inherits it; the engine thread of a batched server is given it with each engine call.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
from dataclasses import replace

from decisio.names import same_fingerprint

LAYOUT_VALUES = ("plain", "header")

# how a request asks for each layout, as the start line and the notes quote it
PLAIN_FIELD, HEADER_FIELD = '"layout": "plain"', '"layout": "header"'
# the layout a request or a registration asked for; None: the server's own
_IN_EFFECT: contextvars.ContextVar[str | None] = contextvars.ContextVar("decisio_layout", default=None)


class LayoutConflict(ValueError):
    """Things loaded together that were trained on different layouts cannot share a server."""


def check_layout(value, where: str):
    if value is not None and value not in LAYOUT_VALUES:
        raise ValueError(f"{where}: layout must be one of {LAYOUT_VALUES}, not {value!r}")
    return value


class LayoutRefused(ValueError):
    """A request's layout that cannot be served where it was sent (the image route, packed mode, an example of a task
    registered under another layout): a 422."""


@contextlib.contextmanager
def using(layout: str | None):
    """Inside the block, prompts are built and fingerprints taken under `layout` (None: the server's own)."""
    token = _IN_EFFECT.set(check_layout(layout, "layout"))
    try:
        yield
    finally:
        _IN_EFFECT.reset(token)


def current() -> str | None:
    """The layout in effect, or None outside `using` (the server's own)."""
    return _IN_EFFECT.get()


@functools.lru_cache(maxsize=64)
def _with_layout(fmt, layout):
    return replace(fmt, layout=layout)


def applied(fmt):
    """`fmt` (a decisio.readout.letters.PromptFormat) under the layout in effect: itself outside `using`."""
    value = _IN_EFFECT.get()
    return fmt if value is None or value == fmt.layout else _with_layout(fmt, value)


def request_value(value, where: str = '"layout"') -> str | None:
    """A request's `layout`: None (absent or null: the server's own) or one of LAYOUT_VALUES; anything else is refused
    (LayoutRefused, a 422) with the values allowed."""
    if value is None:
        return None
    if not isinstance(value, str) or value not in LAYOUT_VALUES:
        raise LayoutRefused(f"{where} must be one of {list(LAYOUT_VALUES)}, or null; not {value!r}")
    return value


def server_fmt(engine):
    """The engine's prompt format as the server serves it, whatever layout is in effect (None: the engine has none)."""
    token = _IN_EFFECT.set(None)
    try:
        return getattr(engine, "fmt", None)
    finally:
        _IN_EFFECT.reset(token)


def notes(records: dict, used: str, adapter: str | None = None) -> list[str]:
    """What a response says when it was served under another layout than what answered it was trained or fitted under.
    `records`: {"adapters": {name: recorded layout or None}, "repository": a tuned repository's recorded layout or None
    (present only for a tuned repository), "temperatures_file": --temperatures-file's recorded layout or None,
    "flavor": {"name", "layout"} or None}. An adapter or a tuned repository that records no layout was trained on the
    plain one (the rule of `decide_layout`); a file that records none says nothing. Never a refusal: the request is
    served as asked."""
    out = []
    if adapter is not None and adapter in (records.get("adapters") or {}):
        rec = records["adapters"][adapter]
        if rec is None and used != "plain":
            out.append(f"the adapter {adapter} records no layout: it was trained under the plain layout")
        elif rec is not None and rec != used:
            out.append(f"the adapter {adapter} was trained, and its temperatures fitted, under the {rec} layout")
    if "repository" in records:
        rec = records["repository"]
        if rec is None and used != "plain":
            out.append("the tuned repository records no layout: it was trained under the plain layout")
        elif rec is not None and rec != used:
            out.append(f"the tuned repository was trained, and its defaults fitted, under the {rec} layout")
    rec = records.get("temperatures_file")
    if rec is not None and rec != used:
        out.append(f"the temperatures (--temperatures-file) were fitted under the {rec} layout")
    flavor = records.get("flavor")
    if flavor and flavor.get("layout") not in (None, used):
        out.append(f"the flavor {flavor['name']} was fitted under the {flavor['layout']} layout")
    return out


def plain_fitted_readout(records, fingerprint_plain: str) -> list[str]:
    """Ids of the calibration/head tasks among `records` that apply to this server under the plain layout: fitted under
    it (its fingerprint equals the server's with the plain layout)."""
    out = []
    for t in records or []:
        if isinstance(t, dict) and t.get("fingerprint") is not None:
            if same_fingerprint(t["fingerprint"], fingerprint_plain):
                out.append(str(t.get("id")))
    return out


def plain_fitted_abstention(records, served_name: str) -> list[str]:
    """Ids of the abstention tasks among `records` fitted for this served name under the plain layout (a record with no
    `layout` was: the field is written only for another layout)."""
    return [
        str(t.get("id"))
        for t in records or []
        if isinstance(t, dict) and t.get("model") in (None, served_name) and (t.get("layout") or "plain") == "plain"
    ]


def decide_layout(
    *,
    explicit: str | None,
    default: str,
    tuned: bool = False,
    recorded: str | None = None,
    adapters: dict | None = None,
    tasks: list | None = None,
    abstention: list | None = None,
) -> dict:
    """The layout this server serves, and why; see the module's account. `adapters`: {name: its recorded layout or
    None}. `tasks` / `abstention`: ids of the stored tasks fitted under the plain layout that apply to this server."""
    adapters = adapters or {}
    tasks, abstention = list(tasks or []), list(abstention or [])
    needs = {}  # who trained on which layout: {description: layout}
    if tuned:
        needs["the tuned repository" + ("" if recorded else " (no recorded layout)")] = recorded or "plain"
    for name, rec in sorted(adapters.items()):
        needs[f"adapter {name}" + ("" if rec else " (no recorded layout)")] = rec or "plain"
    stored = [f"{len(tasks)} calibration task(s) fitted under the plain layout: {tasks}"] if tasks else []
    if abstention:
        stored.append(f"{len(abstention)} abstention task(s) fitted under the plain layout: {abstention}")
    out = {"value": None, "why": None, "default": default, "held_by": [], "notes": []}
    if explicit:
        out.update(value=explicit, why="--layout")
        for who, layout in needs.items():
            if layout != explicit:
                out["notes"].append(f"{who} was trained on the {layout} layout and is served on {explicit}")
        if explicit == "header":
            out["notes"] += [f"{x}; they apply to the requests that ask for {PLAIN_FIELD}" for x in stored]
        return out
    if needs:
        wanted = set(needs.values())
        if len(wanted) > 1:
            raise LayoutConflict(
                "the layouts they were trained on differ ("
                + "; ".join(f"{who}: {layout}" for who, layout in needs.items())
                + "); they cannot share a server's default layout. Serve them separately, or give --layout to choose "
                "the default (and measure); a request may ask for the other layout"
            )
        value = wanted.pop()
        why = "recorded by " + ", ".join(needs)
        if value == "plain" and default == "header":
            out["held_by"] = [who for who, layout in needs.items() if layout == "plain"]
        if value == "header":
            out["notes"] += [f"{x}; they apply to the requests that ask for {PLAIN_FIELD}" for x in stored]
        out.update(value=value, why=why)
        return out
    if default == "header" and stored:
        out.update(value="plain", why="held", held_by=stored)
        return out
    out.update(value=default, why="the default of the base")
    return out


def describe(decision: dict) -> str:
    """The start-up line."""
    value = decision["value"]
    if decision["why"] == "--layout":
        line = f"LAYOUT {value} (--layout)"
    elif decision["held_by"]:
        line = (
            f"LAYOUT {value} (held at plain; this base's default is {decision['default']}): "
            + "; ".join(decision["held_by"])
            + f". A request may ask for the header ({HEADER_FIELD}); to make it the default, register the tasks under "
            "it and start with --layout header (docs/running.md)"
        )
    else:
        line = f"LAYOUT {value} ({decision['why']})"
    return line + "".join(f"\nLAYOUT NOTE: {n}" for n in decision["notes"])
