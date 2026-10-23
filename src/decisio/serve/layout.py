# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Which layout of the user turn a server serves (docs/design/default-layout.md; docs/running.md, "The header layout").

The Gemma bases are served with the header layout (the request's questions listed before the state) unless something the
server loads was fitted or trained on the plain layout, in which case the server holds the plain layout and says so.
Nothing here depends on a request: the layout is decided once, at start, from the flags, the base, the repository and
adapters, and the tasks the stores hold.

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
     this server: a task is kept under the layout it was fitted under, and the default would stop applying it.
"""

from __future__ import annotations

from decisio.names import same_fingerprint

LAYOUT_VALUES = ("plain", "header")


class LayoutConflict(ValueError):
    """Things loaded together that were trained on different layouts cannot share a server."""


def check_layout(value, where: str):
    if value is not None and value not in LAYOUT_VALUES:
        raise ValueError(f"{where}: layout must be one of {LAYOUT_VALUES}, not {value!r}")
    return value


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
            out["notes"] += [f"{x}; they are not applied under header" for x in stored]
        return out
    if needs:
        wanted = set(needs.values())
        if len(wanted) > 1:
            raise LayoutConflict(
                "the layouts they were trained on differ ("
                + "; ".join(f"{who}: {layout}" for who, layout in needs.items())
                + "); they cannot share a server until a request can choose its layout. Serve them separately, or "
                "give --layout to serve all on one (and measure)"
            )
        value = wanted.pop()
        why = "recorded by " + ", ".join(needs)
        if value == "plain" and default == "header":
            out["held_by"] = [who for who, layout in needs.items() if layout == "plain"]
        if value == "header":
            out["notes"] += [f"{x}; they are not applied under header" for x in stored]
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
            + ". Start with --layout header to switch, after registering the tasks again under it (docs/running.md)"
        )
    else:
        line = f"LAYOUT {value} ({decision['why']})"
    return line + "".join(f"\nLAYOUT NOTE: {n}" for n in decision["notes"])
