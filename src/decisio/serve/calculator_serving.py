# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The calculator in the request path (decisio.calculator; docs/calculator.md): which requests get the block, what the
response says about it, and the counts /health reports.

A server shows the block only when its model records the block format it was trained with (a tuned repository's
`defaults.calculator`, `--calculator auto`, the default); a model that records none never sees it, whatever a request
asks: such a request is answered without the block and says so in `x-decisio-warning`. A request may turn the block off
(`"calculator": false`) or bring its own amounts (`"calculator_input"`, rendered by the same code, no parsing).

Within a request, `Decision` holds what was decided and, once the state's context is known, the block itself
(`Annotation`) or why there is none (`Abstain`); `SystemOne.read_state` reads it, so every read of the request (its
questions, a certified task's own read, thinking's second read) sees the same state text. Outside a request
(registering or certifying a task) the server's setting applies to each example's state.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading

# the warnings of a request that asked for the block where none is shown (x-decisio-warning)
NOT_TRAINED = (
    "calculator: no block shown: this server's model was not trained with the calculator block (its repository "
    "records no defaults.calculator), or the server was started with --calculator off"
)
WITH_IMAGES = "calculator: no block shown: a request with images is read without the calculator block"
PACKED = "calculator: no block shown: --mode packed reads no calculator block"
# a request that turns the block off on a server that shows it reads its state as the model was not trained to, so no
# registered task (each fitted on reads with the block) applies to it
TASKS_OFF = (
    '"task": a request with "calculator": false applies no registered task (the tasks of this server were fitted on '
    "reads with the calculator block)"
)

_CURRENT: contextvars.ContextVar[Decision | None] = contextvars.ContextVar("decisio_calculator", default=None)


class Decision:
    """One request's calculator: `on` (the block is shown where the calculator reads the state), the `version`, the
    block made from the request's own amounts (`given`), and once decided the `annotation` (an Annotation or an
    Abstain) for `state` (the request's state object), the `reason` there is none, and a `warning`."""

    __slots__ = ("on", "version", "given", "annotation", "state", "reason", "warning", "asked")

    def __init__(self, on=False, version=None, given=None, reason=None, warning=None, asked=False):
        self.on, self.version, self.given = on, version, given
        self.annotation, self.state = None, None
        self.reason, self.warning, self.asked = reason, warning, asked

    def header_value(self) -> str | None:
        """x-decisio-calculator: `shown; lines=N; version=V`, or `none; reason=R`; None where the server shows no
        block and the request asked for none."""
        from decisio.calculator import Annotation

        if isinstance(self.annotation, Annotation):
            return f"shown; lines={self.annotation.lines}; version={self.annotation.version}"
        if not self.on and not self.asked and self.reason in (None, "off"):
            return None
        return f"none; reason={self.reason or 'unknown'}"


def request_value(value):
    """A request's `"calculator"`: true, false or null (ValueError otherwise)."""
    if value is not None and not isinstance(value, bool):
        raise ValueError('"calculator" must be true, false or null')
    return value


def decide(server_version, want, raw_input, *, images: bool, packed: bool) -> Decision:
    """What a request gets. `server_version`: the format the server shows (None: none); `want`: the request's
    `"calculator"`; `raw_input`: its `"calculator_input"` (validated strictly here, whatever the server shows:
    decisio.calculator.CalculatorInputError, a 422, names the field)."""
    from decisio.calculator import VERSIONS, from_input

    given = None
    if raw_input is not None:
        given = from_input(raw_input, version=server_version or VERSIONS[0])
    asked = want is True or raw_input is not None
    if want is False:
        return Decision(reason="off-by-request", asked=False)
    if server_version is None:
        return Decision(
            reason="off" if not asked else "not-trained", warning=NOT_TRAINED if asked else None, asked=asked
        )
    if images:
        return Decision(reason="images", warning=WITH_IMAGES if asked else None, asked=asked)
    if packed:
        return Decision(reason="packed", warning=PACKED if asked else None, asked=asked)
    return Decision(on=True, version=server_version, given=given, asked=asked)


@contextlib.contextmanager
def deciding(decision: Decision | None):
    """Inside the block, `current()` is this request's decision."""
    token = _CURRENT.set(decision)
    try:
        yield decision
    finally:
        _CURRENT.reset(token)


def current() -> Decision | None:
    return _CURRENT.get()


class Counts:
    """/health systemone.calculator: the version shown, how many requests were read with it on, how many got a block,
    and why the others did not (one count per reason). Never any state text."""

    def __init__(self, version):
        self.version = version
        self._lock = threading.Lock()
        self.requests = self.shown = self.from_input = 0
        self.none: dict[str, int] = {}

    def note(self, decision: Decision) -> None:
        from decisio.calculator import Annotation

        with self._lock:
            if decision.on:
                self.requests += 1
            if isinstance(decision.annotation, Annotation):
                self.shown += 1
                self.from_input += decision.given is not None
            elif decision.on or decision.asked or decision.reason == "off-by-request":
                r = decision.reason or "unknown"
                self.none[r] = self.none.get(r, 0) + 1

    def facts(self) -> dict:
        with self._lock:
            return {
                "version": self.version,
                "requests": self.requests,
                "shown": self.shown,
                "from_input": self.from_input,
                "none": dict(sorted(self.none.items())),
            }
