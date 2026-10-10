# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Whether the engine behind the server is alive, and what the server does when it dies.

A vLLM engine dies in two ways: an exception escapes its forward pass (a CUDA error, an out-of-memory error), or its
engine-core process exits (with `--engine-process separate`, vLLM's own arrangement). Either way it serves nothing
again: in the separate arrangement every later request fails, and in-process every later request hangs. So once the
engine is dead the server

- answers every request at once with 503 and the reason, without touching the engine again;
- reports it on `/health` (503, `"engine": "dead"`), so a health check sees it;
- ends its process with exit code 70 (EX_SOFTWARE) after a short grace period, so that a restart policy (compose's
  `restart: unless-stopped`, a systemd `Restart=`, a Kubernetes pod) starts a new server.

The engine calls decide it (`EngineHealth.call`), and a death is confirmed before it is declared, so that a request
which trips a bug on decisio's side cannot take a healthy server down:

- an exception caused by the request (`ValueError`, `TypeError`, `KeyError`, vLLM's `VLLMClientError`) passes through;
- the engine is dead at once when vLLM says so: its `EngineDeadError` (matched by name), or its flags (`engine_flag`:
  the client's `engine_dead`, set when the engine-core process is gone, and an `errored` flag where the engine has one);
- any other exception fails that request (a 500, as before this module), and the engine is then sent one probe (its
  `probe()`, a forward pass over a one-token prompt) with a timeout of `probe_timeout_s`, or 3 times its slowest recent
  successful call when that is longer (a slow engine is not a dead one); only a probe that fails or does not answer in
  time marks the engine dead. A probe that hangs is how an in-process engine shows its death; a probe still waiting
  when the engine is declared dead by another path ends at once.

After the grace period the process exits once no request is in flight (the server counts them, `request_started` and
`request_finished`), and at the latest `DRAIN_S` later, so a request inside an engine call when the engine is declared
dead gets its answer rather than a dropped connection, unless the engine never returns it.

vLLM's flags are also watched between requests (`watch`), so a death between requests is caught without waiting for one.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

EXIT_ENGINE_DEAD = 70  # EX_SOFTWARE: the process ends because its engine died; a restart policy should start it again
REQUEST_ERRORS = (ValueError, TypeError, KeyError)
PROBE_TIMEOUT_S = 5.0  # a one-token forward pass takes milliseconds; one that has not answered in 5 s never will
PROBE_SLOWEST_FACTOR = 3.0  # ... unless the engine's own recent calls were slower: then 3 times the slowest of them
DRAIN_S = 8.0  # after the grace period, the exit waits at most this long for the requests in flight to be answered


class EngineDead(RuntimeError):
    """The engine behind this server is dead: every request is refused until a new server starts."""


def _named(e: BaseException, name: str) -> bool:
    """`e` is an instance of a class called `name` (vLLM's classes are matched by name, so the CPU stand-in and the MLX
    backend need no vLLM)."""
    return any(c.__name__ == name for c in type(e).__mro__)


def is_request_error(e: BaseException) -> bool:
    """An error caused by the request rather than the engine: decisio's own input errors and vLLM's client errors
    (`vllm.exceptions.VLLMClientError`)."""
    return isinstance(e, REQUEST_ERRORS) or _named(e, "VLLMClientError")


def client_error(e: BaseException) -> tuple[int, str] | None:
    """The HTTP status and detail for an error vLLM says the request caused, or None for any other exception.

    vLLM 0.30.0 raises `VLLMValidationError` (a `VLLMClientError`, not a ValueError) when the rendered prompt is over
    its context, so the routes' `except ValueError` never saw it and the caller got a bare 500. A validation error, or
    an unprocessable entity (an image URL that does not load), is a 422, as for any input over capacity; any other
    client error is a 400, the status for a request the engine cannot answer (docs/api.md)."""
    if not _named(e, "VLLMClientError"):
        return None
    message = str(e)
    if _named(e, "VLLMValidationError") or _named(e, "VLLMUnprocessableEntityError"):
        if "maximum context length" in message:
            return 422, (
                "the rendered prompt (the state with one question) is longer than the context the server was "
                f"started with; shorten the state or the question and send it again. vLLM says: {message}"
            )
        return 422, f"the engine rejected the request: {message}"
    return 400, f"the engine cannot answer the request: {message}"


def is_engine_dead_error(e: BaseException) -> bool:
    """vLLM's own word that its engine is dead (`vllm.v1.engine.exceptions.EngineDeadError`, "Unrecoverable")."""
    return _named(e, "EngineDeadError")


def _llm_engine(engine):
    return getattr(getattr(engine, "llm", None), "llm_engine", None)


def engine_flag(engine) -> str | None:
    """Why vLLM says the engine is dead, from its flags, or None (also for an engine without any: stand-in, MLX)."""
    llm_engine = _llm_engine(engine)
    if llm_engine is None:
        return None
    resources = getattr(getattr(llm_engine, "engine_core", None), "resources", None)
    if getattr(resources, "engine_dead", False) is True:  # vllm.v1.engine.core_client: the core process is gone
        return "vLLM's engine-core process exited"
    if getattr(llm_engine, "errored", False) is True:  # AsyncLLM has it; vLLM 0.30.0's LLMEngine does not
        return "vLLM reports its engine errored"
    return None


def run_probe(probe, timeout_s: float, stop=None) -> tuple[str | None, float]:
    """(None when `probe()` returned within `timeout_s`, else why not; the seconds it took or was given). The probe runs
    in a daemon thread, so a hanging engine holds that thread and not the caller. `stop()` true ends the wait at once
    (the engine was declared dead meanwhile, by another call or the watch thread)."""
    out: dict = {}

    def run():
        try:
            probe()
        except BaseException as e:  # noqa: BLE001 - any failure of the probe is the engine failing it
            out["error"] = e

    t0 = time.perf_counter()
    t = threading.Thread(target=run, daemon=True, name="decisio-engine-probe")
    t.start()
    while t.is_alive() and time.perf_counter() - t0 < timeout_s:
        t.join(min(0.05, max(0.0, timeout_s - (time.perf_counter() - t0))))
        if t.is_alive() and stop is not None and stop():
            return "the engine was declared dead while it was being probed", time.perf_counter() - t0
    took = time.perf_counter() - t0
    if t.is_alive():
        return f"the engine did not answer a probe within {timeout_s:g} s", took
    if "error" in out:
        e = out["error"]
        return f"the engine failed a probe ({type(e).__name__}: {e})"[:500], took
    return None, took


def exit_process(code: int) -> None:
    """Stop the engine's child processes (vLLM's engine core, with `--engine-process separate`) and end this process.
    `os._exit`: the engine is dead, and an orderly shutdown can block on it; the children first, since `os._exit`
    alone orphans the engine core with the GPU held (as `vllm_engine.run_or_die`)."""
    import psutil

    sys.stdout.flush()
    sys.stderr.flush()
    try:  # killed, not terminated: the engine is dead, and multiprocessing's resource tracker ignores SIGTERM
        kids = psutil.Process().children(recursive=True)
        for k in kids:
            k.kill()
        psutil.wait_procs(kids, timeout=10)
    except Exception as e:  # noqa: BLE001 - the process ends either way
        print(f"decisio: stopping the engine's child processes failed: {e!r}", file=sys.stderr, flush=True)
    os._exit(code)


class EngineHealth:
    """The engine's state, shared by every engine of one server (text, image, head)."""

    def __init__(
        self,
        exit_code: int = EXIT_ENGINE_DEAD,
        grace_s: float = 2.0,
        probe_timeout_s: float = PROBE_TIMEOUT_S,
        exit_fn=None,
        drain_s: float = DRAIN_S,
    ):
        self.exit_code, self.grace_s, self.probe_timeout_s, self.drain_s = exit_code, grace_s, probe_timeout_s, drain_s
        self._exit_fn = exit_fn or exit_process
        self._lock = threading.Lock()
        self.reason: str | None = None
        self.dead_since: float | None = None
        self._recent = deque(maxlen=50)  # the durations of the last successful engine calls, in seconds
        self._in_flight = 0

    def request_started(self) -> None:
        with self._lock:
            self._in_flight += 1

    def request_finished(self) -> None:
        with self._lock:
            self._in_flight -= 1

    def calibrate(self, engines) -> dict:
        """Time one probe of each engine at start-up and count it as a recent call, so a server's first request already
        gives a slow engine's probe its time (seen with the stand-in's forward passes made 6 s slow: the first request
        that tripped a bug had nothing to compare with). Returns each probe's milliseconds by engine class."""
        took = {}
        for e in engines:
            probe = getattr(e, "probe", None)
            if probe is None:
                continue
            t0 = time.perf_counter()
            probe()
            took[type(e).__name__] = round((time.perf_counter() - t0) * 1000, 1)
            self._recent.append(took[type(e).__name__] / 1000)
        return took

    def probe_deadline_s(self) -> float:
        """The probe's timeout: `probe_timeout_s`, or 3 times the slowest recent successful call when that is longer."""
        return max(self.probe_timeout_s, PROBE_SLOWEST_FACTOR * max(self._recent, default=0.0))

    @property
    def dead(self) -> bool:
        return self.reason is not None

    def mark_dead(self, reason: str) -> None:
        """Record the death once, and end the process after the grace period (so the requests failing now get their
        answer first)."""
        with self._lock:
            if self.reason is not None:
                return
            self.reason, self.dead_since = reason, time.time()
        print(f"ENGINE DEAD {reason}; exiting with code {self.exit_code} in {self.grace_s} s", file=sys.stderr)
        threading.Thread(target=self._exit_later, daemon=True, name="decisio-engine-dead-exit").start()

    def _exit_later(self) -> None:
        time.sleep(self.grace_s)
        deadline = time.monotonic() + self.drain_s
        while self._in_flight > 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        if self._in_flight == 0:
            time.sleep(0.1)  # the last answer's bytes leave the socket
        self._exit_fn(self.exit_code)

    def check(self) -> None:
        if self.reason is not None:
            raise EngineDead(self.reason)

    def confirm(self, engine, e: BaseException) -> str | None:
        """After `e` escaped one of `engine`'s calls: why the engine is dead, or None when it still answers."""
        what = f"{type(e).__name__}: {e}".strip()[:500]
        if is_engine_dead_error(e):
            return what
        flag = engine_flag(engine)
        if flag:
            return f"{what}; {flag}"
        probe = getattr(engine, "probe", None)
        if probe is None:
            return None
        # a request waiting on its probe gets its answer as soon as the engine is declared dead by another path, not
        # the dropped connection of the exit that follows (seen in the engine-death tests under load)
        failed, took = run_probe(probe, self.probe_deadline_s(), stop=lambda: self.dead)
        if failed:
            return f"{what}; {failed}"
        print(
            f"decisio: a request failed with {what}; the engine answered a probe in {took * 1000:.0f} ms, so it is "
            "alive and the server stays up",
            file=sys.stderr,
            flush=True,
        )
        return None

    def call(self, engine, fn, *args, **kwargs):
        """Run one of `engine`'s calls: refused at once when the engine is dead; a request error passes through; any
        other exception is raised as it is (a 500) when the engine is confirmed alive, and as `EngineDead` when not."""
        self.check()
        t0 = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
        except Exception as e:
            if is_request_error(e):
                raise
            if not self.dead:  # a death recorded meanwhile by another call stands
                why = self.confirm(engine, e)
                if why is None:
                    raise
                self.mark_dead(why)
            raise EngineDead(self.reason) from e
        self._recent.append(time.perf_counter() - t0)
        return out

    def watch(self, engines, interval_s: float = 1.0) -> threading.Thread | None:
        """Read the vLLM engines' flags (`engine_flag`) every `interval_s`; None when no engine has them."""
        engines = [e for e in engines if _llm_engine(e) is not None]
        if not engines:
            return None

        def run():
            while self.reason is None:
                for e in engines:
                    try:
                        why = engine_flag(e)
                    except Exception as err:  # noqa: BLE001 - a flag that cannot be read says nothing about the engine
                        logger.debug("reading the engine's flags failed: %r", err)
                        why = None
                    if why:
                        self.mark_dead(why)
                        return
                time.sleep(interval_s)

        t = threading.Thread(target=run, daemon=True, name="decisio-engine-watch")
        t.start()
        return t


class InFlight:
    """ASGI middleware: counts the requests in flight on `health` (EngineHealth.request_started / request_finished),
    a request finishing once its last response byte has been handed to the server, so the exit after a death waits
    for the answers already under way."""

    def __init__(self, app, health):
        self.app, self.health = app, health

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        self.health.request_started()
        finished = False

        async def sent(message):
            nonlocal finished
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False) and not finished:
                finished = True
                self.health.request_finished()

        try:
            await self.app(scope, receive, sent)
        finally:
            if not finished:
                self.health.request_finished()


def guarded(engine, fn, *args, **kwargs):
    """`fn(*args, **kwargs)`, one of `engine`'s calls, through its health when the server gave it one."""
    health = getattr(engine, "health", None)
    return health.call(engine, fn, *args, **kwargs) if health is not None else fn(*args, **kwargs)
