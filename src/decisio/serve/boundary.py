# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A state's boundary registered after the response (`--register-boundary after`, the default on the Gemma bases).

On the Gemma bases vLLM keeps only a finished request's latest sliding-window checkpoint, which lies inside its
question, so a later, different question about the state would read the whole state again. decisio registers the
state's boundary with a warm-up of the state and one token (decisio.families register_state_boundary). In 0.8.1 the
warm-up went before a single question on a new state, and the caller waited for it (on a 400 W card, 12B +11 to +59 ms,
31B +23 to +72 ms at 300 to 3,000 tokens; RLCD experiments/2026-10-06_t3_latency_and_gpu_tier). Here it goes after:

- the question is answered first, reading the state fresh, and its warm-up is kept pending (`Registrar.defer`);
- the warm-up becomes due when the request's response has been handed to the server (`AfterResponse`, the last body
  byte), or when the engine call returns for a caller without a server (`Ticket`);
- a background thread sends due warm-ups as soon as it can take the engine's lock, and every request that takes the
  lock first sends any that are still due (`Registrar.run_due`), so registrations go before every later question;
- a request about a state whose warm-up is pending but not yet due (a second question sent before the first one's
  response was out) sends the warm-up itself, before its question, as 0.8.1 did.

So a follow-up question sent after the first response reads the state from the cache: it may wait, at most for the
engine call in progress and the warm-up, which reads the whole state again (the engine kept no checkpoint at the
boundary). It reads the state fresh only when its registration was dropped (more than `PENDING_KEPT` pending), failed,
or was evicted; `/health` counts the first two.
"""

from __future__ import annotations

import atexit
import contextvars
import logging
import threading
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

# the registrations deferred while answering the current request; None outside a request (decisio.serve.vllm_engine
# sets one around each engine call when the server has not)
_TICKET: contextvars.ContextVar[Ticket | None] = contextvars.ContextVar("decisio_boundary_ticket", default=None)


def current_ticket() -> Ticket | None:
    return _TICKET.get()


def open_ticket() -> tuple[Ticket, contextvars.Token]:
    ticket = Ticket()
    return ticket, _TICKET.set(ticket)


def close_ticket(ticket: Ticket, token: contextvars.Token) -> None:
    _TICKET.reset(token)
    ticket.release()


class Ticket:
    """The registrations one request deferred; they become due when it is released (its response is out)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items: list[tuple[Registrar, bytes]] = []
        self.released = False

    def add(self, registrar: Registrar, key: bytes) -> None:
        with self._lock:
            if not self.released:
                self._items.append((registrar, key))
                return
        registrar.due([key])

    def release(self) -> None:
        with self._lock:
            self.released = True
            items, self._items = self._items, []
        by: dict[int, tuple[Registrar, list[bytes]]] = {}
        for registrar, key in items:
            by.setdefault(id(registrar), (registrar, []))[1].append(key)
        for registrar, keys in by.values():
            registrar.due(keys)


class AfterResponse:
    """ASGI middleware: the registrations a request deferred become due once its last response byte has been handed to
    the server (or when it ends without one), so the warm-ups never delay the response they follow."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        ticket, token = open_ticket()
        released = False

        async def sent(message):
            nonlocal released
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False) and not released:
                released = True
                ticket.release()

        try:
            await self.app(scope, receive, sent)
        finally:
            _TICKET.reset(token)
            if not released:
                ticket.release()


class Registrar:
    """One engine's pending boundary registrations: (state key) -> (warm-up token ids, adapter, due), oldest first."""

    PENDING_KEPT = 256  # beyond this many pending, the oldest is dropped: a later question on its state reads it again
    CLOSE_WAIT_S = 60.0  # at exit, the wait for a warm-up in progress (a 31,000-token state took 9.2 s on the 31B)

    def __init__(self, engine, background: bool = True):
        self.engine, self.background = engine, background
        self._cond = threading.Condition()
        self._pending: OrderedDict[bytes, list] = OrderedDict()
        self._due = 0
        self._thread: threading.Thread | None = None
        self._closed = False
        self.running = False  # the background thread is sending a registration now (the engine's lock held)
        self.counts = {"deferred": 0, "registered": 0, "dropped": 0, "failed": 0}

    def facts(self) -> dict:
        with self._cond:
            return {"pending": len(self._pending), **self.counts}

    def defer(self, key: bytes, warm, adapter) -> None:
        """Keep the warm-up of a state answered without it (the engine's lock held); due when the request's ticket is
        released."""
        with self._cond:
            if key in self._pending:
                self._pending.move_to_end(key)
            else:
                self._pending[key] = [list(warm), adapter, False]
                self.counts["deferred"] += 1
            while len(self._pending) > self.PENDING_KEPT:
                _, (_, _, due) = self._pending.popitem(last=False)
                self._due -= due
                self.counts["dropped"] += 1
        ticket = current_ticket()
        if ticket is not None:
            ticket.add(self, key)
        else:
            self.due([key])

    def take(self, key: bytes) -> bool:
        """Remove a pending registration the caller sends itself, before its question; whether there was one."""
        with self._cond:
            item = self._pending.pop(key, None)
            if item is not None:
                self._due -= item[2]
            return item is not None

    def due(self, keys) -> None:
        with self._cond:
            for key in keys:
                item = self._pending.get(key)
                if item is not None and not item[2]:
                    item[2] = True
                    self._due += 1
            if self._due and not self._closed:
                self._cond.notify_all()
                if self.background and self._thread is None:
                    self._thread = threading.Thread(target=self._run, daemon=True, name="decisio-boundary-registrar")
                    self._thread.start()
                    atexit.register(self.close)

    def run_due(self) -> tuple[int, float]:
        """Send every due warm-up, in one engine call per adapter; the engine's lock must be held. (states registered,
        ms)."""
        with self._cond:
            items = [(k, w, a) for k, (w, a, d) in self._pending.items() if d]
            for k, _, _ in items:
                del self._pending[k]
            self._due = 0
        if not items:
            return 0, 0.0
        t = time.perf_counter()
        by: dict = {}
        for k, w, a in items:
            by.setdefault(a, []).append((k, w))
        done = 0
        for adapter, group in by.items():
            try:
                self.engine.send_warm([w for _, w in group], adapter)
            except Exception as e:  # noqa: BLE001 - another request's registration never fails the caller's request
                health = getattr(self.engine, "health", None)
                with self._cond:
                    self.counts["failed"] += len(group)
                    if health is not None and health.dead:  # nothing more will be sent
                        self.counts["dropped"] += len(self._pending)
                        self._pending.clear()
                        self._due = 0
                print(
                    f"decisio: registering {len(group)} state boundaries after their responses failed "
                    f"({type(e).__name__}: {e}); a later question on those states reads them again",
                    flush=True,
                )
                continue
            self.engine.boundaries_registered([k for k, _ in group])
            done += len(group)
        with self._cond:
            self.counts["registered"] += done
        return done, (time.perf_counter() - t) * 1000

    def close(self) -> None:
        """At exit (registered with atexit when the thread starts): no warm-up starts any more, the pending ones are
        dropped and counted, and a warm-up in progress is waited for, at most CLOSE_WAIT_S. The registrar's thread is a
        daemon, and an interpreter that shuts down while it is inside vLLM's engine aborts the process (on a card: exit
        by signal 6, "terminate called without an active exception", seen with the 31B)."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self.counts["dropped"] += len(self._pending)
            self._pending.clear()
            self._due = 0
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and self.engine._lock.acquire(timeout=self.CLOSE_WAIT_S):
            self.engine._lock.release()  # the warm-up in progress, if any, has ended; none starts after it
        if thread is not None:
            thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._due and not self._closed:
                    self._cond.wait()
                if self._closed:
                    self._thread = None
                    return
            health = getattr(self.engine, "health", None)
            if health is not None and health.dead:
                with self._cond:
                    self.counts["dropped"] += len(self._pending)
                    self._pending.clear()
                    self._due = 0
                    self._thread = None
                return
            with self.engine._lock:
                if self._closed:
                    return
                self.running = True
                try:
                    self.run_due()
                finally:
                    self.running = False
