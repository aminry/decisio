# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""`--concurrency batched`: the requests waiting for the engine are scored together, in one engine call.

The served default, `--concurrency serial`, takes one request at a time through the engine (`LettersEngine._lock`), so
a request's rows are always scheduled alone and its answer is a function of the request: the same request gives the
same bytes whatever else the server is sent. That is what the certificates promise and what the board's pairs compare.

Batched mode trades that for throughput. One engine thread takes every request waiting at that moment (up to
`max_requests`, those of the oldest request's adapter) and scores all their rows in one engine call, so concurrent
requests share forward passes. Nothing waits on purpose: a request that arrives at an idle server is taken at once,
alone, and goes through the same engine call as a serial server's, so a batch of one is the serial answer byte for
byte, and single-stream latency is unchanged (plus a thread hand-off). A request that arrives while a call runs waits
for it, as it would behind the lock, and goes in the next call with every other request that arrived meanwhile.

What drifts: a question scored beside other requests' rows can differ from the same question alone, because vLLM's
kernels choose their tiles and reduction orders by batch shape (measured on the 585 W card, RLCD
experiments/2026-10-04_t3_vllm_batch_composition: the 31B changed no choice, largest probability difference 0.0046;
the 12B 3 to 5 choices; the Qwen base 21 choices). So an answer depends on the traffic it arrived with. The server
says so: the mode is in `/health` and in the serving-stack fingerprint, so a certificate made under one mode is not
served as certified under the other (`concurrency_changed`), and the batch an answer was scored in is in its timing
(`batch`).

What does not change: within one engine call the rows are the ones a serial server builds (the same tokenization,
padding and labels); the boundary registrations of each request are deferred to that request's own response
(decisio.serve.boundary, the ticket the request was given), and every due registration is still sent before the next
call's questions; think mode, the image route and the hidden-state readout stay one request at a time on the engine's
lock, which is first come, first served (`FairLock`), so they take the engine between batched calls whatever the load.
A certified task's reads, and a task's registration and certification reads, take the serial path on a batched server
too (`serial_reads`): a certificate is made and served on reads that are a function of the request alone.

Standard library only.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import copy
import threading
import time
from collections import Counter, deque

MODES = ("serial", "batched")
MAX_REQUESTS = 16  # --batch-max-requests' default: the most requests in one engine call, counted in requests

# reads that must be a function of the request alone on any server: a certified task's answers, a task's registration
# and certification reads (decisio.serve.systemone). Inside `serial_reads()` a batched server takes them one at a time,
# as a serial server does, so a certificate is made and served on reads no other traffic shares
_SERIAL: contextvars.ContextVar[bool] = contextvars.ContextVar("decisio_serial_reads", default=False)


@contextlib.contextmanager
def serial_reads():
    token = _SERIAL.set(True)
    try:
        yield
    finally:
        _SERIAL.reset(token)


def reads_serial() -> bool:
    return _SERIAL.get()


class FairLock:
    """The engine's lock: first come, first served. `threading.Lock` is not fair: the engine thread of a batched server
    releases it and takes it back at once, so a thinking request, an image request or the hidden-state readout waiting
    on it could wait for ever under constant load. Here a release hands the lock to the oldest waiter. The interface is
    `threading.Lock`'s (acquire with blocking and timeout, release, the context manager, locked)."""

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._held = False
        self._queue: deque = deque()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        with self._cond:
            if not self._held and not self._queue:
                self._held = True
                return True
            if not blocking:
                return False
            me = object()
            self._queue.append(me)
            deadline = None if timeout is None or timeout < 0 else time.monotonic() + timeout
            while self._held or self._queue[0] is not me:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    self._queue.remove(me)
                    self._cond.notify_all()  # the next in line may be free to take it now
                    return False
                self._cond.wait(left)
            self._queue.popleft()
            self._held = True
            return True

    def release(self) -> None:
        with self._cond:
            if not self._held:
                raise RuntimeError("release of an unheld FairLock")
            self._held = False
            self._cond.notify_all()

    def locked(self) -> bool:
        with self._cond:
            return self._held

    def waiting(self) -> int:
        with self._cond:
            return len(self._queue)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def is_engine_failure(engine, error) -> bool:
    """Whether a failed engine call failed for the engine (it is dead, or the error is not one a request causes) rather
    than for one of its requests."""
    from decisio.serve.engine_health import EngineDead, client_error

    health = getattr(engine, "health", None)
    if isinstance(error, EngineDead) or (health is not None and health.dead):
        return True
    return not (isinstance(error, (KeyError, ValueError, AssertionError)) or client_error(error) is not None)


class Waiting:
    """One `answer_many` call waiting for the engine: its requests, adapter and ticket, and, once scored, its result."""

    __slots__ = ("requests", "adapter", "ticket", "behind", "arrived", "done", "result", "error")

    def __init__(self, requests, adapter, ticket, behind):
        self.requests, self.adapter, self.ticket, self.behind = requests, adapter, ticket, behind
        self.arrived = time.perf_counter()
        self.done = threading.Event()
        self.result = None
        self.error: BaseException | None = None


class Combiner:
    """The engine thread of a batched server. `submit` is called from each request's thread and blocks until its
    requests were scored; the thread runs `engine._combined_call(batch)` under the engine's lock."""

    CLOSE_WAIT_S = 60.0  # at exit, the wait for an engine call in progress (as decisio.serve.boundary.Registrar)

    def __init__(self, engine, max_requests: int = MAX_REQUESTS):
        if max_requests < 1:
            raise ValueError("--batch-max-requests must be at least 1")
        self.engine, self.max_requests = engine, int(max_requests)
        self._cond = threading.Condition()
        self._waiting: deque[Waiting] = deque()
        self._thread: threading.Thread | None = None
        self._closed = False
        self.calls = 0
        self.requests = 0
        self.sizes: Counter = Counter()  # requests per engine call -> calls

    def facts(self) -> dict:
        """What /health reports: the mode, the cap, and the batches seen (requests per engine call)."""
        with self._cond:
            return {
                "mode": "batched",
                "max_requests": self.max_requests,
                "calls": self.calls,
                "requests": self.requests,
                "waiting": len(self._waiting),
                "largest": max(self.sizes, default=0),
                "requests_per_call": {str(k): v for k, v in sorted(self.sizes.items())},
            }

    def submit(self, requests, adapter, ticket, behind=False):
        w = Waiting(requests, adapter, ticket, behind)
        with self._cond:
            if self._closed:
                raise RuntimeError("the server is shutting down: no engine call starts any more")
            self._waiting.append(w)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True, name="decisio-combiner")
                self._thread.start()
                atexit.register(self.close)
            self._cond.notify_all()
        w.done.wait()
        if w.error is not None:
            # each waiter raises its own copy: several threads raising one object would interleave its traceback
            raise copy.copy(w.error) from w.error
        return w.result

    def take(self) -> list[Waiting]:
        """The next engine call's requests (the caller holds the engine's lock): the oldest waiting call's adapter, and
        every waiting call of that adapter, oldest first, while the requests stay within `max_requests` (the first is
        always taken, whatever its size)."""
        with self._cond:
            if not self._waiting:
                return []
            adapter = self._waiting[0].adapter
            batch, keep, n = [], deque(), 0
            while self._waiting:
                w = self._waiting.popleft()
                if w.adapter == adapter and (not batch or n + len(w.requests) <= self.max_requests):
                    batch.append(w)
                    n += len(w.requests)
                else:
                    keep.append(w)
            self._waiting = keep
            return batch

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._waiting and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
            with self.engine._lock:
                batch = self.take()  # after the lock: what arrived while the last call ran goes in this one
                if not batch:
                    continue
                try:
                    results = self.engine._combined_call(batch)
                    for w, r in zip(batch, results):
                        w.result = r
                except BaseException as e:  # noqa: BLE001
                    if len(batch) == 1 or is_engine_failure(self.engine, e):
                        for w in batch:  # the engine failed: every request of the call gets the failure
                            w.error = e
                    else:
                        # a request's own error (a state over the context, a question that merges with the state)
                        # fails that request only, as on a serial server: each is scored again alone
                        for w in batch:
                            try:
                                (w.result,) = self.engine._combined_call([w])
                            except BaseException as one:  # noqa: BLE001
                                w.error = one
                finally:
                    with self._cond:
                        self.calls += 1
                        n = sum(len(w.requests) for w in batch)
                        self.requests += n
                        self.sizes[n] += 1
                    for w in batch:
                        w.done.set()

    def close(self) -> None:
        """At exit: no call starts any more, the waiting requests fail, and a call in progress is waited for (at most
        CLOSE_WAIT_S): the thread is a daemon, and an interpreter that shuts down inside vLLM's engine aborts."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            waiting, self._waiting = list(self._waiting), deque()
            self._cond.notify_all()
            thread = self._thread
        for w in waiting:
            w.error = RuntimeError("the server is shutting down: this request was not scored")
            w.done.set()
        if thread is not None and self.engine._lock.acquire(timeout=self.CLOSE_WAIT_S):
            self.engine._lock.release()
        if thread is not None:
            thread.join(timeout=1.0)
