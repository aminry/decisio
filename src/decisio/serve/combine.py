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
call's questions; think mode and the hidden-state readout stay one request at a time on the engine's lock.

Standard library only.
"""

from __future__ import annotations

import atexit
import threading
import time
from collections import Counter, deque

MODES = ("serial", "batched")
MAX_REQUESTS = 16  # --batch-max-requests' default: the most requests in one engine call


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
            raise w.error
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
                except BaseException as e:  # noqa: BLE001 - every request of the call gets the call's failure
                    for w in batch:
                        w.error = e
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
