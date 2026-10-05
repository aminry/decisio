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

Only the engine calls decide it (`EngineHealth.call`): an exception caused by the request (`ValueError`, `TypeError`,
`KeyError`, vLLM's `VLLMClientError`) passes through as a request error, so a malformed request cannot take the server
down; anything else escaping an engine call marks the engine dead. vLLM's own flag for an exited engine-core process
is watched too (`watch`), so a death between requests is caught without waiting for one.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)

EXIT_ENGINE_DEAD = 70  # EX_SOFTWARE: the process ends because its engine died; a restart policy should start it again
REQUEST_ERRORS = (ValueError, TypeError, KeyError)


class EngineDead(RuntimeError):
    """The engine behind this server is dead: every request is refused until a new server starts."""


def is_request_error(e: BaseException) -> bool:
    """An error caused by the request rather than the engine: decisio's own input errors and vLLM's client errors
    (`vllm.exceptions.VLLMClientError`, matched by name so the CPU stand-in needs no vLLM)."""
    return isinstance(e, REQUEST_ERRORS) or any(c.__name__ == "VLLMClientError" for c in type(e).__mro__)


def exit_process(code: int) -> None:
    """Stop the engine's child processes (vLLM's engine core, with `--engine-process separate`) and end this process.
    `os._exit`: the engine is dead, and an orderly shutdown can block on it; the children first, since `os._exit`
    alone orphans the engine core with the GPU held (as `vllm_engine.run_or_die`)."""
    import psutil

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        kids = psutil.Process().children(recursive=True)
        for k in kids:
            k.terminate()
        _, alive = psutil.wait_procs(kids, timeout=5)
        for k in alive:
            k.kill()
        psutil.wait_procs(alive, timeout=5)
    except Exception as e:  # noqa: BLE001 - the process ends either way
        print(f"decisio: stopping the engine's child processes failed: {e!r}", file=sys.stderr, flush=True)
    os._exit(code)


class EngineHealth:
    """The engine's state, shared by every engine of one server (text, image, head)."""

    def __init__(self, exit_code: int = EXIT_ENGINE_DEAD, grace_s: float = 2.0, exit_fn=None):
        self.exit_code, self.grace_s = exit_code, grace_s
        self._exit_fn = exit_fn or exit_process
        self._lock = threading.Lock()
        self.reason: str | None = None
        self.dead_since: float | None = None

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
        self._exit_fn(self.exit_code)

    def check(self) -> None:
        if self.reason is not None:
            raise EngineDead(self.reason)

    def call(self, fn, *args, **kwargs):
        """Run one engine call: refused at once when the engine is dead; a request error passes through; any other
        exception marks the engine dead and is raised as `EngineDead`."""
        self.check()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if is_request_error(e):
                raise
            self.mark_dead(f"{type(e).__name__}: {e}".strip()[:500])
            raise EngineDead(self.reason) from e

    def watch(self, probes, interval_s: float = 1.0) -> threading.Thread | None:
        """Poll each probe (a callable returning the reason the engine is dead, or None) every `interval_s`."""
        probes = [p for p in probes if p is not None]
        if not probes:
            return None

        def run():
            while self.reason is None:
                for p in probes:
                    try:
                        why = p()
                    except Exception as e:  # noqa: BLE001 - a probe that cannot run says nothing about the engine
                        logger.debug("engine probe failed: %r", e)
                        why = None
                    if why:
                        self.mark_dead(why)
                        return
                time.sleep(interval_s)

        t = threading.Thread(target=run, daemon=True, name="decisio-engine-watch")
        t.start()
        return t


def guarded(engine, fn, *args, **kwargs):
    """`fn(*args, **kwargs)` through the engine's health, when the server gave it one (`engine.health`)."""
    health = getattr(engine, "health", None)
    return health.call(fn, *args, **kwargs) if health is not None else fn(*args, **kwargs)
