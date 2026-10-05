# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The server on the CPU stand-in with an engine that dies on command (tests/unit/test_engine_death.py).

    python tests/unit/dying_engine.py ARRANGEMENT CONTROL <server flags>

ARRANGEMENT
  in        as vLLM in the server's process: a forward pass that raises leaves the engine broken, and every later
            forward pass hangs (seen on a card with --engine-process in, 2026-10-05)
  separate  as vLLM's engine-core process: every forward pass goes through a real child process, printed as
            `ENGINE CORE PID <pid>`. Once the child is gone a forward pass raises EngineDeadError (a VLLMServerError,
            as vLLM's client raises it) and vLLM's flag `llm.llm_engine.engine_core.resources.engine_dead` is set;
            a forward pass that raises takes the child down with it
CONTROL     a file read at each forward pass: absent or empty runs it; `raise` raises; `hold` waits until it changes;
            `break` stops the engine without an error (in: this and every later forward pass hangs; separate: the
            child is killed)
POISON      a prompt containing the tokens of " xyzzy" raises IndexError before the forward pass, whatever the engine's
            state: a bug on decisio's side, tripped by one request while the engine is healthy

The model is the stand-in's own (warm-up runs before the stub goes in); only its forward pass is wrapped.
"""

import multiprocessing as mp
import sys
import threading
import time
import types
from pathlib import Path


class VLLMError(Exception):
    pass


class VLLMServerError(VLLMError):
    pass


class EngineDeadError(VLLMServerError):
    """As vllm.exceptions.EngineDeadError: what vLLM's client raises once its engine-core process is gone."""


def core(conn):
    """The engine-core process: answers every forward pass until it is killed."""
    while True:
        conn.send(conn.recv())


class Resources:
    """vLLM's client resources: `engine_dead` once the engine-core process is gone."""

    def __init__(self, proc):
        self.proc = proc

    @property
    def engine_dead(self):
        return not self.proc.is_alive()


class DyingModel:
    def __init__(self, model, arrangement, control, poison, proc=None, conn=None):
        self.model, self.arrangement, self.control, self.proc, self.conn = model, arrangement, control, proc, conn
        self.poison, self.broken = poison, False

    def __getattr__(self, name):
        return getattr(self.model, name)

    def command(self):
        return self.control.read_text().strip() if self.control.exists() else ""

    def alive_or_raise(self):
        if self.proc is not None and not self.proc.is_alive():
            raise EngineDeadError("EngineCore encountered an issue. See stack trace (above) for the root cause.")

    def poisoned(self, args, kwargs):
        x = args[0] if args else kwargs.get("input_ids")
        ids, n = x[0].tolist(), len(self.poison)
        return any(ids[i : i + n] == self.poison for i in range(len(ids) - n + 1))

    def __call__(self, *args, **kwargs):
        if self.poisoned(args, kwargs):
            raise IndexError("list index out of range (an injected bug on decisio's side)")
        if self.broken:
            threading.Event().wait()  # in-process, after a failed forward pass: every later one hangs
        cmd = self.command()
        while cmd == "hold":
            self.alive_or_raise()
            time.sleep(0.05)
            cmd = self.command()
        if cmd == "break":
            if self.proc is not None:
                self.proc.kill()
                self.proc.join()
                self.alive_or_raise()
            self.broken = True
            threading.Event().wait()
        if cmd == "raise":
            if self.proc is not None:
                self.proc.kill()
                self.proc.join()
                self.alive_or_raise()
            self.broken = True
            raise RuntimeError("CUDA error: an illegal memory access was encountered (injected)")
        if self.proc is not None:
            self.alive_or_raise()
            try:
                self.conn.send(1)
                self.conn.recv()
            except (EOFError, OSError) as e:
                raise EngineDeadError("EngineCore encountered an issue.") from e
        return self.model(*args, **kwargs)


def main():
    arrangement, control, flags = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    assert arrangement in ("in", "separate"), arrangement
    proc = conn = None
    if arrangement == "separate":
        ctx = mp.get_context("spawn")
        conn, child = ctx.Pipe()
        proc = ctx.Process(target=core, args=(child,), daemon=True)
        proc.start()
        print(f"ENGINE CORE PID {proc.pid}", flush=True)

    from decisio.serve import hf_letters, vllm_engine

    init = hf_letters.HFLettersEngine.__init__

    def dying_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        poison = self.tok.encode(" xyzzy", add_special_tokens=False)
        self.model = DyingModel(self.model, arrangement, control, poison, proc, conn)
        if proc is not None:
            ns = types.SimpleNamespace
            self.llm = ns(llm_engine=ns(engine_core=ns(resources=Resources(proc))))

    hf_letters.HFLettersEngine.__init__ = dying_init
    sys.argv = ["decisio", *flags]
    vllm_engine.run_or_die(vllm_engine.main)


if __name__ == "__main__":
    main()
