# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Every thread that calls an in-process vLLM engine runs on the stream the engine was warmed up on
(`decisio.serve.engine_stream`). No GPU and no vLLM: a stand-in for vLLM's thread-local stream and its locked workspace.

E1  the failure, in a stand-in: a thread other than the warm-up's asks the locked workspace for a key under its own
    stream and gets vLLM's assertion (what a W4A16 checkpoint's first request did on the card)
E2  binding the engine's stream first (`engine_stream.bind`) gives that thread the warm-up's key
E3  `engine_health.guarded` binds before every engine call, on whatever thread calls, and the probe thread does too
E4  the engines bind through `bind_stream`, which reads the stream captured when they were built; an engine built
    without one (the CPU stand-in, MLX, a fake) binds nothing and calls as before
E5  `capture` returns None where there is no vLLM, and `bind(None)` does nothing

  uv run pytest -q tests/unit/test_engine_stream.py
"""

import threading
from types import SimpleNamespace

from decisio.serve import engine_stream
from decisio.serve.engine_health import EngineHealth, guarded, run_probe


class FakeVLLM:
    """vLLM's per-thread current stream and a workspace that was locked after the warm-up allocated one lock buffer."""

    def __init__(self):
        self.tls = threading.local()
        self.warm = object()
        self.tls.value = self.warm  # the thread that built and warmed the engine (this one)
        self.allocated = {("marlin_locks", self.warm)}

    def set_stream(self, stream):
        self.tls.value = stream

    def current_stream(self):
        if getattr(self.tls, "value", None) is None:  # a thread's first use makes a stream of its own
            self.tls.value = object()
        return self.tls.value

    def marlin_lock_buffer(
        self,
    ):  # the locked workspace manager's get_persistent for ("marlin_locks", current_stream())
        key = ("marlin_locks", self.current_stream())
        if key not in self.allocated:
            raise AssertionError(f"Workspace is locked but {key!r} was not allocated during warmup")
        return key


def on_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            out["error"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join()
    return out


def test_e1_another_thread_trips_the_locked_workspace():
    v = FakeVLLM()
    assert v.marlin_lock_buffer()[1] is v.warm  # the warm-up's thread
    got = on_thread(v.marlin_lock_buffer)
    assert isinstance(got.get("error"), AssertionError) and "marlin_locks" in str(got["error"])


def test_e2_binding_the_engines_stream_first_gives_it_the_warm_up_key():
    v = FakeVLLM()

    def call():
        engine_stream.bind(v.warm, set_stream=v.set_stream)
        return v.marlin_lock_buffer()

    got = on_thread(call)
    assert got.get("value") == ("marlin_locks", v.warm) and "error" not in got


class FakeEngine:
    def __init__(self, v):
        self.v, self.bound = v, 0

    def bind_stream(self):
        self.bound += 1
        engine_stream.bind(self.v.warm, set_stream=self.v.set_stream)

    def call(self):
        return self.v.marlin_lock_buffer()

    def probe(self):
        self.bind_stream()
        self.v.marlin_lock_buffer()


def test_e3_guarded_binds_before_every_call_and_so_does_the_probe_thread():
    v = FakeVLLM()
    eng = FakeEngine(v)
    assert on_thread(lambda: guarded(eng, eng.call)).get("value") == ("marlin_locks", v.warm)
    health = EngineHealth()
    eng.health = health
    assert on_thread(lambda: guarded(eng, eng.call)).get("value") == ("marlin_locks", v.warm)  # through the health too
    assert eng.bound == 2
    why, _ = run_probe(eng.probe, 5.0)  # the probe runs on a daemon thread of its own
    assert why is None and eng.bound == 3
    bare = SimpleNamespace(call=lambda: 7)  # an engine with no bind_stream (stand-in, MLX): called as before
    assert guarded(bare, bare.call) == 7


def test_e4_the_engines_bind_the_stream_they_captured(monkeypatch):
    from decisio.serve.hidden_engine import HiddenEngine
    from decisio.serve.vllm_engine import LettersEngine

    v = FakeVLLM()
    real = engine_stream.bind
    monkeypatch.setattr(engine_stream, "bind", lambda stream: real(stream, set_stream=v.set_stream))
    for cls in (LettersEngine, HiddenEngine):
        eng = SimpleNamespace(_stream=v.warm)
        got = on_thread(lambda: (cls.bind_stream(eng), v.marlin_lock_buffer())[1])
        assert got.get("value") == ("marlin_locks", v.warm), cls.__name__
    none = SimpleNamespace()  # built without a stream (captured None, or an attribute never set)
    assert on_thread(lambda: LettersEngine.bind_stream(none)).get("error") is None


def test_e5_capture_and_bind_without_vllm():
    engine_stream.capture()  # None where vLLM or CUDA is absent, a stream on a card; it never raises
    engine_stream.bind(None)  # nothing to bind
    seen = []
    engine_stream.bind("s", set_stream=seen.append)
    assert seen == ["s"]
