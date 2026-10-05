# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A dead engine is reported, refused fast and ends the server; a failed request on a live engine does not
(decisio.serve.engine_health).

H1  the rule: a request error passes through without a probe; vLLM's word that the engine is dead (EngineDeadError by
    name, the engine_dead flag, an errored flag) kills at once without a probe; any other exception is raised as it is
    while the engine answers its probe, and kills when the probe fails or hangs past the timeout; a poisoned request
    repeated ten times on a healthy engine never kills; once dead every call is refused without touching the engine;
    the death is recorded once and the process ended with code 70 after the grace period; the watch thread reads the
    flags
H2  the server, end to end on the CPU stand-in (tests/unit/dying_engine.py), in both arrangements:
    in        a forward pass raises and leaves the engine hanging: the probe times out, that request and every later
              one get 503, a request waiting on the engine's lock too, and the server exits within the probe's timeout
              plus the grace period (without the fix every later request hung)
    separate  the engine-core process dies, by a failing forward pass or killed between requests or under one:
              503 at once, and /health turns 503 without a request (without the fix: 500 forever)
    and in each: /health answers 503 with the reason and the server exits with code 70, its engine core gone too;
    a poisoned request (a bug on decisio's side) sent ten times to a healthy server gets 500 each time and the server
    stays up and answers; the same request on a dead engine ends it; a request error (400) leaves the engine alive
H3  the container: the Dockerfile's HEALTHCHECK command fails on the dead engine's 503 (and passes on 200), and
    compose's restart policy restarts a container whose process exits non-zero

  uv run pytest -q tests/unit/test_engine_death.py
"""

import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest

from decisio.serve.engine_health import (
    EXIT_ENGINE_DEAD,
    PROBE_SLOWEST_FACTOR,
    PROBE_TIMEOUT_S,
    EngineDead,
    EngineHealth,
    is_engine_dead_error,
    is_request_error,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("DECISIO_STAND_IN_MODEL", "Qwen/Qwen3-0.6B-Base")
# the lag runs (DYING_ENGINE_LAG, tests/unit/dying_engine.py) slow the stand-in down; every bound allows for it
LAG_S = sum(float(x.split("=")[1]) for x in filter(None, os.environ.get("DYING_ENGINE_LAG", "").split(",")))
FAST_S = 5.0 + LAG_S  # a refused request answers in milliseconds; a hanging one never does
GRACE_S = 2.0  # EngineHealth's default, as the server runs it


# ---- H1 the rule -----------------------------------------------------------------------------------------------------


class VLLMClientError(Exception):  # vLLM's class for errors the request caused, matched by name
    pass


class EngineDeadError(Exception):  # vLLM's error for an engine that is gone, matched by name
    pass


class Engine:
    """A fake engine: its probe answers, fails or hangs; `llm.llm_engine` carries vLLM's flags."""

    def __init__(self, probe="answers", engine_dead=False, errored=None):
        self.mode, self.probes = probe, 0
        ns = types.SimpleNamespace
        self.llm = ns(llm_engine=ns(engine_core=ns(resources=ns(engine_dead=engine_dead))))
        if errored is not None:
            self.llm.llm_engine.errored = errored

    def probe(self):
        self.probes += 1
        if self.mode == "slow":
            time.sleep(0.5)
        if self.mode == "fails":
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        if self.mode == "hangs":
            threading.Event().wait()


def health(exits, probe_timeout_s=0.2):
    return EngineHealth(grace_s=0.05, probe_timeout_s=probe_timeout_s, exit_fn=exits.append)


def raising(e):
    def fn():
        raise e

    return fn


def test_h1_request_errors_pass_without_a_probe():
    for e in (ValueError("bad"), TypeError("bad"), KeyError("k"), VLLMClientError("too long")):
        assert is_request_error(e)
        exits, eng = [], Engine(probe="fails")
        h = health(exits)
        with pytest.raises(type(e)):
            h.call(eng, raising(e))
        assert not h.dead and eng.probes == 0 and h.call(eng, lambda: 7) == 7


def test_h1_an_unexpected_error_on_a_live_engine_fails_only_its_request():
    exits, eng = [], Engine()
    h = health(exits)
    e = IndexError("list index out of range")
    with pytest.raises(IndexError) as raised:
        h.call(eng, raising(e))
    assert raised.value is e and not h.dead and eng.probes == 1
    assert h.call(eng, lambda: 7) == 7
    no_probe = types.SimpleNamespace()  # an engine without a probe is never declared dead on a guess
    with pytest.raises(IndexError):
        h.call(no_probe, raising(e))
    time.sleep(0.2)
    assert not h.dead and exits == []


def test_h1_a_poisoned_request_ten_times_never_kills_a_healthy_engine():
    exits, eng = [], Engine()
    h = health(exits)
    for _ in range(10):
        with pytest.raises(AssertionError):
            h.call(eng, raising(AssertionError("an invariant of decisio's broke")))
    time.sleep(0.2)
    assert not h.dead and exits == [] and eng.probes == 10 and h.call(eng, lambda: 7) == 7


@pytest.mark.parametrize(("probe", "why"), [("fails", "failed a probe (RuntimeError"), ("hangs", "within 0.2 s")])
def test_h1_an_unexpected_error_on_a_dead_engine_kills(probe, why):
    exits, eng = [], Engine(probe=probe)
    h = health(exits)
    e = RuntimeError("CUDA error: an illegal memory access was encountered")
    t0 = time.perf_counter()
    with pytest.raises(EngineDead, match="RuntimeError: CUDA error") as raised:
        h.call(eng, raising(e))
    took = time.perf_counter() - t0
    assert raised.value.__cause__ is e and h.dead and why in h.reason and eng.probes == 1
    assert took < 1.0  # a hanging probe is given its timeout, no more
    time.sleep(0.3)
    assert exits == [EXIT_ENGINE_DEAD]


@pytest.mark.parametrize(
    ("eng", "e", "why"),
    [
        (Engine(), EngineDeadError("EngineCore encountered an issue."), "EngineDeadError"),
        (Engine(engine_dead=True), RuntimeError("zmq"), "engine-core process exited"),
        (Engine(errored=True), RuntimeError("background loop"), "errored"),
    ],
)
def test_h1_vllm_saying_dead_kills_at_once_without_a_probe(eng, e, why):
    exits = []
    h = health(exits)
    eng.mode = "fails"
    with pytest.raises(EngineDead):
        h.call(eng, raising(e))
    assert h.dead and why in h.reason and eng.probes == 0
    assert is_engine_dead_error(e) == (why == "EngineDeadError")
    time.sleep(0.3)
    assert exits == [EXIT_ENGINE_DEAD]


def test_h1_a_probe_ends_when_the_engine_is_declared_dead_meanwhile():
    """A request waiting on its probe gets EngineDead as soon as another path (the watch thread, another request)
    declares the engine dead, not after the probe's timeout, and not the dropped connection of the exit that follows."""
    exits, eng = [], Engine(probe="hangs")
    h = health(exits, probe_timeout_s=5.0)
    threading.Timer(0.2, h.mark_dead, args=("vLLM's engine-core process exited",)).start()
    t0 = time.perf_counter()
    with pytest.raises(EngineDead, match="engine-core process exited"):
        h.call(eng, raising(IndexError("list index out of range")))
    assert time.perf_counter() - t0 < 1.0 and eng.probes == 1


def test_h1_a_slow_engine_is_given_time_to_answer_its_probe():
    """The probe's timeout is 3 times the slowest recent successful call when that is longer than probe_timeout_s, so
    an engine that is slow but alive is not declared dead (seen with the stand-in's forward passes made 6 s slow)."""
    exits, eng = [], Engine(probe="slow")  # the probe takes 0.5 s, more than the 0.2 s timeout below
    h = health(exits, probe_timeout_s=0.2)
    assert h.call(eng, lambda: time.sleep(0.4) or 7) == 7  # the engine's calls take 0.4 s: the timeout becomes 1.2 s
    assert h.probe_deadline_s() >= PROBE_SLOWEST_FACTOR * 0.4
    with pytest.raises(IndexError):
        h.call(eng, raising(IndexError("list index out of range")))
    assert not h.dead and eng.probes == 1
    fresh = health(exits, probe_timeout_s=0.2)  # no slow call seen: 0.2 s, and the 0.5 s probe is too slow
    with pytest.raises(EngineDead, match="within 0.2 s"):
        fresh.call(Engine(probe="slow"), raising(IndexError("list index out of range")))


def test_h1_a_start_up_probe_gives_the_first_request_a_slow_engine_s_time():
    """The server times one probe per engine at start-up (calibrate), so even its first request that trips a bug gives
    a slow engine's probe its time."""
    exits, eng = [], Engine(probe="slow")
    h = health(exits, probe_timeout_s=0.2)
    took = h.calibrate([eng, types.SimpleNamespace(), None])  # engines without a probe are skipped
    assert list(took) == ["Engine"] and took["Engine"] >= 500 and h.probe_deadline_s() >= 1.5
    with pytest.raises(IndexError):
        h.call(eng, raising(IndexError("list index out of range")))
    assert not h.dead and eng.probes == 2


def test_h1_the_exit_waits_for_the_requests_in_flight():
    """After the grace period the process ends once no request is in flight, and at the latest drain_s later."""
    exits = []
    h = EngineHealth(grace_s=0.05, drain_s=2.0, exit_fn=exits.append)
    h.request_started()
    h.mark_dead("vLLM's engine-core process exited")
    time.sleep(0.4)
    assert exits == []  # the request is still being answered
    h.request_finished()
    time.sleep(0.4)
    assert exits == [EXIT_ENGINE_DEAD]
    exits = []
    h = EngineHealth(grace_s=0.05, drain_s=0.3, exit_fn=exits.append)
    h.request_started()  # a request the dead engine never returns
    h.mark_dead("the engine did not answer a probe within 5 s")
    time.sleep(0.7)
    assert exits == [EXIT_ENGINE_DEAD]


def test_h1_dead_refuses_without_calling_and_dies_once():
    exits, calls = [], []
    h = health(exits)
    h.mark_dead("first")
    h.mark_dead("second")
    with pytest.raises(EngineDead, match="first"):
        h.call(Engine(), calls.append, 1)
    with pytest.raises(EngineDead):
        h.check()
    time.sleep(0.3)
    assert calls == [] and h.reason == "first" and exits == [EXIT_ENGINE_DEAD]


def test_h1_watch_reads_the_flags():
    exits = []
    h = health(exits)
    assert h.watch([types.SimpleNamespace(), None]) is None  # no engine with vLLM's flags: nothing to watch
    eng = Engine()
    t = h.watch([types.SimpleNamespace(llm=None), eng], interval_s=0.01)
    time.sleep(0.1)
    assert not h.dead
    eng.llm.llm_engine.engine_core.resources.engine_dead = True
    t.join(2)
    assert not t.is_alive() and h.reason == "vLLM's engine-core process exited"


# ---- H2 the server on the CPU stand-in -------------------------------------------------------------------------------

QUESTION = {
    "state": {},
    "model": "m",
    "questions": {"q1": {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "yes", "b": "no"}}},
}


def call(url, path, body=None, timeout=60.0):
    """(status, JSON body, seconds); status None when no answer came within `timeout`."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw), time.perf_counter() - t0
        except ValueError:
            return e.code, raw.decode(errors="replace"), time.perf_counter() - t0
    except TimeoutError:
        return None, None, time.perf_counter() - t0


class DyingServer:
    """tests/unit/dying_engine.py in a child process: the real server, whose engine dies when the test says."""

    def __init__(self, tmp_path, arrangement, lag=None):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.control = tmp_path / "control"
        self.log = (tmp_path / "server.log").open("w+")
        cmd = [sys.executable, str(ROOT / "tests" / "unit" / "dying_engine.py"), arrangement, str(self.control)]
        self.proc = subprocess.Popen(
            [*cmd, "--backend", "hf", "--model", MODEL, "--port", str(self.port)],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "DYING_ENGINE_LAG": lag} if lag else None,
        )
        self.core_pid = None
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(f"the stand-in server exited:\n{self.output()}")
            try:
                if call(self.url, "/health", timeout=5)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        else:
            self.stop()
            raise AssertionError("the stand-in server did not come up")
        if arrangement == "separate":
            self.core_pid = int(re.search(r"ENGINE CORE PID (\d+)", self.output()).group(1))

    def output(self):
        self.log.flush()
        self.log.seek(0)
        return self.log.read()

    def say(self, command):
        """Atomically: the engine reads the file every 50 ms, and write_text truncates it first, so a read in between
        saw it empty and ended a hold as if told to go on (one request answered 200 instead of failing)."""
        tmp = self.control.with_name(self.control.name + ".tmp")
        tmp.write_text(command)
        os.replace(tmp, self.control)

    def exit_code(self, timeout=15.0):
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def stop(self):
        """By the server's own process group (it started its own session), so its engine core goes too."""
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, 9)
            self.proc.wait(timeout=30)
        if self.core_pid is not None and psutil.pid_exists(self.core_pid):
            os.kill(self.core_pid, 9)
        self.log.close()


@pytest.fixture
def server(tmp_path, request):
    s = DyingServer(tmp_path, request.param)
    yield s
    s.stop()


def assert_dead_and_exits(s, why):
    """/health reports the death (its reason holds `why`, or one of `why` when a tuple), a new request is refused at
    once, and the process ends with code 70."""
    status, body, _ = call(s.url, "/health", timeout=FAST_S)
    whys = why if isinstance(why, tuple) else (why,)
    assert status == 503 and body["engine"] == "dead" and any(w in body["reason"] for w in whys), (status, body)
    assert body["exit_code"] == EXIT_ENGINE_DEAD
    status, body, took = call(s.url, "/v1/systemone", QUESTION, timeout=FAST_S)
    assert status == 503 and "the engine is dead" in body["detail"] and took < FAST_S, (status, body, took)
    assert s.exit_code() == EXIT_ENGINE_DEAD, f"the server did not exit with {EXIT_ENGINE_DEAD}:\n{s.output()}"
    assert "ENGINE DEAD" in s.output()
    if s.core_pid is not None:
        time.sleep(0.5)
        assert not psutil.pid_exists(s.core_pid) or psutil.Process(s.core_pid).status() == psutil.STATUS_ZOMBIE


POISONED = {**QUESTION, "questions": {"q1": {**QUESTION["questions"]["q1"], "instructions": "Pick one. xyzzy"}}}
# the probe's timeout (longer with the stand-in's lag), the grace period, and time to stop
DIES_WITHIN_S = PROBE_TIMEOUT_S + GRACE_S + 3.0 + (PROBE_SLOWEST_FACTOR + 1) * LAG_S


@pytest.mark.parametrize("server", ["in", "separate"], indirect=True)
def test_h2_a_forward_pass_that_raises(server):
    """in: the engine is left hanging, as vLLM in-process is, so the probe times out and the server exits within the
    probe's timeout plus the grace period; separate: the core dies with it and vLLM says so, so it is dead at once."""
    status, body, _ = call(server.url, "/v1/systemone", QUESTION)
    assert status == 200 and set(body["answers"]) == {"q1"}
    assert call(server.url, "/v1/answer", {"state": "s", "questions": [{"text": "Which?"}]})[0] == 400
    assert call(server.url, "/health")[0] == 200  # a request error leaves the engine alive
    server.say("raise")
    t0 = time.monotonic()
    status, body, took = call(server.url, "/v1/systemone", QUESTION, timeout=PROBE_TIMEOUT_S + FAST_S)
    assert status == 503, (status, body, took)
    if server.core_pid is None:
        assert "RuntimeError: CUDA error" in body["detail"] and "did not answer a probe" in body["detail"]
        assert PROBE_TIMEOUT_S <= took < PROBE_TIMEOUT_S + FAST_S
    else:
        assert "EngineDeadError" in body["detail"] and took < FAST_S
    assert_dead_and_exits(server, "RuntimeError" if server.core_pid is None else "EngineDeadError")
    assert time.monotonic() - t0 < DIES_WITHIN_S


@pytest.mark.parametrize("server", ["in", "separate"], indirect=True)
def test_h2_a_poisoned_request_never_takes_a_healthy_server_down(server):
    """A bug on decisio's side, tripped by one request with the engine healthy: 500 each time, as before the engine's
    health was watched, and the server stays up and answers."""
    for _ in range(10):
        status, body, took = call(server.url, "/v1/systemone", POISONED, timeout=PROBE_TIMEOUT_S + FAST_S)
        assert status == 500 and took < FAST_S, (status, body, took)
    status, body, _ = call(server.url, "/v1/systemone", QUESTION)
    assert status == 200 and set(body["answers"]) == {"q1"}
    assert call(server.url, "/health")[0] == 200
    assert server.exit_code(timeout=GRACE_S + 1.0) is None, server.output()
    assert server.output().count("the engine answered a probe") == 10


@pytest.mark.parametrize("server", ["in", "separate"], indirect=True)
def test_h2_a_poisoned_request_on_a_dead_engine_ends_it(server):
    """The same request after the engine has died without a word (in: it hangs; separate: the core is killed)."""
    assert call(server.url, "/v1/systemone", QUESTION)[0] == 200
    if server.core_pid is None:
        server.say("break")
    else:
        os.kill(server.core_pid, 9)
    t0 = time.monotonic()
    status, body, _ = call(server.url, "/v1/systemone", POISONED, timeout=PROBE_TIMEOUT_S + FAST_S)
    assert status == 503 and "the engine is dead" in body["detail"], (status, body)
    # separate: vLLM's flag, or the probe through the gone engine core when the request is confirmed before the flag
    # has seen the kill (both say the core is gone; seen on a card box's set-up, 2026-10-05)
    gone = ("engine-core process exited", "failed a probe (EngineDeadError")
    assert_dead_and_exits(server, "did not answer a probe" if server.core_pid is None else gone)
    assert time.monotonic() - t0 < DIES_WITHIN_S


@pytest.mark.parametrize("server", ["in", "separate"], indirect=True)
def test_h2_requests_in_flight_when_it_dies(server):
    """One request inside the forward pass, one waiting for the engine's lock: both answered 503 at once."""
    server.say("hold")
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(call, server.url, "/v1/systemone", QUESTION, 30.0)
        time.sleep(1.0)
        waiting = pool.submit(call, server.url, "/v1/systemone", QUESTION, 30.0)
        time.sleep(1.0)
        assert not first.done() and not waiting.done()
        if server.core_pid is None:
            server.say("raise")
        else:
            os.kill(server.core_pid, 9)
        for f in (first, waiting):
            status, body, _ = f.result()
            assert status == 503 and "the engine is dead" in body["detail"], (status, body)
    # separate: the request in flight sees the core gone (EngineDeadError), or vLLM's flag does first
    gone = ("EngineDeadError", "engine-core process exited")
    assert_dead_and_exits(server, "RuntimeError" if server.core_pid is None else gone)


@pytest.mark.parametrize("server", ["separate"], indirect=True)
def test_h2_engine_core_killed_between_requests(server):
    assert call(server.url, "/v1/systemone", QUESTION)[0] == 200
    os.kill(server.core_pid, 9)
    deadline = time.time() + 10
    while call(server.url, "/health", timeout=FAST_S)[0] == 200 and time.time() < deadline:
        time.sleep(0.1)
    assert_dead_and_exits(server, "engine-core process exited")


def test_h2_a_request_inside_the_engine_when_it_dies_gets_its_answer(tmp_path):
    """separate: the request in flight sees the killed core 4 s late (DYING_ENGINE_LAG alive=4) while vLLM's flag sees
    it at once and the server declares the engine dead; the exit waits for that request's answer instead of dropping
    its connection after the 2 s grace period."""
    server = DyingServer(tmp_path, "separate", lag="alive=4")
    try:
        server.say("hold")
        with ThreadPoolExecutor(1) as pool:
            first = pool.submit(call, server.url, "/v1/systemone", QUESTION, 30.0)
            time.sleep(1.0)
            os.kill(server.core_pid, 9)
            status, body, _ = first.result()
        assert status == 503 and "the engine is dead" in body["detail"], (status, body)
        assert server.exit_code() == EXIT_ENGINE_DEAD, server.output()
    finally:
        server.stop()


def test_h2_a_slow_healthy_engine_is_not_declared_dead(tmp_path):
    """separate: the engine core answers every forward pass 6 s late (DYING_ENGINE_LAG ping=6), longer than the probe's
    5 s; the server's very first request trips a bug and still gets 500, and the server stays up, since the probe waits
    3 times the slowest call seen, the start-up probe's included."""
    server = DyingServer(tmp_path, "separate", lag="ping=6")
    try:
        status, body, took = call(server.url, "/v1/systemone", POISONED, timeout=60)
        assert status == 500, (status, body, took)
        assert call(server.url, "/health")[0] == 200
        assert server.exit_code(timeout=GRACE_S + 1.0) is None, server.output()
        assert "the engine answered a probe" in server.output()
    finally:
        server.stop()


# ---- H3 the container ------------------------------------------------------------------------------------------------


def healthcheck_command():
    text = (ROOT / "Dockerfile").read_text()
    m = re.search(r"^HEALTHCHECK [^\n]*\\\n\s*CMD (\[[^\n]*\])$", text, re.M)
    return json.loads(m.group(1))


def serve_status(status):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/health"
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b'{"ok": false, "engine": "dead"}' if status != 200 else b'{"ok": true}')

        def log_message(self, *args):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


@pytest.mark.parametrize(("status", "healthy"), [(200, True), (503, False)])
def test_h3_dockerfile_healthcheck_fails_on_a_dead_engine(status, healthy):
    cmd = healthcheck_command()
    assert cmd[0] == "python3" and "/health" in cmd[-1]
    httpd = serve_status(status)
    try:
        env = {**os.environ, "DECISIO_PORT": str(httpd.server_address[1])}
        r = subprocess.run([sys.executable, *cmd[1:]], env=env, capture_output=True, timeout=30)
    finally:
        httpd.shutdown()
    assert (r.returncode == 0) is healthy, r.stderr


def test_h3_compose_restarts_a_process_that_exits_non_zero():
    """Docker restarts a container on its process's exit, not on an unhealthy HEALTHCHECK: the exit is what restarts.
    The entrypoint execs the server, so the server's exit code is the container's."""
    m = re.search(r"^\s+restart: (\S+)", (ROOT / "compose.yaml").read_text(), re.M)
    assert m and m.group(1) in ("unless-stopped", "always", "on-failure")
    entrypoint = (ROOT / "docker" / "entrypoint.sh").read_text()
    assert re.search(r"^exec python3 -m decisio\.serve\.vllm_engine", entrypoint, re.M)
