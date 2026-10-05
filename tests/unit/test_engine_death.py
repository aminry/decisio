# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A dead engine is reported, refused fast and ends the server (decisio.serve.engine_health).

H1  an engine call's exception marks the engine dead unless the request caused it (decisio's input errors, vLLM's
    VLLMClientError); once dead every call is refused without touching the engine; the death is recorded once and
    the process is ended with code 70 after the grace period; the watch thread marks it from a probe
H2  the server, end to end on the CPU stand-in (tests/unit/dying_engine.py), in both arrangements:
    in        a forward pass raises: that request and every later one get 503 at once (without the fix every later
              request hung), and so does a request waiting on the engine's lock when it dies
    separate  the engine-core process is killed between requests or under one: /health turns 503 without a request,
              the request in flight and every later one get 503 at once (without the fix: 500 forever)
    and in each, /health answers 503 with the reason and the server exits with code 70, its engine core gone too;
    a request error (400) leaves the engine alive
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
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest

from decisio.serve.engine_health import EXIT_ENGINE_DEAD, EngineDead, EngineHealth, is_request_error

ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("DECISIO_STAND_IN_MODEL", "Qwen/Qwen3-0.6B-Base")
FAST_S = 5.0  # a refused request answers in milliseconds; a hanging one never does


# ---- H1 the engine's state -------------------------------------------------------------------------------------------


class VLLMClientError(Exception):  # vLLM's class for errors the request caused, matched by name
    pass


class EngineDeadError(Exception):  # vLLM's VLLMServerError for an engine-core process that is gone
    pass


def health(exits):
    return EngineHealth(grace_s=0.05, exit_fn=exits.append)


def raising(e):
    def fn():
        raise e

    return fn


def test_h1_request_errors_pass_and_engine_errors_kill():
    for e in (ValueError("bad"), TypeError("bad"), KeyError("k"), VLLMClientError("too long")):
        assert is_request_error(e)
        exits = []
        h = health(exits)
        with pytest.raises(type(e)):
            h.call(raising(e))
        assert not h.dead and h.call(lambda: 7) == 7
    for e in (RuntimeError("CUDA error: an illegal memory access"), EngineDeadError("EngineCore"), MemoryError()):
        assert not is_request_error(e)
        exits = []
        h = health(exits)
        with pytest.raises(EngineDead, match=type(e).__name__) as raised:
            h.call(raising(e))
        assert raised.value.__cause__ is e and h.dead and h.reason.startswith(type(e).__name__)
        time.sleep(0.3)
        assert exits == [EXIT_ENGINE_DEAD]


def test_h1_dead_refuses_without_calling_and_dies_once():
    exits, calls = [], []
    h = health(exits)
    h.mark_dead("first")
    h.mark_dead("second")
    with pytest.raises(EngineDead, match="first"):
        h.call(calls.append, 1)
    with pytest.raises(EngineDead):
        h.check()
    time.sleep(0.3)
    assert calls == [] and h.reason == "first" and exits == [EXIT_ENGINE_DEAD]


def test_h1_watch_marks_from_a_probe():
    exits, state = [], {"why": None}
    h = health(exits)

    def broken_probe():
        raise AttributeError("no engine core")

    assert h.watch([None]) is None
    t = h.watch([broken_probe, lambda: state["why"]], interval_s=0.01)
    time.sleep(0.1)
    assert not h.dead
    state["why"] = "vLLM's engine-core process exited"
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

    def __init__(self, tmp_path, arrangement):
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
        self.control.write_text(command)

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
    """/health reports the death, a new request is refused at once, and the process ends with code 70."""
    status, body, _ = call(s.url, "/health", timeout=FAST_S)
    assert status == 503 and body["engine"] == "dead" and why in body["reason"], (status, body)
    assert body["exit_code"] == EXIT_ENGINE_DEAD
    status, body, took = call(s.url, "/v1/systemone", QUESTION, timeout=FAST_S)
    assert status == 503 and "the engine is dead" in body["detail"] and took < FAST_S, (status, body, took)
    assert s.exit_code() == EXIT_ENGINE_DEAD, f"the server did not exit with {EXIT_ENGINE_DEAD}:\n{s.output()}"
    assert "ENGINE DEAD" in s.output()
    if s.core_pid is not None:
        time.sleep(0.5)
        assert not psutil.pid_exists(s.core_pid) or psutil.Process(s.core_pid).status() == psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("server", ["in", "separate"], indirect=True)
def test_h2_a_forward_pass_that_raises(server):
    status, body, _ = call(server.url, "/v1/systemone", QUESTION)
    assert status == 200 and set(body["answers"]) == {"q1"}
    assert call(server.url, "/v1/answer", {"state": "s", "questions": [{"text": "Which?"}]})[0] == 400
    assert call(server.url, "/health")[0] == 200  # a request error leaves the engine alive
    server.say("raise")
    status, body, took = call(server.url, "/v1/systemone", QUESTION, timeout=FAST_S)
    assert status == 503 and took < FAST_S, (status, body, took)
    why = "RuntimeError: CUDA error" if server.core_pid is None else "EngineDeadError"
    assert why in body["detail"]
    assert_dead_and_exits(server, why)


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
    assert_dead_and_exits(server, "RuntimeError" if server.core_pid is None else "EngineDeadError")


@pytest.mark.parametrize("server", ["separate"], indirect=True)
def test_h2_engine_core_killed_between_requests(server):
    assert call(server.url, "/v1/systemone", QUESTION)[0] == 200
    os.kill(server.core_pid, 9)
    deadline = time.time() + 10
    while call(server.url, "/health", timeout=FAST_S)[0] == 200 and time.time() < deadline:
        time.sleep(0.1)
    assert_dead_and_exits(server, "engine-core process exited")


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
