# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""`--register-boundary after` through the server, on the CPU stand-in (decisio.serve.hf_letters) with a model of
vLLM's prefix cache on a Gemma base: a question reads its state from the cache only when a warm-up registered the
state's boundary, else it reads all of it (decisio.serve.boundary).

H1  the caller does not wait: a first question on a new state is answered while its warm-up is held, and the warm-up
    starts only once the response has been handed to the server
H2  a follow-up sent right after the response, while the warm-up runs, waits for it and reads the state from the
    cache; a request about another state waits for it too (the registration's cost under load)
H3  /health reports where registration happens and counts what was deferred, registered, dropped and failed
H4  an injected failure of the warm-up (the engine still answers its probe) is counted; the server stays up, and the
    next question on that state reads it again and defers its warm-up again
H5  an engine that dies during the warm-up is declared dead: /health turns 503, the exit is called, nothing is pending
H6  --register-boundary before (0.8.1): the warm-up goes ahead of the question, in the caller's request
H7  a process that exits right after its answer waits for the warm-up in progress, and exits cleanly (on a card the
    interpreter aborted with signal 6 when it shut down under a warm-up on the registrar's thread)

    uv run pytest -q tests/unit/test_boundary_after_response.py
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from decisio.families import BASES
from decisio.serve import boundary
from decisio.serve import vllm_engine as sv
from decisio.serve.engine_health import EngineHealth

MODEL = os.environ.get("DECISIO_STAND_IN_MODEL", "Qwen/Qwen3-0.6B-Base")
HIT = 64
Q1 = {"kind": "noul", "instructions": "Is the customer reporting an outage?"}
Q2 = {"kind": "choice", "instructions": "Which team?", "options": ["billing", "technical", "sales", "account"]}


def ticket(i):
    return f"Ticket {i}. Customer writes: our API integration started returning 500 errors on every request. " * 6


class Cached:
    """Mixed into the stand-in: the prefix cache of a Gemma base, modelled. A warm-up registers its state (all but its
    last token); a question reads from the cache the whole hit units of the longest registered state it starts with.
    `gate` holds a warm-up until set; `fail` makes the next warm-ups raise."""

    def setup(self):
        self.family, self.cache_hit_unit, self.match_unit = BASES["gemma-4-12b"], HIT, 16
        self.registered, self.events = set(), []
        self.gate, self.warm_started, self.fail = threading.Event(), threading.Event(), None
        self.gate.set()

    def send_warm(self, warm, adapter=None, mm=None, mm_uuids=None):
        self.events.append(("warm", time.monotonic()))
        self.warm_started.set()
        assert self.gate.wait(60)
        if self.fail is not None:
            ms = sv.guarded(self, self._raise)
        else:
            ms = super().send_warm(warm, adapter, mm, mm_uuids)
        self.registered.update(tuple(w[:-1]) for w in warm)
        return ms

    def _raise(self):
        raise self.fail

    def score_prompts(self, rows, adapter=None, warm=(), skip_cache=False, mm=None, mm_uuids=None):
        if warm:  # --register-boundary before: the warm-up, then the questions
            self.send_warm(warm, adapter)
        self.events.append(("question", time.monotonic()))
        probs, info = super().score_prompts(rows, adapter, (), skip_cache, mm, mm_uuids)
        cached = []
        for ids, _ in rows:
            hit = max((len(p) for p in self.registered if tuple(ids[: len(p)]) == p), default=0)
            cached.append((hit // HIT) * HIT)
        info["cached_tokens"] = cached
        return probs, info


@pytest.fixture(scope="module")
def stand_in():
    from decisio.serve.hf_letters import HFLettersEngine

    eng = type("CachedStandIn", (Cached, HFLettersEngine), {})(MODEL, warm_up=False)
    eng.setup()
    eng._warm_up()
    return eng


@pytest.fixture
def served(stand_in, monkeypatch):
    """A fresh server on the shared stand-in: its cache, registrar, health and boundary registry new."""
    eng = stand_in
    eng.setup()
    for name in ("_registrar", "_boundary_lru"):
        eng.__dict__.pop(name, None)
    eng.register_boundary = "after"
    exits = []
    eng.health = EngineHealth(grace_s=0.05, drain_s=0.3, exit_fn=exits.append)
    releases = []
    release = boundary.Ticket.release

    def recorded(self):
        releases.append(time.monotonic())
        release(self)

    monkeypatch.setattr(boundary.Ticket, "release", recorded)
    with TestClient(sv.make_app(eng, health=eng.health)) as client:
        yield eng, client, releases, exits
    eng.gate.set()


def ask(client, state, question):
    r = client.post("/v1/answer", json={"state": state, "questions": [question]})
    assert r.status_code == 200, r.text
    return r.json()["timing"]


def whole(eng, state, question):
    _, P = eng._prepare_separate(state, [question])
    return (P // HIT) * HIT


def test_h1_the_caller_does_not_wait_for_the_registration(served):
    eng, client, releases, _ = served
    eng.gate.clear()  # the warm-up cannot finish
    t = ask(client, ticket(1), Q1)
    assert t["state_boundary"] == {"registered": 0, "found": 0, "deferred": 1} and t["cached_tokens"] == [0]
    assert eng.warm_started.wait(30)
    warm_at = next(at for kind, at in eng.events if kind == "warm")
    assert warm_at >= releases[0]  # after the response was handed to the server
    eng.gate.set()


def test_h2_a_follow_up_waits_for_the_registration_and_reads_the_state_from_the_cache(served):
    eng, client, _, _ = served
    eng.gate.clear()
    ask(client, ticket(2), Q1)
    assert eng.warm_started.wait(30)
    out = {}
    follow = threading.Thread(target=lambda: out.update(same=ask(client, ticket(2), Q2)))
    other = threading.Thread(target=lambda: out.update(other=ask(client, ticket(3), Q1)))
    follow.start()
    other.start()
    time.sleep(0.5)
    assert follow.is_alive() and other.is_alive()  # both wait behind the warm-up
    eng.gate.set()
    follow.join(60)
    other.join(60)
    same = out["same"]
    assert same["cached_tokens"] == [whole(eng, ticket(2), Q2)] and same["state_boundary"]["found"] == 1
    assert out["other"]["state_boundary"]["deferred"] == 1


def test_h3_health_reports_the_registration(served):
    eng, client, _, _ = served
    ask(client, ticket(4), Q1)
    ask(client, ticket(4), Q2)  # due by now: sent first, if the registrar's thread has not
    h = client.get("/health").json()
    assert h["state_boundary"] == {"pending": 0, "deferred": 1, "registered": 1, "dropped": 0, "failed": 0}


def test_h4_a_failed_warm_up_is_counted_and_the_server_stays_up(served):
    eng, client, _, exits = served
    eng.fail = RuntimeError("an injected failure of the warm-up")
    ask(client, ticket(5), Q1)
    t = ask(client, ticket(5), Q2)  # the failed registration went first; this one reads the state again
    assert t["cached_tokens"] == [0] and t["state_boundary"]["deferred"] == 1
    eng.fail = None
    assert client.get("/health").status_code == 200 and exits == []
    assert eng.registrar.facts()["failed"] == 1


def test_h5_an_engine_that_dies_during_the_warm_up_is_declared_dead(served, monkeypatch):
    eng, client, _, exits = served

    def dead_probe():
        raise RuntimeError("the engine does not answer (injected)")

    monkeypatch.setattr(eng, "probe", dead_probe)
    eng.fail = RuntimeError("CUDA error: an illegal memory access was encountered (injected)")
    ask(client, ticket(6), Q1)
    deadline = time.monotonic() + 30
    while not eng.health.dead and time.monotonic() < deadline:
        time.sleep(0.02)
    assert eng.health.dead
    assert client.get("/health").status_code == 503
    assert client.post("/v1/answer", json={"state": ticket(6), "questions": [Q2]}).status_code == 503
    while not exits and time.monotonic() < deadline:
        time.sleep(0.02)
    assert exits == [70] and eng.registrar.facts()["pending"] == 0


def test_h6_before_sends_the_warm_up_ahead_of_the_question(served):
    eng, client, _, _ = served
    eng.register_boundary = "before"
    t = ask(client, ticket(7), Q1)
    assert t["state_boundary"] == {"registered": 1, "found": 0, "deferred": 0}
    assert t["cached_tokens"] == [whole(eng, ticket(7), Q1)]
    assert [k for k, _ in eng.events] == ["warm", "question"]


EXIT_SCRIPT = """
import sys, time
sys.path.insert(0, {here!r})
from test_boundary_after_response import Cached, MODEL, ticket
from decisio.serve.hf_letters import HFLettersEngine

class Slow(Cached, HFLettersEngine):
    def send_warm(self, warm, adapter=None, mm=None, mm_uuids=None):
        print("WARM START", flush=True)
        time.sleep(1.5)
        ms = super().send_warm(warm, adapter, mm, mm_uuids)
        print("WARM END", flush=True)
        return ms

eng = Slow(MODEL, warm_up=False)
eng.setup()
eng.register_boundary = "after"
eng.answer(ticket(8), [{{"kind": "noul", "instructions": "Is the customer reporting an outage?"}}])
assert eng.warm_started.wait(30)
print("EXIT", flush=True)
sys.exit(0)
"""


@pytest.mark.slow
def test_h7_an_exit_waits_for_the_warm_up_in_progress():
    script = EXIT_SCRIPT.format(here=str(Path(__file__).parent))
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    lines = [x for x in r.stdout.splitlines() if x in ("WARM START", "EXIT", "WARM END")]
    assert r.returncode == 0, r.stderr[-2000:]
    assert lines == ["WARM START", "EXIT", "WARM END"], r.stdout[-2000:]


def test_h8_a_request_records_how_long_it_waited_for_the_engine_and_behind_what(served):
    """#132: under `after` a follow-up waits for the registrar's warm-up; `ran_before` stays 0 (the request sent nothing
    itself), and the wait is `waited_ms`, flagged `waited_behind_registration`; an idle server waits 0."""
    eng, client, _, _ = served
    idle = ask(client, ticket(8), Q1)  # nothing runs before it
    assert idle["waited_ms"] < 50 and idle["waited_behind_registration"] is False
    time.sleep(0.5)  # its registration has finished (the gate is open)
    eng.gate.clear()
    ask(client, ticket(9), Q1)  # its warm-up starts and is held
    assert eng.warm_started.wait(30)
    out = {}
    follow = threading.Thread(target=lambda: out.update(t=ask(client, ticket(10), Q1)))
    follow.start()
    time.sleep(0.6)
    assert follow.is_alive()
    eng.gate.set()
    follow.join(60)
    t = out["t"]
    assert t["waited_ms"] >= 500 and t["waited_behind_registration"] is True
    assert t["state_boundary"].get("ran_before", 0) == 0  # the field that missed the wait
