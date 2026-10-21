# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""`--concurrency batched` (decisio.serve.combine): the requests waiting for the engine scored together in one call.

B1  a lone request goes through the same engine call as on a serial server: the same rows in the same calls (one at a
    time under --multi-question sequential), the same probabilities and the same facts
B2  requests that arrive while a call runs are scored together in the next call, every row in one engine call; each
    caller gets its own probabilities and its own rows' facts (cached tokens, label forms, prefix, boundary counts)
    and says which batch it was in
B3  a call holds the requests of one adapter only, the oldest waiting request's, oldest first, up to the cap
B4  a failed call fails each of its requests, and the next call is served; at exit the waiting requests fail and
    nothing hangs
B5  boundary registrations: each request's deferred warm-up goes to its own ticket and becomes due with its own
    response; two requests about one new state send one warm-up; due ones go before the next call's questions
B6  the mode is in /health and, only when batched, in the serving-stack fingerprint; a certificate made under one mode
    is `certified: false` with `concurrency_changed` under the other, and certified again when the mode is back
B7  the flags: serial is the default; batched is refused on MLX and with --mode packed; the cap is at least 1
B8  on the CPU stand-in through /v1/answer (slow): concurrent clients get exactly the serial answers, in fewer engine
    calls than requests

    uv run pytest -q tests/unit/test_concurrency_batched.py
"""

import threading
import time
from collections import OrderedDict

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_certified_answers import FakeEngine as CertEngine
from test_certified_answers import examples, post_task, wire

from decisio.families import BASES
from decisio.serve import boundary
from decisio.serve import vllm_engine as sv
from decisio.serve.combine import Combiner, FairLock
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore

LABELS = ((7,), (8,), (9,))


class Rows(sv.LettersEngine):
    """LettersEngine without vLLM: a request's rows are its state's tokens plus one token per question; scoring records
    each engine call and gives each row probabilities that depend on the row alone, so any mix-up between callers shows.
    `gate` holds a call until set; `fail` makes the next call raise."""

    def __init__(self, concurrency="serial", multi_question="sequential", register=False):
        self.mode, self.pad_unit, self.match_unit, self.adapters = "separate", None, 16, {"a": object(), "b": object()}
        self.health = None
        self.multi_question, self.concurrency = multi_question, concurrency
        self.family = BASES["gemma-4-12b"] if register else None
        self.cache_hit_unit = 16 if register else None
        self.register_boundary = "after"
        self._lock = FairLock()  # as LettersEngine.__init__ and the stand-in make it
        self.calls, self.warms = [], []
        self.gate, self.started, self.fail = threading.Event(), threading.Event(), None
        self.gate.set()

    def _prepare_separate(self, state, questions):
        prefix = [1000 + ord(c) for c in state]
        return [(prefix + [i + 1], LABELS) for i, _ in enumerate(questions)], len(prefix)

    def send_warm(self, warm, adapter=None, mm=None, mm_uuids=None):
        self.warms.append([tuple(w) for w in warm])
        return 0.1

    def score_prompts(self, rows, adapter=None, warm=(), skip_cache=False, mm=None, mm_uuids=None):
        self.started.set()
        assert self.gate.wait(30)
        if warm:
            self.send_warm(warm, adapter)
        self.calls.append({"rows": [tuple(r) for r, _ in rows], "adapter": adapter, "warm": len(warm)})
        if self.fail is not None:
            e, self.fail = self.fail, None
            raise e
        probs = []
        for ids, _ in rows:
            z = np.array([sum(ids) % 7, sum(ids) % 5, sum(ids) % 3], dtype=float)
            probs.append(np.exp(z) / np.exp(z).sum())
        tag = [sum(ids) for ids, _ in rows]
        return probs, {
            "questions_ms": 1.0,
            "readout_ms": 0.1,
            "warm_ms": 0.0,
            "cached_tokens": tag,
            "engine_prompt_tokens": [len(ids) for ids, _ in rows],
            "label_token_logprobs": [[[float(t)]] for t in tag],
            "engine_timing": [{"queue_ms": 0.0} for _ in rows],
            "cached_tokens_mean": float(np.mean(tag)),
            "prompt_tokens": sum(len(ids) for ids, _ in rows),
        }

    @property
    def _boundaries(self):
        if not hasattr(self, "_boundary_lru"):
            self._boundary_lru = OrderedDict()
        return self._boundary_lru


Q = [{"kind": "choice", "instructions": "Which?", "options": ["x", "y", "z"]}]
Q3 = Q * 3


def in_threads(fn, args):
    out, threads = [None] * len(args), []
    for i, a in enumerate(args):

        def run(i=i, a=a):
            out[i] = fn(*a)

        threads.append(threading.Thread(target=run))
        threads[-1].start()
    for t in threads:
        t.join(30)
        assert not t.is_alive()
    return out


def hold_first(eng, first):
    """Start `first` (a function) on its own thread and hold its engine call, so that what is sent next waits."""
    eng.gate.clear()
    eng.started.clear()
    t = threading.Thread(target=first)
    t.start()
    assert eng.started.wait(10)
    return t


def wait_for_waiting(eng, n):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if len(eng.combiner._waiting) >= n:
            return
        time.sleep(0.005)
    raise AssertionError(f"{n} requests never waited")


# ---- B1 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mq", ["sequential", "warm", "batch"])
def test_b1_a_lone_request_is_the_serial_call(mq):
    serial, batched = Rows("serial", mq), Rows("batched", mq)
    for state, qs in (("a short state", Q), ("another state", Q3)):
        ps, infos = serial.answer(state, qs)
        pb, infob = batched.answer(state, qs)
        assert [p.tobytes() for p in ps] == [p.tobytes() for p in pb]
        drop = ("server_ms", "waited_ms", "prepare_ms")
        assert {k: v for k, v in infos.items() if k not in drop} == {k: v for k, v in infob.items() if k not in drop}
        assert "batch" not in infob and not any(k.startswith("_") for k in infob)
    assert serial.calls == batched.calls  # the same rows in the same engine calls (sequential: one at a time)


# ---- B2 -------------------------------------------------------------------------------------------------------------


def test_b2_requests_waiting_behind_a_call_are_scored_together_each_with_its_own_rows():
    eng = Rows("batched")
    ref = Rows("serial")
    states = [f"state number {i}" for i in range(5)]
    first = hold_first(eng, lambda: eng.answer("the first state", Q))
    results = [None] * 5

    def send(i):
        results[i] = eng.answer(states[i], Q3 if i % 2 else Q)

    threads = [threading.Thread(target=send, args=(i,)) for i in range(5)]
    for th in threads:
        th.start()
    wait_for_waiting(eng, 5)
    eng.gate.set()
    for th in threads + [first]:
        th.join(10)
    assert len(eng.calls) == 2  # the held call, then the five waiting requests in one call
    assert len(eng.calls[1]["rows"]) == sum(3 if i % 2 else 1 for i in range(5))  # every row, whatever --multi-question
    for i, (probs, info) in enumerate(results):
        want, want_info = ref.answer(states[i], Q3 if i % 2 else Q)
        assert [p.tobytes() for p in probs] == [p.tobytes() for p in want]  # the stand-in's rows do not interact
        assert info["cached_tokens"] == want_info["cached_tokens"]  # its own rows' facts, not a neighbour's
        assert info["label_token_logprobs"] == want_info["label_token_logprobs"]
        assert info["engine_prompt_tokens"] == want_info["engine_prompt_tokens"]
        assert info["shared_prefix_tokens"] == want_info["shared_prefix_tokens"]
        assert info["questions"] == len(probs) and info["cached_tokens_mean"] == want_info["cached_tokens_mean"]
        assert info["batch"]["calls"] == 5 and info["batch"]["requests"] == 5 and info["batch"]["rows"] == 9
        assert info["waited_ms"] > 0 and not any(k.startswith("_") for k in info)
    assert sorted(r[1]["batch"]["position"] for r in results) == list(range(5))
    f = eng.concurrency_facts()
    assert (
        f["mode"] == "batched"
        and f["calls"] == 2
        and f["requests"] == 6
        and f["requests_per_call"]
        == {
            "1": 1,
            "5": 1,
        }
    )


# ---- B3 -------------------------------------------------------------------------------------------------------------


def test_b3_one_adapter_per_call_oldest_first_up_to_the_cap():
    eng = Rows("batched")
    eng.batch_max_requests = 3
    order = [None, "a", None, "a", None, None, "b"]
    first = hold_first(eng, lambda: eng.answer("held", Q))
    threads = []
    for i, ad in enumerate(order):
        threads.append(threading.Thread(target=eng.answer, args=(f"s{i}", Q, ad)))
        threads[-1].start()
        wait_for_waiting(eng, i + 1)  # in this order
    eng.gate.set()
    for th in threads + [first]:
        th.join(10)
    # each call: its adapter and its requests (the digit of "s<i>", the row's second token)
    calls = [(c["adapter"], [chr(r[1] - 1000) for r in c["rows"]]) for c in eng.calls[1:]]
    # the oldest's adapter each time, oldest first; three of the four base requests, then the fourth
    assert calls == [(None, ["0", "2", "4"]), ("a", ["1", "3"]), (None, ["5"]), ("b", ["6"])]


# ---- B4 -------------------------------------------------------------------------------------------------------------


def test_b4_every_request_of_a_failed_call_gets_the_failure():
    eng = Rows("batched")
    first = hold_first(eng, lambda: eng.answer("held", Q))
    errors = []

    def send(i):
        try:
            eng.answer(f"fails {i}", Q)
        except RuntimeError as e:
            errors.append(str(e))

    threads = [threading.Thread(target=send, args=(i,)) for i in range(3)]
    for th in threads:
        th.start()
    wait_for_waiting(eng, 3)
    fail_next = {"n": 0}
    score = eng.score_prompts

    def failing(rows, *a, **kw):
        fail_next["n"] += 1
        if fail_next["n"] == 1:  # the call after the held one (which runs the original)
            raise RuntimeError("the engine call failed")
        return score(rows, *a, **kw)

    eng.score_prompts = failing
    eng.gate.set()
    for th in threads + [first]:
        th.join(10)
    assert errors == ["the engine call failed"] * 3
    probs, _ = eng.answer("after the failure", Q)  # the server goes on
    assert len(probs) == 1


def test_b4_at_exit_the_waiting_requests_fail_and_nothing_hangs():
    eng = Rows("batched")
    first = hold_first(eng, lambda: eng.answer("held", Q))
    errors = []

    def send():
        try:
            eng.answer("waiting", Q)
        except RuntimeError as e:
            errors.append(str(e))

    th = threading.Thread(target=send)
    th.start()
    wait_for_waiting(eng, 1)
    closer = threading.Thread(target=eng.combiner.close)
    closer.start()
    th.join(10)
    assert errors and "shutting down" in errors[0]
    eng.gate.set()
    closer.join(10)
    first.join(10)
    assert not closer.is_alive() and not first.is_alive()
    with pytest.raises(RuntimeError, match="shutting down"):
        eng.answer("after close", Q)


def test_b4_the_cap_is_at_least_one():
    with pytest.raises(ValueError, match="at least 1"):
        Combiner(Rows("batched"), 0)


# ---- B5 -------------------------------------------------------------------------------------------------------------


def test_b5_each_request_defers_to_its_own_ticket_and_one_new_state_sends_one_warm_up():
    eng = Rows("batched", register=True)
    eng.registrar.background = False  # due registrations go with the next engine call
    first = hold_first(eng, lambda: eng.answer("held", Q))  # too short to register
    tickets, results = {}, {}

    def send(name, state):
        ticket, token = boundary.open_ticket()
        tickets[name] = ticket
        try:
            results[name] = eng.answer(state, Q)
        finally:
            boundary._TICKET.reset(token)  # released by the test, as the server's response would

    a = threading.Thread(target=send, args=("a", "state A, long enough here"))
    b = threading.Thread(target=send, args=("b", "state B, long enough here"))
    c = threading.Thread(target=send, args=("c", "state B, long enough here"))  # the same new state as b
    for th in (a, b, c):
        th.start()
    wait_for_waiting(eng, 3)
    eng.gate.set()
    for th in (a, b, c, first):
        th.join(10)
    facts = eng.registrar.facts()
    assert facts["deferred"] == 2  # A and B (c's is b's, pending once)
    pending = list(eng.registrar._pending.values())
    assert not any(due for _, _, due in pending)  # none due: no response is out
    tickets["a"].release()  # A's response is out: only A's warm-up is due
    due = [w for w, _, d in eng.registrar._pending.values() if d]
    assert len(due) == 1 and due[0][: len("state A")] == [1000 + ord(ch) for ch in "state A"]
    for name in ("a", "b", "c"):
        sb = results[name][1]["state_boundary"]
        assert sb["deferred"] == 1 and sb["registered"] == 0  # each request's own counts
    eng.answer("next state, long enough here", Q)  # the due warm-up goes before this call's questions
    assert eng.warms[-1] == [tuple(due[0])]


# ---- B6 -------------------------------------------------------------------------------------------------------------


class Batched(CertEngine):
    concurrency, batch_max_requests = "serial", 16


def make(mode="serial"):
    eng = Batched()
    eng.concurrency = mode
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, eng


def test_b6_the_fingerprint_holds_the_mode_only_when_batched():
    off = make()[1].stack_facts()
    assert "concurrency" not in off["serving"]  # a serial server's fingerprint is what it was
    on = make("batched")[1].stack_facts()
    assert on["serving"]["concurrency"] == {"mode": "batched", "max_requests": 16}
    assert {k: v for k, v in on["serving"].items() if k != "concurrency"} == off["serving"]


def test_b6_a_certificate_is_concurrency_changed_under_the_other_mode():
    client, so, eng = make()
    r = post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {}})
    assert r.status_code == 200 and r.json()["certificate"]["certified"], r.text
    msg = "0:0.99:x"

    def flip(mode):
        eng.concurrency = mode
        so.__dict__.pop("_stack", None)
        so.__dict__.pop("_stack_sha", None)

    assert client.post("/v1/systemone", json=wire(msg)).json()["answers"]["q1"]["certified"] is True
    flip("batched")
    a = client.post("/v1/systemone", json=wire(msg)).json()["answers"]["q1"]
    assert a["certified"] is False and a["certified_reason"] == "concurrency_changed"
    status = client.get("/v1/tasks").json()["tasks"][0]["status"]
    assert status["state"] == "suspended" and "serving.concurrency" in status["changed"]
    flip("serial")
    assert client.post("/v1/systemone", json=wire(msg)).json()["answers"]["q1"]["certified"] is True


def test_b6_health_reports_the_mode():
    eng = Rows("batched")
    eng.answer("one", Q)
    assert eng.concurrency_facts()["mode"] == "batched" and eng.concurrency_facts()["calls"] == 1
    assert Rows("serial").concurrency_facts() == {"mode": "serial"}


# ---- B7 -------------------------------------------------------------------------------------------------------------


def test_b7_the_flags():
    ap = sv.build_parser()
    a = ap.parse_args(["--base", "gemma-4-12b"])
    assert a.concurrency == "serial" and a.batch_max_requests == 16
    assert ap.parse_args(["--concurrency", "batched", "--batch-max-requests", "8"]).batch_max_requests == 8
    with pytest.raises(SystemExit):
        ap.parse_args(["--concurrency", "parallel"])


@pytest.mark.parametrize(
    "argv,message",
    [
        (["--backend", "mlx", "--model", "{mlx}", "--concurrency", "batched"], "without --concurrency batched"),
        (["--base", "gemma-4-12b", "--concurrency", "batched", "--mode", "packed"], "--mode packed is one request"),
        (["--base", "gemma-4-12b", "--concurrency", "batched", "--batch-max-requests", "0"], "at least 1"),
    ],
)
def test_b7_refused_combinations(argv, message, tmp_path, monkeypatch, capsys):
    import json

    mlx = tmp_path / "mlx"
    mlx.mkdir()
    (mlx / "config.json").write_text(json.dumps({"model_type": "gemma4_unified"}))
    monkeypatch.delenv("DECISIO_BASE", raising=False)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    monkeypatch.setattr("sys.argv", ["decisio", *[a.format(mlx=mlx) for a in argv]])
    with pytest.raises(SystemExit):
        sv.main()
    assert message in capsys.readouterr().err


# ---- B8 -------------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_b8_concurrent_clients_on_the_stand_in_get_the_serial_answers_in_fewer_calls():
    import os

    from decisio.serve.hf_letters import HFLettersEngine

    model = os.environ.get("DECISIO_STAND_IN_MODEL", "Qwen/Qwen3-0.6B-Base")
    eng = HFLettersEngine(model)
    assert isinstance(eng._lock, FairLock)
    calls = []
    score = eng.score_prompts

    def counted(rows, *a, **kw):
        calls.append(len(rows))
        time.sleep(0.05)  # a call long enough for the others to queue behind it
        return score(rows, *a, **kw)

    eng.score_prompts = counted
    q = [{"kind": "choice", "instructions": "Which team?", "options": ["billing", "technical", "sales"]}]
    states = [f"Ticket {i}: the invoice for order {i} was charged twice." for i in range(12)]
    client = TestClient(sv.make_app(eng))
    serial = [client.post("/v1/answer", json={"state": s, "questions": q}).json()["answers"] for s in states]
    eng.concurrency = "batched"
    calls.clear()
    got = in_threads(
        lambda s: client.post("/v1/answer", json={"state": s, "questions": q}).json()["answers"],
        [(s,) for s in states],
    )
    assert got == serial  # the stand-in scores each row alone: batching changes nothing but the number of calls
    assert len(calls) < len(states)
    assert client.get("/health").json()["concurrency"]["mode"] == "batched"


# ---- review fixes ---------------------------------------------------------------------------------------------------


class Override(Rows):
    """An engine that overrides _answer_separate with the old signature, as the MLX build does."""

    def _answer_separate(self, requests, adapter):
        return super()._answer_separate(requests, adapter)


def test_b1_an_engine_with_its_own_answer_separate_is_served_serially_as_before():
    probs, _ = Override("serial").answer("a short state", Q3)
    assert len(probs) == 3


def test_b4_a_request_that_fails_for_itself_fails_alone():
    eng = Rows("batched")
    prepare = eng._prepare_separate

    def poisoned(state, questions):
        if state.startswith("bad"):
            raise ValueError("a question's text merges with the state's last token")
        return prepare(state, questions)

    eng._prepare_separate = poisoned
    first = hold_first(eng, lambda: eng.answer("held", Q))
    out = {}

    def send(name):
        try:
            out[name] = eng.answer(f"{name} state", Q)[0]
        except ValueError as e:
            out[name] = e

    threads = [threading.Thread(target=send, args=(n,)) for n in ("good1", "bad", "good2")]
    for th in threads:
        th.start()
    wait_for_waiting(eng, 3)
    eng.gate.set()
    for th in threads + [first]:
        th.join(10)
    assert isinstance(out["bad"], ValueError)
    want = Rows("serial")
    for name in ("good1", "good2"):
        assert [p.tobytes() for p in out[name]] == [p.tobytes() for p in want.answer(f"{name} state", Q)[0]]


def test_b5_two_requests_about_one_new_state_send_one_warm_up_before():
    eng = Rows("batched", register=True)
    eng.register_boundary = "before"
    first = hold_first(eng, lambda: eng.answer("held", Q))
    threads = [threading.Thread(target=eng.answer, args=("the same new state, long", Q)) for _ in range(2)]
    for th in threads:
        th.start()
    wait_for_waiting(eng, 2)
    eng.gate.set()
    for th in threads + [first]:
        th.join(10)
    assert eng.calls[-1]["warm"] == 1 and len(eng.warms[-1]) == 1  # one warm-up for both
    assert eng.registrar.facts()["deferred"] == 0  # nothing left pending for the second


def test_b2_the_systemone_response_says_the_batch():
    class Shared(Batched):
        def answer(self, state, questions, adapter=None):
            probs, _ = super().answer(state, questions, adapter)
            return probs, {"batch": {"calls": 2, "requests": 3, "rows": 4, "position": 1}}

    eng = Shared()
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    r = TestClient(app).post("/v1/systemone", json=wire("0:0.9:x"))
    assert r.status_code == 200 and r.headers["x-decisio-batch"] == "requests=3; rows=4"
    alone = make()[0].post("/v1/systemone", json=wire("0:0.9:x"))
    assert "x-decisio-batch" not in alone.headers


# ---- Serve's review -------------------------------------------------------------------------------------------------


def test_b9_the_engine_lock_is_first_come_first_served():
    from decisio.serve.combine import FairLock

    lock, order = FairLock(), []
    lock.acquire()
    threads = []
    for i in range(5):
        threads.append(threading.Thread(target=lambda i=i: (lock.acquire(), order.append(i), lock.release())))
        threads[-1].start()
        deadline = time.monotonic() + 5
        while lock.waiting() < i + 1 and time.monotonic() < deadline:
            time.sleep(0.001)
    lock.release()
    for th in threads:
        th.join(5)
    assert order == [0, 1, 2, 3, 4]
    assert lock.acquire(timeout=0.01) and not lock.acquire(blocking=False)
    t = threading.Thread(target=lambda: order.append(lock.acquire(timeout=0.05)))
    t.start()
    t.join(5)
    assert order[-1] is False and lock.waiting() == 0  # a waiter that timed out leaves the queue
    lock.release()
    with pytest.raises(RuntimeError):
        lock.release()


def test_b9_a_thinking_request_takes_the_engine_under_constant_batched_load():
    eng = Rows("batched")
    score = eng.score_prompts

    def slow(rows, *a, **kw):
        time.sleep(0.005)
        return score(rows, *a, **kw)

    eng.score_prompts = slow
    stop = threading.Event()

    def load(i):
        while not stop.is_set():
            eng.answer(f"load {i}", Q)

    clients = [threading.Thread(target=load, args=(i,)) for i in range(8)]
    for c in clients:
        c.start()
    deadline = time.monotonic() + 5
    while len(eng.calls) < 20 and time.monotonic() < deadline:
        time.sleep(0.001)
    before = len(eng.calls)
    t0 = time.monotonic()
    with eng._lock:  # what think(), the image route and the hidden-state readout do
        waited, calls_meanwhile = time.monotonic() - t0, len(eng.calls) - before
    stop.set()
    for c in clients:
        c.join(10)
    assert calls_meanwhile <= 2 and waited < 1.0  # at most the call in progress and one already queued


def test_b9_reads_inside_serial_reads_take_the_serial_path_on_a_batched_server():
    from decisio.serve.combine import serial_reads

    eng = Rows("batched")
    with serial_reads():
        probs, info = eng.answer("a certified read", Q)
    assert "_combiner" not in eng.__dict__ and "batch" not in info  # the combiner never saw it
    eng.answer("an ordinary read", Q)
    assert eng.combiner.facts()["requests"] == 1


class Recorded(Batched):
    """Records, for every engine read, whether it was asked to take the serial path."""

    def __init__(self):
        super().__init__()
        self.serial = []

    def answer(self, state, questions, adapter=None):
        from decisio.serve.combine import reads_serial

        self.serial.append(reads_serial())
        return super().answer(state, questions, adapter)


def test_b9_a_certified_tasks_reads_registration_and_certification_are_serial_and_the_rest_is_not():
    eng = Recorded()
    eng.concurrency = "batched"
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    client = TestClient(app)
    r = post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {}})
    assert r.status_code == 200 and r.json()["certificate"]["certified"], r.text
    assert eng.serial and all(eng.serial)  # every registration and certification read
    eng.serial.clear()
    a = client.post("/v1/systemone", json=wire("0:0.99:x")).json()["answers"]["q1"]
    assert a["certified"] is True and eng.serial == [True]  # the certified task's own read
    eng.serial.clear()
    other = {"type": "choice", "instructions": "Which colour?", "criteria": {"option_0": "red", "option_1": "blue"}}
    body = {"state": {"message": "1:0.8:x"}, "model": "m", "questions": {"q1": other}}
    assert client.post("/v1/systemone", json=body).status_code == 200 and eng.serial == [False]


def test_b9_the_monitor_skips_answers_scored_in_a_shared_call():
    from decisio.serve.monitor import MixMonitor

    base = {"option_counts": [5, 5], "window": 10, "step": 5, "line": 0.5}
    m = MixMonitor(base)
    m.observe(0, True, 0.9)
    m.observe(1, True, 0.9, shared=True)
    f = m.facts()
    assert f["answers"] == 1 and f["shared_skipped"] == 1 and f["option_counts"] == [1, 0]
    assert "shared_skipped" not in MixMonitor(base).facts()  # a serial server's facts are what they were


def test_b9_a_serial_call_of_several_requests_sends_each_its_warm_up_as_before():
    eng = Rows("serial", register=True)
    eng.register_boundary = "before"
    eng.answer_many([("the same new state, long", Q), ("the same new state, long", Q)])
    assert eng.calls[0]["warm"] == 2  # no de-duplication outside a batched call


@pytest.mark.slow
def test_b10_one_engine_images_and_text_on_a_batched_server():
    """`--one-engine`: one engine serves the text and the image route. On a batched server the image request takes the
    engine's fair lock while text requests around it batch: it waits at most for the call in progress and one queued,
    the text clients keep being served, and every answer, the image's and the text's, is the one a serial server gives
    (the stand-in scores each row alone)."""
    import os

    from PIL import Image

    from decisio.serve.hf_letters import HFImageLettersEngine

    model = os.environ.get("DECISIO_IMAGE_TEST_MODEL", "Qwen/Qwen3.5-0.8B-Base")
    eng = HFImageLettersEngine(model, pad_to="block", pad_where="front")
    assert isinstance(eng._lock, FairLock)
    rng = np.random.default_rng(0)
    picture = Image.fromarray(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8), "RGB")
    iq = [{"kind": "noul", "instructions": "Does the photo show a red item?"}]
    tq = [{"kind": "choice", "instructions": "Which team?", "options": ["billing", "technical", "sales"]}]
    states = [f"Ticket {i}: the invoice for order {i} was charged twice." for i in range(6)]
    serial_image = eng.answer("A photo from a listing.", iq, images=[picture])[0]
    serial_text = {s: eng.answer(s, tq)[0] for s in states}

    calls = []
    score = eng.score_prompts

    def slowed(rows, adapter=None, warm=(), skip_cache=False, mm=None, mm_uuids=None):
        if not mm:  # a text call: long enough for requests to queue behind it
            calls.append(len(rows))
            time.sleep(0.03)
        return score(rows, adapter, warm, skip_cache, mm, mm_uuids)

    eng.score_prompts = slowed
    eng.concurrency = "batched"
    stop, got, bad = threading.Event(), [], []

    def client(s):
        while not stop.is_set():
            p = eng.answer(s, tq)[0]
            got.append(s)
            if [x.tobytes() for x in p] != [x.tobytes() for x in serial_text[s]]:
                bad.append(s)

    clients = [threading.Thread(target=client, args=(s,)) for s in states]
    for c in clients:
        c.start()
    deadline = time.monotonic() + 30
    while len(calls) < 10 and time.monotonic() < deadline:
        time.sleep(0.005)
    before = len(calls)
    image = eng.answer("A photo from a listing.", iq, images=[picture])[0]
    meanwhile = len(calls) - before
    served_after = len(got)
    time.sleep(0.2)  # the text clients go on after the image request
    stop.set()
    for c in clients:
        c.join(30)
    assert [x.tobytes() for x in image] == [x.tobytes() for x in serial_image]
    assert meanwhile <= 2  # the call in progress and at most one queued before it
    assert not bad and len(got) > served_after  # text answers unchanged, and still served after the image
    assert max(calls) > 1  # the text requests did batch
