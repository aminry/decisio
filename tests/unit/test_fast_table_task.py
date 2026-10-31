# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The fast table on a task, through the server (`POST /v1/tasks` with `"fast_table": true`). No model and no encoder
download: a fake engine whose answer is set by the request, a hashed bag-of-words stand-in for the sentence encoder,
and a virtual clock the engine and the encoder advance by the time each is set to take, so the timing at registration,
the replicas' stacks and the load on the host are set by the test.

H1  registration keeps the table where it pays: the stored table, the cascade certificate, the report; a confident
    request is answered by the table without the engine, an unsure one by decisio, both certified by their stage
H2  the timing guard: a table that does not save a quarter of the time is not kept, says why, and the task is served
    as it would be without one
H3  refusals: no certify block, a yes/no question, too few fit examples, the extra missing
H4  another stack: a replica whose CPU differs serves through decisio under decisio's own certificate, times the table
    on its own requests, and takes it up when it pays; a replica on another encoder revision cannot use the table
H5  suspension: the sampled-agreement alarm (or the confidence alarm) switches the table off here at once and, through a
    shared task store, on every replica that follows it; a task directory keeps it on the replica
H6  load: the table's rolling median stops beating decisio's by the registered margin and the replica falls back, says
    why, and takes the table back when it times well again
H7  the record: the listing summarises the arrays, the export holds them and an imported task serves the same answers,
    re-certification keeps the table, redoes its thresholds and lifts a suspension

  uv run pytest -q tests/unit/test_fast_table_task.py
"""

import time as real_time
import types
import zlib

import numpy as np
import pytest

pytest.importorskip("sklearn")
obstore = pytest.importorskip("obstore")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from obstore.store import MemoryStore  # noqa: E402

from decisio.serve import fast_table as FT  # noqa: E402
from decisio.serve import systemone as SO  # noqa: E402
from decisio.serve import task_backends as TB  # noqa: E402
from decisio.serve import task_store_shared as S  # noqa: E402
from decisio.serve.systemone import SystemOne, add_routes  # noqa: E402
from decisio.serve.tasks import TaskStore  # noqa: E402

INS = "Which queue does this message belong to?"
CRIT = {"billing": "billing", "shipping": "shipping", "access": "access", "other": "none of these"}
WORDS = {
    "billing": ["invoice", "refund", "charge", "payment", "receipt", "price"],
    "shipping": ["parcel", "courier", "delivery", "tracking", "address", "package"],
    "access": ["password", "login", "account", "reset", "locked", "token"],
}
FILLER = [f"word{i}" for i in range(300)]


class Clock:
    """The virtual time: the engine and the encoder advance it by what they are set to take."""

    def __init__(self):
        self.t = 0.0
        self.decisio_ms = 60.0
        self.table_ms = 5.0

    def __call__(self):
        return self.t

    def spend(self, ms):
        self.t += ms / 1000.0


CLOCK = Clock()


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [zlib.crc32(w.encode()) % 50000 for w in str(text).split()]


class FakeEngine:
    """The state is `top:confidence|text`: the engine puts that confidence on option `top` (an index) and splits the
    rest; a call takes `CLOCK.decisio_ms`."""

    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"
    calls = 0

    def facts(self):
        return {"engine": "fake", "vllm": "1.0", "base": "fake-base", "quantization": None, "mode": "separate"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None):
        type(self).calls += 1
        CLOCK.spend(CLOCK.decisio_ms)
        head = str(state).split("|")[0] if isinstance(state, str) else str(state["message"]).split("|")[0]
        top, conf = head.split(":")[:2]
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            p = np.full(k, (1 - float(conf)) / (k - 1))
            p[int(top) % k] = float(conf)
            out.append(p)
        return out, {}


def fake_embed(texts):
    """A hashed bag of words over the text after the `|`, L2-normalised; each call takes `CLOCK.table_ms`."""
    CLOCK.spend(CLOCK.table_ms)
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        body = str(t).split("|", 1)[-1]
        for w in body.replace('"', " ").replace("}", " ").split():
            out[i, zlib.crc32(w.encode()) % 256] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1.0, n)


@pytest.fixture(autouse=True)
def virtual_time(monkeypatch):
    CLOCK.t, CLOCK.decisio_ms, CLOCK.table_ms = 0.0, 60.0, 5.0
    FakeEngine.calls = 0
    proxy = types.SimpleNamespace(**{k: getattr(real_time, k) for k in dir(real_time) if not k.startswith("_")})
    proxy.perf_counter = CLOCK
    monkeypatch.setattr(SO, "time", proxy)
    monkeypatch.setattr(FT, "embed", fake_embed)
    monkeypatch.setattr(FT, "encoder", lambda: object())
    monkeypatch.setattr(FT, "cpu_model", lambda: "cpu-A")


def message(gold, rng, right=True, conf=None, confident_text=True):
    """A request: decisio's top is right or not, with the confidence the test gives or one drawn; the text names the
    queue by words from its list (and none of them when `confident_text` is false)."""
    n_words = 4 if confident_text else 0
    words = list(rng.choice(WORDS[gold], n_words, replace=False)) if gold in WORDS else []
    words += list(rng.choice(FILLER, 6, replace=False))
    rng.shuffle(words)
    idx = list(CRIT).index(gold)
    top = idx if right else (idx + 1) % len(CRIT)
    if conf is None:
        conf = float(np.clip(rng.normal(0.93 if right else 0.7, 0.03), 0.4, 0.995))
    return f"{top}:{conf:.4f}|" + " ".join(words)


def wire(text):
    return {
        "state": text,
        "model": "m",
        "questions": {"q1": {"type": "choice", "instructions": INS, "criteria": CRIT}},
    }


def examples(n, seed=0, accuracy=0.9):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        gold = str(rng.choice(list(WORDS)))
        clear = bool(rng.random() < 0.7)  # three in ten say nothing the table can use: it defers them
        out.append(
            {
                "request": wire(message(gold, rng, right=bool(rng.random() < accuracy), confident_text=clear)),
                "answer": gold,
            }
        )
    return out


def make(store=None, name="decisio-test", tasks_dir=None, readonly=False):
    if store is not None:
        backend = S.ObjectBackend(store, "memory://t", readonly=readonly)
        ts = TaskStore("fp")
        abstention, report = TB.load_at_start(ts, backend)
        so = SystemOne(
            FakeEngine(), name, task_store=ts, abstention_tasks=abstention, task_backend=backend, task_report=report
        )
        backend_ = backend
    else:
        so = SystemOne(FakeEngine(), name, task_store=TaskStore("fp"))
        backend_ = None
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, backend_


def make_dir(path, name="decisio-test"):
    """A server with a task directory (`--tasks-dir`): what is in it loads at start."""
    backend = TB.DirBackend(path)
    ts = TaskStore("fp")
    abstention, report = TB.load_at_start(ts, backend)
    so = SystemOne(
        FakeEngine(), name, task_store=ts, abstention_tasks=abstention, task_backend=backend, task_report=report
    )
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, backend


def register(client, n=800, seed=0, **extra):
    body = {"id": "queue", "examples": examples(n, seed), "certify": {"alpha": 0.1, "delta": 0.1}, "fast_table": True}
    body.update(extra)
    return client.post("/v1/tasks", json=body)


def ask(client, gold, right=True, conf=None, seed=1, confident_text=True):
    msg = message(gold, np.random.default_rng(seed), right, conf, confident_text)
    r = client.post("/v1/systemone", json=wire(msg))
    assert r.status_code == 200, r.text
    return r.json()["answers"]["q1"]


# ---- H1 ----------------------------------------------------------------------------------------------------------


def test_h1_the_table_is_kept_where_it_pays_and_answers_what_it_is_sure_of_without_the_model():
    client, so, _ = make()
    r = register(client)
    assert r.status_code == 200, r.text
    t = r.json()
    rep = t["fast_table_report"]
    assert rep["kept"] and rep["reason"] is None and rep["guard"]["saving"] >= FT.MIN_SAVING
    assert t["fast_table"]["encoder"] == FT.ENCODER and t["fast_table"]["revision"] == FT.REVISION
    assert "b64" not in str(t["fast_table"]["coef"]) and t["fast_table"]["coef"]["shape"]
    c = t["certificate"]
    assert c["certified"] and c["table_threshold"] and c["stages"]["table"]["certified_n"] > 0
    assert c["decisio_only"]["threshold"] is not None and "fast table" in c["summary"]
    assert t["status"]["fast_table"]["state"] == "active" and t["status"]["fast_table"]["uses_table"]
    calls = FakeEngine.calls
    a = ask(client, "billing", right=False, conf=0.4)  # decisio is unsure and wrong; the text is clear
    assert a["answered_by"] == "table" and a["choice"] == "billing" and a["certified"] is True
    assert a["certificate"] == c["id"] and FakeEngine.calls == calls  # the model was not asked
    d = ask(client, "billing", right=True, conf=0.95, confident_text=False)  # no clear words: the table defers
    assert d["answered_by"] == "decisio" and FakeEngine.calls == calls + 1
    assert d["choice"] == "billing" and d["certified"] is True  # decisio's own confidence reaches its stage's threshold
    u = ask(client, "billing", right=True, conf=0.45, confident_text=False)
    assert u["answered_by"] == "decisio" and u["certified"] is False and u["certified_reason"] == "below_threshold"


def test_h1_the_served_check_reads_zero_on_the_same_stack():
    client, so, _ = make()
    c = register(client).json()["certificate"]
    sc = c["served_check"]
    assert sc["max_shift"] == 0 and sc["crossed_threshold"] == 0 and sc["top_answer_changed"] == 0


# ---- H2 ----------------------------------------------------------------------------------------------------------


def test_h2_a_table_that_does_not_save_a_quarter_of_the_time_is_not_kept_and_says_why():
    CLOCK.table_ms = 55.0  # nearly decisio's own time: no saving
    client, so, _ = make()
    r = register(client)
    assert r.status_code == 200, r.text
    t = r.json()
    rep = t["fast_table_report"]
    assert not rep["kept"] and "under the 25% required" in rep["reason"] and rep["guard"]["saving"] < 0.25
    assert "fast_table" not in t and t["certificate"]["certified"] and "table_threshold" not in t["certificate"]
    assert t["status"]["fast_table"] == {"state": "off", "uses_table": False, "reason": rep["reason"]}
    calls = FakeEngine.calls
    a = ask(client, "billing", right=True, conf=0.95)
    assert "answered_by" not in a and FakeEngine.calls == calls + 1


# ---- H3 ----------------------------------------------------------------------------------------------------------


def test_h3_refusals():
    client, so, _ = make()
    body = {"id": "queue", "examples": examples(800), "fast_table": True}
    r = client.post("/v1/tasks", json=body)
    assert r.status_code == 422 and "certify" in r.json()["detail"]
    r = register(client, fast_table="yes")
    assert r.status_code == 422 and "true or false" in r.json()["detail"]
    r = register(client, n=150)  # 75 fit examples
    assert r.status_code == 422 and "at least 100" in r.json()["detail"]
    assert not so.task_store.by_key

    def missing():
        raise FT.TableUnavailable("the fast table needs the `table` extra: pip install 'decisio[table]'")

    import decisio.serve.fast_table as module

    module_encoder = module.encoder
    module.encoder = missing
    try:
        r = register(client)
    finally:
        module.encoder = module_encoder
    assert r.status_code == 422 and "decisio[table]" in r.json()["detail"] and not so.task_store.by_key
    dyn = {"id": "dyn", "options": "dynamic", "match": "instructions", "examples": examples(800), "fast_table": True}
    dyn["certify"] = {}
    r = client.post("/v1/tasks", json=dyn)
    assert r.status_code == 422 and "change with every request" in r.json()["detail"] and not so.task_store.by_key
    yn = {"id": "yn", "examples": [], "certify": {}, "fast_table": True}
    yn["examples"] = [
        {"request": {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}, "answer": True}
        for _ in range(300)
    ]
    r = client.post("/v1/tasks", json=yn)
    assert r.status_code == 422 and "choice" in r.json()["detail"]


# ---- H4 ----------------------------------------------------------------------------------------------------------


def test_h4_a_replica_on_another_cpu_serves_through_decisio_until_it_has_timed_the_table_itself(monkeypatch):
    store = MemoryStore()
    a, so_a, _ = make(store, "a")
    assert register(a).status_code == 200
    monkeypatch.setattr(FT, "cpu_model", lambda: "cpu-B")
    b, so_b, bb = make(store, "b")
    assert bb.sync_once()["applied"] == []  # it loaded the task at start
    st = b.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert st["state"] == "timing" and not st["uses_table"] and "cpu" in st["reason"]
    calls = FakeEngine.calls
    x = ask(b, "billing", right=False, conf=0.4)  # the table would be sure; this replica asks the model
    assert x["answered_by"] == "decisio" and FakeEngine.calls == calls + 1
    cert = b.get("/v1/tasks/queue").json()["certificate"]
    assert x["certified"] is False and x["certified_reason"] == "below_threshold"  # decisio's own threshold: 0.4 is low
    y = ask(b, "billing", right=True, conf=0.97)
    assert y["answered_by"] == "decisio" and y["certified"] is True and y["certificate"] == cert["id"]
    for i in range(FT.SHADOW_EVERY * FT.SHADOW_N):  # this replica times the table on its own requests
        ask(b, "access", right=True, conf=0.95, seed=100 + i, confident_text=False)
    st = b.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert st["state"] == "active" and st["retimed"]["saving"] >= FT.MIN_SAVING
    assert a.get("/v1/tasks/queue").json()["status"]["fast_table"]["state"] == "active"  # A never stopped
    z = ask(b, "billing", right=False, conf=0.4)
    assert z["answered_by"] == "table"


def test_h4_a_replica_that_times_the_table_slow_keeps_serving_through_decisio(monkeypatch):
    store = MemoryStore()
    a, so_a, _ = make(store, "a")
    assert register(a).status_code == 200
    monkeypatch.setattr(FT, "cpu_model", lambda: "cpu-B")
    b, so_b, _ = make(store, "b")
    CLOCK.table_ms = 58.0
    for i in range(FT.SHADOW_EVERY * FT.SHADOW_N):
        ask(b, "access", right=True, conf=0.95, seed=200 + i, confident_text=False)
    st = b.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert st["state"] == "timing" and st["last_timing"]["saving"] < FT.MIN_SAVING


def test_h4_another_encoder_revision_cannot_use_the_table(monkeypatch):
    store = MemoryStore()
    a, _, _ = make(store, "a")
    assert register(a).status_code == 200
    monkeypatch.setattr(FT, "REVISION", "0" * 40)
    b, _, _ = make(store, "b")
    st = b.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert st["state"] == "unavailable" and "register the task again" in st["reason"]
    assert ask(b, "billing", right=True, conf=0.97)["answered_by"] == "decisio"


# ---- H5 ----------------------------------------------------------------------------------------------------------


def feed_disagreement(client, n, seed=0):
    """Table answers that decisio would have answered otherwise: the text is clear, decisio's top is another queue."""
    for i in range(n):
        ask(client, "access", right=False, conf=0.9, seed=seed + i)


def test_h5_the_agreement_alarm_switches_the_table_off_here_and_on_every_replica_that_shares_the_store():
    store = MemoryStore()
    a, so_a, ba = make(store, "a")
    b, so_b, bb = make(store, "b")
    assert register(a).status_code == 200
    bb.sync_once()
    assert b.get("/v1/tasks/queue").json()["status"]["fast_table"]["state"] == "active"
    # the certification examples' decisio is right where the table is sure; now it is not: sampled reads disagree
    feed_disagreement(a, FT.SAMPLE_EVERY * 40)
    st = a.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert st["state"] == "suspended" and ("differ from the table's" in st["reason"] or "confidences" in st["reason"])
    t = so_a.task_store.snapshot("queue")
    assert t["fast_table"]["suspended"]["shared"] is True
    bb.sync_once()  # one poll
    sb = b.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert sb["state"] == "suspended" and "suspended by" in sb["reason"]
    calls = FakeEngine.calls
    x = ask(b, "billing", right=True, conf=0.97)
    assert x["answered_by"] == "decisio" and FakeEngine.calls == calls + 1 and x["certified"] is True  # decisio alone


def test_h5_two_replicas_suspending_at_once_both_end_suspended_after_a_poll():
    store = MemoryStore()
    a, so_a, ba = make(store, "a")
    b, so_b, bb = make(store, "b")
    assert register(a).status_code == 200
    bb.sync_once()
    feed_disagreement(a, FT.SAMPLE_EVERY * 40)  # a writes its suspension first
    feed_disagreement(b, FT.SAMPLE_EVERY * 40, seed=5000)  # b has not followed yet: its write is refused (409)
    t = so_b.task_store.snapshot("queue")
    assert t["fast_table"]["suspended"]["shared"] is False and "write_error" in t["fast_table"]["suspended"]
    assert b.get("/v1/tasks/queue").json()["status"]["fast_table"]["state"] == "suspended"
    bb.sync_once()  # b follows a's version
    for c in (a, b):
        assert c.get("/v1/tasks/queue").json()["status"]["fast_table"]["state"] == "suspended"
    assert so_b.task_store.snapshot("queue")["fast_table"]["suspended"]["shared"] is True


def test_h5_without_a_shared_store_the_suspension_stays_on_the_replica():
    a, so_a, _ = make()
    assert register(a).status_code == 200
    feed_disagreement(a, FT.SAMPLE_EVERY * 40)
    assert a.get("/v1/tasks/queue").json()["status"]["fast_table"]["state"] == "suspended"
    t = so_a.task_store.snapshot("queue")
    assert t["fast_table"]["suspended"]["shared"] is False


def test_h5_a_replica_that_cannot_write_suspends_for_itself_and_says_so():
    store = MemoryStore()
    a, so_a, _ = make(store, "a")
    assert register(a).status_code == 200
    c, so_c, bc = make(store, "c", readonly=True)
    feed_disagreement(c, FT.SAMPLE_EVERY * 40)
    st = c.get("/v1/tasks/queue").json()["status"]["fast_table"]
    t = so_c.task_store.snapshot("queue")
    assert st["state"] == "suspended" and t["fast_table"]["suspended"]["shared"] is False
    assert "write_error" in t["fast_table"]["suspended"]
    assert a.get("/v1/tasks/queue").json()["status"]["fast_table"]["state"] == "active"


# ---- H6 ----------------------------------------------------------------------------------------------------------


def test_h6_the_table_falls_back_when_it_stops_beating_decisio_under_load_and_comes_back():
    client, so, _ = make()
    assert register(client).status_code == 200
    CLOCK.table_ms = 50.0  # the host is busy: the encoder takes nearly decisio's time
    for i in range(FT.MIN_LATENCY_SAMPLES + 2):
        ask(client, "billing", right=False, conf=0.4, seed=300 + i)
    st = client.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert st["state"] == "timing" and "under this replica's load" in st["reason"] and not st["uses_table"]
    assert ask(client, "billing", right=False, conf=0.4)["answered_by"] == "decisio"
    CLOCK.table_ms = 5.0  # the load is gone
    for i in range(FT.SHADOW_EVERY * FT.SHADOW_N):
        ask(client, "access", right=True, conf=0.95, seed=400 + i, confident_text=False)
    st = client.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert st["state"] == "active" and st["retimed"]["kept"]


def test_h6_a_table_that_keeps_its_margin_stays_on():
    client, so, _ = make()
    assert register(client).status_code == 200
    for i in range(FT.MIN_LATENCY_SAMPLES * 3):
        ask(client, "billing", right=False, conf=0.4, seed=500 + i)
    st = client.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert st["state"] == "active" and st["table_ms_samples"] >= FT.MIN_LATENCY_SAMPLES


# ---- H7 ----------------------------------------------------------------------------------------------------------


def test_h7_the_listing_summarises_the_export_holds_the_arrays_and_an_import_serves_the_same_answers():
    client, so, _ = make()
    r = register(client)
    listed = client.get("/v1/tasks").json()["tasks"][0]
    full = client.get("/v1/tasks?full=1").json()["tasks"][0]
    assert "b64" not in str(listed["fast_table"]["coef"]) and "b64" in full["fast_table"]["coef"]
    other, so2, _ = make()
    assert other.post("/v1/tasks/import", json={"tasks": [full]}).status_code == 200, "import failed"
    for g, right, conf in (("billing", False, 0.4), ("shipping", True, 0.97), ("access", True, 0.6)):
        a, b = ask(client, g, right, conf, seed=7), ask(other, g, right, conf, seed=7)
        assert a == b
    assert r.json()["certificate"]["id"] == other.get("/v1/tasks/queue").json()["certificate"]["id"]


def test_h7_re_certification_keeps_the_table_redoes_its_thresholds_and_lifts_a_suspension():
    client, so, _ = make()
    first = register(client).json()
    feed_disagreement(client, FT.SAMPLE_EVERY * 40)
    assert client.get("/v1/tasks/queue").json()["status"]["fast_table"]["state"] == "suspended"
    new = examples(300, seed=99)
    r = client.post("/v1/tasks/queue/certify", json={"examples": new})
    assert r.status_code == 200, r.text
    t = r.json()
    assert (
        t["certificate"]["id"] != first["certificate"]["id"] and t["fast_table"]["coef"] == first["fast_table"]["coef"]
    )
    assert t["status"]["fast_table"]["state"] == "active" and "suspended" not in t["fast_table"]


def test_h7_a_task_directory_keeps_the_table_and_its_suspension_across_a_restart(tmp_path):
    a, so_a, ba = make_dir(tmp_path / "d")
    first = register(a).json()
    ba.close()
    b, so_b, bb = make_dir(tmp_path / "d")
    st = b.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert (
        st["state"] == "active" and b.get("/v1/tasks/queue").json()["certificate"]["id"] == first["certificate"]["id"]
    )
    assert ask(b, "billing", right=False, conf=0.4)["answered_by"] == "table"
    feed_disagreement(b, FT.SAMPLE_EVERY * 40)
    assert b.get("/v1/tasks/queue").json()["status"]["fast_table"]["state"] == "suspended"
    bb.close()
    c, so_c, bc = make_dir(tmp_path / "d")
    st = c.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert st["state"] == "suspended" and "suspended by" in st["reason"]  # the directory is this server's own store
    assert ask(c, "billing", right=True, conf=0.97)["answered_by"] == "decisio"
    bc.close()


def test_h7_a_table_fitted_under_another_encoder_revision_is_not_re_certified(monkeypatch):
    client, so, _ = make()
    assert register(client).status_code == 200
    monkeypatch.setattr(FT, "REVISION", "1" * 40)
    r = client.post("/v1/tasks/queue/certify", json={"examples": examples(300, seed=98)})
    assert r.status_code == 422 and "register the task again" in r.json()["detail"]
