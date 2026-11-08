# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The fast table that learns from traffic (`POST /v1/tasks` with `{"id", "question", "fast_table": {"learn": true}}`,
decisio.serve.fast_table_learn). No model and no network: the fake engine, the static and encoder stand-ins and the
virtual clock of tests/unit/test_fast_table_task.py and tests/unit/test_fast_table_kinds.py; refits and the encoder
queue run when a test asks (`FTL.BACKGROUND` off), except where a test starts the threads; exploration is seeded.

L1   a learning task starts with no table: decisio answers, nothing is certified; refusals (a score question, raw
     text, a bad explore share or saving, a dynamic task)
L2   rows are logged with their kind, the features of the students the task may serve and decisio's distribution,
     float16, and no text (the segment files hold none of the request's words)
L3   a refit certifies and promotes a table once there are enough rows; its answers carry certificate_kind agreement
     and its own certificate, and the model is not asked
L4   exploration sends about `explore` of the would-be table answers to decisio (seeded), logged as explored
L5   a refit that does not certify leaves the live table live
L6   a slower refit is not promoted (and the oldest rows past max_rows are dropped at refit)
L7   rollback restores an earlier version, refuses an unknown one and one whose features this server cannot compute
L8   the low-confidence-share guard alone suspends the table on a drifted stream; learning goes on
L9   the disagreement CUSUM alone suspends the table on a drifted stream; learning goes on
L10  the certificate is made on rows logged independently of the table only: deferred rows that would certify do not
L11  a tasks directory: a restart resumes the live version and the log (segment files and their index)
L12  without a tasks directory the status says the rows are in memory and lost at restart; /health's summary
L13  the encoder's background queue drops rows under load without making a request wait, and rows without encoder
     features are not fitted for the encoder
L14  a labelled task that also learns: its examples join the fit, its certificate is gold, answers say so
L15  the worker refits on its own thread once a task has REFIT_ROWS new rows

  uv run pytest -q tests/unit/test_fast_table_learn.py
"""

import re
import threading
import time as real_time

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("obstore")
pytest.importorskip("tokenizers")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from test_fast_table_kinds import STATIC, students  # noqa: E402,F401  (the static and encoder stand-ins: autouse)
from test_fast_table_task import (  # noqa: E402
    CLOCK,
    CRIT,
    INS,
    WORDS,
    FakeEngine,
    examples,
    fake_embed,
    message,
    virtual_time,  # noqa: F401  (the virtual clock: autouse)
    wire,
)

from decisio.serve import fast_table as FT  # noqa: E402
from decisio.serve import fast_table_learn as FTL  # noqa: E402
from decisio.serve import task_backends as TB  # noqa: E402
from decisio.serve.systemone import SystemOne, add_routes  # noqa: E402
from decisio.serve.tasks import TaskStore  # noqa: E402

QUESTION = {"type": "choice", "instructions": INS, "criteria": CRIT}


@pytest.fixture(autouse=True)
def learning_by_hand(monkeypatch):
    monkeypatch.setattr(FTL, "BACKGROUND", False)
    monkeypatch.setattr(FTL, "SEED", 0)


def make(tasks_dir=None):
    backend = None
    if tasks_dir is not None:
        backend = TB.DirBackend(tasks_dir)
        ts = TaskStore("fp")
        abstention, report = TB.load_at_start(ts, backend)
        so = SystemOne(
            FakeEngine(), "decisio-test", task_store=ts, abstention_tasks=abstention, task_backend=backend,
            task_report=report,
        )  # fmt: skip
    else:
        so = SystemOne(FakeEngine(), "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, backend


def learn(client, task_id="queue", question=None, **fast_table):
    body = {"id": task_id, "question": question or QUESTION, "fast_table": {"learn": True, **fast_table}}
    return client.post("/v1/tasks", json=body)


# a text that names its queue (billing) in every word, on which decisio is unsure and wrong (shipping); and one with no
# words, which no table can read, on which decisio is sure
SURE = "1:0.4|invoice refund charge payment receipt price"
UNSURE = "0:0.97|"


def ask(client, text):
    r = client.post("/v1/systemone", json=wire(text))
    assert r.status_code == 200, r.text
    return r.json()["answers"]["q1"]


def traffic(client, n, seed=0, right=1.0, clear=0.7, conf=None):
    """n requests of the task: decisio's top is the text's queue with probability `right`; `clear` of the texts name
    their queue (the rest the table cannot read). Returns [(answer, clear)]."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        gold = str(rng.choice(list(WORDS)))
        is_clear = bool(rng.random() < clear)
        msg = message(gold, rng, right=bool(rng.random() < right), conf=conf, confident_text=is_clear)
        r = client.post("/v1/systemone", json=wire(msg))
        assert r.status_code == 200, r.text
        out.append((r.json()["answers"]["q1"], is_clear))
    return out


def status(client, task_id="queue"):
    return client.get(f"/v1/tasks/{task_id}").json()["status"]["fast_table"]


def live_table(client, so, n=1200, **fast_table):
    """A learning task whose first refit promoted version 1."""
    assert learn(client, **fast_table).status_code == 200
    traffic(client, n)
    so.learning.drain()
    out = so.learning.refit_task("queue")
    assert out["outcome"] == "promoted", out
    return out


def log_rows(so, n, seed=0, kind=FTL.NO_TABLE, right=1.0, clear=0.7, would=False):
    """Rows logged straight into the task's learner, as the request path logs them (the static student's features
    are computed and timed here)."""
    L = so.learning.learners["queue"]
    keys = list(CRIT)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        gold = str(rng.choice(list(WORDS)))
        is_clear = bool(rng.random() < clear)
        text = message(gold, rng, right=True, confident_text=is_clear)
        top = keys.index(gold) if rng.random() < right else int(rng.integers(len(keys)))
        p = np.full(len(keys), 0.05 / (len(keys) - 1))
        p[top] = 0.95
        L.record(text, {}, p, top, 60.0, kind, 0.99 if would else float("nan"), would)


# ---- L1 -------------------------------------------------------------------------------------------------------------


def test_l1_a_learning_task_starts_with_no_table_and_decisio_answers():
    client, so, _ = make()
    r = learn(client)
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["learn"]["request"] == {"kind": "static", "static_model": "base", "min_saving": FT.MIN_SAVING}
    assert t["learn"]["explore"] == 0.02
    assert t["learn"]["max_rows"] == 200_000 and t["n_examples"] == 0 and "certificate" not in t
    st = t["status"]["fast_table"]
    assert st["mode"] == "learning" and st["state"] == "learning" and st["version"] == {"live": None, "kept": []}
    assert st["rows"]["kept"] == 0 and st["kind"] == {"asked": "static", "static_model": "base", "chosen": None}
    calls = FakeEngine.calls
    a = ask(client, UNSURE)
    assert a["answered_by"] == "decisio" and a["choice"] == "billing" and FakeEngine.calls == calls + 1
    assert "certified" not in a and "certificate_kind" not in a
    assert status(client)["rows"]["by_kind"]["no_table"] == 1


@pytest.mark.parametrize(
    "body, words",
    [
        ({"question": {"type": "score", "instructions": "How bad?", "criteria": ["a", "b"]}}, "choice and yes/no"),
        ({"fast_table": {"learn": True, "min_saving": 0.95}}, "fast_table.min_saving"),
        ({"fast_table": {"learn": True, "retain_text": {"days": 7}}}, "retain_text: keeping request text is not"),
        ({"fast_table": {"learn": True, "explore": 0.9}}, "fast_table.explore"),
        ({"fast_table": {"learn": True, "explore": "a lot"}}, "fast_table.explore"),
        ({"fast_table": {"learn": True, "max_rows": 10}}, "fast_table.max_rows"),
        ({"fast_table": {"learn": True, "kind": "hashed"}}, "'hashed' is not available"),
        ({"fast_table": {"learn": "yes"}}, "fast_table.learn: true or false"),
        ({"fast_table": {"explore": 0.1}}, 'for a table that learns from traffic only ("learn": true)'),
        ({"question": "billing?"}, "question: a wire question"),
        ({"options": "dynamic"}, "change with every request"),
        ({"certify": {"holdout": 0.5}}, "do not apply to a learning task"),
        ({"adapter": "metask"}, '"adapter" is not accepted when registering'),
    ],
)
def test_l1_refusals(body, words):
    client, so, _ = make()
    full = {"id": "queue", "question": QUESTION, "fast_table": {"learn": True}, **body}
    r = client.post("/v1/tasks", json=full)
    assert r.status_code == 422 and words in r.json()["detail"], r.text
    assert not so.task_store.by_key and not so.learning.learners


# ---- L2 -------------------------------------------------------------------------------------------------------------


def test_l2_rows_hold_their_kind_features_and_decisio_distribution_and_no_text(tmp_path):
    client, so, backend = make(tmp_path / "tasks")
    assert learn(client).status_code == 200
    sent = traffic(client, 120)
    L = so.learning.learners["queue"]
    rows = L.log.read()
    assert len(rows["id"]) == 120 and set(rows["kind"].tolist()) == {FTL.NO_TABLE}
    assert rows["f_static"].dtype == np.float16 and rows["f_static"].shape == (120, 256)
    assert rows["has_static"].all() and np.allclose(rows["ms_static"], STATIC[0])
    assert rows["dist"].dtype == np.float16 and rows["dist"].shape == (120, len(CRIT))
    assert np.allclose(rows["ms_decisio"], CLOCK.decisio_ms) and (rows["stack"] == rows["stack"][0]).all()
    tops = [list(CRIT).index(a["choice"]) for a, _ in sent]
    assert rows["top"].tolist() == tops and np.allclose(rows["conf"], [a["confidence"] for a, _ in sent], atol=0.2)
    assert "f_encoder" not in rows  # a static task logs the static student only
    L.log.flush()
    learn_dir = tmp_path / "tasks" / "learn" / FTL.task_digest("queue")
    segs = sorted((learn_dir / "log").glob("*.seg"))
    assert len(segs) == 1 and re.fullmatch(r"\d{13}-\d{6}\.seg", segs[0].name)
    assert (learn_dir / "log" / "index.json").exists() and (learn_dir / "state.json").exists()
    data = b"".join(p.read_bytes() for p in learn_dir.rglob("*") if p.is_file())
    for w in ("invoice", "parcel", "password", "word1", "word2"):
        assert w.encode() not in data  # no text, in any file
    st = status(client)
    assert st["rows"]["kept"] == 120 and st["rows"]["bytes"] > 0 and "segment files under" in st["rows"]["stored"]
    backend.close()


def test_l2_a_live_table_logs_deferred_explored_and_spot_check_rows_and_not_its_own_answers():
    client, so, _ = make()
    live_table(client, so, explore=0.1)
    before = dict(status(client)["rows"]["by_kind"])
    sent = traffic(client, 600, seed=3)
    after = status(client)["rows"]["by_kind"]
    by_table = sum(a["answered_by"] == "table" for a, _ in sent)
    new = {k: after[k] - before[k] for k in after}
    assert new["no_table"] == 0 and new["deferred"] > 0 and new["explored"] > 0 and new["spot_check"] > 0
    assert new["spot_check"] == by_table // FT.SAMPLE_EVERY  # the table's other answers are not logged
    assert sum(new.values()) == 600 - by_table + new["spot_check"]


# ---- L3 -------------------------------------------------------------------------------------------------------------


def test_l3_a_refit_certifies_and_promotes_a_table_whose_answers_carry_its_agreement_certificate():
    client, so, _ = make()
    assert learn(client, explore=0.0).status_code == 200  # no exploration: every sure answer below is the table's
    traffic(client, 60)
    early = so.learning.refit_task("queue")
    assert early["outcome"] == "too_few_rows" and "at least 100" in early["reason"]
    traffic(client, 1140, seed=1)
    out = so.learning.refit_task("queue")
    assert out["outcome"] == "promoted" and out["version"] == 1 and out["kind"] == "static"
    st = status(client)
    assert st["state"] == "active" and st["version"]["live"] == 1 and st["kind"]["chosen"] == "static"
    c = st["certificate"]
    assert c["kind"] == "agreement" and c["alpha"] == 0.05 and c["delta"] == 0.1 and c["threshold"] is not None
    assert c["n"] >= 59 and c["upper_bound"] <= 0.05 and set(c["rows"]) == {"no_table", "explored"}
    assert st["saving"] >= FT.MIN_SAVING and st["coverage"]["certified"] > 0.4
    assert st["refit"]["last"]["outcome"] == "promoted"
    calls = FakeEngine.calls
    a = ask(client, SURE)  # a clear text; decisio would be unsure and wrong
    assert a["answered_by"] == "table" and a["choice"] == "billing" and FakeEngine.calls == calls
    assert a["certified"] is True and a["certificate_kind"] == "agreement" and a["certificate"] == c["id"]
    d = ask(client, UNSURE)  # nothing the table can read: decisio answers
    assert d["answered_by"] == "decisio" and "certified" not in d and FakeEngine.calls == calls + 1


# ---- L4 -------------------------------------------------------------------------------------------------------------


def test_l4_exploration_sends_about_explore_of_the_would_be_table_answers_to_decisio():
    client, so, _ = make()
    live_table(client, so, explore=0.1)
    L = so.learning.learners["queue"]
    v = L.live
    before = status(client)["explore"]["explored_would_answer"]
    sent = traffic(client, 1500, seed=4)
    would = L.would_answer
    explored = status(client)["explore"]["explored_would_answer"] - before
    assert would > 800 and abs(explored / would - 0.1) < 0.03, (explored, would)
    by_table = sum(a["answered_by"] == "table" for a, _ in sent)
    assert by_table == would - explored  # an explored request the table was sure of is answered by decisio
    st = status(client)
    assert st["explore"]["fraction"] == 0.1 and st["explore"]["cost"] == round(
        0.1 * v.d["certificate"]["table_share"], 4
    )


# ---- L5 -------------------------------------------------------------------------------------------------------------


def test_l5_a_refit_that_does_not_certify_leaves_the_live_table_live(monkeypatch):
    client, so, _ = make()
    first = live_table(client, so, explore=0.0)
    cert = status(client)["certificate"]["id"]
    traffic(client, 200, seed=5)
    monkeypatch.setattr(FTL.certify, "threshold", lambda *a, **k: np.inf)  # no threshold holds on these rows
    out = so.learning.refit_task("queue")
    assert out["outcome"] == "not_certified" and "the live version stays live" in out["reason"]
    st = status(client)
    assert st["state"] == "active" and st["version"]["live"] == first["version"] == 1
    assert st["certificate"]["id"] == cert and st["refit"]["last"]["outcome"] == "not_certified"
    assert ask(client, SURE)["answered_by"] == "table"


# ---- L6 -------------------------------------------------------------------------------------------------------------


def test_l6_a_slower_refit_is_not_promoted_and_the_oldest_rows_go_at_refit():
    client, so, _ = make()
    live_table(client, so, max_rows=1200)
    live = status(client)
    STATIC[0] = 12.0  # the static student is slower on this host now
    log_rows(so, 1200, seed=6)  # rows read while the table was off: every one counts for the certificate
    out = so.learning.refit_task("queue")
    assert out["outcome"] == "not_promoted" and "not more than the live version 1's" in out["reason"], out
    assert out["kinds"][0]["saving"] < live["saving"]
    st = status(client)
    assert st["version"]["live"] == 1 and st["state"] == "active" and len(st["version"]["kept"]) == 1
    assert st["rows"]["kept"] == 1200 and out["rows"] == 1200  # the first 1200 rows were dropped at the refit
    rows = so.learning.learners["queue"].log.read()
    assert np.allclose(rows["ms_static"], 12.0)


# ---- L7 -------------------------------------------------------------------------------------------------------------


def test_l7_rollback_restores_an_earlier_version(monkeypatch):
    client, so, _ = make()
    STATIC[0] = 8.0
    live_table(client, so, max_rows=1200, explore=0.0)
    v1 = status(client)["certificate"]["id"]
    STATIC[0] = 0.2  # faster now: the next refit saves more
    log_rows(so, 1200, seed=7)
    assert so.learning.refit_task("queue")["outcome"] == "promoted"
    st = status(client)
    assert st["version"]["live"] == 2 and [v["version"] for v in st["version"]["kept"]] == [1, 2]
    v2 = st["certificate"]["id"]
    assert ask(client, SURE)["certificate"] == v2
    r = client.post("/v1/tasks/queue/fast_table/rollback", json={"version": 1})
    assert r.status_code == 200, r.text
    assert r.json()["status"]["fast_table"]["version"]["live"] == 1
    assert ask(client, SURE)["certificate"] == v1
    r = client.post("/v1/tasks/queue/fast_table/rollback", json={"version": 9})
    assert r.status_code == 422 and "no version 9 is kept" in r.json()["detail"]
    assert client.post("/v1/tasks/queue/fast_table/rollback", json={"version": "1"}).status_code == 422
    assert client.post("/v1/tasks/nope/fast_table/rollback", json={"version": 1}).status_code == 404
    monkeypatch.setitem(FT.STATIC_MODELS["base"], "revision", "0" * 40)  # this server's static student is another
    r = client.post("/v1/tasks/queue/fast_table/rollback", json={"version": 2})
    assert r.status_code == 422 and "register the task again" in r.json()["detail"]
    assert status(client)["version"]["live"] == 1


def test_l7_a_labelled_task_without_learning_has_no_versions():
    client, so, _ = make()
    body = {"id": "queue", "examples": examples(800), "certify": {"alpha": 0.1, "delta": 0.1}, "fast_table": True}
    assert client.post("/v1/tasks", json=body).status_code == 200
    r = client.post("/v1/tasks/queue/fast_table/rollback", json={"version": 1})
    assert r.status_code == 422 and "does not learn" in r.json()["detail"]


# ---- L8, L9 ------------------------------------------------------------------------------------------------------


def test_l8_the_low_confidence_share_alone_suspends_the_table_and_learning_goes_on():
    client, so, _ = make()
    live_table(client, so)
    st = status(client)
    assert st["guards"]["low_confidence_share"]["certified_share"] < 0.5
    # the traffic moves to texts the table cannot read; decisio answers them as before, so nothing disagrees
    sent = traffic(client, FTL.LOWCONF_WINDOW + 20, seed=8, clear=0.0)
    st = status(client)
    g = st["guards"]
    assert st["state"] == "suspended" and st["suspended"]["alarm"] == "low_confidence_share"
    assert g["low_confidence_share"]["alarm"] and not g["disagreement_cusum"]["alarm"]
    assert g["disagreement_cusum"]["statistic"] < FTL.CUSUM_LIMIT and "requests had a table confidence" in st["reason"]
    assert sum(a["answered_by"] == "table" for a, _ in sent) < len(sent) * 0.3
    logged = st["rows"]["logged"]
    no_table = st["rows"]["by_kind"]["no_table"]
    after = traffic(client, 50, seed=9)
    assert all(a["answered_by"] == "decisio" for a, _ in after)
    st = status(client)
    assert st["rows"]["logged"] == logged + 50 and st["rows"]["by_kind"]["no_table"] == no_table + 50
    out = so.learning.refit_task("queue")  # refits go on, on the new rows too
    assert out["outcome"] in ("promoted", "not_certified", "not_promoted") and out["rows"] == st["rows"]["kept"]


def test_l9_the_disagreement_cusum_alone_suspends_the_table_and_learning_goes_on():
    client, so, _ = make()
    live_table(client, so)
    # clear texts the table is sure of, which decisio now answers otherwise: only the checks see it
    sent = traffic(client, 600, seed=10, clear=1.0, right=0.0, conf=0.9)
    st = status(client)
    g = st["guards"]
    assert st["state"] == "suspended" and st["suspended"]["alarm"] == "disagreement_cusum", g
    assert g["disagreement_cusum"]["alarm"] and not g["low_confidence_share"]["alarm"]
    assert g["disagreement_cusum"]["disagreed"] >= FTL.CUSUM_LIMIT and "differ from the table's" in st["reason"]
    first = next(i for i, (a, _) in enumerate(sent) if a["answered_by"] == "table")
    assert first < 5 and sent[-1][0]["answered_by"] == "decisio"
    logged = st["rows"]["logged"]
    traffic(client, 30, seed=11)
    assert status(client)["rows"]["logged"] == logged + 30


# ---- L10 ------------------------------------------------------------------------------------------------------


def test_l10_the_certificate_is_made_on_rows_logged_independently_of_the_table_only():
    client, so, _ = make()
    assert learn(client).status_code == 200
    # deferred rows alone would certify (decisio agrees with every clear text), but they are the live table's choice
    log_rows(so, 800, seed=12, kind=FTL.DEFERRED)
    out = so.learning.refit_task("queue")
    assert out["outcome"] == "not_certified" and "logged independently of the table" in out["reason"]
    assert out["kinds"][0]["cert_rows"] == {"no_table": 0, "explored": 0}
    # spot-check rows alone do not certify either: they come only from the table's own answers
    log_rows(so, 800, seed=21, kind=FTL.SPOT_CHECK)
    out = so.learning.refit_task("queue")
    assert out["outcome"] == "not_certified" and out["kinds"][0]["cert_rows"] == {"no_table": 0, "explored": 0}
    # explored rows on which decisio answers at random: the certificate is made on them, and does not hold
    log_rows(so, 800, seed=13, kind=FTL.EXPLORED, right=0.0)
    out = so.learning.refit_task("queue")
    held = out["kinds"][0]["cert_rows"]
    assert out["outcome"] == "not_certified" and held["explored"] >= 59 and held["no_table"] == 0, out
    assert "no confidence threshold keeps the table's disagreements" in out["reason"]
    # the same rows' share, but where decisio is consistent: it holds
    so.learning.drop("queue")
    assert learn(client).status_code == 200
    log_rows(so, 800, seed=12, kind=FTL.DEFERRED)
    log_rows(so, 800, seed=13, kind=FTL.EXPLORED, right=1.0)
    out = so.learning.refit_task("queue")
    rows = status(client)["certificate"]["rows"]
    assert (
        out["outcome"] == "promoted" and rows["explored"] >= 59 and rows["no_table"] == 0 and "spot_check" not in rows
    )


# ---- L11, L12 ------------------------------------------------------------------------------------------------


def test_l11_a_restart_with_a_tasks_directory_resumes_the_live_version_and_the_log(tmp_path):
    client, so, backend = make(tmp_path / "tasks")
    live_table(client, so, explore=0.0)
    traffic(client, 100, seed=14)
    before = status(client)
    so.learning.close()
    backend.close()
    client2, so2, backend2 = make(tmp_path / "tasks")
    st = status(client2)
    assert st["state"] == "active" and st["version"] == before["version"] and st["certificate"] == before["certificate"]
    assert st["rows"]["kept"] == before["rows"]["kept"] and st["rows"]["by_kind"] == before["rows"]["by_kind"]
    a = ask(client2, SURE)
    assert a["answered_by"] == "table" and a["certificate"] == before["certificate"]["id"]
    traffic(client2, 50, seed=16)
    assert status(client2)["rows"]["kept"] > before["rows"]["kept"]  # the log goes on from where it was
    out = so2.learning.refit_task("queue")
    assert out["rows"] == status(client2)["rows"]["kept"]
    assert client2.delete("/v1/tasks/queue").status_code == 200
    assert not (tmp_path / "tasks" / "learn" / FTL.task_digest("queue")).exists()  # its rows go with it
    backend2.close()


def test_l12_without_a_tasks_directory_the_status_says_the_rows_are_in_memory():
    from decisio.serve.vllm_engine import make_app

    client, so, _ = make()
    assert learn(client).status_code == 200
    traffic(client, 10)
    st = status(client)
    assert "in memory" in st["rows"]["stored"] and "lost at restart" in st["rows"]["stored"]
    listed = client.get("/v1/tasks").json()["tasks"][0]
    assert listed["status"]["fast_table"]["rows"]["kept"] == 10
    h = TestClient(make_app(so.engine, so)).get("/health").json()["systemone"]["fast_table_learning"]
    assert h["tasks"] == 1 and h["rows_logged"] == 10 and h["stored"].startswith("memory")
    assert h["worker"]["state"] == "not started" and h["encoder_queue"]["capacity"] == FTL.ENCODE_QUEUE


# ---- L13 ------------------------------------------------------------------------------------------------------


def test_l13_the_encoder_queue_drops_rows_under_load_without_making_a_request_wait(monkeypatch):
    monkeypatch.setattr(FTL, "ENCODE_QUEUE", 4)
    monkeypatch.setattr(FTL, "BACKGROUND", True)
    release = threading.Event()

    def slow_embed(texts, long_state=FT.TRUNC512):
        release.wait(10)  # the host is busy: the encoder takes its time
        return fake_embed(texts, long_state)

    client, so, _ = make()
    assert learn(client, kind="encoder").status_code == 200
    monkeypatch.setattr(FT, "embed", slow_embed)
    t0 = real_time.monotonic()
    sent = traffic(client, 12, seed=17)
    waited = real_time.monotonic() - t0
    assert waited < 5 and all(a["answered_by"] == "decisio" for a, _ in sent)  # no request waited for the encoder
    dropped = status(client)["rows"]["encoder_dropped"]
    assert 12 - FTL.ENCODE_QUEUE - 1 <= dropped <= 12 - FTL.ENCODE_QUEUE
    release.set()
    so.learning.drain()
    rows = so.learning.learners["queue"].log.read()
    assert len(rows["id"]) == 12 and int(rows["has_encoder"].sum()) == 12 - dropped
    assert so.learning.health()["encoder_queue"]["dropped"] == dropped
    so.learning.close()


def test_l13_rows_without_encoder_features_are_not_fitted_for_the_encoder(monkeypatch):
    monkeypatch.setattr(FTL, "ENCODE_QUEUE", 60)
    client, so, _ = make()
    assert learn(client, kind="auto").status_code == 200
    traffic(client, 1200, seed=18)  # no encoder thread: 60 rows wait for their features, the rest are logged without
    assert status(client)["rows"]["encoder_dropped"] == 1140
    so.learning.drain()
    rows = so.learning.learners["queue"].log.read()
    assert int(rows["has_encoder"].sum()) == 60 and rows["has_static"].all()
    out = so.learning.refit_task("queue")
    kinds = {k["kind"]: k for k in out["kinds"]}
    assert kinds["encoder"]["fit_rows"] < 60 and "with encoder features" in kinds["encoder"]["why_not"]
    assert kinds["static"]["fit_rows"] > 700 and out["outcome"] == "promoted" and out["kind"] == "static"


# ---- L14 ------------------------------------------------------------------------------------------------------


def test_l14_a_labelled_task_that_learns_logs_around_its_labelled_table_and_must_beat_it():
    client, so, _ = make()
    body = {
        "id": "queue",
        "examples": examples(800),
        "certify": {"alpha": 0.1, "delta": 0.1},
        "fast_table": {"learn": True, "kind": "static", "explore": 0.05},
    }
    r = client.post("/v1/tasks", json=body)
    assert r.status_code == 200, r.text
    t = r.json()
    st = t["status"]["fast_table"]
    assert st["mode"] == "both" and st["state"] == "learning" and st["labels"] == {"fit": 400, "held_out": 400}
    assert st["labelled_table"]["state"] == "active" and t["learn"]["alpha"] == 0.1
    L = so.learning.learners["queue"]
    L.explore = 0.0  # this one request is not explored
    a = ask(client, SURE)
    assert a["answered_by"] == "table" and a["certificate_kind"] == "gold" and a["certified"] is True
    L.explore = 0.05
    sent = traffic(client, 600, seed=20)
    kinds = status(client)["rows"]["by_kind"]
    assert kinds["no_table"] == 0 and kinds["deferred"] > 0 and kinds["explored"] > 0 and kinds["spot_check"] > 0
    assert all(a.get("certificate_kind") == "gold" for a, _ in sent)
    explored = [a for a, _ in sent if a["answered_by"] == "decisio" and a["confidence"] > 0.8]
    assert explored and all("certified" in a for a in explored)  # decisio's answers keep the task's certificate
    so.task_store.snapshot("queue")["fast_table"]["timing"]["guard"]["saving"] = 0.99  # a labelled table hard to beat
    out = so.learning.refit_task("queue")
    k = out["kinds"][0]
    assert k["certificate_kind"] == "gold" and k["cert_rows"] == {"labelled": 400} and k["fit_rows"] > 400
    assert out["outcome"] == "not_promoted" and "than the labelled table's 99.0%" in out["reason"], out
    assert status(client)["version"]["live"] is None


def test_l14_a_labelled_task_whose_table_did_not_pay_at_registration_learns_one_with_a_gold_certificate():
    client, so, _ = make()
    STATIC[0] = 50.0  # the host was busy when the task was registered: the table did not pay
    body = {
        "id": "queue",
        "examples": examples(800),
        "certify": {"alpha": 0.1, "delta": 0.1},
        "fast_table": {"learn": True, "kind": "static", "explore": 0.0},
    }
    t = client.post("/v1/tasks", json=body).json()
    assert not t["fast_table_report"]["kept"] and "fast_table" not in t
    assert t["status"]["fast_table"]["labelled_table"]["state"] == "off"
    STATIC[0] = 1.0
    traffic(client, 600, seed=22)
    assert status(client)["rows"]["by_kind"]["no_table"] == 600
    out = so.learning.refit_task("queue")
    k = out["kinds"][0]
    assert out["outcome"] == "promoted" and k["certificate_kind"] == "gold" and k["fit_rows"] > 400, out
    st = status(client)
    c = st["certificate"]
    assert c["kind"] == "gold" and c["rows"] == {"labelled": 400} and "wrong against the labelled" in c["promise"]
    assert st["version"]["live"] == 1 and so.learning.learners["queue"].live.d["guard"]["timed_on"] == "traffic"
    b = ask(client, SURE)
    assert b["answered_by"] == "table" and b["certificate_kind"] == "gold" and b["certificate"] == c["id"]
    d = ask(client, UNSURE)  # decisio, under the task's own certificate
    assert d["answered_by"] == "decisio" and d["certificate"] == t["certificate"]["id"]
    assert d["certified"] is True and d["certificate_kind"] == "gold"


# ---- L15 ------------------------------------------------------------------------------------------------------


def test_l15_the_worker_refits_on_its_own_thread_after_refit_rows_new_rows(monkeypatch):
    monkeypatch.setattr(FTL, "BACKGROUND", True)
    monkeypatch.setattr(FTL, "REFIT_ROWS", 300)
    client, so, _ = make()
    assert learn(client).status_code == 200
    traffic(client, 400, seed=21)
    deadline = real_time.monotonic() + 30
    while so.learning.worker["refits"] == 0 and real_time.monotonic() < deadline:
        real_time.sleep(0.05)
    assert so.learning.worker["refits"] >= 1 and so.learning.worker["last"]["task"] == "queue"
    assert so.learning.health()["worker"]["cpu_seconds"] >= 0
    so.learning.close()
