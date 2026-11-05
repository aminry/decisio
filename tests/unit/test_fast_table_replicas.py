# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A fast table that learns from traffic across the replicas of a shared task store (`--task-store`;
decisio.serve.fast_table_replicas; docs/design/task-store.md, "Learning on the shared store").

The store is obstore's in-memory store, which enforces create-only and ETag compare-and-swap writes as S3 does; a small
wrapper (`Faulty`) makes it unreachable or runs a step between a read and a write. Replicas are SystemOne servers on
the fake engine and the student stand-ins of tests/unit/test_fast_table_learn.py, on one virtual wall clock (`WALL`,
for the lease, the flushes and the stats) and the virtual request clock. The task store's poll is `poll`, by hand; no
test sleeps. A version is timed on each replica before it answers there: five requests here (`FT.SHADOW_N`).

R1   the lease: two replicas racing for it, exactly one wins; an expired lease is taken with generation + 1, the old
     holder's renewal fails and it stops training at once
R2   fencing: a holder whose lease moved while it fitted (its clock behind) publishes nothing; its successor does
R3   the state: two updates at once, one retries and both land; one replica's alarm suspends the table on every one
R4   the certificate: rows from both replicas reach one certificate, which names them; another serving stack's rows do
     not; spot-check rows never certify; a labelled task registered on one replica is learned on the trainer with its
     labels
R5   publishing: a version reaches the other replica on its poll, is checked against its digest, timed there and then
     answers; the cluster view on any replica
R6   restart: one replica, then all, resume the promoted version with no refit
R7   the store unreachable: serving goes on without a store call, the buffer fills to its cap, then rows are dropped and
     counted in /health; the lease lapses, nobody promotes; the rows are written when the store comes back
R8   compaction and retention: merged files replace the replica segments, the newest max_rows are kept
R9   a read-only replica: never takes the lease, logs nothing, serves the version, its alarm is its own, no rollback
R10  removal and registration again: the learned rows and versions go (gc takes a late segment), the other replica
     forgets the task; a rollback on one replica reaches every replica; the task store's poll follows the state
R11  two replicas on a real S3-compatible service, when `DECISIO_TEST_S3_ENDPOINT` names one
R12  a segment write that reached the store but whose answer was lost is found there on the retry: its rows are
     counted once

  uv run pytest -q tests/unit/test_fast_table_replicas.py
"""

import collections
import json
import os
import threading
import types
import uuid

import numpy as np
import pytest

pytest.importorskip("sklearn")
obstore = pytest.importorskip("obstore")
pytest.importorskip("tokenizers")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from obstore.store import MemoryStore  # noqa: E402
from test_fast_table_kinds import STATIC, students  # noqa: E402,F401  (the static and encoder stand-ins: autouse)
from test_fast_table_learn import SURE, ask, learn, log_rows, status, traffic  # noqa: E402
from test_fast_table_task import FakeEngine, examples, virtual_time  # noqa: E402,F401  (the virtual clock: autouse)

from decisio.serve import fast_table as FT  # noqa: E402
from decisio.serve import fast_table_learn as FTL  # noqa: E402
from decisio.serve import fast_table_replicas as FTR  # noqa: E402
from decisio.serve import systemone as SO  # noqa: E402
from decisio.serve import task_backends as TB  # noqa: E402
from decisio.serve import task_store_shared as S  # noqa: E402
from decisio.serve.systemone import SystemOne, add_routes  # noqa: E402
from decisio.serve.tasks import TaskStore  # noqa: E402


class Wall:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


WALL = Wall()


@pytest.fixture(autouse=True)
def shared_by_hand(monkeypatch, virtual_time):  # noqa: F811
    monkeypatch.setattr(FTL, "BACKGROUND", False)
    monkeypatch.setattr(FTL, "SEED", 0)
    WALL.t = 1_800_000_000.0
    monkeypatch.setattr(SO.time, "time", WALL)  # the virtual clock's stand-in for the time module
    monkeypatch.setattr(FT, "SHADOW_EVERY", 1)  # a version is timed on every request decisio answers
    monkeypatch.setattr(FT, "SHADOW_N", 5)


class Faulty:
    """`Objects` with faults: `down` (no call reaches the store), and `before[path suffix]` run before a write to a
    path that ends so. Counts the calls."""

    def __init__(self, inner):
        self.inner = inner
        self.down = False
        self.before = {}
        self.calls = collections.Counter()

    def _gate(self, op, path):
        self.calls[op] += 1
        if self.down:
            raise TB.TaskStoreUnavailable(f"the task store cannot be reached ({path}: ConnectionError: refused)")
        if op in ("create", "update"):
            for suffix, fn in list(self.before.items()):
                if path.endswith(suffix):
                    fn(op, path)

    def get(self, path, if_none_match=None):
        self._gate("get", path)
        return self.inner.get(path, if_none_match)

    def create(self, path, data):
        self._gate("create", path)
        return self.inner.create(path, data)

    def update(self, path, data, tag):
        self._gate("update", path)
        return self.inner.update(path, data, tag)

    def put(self, path, data):
        self._gate("put", path)
        return self.inner.put(path, data)

    def delete(self, path):
        self._gate("delete", path)
        return self.inner.delete(path)

    def list(self, prefix):
        self._gate("list", prefix)
        return self.inner.list(prefix)


def once(fn):
    done = []

    def run(*a):
        if not done:
            done.append(1)
            fn()

    return run


def replica(store, name, readonly=False, label="memory://t"):
    """A replica as `--task-store` starts one, with the replica id `name`."""
    backend = S.ObjectBackend(store, label, readonly=readonly, clock=WALL)
    ts = TaskStore("fp")
    abstention, report = TB.load_at_start(ts, backend)
    real = FTR.replica_id
    FTR.replica_id = lambda: name
    try:
        so = SystemOne(FakeEngine(), "decisio-test", task_store=ts, abstention_tasks=abstention, task_backend=backend,
                       task_report=report)  # fmt: skip
    finally:
        FTR.replica_id = real
    app = FastAPI()
    add_routes(app, so)
    return types.SimpleNamespace(client=TestClient(app), so=so, backend=backend, name=name)


def poll(*replicas, advance=0.0):
    """The task store's poll on each replica, in order (`TaskSync.run`), after `advance` seconds."""
    WALL.t += advance
    for r in replicas:
        r.backend.sync_once()
        r.so.learning.sync()


def learner(r, task_id="queue"):
    return r.so.learning.learners[task_id]


def objects(r):
    return r.so.learning.shared.objects


def state(r, task_id="queue"):
    L = learner(r, task_id)
    got = objects(r).get(L.prefix + "STATE.json")
    return json.loads(got[0]) if got else None


def flush(*replicas):
    for r in replicas:
        assert learner(r).log.flush()


def promoted(store, **fast_table):
    """Replicas a (the trainer) and b, with rows from both and version 1 published by a and adopted by b."""
    a, b = replica(store, "a"), replica(store, "b")
    assert learn(a.client, **fast_table).status_code == 200
    poll(a, b)
    log_rows(a.so, 600, seed=1)
    log_rows(b.so, 600, seed=2)
    flush(a, b)
    out = a.so.learning.refit_task("queue")
    assert out["outcome"] == "promoted" and out["version"] == 1, out
    poll(a, b)
    return a, b


def time_it(r, n=None):
    """Requests on `r` until its timing of the live version passes."""
    traffic(r.client, n or FT.SHADOW_N, seed=99)
    assert status(r.client)["state"] == "active", status(r.client)


def stub(objects_, name):
    """A Shared on its own, for the lease alone."""
    sh = FTR.Shared(objects_, replica=name)
    sh.attach(types.SimpleNamespace(now=WALL, learners={}))
    return sh


# ---- R1 -------------------------------------------------------------------------------------------------------------


def test_r1_two_replicas_racing_for_the_lease_exactly_one_wins():
    objs = Faulty(FTR.Objects(MemoryStore()))
    gate = threading.Barrier(2)
    objs.before[FTR.LEASE_PATH] = lambda op, path: gate.wait(10)  # both have read "no lease" before either writes
    a, b = stub(objs, "a"), stub(objs, "b")
    won = {}
    threads = [threading.Thread(target=lambda s=s: won.__setitem__(s.replica, s.lease.step())) for s in (a, b)]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    assert sorted(won.values()) == [False, True]
    winner = next(k for k, v in won.items() if v)
    lease = json.loads(objs.inner.get(FTR.LEASE_PATH)[0])
    assert set(lease) == {"holder", "generation", "expires"}
    assert lease["holder"] == winner and lease["generation"] == 1 and lease["expires"] == WALL() + FTR.LEASE_SECONDS
    assert [s.holds() for s in (a, b)].count(True) == 1


def test_r1_an_expired_lease_is_taken_with_the_next_generation_and_the_old_holder_stops_at_once():
    objs = FTR.Objects(MemoryStore())
    a, b = stub(objs, "a"), stub(objs, "b")
    assert a.lease.step() and not b.lease.step()
    WALL.t += FTR.RENEW_SECONDS
    assert a.lease.step() and not b.lease.step()  # renewed in time: still a's
    assert json.loads(objs.get(FTR.LEASE_PATH)[0])["expires"] == WALL() + FTR.LEASE_SECONDS
    WALL.t += FTR.LEASE_SECONDS + 1  # a stopped renewing
    assert b.lease.step() and b.lease.generation == 2 and b.holds()
    assert json.loads(objs.get(FTR.LEASE_PATH)[0]) == {"holder": "b", "generation": 2, "expires": WALL() + 120}
    assert not a.lease.step()  # its renewal is a compare-and-swap on its own last write: refused
    assert not a.holds() and a.lease.facts()["lost"] == "another replica wrote the lease"
    b.lease.release()  # a clean stop: the next replica takes it at once
    assert a.lease.step() and a.lease.generation == 3


# ---- R2 -------------------------------------------------------------------------------------------------------------


def test_r2_a_stale_holder_cannot_publish_over_its_successor():
    store = MemoryStore()
    a, b = replica(store, "a"), replica(store, "b")
    a.so.learning.now = lambda: WALL() - 200  # a's clock is behind: it believes its lease runs on after it lapsed
    assert learn(a.client).status_code == 200
    poll(a)
    assert a.so.learning.shared.holds()
    log_rows(a.so, 1200, seed=1)
    flush(a)
    poll(b)  # b sees the lease expired and takes it
    lease = json.loads(objects(b).get(FTR.LEASE_PATH)[0])
    assert lease["holder"] == "b" and lease["generation"] == 2
    assert a.so.learning.shared.holds()  # a has not noticed
    out = a.so.learning.refit_task("queue")
    assert out["outcome"] == "fenced" and "the lease moved from generation 1 to 2 (held by b)" in out["reason"], out
    assert not a.so.learning.shared.holds()  # the fence ends a's training at once
    assert state(a)["live"] is None and state(a)["next_version"] == 1  # nothing published
    out = b.so.learning.refit_task("queue")
    assert out["outcome"] == "promoted" and out["version"] == 1
    st = state(b)
    assert st["live"] == 1 and st["generation"] == 2 and st["versions"][0]["generation"] == 2


# ---- R3 -------------------------------------------------------------------------------------------------------------


def test_r3_two_state_updates_at_once_one_retries_and_both_land():
    store = MemoryStore()
    a, b = promoted(store)
    time_it(a)
    time_it(b)
    faulty = Faulty(objects(b))
    b.so.learning.shared.objects = faulty
    # between b's read and its write, the trainer writes the cluster view: b's compare-and-swap loses and is retried
    faulty.before["STATE.json"] = once(lambda: learner(a).aggregate())
    learner(b).suspend("disagreement_cusum", "the spot checks disagree")
    assert status(b.client)["state"] == "suspended" and status(a.client)["state"] == "active"
    poll(b)
    assert learner(b).counters["state_races"] == 1
    st = state(b)
    assert st["suspended"]["by"] == "b" and st["suspended"]["alarm"] == "disagreement_cusum"
    assert st["cluster"]["trainer"] == "a" and st["cluster"]["at"] == WALL()  # the trainer's write is kept
    poll(a)  # any replica's alarm suspends the table for all
    sa = status(a.client)
    assert sa["state"] == "suspended" and sa["suspended"]["by"] == "b"
    assert ask(a.client, SURE)["answered_by"] == "decisio"


# ---- R4 -------------------------------------------------------------------------------------------------------------


def test_r4_rows_from_both_replicas_reach_one_certificate_and_another_stack_does_not():
    store = MemoryStore()
    a, b, c = replica(store, "a"), replica(store, "b"), replica(store, "c")
    c.so.learning._stack_value = "e" * 16  # c serves through another stack (another card, another engine)
    assert learn(a.client).status_code == 200
    poll(a, b, c)
    log_rows(a.so, 500, seed=1)
    log_rows(b.so, 500, seed=2)
    log_rows(c.so, 500, seed=3)
    flush(a, b, c)
    out = a.so.learning.refit_task("queue")
    assert out["outcome"] == "promoted", out
    cert = status(a.client)["certificate"]
    assert set(cert["replicas"]) == {"a", "b"} and min(cert["replicas"].values()) > 50
    assert sum(cert["replicas"].values()) == cert["n"] == sum(cert["rows"].values())
    assert cert["other_stack_rows"] > 50 and cert["stack"] == a.so.learning.stack_sha()
    assert out["rows"] == 1500 and out["kinds"][0]["fit_rows"] < 800  # c's rows are kept, and not fitted on
    poll(a, b, c)
    assert status(c.client)["state"] == "off" and "another serving stack" in status(c.client)["reason"]


def test_r4_spot_check_rows_never_certify():
    store = MemoryStore()
    a, b = replica(store, "a"), replica(store, "b")
    assert learn(a.client).status_code == 200
    poll(a, b)
    log_rows(b.so, 800, seed=4, kind=FTL.SPOT_CHECK, would=True)
    log_rows(a.so, 400, seed=5, kind=FTL.DEFERRED)
    flush(a, b)
    out = a.so.learning.refit_task("queue")
    assert out["outcome"] == "not_certified" and "logged independently of the table" in out["reason"], out
    assert out["kinds"][0]["cert_rows"] == {"no_table": 0, "explored": 0}


def test_r4_a_labelled_task_registered_on_one_replica_is_learned_on_the_trainer_with_its_labels():
    store = MemoryStore()
    a, b = replica(store, "a"), replica(store, "b")
    poll(a)
    STATIC[0] = 50.0  # the host was busy at registration: the labelled table did not pay
    body = {
        "id": "queue",
        "examples": examples(800),
        "certify": {"alpha": 0.1, "delta": 0.1},
        "fast_table": {"learn": True, "kind": "static", "explore": 0.0},
    }
    assert b.client.post("/v1/tasks", json=body).status_code == 200  # registered on b
    STATIC[0] = 1.0
    poll(a, b)
    assert a.so.learning.shared.holds() and objects(a).get(learner(a).prefix + "labels.npz") is not None
    log_rows(a.so, 300, seed=1)
    log_rows(b.so, 300, seed=2)
    flush(a, b)
    out = a.so.learning.refit_task("queue")  # a has the labels from the store
    k = out["kinds"][0]
    assert out["outcome"] == "promoted" and k["certificate_kind"] == "gold" and k["cert_rows"] == {"labelled": 400}
    assert k["fit_rows"] > 400 + 300  # the labelled examples and both replicas' rows
    poll(b)
    time_it(b)
    ans = ask(b.client, SURE)
    assert ans["answered_by"] == "table" and ans["certificate_kind"] == "gold"


# ---- R5 -------------------------------------------------------------------------------------------------------------


def test_r5_a_version_reaches_the_other_replica_is_timed_there_and_then_answers():
    store = MemoryStore()
    a, b = promoted(store)
    sb = status(b.client)
    assert sb["state"] == "timing" and sb["version"]["live"] == 1 and "timed on this replica's" in sb["reason"]
    assert ask(b.client, SURE)["answered_by"] == "decisio"  # not before its own timing passes
    calls = FakeEngine.calls
    time_it(b, FT.SHADOW_N - 1)
    timing = status(b.client)["replica"]["timing"]
    assert timing["passed"] and timing["last"]["kept"] and timing["last"]["saving"] >= FT.MIN_SAVING
    a_ans = ask(b.client, SURE)
    assert a_ans["answered_by"] == "table" and a_ans["certificate"] == status(a.client)["certificate"]["id"]
    assert a_ans["certificate_kind"] == "agreement" and FakeEngine.calls == calls + FT.SHADOW_N - 1
    # the rows b logged while it timed the version are no_table rows: drawn independently of the table
    assert learner(b).counters["no_table"] == 600 + FT.SHADOW_N
    # the cluster view, on any replica, once the trainer has read the replicas' stats (written every STATS_SECONDS)
    for _ in range(int(FTR.STATS_SECONDS // FTR.RENEW_SECONDS) + 1):
        poll(b, a, advance=FTR.RENEW_SECONDS)  # a renews its lease as it goes
    poll(b)
    cluster = status(b.client)["cluster"]
    assert cluster["trainer"] == "a" and set(cluster["replicas"]) == {"a", "b"}
    assert cluster["replicas"]["b"]["state"] == "active" and cluster["replicas"]["a"]["state"] == "timing"
    assert cluster["rows"]["flushed"] == learner(a).log.flushed + learner(b).log.flushed == 1200 + FT.SHADOW_N
    h = TestClient(app_of(b)).get("/health").json()["systemone"]["fast_table_learning"]
    assert h["stored"] == "the shared task store" and h["shared"]["replica"] == "b" and h["shared"]["ok"]
    assert h["shared"]["lease"]["held"] is False and h["shared"]["lease"]["holder"] == "a"


def app_of(r):
    from decisio.serve.vllm_engine import make_app

    return make_app(r.so.engine, r.so)


def test_r5_a_version_whose_bytes_do_not_match_its_digest_is_not_adopted():
    store = MemoryStore()
    a, b = replica(store, "a"), replica(store, "b")
    assert learn(a.client).status_code == 200
    poll(a, b)
    log_rows(a.so, 1200, seed=1)
    flush(a)
    assert a.so.learning.refit_task("queue")["outcome"] == "promoted"
    sha = state(a)["versions"][0]["object"]
    path = learner(a).prefix + f"versions/{sha}.json"
    obstore.put(store, path, b'{"version": 1, "tampered": true}')
    poll(b)
    assert status(b.client)["state"] == "learning" and status(b.client)["version"]["live"] is None
    assert "do not match its digest" in b.so.learning.shared.last_error


# ---- R6 -------------------------------------------------------------------------------------------------------------


def test_r6_a_restart_of_one_replica_and_then_of_all_resumes_the_promoted_version_with_no_refit():
    store = MemoryStore()
    a, b = promoted(store)
    cert = status(a.client)["certificate"]["id"]
    for r in (a, b):
        time_it(r)
    a.so.learning.close()  # a stops cleanly: its rows are written, its lease released
    a2 = replica(store, "a2")
    st = status(a2.client)
    assert st["state"] == "timing" and st["version"]["live"] == 1 and st["certificate"]["id"] == cert
    assert a2.so.learning.shared.holds() and a2.so.learning.shared.lease.generation == 2
    time_it(a2)
    assert ask(a2.client, SURE)["certificate"] == cert
    assert a2.so.learning.worker["refits"] == 0 and state(a2)["next_version"] == 2  # nothing learned again
    traffic(a2.client, 20, seed=5)
    poll(a2, advance=FTL.SEGMENT_SECONDS)
    assert learner(a2).log.flushed > 0  # and it keeps logging
    for r in (a2, b):  # the whole cluster stops
        r.so.learning.close()
    a3, b3 = replica(store, "a3"), replica(store, "b3")
    for r in (a3, b3):
        assert status(r.client)["version"]["live"] == 1
        time_it(r)
        assert ask(r.client, SURE)["certificate"] == cert
        assert r.so.learning.worker["refits"] == 0
    assert state(a3)["next_version"] == 2 and a3.so.learning.shared.lease.generation == 3


# ---- R7 -------------------------------------------------------------------------------------------------------------


def test_r7_with_the_store_unreachable_serving_goes_on_the_buffer_fills_then_rows_are_dropped_and_counted(monkeypatch):
    monkeypatch.setattr(FTR, "BUFFER_BYTES", 40_000)  # about 60 rows
    store = MemoryStore()
    a = replica(store, "a")
    assert learn(a.client).status_code == 200
    poll(a)
    assert a.so.learning.shared.holds()
    faulty = Faulty(objects(a))
    a.so.learning.shared.objects = faulty
    faulty.down = True
    sent = traffic(a.client, 200, seed=6)
    assert all(ans["answered_by"] == "decisio" for ans, _ in sent)
    assert sum(faulty.calls.values()) == 0  # no request made a store call
    L = learner(a)
    held, dropped = L.log.facts()["buffered"], L.log.dropped
    assert 40 < held < 80 and held + dropped == 200 and L.log.pending_bytes <= 40_000
    poll(a, advance=FTL.SEGMENT_SECONDS)  # the poll cannot write the rows, nor renew the lease
    h = TestClient(app_of(a)).get("/health").json()["systemone"]["fast_table_learning"]["shared"]
    want = {"bytes": L.log.pending_bytes, "capacity": 40_000, "rows_waiting": held, "rows_dropped": dropped}
    assert h["buffer"] == want
    assert h["ok"] is False and "cannot be reached" in h["last_error"] and h["lease"]["held"] is False
    assert a.so.learning.refit_task("queue")["outcome"] == "not_trainer"  # nobody promotes
    assert status(a.client)["rows"]["this_replica"]["dropped"] == dropped
    faulty.down = False
    poll(a)
    assert L.log.flushed == held and L.log.facts()["buffered"] == 0
    assert a.so.learning.shared.holds() and a.so.learning.shared.lease.generation == 2


# ---- R8 -------------------------------------------------------------------------------------------------------------


def test_r8_compaction_merges_the_replica_segments_and_retention_keeps_the_newest_rows(monkeypatch):
    monkeypatch.setattr(FTR, "COMPACT_SEGMENTS", 4)
    store = MemoryStore()
    a, b = replica(store, "a"), replica(store, "b")
    assert learn(a.client, max_rows=300).status_code == 200
    poll(a, b)
    for i in range(3):
        WALL.t += 10
        log_rows(a.so, 100, seed=i)
        WALL.t += 10
        log_rows(b.so, 100, seed=10 + i)
        flush(a, b)
    prefix = learner(a).prefix + "log/"
    raw = [p for p in objects(a).list(prefix) if "/merged/" not in p and p.endswith(".seg")]
    assert len(raw) == 6 and {p.split("/")[-2] for p in raw} == {"a", "b"}
    out = a.so.learning.refit_task("queue")
    assert out["rows"] == 300
    left = objects(a).list(prefix)
    merged = [p for p in left if "/merged/" in p]
    assert merged and set(left) == set(merged) | {prefix + "INDEX.json"}  # the replica segments are gone
    index = json.loads(objects(a).get(prefix + "INDEX.json")[0])
    assert [prefix + m for m in index["merged"]] == merged and index["rows"] == 300 and index["generation"] == 1
    rows = learner(a).log.read()
    assert len(rows["id"]) == 300 and rows["t"].min() == WALL.t - 20  # the newest three batches
    assert set(np.unique(rows["replica"]).tolist()) == {b"a", b"b"}
    WALL.t += 10
    log_rows(b.so, 100, seed=20)
    flush(b)
    out = a.so.learning.refit_task("queue")
    assert out["rows"] == 300
    index2 = json.loads(objects(a).get(prefix + "INDEX.json")[0])
    assert not set(index2["merged"]) & set(index["merged"])  # the old merged files were replaced
    assert set(objects(a).list(prefix)) == {prefix + m for m in index2["merged"]} | {prefix + "INDEX.json"}
    assert learner(a).log.read()["t"].min() == WALL.t - 20


# ---- R9 -------------------------------------------------------------------------------------------------------------


def test_r9_a_read_only_replica_never_takes_the_lease_logs_nothing_and_serves_the_version():
    store = MemoryStore()
    a, r = replica(store, "a"), replica(store, "r", readonly=True)
    assert learn(r.client).status_code == 409  # it does not teach
    assert learn(a.client).status_code == 200
    poll(r, a)  # r polls first: it still does not take the lease
    assert json.loads(objects(a).get(FTR.LEASE_PATH)[0])["holder"] == "a"
    traffic(r.client, 50, seed=7)
    poll(r, advance=FTL.SEGMENT_SECONDS)
    assert not [p for p in objects(a).list("tasks/") if "/log/r/" in p or "/stats/r" in p]  # r wrote nothing
    st = status(r.client)
    assert "does not log" in st["rows"]["stored"] and st["rows"]["this_replica"]["logged"] == 0
    log_rows(a.so, 1200, seed=8)
    flush(a)
    poll(a)  # the trainer renews its lease
    assert a.so.learning.refit_task("queue")["outcome"] == "promoted"
    a.so.learning.close()  # the trainer stops and releases the lease
    poll(r, advance=FTR.LEASE_SECONDS)
    assert json.loads(objects(a).get(FTR.LEASE_PATH)[0])["holder"] == "a"  # r never takes it
    assert r.so.learning.shared.lease is None and r.so.learning.refit_task("queue")["outcome"] == "not_trainer"
    time_it(r)
    assert ask(r.client, SURE)["answered_by"] == "table"
    learner(r).suspend("low_confidence_share", "the traffic moved")  # its alarm suspends it alone
    poll(r)
    assert status(r.client)["state"] == "suspended" and status(r.client)["suspended"]["shared"] is False
    assert state(r)["suspended"] is None
    resp = r.client.post("/v1/tasks/queue/fast_table/rollback", json={"version": 1})
    assert resp.status_code == 409 and "--task-store-readonly" in resp.json()["detail"]


# ---- R10 ------------------------------------------------------------------------------------------------------------


def test_r10_a_removal_deletes_the_learned_rows_and_a_new_registration_starts_afresh():
    store = MemoryStore()
    a, b = promoted(store)
    first = learner(a).prefix
    assert objects(a).list(first)
    log_rows(b.so, 10, seed=3)  # b has rows of the first registration still to write
    assert learn(a.client).status_code == 200  # registered again: a new registration, fresh state
    second = learner(a).prefix
    assert second != first and not objects(a).list(first) and state(a)["live"] is None
    flush(b)  # b has not polled yet: its last segment lands under the first registration
    stray = objects(a).list(first)
    assert len(stray) == 1 and "/log/b/" in stray[0]
    gone = a.backend.gc(older_than_days=-1)  # gc removes it (and the superseded task record), not the live prefix
    assert [p for p in gone if p.startswith("tasks/")] == stray
    assert objects(a).list(second) and not objects(a).list(first)
    poll(b)
    assert learner(b).prefix == second and status(b.client)["version"]["live"] is None
    assert a.client.delete("/v1/tasks/queue").status_code == 200
    assert not objects(a).list(second)
    poll(b)
    assert "queue" not in b.so.learning.learners


def test_r10_the_task_stores_poll_follows_the_learned_state_too():
    store = MemoryStore()
    a, b = replica(store, "a"), replica(store, "b")
    assert b.so.learning.sync in b.backend.followers
    assert learn(a.client).status_code == 200
    b.backend.poll_seconds = 0.0
    sync = S.TaskSync(b.backend)
    b.backend.followers.append(sync.stop)  # one pass of the poll, run here
    sync.run()
    assert learner(b).loaded and learner(b).salt == learner(a).salt


def test_r10_a_rollback_on_one_replica_reaches_every_replica():
    store = MemoryStore()
    a, b = promoted(store)
    for r in (a, b):
        time_it(r)
    learner(a).live.saving = 0.0  # any version that saves time replaces it
    log_rows(b.so, 600, seed=9)
    flush(b)
    assert a.so.learning.refit_task("queue")["outcome"] == "promoted"
    poll(a, b)
    assert status(b.client)["version"]["live"] == 2
    resp = b.client.post("/v1/tasks/queue/fast_table/rollback", json={"version": 1})
    assert resp.status_code == 200 and resp.json()["status"]["fast_table"]["version"]["live"] == 1
    poll(a)
    assert status(a.client)["version"]["live"] == 1 and status(a.client)["state"] == "timing"


# ---- R11 ------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("DECISIO_TEST_S3_ENDPOINT"), reason="no S3-compatible service named")
def test_r11_two_replicas_learn_on_a_real_s3_compatible_service():
    """As test_task_store_shared's S10: DECISIO_TEST_S3_ENDPOINT, the AWS_* credentials and region, and a bucket in
    DECISIO_TEST_S3_BUCKET (default decisio-test); a fresh prefix per run."""
    url = f"s3://{os.environ.get('DECISIO_TEST_S3_BUCKET', 'decisio-test')}/learn-{uuid.uuid4().hex[:8]}"
    endpoint = os.environ["DECISIO_TEST_S3_ENDPOINT"]
    a = replica(S.open_store(url, endpoint), "a", label=S.sanitized(url))
    a.backend.probe()
    b = replica(S.open_store(url, endpoint), "b", label=S.sanitized(url))
    assert learn(a.client).status_code == 200
    poll(a, b)
    assert a.so.learning.shared.holds() and not b.so.learning.shared.lease.step()  # one trainer
    log_rows(a.so, 600, seed=1)
    log_rows(b.so, 600, seed=2)
    flush(a, b)
    out = a.so.learning.refit_task("queue")
    assert out["outcome"] == "promoted", out
    assert set(status(a.client)["certificate"]["replicas"]) == {"a", "b"}
    poll(b)
    time_it(b)
    assert ask(b.client, SURE)["answered_by"] == "table"
    WALL.t += FTR.LEASE_SECONDS + 1  # a stops renewing: b takes the lease, a's renewal is refused
    assert b.so.learning.shared.lease.step() and b.so.learning.shared.lease.generation == 2
    assert not a.so.learning.shared.lease.step()
    assert a.client.delete("/v1/tasks/queue").status_code == 200
    assert not objects(a).list(learner(b).prefix)


# ---- R12 ------------------------------------------------------------------------------------------------------------


def test_r12_a_segment_written_whose_answer_was_lost_is_counted_once():
    store = MemoryStore()
    a = replica(store, "a")
    assert learn(a.client).status_code == 200
    poll(a)
    log_rows(a.so, 120, seed=3)
    inner = objects(a)
    real_create, lost = inner.create, []

    def create_then_lose(path, data):
        out = real_create(path, data)
        if path.endswith(".seg") and not lost:  # written, but the answer never arrives
            lost.append(path)
            raise TB.TaskStoreUnavailable(f"the task store cannot be written ({path}: TimeoutError: timed out)")
        return out

    inner.create = create_then_lose
    assert not learner(a).log.flush()  # the rows wait
    inner.create = real_create
    flush(a)  # the retry finds its own bytes under the same name
    segs = [p for p in inner.list(learner(a).prefix + "log/") if p.endswith(".seg")]
    assert segs == lost
    assert len(learner(a).log.read()["id"]) == 120
