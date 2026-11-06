# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The shared task store for replicas (`--task-store`, decisio.serve.task_store_shared; docs/design/task-store.md).

The store is obstore's in-memory store, which enforces create-only and ETag compare-and-swap writes and conditional
GETs as S3, R2, MinIO and GCS do, so the protocol meets the real semantics; replicas are SystemOne servers on
the CPU stand-in engine, following the store by `sync_once` (the poll thread has its own test).

S1  the protocol: a record published is the record applied, versions count, a removal is a tombstone, old tombstones go
S2  two replicas converge: plain, certified and abstention tasks arrive identical, answer alike, no re-fit
S3  the conflict rule: 409 with both versions, before the fit; the named version, `?replace=1`; no silent overwrite
S4  concurrent writes: the same id, one wins and one is told; different ids both land
S5  deletes propagate, and a task removed before a replica saw it is not a conflict
S6  another stack: a replica on another model keeps the task unapplied; one on another engine version suspends the
    certificate
S7  a replica that does not teach: refuses every write at once, still follows
S8  the store failing: a write fails and registers nothing, serving goes on, the replica says so, then recovers; a store
    that cannot enforce preconditions is refused at start; a record that does not match its digest is not applied
S9  the poll thread, the start-up wiring and the flags
S10 a real S3-compatible service (MinIO), when `DECISIO_TEST_S3_ENDPOINT` names one

  uv run pytest -q tests/unit/test_task_store_shared.py
"""

import json
import os
import threading
import time

import pytest

obstore = pytest.importorskip("obstore")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from obstore.store import MemoryStore  # noqa: E402
from test_certified_answers import FakeEngine, ask, examples, post_task, wire  # noqa: E402
from test_task_persistence import abstention_body, other_list  # noqa: E402

from decisio.serve import task_backends as TB  # noqa: E402
from decisio.serve import task_store_shared as S  # noqa: E402
from decisio.serve import vllm_engine  # noqa: E402
from decisio.serve.systemone import SystemOne, add_routes  # noqa: E402
from decisio.serve.tasks import TaskStore  # noqa: E402


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def replica(store, fingerprint="fp", version="1.0", readonly=False, name="decisio-test", clock=None, **kw):
    """A replica as `--task-store` starts one: the backend, what the store holds, then the server."""
    backend = S.ObjectBackend(store, "memory://t", readonly=readonly, clock=clock or time.time, **kw)
    ts = TaskStore(fingerprint)
    abstention, report = TB.load_at_start(ts, backend)
    engine = FakeEngine(version)
    so = SystemOne(engine, name, task_store=ts, abstention_tasks=abstention, task_backend=backend, task_report=report)
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, backend


def by_id(tasks):
    return sorted(tasks, key=lambda t: t["id"])


# ---- S1 -------------------------------------------------------------------------------------------------------------


def test_s1_a_published_record_is_the_record_applied_and_versions_count():
    store = MemoryStore()
    a, _, ba = replica(store)
    r = post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    assert r.status_code == 200 and r.json()["store_version"] == 1
    assert post_task(a, {"id": "t", "examples": examples(40, 0.5, seed=4)}).json()["store_version"] == 2
    b, _, bb = replica(store)  # a replica that starts later loads everything at once
    assert by_id(b.get("/v1/tasks?full=1").json()["tasks"]) == by_id(a.get("/v1/tasks?full=1").json()["tasks"])
    assert b.get("/v1/tasks/t").json()["store_version"] == 2
    manifest, _ = ba._read_manifest()
    (entry,) = manifest["entries"].values()
    assert entry["version"] == 2 and entry["kind"] == "readout" and not entry["deleted"] and manifest["version"] == 2


def test_s1_the_manifest_is_followed_with_a_conditional_get():
    store = MemoryStore()
    a, _, ba = replica(store)
    b, _, bb = replica(store)
    assert bb.sync_once() == {"applied": [], "removed": []}
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    assert bb.sync_once() == {"applied": ["t"], "removed": []}
    assert bb.sync_once() == {"applied": [], "removed": []}  # unchanged: not read again
    assert bb.describe()["manifest_version"] == 1 and bb.describe()["ok"] is True


def test_s1_old_tombstones_are_dropped_by_the_next_writer():
    clock = Clock()
    store = MemoryStore()
    a, _, ba = replica(store, clock=clock)
    post_task(a, {"id": "gone", "examples": examples(40, 0.9)})
    post_task(a, {"id": "kept", "examples": other_list(examples(40, 0.9))})
    assert a.delete("/v1/tasks/gone").status_code == 200
    manifest, _ = ba._read_manifest()
    assert sorted(e["deleted"] for e in manifest["entries"].values()) == [False, True]
    clock.t += 8 * 86400  # past the retention
    a.delete("/v1/tasks/kept")
    manifest, _ = ba._read_manifest()
    (entry,) = manifest["entries"].values()
    assert entry["id"] == "kept" and entry["deleted"]  # "gone" was dropped, "kept" is the fresh tombstone


def test_s1_gc_removes_only_objects_no_entry_points_to():
    store = MemoryStore()
    a, _, ba = replica(store)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    post_task(a, {"id": "t", "examples": examples(40, 0.5, seed=2)})  # the first version's object is now unreferenced
    assert len(ba.gc(older_than_days=-1, dry_run=True)) == 1
    assert len(ba.gc(older_than_days=1)) == 0  # too recent to touch
    assert len(ba.gc(older_than_days=-1)) == 1 and ba.gc(older_than_days=-1) == []
    b, _, _ = replica(store)
    assert [t["id"] for t in b.get("/v1/tasks").json()["tasks"]] == ["t"]  # the live record is still there


# ---- S2 -------------------------------------------------------------------------------------------------------------


def test_s2_two_replicas_converge_on_identical_records_without_a_refit():
    store = MemoryStore()
    a, so_a, ba = replica(store)
    b, so_b, bb = replica(store)
    assert post_task(a, {"id": "plain", "examples": other_list(examples(40, 0.9))}).status_code == 200
    assert post_task(a, {"id": "cert", "examples": examples(400, 0.96), "certify": {}}).status_code == 200
    assert a.post("/v1/abstention/tasks", json=abstention_body()).status_code == 200
    scored = []
    real = so_b.engine.answer
    so_b.engine.answer = lambda *x, **k: scored.append(1) or real(*x, **k)  # B must not read the engine to apply
    assert sorted(bb.sync_once()["applied"]) == ["abs", "cert", "plain"]
    assert scored == []
    assert by_id(b.get("/v1/tasks?full=1").json()["tasks"]) == by_id(a.get("/v1/tasks?full=1").json()["tasks"])
    assert b.get("/v1/abstention/tasks").json()["tasks"] == a.get("/v1/abstention/tasks").json()["tasks"]
    scored.clear()
    assert ask(b, 0, 0.99) == ask(a, 0, 0.99) and ask(b, 1, 0.6) == ask(a, 1, 0.6)
    assert b.get("/v1/tasks/cert").json()["certificate"]["certified"] is True


def test_s2_a_replaced_task_reaches_the_other_replica_and_the_old_one_goes_out_after_the_new_comes_in():
    store = MemoryStore()
    a, so_a, ba = replica(store)
    b, so_b, bb = replica(store)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    bb.sync_once()
    post_task(a, {"id": "t", "examples": other_list(examples(40, 0.9, seed=3))})  # another option list: another key
    bb.sync_once()
    assert len(so_b.task_store.by_key) == 1 and next(iter(so_b.task_store.by_key.values()))["id"] == "t"
    assert b.get("/v1/tasks/t").json()["store_version"] == 2


# ---- S3 -------------------------------------------------------------------------------------------------------------


def test_s3_a_replica_that_has_not_followed_yet_is_told_before_it_fits():
    store = MemoryStore()
    a, _, _ = replica(store)
    b, so_b, bb = replica(store)
    assert post_task(a, {"id": "t", "examples": examples(40, 0.9)}).status_code == 200
    reads = []
    real = so_b.engine.answer
    so_b.engine.answer = lambda *x, **k: reads.append(1) or real(*x, **k)
    r = post_task(b, {"id": "t", "examples": examples(40, 0.5, seed=9)})  # B has not polled
    assert r.status_code == 409 and reads == []  # refused before any example was read
    d = r.json()["detail"]
    assert (
        d["store_version"] == 1 and d["your_version"] == 0 and d["task"] == "t" and "someone taught it" in d["message"]
    )
    assert b.get("/v1/tasks").json()["tasks"] == []  # nothing registered


def test_s3_after_following_a_replica_replaces_what_it_serves_and_the_stale_one_is_refused():
    store = MemoryStore()
    a, _, ba = replica(store)
    b, _, bb = replica(store)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    bb.sync_once()
    assert post_task(b, {"id": "t", "examples": examples(40, 0.5, seed=1)}).json()["store_version"] == 2
    r = post_task(a, {"id": "t", "examples": examples(40, 0.7, seed=2)})  # A still serves version 1
    assert r.status_code == 409 and r.json()["detail"]["store_version"] == 2 and r.json()["detail"]["your_version"] == 1
    assert (
        a.post("/v1/tasks?replace=1", json={"id": "t", "examples": examples(40, 0.7, seed=2)}).json()["store_version"]
        == 3
    )
    ba.sync_once()
    bb.sync_once()
    assert a.get("/v1/tasks/t").json()["store_version"] == 3 == b.get("/v1/tasks/t").json()["store_version"]


def test_s3_a_caller_can_name_the_version_it_was_shown():
    store = MemoryStore()
    a, _, ba = replica(store)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    body = {"id": "t", "examples": examples(40, 0.5, seed=1)}
    wrong = a.post("/v1/tasks", json=body, headers={"x-decisio-replaces": "7"})
    assert wrong.status_code == 409 and wrong.json()["detail"]["your_version"] == 7
    assert a.post("/v1/tasks", json=body, headers={"x-decisio-replaces": "1"}).json()["store_version"] == 2
    assert a.post("/v1/tasks", json=body, headers={"x-decisio-replaces": "x"}).status_code == 422


def test_s3_certification_import_and_removal_follow_the_same_rule():
    store = MemoryStore()
    a, _, ba = replica(store)
    b, _, bb = replica(store)
    post_task(a, {"id": "c", "examples": examples(400, 0.96), "certify": {}})
    r = b.post("/v1/tasks/c/certify", json={"examples": examples(100, 0.96, seed=5, start=5000)})
    assert r.status_code == 404  # B has not got the task yet
    bb.sync_once()
    post_task(a, {"id": "c", "examples": examples(400, 0.95, seed=8), "certify": {}})  # version 2 on A
    r = b.post("/v1/tasks/c/certify", json={"examples": examples(100, 0.96, seed=5, start=5000)})
    assert r.status_code == 409 and r.json()["detail"]["store_version"] == 2
    exported = a.get("/v1/tasks?full=1").json()["tasks"]
    assert b.post("/v1/tasks/import", json={"tasks": exported}).status_code == 409  # B holds version 1 of "c"
    assert b.delete("/v1/tasks/c").status_code == 409
    assert b.delete("/v1/tasks/c?replace=1").status_code == 200


# ---- S4 -------------------------------------------------------------------------------------------------------------


def test_s4_two_replicas_registering_the_same_new_task_at_once_one_wins_and_one_is_told():
    store = MemoryStore()
    a, _, _ = replica(store)
    b, _, _ = replica(store)
    gate, codes = threading.Barrier(2), []

    def go(client, seed):
        gate.wait()
        codes.append(post_task(client, {"id": "t", "examples": examples(40, 0.9, seed=seed)}).status_code)

    threads = [threading.Thread(target=go, args=(c, s)) for c, s in ((a, 1), (b, 2))]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert sorted(codes) == [200, 409]
    c, _, _ = replica(store)
    assert [t["id"] for t in c.get("/v1/tasks").json()["tasks"]] == ["t"]


def test_s4_different_tasks_registered_at_once_both_land():
    store = MemoryStore()
    a, _, _ = replica(store)
    b, _, _ = replica(store)
    gate, codes = threading.Barrier(2), []

    def go(client, body):
        gate.wait()
        codes.append(post_task(client, body).status_code)

    one = {"id": "one", "examples": examples(40, 0.9)}
    two = {"id": "two", "examples": other_list(examples(40, 0.9, seed=2))}
    threads = [threading.Thread(target=go, args=(c, body)) for c, body in ((a, one), (b, two))]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert codes == [200, 200]
    c, _, _ = replica(store)
    assert sorted(t["id"] for t in c.get("/v1/tasks").json()["tasks"]) == ["one", "two"]


def test_s4_a_manifest_write_that_loses_a_race_to_another_task_is_retried(monkeypatch):
    store = MemoryStore()
    a, _, ba = replica(store)
    b, _, bb = replica(store)
    real = ba._write_manifest
    state = {"first": True}

    def racy(manifest, tag):
        if state.pop("first", False):  # another replica publishes a different task between A's read and write
            other = other_list(examples(40, 0.9, seed=5))
            assert post_task(b, {"id": "meanwhile", "examples": other}).status_code == 200
        return real(manifest, tag)

    monkeypatch.setattr(ba, "_write_manifest", racy)
    assert post_task(a, {"id": "mine", "examples": examples(40, 0.9)}).status_code == 200
    c, _, _ = replica(store)
    assert sorted(t["id"] for t in c.get("/v1/tasks").json()["tasks"]) == ["meanwhile", "mine"]
    manifest, _ = ba._read_manifest()
    assert manifest["version"] == 2  # both writes landed, neither replaced the other


# ---- S5 -------------------------------------------------------------------------------------------------------------


def test_s5_a_delete_reaches_the_other_replica_and_the_id_can_be_taught_again():
    store = MemoryStore()
    a, so_a, ba = replica(store)
    b, so_b, bb = replica(store)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    a.post("/v1/abstention/tasks", json=abstention_body())
    bb.sync_once()
    assert a.delete("/v1/tasks/t").status_code == 200 and a.delete("/v1/abstention/tasks/abs").status_code == 200
    assert sorted(bb.sync_once()["removed"]) == ["abs", "t"]
    assert b.get("/v1/tasks").json()["tasks"] == [] and b.get("/v1/abstention/tasks").json()["tasks"] == []
    assert (
        post_task(b, {"id": "t", "examples": examples(40, 0.9)}).json()["store_version"] == 3
    )  # 1, removed 2, again 3


def test_s5_an_id_removed_before_this_replica_saw_it_is_not_a_conflict():
    store = MemoryStore()
    a, _, _ = replica(store)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    a.delete("/v1/tasks/t")
    b, _, bb = replica(store)  # starts after: loads the tombstone only
    assert post_task(b, {"id": "t", "examples": examples(40, 0.9)}).status_code == 200


def test_s5_a_tombstone_dropped_by_retention_before_a_replica_polled_still_removes_the_task():
    clock = Clock()
    store = MemoryStore()
    a, _, ba = replica(store, clock=clock)
    b, so_b, bb = replica(store, clock=clock)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    post_task(a, {"id": "u", "examples": other_list(examples(40, 0.9))})
    bb.sync_once()
    a.delete("/v1/tasks/t")
    clock.t += 9 * 86400
    a.delete("/v1/tasks/u")  # the writer drops the old tombstone of "t"
    assert sorted(bb.sync_once(force=True)["removed"]) == ["t", "u"]
    assert so_b.task_store.by_key == {}


# ---- S6 -------------------------------------------------------------------------------------------------------------


def test_s6_a_replica_on_another_model_keeps_the_task_and_does_not_apply_it():
    store = MemoryStore()
    a, _, _ = replica(store, fingerprint="fp")
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    a.post("/v1/abstention/tasks", json=abstention_body())
    b, so_b, bb = replica(store, fingerprint="another model", name="another served name")
    store_report = b.get("/v1/tasks").json()["store"]
    assert store_report["readout"] == {"loaded": 1, "applied": 0, "other_stack": 1}
    assert store_report["abstention"] == {"loaded": 1, "applied": 0, "other_model": 1}
    assert so_b.task_store.lookup(next(iter(so_b.task_store.by_key))) is None


def test_s6_a_certificate_made_on_another_engine_version_is_suspended_on_that_replica():
    store = MemoryStore()
    a, _, _ = replica(store, version="1.0")
    post_task(a, {"id": "c", "examples": examples(400, 0.96), "certify": {}})
    b, so_b, bb = replica(store, version="2.0")
    ans = b.post("/v1/systemone", json=wire("0:0.99:x")).json()["answers"]["q1"]
    assert ans["certified"] is False and ans["certified_reason"] == "serving_stack_changed"
    ans_a = a.post("/v1/systemone", json=wire("0:0.99:x")).json()["answers"]["q1"]
    assert ans_a["certified"] is True  # the registering replica still serves it certified


# ---- S7 -------------------------------------------------------------------------------------------------------------


def test_s7_a_replica_that_does_not_teach_refuses_every_write_at_once_and_still_follows():
    store = MemoryStore()
    a, _, _ = replica(store)
    r, so_r, br = replica(store, readonly=True)
    reads = []
    real = so_r.engine.answer
    so_r.engine.answer = lambda *x, **k: reads.append(1) or real(*x, **k)
    resp = post_task(r, {"id": "t", "examples": examples(40, 0.9)})
    assert resp.status_code == 409 and "--task-store-readonly" in resp.json()["detail"] and reads == []
    assert r.post("/v1/abstention/tasks", json=abstention_body()).status_code == 409
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    record = a.get("/v1/tasks?full=1").json()["tasks"]
    assert r.post("/v1/tasks/import", json={"tasks": record}).status_code == 409
    assert r.get("/v1/tasks").json()["tasks"] == []  # the import was refused whole
    assert br.sync_once()["applied"] == ["t"]
    assert r.delete("/v1/tasks/t").status_code == 409 and [t["id"] for t in r.get("/v1/tasks").json()["tasks"]] == ["t"]
    assert r.get("/v1/tasks").json()["store"]["saved_to"]["readonly"] is True


# ---- S8 -------------------------------------------------------------------------------------------------------------


def test_s8_a_write_that_the_store_refuses_registers_nothing(monkeypatch):
    store = MemoryStore()
    a, _, ba = replica(store)

    def down(*x, **k):
        raise TB.TaskStoreUnavailable("the task store cannot be written (ConnectionError: boom)")

    monkeypatch.setattr(ba, "_put_object", down)
    r = post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    assert r.status_code == 503 and "the change was not made" in r.json()["detail"]
    assert a.get("/v1/tasks").json()["tasks"] == []
    monkeypatch.undo()
    assert post_task(a, {"id": "t", "examples": examples(40, 0.9)}).status_code == 200


def test_s8_when_the_store_goes_away_the_replica_serves_what_it_has_and_says_so(monkeypatch):
    store = MemoryStore()
    a, _, _ = replica(store)
    post_task(a, {"id": "t", "examples": other_list(examples(40, 0.9))})
    b, so_b, bb = replica(store)
    before = ask(b, 0, 0.9)

    def down(*x, **k):
        raise TB.TaskStoreUnavailable("the task store cannot be read (ConnectionError: boom)")

    monkeypatch.setattr(bb, "_read_manifest", down)
    with pytest.raises(TB.TaskStoreUnavailable):
        bb.sync_once()
    d = b.get("/v1/tasks").json()["store"]["saved_to"]
    assert d["ok"] is False and "cannot be read" in d["last_error"]
    assert ask(b, 0, 0.9) == before and b.get("/health").status_code in (200, 404)  # still answering
    assert b.get("/metrics").text.count("decisio_task_store_ok 0") == 1
    assert post_task(b, {"id": "u", "examples": examples(40, 0.9)}).status_code == 503
    monkeypatch.undo()
    bb.sync_once()
    assert (
        b.get("/v1/tasks").json()["store"]["saved_to"]["ok"] is True
        and "decisio_task_store_ok 1" in b.get("/metrics").text
    )


def test_s8_a_replica_does_not_start_when_it_cannot_read_the_store(monkeypatch):
    backend = S.ObjectBackend(MemoryStore(), "memory://t")

    def down(*x, **k):
        raise TB.TaskStoreUnavailable("the task store cannot be read")

    monkeypatch.setattr(backend, "_read_manifest", down)
    with pytest.raises(TB.TaskStoreUnavailable):
        TB.load_at_start(TaskStore("fp"), backend)


def test_s8_a_store_that_cannot_enforce_preconditions_is_refused(monkeypatch):
    S.ObjectBackend(MemoryStore(), "memory://t").probe()  # a real one passes and leaves nothing behind
    orig = obstore.put

    def ignoring(store, path, data, *, mode=None, **k):
        return orig(store, path, data, **k)  # a store that ignores If-Match and create-only

    monkeypatch.setattr(obstore, "put", ignoring)
    with pytest.raises(TB.TaskPersistError, match="create-only write over an existing object"):
        S.ObjectBackend(MemoryStore(), "memory://t").probe()


def test_s8_a_record_that_does_not_match_its_digest_is_not_applied():
    store = MemoryStore()
    a, _, ba = replica(store)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    manifest, _ = ba._read_manifest()
    digest = next(iter(manifest["entries"].values()))["object"]
    obstore.put(store, f"objects/{digest}.json", b'{"id": "t", "tampered": true}')
    b = S.ObjectBackend(store, "memory://t")
    with pytest.raises(TB.TaskPersistError, match="do not match its digest"):
        b.load()


def test_s8_a_version_that_goes_backwards_is_refused(caplog):
    store = MemoryStore()
    a, _, ba = replica(store)
    b, so_b, bb = replica(store)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    post_task(a, {"id": "t", "examples": examples(40, 0.5, seed=3)})
    bb.sync_once()
    key = next(iter(bb.applied))
    bb.applied[key] = 5  # as if the store had been rolled back to an older manifest
    before = json.dumps(b.get("/v1/tasks?full=1").json()["tasks"], sort_keys=True)
    assert bb.sync_once(force=True) == {"applied": [], "removed": []}
    assert json.dumps(b.get("/v1/tasks?full=1").json()["tasks"], sort_keys=True) == before


# ---- S9 -------------------------------------------------------------------------------------------------------------


def test_s9_the_poll_thread_brings_a_task_taught_elsewhere():
    store = MemoryStore()
    a, _, _ = replica(store)
    b, so_b, bb = replica(store, poll_seconds=0.05)
    sync = S.TaskSync(bb)
    sync.start()
    try:
        post_task(a, {"id": "t", "examples": examples(40, 0.9)})
        deadline = time.time() + 10
        while time.time() < deadline and not so_b.task_store.by_key:
            time.sleep(0.02)
        assert [t["id"] for t in so_b.task_store.by_key.values()] == ["t"]
    finally:
        sync.stop()
        sync.join(5)


def test_s9_credentials_and_query_strings_are_not_logged_or_reported():
    assert S.sanitized("s3://bucket/prefix?x=1") == "s3://bucket/prefix"
    assert S.sanitized("https://user:secret@minio.example:9000/b/p?sig=abc") == "https://minio.example:9000/b/p"


def test_s9_the_flags_are_in_the_parser_and_the_alternatives_are_exclusive(tmp_path):
    a = vllm_engine.build_parser().parse_args(
        [
            "--task-store",
            "s3://b/p",
            "--task-store-endpoint",
            "http://m:9000",
            "--task-store-poll-seconds",
            "2",
            "--task-store-readonly",
        ]
    )
    assert (a.task_store, a.task_store_endpoint, a.task_store_poll_seconds, a.task_store_readonly) == (
        "s3://b/p",
        "http://m:9000",
        2.0,
        True,
    )
    d = vllm_engine.build_parser().parse_args([])
    assert d.task_store is None and d.task_store_poll_seconds == 5.0 and d.task_store_readonly is False
    from test_task_persistence import start_args

    args = start_args(
        tmp_path,
        task_store="memory:///",
        task_store_endpoint=None,
        task_store_readonly=False,
        task_store_poll_seconds=5.0,
    )
    with pytest.raises(SystemExit, match="alternatives"):
        vllm_engine.load_tasks(args, TaskStore("fp"))


def test_s9_the_start_up_wiring_probes_loads_and_refuses_an_empty_store_under_require_tasks(tmp_path, capsys):
    from test_task_persistence import start_args

    args = start_args(
        tmp_path,
        tasks_dir=None,
        task_store="memory:///",
        task_store_endpoint=None,
        task_store_readonly=False,
        task_store_poll_seconds=5.0,
        require_tasks=1,
    )
    with pytest.raises(SystemExit, match=r"--require-tasks 1: 0 task\(s\) apply.*the task store memory:///"):
        vllm_engine.load_tasks(args, TaskStore("fp"))
    ok = start_args(
        tmp_path,
        tasks_dir=None,
        task_store="memory:///",
        task_store_endpoint=None,
        task_store_readonly=True,
        task_store_poll_seconds=5.0,
    )
    _, backend, report = vllm_engine.load_tasks(ok, TaskStore("fp"))
    assert backend.readonly and report["saved_to"]["url"] == "memory:///" and "TASKS" in capsys.readouterr().out
    assert report["from_store"] == {"readout": 0, "abstention": 0}


def test_s9_replicas_seeded_from_the_same_file_do_not_fail_when_one_gets_there_first():
    store = MemoryStore()
    a, _, _ = replica(store)
    post_task(a, {"id": "one", "examples": examples(40, 0.9)})
    seeds = a.get("/v1/tasks?full=1").json()["tasks"]
    other = S.ObjectBackend(MemoryStore(), "memory://t")
    ts = TaskStore("fp")
    # a replica whose store already holds the id skips the seed; one that is read-only leaves it unsaved
    _, report = TB.load_at_start(ts, other, seeds)
    assert report["from_file"]["readout"] == 1
    ro = S.ObjectBackend(MemoryStore(), "memory://t", readonly=True)
    _, report = TB.load_at_start(TaskStore("fp"), ro, seeds)
    assert report["file_tasks_not_saved"] == ["one"]


# ---- S10 ------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("DECISIO_TEST_S3_ENDPOINT"), reason="no S3-compatible service named")
def test_s10_two_replicas_on_a_real_s3_compatible_service():
    """DECISIO_TEST_S3_ENDPOINT=http://127.0.0.1:9000 with AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION in the
    environment and a bucket in DECISIO_TEST_S3_BUCKET (default decisio-test); a fresh prefix per run."""
    import uuid

    url = f"s3://{os.environ.get('DECISIO_TEST_S3_BUCKET', 'decisio-test')}/run-{uuid.uuid4().hex[:8]}"
    endpoint = os.environ["DECISIO_TEST_S3_ENDPOINT"]
    a_back = S.ObjectBackend(S.open_store(url, endpoint), S.sanitized(url))
    a_back.probe()
    ts_a = TaskStore("fp")
    _, rep = TB.load_at_start(ts_a, a_back)
    so_a = SystemOne(FakeEngine(), "decisio-test", task_store=ts_a, task_backend=a_back, task_report=rep)
    app_a = FastAPI()
    add_routes(app_a, so_a)
    a = TestClient(app_a)
    b_back = S.ObjectBackend(S.open_store(url, endpoint), S.sanitized(url))
    ts_b = TaskStore("fp")
    _, rep_b = TB.load_at_start(ts_b, b_back)
    so_b = SystemOne(FakeEngine(), "decisio-test", task_store=ts_b, task_backend=b_back, task_report=rep_b)
    app_b = FastAPI()
    add_routes(app_b, so_b)
    b = TestClient(app_b)
    assert post_task(a, {"id": "t", "examples": examples(40, 0.9)}).json()["store_version"] == 1
    assert b_back.sync_once()["applied"] == ["t"]
    assert b.get("/v1/tasks?full=1").json()["tasks"] == a.get("/v1/tasks?full=1").json()["tasks"]
    assert post_task(b, {"id": "t", "examples": examples(40, 0.5, seed=1)}).json()["store_version"] == 2
    assert post_task(a, {"id": "t", "examples": examples(40, 0.7, seed=2)}).status_code == 409
    assert a.delete("/v1/tasks/t?replace=1").status_code == 200 and b_back.sync_once()["removed"] == ["t"]
