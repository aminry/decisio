# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A task that takes another task's slot (the same question, layout and context) under another id replaces it in the
saved store too: the replaced task's record is removed, so a restart serves the task registered last.

R1  replace, then restart: the newest task serves, the replaced one's file is gone
R2  two restarts, then a third registration in the slot: the newest serves after each restart
R3  three slots (three option sets): replacing in one leaves the others' records alone
R4  the same id registered again in its slot: one file, rewritten
R5  a directory written before this fix (two records in one slot): the most recently written serves, a warning once,
    the start report names the shadowed id, and DELETE of the shadowed id removes its record
R6  an import into an occupied slot removes the replaced task's record
R7  a write that fails rolls back: the replaced task is back in memory and its record is still saved
R8  a seed file (--tasks-file) naming a shadowed id does not displace the directory's newest task
R9  the shared store: a replace tombstones the replaced entry; a store written before the fix (two live entries in one
    slot) loads, and syncs, to the most recently written one
"""

import json
import logging
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient
from obstore.store import MemoryStore
from test_certified_answers import FakeEngine, ask, examples, post_task
from test_task_persistence import fail_writes, server

from decisio.serve import task_backends as TB
from decisio.serve import task_store_shared as S
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore


def files(tmp_path):
    return sorted(p.name for p in (tmp_path / "tasks").glob("readout-*.json"))


def ids_on_disk(tmp_path):
    return sorted(json.loads(p.read_text())["id"] for p in (tmp_path / "tasks").glob("readout-*.json"))


def listed(client):
    return sorted(t["id"] for t in client.get("/v1/tasks").json()["tasks"])


def register(client, task_id, seed=0, **extra):
    r = post_task(client, {"id": task_id, "examples": examples(40, 0.9, seed=seed), **extra})
    assert r.status_code == 200, r.text
    return r.json()


def renamed(items, suffix):
    """The same examples with every option key renamed: another question (its options differ), so another slot."""
    out = []
    for e in items:
        req = json.loads(json.dumps(e["request"]))
        for q in req["questions"].values():
            q["criteria"] = {f"{k}{suffix}": v for k, v in q["criteria"].items()}
        out.append({**e, "request": req, "answer": f"{e['answer']}{suffix}"})
    return out


def restart(tmp_path, backend, **kw):
    backend.close()
    return server(tmp_path, **kw)


# ---- R1 R2 --------------------------------------------------------------------------------------------------------


def test_r1_a_replaced_tasks_record_is_removed_and_the_newest_serves_after_a_restart(tmp_path):
    a, _, backend = server(tmp_path)
    register(a, "first")
    register(a, "second", seed=5)  # the same question: it takes the slot
    assert listed(a) == ["second"]
    assert ids_on_disk(tmp_path) == ["second"]
    b, _, backend = restart(tmp_path, backend)
    assert listed(b) == ["second"]
    assert b.get("/v1/tasks/first").status_code == 404


def test_r2_two_restarts_and_a_third_registration_keep_the_newest(tmp_path):
    a, _, backend = server(tmp_path)
    register(a, "first")
    register(a, "second", seed=5)
    b, _, backend = restart(tmp_path, backend)
    c, _, backend = restart(tmp_path, backend)
    assert listed(c) == ["second"]
    register(c, "third", seed=9)
    assert ids_on_disk(tmp_path) == ["third"]
    d, _, backend = restart(tmp_path, backend)
    e, _, backend = restart(tmp_path, backend)
    assert listed(e) == ["third"] and ids_on_disk(tmp_path) == ["third"]


# ---- R3 R4 --------------------------------------------------------------------------------------------------------


def test_r3_replacing_in_one_slot_leaves_the_other_slots_alone(tmp_path):
    a, _, backend = server(tmp_path)
    register(a, "queue")
    assert post_task(a, {"id": "colour", "examples": renamed(examples(40, 0.9), "_c")}).status_code == 200
    assert post_task(a, {"id": "size", "examples": renamed(examples(40, 0.9), "_s")}).status_code == 200
    register(a, "queue-2", seed=5)  # replaces "queue" only
    assert ids_on_disk(tmp_path) == ["colour", "queue-2", "size"]
    assert post_task(a, {"id": "colour-2", "examples": renamed(examples(40, 0.9, seed=7), "_c")}).status_code == 200
    assert ids_on_disk(tmp_path) == ["colour-2", "queue-2", "size"]
    b, _, backend = restart(tmp_path, backend)
    assert listed(b) == ["colour-2", "queue-2", "size"]


def test_r4_the_same_id_again_rewrites_its_one_file(tmp_path):
    a, _, backend = server(tmp_path)
    register(a, "t")
    before = files(tmp_path)
    register(a, "t", seed=5)
    assert files(tmp_path) == before and ids_on_disk(tmp_path) == ["t"]


# ---- R5: a directory written before this fix --------------------------------------------------------------------


def two_in_one_slot(tmp_path):
    """Two records of one slot on disk, as a server before this fix left them: "older" written first."""
    a, _, backend = server(tmp_path)
    register(a, "older")
    older = a.get("/v1/tasks/older").json()
    exported_older = a.get("/v1/tasks?full=1").json()["tasks"]
    register(a, "newer", seed=5)
    backend.close()
    # put the replaced record back, as the old code left it, and make its file the older one
    path = TB.DirBackend(tmp_path / "tasks")
    path.save("readout", exported_older[0])
    path.close()
    f_old = next(p for p in (tmp_path / "tasks").glob("readout-*.json") if json.loads(p.read_text())["id"] == "older")
    f_new = next(p for p in (tmp_path / "tasks").glob("readout-*.json") if json.loads(p.read_text())["id"] == "newer")
    os.utime(f_old, (1_000_000, 1_000_000))
    os.utime(f_new, (2_000_000, 2_000_000))
    return older


def test_r5_a_slot_with_two_records_serves_the_most_recent_and_warns_once(tmp_path, caplog):
    two_in_one_slot(tmp_path)
    with caplog.at_level(logging.WARNING):
        b, so, backend = server(tmp_path)
    assert listed(b) == ["newer"]
    warned = [r for r in caplog.records if "older" in r.getMessage() and "newer" in r.getMessage()]
    assert len(warned) == 1
    assert so.task_report["superseded"] == [{"id": "older", "by": "newer"}]
    # the shadowed task is not served, and DELETE removes its record
    assert b.get("/v1/tasks/older").status_code == 404
    assert b.delete("/v1/tasks/older").status_code == 200
    assert ids_on_disk(tmp_path) == ["newer"]
    c, so, backend = restart(tmp_path, backend)
    assert listed(c) == ["newer"] and "superseded" not in so.task_report


def test_r5_the_order_follows_the_files_not_their_names(tmp_path):
    two_in_one_slot(tmp_path)
    f_old = next(p for p in (tmp_path / "tasks").glob("readout-*.json") if json.loads(p.read_text())["id"] == "older")
    os.utime(f_old, (3_000_000, 3_000_000))  # now "older" was written last
    b, so, backend = server(tmp_path)
    assert listed(b) == ["older"] and so.task_report["superseded"] == [{"id": "newer", "by": "older"}]


# ---- R6 R7 R8 -----------------------------------------------------------------------------------------------------


def test_r6_an_import_into_an_occupied_slot_removes_the_replaced_record(tmp_path, tmp_path_factory):
    src, _, src_backend = server(tmp_path_factory.mktemp("src"))
    register(src, "imported", seed=5)
    exported = src.get("/v1/tasks?full=1").json()["tasks"]
    a, _, backend = server(tmp_path)
    register(a, "local")
    assert a.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    assert listed(a) == ["imported"] and ids_on_disk(tmp_path) == ["imported"]
    b, _, backend = restart(tmp_path, backend)
    assert listed(b) == ["imported"]


def test_r7_a_replace_that_cannot_be_saved_leaves_the_replaced_task_served_and_saved(tmp_path, monkeypatch):
    a, _, backend = server(tmp_path)
    register(a, "first")
    answer = ask(a, 0, 0.95)
    fail_writes(monkeypatch, backend)
    assert post_task(a, {"id": "second", "examples": examples(40, 0.9, seed=5)}).status_code == 500
    assert listed(a) == ["first"] and ask(a, 0, 0.95) == answer
    monkeypatch.undo()
    assert ids_on_disk(tmp_path) == ["first"]
    b, _, backend = restart(tmp_path, backend)
    assert listed(b) == ["first"]


def test_r8_a_seed_file_naming_a_shadowed_id_does_not_displace_the_newest(tmp_path):
    two_in_one_slot(tmp_path)
    # the seed names "older", which the directory holds (shadowed): left as the directory has it
    raw = next(
        json.loads(p.read_text())
        for p in (tmp_path / "tasks").glob("readout-*.json")
        if json.loads(p.read_text())["id"] == "older"
    )
    c, so, backend = server(tmp_path, file_tasks=[raw])
    assert listed(c) == ["newer"]
    assert "older" in so.task_report.get("file_tasks_left_as_saved", [])


# ---- R9: the shared store ----------------------------------------------------------------------------------------


def shared(store, name="decisio-test"):
    ts = TaskStore("fp")
    backend = S.ObjectBackend(store, "memory://t")
    abstention, report = TB.load_at_start(ts, backend)
    so = SystemOne(
        FakeEngine(), name, task_store=ts, abstention_tasks=abstention, task_backend=backend, task_report=report
    )
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, backend


def live_ids(backend):
    manifest, _ = backend._read_manifest()
    return sorted(e["id"] for e in manifest["entries"].values() if not e.get("deleted"))


def test_r9_a_replace_on_the_shared_store_tombstones_the_replaced_entry(tmp_path):
    store = MemoryStore()
    a, _, backend = shared(store)
    register(a, "first")
    register(a, "second", seed=5)
    assert live_ids(backend) == ["second"]
    b, _, _ = shared(store)
    assert listed(b) == ["second"]


def test_r9_a_store_with_two_live_entries_in_one_slot_loads_and_syncs_to_the_newest(tmp_path):
    store = MemoryStore()
    a, so_a, backend = shared(store)
    register(a, "older")
    older = a.get("/v1/tasks?full=1").json()["tasks"][0]
    follower, so_f, backend_f = shared(store)  # a replica that has applied "older"
    register(a, "newer", seed=5)
    backend.save("readout", older)  # the old code left "older" live; here it is written again after "newer"
    backend.save("readout", next(t for t in a.get("/v1/tasks?full=1").json()["tasks"] if t["id"] == "newer"))
    assert live_ids(backend) == ["newer", "older"]
    b, so_b, _ = shared(store)
    assert listed(b) == ["newer"] and so_b.task_report["superseded"] == [{"id": "older", "by": "newer"}]
    backend_f.sync_once(force=True)
    assert listed(follower) == ["newer"]
