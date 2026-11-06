# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Taught tasks kept on disk (`--tasks-dir`, decisio.serve.task_backends): a restart does not lose them.

P1  the directory backend: one file per task, atomic writes with no leftovers, delete, overwrite; a file that cannot be
    read stops the load naming it; one server owns a directory
P2  a write that fails fails the request, and the task is neither registered nor removed: a new task, a replaced one,
    a certification, an import, a removal
P3  restart: a plain task, a certified task and an abstention task come back as they were, answer alike, and a removal
    survives a restart
P4  another stack: a task saved under it loads and is not applied; a certificate made under another serving stack is
    suspended, exactly as for an import
P5  `--tasks-file` with `--tasks-dir`: the file seeds what the directory does not hold, the directory wins otherwise
P6  what /v1/tasks and /health say, and `--require-tasks`
P7  abstention tasks: import, removal, a task of another served model is kept and not applied

  uv run pytest -q tests/unit/test_task_persistence.py
"""

import json
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_certified_answers import FakeEngine, ask, examples, post_task, wire

from decisio.serve import task_backends as TB
from decisio.serve import vllm_engine
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore


def server(tmp_path, fingerprint="fp", version="1.0", name="decisio-test", file_tasks=None, file_abstention=None, **kw):
    """A server started the way `--tasks-dir` starts one: the store, the backend, what loads, then SystemOne."""
    store = TaskStore(fingerprint)
    backend = TB.DirBackend(tmp_path / "tasks") if tmp_path is not None else None
    abstention, report = TB.load_at_start(store, backend, file_tasks, file_abstention)
    so = SystemOne(
        FakeEngine(version),
        name,
        task_store=store,
        abstention_tasks=abstention,
        task_backend=backend,
        task_report=report,
        **kw,
    )
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, backend


def other_list(items):
    """The same examples asked about another option list, so they are another task."""
    crit = {"option_0": "refunds", "option_1": "outages", "option_2": "accounts"}
    out = []
    for ex in items:
        r = json.loads(json.dumps(ex))
        r["request"]["questions"]["q1"]["criteria"] = crit
        out.append(r)
    return out


def abstention_body(task_id="abs", n=12):
    """Examples for `POST /v1/abstention/tasks` on the fake engine: option_2 is the declared abstain option."""
    ex = []
    for i in range(n):
        unanswerable = i % 3 == 0
        top, conf = (2, 0.8) if unanswerable else (i % 2, 0.9)
        ex.append({"request": wire(f"{top}:{conf}:{i}"), "answer": None if unanswerable else f"option_{i % 2}"})
    return {"id": task_id, "option": {"key": "option_2"}, "match": "option_set", "examples": ex}


# ---- P1 -------------------------------------------------------------------------------------------------------------


def test_p1_one_file_per_task_atomic_with_no_leftovers(tmp_path):
    b = TB.DirBackend(tmp_path / "d")
    b.save("readout", {"id": "t/1 weird id", "x": 1})
    b.save("abstention", {"id": "t/1 weird id", "y": 2})
    b.save("readout", {"id": "t/1 weird id", "x": 3})  # replaces
    names = sorted(p.name for p in (tmp_path / "d").iterdir())
    assert [n for n in names if n.endswith(".json")] == sorted(
        [b.file_for("readout", "t/1 weird id").name, b.file_for("abstention", "t/1 weird id").name]
    )
    assert not [n for n in names if n.startswith(".tmp")]
    got = b.load()
    assert got["readout"] == [{"id": "t/1 weird id", "x": 3}] and got["abstention"] == [{"id": "t/1 weird id", "y": 2}]
    b.delete("readout", "t/1 weird id")
    b.delete("readout", "never was")  # not an error
    assert b.load()["readout"] == [] and len(b.load()["abstention"]) == 1


def test_p1_a_stray_temp_file_is_cleaned_and_a_damaged_file_stops_the_load(tmp_path):
    d = tmp_path / "d"
    b = TB.DirBackend(d)
    b.save("readout", {"id": "a"})
    b.close()
    (d / ".tmp-1-dead").write_text("half")
    b = TB.DirBackend(d)
    assert not (d / ".tmp-1-dead").exists() and [t["id"] for t in b.load()["readout"]] == ["a"]
    b.file_for("readout", "a").write_text('{"id": "a", ')  # a damaged file
    with pytest.raises(TB.TaskPersistError, match="readout-.*cannot be read"):
        b.load()
    b.file_for("readout", "a").write_text("[1]")
    with pytest.raises(TB.TaskPersistError, match="not a task record"):
        b.load()


def test_p1_one_server_owns_a_directory(tmp_path):
    b = TB.DirBackend(tmp_path / "d")
    with pytest.raises(TB.TaskPersistError, match="in use by another server"):
        TB.DirBackend(tmp_path / "d")
    b.close()
    TB.DirBackend(tmp_path / "d").close()


# ---- P2 -------------------------------------------------------------------------------------------------------------


def fail_writes(monkeypatch, backend):
    def boom(*a, **k):
        raise TB.TaskPersistError("task 'x': not saved to the disk (No space left on device)")

    monkeypatch.setattr(backend, "save", boom)


def test_p2_a_new_task_that_cannot_be_saved_is_not_registered(tmp_path, monkeypatch):
    client, so, backend = server(tmp_path)
    fail_writes(monkeypatch, backend)
    r = post_task(client, {"id": "t", "examples": examples(40, 0.9)})
    assert (
        r.status_code == 500 and "not saved" in r.json()["detail"] and "the change was not made" in r.json()["detail"]
    )
    assert client.get("/v1/tasks").json()["tasks"] == []


def test_p2_a_replaced_task_keeps_its_previous_record_when_the_new_one_cannot_be_saved(tmp_path, monkeypatch):
    client, so, backend = server(tmp_path)
    assert post_task(client, {"id": "t", "examples": examples(40, 0.9)}).status_code == 200
    before = client.get("/v1/tasks?full=1").json()["tasks"]
    fail_writes(monkeypatch, backend)
    assert post_task(client, {"id": "t", "examples": examples(40, 0.5, seed=7)}).status_code == 500
    assert client.get("/v1/tasks?full=1").json()["tasks"] == before


def test_p2_a_certification_that_cannot_be_saved_leaves_the_previous_certificate(tmp_path, monkeypatch):
    client, so, backend = server(tmp_path)
    body = {"id": "t", "examples": examples(400, 0.96), "certify": {}}
    assert post_task(client, body).status_code == 200
    before = client.get("/v1/tasks/t").json()
    fail_writes(monkeypatch, backend)
    r = post_task(client, {**body, "examples": examples(400, 0.96, seed=3, start=1000)})
    assert r.status_code == 500
    assert client.get("/v1/tasks/t").json() == before  # still the first certificate, in memory
    r = client.post("/v1/tasks/t/certify", json={"examples": examples(100, 0.95, seed=5, start=5000)})
    assert r.status_code == 500 and client.get("/v1/tasks/t").json() == before


def test_p2_a_removal_that_cannot_be_written_leaves_the_task(tmp_path, monkeypatch):
    client, so, backend = server(tmp_path)
    assert post_task(client, {"id": "t", "examples": examples(40, 0.9)}).status_code == 200

    def boom(*a, **k):
        raise TB.TaskPersistError("task 't': not removed from the disk")

    monkeypatch.setattr(backend, "delete", boom)
    assert client.delete("/v1/tasks/t").status_code == 500
    assert [t["id"] for t in client.get("/v1/tasks").json()["tasks"]] == ["t"]


def test_p2_an_import_that_cannot_be_saved_says_so(tmp_path, monkeypatch):
    client, so, backend = server(tmp_path)
    post_task(client, {"id": "t", "examples": examples(40, 0.9)})
    exported = client.get("/v1/tasks?full=1").json()
    other, so2, backend2 = server(tmp_path / "b")
    fail_writes(monkeypatch, backend2)
    r = other.post("/v1/tasks/import", json={"tasks": exported["tasks"]})
    assert r.status_code == 500 and other.get("/v1/tasks").json()["tasks"] == []


def test_p2_the_file_is_written_before_the_response(tmp_path):
    client, so, backend = server(tmp_path)
    assert post_task(client, {"id": "t", "examples": examples(40, 0.9)}).status_code == 200
    on_disk = json.loads(backend.file_for("readout", "t").read_text())
    assert on_disk["id"] == "t" and on_disk["head"]["applied"] is False


# ---- P3 -------------------------------------------------------------------------------------------------------------


def test_p3_a_restart_brings_every_task_back_as_it_was(tmp_path):
    a, so_a, backend = server(tmp_path)
    assert post_task(a, {"id": "plain", "examples": other_list(examples(40, 0.9))}).status_code == 200
    assert post_task(a, {"id": "cert", "examples": examples(400, 0.96), "certify": {}}).status_code == 200
    assert a.post("/v1/abstention/tasks", json=abstention_body()).status_code == 200
    exported = a.get("/v1/tasks?full=1").json()["tasks"]
    abstention = a.get("/v1/abstention/tasks").json()["tasks"]
    answers = [ask(a, t, c) for t, c in ((0, 0.95), (1, 0.6), (2, 0.99))]
    backend.close()

    b, so_b, backend_b = server(tmp_path)  # the restart: a new process would start the same way
    by_id = lambda tasks: sorted(tasks, key=lambda t: t["id"])  # noqa: E731  (the files load in name order)
    assert by_id(b.get("/v1/tasks?full=1").json()["tasks"]) == by_id(exported)
    assert b.get("/v1/abstention/tasks").json()["tasks"] == abstention
    assert [ask(b, t, c) for t, c in ((0, 0.95), (1, 0.6), (2, 0.99))] == answers
    assert len(exported) == 2 and b.get("/v1/tasks/cert").json()["certificate"]["certified"] is True
    assert so_b.task_report["from_dir"] == {"readout": 2, "abstention": 1}


def test_p3_a_removal_survives_a_restart(tmp_path):
    a, _, backend = server(tmp_path)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    a.post("/v1/abstention/tasks", json=abstention_body())
    assert a.delete("/v1/tasks/t").status_code == 200 and a.delete("/v1/abstention/tasks/abs").status_code == 200
    assert a.delete("/v1/tasks/t").status_code == 404 and a.delete("/v1/abstention/tasks/abs").status_code == 404
    backend.close()
    b, _, _ = server(tmp_path)
    assert b.get("/v1/tasks").json()["tasks"] == [] and b.get("/v1/abstention/tasks").json()["tasks"] == []


def test_p3_a_certificate_with_its_served_check_is_what_is_saved(tmp_path):
    client, so, backend = server(tmp_path)
    post_task(client, {"id": "c", "examples": examples(400, 0.96), "certify": {}})
    on_disk = json.loads(backend.file_for("readout", "c").read_text())
    assert "served_check" in on_disk["certificate"] and on_disk["monitor"]  # written after the check, not before


# ---- P4 -------------------------------------------------------------------------------------------------------------


def test_p4_a_task_of_another_stack_loads_and_is_not_applied(tmp_path):
    a, _, backend = server(tmp_path, fingerprint="fp")
    post_task(a, {"id": "t", "examples": examples(60, 0.9)})
    backend.close()
    b, so_b, _ = server(tmp_path, fingerprint="another model")
    listed = b.get("/v1/tasks").json()
    assert [t["id"] for t in listed["tasks"]] == ["t"]  # kept, so a return to the old stack finds it
    assert listed["store"]["readout"] == {"loaded": 1, "applied": 0, "other_stack": 1}
    key = next(iter(so_b.task_store.by_key))
    assert so_b.task_store.lookup(key) is None  # not applied


def test_p4_a_certificate_made_under_another_serving_stack_is_suspended(tmp_path):
    a, _, backend = server(tmp_path, version="1.0")
    post_task(a, {"id": "c", "examples": examples(400, 0.96), "certify": {}})
    backend.close()
    b, so_b, _ = server(tmp_path, version="2.0")  # the same files on an engine of another version
    r = b.post("/v1/systemone", json=wire("0:0.99:x")).json()["answers"]["q1"]
    assert r["certified"] is False and r["certified_reason"] == "serving_stack_changed"


# ---- P5 -------------------------------------------------------------------------------------------------------------


def exported_tasks(tmp_path_factory, ids):
    src = tmp_path_factory.mktemp("src")
    c, _, backend = server(src)
    for n, i in enumerate(ids):
        ex = examples(40, 0.9, seed=n)
        post_task(c, {"id": i, "examples": other_list(ex) if n else ex})
    out = c.get("/v1/tasks?full=1").json()["tasks"]
    abstention = c.post("/v1/abstention/tasks", json=abstention_body("fa"))
    assert abstention.status_code == 200
    backend.close()
    return out, c.get("/v1/abstention/tasks").json()["tasks"]


def test_p5_the_file_seeds_the_directory_and_the_directory_wins(tmp_path, tmp_path_factory):
    tasks, abstention = exported_tasks(tmp_path_factory, ["one", "two"])
    first, so, backend = server(tmp_path, file_tasks=tasks, file_abstention=abstention)
    assert so.task_report["from_file"] == {"readout": 2, "abstention": 1}
    assert sorted(t["id"] for t in first.get("/v1/tasks").json()["tasks"]) == ["one", "two"]
    assert first.delete("/v1/tasks/two").status_code == 200  # taught since: "two" is gone from the directory
    changed = dict(tasks[0], n_examples=999)
    backend.close()
    again, so2, backend2 = server(tmp_path, file_tasks=[changed, tasks[1]], file_abstention=abstention)
    ids = sorted(t["id"] for t in again.get("/v1/tasks").json()["tasks"])
    assert ids == ["one", "two"]  # "two" was absent from the directory, so the file seeds it again
    one = next(t for t in again.get("/v1/tasks").json()["tasks"] if t["id"] == "one")
    assert one["n_examples"] != 999  # the directory held "one": it stays as saved
    assert so2.task_report["file_tasks_left_as_saved"] == ["one", "fa"]
    assert so2.task_report["from_dir"]["readout"] == 1 and so2.task_report["from_file"]["readout"] == 1


def test_p5_without_a_directory_the_file_loads_as_before(tmp_path_factory):
    tasks, abstention = exported_tasks(tmp_path_factory, ["one"])
    client, so, backend = server(None, file_tasks=tasks, file_abstention=abstention)
    assert backend is None and [t["id"] for t in client.get("/v1/tasks").json()["tasks"]] == ["one"]
    assert so.task_report["saved_to"] is None and client.get("/v1/tasks").json()["store"]["saved_to"] is None


# ---- P6 -------------------------------------------------------------------------------------------------------------


def test_p6_the_listing_says_how_many_loaded_and_from_where(tmp_path):
    a, _, backend = server(tmp_path)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    a.post("/v1/abstention/tasks", json=abstention_body())
    backend.close()
    b, so, _ = server(tmp_path)
    store = b.get("/v1/tasks").json()["store"]
    assert store["saved_to"] == {"backend": "dir", "path": str(tmp_path / "tasks")}
    assert store["readout"] == {"loaded": 1, "applied": 1, "other_stack": 0}
    assert store["abstention"] == {"loaded": 1, "applied": 1, "other_model": 0}
    assert store["at_start"]["from_dir"] == {"readout": 1, "abstention": 1}
    assert so.task_store_report() == store


def test_p6_require_tasks_refuses_a_store_with_fewer(tmp_path):
    store = TaskStore("fp")
    backend = TB.DirBackend(tmp_path / "empty")
    abstention, report = TB.load_at_start(store, backend)
    n = TB.usable(store, abstention, "decisio-test")
    assert n == 0
    msg = TB.start_refusal(n, 1, sum(report["from_dir"].values()), backend.path)
    assert msg and "--require-tasks 1: 0 task(s)" in msg and "untaught answers" in msg and "held 0" in msg
    assert TB.start_refusal(n, 0, 0) is None and TB.start_refusal(n, None, None) is None


def test_p6_require_tasks_counts_only_tasks_that_apply(tmp_path):
    a, _, backend = server(tmp_path)
    post_task(a, {"id": "t", "examples": examples(40, 0.9)})
    a.post("/v1/abstention/tasks", json=abstention_body())
    backend.close()
    for fp, name, want in (("fp", "decisio-test", 2), ("other", "decisio-test", 1), ("other", "another", 0)):
        store = TaskStore(fp)
        b = TB.DirBackend(tmp_path / "tasks")
        abstention, _ = TB.load_at_start(store, b)
        assert TB.usable(store, abstention, name) == want, (fp, name)
        b.close()


def test_p6_the_flags_are_in_the_parser():
    a = vllm_engine.build_parser().parse_args(["--tasks-dir", "/data/tasks", "--require-tasks", "3"])
    assert (a.tasks_dir, a.require_tasks) == ("/data/tasks", 3)
    b = vllm_engine.build_parser().parse_args([])
    assert b.tasks_dir is None and b.require_tasks is None


# ---- P7 -------------------------------------------------------------------------------------------------------------


def test_p7_abstention_tasks_import_and_a_task_of_another_model_is_kept_not_applied(tmp_path):
    a, so, backend = server(tmp_path)
    assert a.post("/v1/abstention/tasks", json=abstention_body()).status_code == 200
    record = a.get("/v1/abstention/tasks").json()["tasks"][0]
    assert record["model"] == "decisio-test"
    backend.close()
    b, so_b, _ = server(tmp_path, name="a different served name")
    listed = b.get("/v1/abstention/tasks").json()
    assert [t["id"] for t in listed["tasks"]] == ["abs"] and listed["store"]["abstention"]["other_model"] == 1
    q = so_b.tasks["abs"]
    assert so_b.task_for(_question(), False) is None or not so_b.abstention_applies(q)
    # import: the export of one server loads into another with the same served name, and is refused by a different one
    c, so_c, _ = server(tmp_path / "c")
    assert c.post("/v1/abstention/tasks/import", json={"tasks": [record]}).status_code == 200
    assert c.get("/v1/abstention/tasks").json()["tasks"] == [record]
    assert b.post("/v1/abstention/tasks/import", json={"tasks": [record]}).status_code == 422
    assert c.post("/v1/abstention/tasks/import", json={"tasks": [{"id": "x"}]}).status_code == 422
    assert c.post("/v1/abstention/tasks/import", json={"nope": 1}).status_code == 422


def _question():
    from decisio.serve.systemone import SystemOneRequest

    return SystemOneRequest.model_validate(wire("0:0.9:x")).questions["q1"]


def test_p7_an_abstention_task_that_cannot_be_saved_is_not_registered(tmp_path, monkeypatch):
    client, so, backend = server(tmp_path)
    fail_writes(monkeypatch, backend)
    assert client.post("/v1/abstention/tasks", json=abstention_body()).status_code == 500
    assert client.get("/v1/abstention/tasks").json()["tasks"] == []
    assert os.listdir(tmp_path / "tasks") == [".lock"]


# ---- P8: the start-up wiring (what main() does before it builds the engine) ------------------------------------------


def start_args(tmp_path, **over):
    import argparse

    kw = dict(
        tasks_file=None,
        abstention_tasks=None,
        tasks_dir=str(tmp_path / "tasks"),
        require_tasks=None,
        served_name="decisio-test",
        task_store=None,
        task_store_endpoint=None,
        task_store_readonly=False,
        task_store_poll_seconds=5.0,
    )
    kw.update(over)
    return argparse.Namespace(**kw)


def test_p8_load_tasks_wires_the_directory_the_file_and_the_requirement(tmp_path, tmp_path_factory, capsys):
    tasks, abstention = exported_tasks(tmp_path_factory, ["one"])
    f, g = tmp_path / "tasks.json", tmp_path / "abstention.json"
    f.write_text(json.dumps({"tasks": tasks}))
    g.write_text(json.dumps(abstention))
    store = TaskStore("fp")
    records, backend, report = vllm_engine.load_tasks(
        start_args(tmp_path, tasks_file=str(f), abstention_tasks=str(g), require_tasks=2), store
    )
    assert backend is not None and [t["id"] for t in records] == ["fa"] and report["usable"] == 2
    assert report["from_file"] == {"readout": 1, "abstention": 1} and "TASKS" in capsys.readouterr().out
    assert store.backend is backend and backend.file_for("readout", "one").exists()
    backend.close()
    # a lost store: an empty directory under --require-tasks refuses to start
    with pytest.raises(SystemExit, match=r"--require-tasks 1: 0 task\(s\) apply.*held 0"):
        vllm_engine.load_tasks(start_args(tmp_path, tasks_dir=str(tmp_path / "lost"), require_tasks=1), TaskStore("fp"))
    # without a directory nothing is saved and the requirement still counts
    _, none, report = vllm_engine.load_tasks(start_args(tmp_path, tasks_dir=None, tasks_file=str(f)), TaskStore("fp"))
    assert none is None and report["usable"] == 1
    with pytest.raises(SystemExit, match="--require-tasks 5"):
        vllm_engine.load_tasks(
            start_args(tmp_path, tasks_dir=None, tasks_file=str(f), require_tasks=5), TaskStore("fp")
        )


def test_p8_a_damaged_or_busy_directory_stops_the_start(tmp_path):
    first, _, backend = server(tmp_path)
    post_task(first, {"id": "t", "examples": examples(40, 0.9)})
    with pytest.raises(SystemExit, match="in use by another server"):
        vllm_engine.load_tasks(start_args(tmp_path), TaskStore("fp"))
    backend.close()
    backend.file_for("readout", "t").write_text("{")
    with pytest.raises(SystemExit, match="cannot be read"):
        vllm_engine.load_tasks(start_args(tmp_path), TaskStore("fp"))
