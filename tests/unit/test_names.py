# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""decisio's names on the wire and in its records, and the earlier `rlcd` spellings it still reads (`decisio.names`).

W1  response headers: x-decisio-route, x-decisio-server-ms and x-decisio-tasks are written and no x-rlcd-* header
    is; the request headers x-decisio-debug / x-decisio-route are read, and the earlier x-rlcd-debug / x-rlcd-route
    still work, the new spelling winning when both are sent; the debug readout is the body's decisio_debug field
W2  record formats: new records are written as decisio-<kind>/1; the task records committed in runs/ (rlcd-task/1
    with rlcd-task-prior/1 and rlcd-intent-head/1 inside) load and are looked up; an abstention task in either
    spelling loads; a record of another kind, or without a format, is refused
W3  the served name: the default is decisio-qwen3.6-35b-a3b-letters; a task fitted under the earlier default name
    applies on a server with the new default (through the store, POST /v1/tasks/import and --tasks-file), and one
    fitted under any other name does not

  uv run pytest -q tests/unit/test_names.py
"""

import gzip
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from decisio import names
from decisio.serve.tasks import TaskStore

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "2026-09-30_plugin-verification"
LEGACY_FP = json.dumps(
    {
        "desnake_labels": True,
        "hide_index_keys": True,
        "model": "moe-fp8",
        "pad_to": "block",
        "pad_where": "front",
        "served_name": names.LEGACY_SERVED_NAME,
    },
    sort_keys=True,
)
NEW_FP = LEGACY_FP.replace(names.LEGACY_SERVED_NAME, names.SERVED_NAME)


def load_json(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def test_w1_request_header_fallback():
    assert names.request_header({"x-decisio-debug": "readout"}, "debug") == "readout"
    assert names.request_header({"x-rlcd-debug": "hidden"}, "debug") == "hidden"
    assert names.request_header({"x-rlcd-route": "image", "x-decisio-route": "text"}, "route") == "text"
    assert names.request_header({}, "route") is None
    assert names.header("tasks") == "x-decisio-tasks"


@pytest.mark.slow
def test_w1_served_headers():
    from fastapi.testclient import TestClient
    from test_ties import TIED, OrderFreeEngine

    from decisio.serve.systemone import SystemOne
    from decisio.serve.vllm_engine import make_app

    eng = OrderFreeEngine(TIED)
    client = TestClient(make_app(eng, SystemOne(eng, "decisio-test", debug_readout=True)))
    body = {
        "state": "a photo",
        "questions": {"q": {"type": "choice", "instructions": "Which drink?", "criteria": {k: None for k in TIED}}},
    }
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    assert r.headers["x-decisio-route"] == "text" and float(r.headers["x-decisio-server-ms"]) >= 0
    assert not [h for h in r.headers if h.lower().startswith("x-rlcd-")]
    assert names.DEBUG_KEY not in r.json() and "rlcd_debug" not in r.json()
    for h in ("x-decisio-debug", "x-rlcd-debug"):
        r = client.post("/v1/systemone", json=body, headers={h: "readout"})
        assert r.status_code == 200 and r.json()[names.DEBUG_KEY]["q"]["path"], (h, r.text)
    for h in ("x-decisio-route", "x-rlcd-route"):
        assert client.post("/v1/systemone", json=body, headers={h: "gpu"}).status_code == 422  # read, refused
        assert client.post("/v1/systemone", json=body, headers={h: "text"}).status_code == 200


def test_w2_new_records_are_decisio():
    from decisio.readout import calibration, intent_head
    from decisio.serve import abstention, tasks

    assert tasks.FORMAT == "decisio-task/1" and calibration.FORMAT == "decisio-task-prior/1"
    assert intent_head.FORMAT == "decisio-intent-head/1" and abstention.FORMAT == "decisio-abstention/1"
    rng = np.random.default_rng(0)
    lp = np.log(rng.dirichlet(np.ones(3), size=12))
    assert calibration.fit_task_prior(list(lp), [i % 3 for i in range(12)])["format"] == "decisio-task-prior/1"


def test_w2_committed_task_records_load():
    record = load_json(RUN / "intent_heads" / "banking77_d0_task.json.gz")
    assert (record["format"], record["calibration"]["format"], record["head"]["format"]) == (
        "rlcd-task/1",
        "rlcd-task-prior/1",
        "rlcd-intent-head/1",
    )
    store = TaskStore(NEW_FP)
    store.load([record])
    task = store.lookup(record["key"])
    assert task is not None and task["head"]["applied"] and isinstance(task["head"]["A"], np.ndarray)
    assert task["head"]["A"].shape == (2048, 77)
    both = load_json(RUN / "latency_tasks.json.gz")
    store.load(both)
    assert sorted(t["id"] for t in store.by_key.values()) == ["r12_banking77_d0", "r12_v3_clinc_d0"]
    # the same record in the new spelling loads the same way
    renamed = {
        **record,
        "format": "decisio-task/1",
        "calibration": {**record["calibration"], "format": "decisio-task-prior/1"},
        "head": {**record["head"], "format": "decisio-intent-head/1"},
    }
    TaskStore(NEW_FP).load([renamed])


@pytest.mark.parametrize("bad", [{"format": "rlcd-abstention/1"}, {"format": "decisio-task/2"}, {}])
def test_w2_other_records_are_refused(bad):
    record = load_json(RUN / "intent_heads" / "banking77_d0_task.json.gz")
    store = TaskStore(NEW_FP)
    with pytest.raises(ValueError, match="format"):
        store.load([{**{k: v for k, v in record.items() if k != "format"}, **bad}])
    with pytest.raises(ValueError, match="calibration"):
        store.load(
            [
                {
                    **record,
                    "calibration": {**record["calibration"], **bad}
                    if bad
                    else {k: v for k, v in record["calibration"].items() if k != "format"},
                }
            ]
        )
    assert store.by_key == {}  # nothing stored from a refused batch


def test_w2_abstention_tasks_in_either_spelling():
    from decisio.serve.systemone import SystemOne

    def task(fmt):
        return {
            "id": "oos",
            "option": {"key": "out of scope"},
            "match": {"option_set": "x"},
            "config": {"format": fmt, "applied": True, "threshold": 0.4},
            "model": names.LEGACY_SERVED_NAME,
        }

    for fmt in ("rlcd-abstention/1", "decisio-abstention/1"):
        assert "oos" in SystemOne(types.SimpleNamespace(), "m", abstention_tasks=[task(fmt)]).tasks
    with pytest.raises(ValueError, match="abstention"):
        SystemOne(types.SimpleNamespace(), "m", abstention_tasks=[task("rlcd-task/1")])


def test_w3_served_name_and_fingerprints():
    from decisio.serve import vllm_engine

    assert names.SERVED_NAME == "decisio-qwen3.6-35b-a3b-letters"
    from decisio.families import QWEN

    # --served-name defaults to the base's, and the Qwen base's is the served name
    assert QWEN.served_name == names.SERVED_NAME
    assert "args.served_name = fam.served_name" in Path(vllm_engine.__file__).read_text()
    assert names.same_fingerprint(LEGACY_FP, NEW_FP) and names.same_fingerprint(NEW_FP, LEGACY_FP)
    assert not names.same_fingerprint(LEGACY_FP, NEW_FP.replace(names.SERVED_NAME, "my-deployment"))
    assert not names.same_fingerprint(LEGACY_FP, LEGACY_FP.replace("moe-fp8", "another-model"))
    assert names.same_fingerprint("fp", "fp") and not names.same_fingerprint("fp", "other")
    # --describe-options sat in the fingerprint before it entered the task key: such records still match
    import json

    with_flag = json.dumps({**json.loads(NEW_FP), "describe_options": True}, sort_keys=True)
    assert names.same_fingerprint(with_flag, NEW_FP)


def test_w3_import_endpoint_takes_legacy_tasks():
    from fastapi.testclient import TestClient

    from decisio.serve.systemone import SystemOne
    from decisio.serve.vllm_engine import make_app

    so = SystemOne(types.SimpleNamespace(), names.SERVED_NAME, task_store=TaskStore(NEW_FP))
    client = TestClient(make_app(types.SimpleNamespace(adapters={}), so))
    tasks = load_json(RUN / "latency_tasks.json.gz")
    r = client.post("/v1/tasks/import", json={"tasks": tasks})
    assert r.status_code == 200 and r.json()["loaded"] == ["r12_banking77_d0", "r12_v3_clinc_d0"], r.text
    so.task_store = TaskStore(NEW_FP.replace(names.SERVED_NAME, "my-deployment"))
    assert client.post("/v1/tasks/import", json={"tasks": tasks}).status_code == 422


def start_with_tasks(monkeypatch, tasks_file, *argv):
    """The server's main() on a stand-in engine up to the point it would listen, with --tasks-file; its SystemOne."""
    import uvicorn

    import decisio.serve.hf_letters as hf
    from decisio.serve import vllm_engine

    class Engine:
        adapters = {}

        def __init__(self, *a, **kw):
            self.pad_unit = 64

        def facts(self):
            return {}

    served = {}
    monkeypatch.setattr(hf, "HFLettersEngine", Engine)
    monkeypatch.setattr(vllm_engine, "make_app", lambda engine, so: served.update(so=so) or object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "decisio",
            "--backend",
            "hf",
            "--model",
            "/models/moe-fp8",
            "--model-class",
            "view",
            "--tasks-file",
            str(tasks_file),
            *argv,
        ],
    )
    vllm_engine.main()
    return served["so"]


def test_w3_tasks_file_at_start_up(monkeypatch, capsys, tmp_path):
    tasks_file = tmp_path / "tasks.json"  # GET /v1/tasks?full=1 as saved to a file
    tasks_file.write_text(json.dumps(load_json(RUN / "latency_tasks.json.gz")))
    earlier = ["--prompt-tail", "compact", "--noul-rendering", "words"]  # the format these tasks were fitted under
    store = start_with_tasks(monkeypatch, tasks_file, *earlier).task_store  # the default served name
    assert "WARNING" not in capsys.readouterr().out
    assert len(store.by_key) == 2 and all(store.lookup(k) is not None for k in store.by_key)
    store = start_with_tasks(monkeypatch, tasks_file, *earlier, "--served-name", "my-deployment").task_store
    assert "WARNING: tasks fitted under another model or rendering" in capsys.readouterr().out
    assert all(store.lookup(k) is None for k in store.by_key)
    # the served default since 2026-10-03 (the spaced layout): tasks fitted under the compact one must be re-registered
    store = start_with_tasks(monkeypatch, tasks_file).task_store
    assert "WARNING: tasks fitted under another model or rendering" in capsys.readouterr().out
    assert all(store.lookup(k) is None for k in store.by_key)
