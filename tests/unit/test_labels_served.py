# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Labels from traffic through the server (decisio.serve.labels, SystemOne, the /v1/tasks/{id}/labels routes).

S1  opt-in: a task registered without a labels block logs nothing and its answers carry no request id; with one, the
    registration response gives the settings, the digest key (once) and what the log stores, and the listing names
    the settings but never the secrets
S2  answers: a request id per logged question (and the x-decisio-request-id header), label_requested when drawn; the
    row keeps the plain readout's label log-probabilities and the served distribution, never the request's text; the
    language tag from meta.lang or the x-decisio-lang header, and the two disagreeing is a 422
S3  the routes: post labels by request id, by digest and by state, the blind queue, the status with the label need,
    declines, the export; a task without labels is a 409, an unknown task a 404, a bad body a 422
S4  under --tasks-dir the rows and labels are written and read back by a new server; removing the task removes them
S5  the settings route starts an epoch; erasing one request and everything before a time; their 422s
S6  a task whose options change with every request opts in: its options and answers are kept as digests only, a label
    is checked against that request's own options, and the plain readout is kept

  uv run pytest -q tests/unit/test_labels_served.py
"""

import json
import time

import numpy as np
from test_noul_certified import NoulEngine, make, noul_examples, wire

from decisio.serve import labels as LB
from decisio.serve import task_backends as TB
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore


def register(client, task_id="angry", labels=None, n=120, **body):
    r = client.post("/v1/tasks", json={"id": task_id, "examples": noul_examples(n), "certify": {},
                                       **({"labels": labels} if labels is not None else {}), **body})  # fmt: skip
    assert r.status_code == 200, r.text
    return r.json()


def ask(client, msg, headers=None, **extra):
    r = client.post("/v1/systemone", json=wire(msg, **extra), headers=headers or {})
    assert r.status_code == 200, r.text
    return r.json()["answers"]["q1"], r.headers


def test_s1_opt_in_only_and_the_secrets_stay_out_of_the_listing():
    client, so = make()
    out = register(client)
    assert "labels" not in out
    a, h = ask(client, "yes:0.93:a")
    assert "request_id" not in a and "x-decisio-request-id" not in h and so.labels_logs == {}
    out = register(client, "angry2", labels={"audit_pi": 1.0}, instructions="Is the customer cross?")
    lb = out["labels"]
    assert lb["settings"]["audit_pi"] == 1.0 and len(lb["digest_key"]) == 64 and lb["stores"]
    assert lb["enabled"] is True and lb["text_stored"] is False and lb["features"] == "full"
    assert lb["location"] == "memory" and lb["retention"]["deleted_with_task"] is True
    assert lb["next_to_label"] == {
        "items": [],
        "pending": 0,
        "available": False,
        "reason": "no traffic yet",
        "budget": {"per_day": None},
    }  # fmt: skip (audit_pi given: no budget)
    listed = client.get("/v1/tasks/angry2").json()
    assert listed["labels"]["settings"] == lb["settings"] == listed["labels"]["registered_settings"]
    assert listed["labels"]["stored_fields"] == lb["stored_fields"] and lb["digest_key"] not in json.dumps(listed)


def _labelled(client):
    register(client, labels={"audit_pi": 1.0})
    return client


def test_s2_answers_carry_ids_and_the_row_keeps_no_text():
    client, so = make()
    _labelled(client)
    a, h = ask(client, "yes:0.93:first", headers={"x-decisio-lang": "de"})
    assert a["request_id"].startswith(so.labels_replica + "-") and a["label_requested"] is True
    assert h["x-decisio-request-id"] == a["request_id"]
    log = so.labels_logs["angry"]
    row = log.rows[(a["request_id"], "q1")]
    assert "first" not in json.dumps(row) and row["lang"] == "de" and row["keys"] == ["yes", "no"]
    assert set(row) == set(LB.stored_fields(LB.parse_settings({"audit_pi": 1.0})))  # the disclosure is the row
    # the plain readout (the engine's own, before the task) and the served distribution are both kept
    assert np.allclose(np.exp(LB.unarr(row["lp"])), [0.93, 0.07], atol=1e-6)
    served = LB.unarr(row["ps"])
    assert np.isclose(served[0], a["noul"], atol=1e-6)
    b, _ = ask(client, "no:0.8:second", meta={"lang": "fr"})
    assert log.rows[(b["request_id"], "q1")]["lang"] == "fr"
    same = client.post(
        "/v1/systemone", json=wire("no:0.8:third", meta={"lang": "fr"}), headers={"x-decisio-lang": "fr"}
    )
    assert same.status_code == 200
    clash = client.post(
        "/v1/systemone", json=wire("no:0.8:third", meta={"lang": "fr"}), headers={"x-decisio-lang": "de"}
    )
    assert clash.status_code == 422 and "differ" in clash.text


def test_s3_the_routes():
    client, so = make()
    _labelled(client)
    a, _ = ask(client, "yes:0.93:one")
    b, _ = ask(client, "no:0.9:two")
    c, _ = ask(client, "yes:0.6:three")
    key = so.task_store.snapshot("angry")["labels"]["secret"]["digest_key"]
    r = client.post("/v1/tasks/angry/labels", json={"labels": [
        {"request_id": a["request_id"], "answer": "yes"},
        {"digest": LB.hk1(key, LB.state_text({"message": "no:0.9:two"})), "answer": "yes"},
        {"state": {"message": "yes:0.6:three"}, "answer": "no"},
    ]})  # fmt: skip
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert [x["status"] for x in res] == ["accepted"] * 3
    assert res[1]["request_id"] == b["request_id"] and res[2]["request_id"] == c["request_id"]
    d, _ = ask(client, "no:0.7:four")
    q = client.get("/v1/tasks/angry/labels/queue").json()
    assert [it["request_id"] for it in q["items"]] == [d["request_id"]] and q["pending"] == 1
    assert set(q["items"][0]) == {"request_id", "question", "asked_at", "due_by"}  # blind
    assert set(q) == {"items", "pending", "epoch", "as_of", "budget", "durable"}
    assert set(q["budget"]) == {"per_day", "rate_per_day"}  # no slice, chance or threshold
    st = client.get("/v1/tasks/angry/labels/status").json()
    assert st["states"]["labelled"] == 3 and {n["alpha"] for n in st["need"]} == {0.05, 0.10, 0.01, 0.005}
    assert st.get("durable") is False  # no --tasks-dir: in memory
    d = client.post("/v1/tasks/angry/labels/decline", json={"request_ids": [d["request_id"]]})
    assert d.status_code == 200 and d.json()["declined"] == 1
    lines = [json.loads(x) for x in client.get("/v1/tasks/angry/labels/export").text.splitlines()]
    assert {x["state"] for x in lines} == {"labelled", "declined"}
    # every stored field is exported, the plain readout too: a new version re-scores the rows from it
    assert all(set(LB.stored_fields(LB.parse_settings({"audit_pi": 1.0}))) <= set(x) for x in lines)
    assert np.allclose(np.exp(LB.unarr(lines[0]["lp"])), [0.93, 0.07], atol=1e-6)
    ev = [json.loads(x) for x in client.get("/v1/tasks/angry/labels/export", params={"view": "eval"}).text.splitlines()]
    assert {x["request_id"] for x in ev} == {
        a["request_id"],
        b["request_id"],
        c["request_id"],
        q["items"][0]["request_id"],
    }
    assert client.get("/v1/tasks/angry/labels/export", params={"view": "fit"}).text == ""  # every row is evaluation's
    assert client.get("/v1/tasks/angry/labels/export", params={"view": "cohort"}).text == ""  # none due yet
    assert client.get("/v1/tasks/angry/labels/export", params={"view": "nope"}).status_code == 422
    assert client.get("/v1/tasks/nope/labels/status").status_code == 404
    assert client.post("/v1/tasks/angry/labels", json={"labels": []}).status_code == 422
    assert client.post("/v1/tasks/angry/labels", json=[1]).status_code == 422
    plain, _ = make()  # a task of the same question replaces this one, so a task without labels has its own server
    register(plain, "plain")
    assert plain.get("/v1/tasks/plain/labels/queue").status_code == 409


def _server(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    store = TaskStore("fp")
    backend = TB.DirBackend(tmp_path / "tasks")
    abstention, report = TB.load_at_start(store, backend, None, None)
    so = SystemOne(NoulEngine("1.0"), "decisio-test", task_store=store, abstention_tasks=abstention,
                   task_backend=backend, task_report=report)  # fmt: skip
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so


def test_s4_the_log_is_written_under_the_tasks_dir_and_removed_with_its_task(tmp_path):
    client, so = _server(tmp_path)
    _labelled(client)
    a, _ = ask(client, "yes:0.93:kept")
    client.post("/v1/tasks/angry/labels", json={"labels": [{"request_id": a["request_id"], "answer": "yes"}]})
    so.flush_labels(force=True)
    root = tmp_path / "tasks" / "labels"
    assert any(root.rglob("rows/*.jsonl")) and any(root.rglob("events/*.jsonl"))
    so.task_backend.close()  # the first server stops (it holds the directory's lock)
    client2, so2 = _server(tmp_path)  # a restart over the same tasks directory
    st = client2.get("/v1/tasks/angry/labels/status")
    assert st.status_code == 200, st.text
    assert st.json()["states"]["labelled"] == 1 and "durable" not in st.json()
    assert client2.delete("/v1/tasks/angry").status_code == 200
    assert not any(root.rglob("*.jsonl"))


def test_s5_settings_and_erasure_routes(tmp_path):
    client, so = _server(tmp_path)
    _labelled(client)
    a, _ = ask(client, "yes:0.93:one")
    b, _ = ask(client, "no:0.9:two")
    r = client.put("/v1/tasks/angry/labels/settings", json={"cert_pi": 0.5})
    assert r.status_code == 200, r.text
    assert r.json()["epoch"]["epoch"] == 1 and r.json()["epoch"]["settings"]["cert_pi"] == 0.5
    assert client.get("/v1/tasks/angry/labels/status").json()["settings"]["cert_pi"] == 0.5
    assert set(client.get("/v1/tasks/angry/labels/status").json()["typical"]) == {"state", "labelled", "until", "held"}
    st = client.get("/v1/tasks/angry/labels/status").json()
    assert {"response_rate", "audit", "last_label_at", "since_last_label_s", "languages"} <= set(st)
    got = client.get("/v1/tasks/angry").json()["labels"]
    assert (
        got["settings"]["cert_pi"] == 0.5 and got["registered_settings"]["cert_pi"] == 0.05
    )  # in force, and registered
    for bad in ({"cert_pi": 0.01}, {"nope": 1}, {}):
        assert client.put("/v1/tasks/angry/labels/settings", json=bad).status_code == 422
    client.post("/v1/tasks/angry/labels", json={"labels": [{"request_id": a["request_id"], "answer": "yes"}]})
    r = client.delete(f"/v1/tasks/angry/labels/{a['request_id']}")
    assert r.status_code == 200 and r.json()["state"] == "erased", r.text
    assert r.json()["matched"] == {"answers": 1, "labels": 1}
    assert client.delete("/v1/tasks/angry/labels/queue").status_code == 422  # not a request id
    so.flush_labels()  # the labels thread's turn purges
    lines = client.get("/v1/tasks/angry/labels/export").text
    assert a["request_id"] not in lines and b["request_id"] in lines
    root = tmp_path / "tasks" / "labels"
    on_disk = "".join(x for f in root.rglob("*.jsonl") for x in f.read_text().splitlines(True) if '"erase"' not in x)
    assert a["request_id"] not in on_disk
    assert client.delete("/v1/tasks/angry/labels").status_code == 422  # neither an id nor a time
    r = client.delete("/v1/tasks/angry/labels", params={"before": time.time() + 60})
    assert r.status_code == 200 and client.get("/v1/tasks/angry/labels/status").json()["rows"] == 0


def test_s7_a_settings_change_discloses_what_is_stored():
    client, so = make()
    out = register(client, labels={"audit_pi": 1.0, "features": "minimal"})
    assert "ps" not in out["labels"]["stored_fields"]
    for features in ("full", "minimal"):
        r = client.put("/v1/tasks/angry/labels/settings", json={"features": features})
        assert r.status_code == 200, r.text
        lb = r.json()["labels"]
        assert lb["features"] == features and "digest_key" not in r.text and lb["stores"]
        a, _ = ask(client, f"yes:0.9:{features}")
        assert set(lb["stored_fields"]) == set(so.labels_logs["angry"].rows[(a["request_id"], "q1")])
        got = client.get("/v1/tasks/angry").json()["labels"]
        assert got["settings"]["features"] == features and got["stored_fields"] == lb["stored_fields"]
        assert got["registered_settings"]["features"] == "minimal"
        listed = next(t for t in client.get("/v1/tasks").json()["tasks"] if t["id"] == "angry")["labels"]
        assert listed["settings"]["features"] == features


def test_s6_a_task_whose_options_change_takes_labels_and_keeps_no_option_text():
    import test_dynamic_task as D

    client, so = D.make()
    r = D.post(client, {"id": "brand", **D.DYN, "examples": D.examples(500), "certify": D.CERT,
                        "labels": {"audit_pi": 1.0}})  # fmt: skip
    assert r.status_code == 200, r.text
    assert "options_digest" in r.json()["labels"]["stored_fields"]
    opts = ["brand01x", "brand02x", "brand03x"]
    out = client.post("/v1/systemone", json=D.request(opts, 2, 0.8, "buy the brand03x product today"))
    a = out.json()["answers"]["q1"]
    assert a["request_id"] and a["label_requested"]
    log = so.labels_logs["brand"]
    row = log.rows[(a["request_id"], "q1")]
    dumped = json.dumps(row)
    assert not any(o in dumped for o in opts) and "brand03x product" not in dumped
    assert all(k.startswith("ok1:") for k in row["keys"]) and row["options_digest"].startswith("hk1:")
    assert row["via"] == "model" and row["lp"] is not None
    bad = client.post("/v1/tasks/brand/labels", json={"labels": [{"request_id": a["request_id"], "answer": "9"}]})
    assert bad.json()["results"][0]["status"] == "invalid_answer" and "options" not in bad.json()["results"][0]
    ok = client.post("/v1/tasks/brand/labels", json={"labels": [{"request_id": a["request_id"], "answer": "3"}]})
    assert ok.json()["results"][0]["status"] == "accepted"
    rec = next(x for x in log.table() if x["request_id"] == a["request_id"])
    assert rec["correct"] is (a["choice"] == "3") and rec["answer"] == log.option_key("3")
    assert not any(o in json.dumps(log.events) for o in opts)
