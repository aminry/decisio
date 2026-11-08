# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Certified yes/no tasks (type "noul"; docs/tasks.md, "Certified answers", "Yes/no questions"), end to end on a fake
engine, with --noul-commit off and on, and under both layouts.

N1  registration: the hold-out split, the fit on the fit part, the threshold, `certificate.certified`, the served check
    reading zero, no catch-all warning; gold given as true/false or "yes"/"no" (a yes/no task's fast table:
    tests/unit/test_noul_fast_table.py)
N2  the certificate is made on the confidences the answers report: the committed ones under --noul-commit (never below
    0.80, so no threshold inside the band), and every answer is certified exactly when its reported top probability,
    max(noul, 1 - noul), reaches the threshold
N3  serving: certified true and false (below_threshold), nothing_certified, the fallback to decisio's own read when the
    task is not applied, serving_stack_changed (another engine, the commit rule switched), layout_mismatch under the
    per-request layout
N4  the monitor: the answer mix over the two answers, and the confidence CUSUM, which under --noul-commit reads the
    confidence before the rule (the committed ones leave no tail below 0.80); a record made before reads as it did
N5  re-certification on a new stack, the monitor's reading kept
N6  status, export and import, and persistence (`--tasks-dir`)
N7  the yes/no rendering (--noul-rendering words or letters) is part of the task key: a task applies under its own,
    and under the other one its answers are certified false with serving_stack_changed, not left without the fields

  uv run pytest -q tests/unit/test_noul_certified.py
"""

import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_certified_answers import FakeEngine
from test_request_layout import FollowsLayout
from test_request_layout import server as layout_server

from decisio.readout import certify
from decisio.serve import monitor
from decisio.serve import task_backends as TB
from decisio.serve.systemone import COMMIT_HI, COMMIT_LO, SystemOne, SystemOneRequest, add_routes, commit_noul
from decisio.serve.tasks import TaskStore

QUESTION = {"type": "noul", "instructions": "Is the customer angry?"}


class NoulEngine(FakeEngine):
    """The state's message is `answer:confidence:id`, answer yes or no: the engine puts that confidence on that answer,
    in the order the question is asked (in words: yes then no; as letters, a two-option choice: No then Yes)."""

    def answer(self, state, questions, adapter=None, images=None):
        said, conf = str(state["message"]).split(":")[:2]
        py = float(conf) if said == "yes" else 1.0 - float(conf)
        return [np.array([py, 1.0 - py]) if q["kind"] == "noul" else np.array([1.0 - py, py]) for q in questions], {}


class LayoutNoulEngine(FollowsLayout, NoulEngine):
    pass


def wire(text, **extra):
    return {"state": {"message": text}, "model": "m", "questions": {"q1": dict(QUESTION)}, **extra}


def noul_examples(n, seed=0, start=0, words=False):
    """n labelled yes/no examples, right nine times in ten: a right answer is sure (about 0.95) except one in eight,
    which is unsure (0.55 to 0.78), and a wrong answer is unsure. Gold is true/false, or "yes"/"no" with `words`."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        yes = bool(rng.random() < 0.5)
        right = rng.random() < 0.9
        says_yes = yes if right else not yes
        sure = right and rng.random() >= 0.125
        conf = float(np.clip(rng.normal(0.95, 0.025), 0.86, 0.995)) if sure else float(rng.uniform(0.55, 0.78))
        gold = ("yes" if yes else "no") if words else yes
        out.append({"request": wire(f"{'yes' if says_yes else 'no'}:{conf:.4f}:{start + i}"), "answer": gold})
    return out


def make(commit=False, rendering="words", version="1.0", store=None, backend=None):
    so = SystemOne(
        NoulEngine(version),
        "decisio-test",
        task_store=store or TaskStore("fp"),
        noul_commit=commit,
        noul_rendering=rendering,
        **({"task_backend": backend} if backend is not None else {}),
    )
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so


def register(client, task_id="angry", n=400, seed=0, **body):
    r = client.post("/v1/tasks", json={"id": task_id, "examples": noul_examples(n, seed), "certify": {}, **body})
    assert r.status_code == 200, r.text
    return r.json()


def ask(client, said, conf, tag="x", **extra):
    r = client.post("/v1/systemone", json=wire(f"{said}:{conf}:{tag}", **extra))
    assert r.status_code == 200, r.text
    return r.json()["answers"]["q1"]


def top(a) -> float:
    """A yes/no answer's top probability, the confidence a certificate is about."""
    return max(a["noul"], 1.0 - a["noul"])


def parsed(ex):
    req = SystemOneRequest(**ex["request"])
    return req.state, req.questions["q1"]


GRID = [0.51, 0.55, 0.62, 0.7, 0.75, 0.79, 0.8, 0.81, 0.84, 0.88, 0.9, 0.93, 0.96, 0.99]


# ---- N1 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("commit", [False, True])
def test_n1_registration_fits_on_the_fit_part_and_certifies_on_the_held_out_part(commit):
    client, so = make(commit)
    t = register(client)
    c = t["certificate"]
    assert t["type"] == "noul" and t["options"] == ["yes", "no"] and t["n_examples"] == 200
    assert c["certified"] and c["threshold"] is not None and c["n"] == 200 and c["id"].startswith("angry#")
    assert c["split"]["method"] == "holdout" and c["split"]["fit_examples"] == c["split"]["certify_examples"] == 200
    assert c["stack"]["serving"]["noul_commit"] is commit and c["stack"]["serving"]["noul_rendering"] == "words"
    assert "warnings" not in c  # the catch-all warning is for choice tasks
    s = c["served_check"]
    assert s["n"] == 20 and s["task_applied"] == 20 and s["max_shift"] == 0.0 and s["crossed_threshold"] == 0
    assert s["top_answer_changed"] == 0
    assert t["status"]["state"] == "ok"


def test_n1_gold_as_words_or_booleans_is_the_same_task():
    a, _ = make()
    b, _ = make()
    ta = a.post("/v1/tasks", json={"id": "angry", "examples": noul_examples(400), "certify": {}}).json()
    tb = b.post("/v1/tasks", json={"id": "angry", "examples": noul_examples(400, words=True), "certify": {}}).json()
    assert ta["certificate"]["id"] == tb["certificate"]["id"] and ta["fit_sha256"] == tb["fit_sha256"]
    bad = noul_examples(80) + [{"request": wire("yes:0.9:z"), "answer": "maybe"}]
    assert a.post("/v1/tasks", json={"id": "x", "examples": bad}).status_code == 422


# ---- N2 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("commit", [False, True])
def test_n2_the_certificate_is_made_on_the_confidences_the_answers_report(commit):
    client, so = make(commit)
    held = noul_examples(200, seed=1, start=10_000)
    t = register(client, certify={"examples": held})
    c = t["certificate"]
    assert c["split"]["method"] == "supplied" and c["n"] == 200
    task = so.task_store.snapshot("angry")
    rows = [so.served_rows(task, [parsed(ex)])[0] for ex in held]
    conf = np.array([max(p) for p in rows])
    gold = np.array([0 if ex["answer"] else 1 for ex in held])
    correct = np.array([int(np.argmax(p)) for p in rows]) == gold
    want = certify.certificate(conf, correct, 0.05, 0.1)
    assert c["threshold"] == want["threshold"] and c["certified_n"] == want["certified_n"]
    raw = [so.served_rows(task, [parsed(ex)], commit=False)[0] for ex in held]
    in_band = [COMMIT_LO < p[0] < COMMIT_HI for p in raw]
    assert sum(in_band) >= 20  # the examples put a good share of the answers inside the band
    if commit:
        # the committed confidences: every answer inside the band is at its edge, so no threshold can fall inside it
        assert all(np.allclose(r, commit_noul(p)) for r, p in zip(rows, raw))
        assert conf.min() >= COMMIT_HI and c["threshold"] >= COMMIT_HI
    else:
        assert all(np.allclose(r, p) for r, p in zip(rows, raw))
        assert conf.min() < COMMIT_HI


@pytest.mark.parametrize("commit", [False, True])
def test_n2_an_answer_is_certified_exactly_when_its_reported_top_probability_reaches_the_threshold(commit):
    client, so = make(commit)
    thr = register(client, certify={"suspend_on_alert": False})["certificate"]["threshold"]
    seen = set()
    for said in ("yes", "no"):
        for conf in GRID:
            a = ask(client, said, conf)
            assert (a["noul"] > 0.5) == (said == "yes")  # the commit rule never changes the answer
            if commit and COMMIT_LO < a["noul"] < COMMIT_HI:
                pytest.fail(f"a committed answer inside the band: {a['noul']}")
            assert a["certified"] is (top(a) >= thr), (said, conf, a, thr)
            if not a["certified"]:
                assert a["certified_reason"] == "below_threshold"
            seen.add(a["certified"])
    assert seen == {True, False}


def test_n2_under_the_commit_rule_an_unsure_answer_reports_the_band_edge_and_is_not_certified_above_it():
    client, so = make(commit=True)
    thr = register(client)["certificate"]["threshold"]
    assert thr > COMMIT_HI  # these examples put wrong answers on the band's edge: the edge is not certified
    for said, edge in (("yes", COMMIT_HI), ("no", COMMIT_LO)):
        a = ask(client, said, 0.66)
        assert a["noul"] == pytest.approx(edge) and a["certified"] is False
        assert a["certified_reason"] == "below_threshold"


# ---- N3 -------------------------------------------------------------------------------------------------------------


def test_n3_nothing_certified_is_said_and_the_task_still_answers():
    client, so = make()
    weak = [{**ex, "answer": not ex["answer"]} if i % 3 == 0 else ex for i, ex in enumerate(noul_examples(400))]
    t = client.post("/v1/tasks", json={"id": "weak", "examples": weak, "certify": {}}).json()
    assert t["certificate"]["threshold"] is None and t["certificate"]["summary"].startswith("Nothing is certified")
    a = ask(client, "yes", 0.99)
    assert a["certified"] is False and a["certified_reason"] == "nothing_certified" and a["noul"] > 0.5


@pytest.mark.parametrize("commit", [False, True])
def test_n3_a_task_not_applied_falls_back_to_decisios_own_read(commit):
    client, so = make(commit)
    register(client)
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    plain, _ = make(commit, version="1.1")  # no task: decisio's own read on the other stack
    other, so2 = make(commit, version="1.1")
    so2.task_store.fingerprint = "another model"  # the task was fitted under another model: kept, not applied
    so2.task_store.load(json.loads(json.dumps(exported)))
    for conf in (0.66, 0.97):
        a, p = ask(other, "yes", conf), ask(plain, "yes", conf)
        assert a["noul"] == p["noul"] and a["certified"] is False
        assert a["certified_reason"] == "serving_stack_changed" and a["certificate"] == exported[0]["certificate"]["id"]
    assert other.get("/v1/tasks/angry").json()["status"]["state"] == "suspended"


@pytest.mark.parametrize("commit", [False, True])
def test_n3_another_engine_or_the_commit_rule_switched_suspends_the_certificate(commit):
    client, so = make(commit)
    register(client)
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    for kw, changed in (({"version": "1.1"}, "engine.vllm"), ({"commit": not commit}, "serving.noul_commit")):
        c2, _ = make(**{"commit": commit, **kw})
        assert c2.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
        a = ask(c2, "yes", 0.99)
        assert a["certified"] is False and a["certified_reason"] == "serving_stack_changed"
        st = c2.get("/v1/tasks/angry").json()["status"]
        assert st["state"] == "suspended" and changed in st["changed"]
    same, _ = make(commit)
    assert same.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    assert ask(same, "yes", 0.99)["certified"] is True


def test_n3_a_two_order_request_is_not_certified():
    client, so = make()
    register(client)
    a = ask(client, "yes", 0.99, orders=2)
    assert a["certified"] is False and a["certified_reason"] == "orders_changed"


@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize("server_layout", ["header", "plain"])
def test_n3_under_the_per_request_layout_the_task_answers_its_layout_and_says_layout_mismatch_otherwise(
    commit, server_layout
):
    eng = LayoutNoulEngine().follows(server_layout)
    client, so = layout_server(eng, server_layout, noul_commit=commit)
    other = "plain" if server_layout == "header" else "header"
    body = {"id": "angry", "layout": other, "examples": noul_examples(400), "certify": {}}
    r = client.post("/v1/tasks", json=body)
    assert r.status_code == 200, r.text
    t = r.json()
    assert set(eng.reads) == {other} and so.task_store.layout_of(t) == other
    assert t["certificate"]["served_check"]["task_applied"] == 20
    hit = ask(client, "yes", 0.99, layout=other)
    assert hit["certified"] is True
    unsure = ask(client, "yes", 0.66, layout=other)
    assert unsure["certified"] is False and unsure["certified_reason"] == "below_threshold"
    if commit:
        assert unsure["noul"] == pytest.approx(COMMIT_HI)
    miss = ask(client, "yes", 0.99)  # the server's layout
    assert miss["certified"] is False and miss["certified_reason"] == "layout_mismatch"
    assert miss["certificate"] == t["certificate"]["id"] and eng.reads[-1] == server_layout
    assert ask(client, "yes", 0.99, layout=server_layout)["certified_reason"] == "layout_mismatch"


# ---- N4 -------------------------------------------------------------------------------------------------------------


def traffic(client, n, conf_of, seed=7):
    rng = np.random.default_rng(seed)
    for i in range(n):
        ask(client, "yes" if rng.random() < 0.5 else "no", conf_of(rng), tag=str(i))


@pytest.mark.parametrize("commit", [False, True])
def test_n4_the_confidence_monitor_alerts_on_unsure_traffic_with_the_commit_rule_on_or_off(commit):
    client, so = make(commit)
    register(client)
    base = client.get("/v1/tasks?full=1").json()["tasks"][0]["monitor"]
    assert len(base["option_counts"]) == 2 and sum(base["option_counts"]) == 200
    cb = base["confidence"]
    assert cb.get("reads") == ("before_noul_commit" if commit else None)
    assert cb["tail_threshold"] < COMMIT_HI  # read before the rule: the examples' tail is inside the band
    # traffic like the examples: quiet
    traffic(client, 600, lambda rng: float(rng.normal(0.95, 0.02)) if rng.random() < 0.9 else rng.uniform(0.55, 0.78))
    st = client.get("/v1/tasks/angry").json()["status"]
    assert "confidence" not in st["monitor"]["signals"] and st["monitor"]["answers"] == 600
    assert st["monitor"]["confidence"].get("reads") == cb.get("reads")
    # then unsure traffic, a balanced mix of yes and no: the answers fall below the examples' tail
    traffic(client, 400, lambda rng: rng.uniform(0.52, 0.62), seed=8)
    st = client.get("/v1/tasks/angry").json()["status"]
    assert st["state"] == "traffic_changed" and "confidence" in st["monitor"]["signals"]
    a = ask(client, "yes", 0.99)
    assert a["certified"] is False and a["certified_reason"] == "traffic_changed"


def test_n4_the_answer_mix_watches_the_two_answers():
    client, so = make()
    register(client, certify={"suspend_on_alert": False})
    traffic(client, 600, lambda rng: 0.97)
    assert "answer_mix" not in client.get("/v1/tasks/angry").json()["status"]["monitor"]["signals"]
    for i in range(600):
        ask(client, "yes", 0.97, tag=f"y{i}")
    st = client.get("/v1/tasks/angry").json()["status"]
    assert "answer_mix" in st["monitor"]["signals"] and "not suspended" in st["text"]
    assert ask(client, "no", 0.97)["certified"] is True  # the answer mix reports only


def test_n4_a_record_made_before_reads_the_reported_confidence_as_its_baseline_did():
    client, so = make(commit=True)
    register(client)
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    exported[0]["monitor"]["confidence"].pop("reads")
    c2, so2 = make(commit=True)
    assert c2.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    m = so2.task_store.monitors["angry"]
    assert (
        not m.reads_before_noul_commit
        and "reads" not in c2.get("/v1/tasks/angry").json()["status"]["monitor"]["confidence"]
    )
    tail = exported[0]["monitor"]["confidence"]["tail_threshold"]
    assert 0.6 < tail < COMMIT_HI
    ask(c2, "yes", 0.6)
    # the reported 0.80 is observed, above the tail, where the 0.60 before the rule would have been below it
    assert m.statistic == 0.0
    ask(client, "yes", 0.6)
    assert so.task_store.monitors["angry"].statistic > 0.0  # the record that says it reads before the rule


def test_n4_the_baseline_records_which_confidence_it_read():
    b = monitor.baseline([0, 1] * 50, 2, "s", np.linspace(0.5, 1.0, 100), conf_reads=monitor.BEFORE_NOUL_COMMIT)
    assert b["confidence"]["reads"] == "before_noul_commit"
    assert "reads" not in monitor.baseline([0, 1] * 50, 2, "s", np.linspace(0.5, 1.0, 100))["confidence"]
    assert monitor.MixMonitor(b).reads_before_noul_commit
    assert not monitor.MixMonitor(monitor.baseline([0, 1] * 50, 2, "s")).reads_before_noul_commit


# ---- N5 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("commit", [False, True])
def test_n5_recertification_on_a_new_stack_restores_the_certificate(commit):
    client, so = make(commit)
    old = register(client)["certificate"]
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    c2, so2 = make(commit, version="1.1")
    assert c2.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    assert ask(c2, "yes", 0.99)["certified_reason"] == "serving_stack_changed"
    r = c2.post("/v1/tasks/angry/certify", json={"examples": noul_examples(400)[:5] + noul_examples(80, 3, 5000)})
    assert r.status_code == 422 and "share an input with the fit" in r.json()["detail"]
    r = c2.post("/v1/tasks/angry/certify", json={"examples": noul_examples(200, seed=3, start=5000)})
    assert r.status_code == 200, r.text
    new = r.json()["certificate"]
    assert new["id"] != old["id"] and new["stack"]["engine"]["vllm"] == "1.1" and new["served_check"]["n"] == 20
    assert r.json()["status"]["state"] == "ok"
    assert ask(c2, "yes", 0.99)["certified"] is True
    assert ask(c2, "no", 0.66)["certified_reason"] == "below_threshold"
    reads = c2.get("/v1/tasks?full=1").json()["tasks"][0]["monitor"]["confidence"].get("reads")
    assert reads == ("before_noul_commit" if commit else None)


# ---- N6 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("commit", [False, True])
def test_n6_status_export_and_import_keep_the_certificate_and_the_monitor(commit):
    client, so = make(commit)
    t = register(client)
    listed = client.get("/v1/tasks/angry").json()
    assert listed["status"]["state"] == "ok" and listed["status"]["monitor"]["option_counts"] == [0, 0]
    assert listed["certificate"]["id"] == t["certificate"]["id"]
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    c2, so2 = make(commit)
    assert c2.post("/v1/tasks/import", json={"tasks": json.loads(json.dumps(exported))}).status_code == 200
    again = c2.get("/v1/tasks?full=1").json()["tasks"][0]
    assert again["certificate"] == exported[0]["certificate"] and again["monitor"] == exported[0]["monitor"]
    assert again["fit_sha256"] == exported[0]["fit_sha256"]
    for conf in GRID:
        assert ask(c2, "yes", conf) == ask(client, "yes", conf)


@pytest.mark.parametrize("commit", [False, True])
def test_n6_a_certified_task_persists_across_a_restart(tmp_path, commit):
    def start():
        store = TaskStore("fp")
        backend = TB.DirBackend(tmp_path / "tasks")
        TB.load_at_start(store, backend, None, None)
        return make(commit, store=store, backend=backend)

    client, so = start()
    t = register(client)
    so.task_backend.close()
    c2, so2 = start()
    got = c2.get("/v1/tasks/angry").json()
    assert got["certificate"]["id"] == t["certificate"]["id"] and got["status"]["state"] == "ok"
    assert so2.task_store.monitors["angry"].reads_before_noul_commit is commit
    assert ask(c2, "yes", 0.99)["certified"] is True and ask(c2, "yes", 0.66)["certified"] is False


# ---- N7 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize("rendering", ["letters", "letters-keys"])
def test_n7_the_rendering_is_part_of_the_task_key(commit, rendering):
    words, so_w = make(commit)
    letters, so_l = make(commit, rendering=rendering)
    tw, tl = register(words), register(letters)
    assert tw["key"] != tl["key"] and tl["certificate"]["stack"]["serving"]["noul_rendering"] == rendering
    assert tl["certificate"]["served_check"]["task_applied"] == 20
    # the letters readout comes back false first and is turned round: the same answers as in words
    for conf in (0.66, 0.99):
        for said in ("yes", "no"):
            assert ask(letters, said, conf)["noul"] == pytest.approx(ask(words, said, conf)["noul"])
    assert ask(letters, "yes", 0.99)["certified"] is True
    # each server's task, imported on the other's: not applied (another key), and said so on every answer
    for src, dst_rendering, plain_rendering in ((words, rendering, rendering), (letters, "words", "words")):
        exported = src.get("/v1/tasks?full=1").json()["tasks"]
        dst, dst_so = make(commit, rendering=dst_rendering)
        plain, _ = make(commit, rendering=plain_rendering)
        assert dst.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
        r = dst.post("/v1/systemone", json=wire("yes:0.99:x"))
        assert "x-decisio-tasks" not in r.headers  # not applied
        a = r.json()["answers"]["q1"]
        assert a["certified"] is False and a["certified_reason"] == "serving_stack_changed"
        assert a["certificate"] == exported[0]["certificate"]["id"]
        assert a["noul"] == ask(plain, "yes", 0.99)["noul"]  # decisio's own read
        st = dst.get("/v1/tasks/angry").json()["status"]
        assert st["state"] == "suspended" and "serving.noul_rendering" in st["changed"]


def test_n7_a_question_with_its_own_task_under_this_rendering_is_not_called_stale():
    words, _ = make()
    register(words, task_id="angry-words")
    exported = words.get("/v1/tasks?full=1").json()["tasks"]
    client, so = make(rendering="letters")
    own = register(client)  # this server's own task of the question, under letters
    assert client.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    assert len(so.task_store.by_key) == 2
    r = client.post("/v1/systemone", json=wire("yes:0.99:x"))
    a = r.json()["answers"]["q1"]
    assert r.headers["x-decisio-tasks"] == "angry" and a["certified"] is True
    assert a["certificate"] == own["certificate"]["id"]
