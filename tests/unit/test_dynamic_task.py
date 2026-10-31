# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A task whose options change with every request, through the server (`POST /v1/tasks` with `"options": "dynamic"`).
No model: a fake engine whose probabilities are set by the request's text.

T1  registration: matched on the instructions, always certified, the re-ranker kept where it helps; the response
    holds the fit's evidence, the certificate and the monitor's baseline, and says no request or option text is stored
T2  serving: a request with other options but the same instructions is answered through the task: the re-ranker
    lifts the option the text names, the answer carries the certificate's fields; other instructions are not matched
T3  refusals: no certify block, a score or yes/no question, two sets of instructions, a null answer where no none option
    is offered, another value of options or match; nothing is stored
T4  a re-ranker that does not help is declined (the certificate is made on decisio's own distribution), and says why
T5  the monitor: requests whose options the examples never contained raise the unseen-options alert, which suspends
    certification and shows in the task's status
T6  the listing shows the size of the prior, the export holds it, and an imported task serves the same answers; a stale
    format is refused
T7  re-certification on new examples keeps the fit and accepts null answers

  uv run pytest -q tests/unit/test_dynamic_task.py
"""

import zlib

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore

INS = "Which of these is the brand of the product in this listing?"
BRANDS = [f"brand{i:02d}x" for i in range(40)]


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [zlib.crc32(w.encode()) % 50000 for w in str(text).split()]


class FakeEngine:
    """The state's message is `top:confidence|text`: the engine puts that confidence on the option `top` (an index
    into the request's options) and splits the rest."""

    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"

    def facts(self):
        return {"engine": "fake", "vllm": "1.0", "base": "fake-base", "quantization": None, "mode": "separate"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None):
        head = str(state["message"]).split("|")[0]
        top, conf = head.split(":")[:2]
        out = []
        for q in questions:
            k = len(q["options"])
            p = np.full(k, (1 - float(conf)) / (k - 1))
            p[int(top) % k] = float(conf)
            out.append(p)
        return out, {}


def make(engine=None):
    so = SystemOne(engine or FakeEngine(), "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so


def request(options, top, conf, text, ins=INS, qtype="choice"):
    crit = {str(i + 1): o for i, o in enumerate(options)}
    crit["none"] = "none of these"
    q = {"type": qtype, "instructions": ins, "criteria": crit}
    return {"state": {"message": f"{top}:{conf}|{text}"}, "model": "m", "questions": {"q1": q}}


def examples(n, seed=0, words=BRANDS):
    """Listings that name their brand; decisio's top is right about half the time, whatever the text says."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.integers(3, 6))
        opts = [str(w) for w in rng.choice(words, k, replace=False)]
        g = int(rng.integers(0, k)) if rng.random() < 0.92 else -1
        text = f"buy the {opts[g]} product today" if g >= 0 else "a generic product with no brand"
        top = g if (g >= 0 and rng.random() < 0.5) else int(rng.integers(0, k + 1))
        conf = float(np.round(rng.uniform(0.5, 0.9), 3))
        out.append({"request": request(opts, top, conf, text), "answer": (str(g + 1) if g >= 0 else None)})
    return out


def calibrated_examples(n, seed=0):
    """Requests whose answer is drawn from decisio's own distribution, with a text that names nothing: decisio is
    calibrated by construction and there is nothing for a re-ranker to learn."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.integers(3, 6))
        opts = [str(w) for w in rng.choice(BRANDS, k, replace=False)]
        top, conf = int(rng.integers(0, k + 1)), float(np.round(rng.uniform(0.5, 0.9), 3))
        p = np.full(k + 1, (1 - conf) / k)
        p[top] = conf
        g = int(rng.choice(k + 1, p=p))
        out.append(
            {"request": request(opts, top, conf, "a generic product"), "answer": (str(g + 1) if g < k else None)}
        )
    return out


def post(client, body):
    return client.post("/v1/tasks", json=body)


DYN = {"options": "dynamic", "match": "instructions"}
CERT = {"alpha": 0.1, "delta": 0.1}


def test_t1_registration_matches_on_instructions_is_certified_and_keeps_the_reranker_that_helps():
    client, so = make()
    r = post(client, {"id": "brand", **DYN, "examples": examples(500), "certify": CERT})
    assert r.status_code == 200, r.text
    t = r.json()
    d = t["dynamic"]
    assert t["key"].startswith("dynamic:") and t["options"] is None and t["type"] == "choice"
    assert (
        d["applied"]
        and d["reason"] == "cross-validated gain"
        and d["none_option"] == "none"
        and d["match"] == "instructions"
    )
    assert d["cv"]["accuracy_reranker"] > d["cv"]["accuracy_decisio"] + 0.15 and len(d["weights"]) == 13
    assert d["prior"] == {"options": len(BRANDS)} and "No request text and no option text" in d["storage"]
    c = t["certificate"]
    assert c["alpha"] == 0.1 and c["certified"] and c["n"] == 250 and c["split"]["fit_examples"] == 250
    assert "warnings" not in c  # a none option is offered in the requests: no 'add a catch-all' warning
    assert t["status"]["state"] == "ok" and t["status"]["monitor"]["confidence"]["limit"] > 0
    m = so.task_store.monitors["brand"]
    assert m.dynamic and m.base["option_counts"] is None
    # the head and the calibration are present and not applied
    assert not t["head"]["applied"] and not t["calibration"]["applied"] and "dynamic" in t["head"]["reason"]


def test_t2_a_request_with_other_options_is_answered_through_the_task():
    client, so = make()
    assert post(client, {"id": "brand", **DYN, "examples": examples(500), "certify": CERT}).status_code == 200
    # decisio is confident on a wrong option; the text names the right one, which the re-ranker learned to trust
    wrong = 0
    for i in range(40):
        opts = ["brand01x", "brand02x", "brand03x", "brand04x"]
        body = request(opts, 0, 0.8, "buy the brand03x product today")
        a = client.post("/v1/systemone", json=body).json()["answers"]["q1"]
        wrong += a["choice"] != "3"
        assert set(a["probabilities"]) == {"1", "2", "3", "4", "none"}
        assert "certified" in a and a["certificate"].startswith("brand#")
    assert wrong <= 4
    # other instructions are another question: not matched, no certified field
    other = request(["brand01x", "brand02x"], 0, 0.8, "buy the brand02x product", ins="Which colour is it?")
    a = client.post("/v1/systemone", json=other).json()["answers"]["q1"]
    assert "certified" not in a


def test_t3_refusals_store_nothing():
    client, so = make()
    ex = examples(400)
    cases = [
        ({"id": "t", **DYN, "examples": ex}, "needs a certify block"),
        ({"id": "t", "options": "static", "examples": ex, "certify": CERT}, "'dynamic' is the only value"),
        (
            {"id": "t", "options": "dynamic", "match": "options", "examples": ex, "certify": CERT},
            "match: 'instructions'",
        ),
        ({"id": "t", **DYN, "examples": ex, "certify": CERT, "none_option": ""}, "none_option"),
        ({"id": "t", **DYN, "examples": ex[:60], "certify": {}}, "held-out"),
    ]
    for body, msg in cases:
        r = post(client, body)
        assert r.status_code == 422 and msg in r.text, (msg, r.status_code, r.text[:200])
    mixed = ex[:200] + [{"request": request(["a1", "a2"], 0, 0.8, "x", ins="Another question?"), "answer": "1"}]
    r = post(client, {"id": "t", **DYN, "examples": mixed, "certify": CERT})
    assert r.status_code == 422 and "share one set of instructions" in r.text
    no_none = {
        "request": {
            "state": {"message": "0:0.8|x"},
            "model": "m",
            "questions": {"q1": {"type": "choice", "instructions": INS, "criteria": {"1": "a", "2": "b"}}},
        },
        "answer": None,
    }
    r = post(client, {"id": "t", **DYN, "examples": ex[:200] + [no_none], "certify": CERT})
    assert r.status_code == 422 and "offers no 'none'" in r.text
    score = {
        "request": {
            "state": {"message": "0:0.8|x"},
            "model": "m",
            "questions": {"q1": {"type": "score", "instructions": INS, "criteria": ["bad", "good"]}},
        },
        "answer": 1,
    }
    r = post(client, {"id": "t", **DYN, "examples": ex[:200] + [score], "certify": CERT})
    assert r.status_code == 422 and "choice question" in r.text
    assert not so.task_store.by_key


def test_t4_a_reranker_that_does_not_help_is_declined_and_the_certificate_is_on_decisios_distribution():
    client, so = make()
    r = post(client, {"id": "brand", **DYN, "examples": calibrated_examples(600), "certify": CERT})
    assert r.status_code == 200, r.text
    t = r.json()
    assert not t["dynamic"]["applied"] and "prefers decisio's own distribution" in t["dynamic"]["reason"]
    assert t["certificate"]["id"].startswith("brand#") and "prior" not in t["dynamic"]
    body = request(["brand01x", "brand02x", "brand03x"], 1, 0.9, "buy the brand03x product")
    a = client.post("/v1/systemone", json=body).json()["answers"]["q1"]
    assert a["choice"] == "2" and "certified" in a  # decisio's own answer, certified or not by the threshold
    # a declined task serves decisio's distribution with the server's temperature, as a task-less question would
    so.temperature = 2.0
    flat = client.post("/v1/systemone", json=body).json()["answers"]["q1"]["probabilities"]
    sharp = np.array([0.1 / 3, 0.9, 0.1 / 3, 0.1 / 3])  # the fake engine's p for this request
    want = sharp**0.5 / (sharp**0.5).sum()  # the engine's p at T = 2: p ** (1 / T), renormalised
    assert abs(flat["2"] - float(want[1])) < 1e-3, (flat, want)


def test_t5_requests_whose_options_the_examples_never_contained_raise_an_alert_and_suspend_certification():
    client, so = make()
    r = post(client, {"id": "brand", **DYN, "examples": examples(500), "certify": CERT})
    assert r.status_code == 200 and r.json()["status"]["monitor"]["unseen_options"]["share"] <= 0.05
    new_words = [f"other{i:02d}y" for i in range(40)]
    states = []
    for i in range(400):
        rng = np.random.default_rng(i)
        opts = [str(w) for w in rng.choice(new_words, 3, replace=False)]
        a = client.post("/v1/systemone", json=request(opts, 0, 0.95, f"buy the {opts[1]} product")).json()["answers"][
            "q1"
        ]
        states.append(a)
    st = client.get("/v1/tasks/brand").json()["status"]
    assert st["state"] == "traffic_changed" and "unseen_options" in st["monitor"]["signals"]
    assert "options the examples never contained" in st["text"] and "suspended" in st["text"]
    assert states[-1]["certified"] is False and states[-1]["certified_reason"] == "traffic_changed"
    seen = client.post("/v1/systemone", json=request(["brand01x", "brand02x"], 0, 0.95, "x")).json()["answers"]["q1"]
    assert seen.get("certified_reason") == "traffic_changed"  # still alerting: the window has not cleared


def test_t5b_the_unseen_signal_is_not_armed_where_options_do_not_recur_and_the_status_says_so():
    client, so = make()
    fresh = [f"fresh{i:04d}z" for i in range(4000)]
    exs = examples(300, seed=3, words=fresh)  # almost no option of one example appears in another
    r = post(client, {"id": "entity", **DYN, "examples": exs, "certify": CERT})
    assert r.status_code == 200, r.text
    u = r.json()["status"]["monitor"]["unseen_options"]
    assert u["armed"] is False and u["share"] > 0.2 and "armed below 20%" in u["why"]
    armed = make()[0]
    r = post(armed, {"id": "brand", **DYN, "examples": examples(500), "certify": CERT})
    assert r.json()["status"]["monitor"]["unseen_options"]["armed"] is True


def test_t6_listing_export_and_import():
    client, so = make()
    assert post(client, {"id": "brand", **DYN, "examples": examples(500), "certify": CERT}).status_code == 200
    listed = client.get("/v1/tasks").json()["tasks"][0]
    assert listed["dynamic"]["prior"] == {"options": len(BRANDS)}
    full = client.get("/v1/tasks?full=1").json()["tasks"][0]
    assert len(full["dynamic"]["prior"]) == len(BRANDS) and len(full["dynamic"]["weights"]) == 13
    body = request(["brand01x", "brand02x", "brand03x"], 0, 0.8, "buy the brand02x product today")
    before = client.post("/v1/systemone", json=body).json()["answers"]["q1"]
    client2, so2 = make()
    assert client2.post("/v1/tasks/import", json={"tasks": [full]}).status_code == 200
    after = client2.post("/v1/systemone", json=body).json()["answers"]["q1"]
    assert after["probabilities"] == before["probabilities"] and after["certificate"] == before["certificate"]
    bad = {**full, "dynamic": {**full["dynamic"], "format": "nonsense/1"}}
    assert client2.post("/v1/tasks/import", json={"tasks": [bad]}).status_code in (400, 422)
    assert client2.delete("/v1/tasks/brand").status_code == 200 and not so2.task_store.by_key


def test_t7_recertification_on_new_examples_accepts_null_answers():
    client, so = make()
    assert post(client, {"id": "brand", **DYN, "examples": examples(500), "certify": CERT}).status_code == 200
    fit = client.get("/v1/tasks?full=1").json()["tasks"][0]
    r = client.post("/v1/tasks/brand/certify", json={"examples": examples(150, seed=77), "certify": CERT})
    assert r.status_code == 200, r.text
    new = r.json()
    assert new["certificate"]["split"]["method"] == "supplied" and new["certificate"]["n"] == 150
    assert new["dynamic"]["weights"] == fit["dynamic"]["weights"]  # the fit is kept


def test_t8_a_dynamic_task_survives_a_restart_from_the_tasks_directory(tmp_path):
    from decisio.serve import task_backends as TB

    def start():
        store = TaskStore("fp")
        backend = TB.DirBackend(tmp_path / "tasks")
        abstention, report = TB.load_at_start(store, backend, None, None)
        so = SystemOne(
            FakeEngine(),
            "decisio-test",
            task_store=store,
            abstention_tasks=abstention,
            task_backend=backend,
            task_report=report,
        )
        app = FastAPI()
        add_routes(app, so)
        return TestClient(app), so

    client, so = start()
    assert post(client, {"id": "brand", **DYN, "examples": examples(500), "certify": CERT}).status_code == 200
    body = request(["brand01x", "brand02x", "brand03x"], 0, 0.8, "buy the brand02x product today")
    before = client.post("/v1/systemone", json=body).json()["answers"]["q1"]
    so.task_backend.close()  # the first server ends: one directory belongs to one server
    client2, so2 = start()  # a restart: the task is loaded from the directory
    assert [t["id"] for t in so2.task_store.by_key.values()] == ["brand"]
    after = client2.post("/v1/systemone", json=body).json()["answers"]["q1"]
    assert after["probabilities"] == before["probabilities"] and after["certified"] == before["certified"]
    assert client2.get("/v1/tasks/brand").json()["status"]["state"] == "ok"


def test_t9_a_two_order_request_is_not_certified_and_does_not_fail():
    client, _ = make()
    assert post(client, {"id": "brand", **DYN, "examples": examples(500), "certify": CERT}).status_code == 200
    body = request(["brand01x", "brand02x", "brand03x"], 0, 0.9, "buy the brand02x product today")
    body["orders"] = 2
    a = client.post("/v1/systemone", json=body)
    assert a.status_code == 200, a.text
    ans = a.json()["answers"]["q1"]
    assert ans["certified"] is False and ans["certified_reason"] == "serving_stack_changed"
