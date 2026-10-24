# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Certified answers and their monitor (`decisio.readout.certify`, `decisio.serve.monitor`, `decisio.serve.stack`,
`POST /v1/tasks` with a `certify` block). No model: a fake engine whose answer is set by the example's text.

C1  the procedure: the thresholds of a reimplementation of shad0w's `auto` procedure on fixed inputs, the
fewest examples
    that can certify (59 at alpha 0.05 and delta 0.1), ties served together, a threshold that holds on a later sample;
C2  the held-out split: deterministic, independent of the order, same input on the same side, disjoint from the fit;
C3  registration: refusal below the minimum with the numbers, the stored certificate (threshold, alpha, delta, n,
certified
    share, the split, the stack), the fit on the fit part only, nothing certified said plainly, score refused;
C4  answers carry certified true or false with the certificate's id and, when false, why;
C5  the stack: the same fingerprint on a fresh server, every answer certified false with the status `suspended:
serving stack
    changed` when it differs, an edited fit suspends too, re-certification on new examples restores it and refuses
    examples
    the fit saw;
C6  the answer-mix monitor: counts of options only, quiet on the certification mix, the alert on a shifted one, in
the task's
    status and in /metrics; a certified task's export and import keep everything;
C7  a task without certification serves and lists exactly as before.

  uv run pytest -q tests/unit/test_certified_answers.py
"""

import json
import zlib

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.readout import certify
from decisio.serve import monitor
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore, example_digest, split_holdout

# ---- C1 -------------------------------------------------------------------------------------

# (n, accuracy, alpha, delta, sum of conf, right, threshold): a reimplementation of the auto procedure
# (equal to shad0w 0.3.3's auto controller on every split) on data drawn by the generator below
REFERENCE = [
    (300, 0.97, 0.05, 0.1, 249.269, 270, 0.804),
    (300, 0.95, 0.05, 0.1, 251.336, 283, 0.786),
    (1000, 0.9, 0.05, 0.1, 822.687, 878, 0.846),
    (200, 0.8, 0.1, 0.1, 161.842, 152, None),
    (100, 0.99, 0.05, 0.1, 80.164, 95, None),
    (60, 1.0, 0.05, 0.1, 50.633, 57, None),
    (500, 0.85, 0.02, 0.05, 413.539, 427, None),
]


def reference_data():
    rng = np.random.default_rng(20261014)
    for n, acc, a, d, csum, right, thr in REFERENCE:
        conf = np.round(np.clip(rng.beta(6, 1.3, n), 0, 1), 3)
        ok = rng.random(n) < np.clip(acc + (conf - 0.85) * 0.8, 0, 1)
        yield n, a, d, conf, ok, csum, right, thr


def test_c1_thresholds_equal_the_reference_reimplementation():
    for n, a, d, conf, ok, csum, right, thr in reference_data():
        assert round(float(conf.sum()), 3) == csum and int(ok.sum()) == right, "the generator's stream changed"
        got = certify.threshold(conf, ok, a, d)
        assert (got == thr) if thr is not None else np.isinf(got)


def test_c1_fewest_examples_that_can_certify():
    assert certify.minimum_examples(0.05, 0.1) == 59 and certify.minimum_examples(0.1, 0.1) == 29
    assert certify.minimum_examples(0.02, 0.05) == 183
    for a, d in ((0.05, 0.1), (0.1, 0.05)):
        n = certify.minimum_examples(a, d)
        right = lambda m: certify.threshold(np.ones(m), np.ones(m, dtype=bool), a, d)  # noqa: E731
        assert np.isfinite(right(n)) and np.isinf(right(n - 1))


def test_c1_ties_are_served_together_and_the_certificate_is_consistent():
    conf = np.array([0.99] * 200 + [0.9] * 100 + [0.6] * 100)
    ok = np.array([True] * 200 + [True] * 98 + [False] * 2 + [True] * 60 + [False] * 40)
    c = certify.certificate(conf, ok, 0.05, 0.1)
    assert c["certified"] and c["threshold"] in (0.99, 0.9) and c["certified_n"] in (200, 300)
    assert c["certified_n"] == int((conf >= c["threshold"]).sum())
    assert c["errors_at_threshold"] == int((~ok[conf >= c["threshold"]]).sum())
    assert (
        c["certified_share"] == c["certified_n"] / len(conf)
        and c["upper_bound_at_threshold"] >= c["error_rate_at_threshold"]
    )
    none = certify.certificate(np.full(100, 0.7), np.array([True, False] * 50), 0.05, 0.1)
    assert not none["certified"] and none["threshold"] is None and none["certified_share"] == 0.0


def test_c1_a_certified_threshold_holds_on_a_later_sample():
    rng = np.random.default_rng(3)
    conf_all = np.clip(rng.beta(6, 1.3, 20000), 0, 1)
    ok_all = rng.random(20000) < np.clip(0.88 + (conf_all - 0.85) * 0.9, 0, 1)
    bad = 0
    for s in range(60):
        idx = rng.permutation(20000)
        cal, rest = idx[:400], idx[400:]
        t = certify.threshold(conf_all[cal], ok_all[cal], 0.05, 0.1)
        if np.isfinite(t):
            served = conf_all[rest] >= t
            bad += float((~ok_all[rest][served]).mean()) > 0.05
    assert bad <= 6  # delta 0.1 over 60 draws: the bound is on the share of draws that break the promise


# ---- C2 -------------------------------------------------------------------------------------


class Q:
    type = "choice"

    def __init__(self, text):
        self.text = text

    def model_dump(self, mode="json"):
        return {"type": "choice", "instructions": self.text}


def test_c2_split_is_deterministic_ordered_by_digest_and_disjoint():
    ex = [(Q("same"), {"i": i // 2}, "a") for i in range(120)]  # pairs share an input
    ex = [(Q("q"), {"i": i // 2}, "a") for i in range(120)]
    fit, held = split_holdout("task", ex, 0.5)
    assert len(held) == 60 and set(fit) | set(held) == set(range(120)) and not set(fit) & set(held)
    again = split_holdout("task", list(reversed(ex)), 0.5)
    assert {example_digest(ex[i][1], ex[i][0]) for i in held} == {
        example_digest(list(reversed(ex))[i][1], list(reversed(ex))[i][0]) for i in again[1]
    }
    sides = {}
    for side, idx in (("fit", fit), ("held", held)):
        for i in idx:
            sides.setdefault(example_digest(ex[i][1], ex[i][0]), set()).add(side)
    assert all(len(v) == 1 for v in sides.values())  # one input, one side
    assert split_holdout("task", ex, 0.5) == (fit, held) and split_holdout("other", ex, 0.5)[1] != held


# ---- C3 to C7: a server on a fake engine
# -----------------------------------------------------------------------------------


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        text = str(text)
        if len(text.strip()) <= 2 and text.strip().isalpha():  # a letter code, with or without its space: one token
            return [1000 + sum(ord(c) * 53**i for i, c in enumerate(text.strip()))]
        return [zlib.crc32(w.encode()) % 50000 for w in text.split()]


CRIT = {"option_0": "billing", "option_1": "shipping", "option_2": "access"}


def wire(text, crit=CRIT, qtype="choice"):
    q = {"type": "choice", "instructions": "Which queue?", "criteria": crit} if qtype == "choice" else None
    return {"state": {"message": text}, "model": "m", "questions": {"q1": q}}


class FakeEngine:
    """The state's message is `top:confidence`: the engine puts that confidence on option `top` and splits the rest."""

    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"

    def __init__(self, version="1.0"):
        self.version = version

    def facts(self):
        return {"engine": "fake", "vllm": self.version, "base": "fake-base", "quantization": None, "mode": "separate"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None):
        top, conf = str(state["message"]).split(":")[:2]
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            p = np.full(k, (1 - float(conf)) / (k - 1))
            p[int(top)] = float(conf)
            out.append(p)
        return out, {}


def make(version="1.0", engine=None):
    eng = engine or FakeEngine(version)
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so


def examples(n, accuracy, seed=0, start=0):
    """n labelled examples: the engine's top option is the gold with probability `accuracy`; its confidence is
    high when it is
    right more often (a ranking the certification can use)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        gold = int(rng.integers(0, 3))
        right = rng.random() < accuracy
        top = gold if right else (gold + 1 + int(rng.integers(0, 2))) % 3
        conf = float(np.clip(rng.normal(0.93 if right else 0.7, 0.04), 0.4, 0.995))
        out.append({"request": wire(f"{top}:{conf:.4f}:{start + i}"), "answer": f"option_{gold}"})
    return out


def post_task(client, body):
    return client.post("/v1/tasks", json=body)


def ask(client, top, conf):
    r = client.post("/v1/systemone", json=wire(f"{top}:{conf}:x"))
    assert r.status_code == 200, r.text
    return r.json()["answers"]["q1"]


def test_c3_refusal_below_the_minimum_says_the_numbers():
    client, so = make()
    r = post_task(client, {"id": "t", "examples": examples(80, 0.95), "certify": {}})  # 40 held out, 59 needed
    assert r.status_code == 422
    msg = r.json()["detail"]
    assert "40 held-out" in msg and "at least 59" in msg and "alpha 0.05" in msg and "delta 0.1" in msg
    assert not so.task_store.by_key  # nothing registered
    r = post_task(client, {"id": "t", "examples": examples(300, 0.95), "certify": {"min_examples": 200}})
    assert (
        r.status_code == 422 and "at least 200" in r.json()["detail"] and "asked for at least 200" in r.json()["detail"]
    )
    for bad in ({"alpha": 2}, {"holdout": 0}, {"nonsense": 1}):
        assert post_task(client, {"id": "t", "examples": examples(300, 0.95), "certify": bad}).status_code == 422


def test_c3_certification_is_stored_with_its_evidence_and_the_fit_sees_the_fit_part_only():
    client, so = make()
    body = {"id": "routing", "examples": examples(400, 0.96), "certify": {"alpha": 0.05, "delta": 0.1}}
    r = post_task(client, body)
    assert r.status_code == 200, r.text
    t = r.json()
    c = t["certificate"]
    assert c["alpha"] == 0.05 and c["delta"] == 0.1 and c["n"] == 200 and c["certified"] and 0.5 < c["threshold"] < 1
    assert c["certified_share"] == c["certified_n"] / 200 and c["id"].startswith("routing#")
    assert (
        c["split"]["method"] == "holdout"
        and c["split"]["fit_examples"] == 200
        and c["split"]["certify_examples"] == 200
    )
    assert c["split"]["disjoint"] and c["split"]["fit_digests"] == 200  # the listing counts them, the export lists them
    assert t["n_examples"] == 200  # the calibration was fitted on the fit part only
    assert c["stack"]["decisio"] and c["stack"]["engine"]["vllm"] == "1.0" and c["stack"]["prompt"]["probe_sha256"]
    assert "certified" in c["summary"] and t["status"]["state"] == "ok"
    full = client.get("/v1/tasks?full=1").json()["tasks"][0]["certificate"]
    assert len(full["split"]["fit_digests"]) == 200
    fit_digests = set(full["split"]["fit_digests"])
    held = {example_digest(s, q)[:16] for q, s, _ in [(Q(""), {}, 0)]}  # noqa: F841 (shape check only)
    assert len(fit_digests) == 200


def test_c3_nothing_certified_is_said_plainly_and_the_task_works_as_before():
    client, so = make()
    r = post_task(client, {"id": "weak", "examples": examples(400, 0.6, seed=2), "certify": {}})
    assert r.status_code == 200
    c = r.json()["certificate"]
    assert not c["certified"] and c["threshold"] is None and c["certified_share"] == 0.0
    assert c["summary"].startswith("Nothing is certified") and "works as before" in c["summary"]
    a = ask(client, 0, 0.99)
    assert a["certified"] is False and a["certified_reason"] == "nothing_certified" and a["choice"] == "option_0"
    assert r.json()["calibration"]["n"] if "n" in r.json()["calibration"] else True


def test_c3_score_questions_are_not_certified():
    client, so = make()
    score = {
        "state": {"m": "0:0.9:1"},
        "model": "m",
        "questions": {"q1": {"type": "score", "criteria": ["low", "mid", "high"]}},
    }
    r = post_task(client, {"id": "s", "examples": [{"request": score, "answer": "1"}] * 120, "certify": {}})
    assert r.status_code == 422 and "score questions are not certified" in r.json()["detail"]


def test_c4_answers_carry_certified_true_or_false():
    client, so = make()
    t = post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {}}).json()
    thr, cid = t["certificate"]["threshold"], t["certificate"]["id"]
    hi, lo = ask(client, 1, 0.99), ask(client, 1, round(thr - 0.2, 3))
    assert hi["certified"] is True and hi["certificate"] == cid and "certified_reason" not in hi
    assert lo["certified"] is False and lo["certified_reason"] == "below_threshold" and lo["certificate"] == cid
    two = client.post("/v1/systemone", json={**wire("1:0.99:x"), "orders": 2})
    assert two.status_code == 200 and two.json()["answers"]["q1"]["certified"] is False  # another form than certified


def test_c7_a_task_without_certification_is_as_before():
    client, so = make()
    r = post_task(client, {"id": "plain", "examples": examples(40, 0.9)})
    assert (
        r.status_code == 200
        and "certificate" not in r.json()
        and "status" not in r.json()
        and "fit_sha256" not in r.json()
    )
    a = ask(client, 0, 0.9)
    assert not {"certified", "certificate", "certified_reason"} & set(a)
    assert "status" not in client.get("/v1/tasks").json()["tasks"][0]
    assert client.get("/metrics").text == ""


def test_c5_the_stack_fingerprint_suspends_and_recertifying_restores():
    client, so = make("1.0")
    task = post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {}}).json()
    thr = task["certificate"]["threshold"]
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    # a fresh server of the same stack: the certificate holds
    c2, so2 = make("1.0")
    assert c2.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    assert ask(c2, 1, 0.99)["certified"] is True
    assert c2.get("/v1/tasks/routing").json()["status"]["state"] == "ok"
    # another engine version: every answer is certified false, the status says why, the calibration still applies
    c3, so3 = make("1.1")
    assert c3.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    a = ask(c3, 1, 0.99)
    assert a["certified"] is False and a["certified_reason"] == "serving_stack_changed" and a["choice"] == "option_1"
    st = c3.get("/v1/tasks/routing").json()["status"]
    assert (
        st["state"] == "suspended"
        and st["text"] == "suspended: serving stack changed"
        and "engine.vllm" in st["changed"]
    )
    assert c3.get("/v1/tasks").json()["tasks"][0]["status"]["state"] == "suspended"
    # re-certifying is an explicit call, on new examples the fit has not seen
    seen = [e for e in examples(400, 0.96)][:5]
    r = c3.post("/v1/tasks/routing/certify", json={"examples": seen + examples(120, 0.96, seed=9, start=1000)})
    assert r.status_code == 422 and "share an input with the fit" in r.json()["detail"]
    r = c3.post("/v1/tasks/routing/certify", json={"examples": examples(40, 0.96, seed=9, start=1000)})
    assert r.status_code == 422 and "at least 59" in r.json()["detail"]
    r = c3.post("/v1/tasks/routing/certify", json={"examples": examples(160, 0.96, seed=9, start=1000)})
    assert r.status_code == 200, r.text
    assert r.json()["status"]["state"] == "ok" and r.json()["certificate"]["split"]["method"] == "supplied"
    assert (
        r.json()["certificate"]["stack"]["engine"]["vllm"] == "1.1"
        and r.json()["certificate"]["id"] != task["certificate"]["id"]
    )
    assert ask(c3, 1, 0.99)["certified"] is True
    assert c3.post("/v1/tasks/ghost/certify", json={"examples": []}).status_code == 404
    assert thr is not None


def test_c5_an_edited_fit_suspends_the_certificate():
    client, so = make()
    post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {}})
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    assert exported[0]["calibration"].get("applied") in (True, False)
    exported[0]["calibration"] = (
        {**exported[0]["calibration"], "bias": [0.5, 0.0, -0.5]}
        if "bias" in exported[0]["calibration"]
        else {
            **exported[0]["calibration"],
            "edited": True,
        }
    )
    c2, so2 = make()
    assert c2.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    a = ask(c2, 1, 0.99)
    assert a["certified"] is False and a["certified_reason"] == "serving_stack_changed"
    assert "calibration.task_fit" in c2.get("/v1/tasks/routing").json()["status"]["changed"]


def test_c6_the_monitor_counts_options_only_and_alerts_on_a_changed_mix():
    client, so = make()
    exs = examples(600, 0.96, seed=5)
    t = post_task(client, {"id": "routing", "examples": exs, "certify": {"suspend_on_alert": False}}).json()
    base = monitor.baseline([0, 1, 2] * 100, 3, "x")
    assert base["window"] == 500 and base["step"] == 100 and len(base["option_counts"]) == 3 and 0 < base["line"] < 0.2
    mix0 = np.array(
        t["certificate"] and so.task_store.by_key[next(iter(so.task_store.by_key))]["monitor"]["option_counts"], float
    )
    mix0 /= mix0.sum()
    rng = np.random.default_rng(1)
    for _ in range(600):  # the certification mix
        ask(client, int(rng.choice(3, p=mix0)), 0.99)
    st = client.get("/v1/tasks/routing").json()["status"]
    assert st["state"] == "ok" and st["monitor"]["answers"] == 600 and st["monitor"]["checked_at"] == 600
    assert set(st["monitor"]) >= {"distance", "line", "option_counts", "window_answers", "alerts"}
    for _ in range(600):  # every answer option 0
        ask(client, 0, 0.99)
    st = client.get("/v1/tasks/routing").json()["status"]
    assert (
        st["state"] == "traffic_changed"
        and st["monitor"]["distance"] > st["monitor"]["line"]
        and "not suspended" in st["text"]
    )
    assert ask(client, 1, 0.99)["certified"] is True  # an alert does not change an answer
    text = client.get("/metrics").text
    assert 'decisio_task_state{task="routing",state="traffic_changed"} 1' in text
    assert (
        'decisio_task_state{task="routing",state="ok"} 0' in text
        and 'decisio_task_mix_distance{task="routing"}' in text
    )
    assert "decisio_task_certified_answers_total" in text and "message" not in text  # counts only
    blob = json.dumps(st)
    assert "0:0.99" not in blob and "billing" not in blob  # no state, no text, no option names in the monitor


def test_c6_export_and_import_keep_the_certificate_and_the_monitor_baseline():
    client, so = make()
    t = post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {}}).json()
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    c2, so2 = make()
    assert c2.post("/v1/tasks/import", json={"tasks": json.loads(json.dumps(exported))}).status_code == 200
    again = c2.get("/v1/tasks?full=1").json()["tasks"][0]
    assert again["certificate"] == exported[0]["certificate"] and again["monitor"] == exported[0]["monitor"]
    assert again["fit_sha256"] == exported[0]["fit_sha256"] == t["fit_sha256"]
    assert c2.delete("/v1/tasks/routing").status_code == 200 and "routing" not in so2.task_store.monitors
    bad = json.loads(json.dumps(exported))
    bad[0]["certificate"]["format"] = "decisio-nothing/1"
    assert make()[0].post("/v1/tasks/import", json={"tasks": bad}).status_code == 422


def test_c5_the_stack_names_what_moves_an_answer():
    from decisio.serve import stack

    client, so = make()
    f = so.stack_facts()
    assert set(f) == {"decisio", "base", "repository", "engine", "prompt", "calibration", "serving", "task_fingerprint"}
    assert (
        f["serving"]["orders"] == 1
        and f["serving"]["noul_commit"] is False
        and f["calibration"]["temperatures"] == {"global": 1.0}
    )
    assert f["engine"] == {
        "engine": "fake",
        "vllm": "1.0",
        "base": "fake-base",
        "quantization": None,
        "mode": "separate",
    }
    other = SystemOne(FakeEngine("1.0"), "x", orders=2, task_store=TaskStore("fp"), temperature=0.8)
    assert sorted(stack.differences(f, other.stack_facts())) == ["calibration.temperatures.global", "serving.orders"]


def test_c5_another_context_length_suspends_the_certificate():
    class E(FakeEngine):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def facts(self):
            return {**super().facts(), "max_model_len": self.n}

    client, so = make(engine=E(98304))
    post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {}})
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    same, _ = make(engine=E(98304))
    assert same.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    assert ask(same, 1, 0.99)["certified"] is True
    shorter, _ = make(engine=E(81920))  # a card whose fit is shorter, or a server started with --max-model-len
    assert shorter.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    a = ask(shorter, 1, 0.99)
    assert a["certified"] is False and a["certified_reason"] == "serving_stack_changed" and a["choice"] == "option_1"
    assert shorter.get("/v1/tasks/routing").json()["status"]["state"] == "suspended"


def test_c5_a_certified_task_fitted_under_another_fingerprint_is_not_applied_and_not_certified():
    client, so = make()
    post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {}})
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    c2, so2 = make()
    so2.task_store.fingerprint = "another model"  # as a server started with --tasks-file on another model
    so2.task_store.load(json.loads(json.dumps(exported)))
    a = ask(c2, 1, 0.99)
    assert a["certified"] is False and a["certified_reason"] == "serving_stack_changed" and a["choice"] == "option_1"
    assert c2.get("/v1/tasks/routing").json()["status"]["state"] == "suspended"


def test_c5_a_checkpoint_path_is_named_by_its_last_component():
    from decisio.serve.stack import stack_facts

    class E(FakeEngine):
        def facts(self):
            return {
                **super().facts(),
                "model": "/Users/someone/.cache/hub/models--x/snapshots/abc123",
                "tokenizer": "org/name",
            }

    f = stack_facts(E(), decisio_version="1", serving={}, temperatures={}, task_fingerprint=None)
    assert f["engine"]["model"] == "abc123" and f["engine"]["tokenizer"] == "org/name"
