# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Per-task calibration and the intent head on `/v1/systemone` (`decisio.serve.tasks`, `POST /v1/tasks`).

T1  a task's identity: the question type and its option list in order (plus the instructions for yes/no and score);
    any change to the list is another task; a question whose rendering the --describe-options rule changes has
    another key with the rule on (so a task fitted under the other rendering is not applied), and every other
    question the same key either way;
T2  the store: calibration is `calibration.fit_task_prior` on the examples' served log-scores, unchanged; the head is
    `intent_head.fit_intent_head` on their hidden readout, fitted only for 10 or more options with at least 5 examples
    per option (else off, with the reason); a re-registration replaces the task; the JSON form round-trips the head's
    arrays exactly;
T3  served (CPU stand-in): no task, or a task of another list, changes nothing, bit for bit; calibration is
    `apply_task_prior` on the served readout and the head `apply_intent_head` on the hidden readout, exactly; the head
    is refused with two orders, calibration is applied per branch in each branch's order; the abstention threshold
    decides on the calibrated distribution; the debug header needs --debug-readout.

  uv run pytest -q tests/unit/test_tasks.py
"""

import numpy as np
import pytest

from decisio.readout import calibration, intent_head
from decisio.readout.debias import log_probs
from decisio.serve.tasks import TaskStore, answer_keys, gold_index, task_key

MODEL = "Qwen/Qwen3-0.6B-Base"
CRIT = {"option_0": "card arrival", "option_1": "exchange rate", "option_2": "top up"}
TOPICS = [
    "card arrival",
    "exchange rate",
    "top up",
    "lost card",
    "pin change",
    "refund",
    "transfer fee",
    "cash withdrawal",
    "account closure",
    "direct debit",
]
CRIT10 = {f"option_{i}": t for i, t in enumerate(TOPICS)}


def wire(utt, crit=CRIT, **extra):
    return {
        "state": {"message": utt},
        "model": "m",
        "questions": {
            "q1": {"type": "choice", "instructions": "Classify the intent of the user's message.", "criteria": crit}
        },
        **extra,
    }


def question(body):
    from decisio.serve.systemone import SystemOneRequest

    return SystemOneRequest.model_validate(body).questions["q1"]


def test_t1_task_identity():
    from decisio.serve.systemone import render_text

    k = task_key(question(wire("a")), render_text)
    assert k == task_key(question(wire("another message")), render_text)  # the state is not the task
    reordered = dict(reversed(list(CRIT.items())))
    renamed = {**CRIT, "option_2": "top up my account"}
    for crit in (reordered, renamed, {**CRIT, "option_3": "other"}):
        assert task_key(question(wire("a", crit)), render_text) != k
    noul = {"state": {}, "questions": {"q1": {"type": "noul", "instructions": "Is it spam?"}}}
    noul2 = {"state": {}, "questions": {"q1": {"type": "noul", "instructions": "Is it toxic?"}}}
    assert task_key(question(noul), render_text) != task_key(question(noul2), render_text)
    assert answer_keys(question(noul)) == ["yes", "no"] and gold_index(question(noul), False) == 1
    with pytest.raises(ValueError):
        gold_index(question(wire("a")), "option_9")


def test_t1_task_key_follows_the_rendering():
    from decisio.serve.systemone import SystemOne, option_set, render_text

    def server(describe_options):
        return SystemOne(engine=None, served_name="m", describe_options=describe_options)

    on, off = server(True), server(False)
    described = question(wire("a", {"CLICK": "Click the target", "DONE": "The goal is complete"}))
    assert on.described(described) and not off.described(described)
    assert task_key(described, render_text, on.described(described)) != task_key(described, render_text)
    assert option_set(described, on.described(described)) != option_set(described)
    # index keys (hidden either way), keys without descriptions, yes/no: the rule changes nothing, nor the key
    for body in (
        wire("a"),
        wire("a", {"billing": None, "access": None}),
        {"state": {}, "questions": {"q1": {"type": "noul", "instructions": "Is it spam?"}}},
    ):
        q = question(body)
        assert not on.described(q) and task_key(q, render_text, on.described(q)) == task_key(q, render_text)


def fake_examples(n_per, K=10, seed=0):
    """(question, state, gold) with a readout that favours slot 0 whatever the gold, and hidden states that carry it."""
    rng = np.random.default_rng(seed)
    q = question(wire("x", CRIT10 if K == 10 else CRIT))
    ex, P, R = [], [], []
    for i in range(n_per * K):
        g = i % K
        z = rng.normal(0, 0.3, K)
        z[0] += 1.0
        z[g] += 0.8
        p = np.exp(z) / np.exp(z).sum()
        h = rng.normal(0, 1, 16).astype(np.float32)
        h[g] += 3.0
        ex.append((q, {"i": i}, f"option_{g}"))
        P.append(p)
        R.append((np.log(p), h))
    return ex, P, R


def test_t2_store_fits_with_the_reference():
    ex, P, R = fake_examples(6)
    store = TaskStore("fp")
    task, fitted = store.register("t", ex, lambda pairs: P, lambda pairs: R)
    labels = [i % 10 for i in range(60)]
    ref = calibration.fit_task_prior([log_probs(p) for p in P], labels, fingerprint="fp", task_id="t", K=10)
    assert task["calibration"] == ref and fitted["labels"] == labels
    head = intent_head.fit_intent_head(
        [lp for lp, _ in R], [h for _, h in R], labels, answer_keys(ex[0][0]), fingerprint="fp", task_id="t"
    )
    assert task["head"]["applied"] and head["applied"]
    assert np.array_equal(task["head"]["A"], head["A"]) and np.array_equal(task["head"]["c"], head["c"])
    assert store.lookup(task["key"]) is task and TaskStore("other").lookup(task["key"]) is None
    # JSON round trip, arrays exact
    other = TaskStore("fp")
    import json

    other.load(json.loads(json.dumps([TaskStore.public(task, full=True)])))
    t2 = other.lookup(task["key"])
    assert np.array_equal(t2["head"]["A"], task["head"]["A"]) and np.array_equal(t2["head"]["c"], task["head"]["c"])
    assert TaskStore.public(task)["head"]["A"] == {"shape": [16, 10]}
    # fewer than 5 per option: no head, the reason stored; without a hidden engine likewise
    ex4, P4, R4 = fake_examples(4)
    t4, _ = store.register("t", ex4, lambda pairs: P4, lambda pairs: R4)
    assert not t4["head"]["applied"] and "at least 5" in t4["head"]["reason"] and len(store.by_key) == 1
    t0, _ = store.register("t", ex, lambda pairs: P, None)
    assert not t0["head"]["applied"] and "--head-engine" in t0["head"]["reason"]
    # fewer than 10 options (a yes/no or multiple-choice question): calibration only, whatever the examples
    ex3, P3, R3 = fake_examples(8, K=3)
    t3, f3 = store.register("t3", ex3, lambda pairs: P3, lambda pairs: R3)
    assert not t3["head"]["applied"] and "at least 10 options" in t3["head"]["reason"] and f3["readout"] is None
    store.remove("t3")
    with pytest.raises(ValueError):  # one task per registration
        store.register("mixed", ex[:3] + [(question(wire("x", {"a": "b", "c": "d"})), {}, "a")], lambda p: P, None)
    assert store.remove("t") and not store.by_key


@pytest.fixture(scope="module")
def served():
    from fastapi.testclient import TestClient

    from decisio.serve.hf_letters import HFLettersEngine
    from decisio.serve.hidden_engine import HFHiddenEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.vllm_engine import make_app

    eng = HFLettersEngine(MODEL, pad_to="block", pad_where="front")
    hid = HFHiddenEngine(MODEL, pad_to="block", pad_where="front")
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"), hidden_engine=hid, debug_readout=True)
    return TestClient(make_app(eng, so)), so


UTTS = {
    "option_0": [
        "where is my card",
        "my card has not arrived",
        "card still not here",
        "when will my card come",
        "has my new card shipped",
    ],
    "option_1": [
        "what is the euro rate",
        "exchange rate for dollars",
        "how much is a pound in euros",
        "rate for yen today",
        "currency conversion rate",
    ],
    "option_2": [
        "add money to my account",
        "top up with my visa",
        "how do I top up",
        "put 50 on my card",
        "load funds into the app",
    ],
}
EXAMPLES = [{"request": wire(u), "answer": k} for k, us in UTTS.items() for u in us]
PHRASES = [
    "I need help with {}",
    "question about {}",
    "what should I do about {}",
    "tell me about {}",
    "can you explain {}",
]
EXAMPLES10 = [{"request": wire(p.format(t), CRIT10), "answer": k} for k, t in CRIT10.items() for p in PHRASES]


def post(client, body, **headers):
    r = client.post("/v1/systemone", json=body, headers=headers)
    assert r.status_code == 200, r.text
    return r


def test_t3_no_task_no_change(served):
    client, so = served
    body = wire("my card is late")
    base = post(client, body).json()
    assert "x-decisio-tasks" not in post(client, body).headers
    so.task_store.by_key.clear()
    r = client.post(
        "/v1/tasks",
        json={
            "id": "other",
            "examples": [
                {"request": wire(u, {"a": "billing", "b": "shipping"}), "answer": "a" if i % 2 else "b"}
                for i, u in enumerate(["x"] * 10 + ["y"] * 10)
            ],
        },
    )
    assert r.status_code == 200, r.text
    assert post(client, body).json() == base  # a task of another list: nothing
    so.tasks_enabled = False
    assert post(client, wire("x", {"a": "billing", "b": "shipping"})).headers.get("x-decisio-tasks") is None
    so.tasks_enabled = True
    so.task_store.by_key.clear()


def test_t3_calibration_served_exactly(served):
    client, so = served
    from decisio.serve.systemone import second_order, to_engine_question

    so.task_store.by_key.clear()
    r = client.post("/v1/tasks", json={"id": "cal", "examples": EXAMPLES})
    assert r.status_code == 200, r.text
    task = so.task_store.lookup(r.json()["key"])
    # 3 options: calibration only
    assert not task["head"]["applied"] and "at least 10 options" in task["head"]["reason"]
    # force a visible bias (the fit on 15 CPU examples may decline) and check the served arithmetic
    task["calibration"] = {**task["calibration"], "applied": True, "bias": [0.7, -0.2, -0.5]}
    task["head"] = {**task["head"], "applied": False}
    body = wire("I want to top up")
    r = post(client, body, **{"x-decisio-debug": "readout"})
    assert r.headers["x-decisio-tasks"] == "cal"
    d = r.json()["decisio_debug"]["q1"]
    assert d["path"] == "calibration"
    want = calibration.apply_task_prior(log_probs(np.array(d["p"])), task["calibration"])
    assert list(r.json()["answers"]["q1"]["probabilities"].values()) == want.tolist()
    # two orders: the bias per displayed slot in each branch, then the average
    r2 = post(client, {**body, "orders": 2}).json()["answers"]["q1"]
    q = question(body)
    eq, keys = to_engine_question(q, so.hide_index_keys, so.desnake_labels)
    q2, perm = second_order("q1", eq, keys)
    probs, _ = so.engine.answer(body["state"], [eq, q2])
    b1 = calibration.apply_task_prior(log_probs(np.asarray(probs[0], dtype=np.float64)), task["calibration"])
    shown = calibration.apply_task_prior(log_probs(np.asarray(probs[1], dtype=np.float64)), task["calibration"])
    b2 = np.zeros(3)
    for pos, orig in enumerate(perm):
        b2[orig] = shown[pos]
    assert list(r2["probabilities"].values()) == ((b1 + b2) / 2).tolist()
    # abstention composes after: the threshold decides on the calibrated distribution
    from decisio.serve.systemone import option_set

    so.tasks = {
        "abs": {
            "id": "abs",
            "option": {"key": "option_2"},
            "match": {"option_set": option_set(q)},
            "config": {"applied": True, "threshold": float(want[2]) - 1e-9},
        }
    }
    a = post(client, body).json()["answers"]["q1"]
    assert a["choice"] == "option_2" and list(a["probabilities"].values()) == want.tolist()
    so.tasks = {"abs": {**so.tasks["abs"], "config": {"applied": True, "threshold": float(want[2]) + 1e-9}}}
    a = post(client, body).json()["answers"]["q1"]
    assert a["choice"] == ["option_0", "option_1"][int(np.argmax(want[:2]))]
    so.tasks = {}
    so.task_store.by_key.clear()


def test_t3_head_served_exactly(served):
    client, so = served
    so.task_store.by_key.clear()
    r = client.post("/v1/tasks", json={"id": "intent", "examples": EXAMPLES10}, headers={"x-decisio-debug": "hidden"})
    assert r.status_code == 200, r.text
    out = r.json()
    dbg = out["decisio_debug"]
    labels = dbg["labels"]
    assert labels == [k for k in range(10) for _ in range(5)] and out["per_option_min"] == 5
    # the stored fit is the reference fit of the served readout, bit for bit
    lps = [np.array(lp) for lp, _ in dbg["readout"]]
    H = [np.array(h, dtype=np.float32) for _, h in dbg["readout"]]
    ref = intent_head.fit_intent_head(lps, H, labels, list(CRIT10), fingerprint="fp", task_id="intent")
    task = so.task_store.lookup(out["key"])
    assert task["head"]["applied"] == ref["applied"]
    assert task["calibration"] == calibration.fit_task_prior(
        [np.array(x) for x in dbg["lps"]], labels, fingerprint="fp", task_id="intent", K=10
    )
    if not ref["applied"]:
        pytest.skip("cross-validation declined the head on the CPU stand-in's 50 examples")
    assert np.array_equal(task["head"]["A"], ref["A"]) and np.array_equal(task["head"]["c"], ref["c"])
    body = wire("please top up my balance", CRIT10)
    r = post(client, body, **{"x-decisio-debug": "readout"})
    d = r.json()["decisio_debug"]["q1"]
    assert d["path"] == "head" and r.headers["x-decisio-tasks"] == "intent"
    want = intent_head.apply_intent_head(
        np.array(d["hidden_lp"]), np.array(d["h"], dtype=np.float32), task["head"], list(CRIT10)
    )
    assert list(r.json()["answers"]["q1"]["probabilities"].values()) == want.tolist()
    # refused with two orders; a changed list is another task (served plain)
    bad = client.post("/v1/systemone", json={**body, "orders": 2})
    assert bad.status_code == 400 and "two-order" in bad.json()["detail"]
    reordered = dict(reversed(list(CRIT10.items())))
    so.tasks_enabled = False
    plain = post(client, wire("please top up my balance", reordered)).json()
    so.tasks_enabled = True
    r = post(client, wire("please top up my balance", reordered))
    assert "x-decisio-tasks" not in r.headers and r.json() == plain
    listed = client.get("/v1/tasks").json()["tasks"]
    assert [t["id"] for t in listed] == ["intent"] and listed[0]["head"]["A"] == {"shape": list(ref["A"].shape)}
    full = client.get("/v1/tasks", params={"full": 1}).json()["tasks"]
    assert client.delete("/v1/tasks/intent").status_code == 200 and not so.task_store.by_key
    assert client.post("/v1/tasks/import", json={"tasks": listed}).status_code == 422  # no arrays
    assert client.post("/v1/tasks/import", json={"tasks": [{**full[0], "fingerprint": "x"}]}).status_code == 422
    assert client.post("/v1/tasks/import", json={"tasks": full}).status_code == 200
    again = post(client, body, **{"x-decisio-debug": "readout"}).json()["answers"]["q1"]["probabilities"]
    assert list(again.values()) == want.tolist()
    assert client.delete("/v1/tasks/intent").status_code == 200 and not so.task_store.by_key


def test_t3_debug_and_bad_registrations(served):
    client, so = served
    so.debug_readout = False
    assert client.post("/v1/systemone", json=wire("x"), headers={"x-decisio-debug": "readout"}).status_code == 422
    assert "decisio_debug" not in post(client, wire("x")).json()
    so.debug_readout = True
    two = {
        "state": {},
        "questions": {"a": {"type": "noul", "instructions": "?"}, "b": {"type": "noul", "instructions": "!"}},
    }
    assert client.post("/v1/tasks", json={"id": "x", "examples": [{"request": two, "answer": True}]}).status_code == 422
    assert (
        client.post(
            "/v1/tasks", json={"id": "x", "examples": [{"request": wire("a"), "answer": "option_7"}]}
        ).status_code
        == 422
    )
    assert client.post("/v1/tasks", json={"examples": []}).status_code == 422
    assert not so.task_store.by_key


def test_t2_certain_readout_registers():
    """A readout so confident that the fitted prior changes nothing (every fold's log-loss change exactly 0) leaves
    calibration off with a finite record that JSON can carry."""
    import json

    import numpy as np

    from decisio.readout import calibration

    K, n = 4, 20
    labels = [i % K for i in range(n)]
    lps = [np.log(np.clip(np.eye(K)[y], 1e-300, None)) for y in labels]  # p = 1 on the answer, 0 elsewhere
    rec = calibration.fit_task_prior(lps, labels, fingerprint="fp", task_id="t", K=K)
    assert rec["applied"] is False and rec["cv_t"] == 0.0
    json.dumps(rec, allow_nan=False)


def test_t2_certain_readout_over_http():
    """POST /v1/tasks on a readout certain on every example answers 200 (it answered 500: the record held +inf)."""
    import types

    import numpy as np
    from fastapi.testclient import TestClient

    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore
    from decisio.serve.vllm_engine import make_app

    keys = ["billing", "access", "sales", "other"]

    def answer(state, questions, adapter=None):  # probability 1 on the option the state names
        return [np.eye(len(keys))[keys.index(state)] for _ in questions], {}

    engine = types.SimpleNamespace(adapters={}, answer=answer)
    so = SystemOne(engine, "test", task_store=TaskStore("fp"))
    client = TestClient(make_app(engine, so))
    q = {"type": "choice", "instructions": "Which team?", "criteria": {k: None for k in keys}}
    examples = [{"request": {"state": k, "questions": {"q": q}}, "answer": k} for k in keys * 5]
    r = client.post("/v1/tasks", json={"id": "certain", "examples": examples})
    assert r.status_code == 200, r.text
    assert r.json()["calibration"]["applied"] is False and r.json()["calibration"]["cv_t"] == 0.0
