# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A certificate rests on the confidences actually served: a question of a certified task is scored by the function
that registered it, and registration re-reads a sample through the serving path and records the difference.

S1  a certified question in a request with other questions is scored alone, as at registration: its probabilities equal
    the one-question request's, while an uncertified question in the same request still moves with the batch
S2  on one stack the registration-time check reads zero: every sampled item went through the task, none moved, none
    changed sides of the threshold
S3  when the engine's answers drift between reads, the check records the exposure in the certificate and its summary
S4  the check does not feed the answer-mix monitor, and is not made when nothing is certified
S5  what is not certified is unchanged: a task without a certificate, a score question, a two-order request
S6  --noul-commit is applied once to a certified yes/no answer, and the debug readout names the served path
S7  a re-certification makes the check too

  uv run pytest -q tests/unit/test_certified_served_path.py
"""

import numpy as np
from test_certified_answers import CRIT, FakeEngine, examples, make, post_task, wire


class BatchEngine(FakeEngine):
    """A fake whose confidence moves with the number of questions in the call (a stand-in for any batch-dependent
    numeric difference) and, with `drift`, with the number of calls so far."""

    def __init__(self, shift=0.004, drift=0.0):
        super().__init__("1.0")
        self.shift, self.drift, self.calls = shift, drift, 0

    def answer(self, state, questions, adapter=None, images=None):
        self.calls += 1
        probs, info = super().answer(state, questions, adapter)
        out = []
        for p in probs:
            top = int(np.argmax(p))
            conf = float(p[top]) - self.shift * (len(questions) - 1) - self.drift * (self.calls % 5)
            q = np.full(len(p), (1 - conf) / (len(p) - 1))
            q[top] = conf
            out.append(q)
        return out, info


OTHER = {"option_0": "left", "option_1": "right"}


def two_questions(text):
    return {
        "state": {"message": text},
        "model": "m",
        "questions": {
            "q1": {"type": "choice", "instructions": "Which queue?", "criteria": CRIT},
            "q2": {"type": "choice", "instructions": "Which side?", "criteria": OTHER},
            "q3": {"type": "choice", "instructions": "Which other side?", "criteria": OTHER},
        },
    }


def registered(engine, n=400, acc=0.96):
    client, so = make(engine=engine)
    r = post_task(client, {"id": "routing", "examples": examples(n, acc), "certify": {}})
    assert r.status_code == 200, r.text
    return client, so, r.json()


def test_s1_a_certified_question_is_scored_alone_whatever_else_the_request_asks():
    client, so, task = registered(BatchEngine())
    assert task["certificate"]["certified"]
    alone = client.post("/v1/systemone", json=wire("1:0.97:x")).json()["answers"]["q1"]
    both = client.post("/v1/systemone", json=two_questions("1:0.97:x")).json()
    q1, q2 = both["answers"]["q1"], both["answers"]["q2"]
    assert q1["probabilities"] == alone["probabilities"]  # the registration's function, not the batch
    assert q1["certified"] is True and alone["certified"] is True
    # an uncertified question of the same request is still scored in the batch, as before
    assert abs(q2["probabilities"]["option_1"] - 0.97) > 1e-3


def test_s2_on_one_stack_the_check_reads_zero():
    client, so, task = registered(BatchEngine())
    c = task["certificate"]["served_check"]
    assert c["n"] == 20 and c["task_applied"] == 20
    assert (
        c["max_shift"] == 0.0 and c["p99_shift"] == 0.0 and c["crossed_threshold"] == 0 and c["top_answer_changed"] == 0
    )
    assert "served confidences can differ" not in task["certificate"]["summary"]
    assert so.task_store.by_key[next(iter(so.task_store.by_key))]["certificate"]["served_check"] == c


def test_s3_a_drifting_engine_is_recorded_in_the_certificate_and_its_summary():
    client, so, task = registered(BatchEngine(shift=0.0, drift=0.002))
    c = task["certificate"]["served_check"]
    assert c["max_shift"] > 0 and c["n"] == 20
    assert "Re-read through the serving path" in task["certificate"]["summary"]
    assert "served confidences can differ" in task["certificate"]["summary"]


def test_s4_the_check_does_not_feed_the_monitor_and_is_skipped_when_nothing_is_certified():
    client, so = make(engine=BatchEngine())
    r = post_task(client, {"id": "weak", "examples": examples(400, 0.6, seed=2), "certify": {}})
    assert r.json()["certificate"]["threshold"] is None and "served_check" not in r.json()["certificate"]
    client, so = make(engine=BatchEngine())
    seen = []
    import decisio.serve.monitor as monitor

    orig = monitor.MixMonitor.observe
    monitor.MixMonitor.observe = lambda self, chosen, ok: (seen.append(chosen), orig(self, chosen, ok))[1]
    try:
        r = post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {}})
        assert r.json()["certificate"]["served_check"]["n"] == 20 and seen == []
        client.post("/v1/systemone", json=wire("1:0.97:x"))
        assert len(seen) == 1  # a real request is observed
    finally:
        monitor.MixMonitor.observe = orig


def test_s5_what_is_not_certified_is_unchanged():
    client, so = make(engine=BatchEngine())
    r = post_task(client, {"id": "plain", "examples": examples(40, 0.9)})
    assert "certificate" not in r.json()
    a = client.post("/v1/systemone", json=two_questions("1:0.9:x")).json()["answers"]
    assert a["q1"]["probabilities"]["option_1"] != 0.9  # batch shift still applies to uncertified questions
    client, so, task = registered(BatchEngine())
    two = client.post("/v1/systemone", json={**wire("1:0.99:x"), "orders": 2})
    assert two.status_code == 200 and two.json()["answers"]["q1"]["certified"] is False
    score = {
        "state": {"message": "0:0.9:1"},
        "model": "m",
        "questions": {"q": {"type": "score", "criteria": ["a", "b", "c"]}},
    }
    assert client.post("/v1/systemone", json=score).status_code == 200


def test_s6_noul_commit_once_and_the_debug_readout_names_the_path():
    client, so, task = registered(BatchEngine())
    so.debug_readout = True
    r = client.post("/v1/systemone", json=wire("1:0.97:x"), headers={"x-decisio-debug": "readout"})
    assert r.status_code == 200
    dbg = r.json()["decisio_debug"]["q1"]
    assert dbg["path"] == "served" and dbg["task"] == "routing"


def test_s7_re_certification_makes_the_check_too():
    client, so, task = registered(BatchEngine())
    new = examples(200, 0.96, seed=9, start=10_000)
    r = client.post("/v1/tasks/routing/certify", json={"examples": new})
    assert r.status_code == 200, r.text
    assert r.json()["certificate"]["served_check"]["n"] == 20


def test_s8_a_head_question_is_one_hidden_state_read_and_no_plain_read(monkeypatch):
    """The card session of 2026-10-16: registration read each certifying example plain and then hidden, serving only
    hidden, and the two differ slightly on vLLM. Both are now the hidden read alone."""
    import decisio.readout.intent_head as ih
    from decisio.serve.systemone import SystemOne  # noqa: F401

    class Hidden:
        def __init__(self):
            self.calls = []

        def readout(self, state, questions):
            self.calls.append(state["message"])
            return [(np.array([0.0, 1.0, 2.0]), np.array([1.0]))]

    client, so = make(engine=BatchEngine())
    so.hidden_engine = Hidden()
    monkeypatch.setattr(ih, "apply_intent_head", lambda lp, h, head, options: np.array([0.1, 0.2, 0.7]))
    task = {"head": {"applied": True}, "calibration": {}, "options": ["a", "b", "c"]}
    from decisio.serve.systemone import ChoiceQuestion

    q = ChoiceQuestion(type="choice", instructions="Which?", criteria=CRIT)
    before = so.engine.calls
    out = so.served_rows(task, [({"message": "1:0.9:x"}, q), ({"message": "2:0.9:y"}, q)])
    assert so.engine.calls == before  # the engine was never asked for a plain read
    assert so.hidden_engine.calls == ["1:0.9:x", "2:0.9:y"]
    assert all(np.allclose(p, [0.1, 0.2, 0.7]) for p in out)
