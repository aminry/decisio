# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Label review in the registration response (decisio.readout.label_review; docs/tasks.md, "Label review").

L1  parity: on the stored BANKING77 readouts with planted wrong labels, the out-of-fold distributions and the flagged
    set equal the reference procedure's (5 folds of one permutation; the pipeline: head where kept, else calibration
    fitted on top of the temperature where kept, else the question at its temperature), bit for bit
L2  no flags when every out-of-fold answer is certain and right
L3  the field: present with an empty list when nothing is flagged; with a reason and no flags below 20 examples, for a
    score question, for a choice question of two options (a yes/no one is reviewed) and on every base but Gemma 4 12B
    (every format, the base and its tuned repositories)
L4  determinism: the same request gives the same review
L5  over HTTP, a wrong label is flagged at its place in the request, in the fit half or the held-out half of a
    certified task alike; the labels are never changed

  uv run pytest -q tests/unit/test_label_review.py
"""

import types
from pathlib import Path

import numpy as np
import pytest

from decisio.readout import calibration, intent_head, label_review
from decisio.readout.debias import log_probs
from decisio.serve.temperature import apply_temperature

READOUTS = Path(__file__).resolve().parents[1] / "data" / "intent_readouts"
T = 3.592


def _reference(lp, h, y, keys, served, g):
    """The reference procedure, as the study that chose the rule ran it: out-of-fold distributions from 5 folds of one
    permutation; per fold, the calibration fitted on log p / T, a head for 10+ options with MIN_PER_OPTION of each,
    served by the head when kept, else the calibration when kept, else the stored untaught answer."""
    K = len(keys)
    perm = g.permutation(len(y))
    parts = np.array_split(perm, min(5, len(y)))
    P = np.zeros((len(y), K))
    for k, te in enumerate(parts):
        tr = np.concatenate([parts[j] for j in range(len(parts)) if j != k])
        cal = calibration.fit_task_prior([log_probs(np.exp(lp[i])) / T for i in tr], y[tr].tolist(), K=K)
        head = None
        if K >= 10 and np.bincount(y[tr], minlength=K).min() >= intent_head.MIN_PER_OPTION:
            head = intent_head.fit_intent_head([lp[i] for i in tr], [h[i] for i in tr], y[tr].tolist(), keys)
        if head is not None and head.get("applied"):
            P[te] = [intent_head.apply_intent_head(lp[i], h[i], head, keys) for i in te]
        elif cal.get("applied"):
            P[te] = [calibration.apply_task_prior(log_probs(np.exp(lp[i])) / T, cal) for i in te]
        else:
            P[te] = served[te]
    rows = np.arange(len(y))
    Q = P.copy()
    Q[rows, y] = -1
    flagged = (P[rows, y] <= 0.02) & (P[rows, Q.argmax(1)] >= 0.9)
    return P, flagged


@pytest.mark.slow
def test_l1_parity_with_the_reference_procedure():
    d = np.load(READOUTS / "banking77_examples.npz")
    rng = np.random.default_rng(7)
    pick = rng.permutation(len(d["y"]))  # all 770 (10 per intent): every fold fits a head
    lp, h, y = d["lp"][pick], d["h"][pick], d["y"][pick].copy()
    K = lp.shape[1]
    keys = [f"i{k:02d}" for k in range(K)]
    planted = rng.random(len(y)) < 0.10
    y[planted] = (y[planted] + rng.integers(1, K, planted.sum())) % K
    lps = [log_probs(np.exp(x)) for x in lp]
    served = np.asarray([apply_temperature(np.exp(x), T) for x in lp])
    readout = [(lp[i], h[i]) for i in range(len(y))]
    for seed in (0, 1):
        P_ref, f_ref = _reference(lp, h, y, keys, served, np.random.default_rng(seed))
        parts = label_review.folds(len(y), np.random.default_rng(seed))
        P = label_review.out_of_fold(lps, y.tolist(), keys, T, readout, parts, 10)
        assert np.array_equal(P, P_ref)
        rec = label_review.review(lps, y.tolist(), keys, T, readout, "t", rng=np.random.default_rng(seed))
        assert sorted(f["index"] for f in rec["flags"]) == np.flatnonzero(f_ref).tolist()
        assert rec["reviewed"] and rec["examples"] == len(y) and rec["folds"] == 5
    assert f_ref.any() and (f_ref & planted).any()  # the check has something to compare


G12 = types.SimpleNamespace(key="gemma-4-12b")


def _server(answer_fn, family=G12):
    from fastapi.testclient import TestClient

    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore
    from decisio.serve.vllm_engine import make_app

    engine = types.SimpleNamespace(adapters={}, answer=answer_fn, family=family)
    so = SystemOne(engine, "test", task_store=TaskStore("fp"))
    return TestClient(make_app(engine, so))


KEYS = ["billing", "access", "sales", "other"]
Q = {"type": "choice", "instructions": "Which team?", "criteria": {k: None for k in KEYS}}


def _sure(state, questions, adapter=None):
    """0.97 on the team the state names, 0.01 on each other."""
    p = np.full(len(KEYS), 0.01)
    p[KEYS.index(state.split()[0])] = 0.97
    return [p for _ in questions], {}


def _examples(n, wrong=()):
    out = []
    for i in range(n):
        team = KEYS[i % len(KEYS)]
        label = KEYS[(i + 1) % len(KEYS)] if i in wrong else team
        out.append({"request": {"state": f"{team} message {i}", "questions": {"q": Q}}, "answer": label})
    return out


@pytest.mark.parametrize("key", ["gemma-4-12b", "gemma-4-12b-fp8"])
def test_l2_l3_no_flags_when_every_answer_is_certain_and_right(key):
    client = _server(_sure, family=types.SimpleNamespace(key=key))  # every format of the 12B
    r = client.post("/v1/tasks", json={"id": "clean", "examples": _examples(24)})
    assert r.status_code == 200, r.text
    rev = r.json()["label_review"]
    assert rev["reviewed"] is True and rev["examples"] == 24 and rev["flags"] == []


def test_l3_too_few_a_score_question_and_an_unchecked_base():
    client = _server(_sure)
    rev = client.post("/v1/tasks", json={"id": "few", "examples": _examples(19)}).json()["label_review"]
    assert rev["reviewed"] is False and "at least 20" in rev["reason"] and "flags" not in rev
    for key in ("gemma-4-31b", "gemma-4-31b-bf16", "qwen3.6-35b-a3b", "qwen3.5-4b"):  # not checked
        other = _server(_sure, family=types.SimpleNamespace(key=key))
        rev = other.post("/v1/tasks", json={"id": "b", "examples": _examples(24)}).json()["label_review"]
        assert rev["reviewed"] is False and f"not available on {key}" in rev["reason"]
    sq = {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]}

    def score_answer(state, questions, adapter=None):
        return [np.array([0.2, 0.3, 0.5]) for _ in questions], {}

    ex = [{"request": {"state": f"m {i}", "questions": {"q": sq}}, "answer": str(i % 3)} for i in range(24)]
    r = _server(score_answer).post("/v1/tasks", json={"id": "s", "examples": ex})
    assert r.status_code == 200, r.text
    assert r.json()["label_review"] == {"reviewed": False, "reason": "score questions are not reviewed"}
    # a choice question of two options is not reviewed (too many correct labels were flagged there); yes/no is
    two = [{"request": {"state": f"m {i}", "questions": {"q": {"type": "choice", "instructions": "Which?",
                                                              "criteria": {"a": None, "b": None}}}},
            "answer": "ab"[i % 2]} for i in range(24)]  # fmt: skip

    def two_answer(state, questions, adapter=None):
        return [np.array([0.9, 0.1]) for _ in questions], {}

    r = _server(two_answer).post("/v1/tasks", json={"id": "two", "examples": two})
    assert r.status_code == 200, r.text
    assert r.json()["label_review"] == {"reviewed": False, "reason": label_review.TWO_OPTION_REASON}


def test_l4_l5_a_wrong_label_is_flagged_at_its_place_and_kept():
    client = _server(_sure)
    body = {"id": "noisy", "examples": _examples(40, wrong={5, 30})}
    a = client.post("/v1/tasks", json=body).json()
    b = client.post("/v1/tasks", json=body).json()
    assert a["label_review"] == b["label_review"]  # L4
    flags = a["label_review"]["flags"]
    assert [(f["list"], f["index"]) for f in flags] == [("examples", 5), ("examples", 30)]
    assert flags[0]["label"] == "sales" and flags[0]["suggested"] == "access"
    assert flags[0]["p_label"] <= 0.02 and flags[0]["p_suggested"] >= 0.9
    assert a["n_examples"] == 40  # every example fitted as given

    # a certified task: the review covers the held-out half and the supplied list too, at their places
    body = {
        "id": "cert",
        "examples": _examples(40, wrong={3}),
        "certify": {"alpha": 0.1, "delta": 0.1, "examples": _examples(32, wrong={7})[0:32]},
    }
    for e in body["certify"]["examples"]:
        e["request"]["state"] = e["request"]["state"].replace("message", "held")
    r = client.post("/v1/tasks", json=body)
    assert r.status_code == 200, r.text
    rev = r.json()["label_review"]
    assert rev["examples"] == 72
    assert [(f["list"], f["index"]) for f in rev["flags"]] == [("examples", 3), ("certify.examples", 7)]
