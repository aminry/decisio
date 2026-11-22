# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The label monitor (decisio.serve.labels: `detector_states`, `TaskLabels.drift`): an e-detector over the audit and
cert labels of a certificate's certified answers, which suspends certification when their error rises above alpha.

M1  the statistic and its alert and clear states equal the reference implementation the detection and
    false-suspension figures were measured with, label by label, on random sequences
M2  under the null (errors at exactly alpha) it alerts at most about once per h labels
M3  a rise from alpha to 0.30 is detected after the change and never before it
M4  the pooling rule: audit and cert rows are read together only when every certified answer had the same chance
    of being drawn; across a change of chances, the audit rows alone, and the status says which
M5  the strict rule: a drawn answer that expires unlabelled counts as an error at its due time
M6  through the server: an alert makes answers certified false with certified_reason label_drift, the task's status
    says label_drift, and a new certificate starts the monitor again
M7  the monitor's claims are the reviewed sentences, each only in its tested scope (the share of answers drawn, the
    alpha and the split)

  uv run pytest -q tests/unit/test_labels_monitor.py
"""

import numpy as np
import pytest

from decisio.serve import labels as LB
from decisio.serve.fast_table_learn import MemoryPlace

KEYS = ["billing", "tech", "sales"]


def reference_alerting(times, err, alpha, h, T):
    """The reference implementation, verbatim in its arithmetic: the alert state by answer time."""
    U_GRID = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
    lam = U_GRID / alpha
    R = np.zeros(len(lam))
    state = np.zeros(T + 1, bool)
    on = False
    change = np.zeros(T + 1, np.int8)
    order = np.argsort(times, kind="stable")
    for t, e in zip(times[order], err[order]):
        R = (1.0 + R) * (1.0 + lam * (float(e) - alpha))
        m = float(R.mean())
        if not on and m >= h:
            on = True
            change[min(int(t), T)] += 1
        elif on and m < h / 2:
            on = False
            change[min(int(t), T)] -= 1
    cur = False
    for t in range(T):
        if change[t] > 0:
            cur = True
        elif change[t] < 0:
            cur = False
        state[t] = cur
    return state[:T]


def test_m1_equal_to_the_reference():
    g = np.random.default_rng(0)
    for trial in range(20):
        alpha = float(g.choice([0.005, 0.01, 0.05, 0.1]))
        n = int(g.integers(200, 3000))
        rise = int(g.integers(0, n))
        err = np.concatenate([g.random(rise) < alpha, g.random(n - rise) < float(g.uniform(alpha, 0.5))]).astype(float)
        h = float(g.choice([20.0, 200.0, 1500.0]))
        ours = [on for _, on in LB.detector_states(err, alpha, h)]
        with np.errstate(over="ignore"):  # the reference overflows to infinity on long runs of errors, as ours does
            ref = reference_alerting(np.arange(n, dtype=float), err, alpha, h, n)
        assert list(ref) == ours, trial
    assert LB.detector_threshold(0.05) == 10 * 5000 * 0.05


def test_m2_the_null_alerts_at_most_once_per_h_labels():
    alpha, h = 0.05, 200.0
    err = (np.random.default_rng(3).random(400_000) < alpha).astype(float)
    on = np.array([x for _, x in LB.detector_states(err, alpha, h)])
    onsets = int((on & ~np.concatenate([[False], on[:-1]])).sum())
    assert onsets <= 1.2 * len(err) / h + 5


def test_m3_a_rise_is_detected_after_the_change_only():
    g = np.random.default_rng(4)
    err = np.concatenate([g.random(1000) < 0.05, g.random(400) < 0.3]).astype(float)
    on = [x for _, x in LB.detector_states(err, 0.05, 1500.0)]
    assert not any(on[:1000]) and any(on[1000:])


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _log(clock, **settings):
    s = LB.parse_settings({"audit_pi": 0.5, "cert_pi": 0.5, "margin_pi_max": 0, "tau_after": 10_000, **settings})
    tl = LB.TaskLabels("t", s, LB.new_secrets(), MemoryPlace(), "r1", clock)
    tl.cert_id, tl.cert_alpha, tl.cert_threshold = "t#1", 0.05, 0.6
    return tl


def _serve(tl, i, conf=0.9, cert="t#1"):
    rid = LB.request_id("r1", tl.clock())
    tl.observe(rid, "q", text=f"t{i}", keys=KEYS, served=[conf, (1 - conf) / 2, (1 - conf) / 2], lp=None, h=None,
               certified=True, version="v1", stack="fp", lang="en", certificate=cert)  # fmt: skip
    return rid


def test_m4_the_pooling_rule():
    clock = Clock()
    tl = _log(clock)
    rids = [_serve(tl, i) for i in range(40)]
    tl.post([{"request_id": r, "answer": "billing"} for r in rids])
    d = tl.drift()
    drawn = sum(
        1 for r in rids if tl.rows[(r, "q")]["draws"]["audit"]["in"] or tl.rows[(r, "q")]["draws"]["cert"]["in"]
    )
    assert d["pooling"] == "pooled" and d["labels"] == drawn and d["h"] == pytest.approx(LB.detector_threshold(0.75))
    tl.set_settings({"cert_pi": 0.2})  # a change of chances: the region's rows no longer share one
    more = [_serve(tl, 100 + i) for i in range(40)]
    tl.post([{"request_id": r, "answer": "billing"} for r in more])
    d = tl.drift()
    audit = sum(1 for r in rids + more if tl.rows[(r, "q")]["draws"]["audit"]["in"])
    assert d["pooling"] == "audit" and d["labels"] == audit
    assert tl.overview()["label_monitor"]["pooling"] == "audit"
    other = _serve(tl, 999, cert="t#2")  # another certificate's answer is not this certificate's
    tl.post([{"request_id": other, "answer": "tech"}])
    assert tl.drift()["labels"] == audit


def test_m5_an_expired_row_counts_as_an_error_at_its_due_time():
    clock = Clock()
    tl = _log(clock, cert_pi=0, audit_pi=1.0)
    rids = [_serve(tl, i) for i in range(5)]
    tl.post([{"request_id": r, "answer": "billing"} for r in rids[:4]])  # four right, one pending
    h = tl.drift()["h"]
    assert tl.drift()["labels"] == 4
    assert tl.drift()["statistic"] == pytest.approx(LB.detector_states([0, 0, 0, 0], 0.05, h)[-1][0], abs=1e-4)
    clock.t += 15 * 86400  # past due: the fifth is an error
    d = tl.drift()
    assert d["labels"] == 5
    s4 = LB.detector_states([0, 0, 0, 0], 0.05, d["h"])[-1][0]
    s5 = LB.detector_states([0, 0, 0, 0, 1], 0.05, d["h"])[-1][0]
    assert d["statistic"] == pytest.approx(s5, abs=1e-4) and s5 > s4


def test_m6_through_the_server():
    from test_labels_served import ask, make, register

    client, so = make()
    register(client, labels={"audit_pi": 1.0, "cert_pi": 1.0, "margin_pi_max": 0}, n=600)  # enough to certify
    t = so.task_store.snapshot("angry")
    cert = t["certificate"]
    assert cert["threshold"] is not None
    rids = []
    for i in range(60):
        a, _ = ask(client, f"yes:0.97:wrong {i}")
        assert a["certified"] is True
        rids.append(a["request_id"])
    log = so.labels_logs["angry"]
    client.post("/v1/tasks/angry/labels", json={"labels": [{"request_id": r, "answer": "no"} for r in rids]})
    so.flush_labels(force=True)  # the labels thread's turn: every certified answer was wrong
    assert log.drift_state["alerting"] is True
    a, _ = ask(client, "yes:0.97:after")
    assert a["certified"] is False and a["certified_reason"] == "label_drift"
    st = client.get("/v1/tasks/angry").json()["status"]
    assert st["state"] == "label_drift" and st["label_monitor"]["alerting"] is True
    log.drift_state = {**log.drift_state, "certificate": "angry#other"}  # a new certificate starts it again
    a, _ = ask(client, "yes:0.97:new")
    assert a["certified"] is True


def test_m7_claims_in_their_tested_scope():
    clock = Clock()
    tl = _log(clock, audit_pi=0.02, cert_pi=0.02)  # explicit chances: no comparison claim
    for i in range(300):
        clock.t += 60
        _serve(tl, i)
    d = tl.drift()
    assert d["label_share"] >= 0.03 and d["claims"] == [LB.FALSE_SUSPENSION_CLAIM]
    lenient = LB.TaskLabels("t", LB.parse_settings({"per_day": 400}), LB.new_secrets(), MemoryPlace(), "r1", clock)
    lenient.cert_id, lenient.cert_alpha, lenient.cert_threshold = "t#1", 0.05, 0.6
    for i in range(300):
        clock.t += 60
        _serve(lenient, i)
    assert LB.COMPARISON_CLAIM in lenient.drift()["claims"]
    lenient.cert_alpha = 0.01  # a strict alpha: its numerical claim is detection's, not the comparison
    assert LB.COMPARISON_CLAIM not in lenient.drift()["claims"]
