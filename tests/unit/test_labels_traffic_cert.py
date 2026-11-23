# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Certification from traffic labels (decisio.serve.labels: `traffic_threshold`, `t_floor`, the looks).

C1  parity with the frozen form the method was validated with: the floor is the frozen t_floor, the certificate is
    decisio's fixed-sequence arm at the full delta on the labelled certification answers, on a fixture
C2  a look: the cohort is the first n answers the cert slice drew above the floor; its certificate is made once,
    when every one of them is labelled, expired or declined, and equals C1's on the same answers
C3  looks fixed in advance: with looks 100 and 200, each runs at delta / 2, and nothing is made between them
C4  the asserts: answers below a raised floor are audit-only, answers served under another stack are left out
C5  missing labels at alpha 0.01: label response under 1 - alpha gives labels_missing and no certificate
C7  pooling across epochs needs containment: an earlier epoch pools only while its floor is at or below the
    current one (a falling floor leaves it out, a rising one takes it in)
C6  through the server: a task that opted into labels and asks for a certificate certifies from its traffic labels
    (cert region on the label-free score at c_max, rows keep it); the certificate from examples serves until the first
    look, which replaces it; certified answers carry the traffic certificate and the label-free readout; a task
    without labels keeps its certificate from examples, logs nothing and is told what it needs; bad looks and
    target shares are refused

  uv run pytest -q tests/unit/test_labels_traffic_cert.py
"""

import math

import numpy as np
import pytest

from decisio.readout import certify
from decisio.serve import labels as LB
from decisio.serve.fast_table_learn import MemoryPlace

KEYS = ["billing", "tech", "sales"]


def frozen_t_floor(U, c):
    """The frozen form's floor, verbatim."""
    s = np.sort(np.asarray(U, float))[::-1]
    return float(s[min(len(s) - 1, max(0, math.ceil(c * len(s)) - 1))])


def frozen_c1(conf, correct, alpha, delta):
    """The frozen form's certificate (C1), verbatim: decisio's fixed-sequence arm at the full delta."""
    if len(conf) == 0:
        return math.inf
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    o = np.argsort(-conf)
    return certify._fixed_sequence(conf[o], np.cumsum(~correct[o]), alpha, delta)


def test_c1_parity_with_the_frozen_form():
    g = np.random.default_rng(5)
    for trial in range(30):
        n = int(g.integers(50, 800))
        s = 0.3 + 0.7 * g.random(n)
        correct = g.random(n) > 0.15 * (1.2 - s)
        for alpha in (0.10, 0.05, 0.01):
            assert LB.traffic_threshold(s, correct, alpha, 0.1) == frozen_c1(s, correct, alpha, 0.1), (trial, alpha)
        U = g.random(int(g.integers(10, 5000)))
        for X in (0.10, 0.30, 0.50, None):
            c = LB.c_max_of(X)
            assert LB.t_floor(U, c) == frozen_t_floor(U, c)
    assert LB.c_max_of(None) == 0.45 and LB.c_max_of(0.3) == pytest.approx(0.45) and LB.c_max_of(0.8) == 1.0


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _log(clock, looks=(100,), alpha=0.1, tau_after=200, due_days=14):
    s = LB.parse_settings({"audit_pi": 0, "cert_pi": 1.0, "margin_pi_max": 0, "tau_after": tau_after,
                           "region": 0.45, "cert_score": "label_free", "due_days": due_days})  # fmt: skip
    tl = LB.TaskLabels("t", s, LB.new_secrets(), MemoryPlace(), "r1", clock)
    tl.traffic = {
        "looks": list(looks),
        "alpha": alpha,
        "delta": 0.1,
        "target_share": None,
        "c_max": 0.45,
        "stack": "fp",
    }
    if tau_after >= 10_000:  # a floor already set (at 0: every answer is above it), so the cohort starts at once
        tl.epochs = [*tl.epochs, {**tl.epochs[-1], "epoch": 1, "tau_min": 0.0, "tau_set": True}]
    return tl


def _serve(tl, s0, i, stack="fp"):
    clock = tl.clock
    clock.t += 1
    rid = LB.request_id("r1", clock())
    tl.observe(rid, "q", text=f"t{i}", keys=KEYS, served=[s0, (1 - s0) / 2, (1 - s0) / 2], lp=None, h=None,
               certified=True, version="v1", stack=stack, lang="en", s0=s0, s0_top="billing")  # fmt: skip
    return rid


def _traffic(tl, g, n, error=lambda s: 0.15 * (1.2 - s)):
    rows = []
    for i in range(n):
        s0 = float(0.34 + 0.66 * g.random())
        rows.append((_serve(tl, s0, i), s0, g.random() >= error(s0)))
    return rows


def test_c2_a_look_equals_the_frozen_certificate_on_its_cohort():
    clock = Clock()
    g = np.random.default_rng(1)
    tl = _log(clock, looks=(100,))
    rows = _traffic(tl, g, 200)  # the first tau_after answers: all traffic is the region, every answer drawn
    tl.tick()
    assert tl.epoch["tau_min"] == pytest.approx(frozen_t_floor([s for _, s, _ in rows], 0.45))
    later = _traffic(tl, g, 600)  # after the floor: the cert slice draws at or above it only
    cohort = [(rid, s, ok) for rid, s, ok in later if tl.rows[(rid, "q")]["draws"]["cert"]["in"]][:100]
    assert all(s >= tl.epoch["tau_min"] for _, s, _ in cohort)
    tl.post([{"request_id": rid, "answer": "billing" if ok else "tech"} for rid, _, ok in cohort[:99]])
    tl.tick()
    assert tl.traffic_in_force() is None and tl.traffic_status()["next_look"]["closed"] == 99  # one still pending
    tl.post([{"request_id": cohort[99][0], "answer": "billing" if cohort[99][2] else "tech"}])
    tl.tick()
    c = tl.traffic_in_force()
    want = frozen_c1([s for _, s, _ in cohort], [ok for _, _, ok in cohort], 0.1, 0.1)
    assert c["look"] == 1 and c["procedure"] == LB.TRAFFIC_PROCEDURE and c["counts"]["requested"] == 100
    assert (c["threshold"] or math.inf) == want and c["certified"] == math.isfinite(want)
    tl.tick()
    assert len(tl.traffic_certs) == 1  # made once


def test_c3_looks_fixed_in_advance_at_delta_over_k():
    clock = Clock()
    g = np.random.default_rng(2)
    tl = _log(clock, looks=(100, 200), tau_after=10_000)  # the region stays all traffic
    rows = _traffic(tl, g, 300, error=lambda s: 0.02)
    tl.post([{"request_id": rid, "answer": "billing" if ok else "tech"} for rid, _, ok in rows[:150]])
    tl.tick()
    first = tl.traffic_in_force()
    assert first["look"] == 1 and first["delta_look"] == pytest.approx(0.05)
    want = frozen_c1([s for _, s, _ in rows[:100]], [ok for _, _, ok in rows[:100]], 0.1, 0.05)
    assert (first["threshold"] or math.inf) == want
    tl.tick()
    assert len(tl.traffic_certs) == 1  # 150 labelled: nothing between the looks
    tl.post([{"request_id": rid, "answer": "billing" if ok else "tech"} for rid, _, ok in rows[150:200]])
    tl.tick()
    second = tl.traffic_in_force()
    assert second["look"] == 2 and second["delta_look"] == pytest.approx(0.05) and len(tl.traffic_certs) == 2
    assert (second["threshold"] or math.inf) == frozen_c1(
        [s for _, s, _ in rows[:200]], [ok for _, _, ok in rows[:200]], 0.1, 0.05
    )


def test_c4_the_asserts():
    clock = Clock()
    g = np.random.default_rng(3)
    tl = _log(clock, looks=(60,), tau_after=10_000)
    rows = _traffic(tl, g, 50, error=lambda s: 0.0) + [(_serve(tl, 0.9, 900 + i, stack="other"), 0.9, True)
                                                         for i in range(10)]  # fmt: skip
    tl.epochs = [*tl.epochs, {**tl.epochs[-1], "epoch": 2, "tau_min": 0.6, "tau_set": True}]  # the floor rises
    tl.post([{"request_id": rid, "answer": "billing"} for rid, _, _ in rows])
    tl.tick()
    c = tl.traffic_in_force()
    below = sum(1 for _, s, _ in rows[:50] if s < 0.6)
    assert c["counts"]["audit_only"] == below and c["counts"]["other_stack"] == 10
    assert c["counts"]["requested"] == 50 - below and c["t_min"] == 0.6
    assert c["threshold"] is None or c["threshold"] >= 0.6  # every candidate threshold at or above the floor


def test_c5_missing_labels_at_a_strict_alpha():
    clock = Clock()
    g = np.random.default_rng(4)
    tl = _log(clock, looks=(100,), alpha=0.01, tau_after=10_000, due_days=1)
    rows = _traffic(tl, g, 100, error=lambda s: 0.0)
    tl.post([{"request_id": rid, "answer": "billing"} for rid, _, _ in rows[:98]])  # two never labelled
    clock.t += 2 * 86400  # they expire
    tl.tick()
    c = tl.traffic_in_force()
    assert c["certified"] is False and c["reason"] == "labels_missing" and c["counts"]["missing"] == 2


def test_c6_through_the_server():
    from test_labels_served import ask, make

    client, so = make()
    body = {"id": "angry", "examples": __import__("test_noul_certified").noul_examples(600),
            "certify": {"looks": [80]},
            "labels": {"audit_pi": 0, "cert_pi": 1.0, "margin_pi_max": 0, "tau_after": 20}}  # fmt: skip
    r = client.post("/v1/tasks", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["traffic_certification"]["enabled"] is True and out["traffic_certification"]["looks"] == [80]
    assert out["labels"]["settings"]["cert_score"] == "label_free" and out["labels"]["settings"]["region"] == 0.45
    assert "s0" in out["labels"]["stored_fields"]
    examples_cert = so.task_store.snapshot("angry")["certificate"]["id"]
    for i in range(20):
        a, _ = ask(client, f"yes:{0.6 + 0.02 * i:.2f}:warm {i}")
        assert a["certificate"] == examples_cert  # the certificate from examples serves until the first look
    so.flush_labels(force=True)  # the floor is set from the first 20 answers
    log = so.labels_logs["angry"]
    assert log.epoch["tau_min"] > 0 and log.traffic["looks"] == [80]
    rids = []
    for i in range(200):
        a, _ = ask(client, f"yes:0.97:after {i}")
        rids.append(a["request_id"])
    drawn = [r for r in rids if log.rows[(r, "q1")]["draws"]["cert"]["in"]]
    assert all(log.rows[(r, "q1")]["s0"] >= log.epoch["tau_min"] for r in drawn)
    assert len(drawn) >= 80
    client.post("/v1/tasks/angry/labels", json={"labels": [{"request_id": r, "answer": "yes"} for r in drawn[:80]]})
    so.flush_labels(force=True)  # the first look's cohort is complete: its certificate is made
    tc = log.traffic_in_force()
    assert tc is not None and tc["look"] == 1 and tc["procedure"] == LB.TRAFFIC_PROCEDURE
    assert tc["certified"] is True and tc["counts"]["requested"] == 80
    a, _ = ask(client, "yes:0.97:certified")
    assert a["certificate"] == tc["id"] and a["certificate_kind"] == "traffic" and a["certified"] is True
    pf = LB.label_free(np.log([0.97, 0.03]), so.temperature_of("noul", None, 2))
    assert a["noul"] == pytest.approx(float(pf[0]), abs=1e-6)  # served with the label-free readout
    st = client.get("/v1/tasks/angry").json()["status"]
    assert st["traffic_certification"]["in_force"]["id"] == tc["id"]
    # without labels: the certificate from examples, nothing logged, and told what certifying from traffic needs
    plain, so2 = make()
    r = plain.post("/v1/tasks", json={"id": "calm", "examples": __import__("test_noul_certified").noul_examples(600),
                                      "certify": {}, "instructions": "Is the customer calm?"})  # fmt: skip
    assert r.status_code == 200 and r.json()["traffic_certification"]["enabled"] is False
    assert "labels" not in r.json() and so2.labels_logs == {}
    for bad in ({"looks": [0]}, {"looks": [200, 100]}, {"looks": 300}, {"target_share": 0}, {"target_share": 1.5}):
        r = plain.post("/v1/tasks", json={"id": "x", "examples": __import__("test_noul_certified").noul_examples(120),
                                          "certify": bad})  # fmt: skip
        assert r.status_code == 422, bad


def test_c7_pooling_needs_containment():
    clock = Clock()
    tl = _log(clock, looks=(60,), tau_after=10_000)

    def floor(tau):
        tl.epochs = [*tl.epochs, {**tl.epochs[-1], "epoch": len(tl.epochs), "tau_min": tau, "tau_set": True}]

    floor(0.6)
    high = [_serve(tl, 0.8, i) for i in range(30)]  # drawn above a floor of 0.6
    floor(0.3)  # the floor falls: the 0.6 epoch never drew the band 0.3 to 0.6
    low = [_serve(tl, 0.5, 100 + i) for i in range(30)]
    assert [k[0] for k, _ in tl.traffic_cohort()] == low
    tl.post([{"request_id": r, "answer": "billing"} for r in high + low])
    tl.tick()
    assert tl.traffic_in_force() is None and tl.traffic_status()["next_look"]["drawn"] == 30
    floor(0.7)  # the floor rises above both: both epochs drew all of its band, so both pool
    assert [k[0] for k, _ in tl.traffic_cohort()] == high + low
    tl.tick()
    c = tl.traffic_in_force()
    assert c["counts"]["audit_only"] == 30 and c["counts"]["requested"] == 30  # answers under the new floor
