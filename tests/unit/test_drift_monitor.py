# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The traffic monitor of a certified task: a confidence signal beside the answer mix, suspension on alert, and the
warning for a task with no catch-all option. No model: a fake engine whose answer is set by the example's text.

The card replay of 2026-10-16 (the 31B, CLINC150, 10 examples per intent, 5,363 messages, the out-of-scope share
rising from message 1,000) is the case: the answer mix alerted after message 5,200 and 976 wrong certified answers, and
certification stayed on. The signals are replayed on that stream in `runs/2026-10-17_drift-monitor/` and on a card
by `experiments/2026-10-17_serve_drift_monitor/CELL_DRIFT_REPLAY.md` (Serve's RLCD folder).

M1  the confidence baseline: a tail share, a limit that a normal stream does not reach (at most 1 percent of
    5,000-answer streams) and a shifted stream does quickly; none for too few examples
M2  the monitor: the statistic alerts on a rise in low confidences, clears with hysteresis, shows in the facts; a task
    made before it existed runs the answer mix alone
M3  suspension: on by default, every answer certified false with the reason `traffic_changed` while the alert is on,
    back when it clears, off by `"suspend_on_alert": false`, kept by a re-certification, the settings validated
M4  the warning for a choice task with no catch-all option (and none for one with, or for yes/no)
M5  the metrics carry the statistic

  uv run pytest -q tests/unit/test_drift_monitor.py
"""

import numpy as np
import pytest
from test_certified_answers import ask, examples, make, post_task

from decisio.serve import monitor
from decisio.serve.systemone import has_catch_all

rng0 = np.random.default_rng(11)
CERT_CONF = np.clip(rng0.beta(8, 1.2, 2000), 0, 0.999)  # a skewed set of top probabilities, most near 1


# ---- M1 -----------------------------------------------------------------------------------------------------------


def stream(share, n, thr, seed):
    r = np.random.default_rng(seed)
    low = r.random(n) < share
    return np.where(low, thr * 0.5, 0.97)


def alarm_at(base, confs):
    m = monitor.MixMonitor(monitor.baseline([0, 1, 2] * 50, 3, "t", CERT_CONF))
    for i, c in enumerate(confs, 1):
        m.observe(i % 3, True, float(c))
        if m.conf_alerting:
            return i
    return None


def test_m1_the_confidence_baseline_and_its_limit():
    b = monitor.confidence_baseline(CERT_CONF, "t")
    assert b["tail_quantile"] == 0.05 and abs(b["tail_share"] - 0.05) < 0.01
    assert b["p_ok"] == pytest.approx(1.3 * b["tail_share"]) and b["p_bad"] == pytest.approx(2.5 * b["tail_share"])
    assert b["limit"] > 3 and b["limit"] == monitor.confidence_baseline(CERT_CONF, "t")["limit"]  # seeded: the same
    assert monitor.confidence_baseline(CERT_CONF[:10], "t") is None  # too few examples
    assert (
        monitor.confidence_baseline(np.where(np.arange(100) < 60, 0.1, 0.9), "t")["tail_share"] > 0
    )  # a heavy tail is fine
    tied = monitor.confidence_baseline(np.full(100, 0.5), "t")  # all tied: share 0, floored at 1/n
    assert tied is None or tied["tail_share"] == pytest.approx(0.01)
    base = monitor.baseline([0, 1, 2] * 50, 3, "t", CERT_CONF)
    assert base["confidence"] == b and "confidence" not in monitor.baseline([0, 1, 2] * 50, 3, "t")


def test_m1_a_normal_stream_stays_quiet_and_a_shifted_one_alarms_quickly():
    b = monitor.confidence_baseline(CERT_CONF, "t")
    thr = b["tail_threshold"]
    # normal traffic: the normal share (1.3 x the examples') or the examples' own
    quiet = [
        alarm_at(monitor.baseline([0, 1, 2] * 50, 3, "t", CERT_CONF), stream(b["p_ok"], 5000, thr, s))
        for s in range(60)
    ]
    assert sum(a is not None for a in quiet) <= 3  # the limit allows 1 percent of 5,000-answer streams
    # the tail share doubling past the alarm level: detected well inside a window
    base = monitor.baseline([0, 1, 2] * 50, 3, "t", CERT_CONF)
    hits = [alarm_at(base, stream(2 * b["p_bad"], 3000, thr, s)) for s in range(30)]
    assert all(h is not None and h < 400 for h in hits)


# ---- M2 -----------------------------------------------------------------------------------------------------------


def test_m2_the_statistic_alerts_and_clears_with_hysteresis():
    base = monitor.baseline([0, 1, 2] * 50, 3, "t", CERT_CONF)
    thr, limit = base["confidence"]["tail_threshold"], base["confidence"]["limit"]
    m = monitor.MixMonitor(base)
    for _ in range(100):
        m.observe(0, True, 0.97)
    assert m.facts()["state"] == "ok" and m.facts()["confidence"]["statistic"] == 0.0 and m.facts()["signals"] == []
    n = 0
    while not m.conf_alerting:
        m.observe(0, True, thr * 0.5)
        n += 1
        assert n < 200
    f = m.facts()
    assert f["state"] == "traffic_changed" and f["signals"] == ["confidence"] and f["first_alert_at"] == 100 + n
    assert f["confidence"]["statistic"] > limit and f["confidence"]["alerts"] == 1 and m.alerting
    while m.facts()["confidence"]["statistic"] > limit / 2 + 1e-9:  # normal traffic brings it down...
        assert m.alerting  # ...and the alert holds until the statistic is under half the limit
        m.observe(0, True, 0.97)
    m.observe(0, True, 0.97)
    assert not m.alerting and m.facts()["state"] == "ok"
    m.observe(0, True, 0.97)  # a confidence is optional: the mix signal alone still counts
    m2 = monitor.MixMonitor(monitor.baseline([0, 1, 2] * 50, 3, "t"))  # a task made before the signal existed
    for _ in range(50):
        m2.observe(1, True, 0.1)
    assert "confidence" not in m2.facts() and m2.facts()["state"] == "ok"


# ---- M3 -----------------------------------------------------------------------------------------------------------


def registered(body_extra=None, n=400):
    client, so = make()
    r = post_task(client, {"id": "routing", "examples": examples(n, 0.96), "certify": body_extra or {}})
    assert r.status_code == 200, r.text
    return client, so, r.json()


def test_m3_traffic_that_looks_unlike_the_examples_suspends_certification_by_default():
    client, so, task = registered()
    c = task["certificate"]
    assert c["suspend_on_alert"] is True and "confidence" in so.task_store.by_key[task["key"]]["monitor"]
    thr = so.task_store.by_key[task["key"]]["monitor"]["confidence"]["tail_threshold"]
    hi = ask(client, 1, 0.99)
    assert hi["certified"] is True
    seen = []
    for _ in range(120):  # messages the model is unsure about: a tail far above the examples'
        seen.append(ask(client, 1, round(thr * 0.5, 4)))
    st = client.get("/v1/tasks/routing").json()["status"]
    assert st["state"] == "traffic_changed" and st["monitor"]["signals"] == ["confidence"]
    assert "certification is suspended" in st["text"]
    a = ask(client, 1, 0.99)  # a confident answer, certified before, is not while the alert is on
    assert a["certified"] is False and a["certified_reason"] == "traffic_changed" and a["certificate"] == c["id"]
    # the alert clears when the traffic is normal again, and certification returns
    mix0 = np.array(so.task_store.by_key[task["key"]]["monitor"]["option_counts"], float)
    rng = np.random.default_rng(2)
    for _ in range(900):  # traffic like the examples: their mix of options, confident
        ask(client, int(rng.choice(3, p=mix0 / mix0.sum())), 0.99)
    assert ask(client, 1, 0.99)["certified"] is True
    assert client.get("/v1/tasks/routing").json()["status"]["state"] == "ok"
    # the answers themselves never changed
    assert all(x["choice"] == "option_1" for x in seen)


def test_m3_suspension_can_be_turned_off_and_is_kept_by_re_certification():
    client, so, task = registered({"suspend_on_alert": False})
    thr = so.task_store.by_key[task["key"]]["monitor"]["confidence"]["tail_threshold"]
    for _ in range(120):
        ask(client, 1, round(thr * 0.5, 4))
    st = client.get("/v1/tasks/routing").json()["status"]
    assert st["state"] == "traffic_changed" and "not suspended" in st["text"]
    assert ask(client, 1, 0.99)["certified"] is True  # reported, not acted on
    r = client.post("/v1/tasks/routing/certify", json={"examples": examples(200, 0.96, seed=9, start=10_000)})
    assert r.status_code == 200 and r.json()["certificate"]["suspend_on_alert"] is False  # the setting is inherited
    r = client.post(
        "/v1/tasks/routing/certify",
        json={"examples": examples(200, 0.96, seed=9, start=10_000), "suspend_on_alert": True},
    )
    assert r.json()["certificate"]["suspend_on_alert"] is True


def test_m3_the_setting_is_validated():
    client, so = make()
    r = post_task(client, {"id": "t", "examples": examples(400, 0.96), "certify": {"suspend_on_alert": "yes"}})
    assert r.status_code == 422 and "suspend_on_alert is true or false" in r.json()["detail"]
    r = post_task(client, {"id": "t", "examples": examples(400, 0.96), "certify": {"suspend": True}})
    assert r.status_code == 422 and "suspend_on_alert" in r.json()["detail"]


# ---- M4 -----------------------------------------------------------------------------------------------------------


def test_m4_a_choice_task_with_no_catch_all_option_is_warned():
    client, so, task = registered()
    w = task["certificate"]["warnings"]
    assert w[0]["code"] == "no_catch_all_option" and "out of scope" in w[0]["text"]
    assert "Warning:" in task["certificate"]["summary"]
    for keys, expect in (
        (["billing", "other"], True),
        (["billing", "out_of_scope"], True),
        (["a", "None of the above"], True),
        (["a", "unknown"], True),
        (["a", "can't tell"], True),
        (["billing", "shipping", "access"], False),
        (["otherwise_billing", "mother"], False),  # whole words only
    ):
        assert has_catch_all(keys) is expect, keys


def test_m4_no_warning_for_a_task_with_a_catch_all_option():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from test_certified_answers import CRIT, FakeEngine

    from decisio.serve.systemone import SystemOne, add_routes
    from decisio.serve.tasks import TaskStore

    crit = {**CRIT, "other": "something else"}
    so = SystemOne(FakeEngine(), "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    client = TestClient(app)
    exs = examples(400, 0.96)
    for e in exs:
        e["request"]["questions"]["q1"]["criteria"] = crit
    r = post_task(client, {"id": "withother", "examples": exs, "certify": {}})
    assert r.status_code == 200, r.text
    assert "warnings" not in r.json()["certificate"] and "Warning" not in r.json()["certificate"]["summary"]


# ---- M5 -----------------------------------------------------------------------------------------------------------


def test_m5_the_metrics_carry_the_statistic():
    client, so, task = registered()
    ask(client, 1, 0.99)
    text = client.get("/metrics").text
    assert 'decisio_task_confidence_statistic{task="routing"}' in text
    assert 'decisio_task_confidence_limit{task="routing"}' in text and "decisio_task_confidence_alerts_total" in text


def test_m3_the_answer_mix_alone_reports_and_does_not_suspend():
    """On the card's BANKING77 draws the mix signal alerted at message 500 on normal traffic from another split (the
    examples' mix is the training split's, the traffic is the test split's); the confidence signal stayed quiet.
    Only the confidence signal suspends."""
    client, so, task = registered()
    for _ in range(600):  # every answer option 0, confident: the mix moves, the confidences do not
        ask(client, 0, 0.99)
    st = client.get("/v1/tasks/routing").json()["status"]
    assert st["state"] == "traffic_changed" and st["monitor"]["signals"] == ["answer_mix"]
    assert "not suspended" in st["text"]
    assert ask(client, 1, 0.99)["certified"] is True


def legacy_export(exported):
    """A task as decisio 0.14.0 and 0.15.0 exported it: a certificate with no `suspend_on_alert` and no `warnings`,
    and a monitor baseline with no confidence block."""
    import copy

    out = copy.deepcopy(exported)
    for t in out:
        t["certificate"].pop("suspend_on_alert", None)
        t["certificate"].pop("warnings", None)
        t["monitor"].pop("confidence", None)
    return out


def test_m3_a_certificate_made_before_the_signal_takes_the_default_on_re_certification_and_suspends():
    """S21's drift cell (RLCD, Lab 2): Lab 3's task was certified by 0.14.0, so `old.get("suspend_on_alert", False)`
    gave a re-certification false and "on by default" held for no upgrading user."""
    client, so, task = registered()
    legacy = legacy_export(client.get("/v1/tasks?full=1").json()["tasks"])
    assert "suspend_on_alert" not in legacy[0]["certificate"] and "confidence" not in legacy[0]["monitor"]
    c2, so2 = make()
    assert c2.post("/v1/tasks/import", json={"tasks": legacy}).status_code == 200
    # imported as it was: no confidence signal yet, so nothing can suspend, and an answer is served as before
    assert ask(c2, 1, 0.99)["certified"] is True
    st = c2.get("/v1/tasks/routing").json()["status"]
    assert "confidence" not in st["monitor"]
    # re-certified with no setting named: the default, true
    r = c2.post("/v1/tasks/routing/certify", json={"examples": examples(200, 0.96, seed=9, start=10_000)})
    assert r.status_code == 200, r.text
    assert r.json()["certificate"]["suspend_on_alert"] is True
    thr = so2.task_store.by_key[r.json()["key"]]["monitor"]["confidence"]["tail_threshold"]
    for _ in range(120):
        ask(c2, 1, round(thr * 0.5, 4))
    a = ask(c2, 1, 0.99)
    assert a["certified"] is False and a["certified_reason"] == "traffic_changed"
    assert "certification is suspended" in c2.get("/v1/tasks/routing").json()["status"]["text"]


def test_m3_a_stored_certificate_without_the_key_reads_as_the_default_when_it_alerts():
    """A task whose certificate has no `suspend_on_alert` but whose monitor has the confidence block (an edited export)
    is read with the default, not as false."""
    client, so, task = registered()
    t = so.task_store.by_key[task["key"]]
    thr = t["monitor"]["confidence"]["tail_threshold"]
    t["certificate"].pop("suspend_on_alert")
    for _ in range(120):
        ask(client, 1, round(thr * 0.5, 4))
    a = ask(client, 1, 0.99)
    assert a["certified"] is False and a["certified_reason"] == "traffic_changed"
    assert "certification is suspended" in client.get("/v1/tasks/routing").json()["status"]["text"]
