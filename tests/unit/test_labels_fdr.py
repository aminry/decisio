# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The average-error (FDR) mode (decisio.serve.labels: `fdr_pvalues`, `bh`, the windows).

F1  parity with the frozen form the mode was validated with: Mondrian conformal p-values and Benjamini-Hochberg, on a
    fixture with several buckets
F2  marks only when a window closes: a window of N answers in serve order closes once its newest answer has settled;
    it is closed once, its marks recorded; the next window starts after it
F3  an empty calibration bucket marks nothing in it; Benjamini-Hochberg runs over the whole window, not per bucket
F4  through the server: the mode needs labels on (refused otherwise); the registration says the reviewed promise; no
    answer is ever certified; the marks come in the status and the export

  uv run pytest -q tests/unit/test_labels_fdr.py
"""

import numpy as np

from decisio.serve import labels as LB
from decisio.serve.fast_table_learn import MemoryPlace

KEYS = ["billing", "tech", "sales"]


def frozen_bh(p, alpha):
    """The frozen form's Benjamini-Hochberg, verbatim."""
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    o = np.argsort(p, kind="stable")
    ok = p[o] <= alpha * np.arange(1, m + 1) / m
    sel = np.zeros(m, bool)
    if ok.any():
        sel[o[: np.nonzero(ok)[0].max() + 1]] = True
    return sel


def frozen_pvalues(s_cal, wrong_cal, b_cal, s_bat, b_bat):
    """The frozen form's Mondrian conformal p-values, verbatim."""
    p = np.ones(len(s_bat))
    for b in np.unique(b_bat):
        mb = b_bat == b
        mc = b_cal == b
        n = int(mc.sum())
        if n == 0:
            continue
        w = np.sort(s_cal[mc & wrong_cal])
        ge = len(w) - np.searchsorted(w, s_bat[mb], side="left")
        p[mb] = (1.0 + ge) / (n + 1.0)
    return p


def test_f1_parity_with_the_frozen_form():
    g = np.random.default_rng(7)
    for trial in range(30):
        n, m = int(g.integers(20, 600)), int(g.integers(50, 400))
        s_cal, s_bat = 0.3 + 0.7 * g.random(n), 0.3 + 0.7 * g.random(m)
        wrong = g.random(n) < 0.2 * (1.2 - s_cal)
        b_cal, b_bat = g.integers(0, 3, n), g.integers(0, 4, m)  # bucket 3 has no calibration answer
        ours = LB.fdr_pvalues(s_cal, wrong, b_cal, s_bat, b_bat)
        ref = frozen_pvalues(s_cal, wrong, b_cal, s_bat, b_bat)
        assert np.array_equal(ours, ref), trial
        for alpha in (0.10, 0.05, 0.01):
            assert np.array_equal(LB.bh(ours, alpha), frozen_bh(ref, alpha)), (trial, alpha)
    assert LB.option_cell(2, "noul") == "yes/no" and LB.option_cell(5, "choice") == "4-5"
    assert LB.option_cell(31, "choice") == "30+"


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _log(clock, window=50, alpha=0.1):
    s = LB.parse_settings({"audit_pi": 1.0, "cert_pi": 0, "margin_pi_max": 0, "tau_after": 10_000})  # all audit-drawn
    tl = LB.TaskLabels("t", s, LB.new_secrets(), MemoryPlace(), "r1", clock)
    tl.fdr = {"alpha": alpha, "window": window, "qtype": "choice"}
    return tl


def _serve(tl, s0, i, lang="en"):
    tl.clock.t += 1
    rid = LB.request_id("r1", tl.clock())
    tl.observe(rid, "q", text=f"t{i}", keys=KEYS, served=[s0, (1 - s0) / 2, (1 - s0) / 2], lp=None, h=None,
               certified=None, version="v1", stack="fp", lang=lang, s0=s0, s0_top="billing")  # fmt: skip
    return rid


def _label(tl, rids, wrong):
    tl.post([{"request_id": r, "answer": "tech" if w else "billing"} for r, w in zip(rids, wrong)])


def test_f2_marks_only_when_a_window_closes():
    clock = Clock()
    g = np.random.default_rng(1)
    tl = _log(clock, window=50)
    s0 = [0.5 + 0.5 * (i % 10) / 10 for i in range(150)]
    rids = [_serve(tl, s, i) for i, s in enumerate(s0)]
    wrong = [bool(g.random() < 0.4 * (1.1 - s)) for s in s0]
    _label(tl, rids[50:150], wrong[50:150])  # calibration: audit labels outside the first window
    tl.tick()
    assert tl.fdr_windows == []  # full windows, but not settled yet
    clock.t += LB.FDR_SETTLE + 1
    tl.tick()
    assert len(tl.fdr_windows) == 1  # one window per turn
    w0 = tl.fdr_windows[0]
    assert [tuple(x)[0] for x in w0["answers"]] == rids[:50] and w0["calibration"]["n"] == 100
    p = LB.fdr_pvalues(s0[50:150], wrong[50:150], ["3"] * 100, s0[:50], ["3"] * 50)
    want = {r for r, m in zip(rids[:50], LB.bh(p, 0.1)) if m}
    assert {tuple(x)[0] for x in w0["marked"]} == want  # BH over the window against the audit labels outside it
    tl.tick()
    w1 = tl.fdr_windows[1]
    assert w1["calibration"]["n"] == 50  # its own 50 are left out of its calibration
    st = tl.fdr_status()
    assert st["label"] == "Average error (FDR)" and "at most 10%" in st["promise"] and st["calibration"]["floor"] == 9
    assert len(tl.fdr_marks()) == 100


def test_f3_the_floor_and_bh_over_the_window():
    clock = Clock()
    tl = _log(clock, window=40)
    cal = [_serve(tl, 0.9, i) for i in range(8)]  # 8 audit labels: under the floor of 9 at alpha 0.1
    _label(tl, cal, [False] * 8)
    for i in range(80):  # two windows: the second has all 8 calibration rows outside it
        _serve(tl, 0.95, 100 + i)
    clock.t += LB.FDR_SETTLE + 1
    tl.tick()
    assert tl.fdr_windows[0]["marked"] == []  # the first window holds the calibration rows themselves
    tl.tick()
    assert len(tl.fdr_windows) == 2 and tl.fdr_windows[1]["calibration"]["per_bucket"] == {"3": 8}
    assert tl.fdr_windows[1]["marked"] == []  # 8 audit labels in the bucket, floor 9: nothing marked
    assert tl.fdr_windows[1]["calibration"]["below_floor"] == ["3"]
    st = tl.fdr_status()
    assert st["calibration"]["below_floor"] == ["3"] and "until each has 9 labelled audit answers: 3." in (
        st["calibration"]["reason"])  # fmt: skip
    assert st["windows"][-1]["below_floor"] == ["3"]
    tl2 = _log(clock, window=40)
    cal = [_serve(tl2, 0.9 if i < 30 else 0.4, 200 + i) for i in range(40)]
    _label(tl2, cal, [i >= 30 for i in range(40)])
    win = [_serve(tl2, 0.95 if i < 20 else 0.35, 300 + i, lang="en") for i in range(40)]
    clock.t += LB.FDR_SETTLE + 1
    tl2.tick()
    tl2.tick()
    w = tl2.fdr_windows[1]
    assert [tuple(x)[0] for x in w["answers"]] == win
    p = LB.fdr_pvalues([0.9] * 30 + [0.4] * 10, [False] * 30 + [True] * 10, ["3"] * 40, [0.95] * 20 + [0.35] * 20,
                       ["3"] * 40)  # fmt: skip
    assert len(w["marked"]) == int(LB.bh(p, 0.1).sum()) and 0 < len(w["marked"]) < 40


def test_f4_through_the_server(monkeypatch):
    from test_labels_served import ask, make
    from test_noul_certified import noul_examples

    monkeypatch.setattr(LB, "FDR_SETTLE", 0)
    client, so = make()
    r = client.post("/v1/tasks", json={"id": "angry", "examples": noul_examples(120),
                                       "certify": {"mode": "average_error", "window": 30}})  # fmt: skip
    assert r.status_code == 422 and "labels" in r.text  # windows of logged answers need labels on
    r = client.post("/v1/tasks", json={"id": "angry", "examples": noul_examples(120), "labels": {"audit_pi": 1.0},
                                       "certify": {"mode": "average_error", "alpha": 0.1, "window": 30}})  # fmt: skip
    assert r.status_code == 200, r.text
    ae = r.json()["average_error"]
    assert ae["label"] == "Average error (FDR)" and ae["window"] == 30 and ae["promise"] == LB.fdr_sentence(0.1)
    assert ae["procedure"] == "decisio-certify/average-error-conformal-bh.v1" and ae["delta"] is None
    assert "certificate" not in so.task_store.snapshot("angry")
    rids = []
    for i in range(65):
        a, _ = ask(client, f"yes:{0.6 + 0.005 * i:.3f}:m {i}")
        assert "certified" not in a
        rids.append(a["request_id"])
    client.post("/v1/tasks/angry/labels", json={"labels": [{"request_id": r, "answer": "yes"} for r in rids[30:65]]})
    so.flush_labels(force=True)
    st = client.get("/v1/tasks/angry/labels/status").json()["average_error"]
    assert len(st["windows"]) == 1 and st["windows"][0]["size"] == 30
    # the next window holds answers 30-59, so only the 5 labelled answers after it calibrate it
    assert st["calibration"]["audit_labels"] == {"yes/no": 5} and st["calibration"]["floor"] == 9
    assert st["calibration"]["below_floor"] == ["yes/no"] and st["calibration"]["reason"]
    lines = client.get("/v1/tasks/angry/labels/export", params={"view": "marks"}).text.splitlines()
    assert len(lines) == 30
    bad = {"id": "x", "examples": noul_examples(120), "labels": True, "certify": {"mode": "average_error", "window": 5}}
    assert client.post("/v1/tasks", json=bad).status_code == 422
