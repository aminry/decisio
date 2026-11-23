# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The average-error (FDR) mode (decisio.serve.labels: `fdr_pvalues`, `bh`, the windows).

F1  parity with the frozen form the mode was validated with: Mondrian conformal p-values and Benjamini-Hochberg, on a
    fixture with several buckets; with one bucket, the frozen one-bucket conformal selection
F2  marks only when a window closes: a window of N answers in serve order closes once its newest answer has settled;
    it is closed once, its marks recorded; the next window starts after it; its calibration is the audit answers
    outside it drawn more than due_days before it closes
F3  a bucket under the floor marks nothing in it; Benjamini-Hochberg runs over the whole window, not per bucket
F4  through the server: the mode needs labels on (refused otherwise); the registration says the reviewed promise; no
    answer is ever certified; the marks come in the status and the export
F5  re-scored from the logged readout under the current stack: calibration answers from another stack or without a
    readout are left out, and window answers of either kind are never marked (counted as unmarkable)
F6  missing audit labels: an expired or declined audit answer counts as wrong at its score; a bucket whose audit
    answers were all skipped never marks; counting a skipped answer wrong never adds a mark over its having been
    labelled; calibration.audit_response
F7  erasure: a window names its answers by keyed digest only, so after the purge no request id of an erased answer is
    stored; the marks export leaves an erased answer out at once

  uv run pytest -q tests/unit/test_labels_fdr.py
"""

import numpy as np
import pytest

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


def frozen_conformal_select(score_cal, correct_cal, score_batch, alpha):
    """The frozen one-bucket form (conformal selection, then BH over the batch), verbatim."""
    n = len(score_cal)
    w = np.sort(score_cal[~correct_cal])
    ge = len(w) - np.searchsorted(w, score_batch, side="left")
    p = (1.0 + ge) / (n + 1.0)
    m = len(p)
    order = np.argsort(p)
    ok = p[order] <= alpha * np.arange(1, m + 1) / m
    sel = np.zeros(m, bool)
    if ok.any():
        kmax = np.nonzero(ok)[0].max()
        sel[order[: kmax + 1]] = True
    return sel


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
        one = np.zeros(n, int), np.zeros(m, int)  # one bucket: the frozen one-bucket selection
        for alpha in (0.10, 0.05):
            ours1 = LB.bh(LB.fdr_pvalues(s_cal, wrong, one[0], s_bat, one[1]), alpha)
            assert np.array_equal(ours1, frozen_conformal_select(s_cal, ~wrong, s_bat, alpha)), (trial, alpha)
    assert LB.option_cell(2, "noul") == "yes/no" and LB.option_cell(5, "choice") == "4-5"
    assert LB.option_cell(31, "choice") == "30+"


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


DAY = 86_400


def _log(clock, window=50, alpha=0.1):
    s = LB.parse_settings({"audit_pi": 1.0, "cert_pi": 0, "margin_pi_max": 0, "tau_after": 10_000,
                           "due_days": 1})  # every answer audit-drawn; a drawn answer is due in a day  # fmt: skip
    tl = LB.TaskLabels("t", s, LB.new_secrets(), MemoryPlace(), "r1", clock)
    tl.fdr = {"alpha": alpha, "window": window, "qtype": "choice"}
    return tl


def _serve(tl, s0, i, lang="en", stack="fp", lp=False):
    tl.clock.t += 1
    rid = LB.request_id("r1", tl.clock())
    dist = [s0, (1 - s0) / 2, (1 - s0) / 2]
    tl.observe(rid, "q", text=f"t{i}", keys=KEYS, served=dist, lp=np.log(dist) if lp else None, h=None,
               certified=None, version="v1", stack=stack, lang=lang, s0=s0, s0_top="billing")  # fmt: skip
    return rid


def _label(tl, rids, wrong):
    tl.post([{"request_id": r, "answer": "tech" if w else "billing"} for r, w in zip(rids, wrong)])


def _ak(tl, rids):
    """The keyed digests a window names its answers by."""
    return [tl.akey((r, "q")) for r in rids]


def _due(tl):
    """Move past due_days: every answer served so far is closed, so it can calibrate."""
    tl.clock.t += DAY + 1


def _settle(tl):
    tl.clock.t += LB.FDR_SETTLE + 1


def test_f2_marks_only_when_a_window_closes():
    clock = Clock()
    g = np.random.default_rng(1)
    tl = _log(clock, window=50)
    s_cal = [0.5 + 0.5 * (i % 10) / 10 for i in range(100)]
    cal = [_serve(tl, s, i) for i, s in enumerate(s_cal)]
    tl.tick()
    assert tl.fdr_windows == []  # full windows, but not settled yet
    w_cal = [bool(g.random() < 0.4 * (1.1 - s)) for s in s_cal]
    _label(tl, cal, w_cal)
    _due(tl)
    s_win = [0.55 + 0.45 * (i % 9) / 9 for i in range(50)]
    win = [_serve(tl, s, 100 + i) for i, s in enumerate(s_win)]
    tl.tick()
    assert len(tl.fdr_windows) == 1  # one window per turn
    assert tl.fdr_windows[0]["calibration"]["n"] == 50  # the other 50 calibration answers: its own are left out
    tl.tick()
    tl.tick()
    assert len(tl.fdr_windows) == 2  # the third window's answers have not settled
    _settle(tl)
    tl.tick()
    w = tl.fdr_windows[2]
    assert w["answers"] == _ak(tl, win) and w["calibration"]["n"] == 100
    assert w["calibration"]["audit_response"] == 1.0
    p = LB.fdr_pvalues(s_cal, w_cal, ["3"] * 100, s_win, ["3"] * 50)
    want = {r for r, m in zip(win, LB.bh(p, 0.1)) if m}
    assert want and set(w["marked"]) == set(_ak(tl, want))  # BH over the window against the closed audit answers
    st = tl.fdr_status()
    assert st["label"] == "Average error (FDR)" and "at most 10%" in st["promise"] and st["calibration"]["floor"] == 9
    assert len(tl.fdr_marks()) == 150


def test_f2b_young_audit_answers_wait_even_when_labelled():
    clock = Clock()
    tl = _log(clock, window=30)
    cal = [_serve(tl, 0.7, i) for i in range(30)]
    _label(tl, cal, [False] * 30)  # labelled at once, but drawn less than due_days ago: not in the cohort yet
    win = [_serve(tl, 0.95, 100 + i) for i in range(30)]
    _settle(tl)
    tl.tick()
    tl.tick()
    assert tl.fdr_windows[1]["calibration"]["n"] == 0 and tl.fdr_windows[1]["marked"] == []
    assert tl.fdr_windows[1]["answers"] == _ak(tl, win)


def test_f3_the_floor_and_bh_over_the_window():
    clock = Clock()
    tl = _log(clock, window=40)
    cal = [_serve(tl, 0.9, i) for i in range(8)]  # 8 audit labels: under the floor of 9 at alpha 0.1
    _label(tl, cal, [False] * 8)
    _due(tl)
    for i in range(80):  # two windows: the second has all 8 calibration rows outside it
        _serve(tl, 0.95, 100 + i)
    _settle(tl)
    tl.tick()
    assert tl.fdr_windows[0]["marked"] == []  # the first window holds the calibration rows themselves
    tl.tick()
    assert len(tl.fdr_windows) == 2 and tl.fdr_windows[1]["calibration"]["per_bucket"] == {"3": 8}
    assert tl.fdr_windows[1]["marked"] == []  # 8 audit labels in the bucket, floor 9: nothing marked
    assert tl.fdr_windows[1]["calibration"]["below_floor"] == ["3"]
    st = tl.fdr_status()
    assert st["calibration"]["below_floor"] == ["3"] and "until each has 9 closed audit answers" in (
        st["calibration"]["reason"])  # fmt: skip
    assert st["windows"][-1]["below_floor"] == ["3"]
    tl2 = _log(clock, window=40)
    cal = [_serve(tl2, 0.9 if i < 30 else 0.4, 200 + i) for i in range(40)]
    _label(tl2, cal, [i >= 30 for i in range(40)])
    _due(tl2)
    win = [_serve(tl2, 0.95 if i < 20 else 0.35, 300 + i) for i in range(40)]
    _settle(tl2)
    tl2.tick()
    tl2.tick()
    w = tl2.fdr_windows[1]
    assert w["answers"] == _ak(tl2, win)
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
    r = client.post("/v1/tasks", json={"id": "angry", "examples": noul_examples(120),
                                       "labels": {"audit_pi": 1.0, "due_days": 1},
                                       "certify": {"mode": "average_error", "alpha": 0.1, "window": 30}})  # fmt: skip
    assert r.status_code == 200, r.text
    ae = r.json()["average_error"]
    assert ae["label"] == "Average error (FDR)" and ae["window"] == 30 and ae["promise"] == LB.fdr_sentence(0.1)
    assert ae["procedure"] == "decisio-certify/average-error-conformal-bh.v1" and ae["delta"] is None
    assert "certificate" not in so.task_store.snapshot("angry")
    rids = []
    for i in range(30):
        a, _ = ask(client, f"yes:{0.6 + 0.01 * i:.3f}:cal {i}")
        assert "certified" not in a
        rids.append(a["request_id"])
    client.post("/v1/tasks/angry/labels", json={"labels": [{"request_id": r, "answer": "yes"} for r in rids]})
    log = so.labels_logs["angry"]
    real = log.clock
    log.clock = lambda: real() + DAY + 60  # a day on: the 30 audit answers are closed
    for i in range(35):
        a, _ = ask(client, f"yes:{0.9 + 0.002 * i:.3f}:m {i}")
        assert "certified" not in a
    so.flush_labels(force=True)
    st = client.get("/v1/tasks/angry/labels/status").json()["average_error"]
    assert len(st["windows"]) == 1 and st["windows"][0]["size"] == 30
    assert st["calibration"]["audit_labels"] == {"yes/no": 30} and st["calibration"]["audit_response"] == 1.0
    so.flush_labels(force=True)
    st = client.get("/v1/tasks/angry/labels/status").json()["average_error"]
    assert len(st["windows"]) == 2 and st["windows"][1]["marked"] > 0
    lines = client.get("/v1/tasks/angry/labels/export", params={"view": "marks"}).text.splitlines()
    assert len(lines) == 60
    bad = {"id": "x", "examples": noul_examples(120), "labels": True, "certify": {"mode": "average_error", "window": 5}}
    assert client.post("/v1/tasks", json=bad).status_code == 422


def test_f5_other_stacks_and_missing_readouts_are_left_out():
    clock = Clock()
    tl = _log(clock, window=45)
    tl.fdr["stack"] = "fp"
    tl.fdr_temperature = lambda k: 1.0  # re-scored from the logged readout: softmax(lp / 1) is the logged distribution
    s_ok = [float(x) for x in np.linspace(0.6, 0.94, 30)]
    wrong_ok = [x < 0.7 for x in s_ok]  # the wrong calibration answers score low
    cal = [_serve(tl, s, i, lp=True) for i, s in enumerate(s_ok)]
    other = [_serve(tl, 0.99, 100 + i, stack="other", lp=True) for i in range(10)]  # another stack: left out
    bare = [_serve(tl, 0.99, 200 + i) for i in range(5)]  # no plain readout: left out
    _label(tl, cal, wrong_ok)
    _label(tl, other + bare, [True] * 15)  # all wrong: would pull every p-value up if they were used
    _due(tl)
    win_s = [float(x) for x in np.linspace(0.95, 0.99, 35)]  # above every wrong one: marked, unless the 0.99 wrong
    # answers of another stack (or without a readout) were let into the calibration
    win = [_serve(tl, s, 300 + i, lp=True) for i, s in enumerate(win_s)]
    win_other = [_serve(tl, 0.999, 400 + i, stack="other", lp=True) for i in range(5)]
    win_bare = [_serve(tl, 0.999, 500 + i) for i in range(5)]
    _settle(tl)
    tl.tick()
    tl.tick()
    w = tl.fdr_windows[1]
    assert w["answers"] == _ak(tl, win + win_other + win_bare)
    assert w["calibration"]["n"] == 30 and w["calibration"]["left_out"] == {"other_stack": 10, "no_lp": 5}
    assert w["unmarkable"] == {"other_stack": 5, "no_lp": 5}
    rescore = lambda s: float(LB.label_free(np.log([s, (1 - s) / 2, (1 - s) / 2]), 1.0).max())  # noqa: E731
    p = LB.fdr_pvalues([rescore(x) for x in s_ok], wrong_ok, ["3"] * 30, [rescore(x) for x in win_s] + [0.0] * 10,
                       ["3"] * 45)  # fmt: skip
    p[35:] = 1.0
    want = {r for r, m in zip(win + win_other + win_bare, LB.bh(p, 0.1)) if m}
    assert want and set(w["marked"]) == set(_ak(tl, want)) and not want & set(win_other + win_bare)
    st = tl.fdr_status()
    assert st["windows"][-1]["unmarkable"] == {"other_stack": 5, "no_lp": 5}


def _window_marks(cal_s, cal_label, win_s, window):
    """The marks of the window after a calibration whose answers got `cal_label` (True wrong, False right, None
    skipped: expired)."""
    clock = Clock()
    tl = _log(clock, window=window)
    cal = [_serve(tl, s, i) for i, s in enumerate(cal_s)]
    got = [(r, w) for r, w in zip(cal, cal_label) if w is not None]
    if got:
        _label(tl, [r for r, _ in got], [w for _, w in got])
    _due(tl)
    [_serve(tl, s, 1000 + i) for i, s in enumerate(win_s)]
    _settle(tl)
    tl.tick()
    tl.tick()
    w = tl.fdr_windows[1]
    back = {tl.akey(k): k[0] for k in tl.rows}
    return {back[ak][-6:] for ak in w["marked"]}, w, tl  # request ids end in the serve time: same across runs


def test_f6_missing_audit_labels_count_wrong():
    # a bucket whose audit answers were all skipped never marks, even above all of them
    cal_s = [float(x) for x in np.linspace(0.6, 0.9, 30)]
    marked, w, tl = _window_marks(cal_s, [None] * 30, [0.97] * 30, window=30)
    assert marked == set() and w["calibration"]["audit_response"] == 0.0 and w["calibration"]["n"] == 30
    assert w["calibration"]["below_floor"] == ["3"] and tl.fdr_status()["calibration"]["audit_response"] == 0.0
    # counting a skipped answer wrong never adds a mark over its having been labelled right
    g = np.random.default_rng(11)
    for trial in range(8):
        cal_s = [float(0.5 + 0.5 * g.random()) for _ in range(60)]
        truth = [bool(g.random() < 0.3 * (1.1 - s)) for s in cal_s]
        skip = g.random(60) < 0.3
        win_s = [float(0.5 + 0.5 * g.random()) for _ in range(60)]
        imputed, wi, _ = _window_marks(cal_s, [None if k else t for k, t in zip(skip, truth)], win_s, window=60)
        right, _, _ = _window_marks(cal_s, [False if k else t for k, t in zip(skip, truth)], win_s, window=60)
        assert imputed <= right, trial
        p = LB.fdr_pvalues(cal_s, [bool(k or t) for k, t in zip(skip, truth)], ["3"] * 60, win_s, ["3"] * 60)
        assert len(imputed) == int(LB.bh(p, 0.1).sum()), trial  # an expired answer counts wrong at its score
        assert wi["calibration"]["audit_response"] == pytest.approx(1 - skip.mean())


def test_f7_an_erased_answer_leaves_the_marks():
    from decisio.serve.fast_table_learn import MemoryPlace as Place

    clock = Clock()
    place = Place()
    s = LB.parse_settings({"audit_pi": 1.0, "cert_pi": 0, "margin_pi_max": 0, "tau_after": 10_000, "due_days": 1})
    tl = LB.TaskLabels("t", s, LB.new_secrets(), place, "r1", clock)
    tl.fdr = {"alpha": 0.1, "window": 30, "qtype": "choice"}
    cal = [_serve(tl, 0.7, i) for i in range(30)]
    _label(tl, cal, [False] * 30)
    _due(tl)
    win = [_serve(tl, 0.95, 100 + i) for i in range(30)]
    _settle(tl)
    tl.tick()
    tl.tick()
    marks = tl.fdr_marks()
    assert len(marks) == 60 and {m["request_id"] for m in marks} == set(cal + win)
    gone = win[0]
    tl.erase(request_id=gone)
    marks = tl.fdr_marks()
    assert len(marks) == 59 and gone not in {m["request_id"] for m in marks}  # at once
    clock.t += 100
    tl.tick()
    stored = [
        x for k in ("rows/", "events/") for n in place.names(k) for x in (place.get(n) or b"").decode().splitlines()
    ]
    assert tl.counts()["erasures"]["purged"] == 1
    assert not any(gone in x for x in stored if '"type": "erase"' not in x)  # only the erase event's opaque id
    assert all(isinstance(a, str) and a.startswith("ak:") for w in tl.fdr_windows for a in w["answers"])
