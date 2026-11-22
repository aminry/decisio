# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Labels from traffic, the log of one task (decisio.serve.labels).

L1  settings: no block, false or enabled false is off; true is the defaults; an unknown or out-of-range setting is
    refused
L2  the draw: the same on every replica for the same request id, independent of the answer, a pi of 0 or at least
    PI_MIN, and each slice drawn at its pi
L3  logging a served question: a request id, label_requested exactly when drawn, no request text kept, the plain
    readout and h only with full features (h only on drawn rows)
L4  labels: by request id, by a digest the client computes, by text (hashed, never kept), imports; duplicates,
    revisions, invalid answers, unknown references; written before the write returns, and read back after a restart
L5  the queue: the drawn questions still to label, oldest first, blind (no served answer, confidence, slice or
    probability); declines leave it,
    rows past due expire
L6  the label need: the cert slice's labelled answers and errors, expired and declined rows counted as errors
L7  a certificate never uses a label its version was fitted on
L8  the cert slice's region: all traffic until tau_after answers, then a new epoch with tau_min set
L9  past max_rows the oldest undrawn rows go and every drawn row stays, also when a log is read back
L10 a label write that fails to reach the place raises and leaves nothing acknowledged; a failed segment write keeps
    its rows for the next try; a write in progress does not hold up serving
L11 the need counts only the current region at one chance (exact pooling), says when a level will not certify, and is
    quick when it never will
L12 (slow) each slice's inclusion matches its pi within binomial noise on 200,000 rows, the margin slice too
L13 the need pools an earlier epoch's rows only when its region contained the current one (its tau_min at most the
    current), whatever its cert pi: a tau_min that falls leaves the older epoch out, one that rises keeps it; a
    label keeps the source its poster gave
L14 an undrawn answer keeps its served distribution only in the uniform sample (reservoir / max_rows), the same on
    every replica; a drawn one always does
L15 erasure on one server: by request id and by time, gone from memory at once and from every file (purged), and
    still gone when the log is read back
L16 the merge: many files become a few, nothing kept is lost, an event or row read twice counts once, and a log read
    back from the merged files has the same table
L17 settings after registration: a new epoch, draws at the new chances, rows keep their epoch; a region change
    moves tau_min; bad settings are refused
L18 roles: an evaluation draw (audit or cert) makes a row evaluation-only; fit rows are margin rows drawn by no
    evaluation slice, weighted 1 / (pi_margin (1 - p_eval)), and self labels on undrawn rows (no weight); the cohort is
    closed at now - due_days and refuses a cutoff with pending rows; the region reaches the certified threshold
L19 the per-day budget: the registration's chances assume tau_after answers a day; the region epoch sets them from
    the measured rate so the expected labels a day meet per_day; a new budget epoch only after a day and only when
    the rate has left 0.5x to 2x; per_day and explicit chances switch each other off

  uv run pytest -q tests/unit/test_labels.py
"""

import json

import numpy as np
import pytest

from decisio.serve import labels as LB
from decisio.serve.fast_table_learn import DirPlace, MemoryPlace

KEYS = ["billing", "tech", "sales"]


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _task(place=None, clock=None, **settings):
    s = LB.parse_settings(settings or True)
    return LB.TaskLabels("t", s, LB.new_secrets(), place or MemoryPlace(), "r1", clock or Clock())


def _serve(tl, text, p, rid=None, h=None):
    rid = rid or LB.request_id("r1")
    lp = np.log(np.asarray(p))
    return rid, tl.observe(rid, "q", text=text, keys=KEYS, served=p, lp=lp, h=h, certified=True,
                           version="v1", stack="fp", lang="en")  # fmt: skip


def test_l1_settings():
    assert LB.parse_settings(None) is None and LB.parse_settings(False) is None
    assert LB.parse_settings({"enabled": False}) is None
    d = LB.parse_settings(True)  # a budget of 20 a day; the registration's epoch assumes tau_after (200) answers a day
    assert d["per_day"] == 20 and {k: d[k] for k in LB.PI_KEYS} == {
        "audit_pi": 0.02, "cert_pi": 0.045, "margin_pi_min": 0.0, "margin_pi_max": 0.08}  # fmt: skip
    assert LB.parse_settings({"cert_pi": 0.3})["per_day"] is None  # chances given: no budget
    assert LB.parse_settings({"cert_pi": 0.3})["cert_pi"] == 0.3
    for bad in ({"nope": 1}, {"cert_pi": 0.01}, {"features": "all"}, {"margin_pi_min": 0.3, "margin_pi_max": 0.2},
                {"due_days": 0}, {"show_answer": True}, {"enabled": 1}, "on", {"per_day": 0},
                {"per_day": 10, "cert_pi": 0.2}, {"shares": {"audit": 0.5}}):  # fmt: skip
        with pytest.raises(LB.LabelsError):
            LB.parse_settings(bad)


def test_l2_the_draw_is_reproducible_answer_free_and_at_its_pi():
    salt = LB.new_secrets()["salt"]
    epoch = {"epoch": 0, "tau_min": 0.5, "settings": LB.parse_settings({"audit_pi": 0.1, "cert_pi": 0.3})}
    a = LB.draw(salt, "r1-X", "q", 0.9, 0.5, epoch)
    assert a == LB.draw(salt, "r1-X", "q", 0.9, 0.5, epoch)  # any replica draws the same from the id
    # the uniforms are the id's alone: a different answer (confidence, margin) changes pi, never u
    b = LB.draw(salt, "r1-X", "q", 0.2, 0.01, epoch)
    assert all(a[s]["u"] == b[s]["u"] for s in LB.SLICES) and b["cert"]["pi"] == 0.0
    assert LB.floor_pi(0.019) == 0.0 and LB.floor_pi(0.02) == 0.02
    rng = np.random.default_rng(0)
    n = 20_000
    hits = {s: 0 for s in LB.SLICES}
    for i in range(n):
        d = LB.draw(salt, f"r1-{i}", "q", 0.9, float(rng.uniform(0.5, 1.0)), epoch)
        for s in LB.SLICES:
            hits[s] += d[s]["in"]
    for s, pi in (("audit", 0.1), ("cert", 0.3)):
        assert abs(hits[s] / n - pi) < 4 * (pi * (1 - pi) / n) ** 0.5, (s, hits[s] / n)


def test_l3_logging_keeps_no_text_and_draws():
    tl = _task(audit_pi=1.0)  # every answer drawn into the audit slice
    rid, add = _serve(tl, "My card was charged twice", [0.7, 0.2, 0.1], h=np.ones(4))
    assert add == {"request_id": rid, "label_requested": True}
    row = tl.rows[(rid, "q")]
    assert "My card" not in json.dumps(row) and row["top"] == "billing" and row["lang"] == "en"
    assert np.allclose(LB.unarr(row["lp"]), np.log([0.7, 0.2, 0.1])) and LB.unarr(row["h"]).shape == (4,)
    quiet = _task(audit_pi=0, cert_pi=0, margin_pi_max=0)
    rid2, add2 = _serve(quiet, "x", [0.7, 0.2, 0.1], h=np.ones(4))
    assert add2 == {"request_id": rid2} and quiet.rows[(rid2, "q")]["h"] is None  # h only on drawn rows
    minimal = _task(features="minimal", audit_pi=1.0)
    rid3, _ = _serve(minimal, "x", [0.7, 0.2, 0.1], h=np.ones(4))
    assert not {"ps", "lp", "h"} & set(minimal.rows[(rid3, "q")])


def test_l4_labels_every_reference_and_a_restart(tmp_path):
    clock = Clock()
    place = DirPlace(tmp_path / "labels")
    tl = _task(place=place, clock=clock, audit_pi=1.0)
    rid, _ = _serve(tl, "My card was charged twice", [0.7, 0.2, 0.1])
    rid_b, _ = _serve(tl, "The app crashes", [0.2, 0.7, 0.1])
    tl.flush(force=True)
    digest = LB.hk1(tl.secret["digest_key"], "The   app crashes ")  # the client's digest of its own text
    out = tl.post([
        {"request_id": rid, "answer": "billing"},
        {"digest": digest, "answer": "tech"},
        {"text": "Something new entirely", "answer": "sales"},
        {"request_id": "r1-NOPE", "answer": "tech"},
        {"request_id": rid, "answer": "refunds"},
    ])  # fmt: skip
    st = [r["status"] for r in out["results"]]
    assert st == ["accepted", "accepted", "accepted", "unknown_ref", "invalid_answer"]
    assert out["results"][1]["request_id"] == rid_b and out["results"][2]["source"] == "import"
    assert "Something new" not in json.dumps(tl.events)  # a text label keeps the digest, never the text
    assert tl.post([{"request_id": rid, "answer": "billing"}])["results"][0]["status"] == "duplicate"
    assert tl.post([{"request_id": rid, "answer": "sales"}])["results"][0]["status"] == "revised"
    with pytest.raises(LB.LabelsError):
        tl.post([])
    # every acknowledged label is on the place: a new log over the same place reads it all back
    back = LB.TaskLabels("t", tl.settings, tl.secret, DirPlace(tmp_path / "labels"), "r1", clock)
    back.sync()
    labelled = {(r["request_id"], r.get("answer")) for r in back.table() if r["state"] == "labelled"}
    assert labelled == {(rid, "sales"), (rid_b, "tech")}


def test_l5_the_queue_declines_and_expiry():
    clock = Clock()
    tl = _task(clock=clock, audit_pi=1.0)
    rids = []
    for i in range(3):
        clock.t += 1
        rids.append(_serve(tl, f"text {i}", [0.6, 0.3, 0.1])[0])
    q = tl.queue(limit=2)
    assert [it["request_id"] for it in q["items"]] == rids[:2] and q["pending"] == 3
    assert set(q["items"][0]) == {"request_id", "question", "asked_at", "due_by"}  # blind: no answer, no confidence
    tl.decline([rids[0]])
    assert [it["request_id"] for it in tl.queue()["items"]] == rids[1:]
    clock.t += 15 * 86400  # past the default 14 days
    assert tl.queue()["pending"] == 0 and tl.counts()["states"]["expired"] == 2


def test_l6_the_need_counts_missing_labels_as_errors():
    from decisio.readout import certify

    clock = Clock()
    tl = _task(clock=clock, audit_pi=0, cert_pi=1.0, margin_pi_max=0, tau_after=10_000)
    rids = [_serve(tl, f"t{i}", [0.8, 0.1, 0.1])[0] for i in range(60)]
    tl.post([{"request_id": r, "answer": "billing"} for r in rids[:58]] + [{"request_id": rids[58], "answer": "tech"}])
    tl.decline([rids[59]])
    need = {n["alpha"]: n for n in tl.need()}
    a = need[0.05]
    assert a["region_labels"] == 60 and a["region_errors"] == 2  # one wrong answer, one declined
    assert a["upper_bound"] == pytest.approx(float(certify.upper_bound(2, 60, 0.1)), abs=1e-5)
    m = 60 + a["more_if_no_error"]
    assert certify.upper_bound(2, m, 0.1) <= 0.05 < certify.upper_bound(2, m - 1, 0.1)


def test_l7_no_certificate_on_its_own_fit_labels():
    tl = _task(audit_pi=1.0)
    rids = [_serve(tl, f"t{i}", [0.8, 0.1, 0.1])[0] for i in range(4)]
    tl.record_fit_use("v2", [(rids[0], "q"), (rids[1], "q")])
    fit = tl.fit_use(["v2"])
    LB.assert_disjoint(fit, [(rids[2], "q"), (rids[3], "q")])
    with pytest.raises(LB.LabelsError, match="fitted on"):
        LB.assert_disjoint(fit, [(rids[1], "q"), (rids[2], "q")])


def test_l8_the_region_is_all_traffic_then_the_top_share():
    tl = _task(audit_pi=0, cert_pi=1.0, margin_pi_max=0, tau_after=50, region=0.5)
    confs = np.linspace(0.4, 0.99, 50)
    for c in confs:
        _serve(tl, f"t{c}", [c, (1 - c) / 2, (1 - c) / 2])
    assert len(tl.epochs) == 1  # the request path never writes: the labels thread starts the epoch
    tl.tick()
    assert len(tl.epochs) == 2 and tl.epoch["tau_min"] == pytest.approx(float(np.quantile(confs, 0.5)))
    low = _serve(tl, "low", [0.41, 0.3, 0.29])[0]
    assert not tl.rows[(low, "q")]["draws"]["cert"]["in"] and tl.rows[(low, "q")]["epoch"] == 1


def test_l9_max_rows_drops_the_oldest_undrawn_rows(tmp_path):
    clock = Clock()
    tl = _task(place=DirPlace(tmp_path / "labels"), clock=clock, audit_pi=0.3, max_rows=10)
    drawn, undrawn = [], []
    for i in range(60):
        clock.t += 1
        rid, add = _serve(tl, f"t{i}", [0.6, 0.3, 0.1])
        (drawn if add.get("label_requested") else undrawn).append(rid)
    tl.flush(force=True)
    assert len(drawn) >= 11  # 60 at 0.3: about 18, more than max_rows
    kept_undrawn = [r for r in undrawn if (r, "q") in tl.rows]
    assert all((r, "q") in tl.rows for r in drawn)  # a drawn row's label may still come
    assert kept_undrawn == undrawn[len(undrawn) - len(kept_undrawn) :] and len(kept_undrawn) <= 1  # the newest only
    back = LB.TaskLabels("t", tl.settings, tl.secret, DirPlace(tmp_path / "labels"), "r1", clock)
    back.sync()
    assert {k[0] for k in back.rows} == set(drawn)  # read back: the same rule, every undrawn row over the cap goes


class FailingPlace(MemoryPlace):
    def __init__(self):
        super().__init__()
        self.fail = False

    def put(self, name, data):
        if self.fail:
            raise OSError("disk full")
        super().put(name, data)


def test_l10_nothing_unsaved_is_acknowledged_and_serving_does_not_wait():
    import threading

    place = FailingPlace()
    tl = _task(place=place, audit_pi=1.0)
    rid, _ = _serve(tl, "a", [0.7, 0.2, 0.1])
    place.fail = True
    with pytest.raises(OSError):
        tl.post([{"request_id": rid, "answer": "billing"}])
    assert tl.events == [] and tl.state_of((rid, "q"), tl.latest, tl.clock()) == "pending"
    with pytest.raises(OSError):
        tl.flush(force=True)
    assert len(tl._pending_rows) == 1  # kept for the next try
    place.fail = False
    assert tl.flush(force=True) == 1
    assert tl.post([{"request_id": rid, "answer": "billing"}])["results"][0]["status"] == "accepted"

    class SlowPlace(MemoryPlace):
        def __init__(self):
            super().__init__()
            self.entered, self.release = threading.Event(), threading.Event()

        def put(self, name, data):
            if name.startswith("events/"):
                self.entered.set()
                self.release.wait(5)
            super().put(name, data)

    slow = SlowPlace()
    tl = _task(place=slow, audit_pi=1.0)
    rid, _ = _serve(tl, "a", [0.7, 0.2, 0.1])
    th = threading.Thread(target=tl.post, args=([{"request_id": rid, "answer": "billing"}],))
    th.start()
    assert slow.entered.wait(5)
    done = threading.Event()
    threading.Thread(target=lambda: (_serve(tl, "b", [0.6, 0.3, 0.1]), done.set())).start()
    assert done.wait(2), "serving waited on a label write"
    slow.release.set()
    th.join(5)
    assert tl.state_of((rid, "q"), tl.latest, tl.clock()) == "labelled"


def test_l11_the_need_is_the_current_region_at_one_chance():
    tl = _task(audit_pi=0, cert_pi=1.0, margin_pi_max=0, tau_after=20, region=0.5)
    low = [_serve(tl, f"low{i}", [0.5, 0.3, 0.2])[0] for i in range(10)]  # epoch 0: all traffic is the region
    high = [_serve(tl, f"high{i}", [0.9, 0.05, 0.05])[0] for i in range(10)]
    tl.tick()
    assert tl.epoch["epoch"] == 1 and tl.epoch["tau_min"] > 0.5
    tl.post([{"request_id": r, "answer": "billing"} for r in low + high])
    a = {n["alpha"]: n for n in tl.need()}[0.05]
    assert a["region_labels"] == 10 and a["pooling"] == "exact"  # the low rows are below the current region
    bad = _task(audit_pi=0, cert_pi=1.0, margin_pi_max=0, tau_after=10_000)
    rids = [_serve(bad, f"t{i}", [0.8, 0.1, 0.1])[0] for i in range(20)]
    bad.post([{"request_id": r, "answer": "tech"} for r in rids])  # every answer wrong
    import time as _time

    t0 = _time.perf_counter()
    need = bad.need()
    assert _time.perf_counter() - t0 < 1.0
    assert all("reason" in n and n["region_errors"] == 20 for n in need)
    assert need[0]["more_if_no_error"] > 300  # finite: enough right answers would bring the bound under alpha


@pytest.mark.slow
def test_l12_inclusion_matches_pi_on_200000_rows():
    salt = LB.new_secrets()["salt"]
    s = LB.parse_settings({"audit_pi": 0.05, "cert_pi": 0.2})
    epoch = {"epoch": 1, "tau_min": 0.7, "settings": s}
    rng = np.random.default_rng(1)
    n = 200_000
    confs, margins = rng.uniform(0.34, 1.0, n), rng.uniform(0.0, 0.6, n)
    want = {x: 0.0 for x in LB.SLICES}
    got = {x: 0 for x in LB.SLICES}
    var = {x: 0.0 for x in LB.SLICES}
    for i in range(n):
        d = LB.draw(salt, f"r-{i}", "q", float(confs[i]), float(margins[i]), epoch)
        for x in LB.SLICES:
            pi = d[x]["pi"]
            want[x] += pi
            var[x] += pi * (1 - pi)
            got[x] += d[x]["in"]
    for x in LB.SLICES:
        assert abs(got[x] - want[x]) < 4 * var[x] ** 0.5, (x, got[x], want[x])


def test_l13_pooling_by_region_containment_and_the_real_source():
    tl = _task(audit_pi=0, cert_pi=1.0, margin_pi_max=0, tau_after=10_000)

    def epoch(tau, cert_pi):
        last = tl.epochs[-1]
        tl.epochs = [*tl.epochs, {"epoch": len(tl.epochs), "start": tl.clock(), "tau_min": tau, "tau_set": True,
                                  "settings": {**last["settings"], "cert_pi": cert_pi}, "why": "test"}]  # fmt: skip

    def serve_label(n, conf):
        rids = [_serve(tl, f"e{tl.epoch['epoch']}-{conf}-{i}", [conf, (1 - conf) / 2, (1 - conf) / 2])[0]
                for i in range(n)]  # fmt: skip
        tl.post([{"request_id": r, "answer": "billing"} for r in rids])
        return rids

    epoch(0.8, 1.0)
    serve_label(5, 0.9)  # epoch 1: region 0.8 and up
    epoch(0.6, 0.5)  # tau_min falls: epoch 1 drew nothing in [0.6, 0.8)
    drawn2 = [r for r in serve_label(40, 0.7) + serve_label(40, 0.9) if tl.rows[(r, "q")]["draws"]["cert"]["in"]]
    a = {n["alpha"]: n for n in tl.need()}[0.05]
    assert a["region_labels"] == len(drawn2) and a["pooling"] == "exact"  # epoch 1 left out
    epoch(0.7, 1.0)  # tau_min rises: epoch 2's region (0.6 and up) held it, epoch 1's (0.8) did not
    serve_label(3, 0.75)
    a = {n["alpha"]: n for n in tl.need()}[0.05]
    assert a["region_labels"] == len(drawn2) + 3  # epoch 2's rows at 0.7 and up (all of them here), epoch 3's three
    assert a["region_pending"] == 0
    rid = _serve(tl, "pending one", [0.9, 0.05, 0.05])[0]
    assert {n["alpha"]: n for n in tl.need()}[0.05]["region_pending"] == 1 and (rid, "q") in tl.rows
    tl.post([{"request_id": rid, "answer": "billing"}])
    assert tl.latest[(rid, "q")]["source"] == "user"  # the poster's own source, not rewritten to "queue"


def test_l14_the_served_distribution_of_undrawn_answers_is_a_uniform_sample():
    tl = _task(audit_pi=0.1, max_rows=10_000, reservoir=1_000)  # a sample at 0.1
    n = 4_000
    rids = [_serve(tl, f"t{i}", [0.6, 0.3, 0.1])[0] for i in range(n)]
    rows = [tl.rows[(r, "q")] for r in rids]
    undrawn = [r for r in rows if not r["drawn"]]
    kept = [r for r in undrawn if r["ps"] is not None]
    assert all(r["ps"] is not None for r in rows if r["drawn"])
    assert all((r["ps"] is not None) == r["sample"]["in"] for r in undrawn) and undrawn[0]["sample"]["pi"] == 0.1
    assert abs(len(kept) / len(undrawn) - 0.1) < 4 * (0.09 / len(undrawn)) ** 0.5
    other = LB.TaskLabels("t", tl.settings, tl.secret, MemoryPlace(), "r2", tl.clock)
    assert other._sample(rids[0], "q") == tl.rows[(rids[0], "q")]["sample"]  # any replica draws the same


def _files_text(place) -> str:
    """Every row and label on the place, as text (an erase event keeps the id it erased, an opaque id, and nothing
    else of the request)."""
    files = [place.get(n) or b"" for k in ("rows/", "events/") for n in place.names(k)]
    lines = [x for data in files for x in data.decode().splitlines()]
    return "\n".join(x for x in lines if '"type": "erase"' not in x)


def test_l15_erasure_on_one_server(tmp_path):
    clock = Clock()
    place = DirPlace(tmp_path / "labels")
    tl = _task(place=place, clock=clock, audit_pi=1.0)
    old = [_serve(tl, f"old {i}", [0.7, 0.2, 0.1])[0] for i in range(3)]
    clock.t += 100
    keep, _ = _serve(tl, "keep", [0.7, 0.2, 0.1])
    gone, _ = _serve(tl, "gone", [0.7, 0.2, 0.1])
    tl.flush(force=True)
    tl.post([{"request_id": r, "answer": "billing"} for r in (*old, keep, gone)])
    out = tl.erase(request_id=gone)
    assert out["state"] == "erased" and out["matched"] == {"answers": 1, "labels": 1}
    assert (gone, "q") not in tl.rows and gone in _files_text(place)  # gone from every view; the bytes at the merge
    tl.tick()  # the labels thread's next turn: one server purges at once
    assert gone not in _files_text(place) and tl.counts()["erasures"] == {"erased": 1, "purged": 1}
    out = tl.erase(before=clock.t - 50)
    assert out["matched"]["answers"] == 3
    tl.tick()
    assert all(r not in _files_text(place) for r in old)
    assert {r["request_id"] for r in tl.table()} == {keep}
    assert tl.counts()["erasures"] == {"erased": 2, "purged": 2}
    late, _ = _serve(tl, "late", [0.7, 0.2, 0.1])  # served after: kept
    tl.flush(force=True)
    back = LB.TaskLabels("t", tl.registered, tl.secret, DirPlace(tmp_path / "labels"), "r1", clock)
    back.sync()
    assert {r["request_id"] for r in back.table()} == {keep, late}
    assert back.table()[0]["state"] == "labelled"
    with pytest.raises(LB.LabelsError):
        tl.erase()
    with pytest.raises(LB.LabelsError, match="not a request id"):
        tl.erase(request_id="queue")


def test_l16_the_merge_keeps_everything_kept(monkeypatch, tmp_path):
    clock = Clock()
    place = DirPlace(tmp_path / "labels")
    tl = _task(place=place, clock=clock, audit_pi=0.5)
    rids = []
    for i in range(40):
        clock.t += 1
        rids.append(_serve(tl, f"t{i}", [0.6, 0.3, 0.1])[0])
        if i % 4 == 3:
            tl.flush(force=True)
    tl.post([{"request_id": r, "answer": "tech"} for r in rids[:10]])
    tl.post([{"request_id": rids[0], "answer": "billing"}])  # a revision
    monkeypatch.setattr(LB, "COMPACT_FILES", 5)
    before = tl.table()
    n_files = len(place.names("rows/")) + len(place.names("events/"))
    tl.tick()
    after = len(place.names("rows/")) + len(place.names("events/"))
    assert tl.merges == 1 and n_files >= 12 and after == 2
    assert tl.table() == before and tl.revisions == 1
    back = LB.TaskLabels("t", tl.registered, tl.secret, DirPlace(tmp_path / "labels"), "r1", clock)
    back.sync()
    assert back.table() == before and back.revisions == 1
    # a merged file read twice (a second merger's copy) changes nothing
    for name in place.names("rows/") + place.names("events/"):
        place.put(name.replace("-merged.jsonl", "-merged2.jsonl"), place.get(name))
    again = LB.TaskLabels("t", tl.registered, tl.secret, DirPlace(tmp_path / "labels"), "r1", clock)
    again.sync()
    assert again.table() == before and again.revisions == 1 and len(again.events) == len(back.events)


def test_l17_settings_after_registration():
    tl = _task(audit_pi=0, cert_pi=1.0, margin_pi_max=0, tau_after=20, region=0.5)
    for i in range(20):
        _serve(tl, f"a{i}", [0.5 + i / 50, 0.3, 0.2])
    tl.tick()
    tau1 = tl.epoch["tau_min"]
    assert tl.epoch["epoch"] == 1 and tau1 > 0
    e = tl.set_settings({"audit_pi": 1.0})
    assert e["epoch"] == 2 and tl.settings["audit_pi"] == 1.0 and e["tau_min"] == tau1
    rid, add = _serve(tl, "after", [0.6, 0.3, 0.1])
    assert add["label_requested"] and tl.rows[(rid, "q")]["epoch"] == 2
    assert all(tl.rows[k]["epoch"] < 2 for k in tl.rows if k[0] != rid)  # the rows before keep their epoch
    e = tl.set_settings({"region": 0.25})
    assert e["tau_min"] > tau1  # a smaller region: a higher lowest confidence
    for bad in ({}, {"cert_pi": 0.01}, {"nope": 1}, {"enabled": False}, "x"):
        with pytest.raises(LB.LabelsError):
            tl.set_settings(bad)
    assert tl.epoch["epoch"] == 3 and tl.registered["audit_pi"] == 0


def test_l18_roles_the_cohort_and_the_region():
    clock = Clock()
    tl = _task(clock=clock, audit_pi=0.3, cert_pi=0.3, margin_pi_min=0.5, margin_pi_max=0.5, tau_after=10_000)
    rids = [_serve(tl, f"t{i}", [0.6, 0.3, 0.1])[0] for i in range(400)]
    tl.post([{"request_id": r, "answer": "billing"} for r in rids])
    rows = {r: tl.rows[(r, "q")]["draws"] for r in rids}
    ev = {r["request_id"] for r in tl.eval_rows()}
    fit = {r["request_id"]: r for r in tl.fit_rows()}
    assert ev == {r for r, d in rows.items() if d["audit"]["in"] or d["cert"]["in"]}
    assert not ev & set(fit)  # an evaluation draw makes a row evaluation-only
    assert set(fit) == set(rids) - ev  # margin rows (weighted) and self labels on undrawn rows
    both = [r for r, d in rows.items() if d["margin"]["in"] and r in ev]
    assert both  # rows drawn by margin and by an evaluation slice exist, and none is a fit row
    w = next(x for x in fit.values() if x["selection"] == "slice")["weight"]
    assert w == pytest.approx(1 / (0.5 * (1 - (1 - (1 - 0.3) * (1 - 0.3)))))
    assert all(x["weight"] is None for x in fit.values() if x["selection"] == "self")
    LB.assert_disjoint([(r, "q") for r in fit], [(r, "q") for r in ev])
    clock.t += 1
    late = next(
        r
        for r in (_serve(tl, f"late{i}", [0.6, 0.3, 0.1])[0] for i in range(50))
        if tl.rows[(r, "q")]["draws"]["audit"]["in"]
    )
    with pytest.raises(LB.LabelsError, match="pending"):  # a drawn row still pending at that cutoff: the cohort waits
        tl.cohort(cutoff=clock.t)
    assert tl.cohort() == []  # nothing is due yet: every row is younger than due_days
    clock.t += 15 * 86400
    co = tl.cohort()
    assert {r["request_id"] for r in co} == {r["request_id"] for r in tl.eval_rows()} and late in {
        r["request_id"] for r in co
    }
    assert all(not r["strict_error"] for r in co if r["request_id"] in ev)
    assert next(r for r in co if r["request_id"] == late)["strict_error"]  # expired: an error under the strict rule
    reg = _task(audit_pi=0, cert_pi=1.0, margin_pi_max=0, tau_after=50)
    for i in range(50):
        _serve(reg, f"r{i}", [0.5 + i / 100, 0.3, 0.2])
    reg.cert_threshold = 0.52  # the task's certified threshold sits below the top-80% cut
    reg.tick()
    assert reg.epoch["tau_min"] == pytest.approx(0.52)


def test_l19_the_per_day_budget():
    clock = Clock()
    tl = _task(clock=clock)  # per_day 20
    rng = np.random.default_rng(3)
    for i in range(200):  # 200 answers over 10 days: 20 a day
        clock.t += 72 * 60
        c = float(rng.uniform(0.45, 0.6))  # near ties: the margin slice can spend its share
        _serve(tl, f"t{i}", [c, 0.99 - c, 0.01])
    assert tl.epoch["epoch"] == 0 and tl.settings["cert_pi"] == 0.045
    tl.tick()
    e = tl.epoch
    assert e["epoch"] == 1 and e["budget"]["rate_per_day"] == pytest.approx(20, rel=0.05)
    assert e["settings"]["audit_pi"] == pytest.approx(20 * 0.15 / e["budget"]["rate_per_day"], rel=1e-3)
    assert e["budget"]["expected_per_day"] == pytest.approx(20, abs=0.5)  # no chance clamped: the budget is spent
    assert tl.budget()["per_day"] == 20
    clock.t += 3600
    for i in range(200):  # a burst: the rate triples, but within a day of the epoch nothing moves
        _serve(tl, f"b{i}", [0.9, 0.06, 0.04])
    tl.tick()
    assert tl.epoch["epoch"] == 1
    clock.t += 86_400
    for i in range(400):
        _serve(tl, f"c{i}", [0.9, 0.06, 0.04])
    tl.tick()
    assert tl.epoch["epoch"] == 2 and "rate moved" in tl.epoch["why"]
    assert tl.epoch["settings"]["audit_pi"] < e["settings"]["audit_pi"]  # more traffic: smaller chances
    assert tl.epoch["tau_min"] == e["tau_min"]
    up = tl.set_settings({"per_day": 80})
    assert up["settings"]["per_day"] == 80 and up["settings"]["audit_pi"] > tl.epochs[-2]["settings"]["audit_pi"]
    off = tl.set_settings({"cert_pi": 0.3})
    assert off["settings"]["per_day"] is None and off["settings"]["cert_pi"] == 0.3 and "budget" not in off
