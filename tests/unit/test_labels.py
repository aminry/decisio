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
L7  a certificate never uses a label its version was fitted on; the fit record holds keyed digests, never a request id
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
    still gone when the log is read back; no request id it erased is on the place, erase events included
L15b an erase event in an earlier version's form (with the request id) is honoured, and a merge rewrites it as its keyed
     digest
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
L20 the typical slice: levels from the reservoir rank a typical input above an outlier; with the student present a
    budget task starts the slice (an epoch), draws it at its level, ends it after typical_labels (its share back to
    margin), and its labels are fit labels weighted by the chance of either fit slice; without the student it is off;
    the student reads a bounded prefix of the input
L21 the overview: the response rate, the audit accuracy and its strict form with their intervals, the last label,
    and per language the counts and the labels still needed; days to the need at the region's label rate
L22 ttl_days: answers and labels older than it are erased by the labels thread, at most once an hour, through the
    erase path; it is at least due_days
L23 a task certified at alpha 0.01 or below, in budget mode with the default shares, moves to audit 0.10, cert 0.80,
    margin 0.10, typical 0 (an epoch); given shares and lenient alphas are kept
L24 a task certified at alpha 0.01 or below needs labels on at least 5% of its answers: explicit chances that
    cannot reach it are refused with the cert_pi that would, a budget gets a warning at registration, and the
    overview reports the share drawn and warns below it
L25 status.budget gives the labels a day expected at the measured rate and flags spending above per_day (the 0.02
    floor on heavy traffic); the rate counts an answer once, also when it is read again after max_rows dropped it
L26 detection at a strict alpha: the claim holds when the split's q (audit or cert, in the region) reaches the
    validated reference (10/80/10/0 at 5%); otherwise the exact warning and what would reach it; never for a
    lenient alpha
L27 status reads never hold up serving: observe() stays fast while another thread polls the queue, the overview,
    the need and the budget (each scan reads a snapshot outside the log's lock)

  uv run pytest -q tests/unit/test_labels.py
"""

import json
import zlib

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
        "audit_pi": 0.02, "cert_pi": 0.045, "margin_pi_min": 0.0, "margin_pi_max": 0.08,
        "typical_pi_max": 0.0}  # fmt: skip
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
    assert set(out["need"]) == {0.05, 0.10, 0.01, 0.005}  # the label need comes with every write
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
    assert tl.fit_use(["v2"]) == {tl.akey((rids[0], "q")), tl.akey((rids[1], "q"))}
    tl.assert_disjoint_from_fits(["v2"], [(rids[2], "q"), (rids[3], "q")])
    with pytest.raises(LB.LabelsError, match="fitted on"):
        tl.assert_disjoint_from_fits(["v2"], [(rids[1], "q"), (rids[2], "q")])
    rec = next(e for e in tl.events if e["type"] == "fit_use")
    assert not any(r in json.dumps(rec) for r in rids)  # the record holds keyed digests, never a request id
    tl.events.append({"type": "fit_use", "version": "v1", "labels": [[rids[3], "q"]]})  # an earlier version's record
    assert tl.fit_use(["v1"]) == {tl.akey((rids[3], "q"))}


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
    gone = undrawn[0]  # dropped past max_rows: the label says so
    assert tl.post([{"request_id": gone, "answer": "billing"}])["results"][0]["status"] == "expired_ref"
    assert tl.post([{"request_id": "r1-" + LB.ulid(clock.t + 5), "answer": "billing"}])["results"][0]["status"] == (
        "unknown_ref")  # fmt: skip
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
    s = LB.parse_settings({"audit_pi": 0.05, "cert_pi": 0.2, "typical_pi_max": 0.4})
    epoch = {"epoch": 1, "tau_min": 0.7, "settings": s}
    rng = np.random.default_rng(1)
    n = 200_000
    confs, margins = rng.uniform(0.34, 1.0, n), rng.uniform(0.0, 0.6, n)
    levels = rng.integers(0, LB.TYPICAL_LEVELS + 1, n)  # 0: no level yet (the typical slice draws nothing)
    want = {x: 0.0 for x in LB.SLICES}
    got = {x: 0 for x in LB.SLICES}
    var = {x: 0.0 for x in LB.SLICES}
    for i in range(n):
        d = LB.draw(salt, f"r-{i}", "q", float(confs[i]), float(margins[i]), epoch, typical=int(levels[i]) or None)
        for x in LB.SLICES:
            pi = d[x]["pi"]
            want[x] += pi
            var[x] += pi * (1 - pi)
            got[x] += d[x]["in"]
    for x in LB.SLICES:
        assert want[x] > 0, x  # every slice draws in this test, the typical one too
        assert abs(got[x] - want[x]) <= 4 * var[x] ** 0.5, (x, got[x], want[x])


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
    """Every row and event on the place, as text, erase events included (they keep a keyed digest of the request id
    they erased, never the id)."""
    files = [place.get(n) or b"" for k in ("rows/", "events/") for n in place.names(k)]
    return "\n".join(x for data in files for x in data.decode().splitlines())


def test_l15b_an_erase_event_from_before_keeps_only_its_digest_after_a_merge(tmp_path):
    clock = Clock()
    place = DirPlace(tmp_path / "labels")
    tl = _task(place=place, clock=clock, audit_pi=1.0)
    keep, _ = _serve(tl, "keep", [0.7, 0.2, 0.1])
    gone, _ = _serve(tl, "gone", [0.7, 0.2, 0.1])
    tl.flush(force=True)
    tl.post([{"request_id": r, "answer": "billing"} for r in (keep, gone)])
    old = {
        "type": "erase",
        "request_id": gone,
        "before": None,
        "received": clock.t,
        "replica": "r1",
    }  # as earlier versions wrote it
    with tl._write:
        tl._write_events([old])
    assert '"request_id": "' + gone in _files_text(place)  # written as an older version wrote it
    back = LB.TaskLabels("t", tl.registered, tl.secret, DirPlace(tmp_path / "labels"), "r1", clock)
    back.sync()
    assert {r["request_id"] for r in back.table()} == {keep}  # honoured: the answer and its label are erased
    erase = next(e for e in back.events if e["type"] == "erase")
    assert erase["rid_key"] == back.rkey(gone) and "request_id" not in erase
    back.tick()  # the merge rewrites the log: the erase event in its new form, the request id gone
    assert gone not in _files_text(place) and back.counts()["erasures"] == {"erased": 1, "purged": 1}
    again = LB.TaskLabels("t", tl.registered, tl.secret, DirPlace(tmp_path / "labels"), "r1", clock)
    again.sync()
    assert {r["request_id"] for r in again.table()} == {keep}
    # one purged by the earlier version already (nothing waits for a purge): the next turn merges it away all the same
    late, _ = _serve(again, "late", [0.7, 0.2, 0.1])
    again.flush(force=True)
    with again._write:
        again._write_events([{"type": "erase", "request_id": late, "before": None, "received": clock.t,
                              "replica": "r1", "id": "01OLDERASE0000000000000000"}])  # fmt: skip
        again._write_events([{"type": "purged", "of": ["01OLDERASE0000000000000000"], "received": clock.t,
                              "replica": "r1"}])  # fmt: skip
    assert late in _files_text(place)
    last = LB.TaskLabels("t", tl.registered, tl.secret, DirPlace(tmp_path / "labels"), "r1", clock)
    last.sync()
    assert last.unpurged(ripe=True) == [] and last._legacy_erase
    last.tick()
    assert late not in _files_text(place) and not last._legacy_erase


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


def _fake_embed(dim=16, seed=0):
    """Texts "c<k>..." near centre k of a few clusters, "x..." far from all: a stand-in for the static student."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(3, dim))

    def embed(text):
        r = np.random.default_rng(zlib.crc32(text.encode()))
        v = (rng.normal(size=dim) * 3 if text.startswith("x") else centres[int(text[1]) % 3]) + r.normal(size=dim) * 0.3
        return v / np.linalg.norm(v)

    return embed


def test_l20_the_typical_slice():
    typ = LB.Typicality(1)
    embed = _fake_embed()
    for i in range(200):  # most inputs like cluster 0, some like cluster 1, a few unlike anything
        typ.add(embed(f"c0-{i}" if i < 160 else f"c1-{i}" if i < 190 else f"x-{i}"))
    typ.refresh()
    assert typ.level(embed("c0-new")) >= 3 and typ.level(embed("x-outlier")) == 1

    clock = Clock()
    tl = _task(clock=clock, typical_labels=5, per_day=400)  # a budget the typical slice can draw from
    assert tl.typical_status()["state"].startswith("off: no static student")
    tl.embed = embed
    tl.tick()
    assert tl.epoch["why"] == "the typical slice started" and tl.settings["typical_pi_max"] > 0
    assert tl.settings["margin_pi_max"] <= tl.epochs[0]["settings"]["margin_pi_max"]  # margin gave up its share
    rids = [_serve(tl, f"c{i % 3}-{i}", [0.6, 0.3, 0.1])[0] for i in range(80)]  # the reservoir fills
    tl.tick()
    assert tl.typical_status()["state"] == "on"
    rids += [_serve(tl, f"c{i % 3}-t{i}", [0.6, 0.3, 0.1])[0] for i in range(400)]
    typ_rows = [r for r in rids if tl.rows[(r, "q")]["draws"]["typical"]["in"]]
    assert len(typ_rows) >= 20 and all(tl.rows[(r, "q")]["draws"]["typical"]["pi"] > 0 for r in typ_rows)
    tl.post([{"request_id": r, "answer": "billing"} for r in typ_rows[:5]])
    fit = {x["request_id"]: x for x in tl.fit_rows()}
    for r in typ_rows[:5]:
        d = tl.rows[(r, "q")]["draws"]
        if d["audit"]["in"] or d["cert"]["in"]:
            continue  # evaluation first
        p_fit = 1 - (1 - d["margin"]["pi"]) * (1 - d["typical"]["pi"])
        assert fit[r]["weight"] == pytest.approx(1 / (p_fit * (1 - tl.p_eval(tl.rows[(r, "q")]))))
    tl.tick()
    assert tl.epoch["why"] == "the typical slice ended" and tl.settings["typical_pi_max"] == 0.0
    assert tl.typical_status()["state"] == "ended" and not tl.typical_wanted()
    late, _ = _serve(tl, "c1-late", [0.6, 0.3, 0.1])
    assert tl.rows[(late, "q")]["draws"]["typical"]["pi"] == 0.0
    seen = []
    capped = _task(typical_pi_max=0.3)
    capped.embed = lambda t: (seen.append(len(t)), embed("c0-" + t[:5]))[1]
    _serve(capped, "c0-" + "x" * 50_000, [0.6, 0.3, 0.1])
    assert seen == [LB.TYPICAL_TEXT]  # the student reads a bounded prefix whatever the state's length
    assert LB.script_class("я" * 600 + "a" * 50_000) == "cyrillic"  # the script class too: its first 500 characters


def test_l21_the_overview():
    clock = Clock()
    tl = _task(clock=clock, audit_pi=1.0, cert_pi=1.0, margin_pi_max=0, tau_after=10_000)
    de = [_serve(tl, f"de{i}", [0.8, 0.1, 0.1])[0] for i in range(10)]
    for r in de:
        tl.rows[(r, "q")]["lang"] = "de"
    en = [_serve(tl, f"en{i}", [0.8, 0.1, 0.1])[0] for i in range(10)]
    tl.post([{"request_id": r, "answer": "billing"} for r in de[:8]] + [{"request_id": en[0], "answer": "tech"}])
    tl.decline([de[8]])
    o = tl.overview()
    assert o["response_rate"] == pytest.approx(9 / 10)  # 9 labelled of 10 closed (one declined; the rest pending)
    assert o["audit"]["labelled"] == 9 and o["audit"]["accuracy"] == pytest.approx(8 / 9, abs=1e-4)
    assert o["audit"]["strict_accuracy"] == pytest.approx(8 / 10, abs=1e-4)
    lo, hi = o["audit"]["interval_90"]
    assert lo < 8 / 9 < hi and o["since_last_label_s"] == 0.0
    assert o["languages"]["de"] == {"answers": 10, "drawn": 10, "labelled": 8, "need": o["languages"]["de"]["need"]}
    assert set(o["languages"]) == {"de", "en"} and set(o["languages"]["de"]["need"]) == {0.05, 0.10, 0.01, 0.005}
    n = {x["alpha"]: x for x in tl.need(lang="de")}[0.05]
    assert n["region_labels"] == 9 and n["region_errors"] == 1  # 8 right, 1 declined
    assert n["region_labels_per_day"] == pytest.approx(8 / 7, abs=0.01) and n["days_if_no_error"] == pytest.approx(
        n["more_if_no_error"] / (8 / 7), abs=0.1)  # fmt: skip
    assert LB.accuracy_interval(0, 0) is None and LB.accuracy_interval(5, 5)[1] == 1.0


def test_l22_ttl_days():
    with pytest.raises(LB.LabelsError):
        LB.parse_settings({"ttl_days": 7})  # under due_days (14)
    clock = Clock()
    tl = _task(clock=clock, audit_pi=1.0, ttl_days=30)
    old = [_serve(tl, f"old{i}", [0.7, 0.2, 0.1])[0] for i in range(3)]
    tl.post([{"request_id": r, "answer": "billing"} for r in old])
    clock.t += 20 * 86400
    new = _serve(tl, "new", [0.7, 0.2, 0.1])[0]
    tl.tick()
    assert {k[0] for k in tl.rows} == set(old) | {new}  # nothing is 30 days old yet
    clock.t += 11 * 86400
    tl.tick()
    assert {k[0] for k in tl.rows} == {new} and tl.counts()["erasures"]["erased"] == 1
    tl.tick()
    assert tl.counts()["erasures"]["erased"] == 1  # once an hour at most, and only when something is old


def test_l23_shares_for_a_strict_alpha():
    tl = _task()
    tl.cert_alpha = 0.05
    tl.tick()
    assert tl.epoch["epoch"] == 0  # a lenient alpha keeps the default shares
    tl.cert_alpha = 0.005
    tl.tick()
    e = tl.epoch
    assert e["epoch"] == 1 and e["settings"]["shares"] == LB.STRICT_SHARES and "alpha 0.005" in e["why"]
    assert e["settings"]["cert_pi"] > tl.epochs[0]["settings"]["cert_pi"] and e["settings"]["typical_pi_max"] == 0.0
    assert e["budget"]["per_day"] == 20
    tl.tick()
    assert tl.epoch["epoch"] == 1  # once
    given = _task(shares={"audit": 0.2, "cert": 0.5, "margin": 0.3, "typical": 0.0})
    given.cert_alpha = 0.01
    given.tick()
    assert given.epoch["epoch"] == 0  # shares given at registration are kept


def test_l24_a_strict_alpha_needs_a_tenth_of_its_answers_labelled():
    low = LB.parse_settings({"audit_pi": 0.02, "cert_pi": 0.02, "margin_pi_max": 0})  # at most 3.6%
    err, _ = LB.strict_check(low, 0.01)
    assert err and "at least 5%" in err and "cert_pi to at least" in err
    assert LB.strict_check(low, 0.05) == (None, [])  # a lenient alpha: no requirement
    ok = LB.parse_settings({"audit_pi": 0.02, "cert_pi": 0.2, "margin_pi_max": 0})
    assert LB.strict_check(ok, 0.005) == (None, [])
    err, warn = LB.strict_check(LB.parse_settings(True), 0.01)
    assert err is None and "400 answers a day" in warn[0]
    block = LB.registration_block({"settings": LB.parse_settings(True), "secret": LB.new_secrets()}, "memory", 0.01)
    assert block["warnings"] == warn
    tl = _task(audit_pi=0.02, cert_pi=0.02, margin_pi_max=0)
    tl.cert_alpha = 0.01
    for i in range(150):
        _serve(tl, f"t{i}", [0.7, 0.2, 0.1])
    o = tl.overview()
    assert o["label_share"] < 0.05 and "under the 5%" in o["warnings"][0]


def test_l25_the_budget_shows_overspend_and_the_rate_counts_once():
    clock = Clock()
    tl = _task(clock=clock, max_rows=500, tau_after=100)  # per_day 20
    for i in range(1000):  # 1,000 answers in a day: the floors draw more than 20 a day
        clock.t += 86.4
        _serve(tl, f"t{i}", [0.7, 0.2, 0.1])
    tl.tick()
    b = tl.budget()
    assert b["expected_per_day"] > 25 and b["over_budget"] is True
    w = next(w for w in tl.overview()["warnings"] if "above per_day (20)" in w)
    assert "counts as an error" in w and "register without a certificate" in w
    assert tl.settings["audit_pi"] == LB.PI_MIN  # display only: no chance moves
    block = LB.registration_block({"settings": LB.parse_settings(True), "secret": LB.new_secrets()}, "memory")
    assert block["budget_floor"] == {"min_share": 0.036, "above_answers_per_day": 556} and "warnings" not in block
    small = LB.registration_block({"settings": LB.parse_settings({"per_day": 2}), "secret": LB.new_secrets()}, "memory")
    assert "more than per_day (2)" in small["warnings"][0]  # the assumed first day already overspends
    rate = tl.rate()
    gone = next(iter(tl._undrawn))
    row = dict(tl.rows[gone])
    assert tl._evict()  # dropped past max_rows
    tl._index(row)  # read again from a merged file
    assert tl.rate() == rate  # not counted twice


def test_l26_the_detection_claim_only_for_a_validated_split():
    ref = LB.reference_q(0.8)
    assert ref == pytest.approx(1 - (1 - 0.005) * (1 - 0.05))
    explicit = LB.parse_settings({"audit_pi": 0.02, "cert_pi": 0.5, "margin_pi_max": 0.2})  # a high q, no named split
    d = LB.detection(explicit, 0.8, label_share=0.4)
    assert d["validated"] is False and d["warning"] == "Drift detection speed has not been validated for this split."
    assert "claim" not in d and d["q"] > d["q_reference"]  # q is a diagnostic, never the claim's condition
    strict = {**LB.parse_settings(True), "shares": dict(LB.STRICT_SHARES)}
    assert LB.detection(strict, 0.8, label_share=0.06)["claim"] == LB.DETECTION_CLAIM
    assert LB.detection(strict, 0.8, label_share=0.04)["warning"] == LB.DETECTION_UNVALIDATED  # under its 5%
    assert "the claim applies once at least 5%" in LB.detection(strict, 0.8)["pending"]  # not measured yet
    default = LB.parse_settings(True)  # 15/45/30/10: validated at 10%
    assert "claim" in LB.detection(default, 0.8, label_share=0.11) and "warning" in LB.detection(
        default, 0.8, label_share=0.07
    )
    custom = LB.parse_settings({"shares": {"audit": 0.5, "cert": 0.2, "margin": 0.3, "typical": 0.0}})
    custom.update(LB.budget_pis(custom, rate=4_000, region_share=0.8, margin_weight=0.5))
    d = LB.detection(custom, 0.8, rate=4_000, label_share=0.5)
    assert d["warning"] == LB.DETECTION_UNVALIDATED and d["needed"]["label_share"] > 0.05
    block = LB.registration_block({"settings": explicit, "secret": LB.new_secrets()}, "memory", alpha=0.01)
    assert block["detection"]["validated"] is False and LB.DETECTION_UNVALIDATED in block["warnings"]
    block = LB.registration_block({"settings": default, "secret": LB.new_secrets()}, "memory", alpha=0.01)
    assert "pending" in block["detection"] and LB.DETECTION_UNVALIDATED not in block.get("warnings", [])
    assert "detection" not in LB.registration_block({"settings": explicit, "secret": LB.new_secrets()}, "memory", 0.05)
    tl = _task(audit_pi=0.02, cert_pi=0.02, margin_pi_max=0)
    assert tl.detection() is None  # no certificate known yet
    tl.cert_alpha = 0.01
    assert tl.overview()["detection"]["warning"] == LB.DETECTION_UNVALIDATED


def test_l27_status_reads_never_hold_up_serving(monkeypatch):
    tl = _task(audit_pi=0.3, cert_pi=0.3)
    rids = [_serve(tl, f"t{i}", [0.6, 0.3, 0.1])[0] for i in range(300)]
    tl.post([{"request_id": r, "answer": "billing"} for r in rids[:100]])
    held = []
    real = LB.TaskLabels._state

    def state(row, ev, now, due):  # every scan works row by row outside the lock the answer path takes
        held.append(tl._lock._is_owned())
        return real(row, ev, now, due)

    monkeypatch.setattr(LB.TaskLabels, "_state", staticmethod(state))
    for read in (tl.queue, tl.counts, tl.overview, tl.need, tl.table, tl.eval_rows, tl.fit_rows, tl.budget):
        read()
    assert held and not any(held)
    hours = len(tl._hours)
    assert hours >= 1 and tl._drawn_share() == pytest.approx(
        np.mean([1 - np.prod([1 - d["pi"] for d in r["draws"].values()]) for r in tl.rows.values()])
    )  # the budget's drawn share comes from hourly sums, not a scan of the rows
