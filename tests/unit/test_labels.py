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
    assert LB.parse_settings(True) == LB.DEFAULTS
    assert LB.parse_settings({"cert_pi": 0.3})["cert_pi"] == 0.3
    for bad in ({"nope": 1}, {"cert_pi": 0.01}, {"features": "all"}, {"margin_pi_min": 0.3, "margin_pi_max": 0.2},
                {"due_days": 0}, {"show_answer": True}, "on"):  # fmt: skip
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
    assert {k[0] for k in back.rows} == set(drawn)  # read back: the same rule, every undrawn row over the cap goes
