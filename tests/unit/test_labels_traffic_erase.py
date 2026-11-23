# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Erasure and certificates from traffic labels (decisio.serve.labels: tombstones, the recompute).

E1  a look's certificate records its cohort, as keyed digests (no request id)
E2  an erasure that removes an answer of the certificate in force's cohort recomputes it at the merger's next turn:
    same look, floor and stack, the erased answer counted wrong at confidence 1, a new id naming the old one, which
    is withdrawn; nothing is recomputed again until another answer of it is erased
E3  erasing every wrong label of a cohort, after its look or before it, never yields a certificate the unerased
    cohort did not have (a lower threshold, or one where it had none)
E4  across replicas: an answer served on one replica and erased on another before it read it still leaves a
    tombstone (the merger writes it when it reads the row); the other replica sees it; after the purge no request id
    of the erased answer is on the store, and a fresh replica counts the tombstone in the cohort
E5  the label monitor counts an erased drawn answer it read as missing, never as labelled
E6  through the server: erasing an answer of the certificate's cohort serves the recomputed certificate

  uv run pytest -q tests/unit/test_labels_traffic_erase.py
"""

import json
import math

import numpy as np
from test_labels_replicas import PREFIX, _pair, _store_text
from test_labels_traffic_cert import KEYS, Clock, _log, frozen_c1

from decisio.serve import labels as LB


def _serve(tl, s0, i, stack="fp", certificate=None):
    tl.clock.t += 1
    rid = LB.request_id(tl.replica, tl.clock())
    tl.observe(rid, "q", text=f"t{i}", keys=KEYS, served=[s0, (1 - s0) / 2, (1 - s0) / 2], lp=None, h=None,
               certified=certificate is not None, version="v1", stack=stack, lang="en", certificate=certificate,
               s0=s0, s0_top="billing")  # fmt: skip
    return rid


def _cohort(tl, g, n, error=lambda s: 0.15 * (1.2 - s)):
    out = []
    for i in range(n):
        s0 = float(0.4 + 0.6 * g.random())
        out.append((_serve(tl, s0, i), s0, bool(g.random() >= error(s0))))
    return out


def _label(tl, rows):
    tl.post([{"request_id": rid, "answer": "billing" if ok else "tech"} for rid, _, ok in rows])


def _thr(c):
    return math.inf if c["threshold"] is None else c["threshold"]


def test_e1_the_cohort_is_recorded_as_keyed_digests():
    tl = _log(Clock(), looks=(60,), tau_after=10_000)
    rows = _cohort(tl, np.random.default_rng(1), 60)
    _label(tl, rows)
    tl.tick()
    c = tl.traffic_in_force()
    assert c["cohort"] == [tl.akey((rid, "q")) for rid, _, _ in rows] and c["counts"]["erased"] == 0
    assert not any(rid in json.dumps(c) for rid, _, _ in rows)  # no request id in the certificate


def test_e2_an_erasure_recomputes_the_certificate_in_force():
    tl = _log(Clock(), looks=(60,), tau_after=10_000)
    rows = _cohort(tl, np.random.default_rng(2), 60, error=lambda s: 0.02)
    _label(tl, rows)
    tl.tick()
    old = tl.traffic_in_force()
    gone = next(i for i, (_, _, ok) in enumerate(rows) if ok)
    tl.erase(request_id=rows[gone][0])
    tl.tick()
    new = tl.traffic_in_force()
    assert new["id"] != old["id"] and new["recompute_of"] == old["id"] and new["look"] == 1
    assert new["counts"]["erased"] == 1 and new["cohort"] == old["cohort"] and new["t_min"] == old["t_min"]
    s = [1.0 if i == gone else x for i, (_, x, _) in enumerate(rows)]  # erased: wrong at confidence 1
    ok = [False if i == gone else y for i, (_, _, y) in enumerate(rows)]
    assert _thr(new) == frozen_c1(s, ok, 0.1, 0.1)
    assert tl.traffic_status()["in_force"]["id"] == new["id"]  # the old one is withdrawn
    tl.tick()
    assert len(tl.traffic_certs) == 2  # nothing more to recompute
    second = next(i for i, (_, _, y) in enumerate(rows) if y and i != gone)
    tl.erase(request_id=rows[second][0])
    tl.tick()
    last = tl.traffic_in_force()
    assert last["recompute_of"] == new["id"] and last["counts"]["erased"] == 2 and len(tl.traffic_certs) == 3


def test_e3_erasing_wrong_labels_never_buys_a_certificate():
    for seed in range(12):
        # after the look
        tl = _log(Clock(), looks=(80,), tau_after=10_000)
        rows = _cohort(tl, np.random.default_rng(seed), 80, error=lambda s: 0.08 * (1.3 - s))
        _label(tl, rows)
        tl.tick()
        t0 = _thr(tl.traffic_in_force())
        for rid, _, ok in rows:
            if not ok:
                tl.erase(request_id=rid)
        tl.tick()
        c = tl.traffic_in_force()
        assert _thr(c) >= t0, seed
        assert c["counts"]["erased"] == sum(1 for _, _, ok in rows if not ok)
        # before the look: the tombstones keep their place in the cohort
        tl2 = _log(Clock(), looks=(80,), tau_after=10_000)
        rows2 = _cohort(tl2, np.random.default_rng(seed), 80, error=lambda s: 0.08 * (1.3 - s))
        _label(tl2, rows2)
        for rid, _, ok in rows2:
            if not ok:
                tl2.erase(request_id=rid)
        tl2.tick()
        c2 = tl2.traffic_in_force()
        assert c2 is not None and "recompute_of" not in c2 and _thr(c2) >= t0, seed
        assert c2["counts"]["erased"] == sum(1 for _, _, ok in rows2 if not ok)


def _traffic_pair(clock):
    a, b, objects = _pair(clock=clock, audit_pi=0, cert_pi=1.0, margin_pi_max=0, tau_after=10_000,
                          cert_score="label_free")  # fmt: skip
    for tl in (a, b):
        tl.traffic = {"looks": [40], "alpha": 0.1, "delta": 0.1, "target_share": None, "c_max": 0.45, "stack": "fp"}
        tl.epochs = [*tl.epochs, {**tl.epochs[-1], "epoch": 1, "tau_min": 0.0, "tau_set": True}]
    return a, b, objects


def test_e4_tombstones_across_replicas_and_the_purge():
    clock = Clock()
    a, b, objects = _traffic_pair(clock)  # a is the merger
    rows = _cohort(b, np.random.default_rng(4), 40, error=lambda s: 0.02)
    b.flush(force=True)
    gone = rows[3][0]
    a.erase(request_id=gone)  # before a has read b's answers
    _label(b, [r for r in rows if r[0] != gone])
    b.flush(force=True)
    a.tick(merge=True)  # reads b's files, drops the erased row, writes its tombstone, makes the look
    c = a.traffic_in_force()
    assert c is not None and c["counts"]["erased"] == 1 and c["counts"]["requested"] == 40
    assert c["cohort"][3] == a.akey((gone, "q"))
    b.sync()
    assert a.akey((gone, "q")) in b.erased_draws and b.traffic_in_force()["id"] == c["id"]
    clock.t += LB.PURGE_AFTER + 1
    a.tick(merge=True)  # the purge
    assert gone not in _store_text(objects) and a.counts()["erasures"]["purged"] == 1
    fresh = LB.TaskLabels("t", a.settings, a.secret, LB.SharedPlace(objects, PREFIX, "host-c-3"), "host-c-3", clock)
    fresh.traffic, fresh.epochs = dict(a.traffic), list(a.epochs)
    fresh.sync()
    cohort = fresh.traffic_cohort()
    assert len(cohort) == 40 and cohort[3][1].get("erased") and fresh.traffic_in_force()["id"] == c["id"]


def test_e5_the_label_monitor_counts_an_erased_answer_missing():
    tl = _log(Clock(), looks=(40,), tau_after=10_000)
    rows = _cohort(tl, np.random.default_rng(5), 40, error=lambda s: 0.0)
    _label(tl, rows)
    tl.tick()
    c = tl.traffic_in_force()
    tl.cert_id, tl.cert_alpha, tl.cert_threshold, tl.cert_label_free = c["id"], 0.1, 0.5, True
    served = [_serve(tl, 0.9, 100 + i, certificate=c["id"]) for i in range(10)]
    _label(tl, [(r, 0.9, True) for r in served])
    before = tl.drift()
    assert before["missing"] == 0 and before["labels"] == 10
    tl.erase(request_id=served[0])
    after = tl.drift()
    assert after["missing"] == 1 and after["labels"] == 9 and after["label_response"] == 0.9


def test_e6_through_the_server():
    from test_labels_served import ask
    from test_labels_traffic_cert import _certified_server

    client, so, tc = _certified_server()
    log = so.labels_logs["angry"]
    keys = {log.akey(k): k for k in log.rows}
    gone = keys[tc["cohort"][0]][0]
    r = client.delete(f"/v1/tasks/angry/labels/{gone}")
    assert r.status_code == 200, r.text
    so.flush_labels(force=True)
    new = log.traffic_in_force()
    assert new["recompute_of"] == tc["id"] and new["counts"]["erased"] == 1
    a, _ = ask(client, "yes:0.97:after the erasure")
    assert a["certificate"] == new["id"]
    st = client.get("/v1/tasks/angry").json()["status"]
    assert st["traffic_certification"]["in_force"]["recompute_of"] == tc["id"]
