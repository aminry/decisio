# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A yes/no table learned from traffic across the replicas of a shared task store: the replicas, the store, the clocks
and the stand-ins of tests/unit/test_fast_table_replicas.py, with a yes/no question.

YR1  the lease holder promotes a version calibrated on its fit rows; the other replica adopts it from the version file
     with its calibration, times it against the task's own bar before it answers, and then answers with the table's
     own P(yes) under the agreement certificate
YR2  one replica's alarm suspends the table on every replica; a rollback on one replica reaches every replica, and
     the version it restores carries its own calibration

  uv run pytest -q tests/unit/test_noul_fast_table_replicas.py
"""

import math

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("obstore")
pytest.importorskip("tokenizers")

from obstore.store import MemoryStore  # noqa: E402
from test_fast_table_kinds import fake_static_embed, students  # noqa: E402,F401  (the static stand-in: autouse)
from test_fast_table_learn import status  # noqa: E402
from test_fast_table_replicas import poll, replica, shared_by_hand  # noqa: E402,F401  (the shared clocks: autouse)
from test_fast_table_task import virtual_time  # noqa: E402,F401  (the virtual clock: autouse)
from test_noul_fast_table import CLEAR_YES, QUESTION, ask, body, state  # noqa: E402

from decisio.serve import fast_table as FT  # noqa: E402
from decisio.serve import fast_table_learn as FTL  # noqa: E402

TASK = "angry"


def learner(r):
    return r.so.learning.learners[TASK]


def rows(r, n, seed):
    """n yes/no rows logged on replica r as its request path logs them while no table is live: decisio right and sure,
    seven texts in ten say the answer."""
    L = learner(r)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yes = bool(rng.random() < 0.5)
        words = 4 if rng.random() < 0.7 else 0
        p = np.array([0.95, 0.05]) if yes else np.array([0.05, 0.95])
        L.record(state(yes, 0.95, body(yes, rng, words)), {}, p, 0 if yes else 1, 60.0, FTL.NO_TABLE, math.nan, False)
    assert L.log.flush()


def promoted(store, **fast_table):
    a, b = replica(store, "a"), replica(store, "b")
    body_ = {"id": TASK, "question": QUESTION, "fast_table": {"learn": True, **fast_table}}
    r = a.client.post("/v1/tasks", json=body_)
    assert r.status_code == 200, r.text
    poll(a, b)
    rows(a, 600, seed=1)
    rows(b, 600, seed=2)
    out = a.so.learning.refit_task(TASK)
    assert out["outcome"] == "promoted" and out["version"] == 1, out
    poll(a, b)
    return a, b


def test_yr1_the_holder_promotes_a_calibrated_version_and_the_other_replica_times_it_against_the_tasks_bar():
    store = MemoryStore()
    a, b = promoted(store, min_saving=0.3)
    va, vb = learner(a).live, learner(b).live
    cal = va.record["calibration"]
    assert cal["method"] == "cross_fit" and cal["reference"] == "decisio" and cal["n"] == va.d["fit_rows"]
    assert vb is not va and vb.number == 1 and vb.record["calibration"] == cal  # read back from the version file
    assert learner(b).min_saving == 0.3 and status(b.client, TASK)["min_saving"] == 0.3
    sb = status(b.client, TASK)
    assert sb["state"] == "timing"
    assert ask(b.client, CLEAR_YES)["answered_by"] == "decisio"  # not before its own timing passes
    for i in range(FT.SHADOW_N):
        ask(b.client, state(True, 0.95, body(True, np.random.default_rng(i), 0)))
    timing = status(b.client, TASK)["replica"]["timing"]
    assert timing["passed"] and timing["last"]["required"] == 0.3 and timing["last"]["saving"] >= 0.3
    learner(b).explore = 0.0
    x = ask(b.client, CLEAR_YES)
    assert x["answered_by"] == "table" and x["certificate_kind"] == "agreement"
    own = float(FT.predict(vb.record, fake_static_embed("base", [CLEAR_YES]))[0, 0])
    assert x["noul"] == pytest.approx(own, abs=1e-12) and own > 0.5


def test_yr1_a_replica_whose_timing_falls_short_of_the_tasks_bar_keeps_serving_through_decisio():
    store = MemoryStore()
    a, b = promoted(store, min_saving=0.7)  # the trainer's own timing cleared it; this replica's must too
    import test_fast_table_kinds as K

    K.STATIC[0] = 20.0  # this replica's student is slow: a third of decisio's time
    for i in range(FT.SHADOW_N):
        ask(b.client, state(True, 0.95, body(True, np.random.default_rng(i), 0)))
    timing = status(b.client, TASK)["replica"]["timing"]
    assert not timing["passed"] and timing["last"]["required"] == 0.7 and 0.25 <= timing["last"]["saving"] < 0.7
    assert ask(b.client, CLEAR_YES)["answered_by"] == "decisio"


def test_yr2_suspension_and_rollback_reach_every_replica_with_each_versions_calibration():
    store = MemoryStore()
    a, b = promoted(store)
    for r in (a, b):
        for i in range(FT.SHADOW_N):
            ask(r.client, state(True, 0.95, body(True, np.random.default_rng(i), 0)))
        assert status(r.client, TASK)["state"] == "active"
    cal1 = learner(a).live.record["calibration"]
    learner(b).suspend("disagreement_cusum", "a test alarm on b")
    poll(b, a)
    assert status(a.client, TASK)["state"] == "suspended" and status(b.client, TASK)["state"] == "suspended"
    rows(b, 600, seed=9)
    assert a.so.learning.refit_task(TASK)["outcome"] == "promoted"  # a suspended version is replaced by the next
    poll(a, b)
    assert status(b.client, TASK)["version"]["live"] == 2
    assert learner(b).live.record["calibration"]["n"] == learner(b).live.d["fit_rows"] != cal1["n"]
    resp = b.client.post(f"/v1/tasks/{TASK}/fast_table/rollback", json={"version": 1})
    assert resp.status_code == 200 and resp.json()["status"]["fast_table"]["version"]["live"] == 1
    poll(a)
    assert status(a.client, TASK)["version"]["live"] == 1 and learner(a).live.record["calibration"] == cal1
