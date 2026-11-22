# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Labels from traffic on a shared task store (decisio.serve.labels.SharedPlace; --task-store).

R1  each replica writes only its own files, create-only; another replica reads them, each file once, and a listing
    resumes after the last file it saw
R2  a label posted to one replica for an answer another served is kept at once (`not_joined`) and joined when the
    serving replica's segment is read, on every replica; a label for an id this replica issued and does not hold is
    still `unknown_ref`
R3  the latest label wins by receipt time whatever order the replicas' files are read in; revisions are counted once
R4  the epoch that sets the cert region is stored by compare-and-swap: two replicas that both propose it end with
    the first one stored
R5  a store out of reach: a label write raises and nothing is acknowledged; rows wait in memory and are written once
    the store is back; past PENDING_MAX, rows serve but go unsaved and are counted
R6  a replica that does not write (--task-store-readonly) logs no answers, takes no labels, and reads the others'
R7  through the servers: a label posted to replica B for replica A's answer, the status on both, a read-only replica's
    409, a 503 while the store is down, the store's gc keeping a live log and removing a removed task's
R8  (with DECISIO_TEST_S3_ENDPOINT) two logs on a real S3-compatible service: listings, create-only writes and the
    compare-and-swap of the epochs
R11 a read that fails after the listing loses nothing: once the store is back every row is held (the listing moves
    only past files read); a file name with a long sequence parses; a label for an id no replica ever holds is dropped
    after its due window and counted
R9  erasure on a shared store: gone at once on the replica that erased and on the others when they read it; the
    merger purges it only PURGE_AFTER later, then no object holds it and every replica says purged
R10 two merges at once lose nothing and count nothing twice; settings changed on one replica reach the others

  uv run pytest -q tests/unit/test_labels_replicas.py
"""

import os
import time
import uuid

import numpy as np
import pytest

obstore = pytest.importorskip("obstore")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from obstore.store import MemoryStore  # noqa: E402
from test_labels_served import ask  # noqa: E402
from test_noul_certified import NoulEngine, noul_examples  # noqa: E402

from decisio.serve import fast_table_replicas as FTR  # noqa: E402
from decisio.serve import labels as LB  # noqa: E402
from decisio.serve import task_backends as TB  # noqa: E402
from decisio.serve import task_store_shared as S  # noqa: E402
from decisio.serve.systemone import SystemOne, add_routes  # noqa: E402
from decisio.serve.tasks import TaskStore  # noqa: E402

KEYS = ["billing", "tech", "sales"]
PREFIX = "labels/d1/s1/"


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class Faulty:
    """`Objects` that can be out of reach (`down`), counting the calls and the listing offsets."""

    def __init__(self, inner):
        self.inner, self.down, self.offsets = inner, False, []

    def __getattr__(self, op):
        fn = getattr(self.inner, op)

        def call(*a, **k):
            if self.down:
                raise TB.TaskStoreUnavailable("the task store cannot be reached (ConnectionError: refused)")
            if op == "list":
                self.offsets.append(k.get("offset"))
            return fn(*a, **k)

        return call


def _pair(clock=None, **settings):
    """Two replicas' logs of one task on one store, with the same secrets (the task record's)."""
    objects = Faulty(FTR.Objects(MemoryStore()))
    s = LB.parse_settings(settings or True)
    secret = LB.new_secrets()
    clock = clock or Clock()
    a = LB.TaskLabels("t", s, secret, LB.SharedPlace(objects, PREFIX, "host-a-1"), "host-a-1", clock)
    b = LB.TaskLabels("t", s, secret, LB.SharedPlace(objects, PREFIX, "host-b-2"), "host-b-2", clock)
    return a, b, objects


def _serve(tl, text, p=(0.7, 0.2, 0.1)):
    rid = LB.request_id(tl.replica, tl.clock())
    add = tl.observe(rid, "q", text=text, keys=KEYS, served=list(p), lp=np.log(p), h=None, certified=True,
                     version="v1", stack="fp", lang="en")  # fmt: skip
    return rid, add


def test_r1_own_files_create_only_read_once_listing_resumes():
    a, b, objects = _pair(audit_pi=1.0)
    rids = [_serve(a, f"t{i}")[0] for i in range(3)]
    assert a.flush(force=True) == 3
    paths = objects.list(PREFIX)
    assert paths and all(p.startswith(PREFIX + "host-a-1/rows/") for p in paths)
    b.sync()
    assert {k[0] for k in b.rows} == set(rids) and all(b.rows[(r, "q")]["replica"] == "host-a-1" for r in rids)
    objects.offsets.clear()
    b.sync()  # nothing new: the listing resumes after the last file seen
    assert any(o is not None for o in objects.offsets) and len(b.rows) == 3
    with pytest.raises(FTR.Exists):  # create-only: a file is never written twice
        objects.create(paths[0], b"x")


def test_r2_a_label_on_another_replica_is_kept_then_joined():
    a, b, _ = _pair(audit_pi=1.0)
    rid, _ = _serve(a, "My card was charged twice")
    out = b.post([{"request_id": rid, "answer": "billing"}])
    assert out["results"][0]["status"] == "not_joined" and out["not_joined"] == 1
    mine = LB.request_id("host-b-2", b.clock())
    assert b.post([{"request_id": mine, "answer": "billing"}])["results"][0]["status"] == "unknown_ref"
    a.flush(force=True)
    b.sync()
    assert b.counts()["states"] == {"labelled": 1} and b.counts()["not_joined"] == 0
    a.sync()  # the label event, from b's files
    row = next(r for r in a.table() if r["request_id"] == rid)
    assert row["state"] == "labelled" and row["selection"] == "slice" and row["replica_labelled"] == "host-b-2"
    # a label naming an option the row does not have is ignored when it is joined, and counted
    rid2, _ = _serve(a, "x")
    b.post([{"request_id": rid2, "answer": "nope"}])
    a.flush(force=True)
    b.sync()
    assert b.counts()["invalid_on_join"] == 1 and b.counts()["states"]["pending"] == 1


def test_r3_the_latest_label_wins_whatever_the_read_order():
    clock = Clock()
    a, b, _ = _pair(clock=clock, audit_pi=1.0)
    rid, _ = _serve(a, "x")
    a.flush(force=True)
    b.sync()
    b.post([{"request_id": rid, "answer": "tech"}])  # first, on b
    clock.t += 5
    a.post([{"request_id": rid, "answer": "sales"}])  # later, on a
    for log in (a, b):
        log.sync()
        row = next(r for r in log.table() if r["request_id"] == rid)
        assert row["answer"] == "sales" and log.revisions == 1
    c = LB.TaskLabels("t", a.settings, a.secret, LB.SharedPlace(a.place.objects, PREFIX, "host-c-3"), "host-c-3", clock)
    c.sync()  # a fresh replica reads both files, in name order: the same answer, the same count
    assert next(r for r in c.table() if r["request_id"] == rid)["answer"] == "sales" and c.revisions == 1


def test_r4_one_epoch_by_compare_and_swap():
    a, b, objects = _pair(audit_pi=0, cert_pi=1.0, margin_pi_max=0, tau_after=10, region=0.5)
    for i in range(10):
        _serve(a, f"a{i}", (0.9, 0.05, 0.05))
        _serve(b, f"b{i}", (0.5, 0.3, 0.2))
    a.flush(force=True)
    b.flush(force=True)
    a._maybe_tau()  # a proposes first: its rows only (b's not read yet)
    b._maybe_tau()  # b's proposal loses the compare-and-swap and adopts a's
    assert a.epoch == b.epoch and a.epoch["by"] == "host-a-1" and a.epoch["tau_min"] == pytest.approx(0.9)
    c = LB.TaskLabels("t", a.settings, a.secret, LB.SharedPlace(objects, PREFIX, "host-c-3"), "host-c-3", a.clock)
    c.sync()
    assert c.epoch == a.epoch


def test_r5_a_store_out_of_reach(monkeypatch):
    a, b, objects = _pair(audit_pi=1.0)
    rid, _ = _serve(a, "x")
    a.flush(force=True)
    objects.down = True
    with pytest.raises(TB.TaskStoreUnavailable):
        a.post([{"request_id": rid, "answer": "billing"}])
    assert a.counts()["states"] == {"pending": 1}  # nothing acknowledged, nothing in memory
    _serve(a, "y")
    with pytest.raises(TB.TaskStoreUnavailable):
        a.flush(force=True)
    assert len(a._pending_rows) == 1
    monkeypatch.setattr(LB, "PENDING_MAX", 1)
    _, add = _serve(a, "z")
    assert add["request_id"] and a.counts()["unsaved_rows"] == 1  # served, not written
    objects.down = False
    assert a.flush(force=True) == 1
    b.sync()
    assert len(b.rows) == 2  # the first two; the unsaved one is only on a


def test_r6_a_replica_that_does_not_write():
    a, _, objects = _pair(audit_pi=1.0)
    ro = LB.TaskLabels("t", a.settings, a.secret, LB.SharedPlace(objects, PREFIX, "ro-9", writes=False), "ro-9",
                       a.clock)  # fmt: skip
    assert ro.observe("ro-9-X", "q", text="x", keys=KEYS, served=[0.7, 0.2, 0.1], lp=None, h=None, certified=True,
                      version="v1", stack="fp", lang=None) == {}  # fmt: skip
    rid, _ = _serve(a, "x")
    a.flush(force=True)
    ro.sync()
    assert (rid, "q") in ro.rows
    with pytest.raises(LB.LabelsReadOnly):
        ro.post([{"request_id": rid, "answer": "billing"}])
    with pytest.raises(LB.LabelsReadOnly):
        ro.decline([rid])
    assert not any("/ro-9/" in p for p in objects.list(PREFIX))


# ---- through the servers ----------------------------------------------------------------------------------------


def _replica(store, readonly=False):
    backend = S.ObjectBackend(store, "memory://t", readonly=readonly, clock=time.time)
    ts = TaskStore("fp")
    abstention, report = TB.load_at_start(ts, backend)
    so = SystemOne(NoulEngine("1.0"), "decisio-test", task_store=ts, abstention_tasks=abstention,
                   task_backend=backend, task_report=report)  # fmt: skip
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, backend


def test_r7_two_replicas_a_reader_and_the_store_gc():
    store = MemoryStore()
    ca, sa, ba = _replica(store)
    cb, sb, bb = _replica(store)
    r = ca.post("/v1/tasks", json={"id": "angry", "examples": noul_examples(120), "certify": {},
                                   "labels": {"audit_pi": 1.0}})  # fmt: skip
    assert r.status_code == 200, r.text
    assert r.json()["labels"]["location"] == "task_store"
    bb.sync_once()  # replica B follows the task
    a, _ = ask(ca, "yes:0.93:first")
    assert a["request_id"].startswith(sa.labels_replica + "-")
    out = cb.post("/v1/tasks/angry/labels", json={"labels": [{"request_id": a["request_id"], "answer": "yes"}]})
    assert out.status_code == 200, out.text
    assert out.json()["results"][0]["status"] == "not_joined" and "durable" not in out.json()
    sa.flush_labels(force=True)  # A's labels thread writes its segment
    for c in (cb, ca):
        st = c.get("/v1/tasks/angry/labels/status").json()
        assert st["states"] == {"labelled": 1}, st
    # a read-only replica reads the log and refuses writes
    cr, sr, br = _replica(store, readonly=True)
    q = cr.get("/v1/tasks/angry/labels/status")
    assert q.status_code == 200 and q.json()["states"] == {"labelled": 1}
    w = cr.post("/v1/tasks/angry/labels", json={"labels": [{"request_id": a["request_id"], "answer": "no"}]})
    assert w.status_code == 409 and "readonly" in w.text
    assert cr.put("/v1/tasks/angry/labels/settings", json={"cert_pi": 0.5}).status_code == 409
    assert cr.delete(f"/v1/tasks/angry/labels/{a['request_id']}").status_code == 409
    b_ans, h = ask(cr, "no:0.9:on the reader")
    assert "request_id" not in b_ans and "x-decisio-request-id" not in h  # the reader logs nothing
    # the store out of reach: a label write is a 503, a read goes on with what the replica holds
    log = sb.labels_logs["angry"]
    log.place.objects = Faulty(log.place.objects)
    log.place.objects.down = True
    w = cb.post("/v1/tasks/angry/labels", json={"labels": [{"request_id": a["request_id"], "answer": "no"}]})
    assert w.status_code == 503, w.text
    st = cb.get("/v1/tasks/angry/labels/status").json()
    assert st["synced"] is False and st["states"] == {"labelled": 1}
    exp = cb.get("/v1/tasks/angry/labels/export")
    assert exp.status_code == 200 and "x-decisio-warning" in exp.headers  # a stale view says so
    log.place.objects.down = False
    # the store's gc keeps a live task's log, and removes a removed task's
    assert not [p for p in ba.gc(older_than_days=-1, dry_run=True) if p.startswith("labels/")]
    assert ca.delete("/v1/tasks/angry").status_code == 200
    left = [x for x in FTR.Objects(store).list("labels/") if x != "labels/LEASE.json"]
    assert not left  # the removing replica deletes the log (the merger's lease stays)
    sb.flush_labels(force=True)  # B sees the task gone and drops its log without writing
    bb.sync_once()
    sb.flush_labels(force=True)
    assert "angry" not in sb.labels_logs and FTR.Objects(store).list("labels/") == ["labels/LEASE.json"]


@pytest.mark.skipif(not os.environ.get("DECISIO_TEST_S3_ENDPOINT"), reason="no S3-compatible service named")
def test_r8_two_label_logs_on_a_real_s3_compatible_service():
    """As test_task_store_shared's S10: DECISIO_TEST_S3_ENDPOINT, the AWS_* credentials and region, and a bucket in
    DECISIO_TEST_S3_BUCKET (default decisio-test); a fresh prefix per run. The listing with an offset and a delimiter,
    the create-only writes and the compare-and-swap on a real service."""
    url = f"s3://{os.environ.get('DECISIO_TEST_S3_BUCKET', 'decisio-test')}/labels-{uuid.uuid4().hex[:8]}"
    endpoint = os.environ["DECISIO_TEST_S3_ENDPOINT"]
    objects = FTR.Objects(S.open_store(url, endpoint))
    s = LB.parse_settings({"audit_pi": 1.0, "tau_after": 3})
    secret, clock = LB.new_secrets(), Clock(time.time())
    a = LB.TaskLabels("t", s, secret, LB.SharedPlace(objects, PREFIX, "host-a-1"), "host-a-1", clock)
    b = LB.TaskLabels("t", s, secret, LB.SharedPlace(objects, PREFIX, "host-b-2"), "host-b-2", clock)
    rids = [_serve(a, f"t{i}")[0] for i in range(3)]
    assert b.post([{"request_id": rids[0], "answer": "billing"}])["results"][0]["status"] == "not_joined"
    a.tick(force=True)  # writes the segment and stores epoch 1 by a create
    clock.t += 1
    rids.append(_serve(a, "t3")[0])
    a.flush(force=True)
    b.tick()  # reads both of a's segments (the second after the listing's offset) and adopts the epoch
    assert {k[0] for k in b.rows} == set(rids) and b.counts()["states"]["labelled"] == 1
    assert b.epoch["epoch"] == 1 and b.epoch == a.epoch
    stored = b._swap_epochs(lambda cur: cur + [{**cur[-1], "epoch": 2}])  # a compare-and-swap on the stored copy
    a.sync()
    assert len(stored) == 3 and a.epoch["epoch"] == 2
    for path in objects.list(PREFIX):
        objects.delete(path)


def test_r11_a_failed_read_after_the_listing_loses_nothing():
    a, b, objects = _pair(audit_pi=1.0)
    first = [_serve(a, f"one{i}")[0] for i in range(3)]
    a.flush(force=True)
    second = [_serve(a, f"two{i}")[0] for i in range(3)]
    a.flush(force=True)  # two row segments from replica a
    real_get, calls = objects.inner.get, {"n": 0}

    def flaky_get(path, *args):
        if "/rows/" in path:
            calls["n"] += 1
            if calls["n"] == 2:  # the store drops after the listing, on the second segment
                raise TB.TaskStoreUnavailable("the task store cannot be reached")
        return real_get(path, *args)

    objects.inner.get = flaky_get
    with pytest.raises(TB.TaskStoreUnavailable):
        b.sync()
    objects.inner.get = real_get
    b.sync()
    assert {k[0] for k in b.rows} == set(first + second)
    # a file name whose sequence outgrew its width still parses
    place = b.place
    assert place._split("rows/1800000000000-1000000-host-a-1.jsonl") == ("rows", "1800000000000-1000000", "host-a-1")
    with pytest.raises(LB.LabelsError):
        place._split("rows/oops.jsonl")


def test_r11b_a_label_for_an_id_no_replica_holds_expires():
    clock = Clock()
    a, b, _ = _pair(clock=clock, audit_pi=1.0)
    made_up = LB.request_id("host-a-1", clock.t)  # well formed, from another replica, never served
    assert b.post([{"request_id": made_up, "answer": "billing"}])["results"][0]["status"] == "not_joined"
    b.sync()
    assert b.counts()["not_joined"] == 1
    clock.t += 15 * 86400
    b.sync()
    assert b.counts()["not_joined"] == 0 and b.counts()["unjoined_expired"] == 1


def _store_text(objects) -> str:
    """Every row and label on the store, as text (an erase event keeps only the opaque id it erased)."""
    lines = [x for p in objects.list(PREFIX) if not p.endswith("EPOCHS.json")
             for x in objects.get(p)[0].decode().splitlines()]  # fmt: skip
    return "\n".join(x for x in lines if '"type": "erase"' not in x)


def test_r9_erasure_on_a_shared_store():
    clock = Clock()
    a, b, objects = _pair(clock=clock, audit_pi=1.0)
    gone, _ = _serve(a, "my address is 1 Main St")
    keep, _ = _serve(a, "keep")
    a.flush(force=True)
    b.sync()
    b.post([{"request_id": gone, "answer": "billing"}, {"request_id": keep, "answer": "tech"}])
    out = a.erase(request_id=gone)
    assert out["state"] == "erased" and (gone, "q") not in a.rows  # at once here
    b.sync()
    assert (gone, "q") not in b.rows and b.counts()["states"] == {"labelled": 1}
    a.tick(merge=True)  # too soon to purge: another replica may still write it
    assert gone in _store_text(objects) and a.unpurged() == [out["erase_id"]]
    clock.t += LB.PURGE_AFTER + 1
    a.tick(merge=True)
    assert gone not in _store_text(objects) and keep in _store_text(objects)
    b.sync()
    assert b.counts()["erasures"] == {"erased": 1, "purged": 1}
    c = LB.TaskLabels("t", a.registered, a.secret, LB.SharedPlace(objects, PREFIX, "host-c-3"), "host-c-3", clock)
    c.sync()
    assert {r["request_id"] for r in c.table()} == {keep} and c.table()[0]["answer"] == "tech"


def test_r10_two_merges_at_once_and_settings_from_another_replica():
    a, b, objects = _pair(audit_pi=0.5)
    rids = [_serve(a, f"a{i}")[0] for i in range(20)] + [_serve(b, f"b{i}")[0] for i in range(20)]
    a.flush(force=True)
    b.flush(force=True)
    a.sync()
    b.sync()
    a.post([{"request_id": r, "answer": "tech"} for r in rids[:5]])
    a.sync()
    b.sync()
    want = a.table()
    a.compact()  # both merge from the same files: the second deletes what the first already did
    b.compact()
    c = LB.TaskLabels("t", a.registered, a.secret, LB.SharedPlace(objects, PREFIX, "host-c-3"), "host-c-3", a.clock)
    c.sync()
    assert c.table() == want and len({e["id"] for e in c.events}) == len(c.events)
    e = b.set_settings({"cert_pi": 0.5})
    a.sync()
    assert a.epoch == e and a.settings["cert_pi"] == 0.5
