# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A fast table that learns from traffic, learned across the replicas of a shared task store (`--task-store`;
docs/design/task-store.md, "Learning on the shared store"). One server's learning is decisio.serve.fast_table_learn's,
and nothing here changes it: this module adds where the rows, the versions and the state live when replicas share them.

Under the store's prefix, beside the task store's own `MANIFEST.json` and `objects/`:

    learner/LEASE.json                     {"holder", "generation", "expires"}: the one replica that trains
    tasks/<digest>/STATE.json              the task's learned-version state, rewritten only by compare-and-swap
    tasks/<digest>/versions/<sha256>.json  one promoted version, immutable, named by the digest of its bytes
    tasks/<digest>/log/<replica>/<start ms>-<seq>.seg   one replica's rows, written once (create-only)
    tasks/<digest>/log/merged/<generation>-<token>-<n>.seg   rows the trainer compacted, written once
    tasks/<digest>/log/INDEX.json          the merged files and the replica segments they hold (the trainer's)
    tasks/<digest>/labels.npz              a labelled task's examples' features, written at registration
    tasks/<digest>/stats/<replica>.json    one replica's counts and guards, rewritten every `STATS_SECONDS`

`<digest>` names one registration of a task id (its `learn.registration`): a task registered again starts afresh under
another, as one server drops its rows.

Every replica that writes to the store logs the rows decisio reads, as one server does, each row with the replica's id
and serving-stack fingerprint, buffered in memory and written as a new segment at `fast_table_learn.SEGMENT_BYTES` or
`SEGMENT_SECONDS`, whichever comes first, and only when there are rows. The request path never touches the store: the
task store's poll thread writes the segments, and while the store cannot be reached the rows wait in memory, up to
`BUFFER_BYTES` for the server, past which new rows are dropped and counted (`/health`).

One replica trains, cluster-wide: the holder of the lease. It is taken with a create when there is none, or by a
compare-and-swap on the copy read when that copy has expired (or is this replica's own, after a failed renewal), and
each take is generation + 1. It lasts `LEASE_SECONDS` and is renewed every `RENEW_SECONDS` by a compare-and-swap on
the holder's own last write; a renewal that fails stops training at once. The holder's refit worker reads every
replica's segments, keeps the newest `max_rows`, fits and certifies exactly as one server does, on the rows whose
serving-stack fingerprint is its own (the certificate also records the replicas and how many rows each gave), and
compacts the log. A version is published fenced: its object is written, the lease is re-read, and the state is moved by
compare-and-swap only if the lease's generation is still the publisher's and no later generation has published; a
holder that lost the lease while it was fitting publishes nothing.

Every replica follows the state on the task store's poll (a conditional GET), fetches the versions it does not hold,
checks each against its digest and its feature function, and swaps the live one in one assignment. A version is timed
on each replica's own requests (`fast_table.SHADOW_EVERY`, `SHADOW_N`, `timing_guard`) before its table answers there:
the host differs. A replica's alarm suspends the version for every replica through the state. A replica that starts
loads the promoted versions and serves them once its own timing passes; nothing is learned again.

A replica started with `--task-store-readonly` never writes to the store: it does not take the lease, log rows, write
its stats or share an alarm (its own suspends it alone), and it refuses a rollback. It follows the state and serves the
versions as any replica does.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import logging
import os
import re
import socket
import threading
import uuid

import numpy as np

from decisio.names import record_format
from decisio.serve import fast_table as FT
from decisio.serve import fast_table_learn as FTL
from decisio.serve.task_backends import TaskPersistError, TaskStoreReadOnly, TaskStoreUnavailable, dumps

logger = logging.getLogger("decisio.serve")

STATE_FORMAT = record_format("fast-table-shared-state")
INDEX_FORMAT = record_format("fast-table-shared-index")
STATS_FORMAT = record_format("fast-table-shared-stats")
VERSION_FORMAT = record_format("fast-table-version")

LEASE_PATH = "learner/LEASE.json"
LEASE_SECONDS = 120.0
RENEW_SECONDS = 40.0
STATS_SECONDS = 300.0  # each replica writes its stats this often; the trainer reads them as often
STATS_STALE = 3 * STATS_SECONDS  # a replica whose stats are older has stopped: its file goes, its rows are counted
BUFFER_BYTES = 256 << 20  # the rows a server holds while the store cannot be reached, for all its tasks
COMPACT_SEGMENTS = 16  # the trainer compacts once this many replica segments are not yet merged
MERGED_BYTES = 64 << 20  # the largest merged file
MAX_RACES = 8  # a state write retried because another replica wrote first, at most this often
REPLICA_BYTES = 64  # a replica id's longest form in a row


class LeaseLost(Exception):
    """This replica does not hold the trainer's lease, or no longer does: it publishes nothing."""


class StateGone(TaskPersistError):
    """The task's learning state is not in the store (the task was removed, or registered again)."""


class Exists(Exception):
    """A create-only write found the object there."""


class Raced(Exception):
    """A compare-and-swap found another version of the object than the one it names."""


NOT_MODIFIED = object()


def replica_id() -> str:
    """This process's replica id: the host name, made safe for a path, and a random suffix (a restarted replica is a
    new one)."""
    host = re.sub(r"[^A-Za-z0-9._-]", "-", socket.gethostname())[:40] or "replica"
    return f"{host}-{uuid.uuid4().hex[:8]}"


def learning_digest(task_id: str, registration: str | None = None) -> str:
    """The name of a registration's prefix under `tasks/` (a record without a registration id: its task id's)."""
    return FTL.task_digest(f"{task_id}\n{registration}" if registration else task_id)


def _bytes(obj) -> bytes:
    return dumps(obj).encode()


# ---- the store ---------------------------------------------------------------------------------------------------


class Objects:
    """The object-store operations learning uses, on an obstore store. Any failure to reach the store is
    TaskStoreUnavailable; a create over an object is `Exists`, a compare-and-swap on a stale tag is `Raced`. A tag is
    (e_tag, version)."""

    def __init__(self, store):
        self.store = store

    @staticmethod
    def _obs():
        import obstore
        from obstore import exceptions as ex

        return obstore, ex

    @staticmethod
    def _down(what: str, path: str, e: Exception) -> TaskStoreUnavailable:
        return TaskStoreUnavailable(f"the task store cannot be {what} ({path}: {type(e).__name__}: {e})")

    def get(self, path: str, if_none_match: str | None = None):
        """(bytes, tag), None when there is no such object, or NOT_MODIFIED when its e_tag is still `if_none_match`."""
        obs, ex = self._obs()
        try:
            got = (
                obs.get(self.store, path, options={"if_none_match": if_none_match})
                if if_none_match
                else obs.get(self.store, path)
            )
            data = bytes(got.bytes())
        except ex.NotModifiedError:
            return NOT_MODIFIED
        except (FileNotFoundError, ex.NotFoundError):
            return None
        except Exception as e:
            raise self._down("read", path, e) from e
        return data, (got.meta.get("e_tag"), got.meta.get("version"))

    def _put(self, path: str, data: bytes, mode):
        obs, ex = self._obs()
        try:
            r = obs.put(self.store, path, data, mode=mode)
        except ex.AlreadyExistsError as e:
            raise Exists(path) from e
        except ex.PreconditionError as e:
            raise Raced(path) from e
        except Exception as e:
            raise self._down("written", path, e) from e
        return r.get("e_tag"), r.get("version")

    def create(self, path: str, data: bytes):
        return self._put(path, data, "create")

    def update(self, path: str, data: bytes, tag):
        return self._put(path, data, {k: v for k, v in zip(("e_tag", "version"), tag) if v is not None})

    def put(self, path: str, data: bytes):
        return self._put(path, data, "overwrite")

    def delete(self, path: str) -> None:
        obs, _ = self._obs()
        try:
            obs.delete(self.store, path)
        except FileNotFoundError:
            return
        except Exception as e:
            raise self._down("written", path, e) from e

    def list(self, prefix: str) -> list[str]:
        obs, _ = self._obs()
        try:
            return sorted(item["path"] for batch in obs.list(self.store, prefix) for item in batch)
        except Exception as e:
            raise self._down("listed", prefix, e) from e


# ---- the lease ---------------------------------------------------------------------------------------------------


class Lease:
    """The trainer's lease (`learner/LEASE.json`), as the module's account says. `step` takes or renews it; `fence`
    re-reads it before a write only the holder may make."""

    def __init__(self, shared: Shared):
        self.sh = shared
        self.generation: int | None = None  # the generation this replica holds, or None
        self.expires = 0.0
        self._tag = None
        self.seen: dict | None = None  # the lease as last read or written
        self.takes = 0
        self.losses = 0
        self.lost_reason: str | None = None
        self._lock = threading.Lock()

    @property
    def replica(self) -> str:
        return self.sh.replica

    def held(self) -> bool:
        return self.generation is not None and self.sh.now() < self.expires

    def _drop(self, why: str) -> None:
        if self.generation is not None:
            self.losses += 1
            logger.warning("fast-table learning: this replica lost the trainer's lease (generation %s): %s",
                           self.generation, why)  # fmt: skip
        self.generation, self._tag, self.lost_reason = None, None, why

    def step(self) -> bool:
        """Renew the lease when held, else take it when there is none, it has expired, or it is this replica's own;
        True when this replica holds it after. TaskStoreUnavailable when the store cannot be reached (a holder has lost
        the lease by then)."""
        objects = self.sh.objects
        with self._lock:
            now = self.sh.now()
            if self.generation is not None:
                body = {"holder": self.replica, "generation": self.generation, "expires": now + LEASE_SECONDS}
                try:
                    self._tag = objects.update(LEASE_PATH, _bytes(body), self._tag)
                except Raced:
                    self._drop("another replica wrote the lease")
                    return False
                except TaskStoreUnavailable:
                    self._drop("the store could not be reached to renew it")
                    raise
                self.expires, self.seen = body["expires"], body
                return True
            got = objects.get(LEASE_PATH)
            if got is None:
                body = {"holder": self.replica, "generation": 1, "expires": now + LEASE_SECONDS}
                try:
                    tag = objects.create(LEASE_PATH, _bytes(body))
                except Exists:
                    return False
            else:
                cur = json.loads(got[0])
                self.seen = cur
                if now < float(cur["expires"]) and cur.get("holder") != self.replica:
                    return False
                g = int(cur["generation"]) + 1
                body = {"holder": self.replica, "generation": g, "expires": now + LEASE_SECONDS}
                try:
                    tag = objects.update(LEASE_PATH, _bytes(body), got[1])
                except Raced:
                    return False
            self.generation, self.expires, self._tag, self.seen = body["generation"], body["expires"], tag, body
            self.takes += 1
            self.lost_reason = None
            logger.info("fast-table learning: this replica (%s) trains, lease generation %s", self.replica,
                        self.generation)  # fmt: skip
            return True

    def fence(self) -> int:
        """The generation this replica holds, after reading the lease again: LeaseLost when it has lapsed here or
        another replica has taken it since."""
        g = self.generation
        if g is None or not self.held():
            raise LeaseLost("this replica does not hold the trainer's lease")
        got = self.sh.objects.get(LEASE_PATH)
        cur = json.loads(got[0]) if got else None
        if cur is None or cur.get("holder") != self.replica or int(cur.get("generation", -1)) != g:
            with self._lock:
                self._drop("another replica took the lease")
            raise LeaseLost(
                f"the lease moved from generation {g} to "
                + (f"{cur.get('generation')} (held by {cur.get('holder')})" if cur else "none")
            )
        return g

    def release(self) -> None:
        """At a clean stop: the lease is written as expired, so another replica takes it at its next try."""
        with self._lock:
            if self.generation is None:
                return
            body = {"holder": self.replica, "generation": self.generation, "expires": self.sh.now()}
            try:
                self.sh.objects.update(LEASE_PATH, _bytes(body), self._tag)
            except (Raced, TaskStoreUnavailable):
                pass
            self.generation, self._tag = None, None

    def facts(self) -> dict:
        seen = self.seen or {}
        now = self.sh.now()
        return {
            "held": self.held(),
            "generation": self.generation if self.generation is not None else seen.get("generation"),
            "holder": seen.get("holder"),
            "expires_in_s": round(float(seen["expires"]) - now, 1) if seen.get("expires") is not None else None,
            "takes": self.takes,
            "losses": self.losses,
            **({"lost": self.lost_reason} if self.lost_reason and not self.held() else {}),
        }


# ---- the log -----------------------------------------------------------------------------------------------------


class SharedLog:
    """One replica's rows of a task on the shared store, and the trainer's reading of every replica's (`collect`).
    `append` never touches the store: a full buffer is sealed for the poll thread to write (`upload`)."""

    def __init__(self, learner: SharedLearner):
        self.L = learner
        self.prefix = learner.prefix + "log/"
        self.students, self.k = dict(learner.ids), learner.k
        self.segment_bytes, self.segment_seconds = FTL.SEGMENT_BYTES, FTL.SEGMENT_SECONDS
        self._lock = threading.Lock()  # the buffer (the request path); never held over a store call
        self._upload_lock = threading.Lock()  # one writer of this log's segments at a time
        self._buf: list[dict] = []
        self._buf_bytes = 0
        self._buf_start: float | None = None
        # [start, rows, bytes, (name, data) once a write was tried] waiting to be written
        self._sealed: collections.deque = collections.deque()
        self._sealed_bytes = 0
        self.next_id = 0
        self.seq = 0
        self.flushed = 0  # rows this replica has written to the store
        self.segments = 0
        self.dropped = 0  # rows dropped because the buffer was full
        self.cluster_kept: int | None = None  # the rows the trainer kept at its last refit (from the state)
        self.last_error: str | None = None

    @property
    def sh(self) -> Shared:
        return self.L.sh

    @property
    def pending_bytes(self) -> int:
        return self._buf_bytes + self._sealed_bytes

    def append(self, row: dict) -> bool:
        """Buffer one row; False when the server's buffer is full and the row is dropped (counted)."""
        nb = FTL._row_bytes(row)
        with self._lock:
            if self.sh.buffered_bytes() + nb > self.sh.buffer_bytes:
                self.dropped += 1
                return False
            row["id"] = self.next_id
            self.next_id += 1
            if not self._buf:
                self._buf_start = self.L.m.now()
            self._buf.append(row)
            self._buf_bytes += nb
            if self._buf_bytes >= self.segment_bytes:
                self._seal()
            return True

    def _seal(self) -> None:
        if self._buf:
            self._sealed.append([self._buf_start, self._buf, self._buf_bytes, None])
            self._sealed_bytes += self._buf_bytes
            self._buf, self._buf_bytes, self._buf_start = [], 0, None

    def due(self) -> bool:
        return bool(self._buf) and (
            self._buf_bytes >= self.segment_bytes or self.L.m.now() - (self._buf_start or 0) >= self.segment_seconds
        )

    def tick(self) -> bool:
        """Seal the buffer when it is due (`upload` writes it); never writes, so False."""
        with self._lock:
            if self.due():
                self._seal()
        return False

    def flush(self) -> bool:
        """Seal the buffer and write what waits; False (the rows keep waiting) when the store cannot be reached."""
        with self._lock:
            self._seal()
        try:
            self.upload()
        except TaskStoreUnavailable as e:
            self.last_error = str(e)
            return False
        return True

    def upload(self) -> int:
        """Write the sealed segments, oldest first, each to a new name; returns how many. TaskStoreUnavailable stops
        it, and the rest wait."""
        n = 0
        with self._upload_lock:
            while True:
                with self._lock:
                    if not self._sealed:
                        break
                    entry = self._sealed[0]
                    start, rows, nb = entry[:3]
                self._write(entry)
                with self._lock:
                    if self._sealed and self._sealed[0][1] is rows:  # not discarded meanwhile (the task went)
                        self._sealed.popleft()
                        self._sealed_bytes -= nb
                    self.flushed += len(rows)
                    self.segments += 1
                n += 1
        self.last_error = None
        return n

    def _write(self, entry: list) -> None:
        """Write a sealed segment. Its name and bytes are fixed at the first try, so a write that reached the store but
        whose answer was lost is found there on the retry (the same bytes under the same name) and never written twice
        under another name: a row is counted once."""
        if entry[3] is None:
            entry[3] = (self._next_name(entry[0]), self._encode(entry[0], entry[1]))
        name, data = entry[3]
        while True:
            try:
                self.sh.objects.create(name, data)  # a new name, never written again
                return
            except Exists:
                got = self.sh.objects.get(name)
                if got is not None and got[0] == data:
                    return  # an earlier try wrote it
                name = self._next_name(entry[0])
                entry[3] = (name, data)

    def _next_name(self, start: float) -> str:
        name = f"{self.prefix}{self.sh.replica}/{int(start * 1000):013d}-{self.seq:06d}.seg"
        self.seq += 1
        return name

    def _encode(self, start: float, rows: list[dict]) -> bytes:
        arrays = FTL.columns(rows, list(self.students), self.k)
        arrays["replica"] = np.full(len(rows), self.sh.replica.encode()[:REPLICA_BYTES], dtype=f"S{REPLICA_BYTES}")
        meta = {
            "format": FTL.LOG_FORMAT,
            "students": self.students,
            "k": self.k,
            "replica": self.sh.replica,
            "start": start,
            "end": rows[-1]["t"],
            "rows": len(rows),
        }
        return FTL._npz(arrays, meta)

    def discard(self) -> None:
        """The task is gone: what waits is not written."""
        with self._lock:
            self._buf, self._buf_bytes, self._buf_start = [], 0, None
            self._sealed.clear()
            self._sealed_bytes = 0

    # the trainer's side

    def _parts(self, names: list[str]) -> tuple[list, set]:
        """(the files' columns, the names that could not be read)."""
        parts, bad = [], set()
        for name in names:
            got = self.sh.objects.get(self.prefix + name)
            if got is None:
                continue
            try:
                arrays, meta = FTL._unnpz(got[0])
            except Exception as e:  # a damaged file: named, left in place, not read, never compacted
                logger.warning("fast-table log: %s cannot be read (%s); it is left out", self.prefix + name, e)
                bad.add(name)
                continue
            if "replica" not in arrays:
                who = str(meta.get("replica", "")).encode()[:REPLICA_BYTES]
                arrays["replica"] = np.full(len(arrays["id"]), who, dtype=f"S{REPLICA_BYTES}")
            parts.append((arrays, meta.get("students", {})))
        return parts, bad

    def _listing(self):
        """(the replica segments in the store, the index and its tag)."""
        names = [p[len(self.prefix) :] for p in self.sh.objects.list(self.prefix)]
        raw = sorted(n for n in names if n.endswith(".seg") and not n.startswith("merged/"))
        got = self.sh.objects.get(self.prefix + "INDEX.json")
        if got is None:
            return raw, {"format": INDEX_FORMAT, "merged": [], "consumed": []}, None
        return raw, json.loads(got[0]), got[1]

    def read(self) -> dict:
        """Every replica's kept rows, oldest first, as columns (with `replica`); this replica's buffer included."""
        return self.collect(None, compact=False)[0]

    def collect(self, max_rows: int | None, compact: bool = True) -> tuple[dict, int]:
        """(the newest `max_rows` rows of every replica, oldest first, and how many there were), this replica's own
        written first. With `compact`, a holder of the lease that finds `COMPACT_SEGMENTS` unmerged segments, or rows
        past `max_rows`, writes the kept rows as merged files and deletes what they replace."""
        self.flush()
        raw, index, tag = self._listing()
        consumed = set(index.get("consumed", [])) & set(raw)  # a consumed name no longer listed was deleted
        fresh = [n for n in raw if n not in consumed]
        parts, bad = self._parts(list(index.get("merged", [])) + fresh)
        with self._lock:
            waiting = [r for e in self._sealed for r in e[1]] + list(self._buf)
        if waiting:
            cols = FTL.columns(waiting, list(self.students), self.k)
            cols["replica"] = np.full(len(waiting), self.sh.replica.encode()[:REPLICA_BYTES], f"S{REPLICA_BYTES}")
            parts.append((cols, self.students))
        rows = FTL.concat(parts, self.students, self.k, extra=("replica",))
        order = np.argsort(rows["t"], kind="stable")
        rows = {k: v[order] for k, v in rows.items()}
        n_all = len(rows["id"])
        if max_rows is not None and n_all > max_rows:  # retention: the oldest rows go
            rows = {k: v[n_all - max_rows :] for k, v in rows.items()}
        lease = self.sh.lease
        due = len(fresh) >= COMPACT_SEGMENTS or (max_rows is not None and n_all > max_rows)
        if compact and due and lease is not None and lease.held() and not waiting:
            try:
                self._compact(rows, index, tag, consumed | (set(fresh) - bad), bad)
            except (LeaseLost, TaskStoreUnavailable, Exists, Raced) as e:
                logger.warning("task %s: the fast-table log was not compacted: %s", self.L.task_id, e)
        return rows, n_all

    def _compact(self, rows: dict, index: dict, tag, consumed: set, bad: set) -> None:
        """Write `rows` as merged files, name them in the index (fenced, by compare-and-swap), then delete the files
        they replace (a merged file that could not be read stays named, and in place)."""
        objects = self.sh.objects
        g = self.sh.lease.fence()
        cols = {k: v for k, v in rows.items() if not k.startswith("has_")}
        n = len(cols["id"])
        per_row = max(1, sum(v.nbytes for v in cols.values()) // max(n, 1))
        chunk = max(1, MERGED_BYTES // per_row)
        token = uuid.uuid4().hex[:8]
        names = []
        for i, s in enumerate(range(0, n, chunk)):
            part = {k: v[s : s + chunk] for k, v in cols.items()}
            meta = {"format": FTL.LOG_FORMAT, "students": self.students, "k": self.k, "merged": True,
                    "generation": g, "rows": int(len(part["id"]))}  # fmt: skip
            name = f"merged/{g:06d}-{token}-{i:04d}.seg"
            objects.create(self.prefix + name, FTL._npz(part, meta))
            names.append(name)
        kept_bad = [m for m in index.get("merged", []) if m in bad]
        new = {"format": INDEX_FORMAT, "merged": names + kept_bad, "consumed": sorted(consumed), "generation": g,
               "rows": n, "at": self.L.m.now()}  # fmt: skip
        try:
            self.sh.lease.fence()
            if tag is None:
                objects.create(self.prefix + "INDEX.json", _bytes(new))
            else:
                objects.update(self.prefix + "INDEX.json", _bytes(new), tag)
        except (LeaseLost, Exists, Raced):
            for name in names:
                objects.delete(self.prefix + name)
            raise
        for name in [m for m in index.get("merged", []) if m not in bad] + sorted(consumed):
            objects.delete(self.prefix + name)

    def facts(self) -> dict:
        with self._lock:
            waiting = len(self._buf) + sum(len(e[1]) for e in self._sealed)
        return {
            "kept": int(self.cluster_kept or 0),
            "bytes": self.pending_bytes,
            "segments": self.segments,
            "buffered": waiting,
        }


# ---- one task's learning on the shared store ---------------------------------------------------------------------


class _Where:
    """Where a shared learner's files are, for its status."""

    persistent = True

    def __init__(self, prefix: str):
        self.prefix = prefix.rstrip("/")

    def describe(self) -> str:
        return f"the shared task store, {self.prefix}"


class SharedLearner(FTL.Learner):
    """One learning task on a replica of a shared store: fast_table_learn.Learner's request path, guards and refit,
    with the rows, versions and state in the store (the module's account). Makes no store call when it is made: its
    state is read on the next poll (`sync`), and until then it logs nothing and serves no table."""

    def __init__(self, manager, task, place=None):
        self.sh: Shared = manager.shared
        self.prefix = f"tasks/{learning_digest(task['id'], task['learn'].get('registration'))}/"
        super().__init__(manager, task, _Where(self.prefix))
        self.salt = None  # the store's, once the state is read: no row is logged before
        self.loaded = False
        self.st: dict | None = None
        self._st_tag = None
        self._io = threading.RLock()  # one state read or write at a time on this replica
        self._version_cache: dict[str, dict] = {}  # sha256 -> version (immutable, so fetched once)
        self.epoch = None  # the state's epoch this replica has adopted (moves at a promotion or a rollback)
        self.timing: dict | None = None  # this replica's timing of the live version
        self.pending_suspension: dict | None = None
        self.pending_labels: bytes | None = None
        self._labels_checked = False
        self.cluster: dict | None = None
        self.rows_at_refit = 0
        self.peer_rows: dict[str, int] = {}  # the trainer's view: rows each other replica has written
        self._stats_at: float | None = None
        self._cluster_at: float | None = None

    # the store's names

    def _path(self, name: str) -> str:
        return self.prefix + name

    # ---- hooks of the one-server learner ---------------------------------------------------------------------------

    def _load_state(self) -> dict:
        return {}

    def _open_log(self):
        return SharedLog(self)

    def _load_labels(self):
        return None

    def persist(self) -> None:
        """Nothing: what this replica changes is written to the store by the poll (a suspension), by the refit (its
        outcome and its version) or by the route (a rollback)."""

    def save_labels(self, labels: dict) -> None:
        arrays = {k: np.asarray(v) for k, v in labels.items() if k != "students"}
        meta = {"format": FTL.LABELS_FORMAT, "students": labels.get("students", {})}
        self.labels = {"arrays": arrays, "meta": meta}
        self.pending_labels = FTL._npz(arrays, meta)  # written by the next sync, after the state

    def record(self, text, feats, p, top, ms_decisio, kind, table_conf=float("nan"), would_answer=False) -> None:
        if not self.sh.writes:
            return  # a read-only replica logs nothing
        if self.salt is None:
            with self.lock:
                self.counters["not_logged_no_state"] += 1
            return
        super().record(text, feats, p, top, ms_decisio, kind, table_conf, would_answer)

    def append(self, row: dict) -> None:
        if not self.log.append(row):
            return  # the buffer is full: dropped, and counted by the log
        with self.lock:
            self.counters["logged"] += 1
            self.counters[FTL.ROW_KINDS[row["kind"]]] += 1
            if row["kind"] == FTL.EXPLORED and row["would"]:
                self.counters["explored_would_answer"] += 1
            self.since_refit += 1

    def serving(self):
        v = super().serving()
        if v is None or not self._timed_here(v):
            return None
        return v

    def state(self) -> str:
        st = super().state()
        if st == "active" and not self._timed_here(self.live):
            return "timing"
        return st

    def _timed_here(self, v) -> bool:
        t = self.timing
        return t is not None and t["version"] is v and t["passed"]

    def timing_due(self):
        v = self.live
        if v is None or self.suspended or self.off_reason:
            return None
        with self.lock:
            t = self.timing
            if t is None or t["version"] is not v or t["passed"]:
                return None
            t["requests"] += 1
            return v if t["requests"] % FT.SHADOW_EVERY == 0 else None

    def timed(self, v, table_ms: float, decisio_ms: float) -> None:
        with self.lock:
            t = self.timing
            if t is None or t["version"] is not v or t["passed"]:
                return
            t["pairs"].append((float(table_ms), float(decisio_ms)))
            if len(t["pairs"]) < FT.SHADOW_N:
                return
            g = FT.timing_guard([a for a, _ in t["pairs"]], [b for _, b in t["pairs"]],
                                v.d["certificate"]["table_share"])  # fmt: skip
            t["last"], t["pairs"] = g, []
            if not g["kept"]:
                return
            t["passed"], t["passed_at"] = True, self.m.now()
            self.low.clear()  # the guards start with the table's first answers here
            self.cusum = 0.0
            self.table_answers = self.evaluated = self.would_answer = 0
        logger.info("task %s: the learned fast table version %s passed this replica's timing (saving %.0f%%)",
                    self.task_id, v.number, 100 * g["saving"])  # fmt: skip

    def suspend(self, alarm: str, reason: str) -> None:
        super().suspend(alarm, reason)
        with self.lock:
            if self.suspended and "epoch" not in self.suspended:
                self.suspended = {**self.suspended, "by": self.sh.replica, "epoch": self.epoch,
                                  "shared": self.sh.writes}  # fmt: skip
                if self.sh.writes:
                    self.pending_suspension = self.suspended  # every replica's, once the poll writes it

    def rollback(self, number: int):
        if not self.sh.writes:
            raise TaskStoreReadOnly(
                "this replica does not write to the task store (--task-store-readonly): roll the version back on a "
                "replica that does"
            )
        d = next((v for v in self.versions if v["version"] == number), None)
        if d is None:
            kept = [v["version"] for v in self.versions]
            raise ValueError(f"fast_table.rollback: no version {number} is kept (the kept versions are {kept})")
        why = FT.cannot_compute(d["features"])
        if why:
            raise ValueError(f"fast_table.rollback: version {number} was {why}")

        def go_back(st):
            if number not in [e["version"] for e in st["versions"]]:
                raise ValueError(f"fast_table.rollback: version {number} is no longer kept")
            st.update(live=number, epoch=int(st["epoch"]) + 1, suspended=None)
            return st

        self.update_state(go_back)
        with self.lock:
            self.counters["rollbacks"] += 1
        return self.live

    # ---- the refit, on the lease holder ----------------------------------------------------------------------------

    def due(self) -> bool:
        if not self.loaded or not self.sh.holds():
            return False
        total = self.cluster_rows()
        if total < FT.MIN_EXAMPLES:
            return False
        since = total - self.rows_at_refit if total >= self.rows_at_refit else total
        return since >= FTL.REFIT_ROWS or (since > 0 and self.m.now() - self.last_refit_at >= FTL.REFIT_SECONDS)

    def cluster_rows(self) -> int:
        """The rows every replica has logged for the task, as the trainer last read their stats, and its own."""
        return sum(self.peer_rows.values()) + self.log.flushed + self.log.facts()["buffered"]

    def refit(self) -> dict:
        lease = self.sh.lease
        if not self.sh.holds():
            holder = (lease.seen or {}).get("holder") if lease is not None else None
            return {"at": self.m.now(), "rows": 0, "outcome": "not_trainer",
                    "reason": f"the trainer's lease is held by {holder or 'no replica'}"}  # fmt: skip
        if not self.loaded:
            return {"at": self.m.now(), "rows": 0, "outcome": "not_loaded",
                    "reason": "the task's learning state has not been read from the store"}  # fmt: skip
        g = lease.generation
        seen = self.cluster_rows()
        try:
            self._ensure_labels()
            out = super().refit()
        except TaskPersistError as e:  # the store went away while the rows were read: nothing changes
            out = {"at": self.m.now(), "rows": 0, "outcome": "error", "reason": str(e)}
            with self.lock:
                self.last_refit = out
                self.counters["refits"] += 1
                self.counters["refit_error"] += 1
            return out

        def outcome(st):
            if not self.sh.holds() or int(st.get("generation", 0)) > g:
                return None  # the lease moved: the next trainer's account stands
            st.update(last_refit=out, last_refit_at=out["at"], rows_at_refit=seen, rows_kept=out["rows"])
            return st

        try:
            self.update_state(outcome)
        except TaskPersistError as e:
            logger.warning("task %s: the refit's outcome could not be written to the store: %s", self.task_id, e)
        return out

    def _ensure_labels(self) -> None:
        if self.labels is not None or self._labels_checked:
            return
        got = self.sh.objects.get(self._path("labels.npz"))
        self._labels_checked = True
        if got is not None:
            arrays, meta = FTL._unnpz(got[0])
            self.labels = {"arrays": arrays, "meta": meta}

    def _kept_rows(self):
        return self.log.collect(self.max_rows)

    def _certificate_rows(self, rows: dict, ci, held_unbiased) -> dict:
        names, counts = np.unique(rows["replica"][ci], return_counts=True)
        other = held_unbiased & (rows["stack"] != self.stack_sha().encode())
        return {
            "replicas": {n.decode(): int(c) for n, c in zip(names, counts)},
            "other_stack_rows": int(other.sum()),
        }

    def _refit(self, rows: dict) -> dict:
        try:
            return super()._refit(rows)
        except LeaseLost as e:
            return {"outcome": "fenced", "reason": f"not published: {e}"}

    def _promote(self, c: dict, stack: str) -> dict:
        """Publish a version, fenced: its object, then the lease read again, then the state by compare-and-swap."""
        lease = self.sh.lease
        if lease is None or not lease.held():
            raise LeaseLost("this replica does not hold the trainer's lease")
        g = lease.generation
        with self.lock:
            n = self.next_version
        ident = FTL.sha256([self.task_id, n, c["threshold"], stack])[:8]
        d = {
            "format": VERSION_FORMAT,
            "version": n,
            "kind": c["kind"],
            "features": c["features"],
            "record": c["record"],
            "threshold": c["threshold"],
            "alpha": self.alpha,
            "certificate": {**c["certificate"], "id": f"{self.task_id}#learned-v{n}-{ident}"},
            "guard": c["guard"],
            "low_share": c["low_share"],
            "stack": stack,
            "promoted_at": self.m.now(),
            "fit_rows": c["fit_rows"],
            "generation": g,
            "replica": self.sh.replica,
            "registration": self.registration,
        }
        data = _bytes(d)
        sha = hashlib.sha256(data).hexdigest()
        try:
            self.sh.objects.create(self._path(f"versions/{sha}.json"), data)
        except Exists:
            pass  # the same bytes
        self._version_cache[sha] = json.loads(data)
        lease.fence()  # a holder that lost the lease while it fitted stops here

        def publish(st):
            if int(st.get("generation", 0)) > g:
                raise LeaseLost(f"generation {st['generation']} has published since")
            if int(st["next_version"]) != n:
                raise LeaseLost(f"version {st['next_version'] - 1} was published meanwhile")
            st["versions"] = (st["versions"] + [{"version": n, "object": sha, "generation": g}])[-FTL.VERSIONS_KEPT :]
            st.update(next_version=n + 1, live=n, epoch=int(st["epoch"]) + 1, suspended=None, generation=g)
            return st

        self.update_state(publish)
        with self.lock:
            self.counters["promotions"] += 1
        logger.info("task %s: the learned fast table version %s (%s) is published (lease generation %s)",
                    self.task_id, n, c["kind"], g)  # fmt: skip
        return self._version_cache[sha]

    # ---- the state -------------------------------------------------------------------------------------------------

    def _fresh_state(self) -> dict:
        seed = self.m.seed_for("salt:" + self.task_id)
        salt = os.urandom(16) if seed is None else seed.to_bytes(8, "little") * 2
        return {
            "format": STATE_FORMAT,
            "task_id": self.task_id,
            "key": self.key,
            "registration": self.registration,
            "salt": salt.hex(),
            "versions": [],
            "next_version": 1,
            "live": None,
            "epoch": 0,
            "suspended": None,
            "generation": 0,
            "serial": 0,
            "last_refit": None,
            "last_refit_at": self.m.now(),
            "rows_at_refit": 0,
            "rows_kept": None,
            "cluster": None,
            "created": self.m.now(),
        }

    def _version(self, sha: str) -> dict:
        d = self._version_cache.get(sha)
        if d is not None:
            return d
        got = self.sh.objects.get(self._path(f"versions/{sha}.json"))
        if got is None:
            raise TaskPersistError(f"task {self.task_id}: learned version object {sha[:12]} is not in the store")
        if hashlib.sha256(got[0]).hexdigest() != sha:
            raise TaskPersistError(
                f"task {self.task_id}: learned version {sha[:12]}: its bytes do not match its digest; not applied"
            )
        d = json.loads(got[0])
        if FT.record_features(d["record"]) != d["features"] or FTL.feature_id(d["features"]) not in self.ids.values():
            raise TaskPersistError(
                f"task {self.task_id}: learned version {d.get('version')} was fitted with another feature function "
                "than the task's students"
            )
        self._version_cache[sha] = d
        return d

    def apply_state(self, st: dict, tag) -> None:
        """Adopt the store's state: its versions (each checked), its live one (a new one is timed here first) and a
        suspension. All or nothing: a state that does not check out leaves this replica as it was."""
        if st.get("format") != STATE_FORMAT or st.get("key") != self.key or st.get("registration") != self.registration:
            raise TaskPersistError(f"task {self.task_id}: the learning state in the store is not this task's")
        if self.st is not None and int(st.get("serial", 0)) < int(self.st.get("serial", 0)):
            return  # older than what this replica holds
        kept = [self._version(e["object"]) for e in st["versions"]]
        live = st.get("live")
        d = next((x for x in kept if x["version"] == live), None) if live is not None else None
        with self.lock:
            self.st, self._st_tag = st, tag
            self.salt = bytes.fromhex(st["salt"])
            self.versions = kept
            self.next_version = int(st["next_version"])
            self.last_refit = st.get("last_refit")
            self.last_refit_at = float(st.get("last_refit_at") or self.last_refit_at)
            self.rows_at_refit = int(st.get("rows_at_refit") or 0)
            self.cluster = st.get("cluster")
            self.log.cluster_kept = st.get("rows_kept")
            if st["epoch"] != self.epoch:  # a promotion or a rollback (or this replica's start)
                self.epoch = st["epoch"]
                self.pending_suspension = None
                if d is None:
                    self.live, self.suspended, self.timing = None, None, None
                else:
                    self._set_live(FTL.Version(d), suspended=st.get("suspended"))
                    self.timing = {"version": self.live, "pairs": [], "requests": 0, "passed": False, "last": None,
                                   "since": self.m.now()}  # fmt: skip
            elif st.get("suspended") and not self.suspended and self.live is not None:
                self.suspended = st["suspended"]  # another replica's alarm
                self.low.clear()
                self.cusum = 0.0
            self.loaded = True

    def update_state(self, change) -> dict | None:
        """Move the state by compare-and-swap: `change(copy)` returns the new state, or None to write nothing; a write
        that loses to another replica's reads the state again and retries."""
        path = self._path("STATE.json")
        with self._io:
            st, tag = self.st, self._st_tag
            for _ in range(MAX_RACES):
                if st is None:
                    got = self.sh.objects.get(path)
                    if got is None:
                        raise StateGone(f"task {self.task_id}: its learning state is not in the store")
                    st, tag = json.loads(got[0]), got[1]
                new = change(copy.deepcopy(st))
                if new is None:
                    return None
                new["serial"] = int(st.get("serial", 0)) + 1
                new["updated"] = self.m.now()
                try:
                    new_tag = self.sh.objects.update(path, _bytes(new), tag)
                except Raced:
                    with self.lock:
                        self.counters["state_races"] += 1
                    st = None
                    continue
                self.apply_state(new, new_tag)
                return new
        raise TaskStoreUnavailable(f"task {self.task_id}: the learning state kept changing under this write; try again")

    def sync(self) -> None:
        """One poll: follow the state (made when there is none), then, on a replica that writes, the labels, the
        segments, a suspension, this replica's stats and, on the trainer, the cluster's."""
        path = self._path("STATE.json")
        with self._io:
            got = self.sh.objects.get(path, if_none_match=self._st_tag[0] if self._st_tag else None)
            if got is None and not self.loaded and self.sh.writes:
                fresh = self._fresh_state()
                try:
                    tag = self.sh.objects.create(path, _bytes(fresh))
                    got = (_bytes(fresh), tag)
                except Exists:
                    got = self.sh.objects.get(path)
            if got is not None and got is not NOT_MODIFIED:
                self.apply_state(json.loads(got[0]), got[1])
        if not self.sh.writes or not self.loaded:
            return
        if self.pending_labels is not None:
            self.sh.objects.put(self._path("labels.npz"), self.pending_labels)
            self.pending_labels = None
        self.log.tick()
        self.log.upload()
        sus = self.pending_suspension
        if sus is not None:

            def suspend_all(st):
                if st["epoch"] != sus["epoch"] or st.get("suspended"):
                    return None  # another version is live, or it is suspended already
                st["suspended"] = sus
                return st

            self.update_state(suspend_all)
            with self.lock:
                if self.pending_suspension is sus:
                    self.pending_suspension = None
        now = self.m.now()
        if self._stats_at is None or now - self._stats_at >= STATS_SECONDS:
            self.sh.objects.put(self._path(f"stats/{self.sh.replica}.json"), _bytes(self.stats()))
            self._stats_at = now
        if self.sh.holds() and (self._cluster_at is None or now - self._cluster_at >= STATS_SECONDS):
            self.aggregate()
            self._cluster_at = now

    def stats(self) -> dict:
        """What this replica tells the trainer: its rows, its state and its guards."""
        v = self.live
        with self.lock:
            c = dict(self.counters)
            low = list(self.low)
        return {
            "format": STATS_FORMAT,
            "replica": self.sh.replica,
            "at": self.m.now(),
            "stack": self.stack_sha(),
            "state": self.state(),
            "live": v.number if v is not None else None,
            "rows": {
                "logged": c.get("logged", 0),
                "flushed": self.log.flushed,
                "waiting": self.log.facts()["buffered"],
                "dropped": self.log.dropped,
                "by_kind": {k: c.get(k, 0) for k in FTL.ROW_KINDS},
            },
            "guards": {
                "low_confidence_share": sum(low) / len(low) if low else None,
                "cusum": round(self.cusum, 4),
                "checks": c.get("checks", 0),
                "disagreed": c.get("checks_disagreed", 0),
            },
            "coverage": self.would_answer / self.evaluated if self.evaluated else None,
            "requests_read": self.evaluated,
            "suspended": self.suspended.get("alarm") if self.suspended else None,
        }

    def aggregate(self) -> None:
        """The trainer reads every replica's stats into the state's cluster view; a replica silent for `STATS_STALE`
        has stopped, its file goes and its rows are kept in the count."""
        lease = self.sh.lease
        g = lease.generation if lease is not None else None
        now = self.m.now()
        retired = int(((self.st or {}).get("cluster") or {}).get("rows_retired") or 0)
        replicas, gone = {}, []
        for path in self.sh.objects.list(self._path("stats/")):
            got = self.sh.objects.get(path)
            if got is None:
                continue
            try:
                s = json.loads(got[0])
            except ValueError:
                continue
            if now - float(s.get("at", 0)) > STATS_STALE and s.get("replica") != self.sh.replica:
                gone.append((path, int(s.get("rows", {}).get("flushed", 0))))
                continue
            replicas[s["replica"]] = {k: s.get(k) for k in ("at", "stack", "state", "live", "rows", "guards",
                                                             "coverage", "requests_read", "suspended")}  # fmt: skip
        for path, flushed in gone:
            self.sh.objects.delete(path)
            retired += flushed
        self.peer_rows = {r: int(s["rows"]["flushed"]) for r, s in replicas.items() if r != self.sh.replica}
        if retired:
            self.peer_rows["(stopped replicas)"] = retired
        cluster = {
            "at": now,
            "trainer": self.sh.replica,
            "generation": g,
            "replicas": replicas,
            "rows_retired": retired,
            "rows": {
                k: sum(int(s["rows"].get(k, 0)) for s in replicas.values()) for k in ("logged", "flushed", "dropped")
            },
        }
        cluster["rows"]["flushed"] += retired

        def write(st):
            if int(st.get("generation", 0)) > (g or 0):
                return None
            st["cluster"] = cluster
            return st

        self.update_state(write)

    def remove_all(self) -> None:
        """Delete everything this registration holds in the store (the task was removed or registered again); what
        is left by a replica still writing goes with `python -m decisio.serve.task_store_shared gc`."""
        self.log.discard()
        for path in self.sh.objects.list(self.prefix):
            self.sh.objects.delete(path)

    # ---- status ----------------------------------------------------------------------------------------------------

    def facts(self) -> dict:
        out = super().facts()
        st = out["state"]
        t = self.timing
        if st == "timing":
            done = len(t["pairs"]) if t else 0
            out["reason"] = (
                f"version {self.live.number} is timed on this replica's requests before its table answers here "
                f"({done} of {FT.SHADOW_N} paired timings; decisio answers meanwhile)"
            )
        elif st == "learning" and not self.loaded:
            out["reason"] = "the task's learning state has not been read from the shared store yet"
        log = self.log
        out["rows"].update(
            kept=log.cluster_kept,
            stored=(
                f"segment files in the shared task store under {self.prefix}log/, each replica its own"
                if self.sh.writes
                else "this replica does not log (--task-store-readonly); the replicas that write to the store do"
            ),
            this_replica={
                "logged": self.counters.get("logged", 0),
                "written": log.flushed,
                "waiting": log.facts()["buffered"],
                "dropped": log.dropped,
                "segments": log.segments,
            },
        )
        lease = self.sh.lease
        out["replica"] = {
            "id": self.sh.replica,
            "readonly": not self.sh.writes,
            "trainer": bool(lease is not None and lease.held()),
            "timing": (
                {
                    "version": t["version"].number,
                    "passed": t["passed"],
                    "pairs": len(t["pairs"]),
                    "needed": FT.SHADOW_N,
                    **({"last": t["last"]} if t.get("last") else {}),
                }
                if t
                else None
            ),
        }
        out["cluster"] = self.cluster
        return out


# ---- this server's part ------------------------------------------------------------------------------------------


class Shared:
    """This server's part in learning shared through a task store: its replica id, the lease, the buffer budget, and
    the poll (`tick`) that follows and writes every task's state. `objects`: `Objects` on the task store's own store."""

    def __init__(self, objects, readonly: bool = False, replica: str | None = None, buffer_bytes: int | None = None):
        self.objects = objects
        self.writes = not readonly
        self.replica = replica or replica_id()
        self.buffer_bytes = BUFFER_BYTES if buffer_bytes is None else int(buffer_bytes)
        self.m = None  # the Learning this belongs to (`attach`)
        self.lease = Lease(self) if self.writes else None
        self._next_lease = 0.0
        self._tick_lock = threading.Lock()
        self.last_sync: float | None = None
        self.last_error: str | None = None
        self.closed = False

    def attach(self, learning) -> None:
        self.m = learning

    def now(self) -> float:
        return self.m.now()

    def learner(self, t) -> SharedLearner:
        return SharedLearner(self.m, t)

    def holds(self) -> bool:
        return self.lease is not None and self.lease.held()

    def learners(self) -> list[SharedLearner]:
        return [L for L in list(self.m.learners.values()) if isinstance(L, SharedLearner)]

    def buffered_bytes(self) -> int:
        return sum(L.log.pending_bytes for L in self.learners())

    def started(self, L: SharedLearner) -> None:
        """A task registered on this replica: its state (and labels) written now, when the store can be reached; the
        poll does it otherwise."""
        try:
            L.sync()
        except (TaskPersistError, LeaseLost) as e:
            logger.warning("task %s: the learning state could not be written to the store yet (%s)", L.task_id, e)

    def remove(self, L: SharedLearner) -> None:
        try:
            L.remove_all()
        except TaskPersistError as e:
            logger.warning("task %s: the learned rows and versions were not all deleted from the store (%s); "
                           "`python -m decisio.serve.task_store_shared gc` removes them", L.task_id, e)  # fmt: skip

    def tick(self) -> None:
        """One poll, on the task store's poll thread: the lease (every `RENEW_SECONDS`, when a task learns), then every
        task's `sync`. A failure is kept for /health and retried at the next poll; serving is not touched."""
        with self._tick_lock:
            if self.closed:
                return
            errors = []
            learners = self.learners()
            if self.lease is not None and learners and self.now() >= self._next_lease:
                try:
                    self.lease.step()
                    self._next_lease = self.now() + RENEW_SECONDS
                except TaskStoreUnavailable as e:
                    errors.append(str(e))
            for L in learners:
                try:
                    L.sync()
                except (TaskPersistError, LeaseLost) as e:
                    errors.append(f"task {L.task_id}: {e}")
            self.last_sync = self.now()
            self.last_error = errors[0] if errors else None
            if errors:
                logger.debug("fast-table learning: %s", "; ".join(errors))

    def close(self) -> None:
        """The server is stopping: no more polls, and the lease is released."""
        with self._tick_lock:
            self.closed = True
        if self.lease is not None:
            self.lease.release()

    def health(self) -> dict:
        learners = self.learners()
        return {
            "replica": self.replica,
            "readonly": not self.writes,
            "logs": self.writes,
            "lease": self.lease.facts() if self.lease is not None else None,
            "buffer": {
                "bytes": self.buffered_bytes(),
                "capacity": self.buffer_bytes,
                "rows_waiting": sum(L.log.facts()["buffered"] for L in learners),
                "rows_dropped": sum(L.log.dropped for L in learners),
            },
            "ok": self.last_error is None and self.last_sync is not None,
            "last_error": self.last_error,
            "last_sync_age_s": None if self.last_sync is None else round(max(0.0, self.now() - self.last_sync), 1),
        }
