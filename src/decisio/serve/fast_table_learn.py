# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A fast table that learns from a task's own traffic: decisio answers, its answers are logged as features, and a
background refit fits and certifies a table on them (`"fast_table": {"learn": true, ...}`).

A learning task is registered with a question and no examples, and is matched to requests as any task is (its question
type and options, `decisio.serve.tasks.task_key`). Choice and yes/no questions, as the table is. Until a table is
certified, decisio answers every request of the task. Each request decisio reads for it is logged as a row: the features
of the students the task may serve (`static`: the static student's vector, float16; `encoder`: the encoder's, float16,
computed after the response by a bounded background queue that drops rows under load rather than make a caller wait),
decisio's served distribution (float16), its confidence and choice, the time decisio's read took and each student's, the
row's kind, the serving-stack fingerprint and the time. No text is kept. A row's kind:

  no_table    no table was live (or it was suspended): decisio read the request, as it reads all of them then
  deferred    the live table's confidence was under its threshold, so decisio answered
  explored    a request drawn at random (a share `explore`, default 2%) that decisio answered while the table was live;
              for a request the table would have answered, the caller gets decisio's answer and the time decisio takes
  spot_check  one table answer in `fast_table.SAMPLE_EVERY` also read by decisio; the caller gets the table's answer

Refits run on one background thread per server, one at a time, at low priority where the platform allows it (nice 10 on
Linux; elsewhere the thread's normal priority), with scikit-learn's BLAS limited to one thread: every `REFIT_ROWS` new
rows or every `REFIT_SECONDS`, once a task has `fast_table.MIN_EXAMPLES` rows. A refit fits each student the task asks
for on decisio's choices, each row weighted by one over the chance it was logged (1 for a no_table or deferred row; for
an explored or spot-check row the table would have answered, one over `explore` plus the spot-check share of the rest).
Rows are split once by a keyed digest of their text: `HOLDOUT` of the digests are held out, and a held-out row is never
fitted on. The certificate is made only on the held-out rows logged independently of the table's confidence
(`CERTIFY_KINDS`: no_table and explored rows. A deferred row is the live table's choice of hard requests and a
spot-check row is drawn only from the requests the table answered, so neither stands for the traffic and neither
enters a certificate; spot checks feed the agreement CUSUM): the table's threshold is the lowest at which its choices
differ from decisio's on at most `alpha` of the held-out rows it answers, with probability 1 - delta
(`decisio.readout.certify.threshold`). That is
an agreement certificate: it bounds how often the table's answer differs from decisio's, not how often it is wrong. A
task that also has labelled examples (`"learn": true` on a labelled registration) adds them to every fit, and its
certificate is made against their gold answers on its held-out labelled examples (a gold certificate) when there are at
least `certify.minimum_examples(alpha, delta)` of them with the student's features; otherwise against decisio.

A refit's version goes live only if its certificate holds, its expected saving (`fast_table.timing_guard`, from the
times logged on this host) is at least the task's `min_saving` (`fast_table.record_min_saving` of its request: the
default `fast_table.MIN_SAVING` for a task registered with this setting, `fast_table.LEGACY_MIN_SAVING` for one
registered before it), and it saves more than the live version did when it was promoted (`auto`: the students are
compared as `fast_table.choose` does). A refit that fails leaves the live version live. The last `VERSIONS_KEPT`
promoted versions are kept (the record and the certificate, not the rows), numbered, and any of them can be restored
(`rollback`). A version is served only with its own feature function and on the serving stack it was certified on. The
swap is one assignment, so a request is answered by one version or the other.

A yes/no task's table (type "noul") is a two-class one. Each refit calibrates its P(yes) on the rows its certificate is
made on (`fast_table.fit_temperature`, against the certificate's reference answers: the labelled answers of a gold
certificate, decisio's of an agreement one), and on a server whose --noul-commit rule is on its threshold, its
sureness and its low-confidence guard read its confidence after the rule (`commit_band`), as decisio's certified
yes/no answers are read. Its logged rows hold decisio's distribution before the rule.

Two drift guards watch a live table, and either alarm suspends it (decisio answers every request; logging and refits go
on, and the next version that certifies goes live): the low-confidence share, the share of the last `LOWCONF_WINDOW`
requests whose table confidence is under the threshold, rising more than `LOWCONF_MARGIN` above its share on the
certificate's rows; and a one-sided CUSUM on disagreement, fed by the spot checks and by the explored rows the table
would have answered (reference alpha, alarm at `CUSUM_LIMIT`).

Rows are kept per task in append-only segment files under the tasks directory (`<tasks-dir>/learn/<task digest>/log/
<start ms>-<sequence>.seg`, numpy's npz, with a small JSON index), flushed when the buffer holds `SEGMENT_BYTES` or is
`SEGMENT_SECONDS` old, never rewritten; at most `max_rows` (default 200,000) are kept, the oldest dropped at refit.
Without a tasks directory they are kept in memory and lost at restart, and the task's status says so. The log and the
task's learning state sit behind a small storage interface (`DirPlace`, `MemoryPlace`: put, get, delete, list). On a
shared task store (`--task-store`) the replicas learn together, one of them training: `Learning(shared=...)` and
decisio.serve.fast_table_replicas, which change nothing of the above for one server.

Standard library, numpy and the `table` extra (scikit-learn).
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import hashlib
import io
import json
import logging
import math
import os
import queue
import shutil
import sys
import threading
import time
import uuid
import weakref
from pathlib import Path

import numpy as np

from decisio.names import record_format
from decisio.readout import certify
from decisio.serve import fast_table as FT
from decisio.serve.stack import sha256

logger = logging.getLogger("decisio.serve")

FORMAT = record_format("fast-table-learning")
LOG_FORMAT = record_format("fast-table-log")
LABELS_FORMAT = record_format("fast-table-labels")

ROW_KINDS = ("no_table", "deferred", "explored", "spot_check")
NO_TABLE, DEFERRED, EXPLORED, SPOT_CHECK = range(4)
# the rows drawn independently of the table's confidence: every request while no table is live, and a fixed share of
# every request a live table reads; spot checks come only from the table's own answers and over-sample its region
CERTIFY_KINDS = (NO_TABLE, EXPLORED)
STUDENTS = ("static", "encoder")  # the feature columns a row may hold

DEFAULT_EXPLORE = 0.02  # the smallest share that kept the certificate honest on long replayed streams
MAX_EXPLORE = 0.5
DEFAULT_MAX_ROWS = 200_000
DEFAULT_ALPHA, DEFAULT_DELTA = 0.05, 0.1
REFIT_ROWS = 2_000  # a refit every this many new rows
REFIT_SECONDS = 3_600  # or every hour, whichever comes first
WORKER_TICK = 30.0  # how often the worker looks for a due refit and an old log buffer (seconds)
SEGMENT_BYTES = 4 << 20
SEGMENT_SECONDS = 300
HOLDOUT = 0.3  # the share of row digests held out of every fit, for certificates
VERSIONS_KEPT = 5
LOWCONF_WINDOW = 200
LOWCONF_MARGIN = 0.15
CUSUM_LIMIT = 5.0
ENCODE_QUEUE = 256  # rows waiting for their encoder features; a fuller queue logs the row without them
LABEL_WEIGHT = 2.0  # a labelled example's weight in a refit's fit, against a traffic row's 1
NICE = 10
TIMED_PREDICTS = 20  # rows a refit times the classifier on, one at a time
BACKGROUND = True  # a server's refits and encoder rows run on their own threads (tests run them when they ask)
SEED = None  # exploration's seed (None: fresh entropy); per task, mixed with the task's id

RETAIN_TEXT_REFUSED = (
    "fast_table.retain_text: keeping request text is not available in this version; a learning table keeps the "
    "students' features and decisio's distributions only, never the text"
)
LEARN_KEYS = ("learn", "explore", "max_rows", "retain_text")


# ---- the request ----------------------------------------------------------------------------------------------------


def split_request(value):
    """(the table block, as `fast_table.parse_request` reads it, and the learning settings or None) of a registration's
    `fast_table`. ValueError says what is wrong."""
    if not isinstance(value, dict) or not any(k in value for k in LEARN_KEYS):
        return value, None
    if "retain_text" in value:
        raise ValueError(RETAIN_TEXT_REFUSED)
    learn = value.get("learn", False)
    if not isinstance(learn, bool):
        raise ValueError("fast_table.learn: true or false")
    rest = {k: v for k, v in value.items() if k not in LEARN_KEYS}
    if not learn:
        extra = [k for k in ("explore", "max_rows") if k in value]
        if extra:
            raise ValueError(f'fast_table.{extra[0]}: for a table that learns from traffic only ("learn": true)')
        return rest or True, None
    table = FT.parse_request(rest)
    explore = value.get("explore", DEFAULT_EXPLORE)
    if isinstance(explore, bool) or not isinstance(explore, (int, float)) or not 0 <= explore <= MAX_EXPLORE:
        raise ValueError(
            f"fast_table.explore: the share of requests decisio answers anyway, from 0 to {MAX_EXPLORE} "
            f"(default {DEFAULT_EXPLORE})"
        )
    max_rows = value.get("max_rows", DEFAULT_MAX_ROWS)
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < FT.MIN_EXAMPLES:
        raise ValueError(
            f"fast_table.max_rows: the rows kept for the task, an integer of at least {FT.MIN_EXAMPLES} "
            f"(default {DEFAULT_MAX_ROWS})"
        )
    return table, {"request": table, "explore": float(explore), "max_rows": int(max_rows)}


def feature_id(feat: dict) -> str:
    """A short identity of a feature function: rows logged with another function are not fitted for this one."""
    return sha256(feat)[:16]


def task_digest(task_id: str) -> str:
    return hashlib.sha256(str(task_id).encode()).hexdigest()[:32]


# ---- storage --------------------------------------------------------------------------------------------------------


class DirPlace:
    """A task's learning files in a directory: each `put` writes a temporary file, fsyncs it and renames it over the
    name, so a crash leaves the old file or the new one."""

    persistent = True

    def __init__(self, root):
        self.root = Path(root)

    def describe(self) -> str:
        return str(self.root)

    def _path(self, name: str) -> Path:
        return self.root / name

    def put(self, name: str, data: bytes) -> None:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        with contextlib.suppress(OSError):
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def get(self, name: str) -> bytes | None:
        try:
            return self._path(name).read_bytes()
        except FileNotFoundError:
            return None

    def delete(self, name: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._path(name).unlink()

    def names(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.is_dir():
            return []
        return sorted(f"{prefix}{p.name}" for p in base.iterdir() if p.is_file() and not p.name.startswith(".tmp-"))

    def remove_all(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class MemoryPlace:
    """A task's learning files in memory: lost at restart."""

    persistent = False

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def describe(self) -> str:
        return "memory"

    def put(self, name: str, data: bytes) -> None:
        with self._lock:
            self.files[name] = bytes(data)

    def get(self, name: str) -> bytes | None:
        return self.files.get(name)

    def delete(self, name: str) -> None:
        with self._lock:
            self.files.pop(name, None)

    def names(self, prefix: str) -> list[str]:
        return sorted(n for n in list(self.files) if n.startswith(prefix) and "/" not in n[len(prefix) :])

    def remove_all(self) -> None:
        with self._lock:
            self.files.clear()


def _npz(arrays: dict, meta: dict) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, meta=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8), **arrays)
    return buf.getvalue()


def _unnpz(data: bytes) -> tuple[dict, dict]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(bytes(z["meta"]).decode())
    return arrays, meta


def columns(rows: list[dict], students: list[str], k: int) -> dict:
    """The rows (dicts, as `Learner.record` makes them) as column arrays."""
    n = len(rows)
    out = {
        "id": np.array([r["id"] for r in rows], dtype=np.int64),
        "t": np.array([r["t"] for r in rows], dtype=np.float64),
        "kind": np.array([r["kind"] for r in rows], dtype=np.uint8),
        "digest": np.array([r["digest"] for r in rows], dtype=np.uint64),
        "stack": np.array([r["stack"] for r in rows], dtype="S16"),
        "top": np.array([r["top"] for r in rows], dtype=np.int16),
        "would": np.array([r.get("would", False) for r in rows], dtype=bool),
        "conf": np.array([r["conf"] for r in rows], dtype=np.float32),
        "ms_decisio": np.array([r["ms_decisio"] for r in rows], dtype=np.float32),
        "table_conf": np.array([r["table_conf"] for r in rows], dtype=np.float32),
        "dist": np.stack([r["dist"] for r in rows]).astype(np.float16) if n else np.zeros((0, k), np.float16),
    }
    for s in students:
        vecs = [r.get(f"f_{s}") for r in rows]
        dim = next((len(v) for v in vecs if v is not None), 0)
        f = np.zeros((n, dim), dtype=np.float16)
        for i, v in enumerate(vecs):
            if v is not None:
                f[i] = v
        out[f"f_{s}"] = f
        out[f"ms_{s}"] = np.array([r.get(f"ms_{s}", math.nan) for r in rows], dtype=np.float32)
    return out


def _row_bytes(row: dict) -> int:
    n = 96 + row["dist"].nbytes
    for s in STUDENTS:
        v = row.get(f"f_{s}")
        if v is not None:
            n += v.nbytes + 4
    return n


def concat(parts, students: dict, k: int, drop: int = 0, extra=()) -> dict:
    """Segments' columns [(arrays, their students' feature ids)] as one set of columns, the rows below id `drop` left
    out. A student's features from a segment logged with another feature function are left out (zeros, `has_<student>`
    false). `extra`: more per-row columns every part has."""
    names = ("id", "t", "kind", "digest", "stack", "top", "would", "conf", "ms_decisio", "table_conf", "dist", *extra)
    empty = columns([], list(students), k)
    if not parts:
        out = {n: empty[n] if n in empty else np.zeros(0, "S1") for n in names}
    else:
        out = {n: np.concatenate([a[n] for a, _ in parts]) for n in names}
    for s, fid in students.items():
        dims = [a[f"f_{s}"].shape[1] for a, ids in parts if ids.get(s) == fid and f"f_{s}" in a]
        dim = max(dims, default=0)
        feats, ms, has = [], [], []
        for a, ids in parts:
            n = len(a["id"])
            ok = ids.get(s) == fid and f"f_{s}" in a and a[f"f_{s}"].shape[1] == dim and dim > 0
            if ok:
                feats.append(a[f"f_{s}"])
                m = a[f"ms_{s}"].astype(np.float32)
                ms.append(m)
                has.append(np.isfinite(m))
            else:
                feats.append(np.zeros((n, dim), np.float16))
                ms.append(np.full(n, np.nan, np.float32))
                has.append(np.zeros(n, bool))
        out[f"f_{s}"] = np.concatenate(feats) if feats else np.zeros((0, dim), np.float16)
        out[f"ms_{s}"] = np.concatenate(ms) if ms else np.zeros(0, np.float32)
        out[f"has_{s}"] = np.concatenate(has) if has else np.zeros(0, bool)
    keep = out["id"] >= drop
    return {n: v[keep] for n, v in out.items()}


class SegmentLog:
    """A task's rows: `append` a row, `read` every kept row, `drop_before` a row id. Rows are buffered and written as
    immutable segments (`flush`) when the buffer holds `segment_bytes` or is `segment_seconds` old; a small index
    names the segments and the first row id kept. A segment the index does not name (a flush interrupted between the
    two writes) is adopted when the log is opened."""

    def __init__(self, place, students: dict, k: int, clock=time.time, segment_bytes=SEGMENT_BYTES,
                 segment_seconds=SEGMENT_SECONDS):  # fmt: skip
        self.place, self.students, self.k, self.clock = place, dict(students), k, clock  # students: {name: feature id}
        self.segment_bytes, self.segment_seconds = segment_bytes, segment_seconds
        self._lock = threading.RLock()
        self._buf: list[dict] = []
        self._buf_bytes = 0
        self._buf_start: float | None = None
        self.index = self._open()

    # the index

    def _open(self) -> dict:
        raw = self.place.get("log/index.json")
        index = json.loads(raw) if raw else {"format": LOG_FORMAT, "segments": [], "drop_before": 0, "next_id": 0}
        present = set(self.place.names("log/"))
        segs = [s for s in index["segments"] if f"log/{s['name']}" in present]
        listed = {s["name"] for s in segs}
        for name in sorted(present):
            short = name[len("log/") :]
            if short.endswith(".seg") and short not in listed:
                try:
                    arrays, meta = _unnpz(self.place.get(name))
                except Exception as e:  # a damaged segment: named, left in place, not read
                    logger.warning("fast-table log: %s cannot be read (%s); it is left out", name, e)
                    continue
                segs.append({**self._entry(short, arrays, meta), "adopted": True})
        segs.sort(key=lambda s: s["name"])
        index["segments"] = segs
        last = max((s["last"] for s in segs), default=-1)
        index["next_id"] = max(int(index.get("next_id", 0)), last + 1)
        index["seq"] = max(int(index.get("seq", 0)), len(segs))
        return index

    def _entry(self, name, arrays, meta) -> dict:
        ids = arrays["id"]
        return {
            "name": name,
            "rows": int(len(ids)),
            "first": int(ids.min()) if len(ids) else 0,
            "last": int(ids.max()) if len(ids) else -1,
            "bytes": int(meta.get("bytes", 0)),
            "start": meta.get("start"),
            "students": meta.get("students", {}),
        }

    def _write_index(self) -> None:
        self.place.put("log/index.json", json.dumps(self.index, sort_keys=True).encode())

    # rows

    def append(self, row: dict) -> int:
        with self._lock:
            row["id"] = self.index["next_id"]
            self.index["next_id"] += 1
            if not self._buf:
                self._buf_start = self.clock()
            self._buf.append(row)
            self._buf_bytes += _row_bytes(row)
            if self._buf_bytes >= self.segment_bytes:
                self.flush()
            return row["id"]

    def due(self) -> bool:
        return bool(self._buf) and (
            self._buf_bytes >= self.segment_bytes or self.clock() - (self._buf_start or 0) >= self.segment_seconds
        )

    def tick(self) -> bool:
        """Flush the buffer when it is due; True when it did."""
        with self._lock:
            if self.due():
                self.flush()
                return True
            return False

    def flush(self) -> None:
        """Write the buffered rows as one new segment (nothing when the buffer is empty)."""
        with self._lock:
            if not self._buf:
                return
            rows = self._buf
            arrays = columns(rows, list(self.students), self.k)
            seq = int(self.index.get("seq", 0))
            self.index["seq"] = seq + 1
            name = f"{int((self._buf_start or self.clock()) * 1000):013d}-{seq:06d}.seg"
            meta = {
                "format": LOG_FORMAT,
                "students": self.students,
                "k": self.k,
                "start": self._buf_start,
                "end": self.clock(),
                "bytes": self._buf_bytes,
            }
            data = _npz(arrays, meta)
            meta["bytes"] = len(data)
            self.place.put(f"log/{name}", data)  # a new name, never written again
            self.index["segments"].append({**self._entry(name, arrays, meta), "bytes": len(data)})
            self._write_index()
            self._buf, self._buf_bytes, self._buf_start = [], 0, None

    def read(self) -> dict:
        """Every kept row, oldest first, as columns. A student's features from a segment logged with another feature
        function are left out (zeros, `has_<student>` false)."""
        with self._lock:
            segs = list(self.index["segments"])
            buf = list(self._buf)
            drop = self.index["drop_before"]
        parts = []
        for s in segs:
            if s["last"] < drop:
                continue
            data = self.place.get(f"log/{s['name']}")
            if data is None:
                continue
            arrays, meta = _unnpz(data)
            parts.append((arrays, meta.get("students", {})))
        if buf:
            parts.append((columns(buf, list(self.students), self.k), self.students))
        return self._concat(parts, drop)

    def _concat(self, parts, drop) -> dict:
        return concat(parts, self.students, self.k, drop)

    def drop_before(self, row_id: int) -> None:
        """Drop every row whose id is below `row_id`: segments wholly below it are deleted, the rest skip them."""
        with self._lock:
            self.index["drop_before"] = max(int(row_id), int(self.index["drop_before"]))
            gone = [s for s in self.index["segments"] if s["last"] < self.index["drop_before"]]
            self.index["segments"] = [s for s in self.index["segments"] if s["last"] >= self.index["drop_before"]]
            self._buf = [r for r in self._buf if r["id"] >= self.index["drop_before"]]
            self._write_index()
            for s in gone:
                self.place.delete(f"log/{s['name']}")

    def facts(self) -> dict:
        with self._lock:
            drop = self.index["drop_before"]
            kept = sum(max(0, s["last"] - max(s["first"], drop) + 1) for s in self.index["segments"])
            return {
                "kept": kept + len(self._buf),
                "bytes": sum(s["bytes"] for s in self.index["segments"]) + self._buf_bytes,
                "segments": len(self.index["segments"]),
                "buffered": len(self._buf),
            }


# ---- a version ---------------------------------------------------------------------------------------------------


class Version:
    """A promoted table: its record, threshold and certificate, as stored (`d`). Never changed once made."""

    __slots__ = ("d", "number", "kind", "features", "record", "threshold", "alpha", "certificate_kind", "cert_id",
                 "saving", "low_share", "stack")  # fmt: skip

    def __init__(self, d: dict):
        self.d = d
        self.number = d["version"]
        self.kind = d["kind"]
        self.features = d["features"]
        rec = d["record"]
        self.record = {**rec, "coef": FT.unpack(rec["coef"]), "intercept": FT.unpack(rec["intercept"])}
        self.threshold = d["threshold"]
        self.alpha = d["alpha"]
        self.certificate_kind = d["certificate"]["kind"]
        self.cert_id = d["certificate"]["id"]
        self.saving = d["guard"]["saving"]
        self.low_share = d["low_share"]
        self.stack = d["stack"]


def _summary(d: dict) -> dict:
    c = d["certificate"]
    return {
        "version": d["version"],
        "kind": d["kind"],
        "certificate_kind": c["kind"],
        "threshold": d["threshold"],
        "coverage": c["table_share"],
        "saving": d["guard"]["saving"],
        "promoted_at": d["promoted_at"],
        "certificate": c["id"],
    }


# ---- one task's learning ---------------------------------------------------------------------------------------


class Learner:
    """One learning task on this server: its log, its versions, the live one and its guards."""

    def __init__(self, manager: Learning, task: dict, place):
        self.m = manager
        self.task_id, self.key = task["id"], task["key"]
        self.options = list(task["options"])
        self.k = len(self.options)
        self.noul = task.get("type") == "noul"
        # a yes/no table's confidence is read after --noul-commit when the server's rule is on (its stack says so, and
        # a version is served on its own stack only)
        self.commit_band = manager.commit_band() if self.noul else None
        s = task["learn"]
        self.settings = s
        self.request = s["request"]
        self.min_saving = FT.record_min_saving(self.request)
        self.explore = float(s["explore"])
        self.max_rows = int(s["max_rows"])
        self.alpha, self.delta = float(s["alpha"]), float(s["delta"])
        self.mode = "both" if task.get("certificate") else "learning"
        self.registration = s.get("registration")  # on a shared store, which registration of the id this is
        self.place = place
        self.lock = threading.RLock()
        self.students = {f["kind"]: f for f in FT.requested_features(self.request)}
        self.ids = {k: feature_id(f) for k, f in self.students.items()}
        # the layout the task was fitted under (decisio.serve.layout; None: the server's own): its rows, versions and
        # certificates are on that layout's serving stack, the one its requests are answered on
        self.layout = manager.layout_of(task)
        st = self._load_state()
        self.log = self._open_log()
        self.unavailable: dict[str, str] = {}  # a student this server cannot load, and why
        # the key of the rows' digests: random, kept with the task's state (a set seed makes it the seed's)
        seed = manager.seed_for("salt:" + self.task_id)
        fresh = os.urandom(16) if seed is None else seed.to_bytes(8, "little") * 2
        self.salt = bytes.fromhex(st["salt"]) if st.get("salt") else fresh
        self.versions: list[dict] = st.get("versions", [])
        self.next_version = int(st.get("next_version", 1 + max((v["version"] for v in self.versions), default=0)))
        self.suspended = st.get("suspended")
        self.counters = collections.Counter(st.get("counters", {}))
        self.last_refit = st.get("last_refit")
        self.last_refit_at = float(st.get("last_refit_at") or manager.now())
        self.since_refit = int(st.get("since_refit", 0))
        self.labels = self._load_labels()
        self.rng = np.random.default_rng(manager.seed_for(self.task_id))
        self.low: collections.deque = collections.deque(maxlen=LOWCONF_WINDOW)
        self.cusum = 0.0
        self.table_answers = 0
        self.evaluated = 0
        self.would_answer = 0
        self.live: Version | None = None
        self._checked = None
        live = st.get("live")
        if live is not None:
            d = next((v for v in self.versions if v["version"] == live), None)
            if d is not None:
                self._set_live(Version(d), suspended=self.suspended)

    def stack_sha(self) -> str:
        """The serving stack (short) this task's requests are answered on: its layout's."""
        return self.m.stack_sha(self.layout)

    # ---- state ---------------------------------------------------------------------------------------------------

    def _load_state(self) -> dict:
        raw = self.place.get("state.json")
        st = json.loads(raw) if raw else {}
        if st and st.get("key") != self.key:  # the rows and versions of another question under this id: not ours
            self.place.remove_all()
            st = {}
        return st

    def _open_log(self):
        return SegmentLog(self.place, self.ids, self.k, clock=self.m.now)

    def _load_labels(self):
        raw = self.place.get("labels.npz")
        if raw is None:
            return None
        arrays, meta = _unnpz(raw)
        return {"arrays": arrays, "meta": meta}

    def save_labels(self, labels: dict) -> None:
        """The labelled examples' features and gold answers, written once at registration (`labels.npz`)."""
        arrays = {k: np.asarray(v) for k, v in labels.items() if k != "students"}
        meta = {"format": LABELS_FORMAT, "students": labels.get("students", {})}
        self.place.put("labels.npz", _npz(arrays, meta))
        self.labels = {"arrays": arrays, "meta": meta}

    def persist(self) -> None:
        with self.lock:
            st = {
                "format": FORMAT,
                "task_id": self.task_id,
                "key": self.key,
                "salt": self.salt.hex(),
                "versions": self.versions,
                "next_version": self.next_version,
                "live": self.live.number if self.live is not None else None,
                "suspended": self.suspended,
                "counters": dict(self.counters),
                "last_refit": self.last_refit,
                "last_refit_at": self.last_refit_at,
                "since_refit": self.since_refit,
            }
        try:
            self.place.put("state.json", json.dumps(st, sort_keys=True, default=float).encode())
        except OSError as e:  # serving goes on; the state is in memory
            logger.warning("task %s: the fast table's learning state could not be written: %s", self.task_id, e)

    def _set_live(self, v: Version, suspended=None) -> None:
        self.suspended = suspended
        self.low.clear()
        self.cusum = 0.0
        self.table_answers = self.evaluated = self.would_answer = 0
        self._checked = None
        self.live = v  # one assignment: a request reads the old version or this one

    @property
    def off_reason(self) -> str | None:
        """Why the live version cannot be served here (another feature function, another serving stack), or None;
        found once per version, when it is first served."""
        v = self.live
        if v is None:
            return None
        if self._checked is None or self._checked[0] is not v:
            why = FT.cannot_compute(v.features)
            if why:
                why = f"version {v.number} was {why}"
            elif v.stack != self.stack_sha():
                why = (
                    f"version {v.number} was certified on another serving stack (decisio's answers there are another "
                    "teacher's): this server serves through decisio, and its refits make a new version from its own "
                    "rows"
                )
            self._checked = (v, why)
        return self._checked[1]

    def state(self) -> str:
        if self.live is None:
            return "learning"
        if self.off_reason:
            return "off"
        return "suspended" if self.suspended else "active"

    def serving(self) -> Version | None:
        """The version that answers now, or None."""
        v = self.live
        if v is None or self.suspended or self.off_reason:
            return None
        return v

    def has_versions(self) -> bool:
        return bool(self.versions)

    def timing_due(self) -> Version | None:
        """A live version this server is timing before it answers, when this request is to time it too (a version
        learned on a shared store is timed on each replica first: decisio.serve.fast_table_replicas); None on one
        server, whose refits timed it on this host."""
        return None

    def timed(self, v: Version, table_ms: float, decisio_ms: float) -> None:
        """One request timed both ways for `timing_due`'s version."""

    # ---- the request path ----------------------------------------------------------------------------------------

    def digest(self, text: str) -> int:
        h = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), key=self.salt, digest_size=8).digest()
        return int.from_bytes(h, "little")

    def table_read(self, v: Version, text: str):
        """(features, their ms, the table's distribution) of one request by version v."""
        perf = self.m.perf
        t0 = perf()
        X = FT.featurize(v.features, [text])
        t1 = perf()
        P = FT.predict(v.record, X)[0]
        return X[0], (t1 - t0) * 1000.0, P

    def explore_draw(self) -> bool:
        if self.explore <= 0:
            return False
        with self.lock:
            return bool(self.rng.random() < self.explore)

    def spot_check_due(self) -> bool:
        with self.lock:
            self.table_answers += 1
            return self.table_answers % FT.SAMPLE_EVERY == 0

    def observe_confidence(self, v: Version, sure: bool) -> None:
        """The low-confidence-share guard: one request the live table read."""
        with self.lock:
            if self.live is not v:
                return
            self.evaluated += 1
            self.would_answer += bool(sure)
            self.low.append(not sure)
            if len(self.low) == LOWCONF_WINDOW:
                share = sum(self.low) / LOWCONF_WINDOW
                if share > v.low_share + LOWCONF_MARGIN:
                    self.suspend(
                        "low_confidence_share",
                        f"{share:.0%} of the last {LOWCONF_WINDOW} requests had a table confidence under the "
                        f"threshold, against {v.low_share:.0%} on the certificate's rows (alarm above "
                        f"+{LOWCONF_MARGIN:.0%}): the traffic moved away from what the table learned",
                    )

    def observe_check(self, v: Version, disagree: bool) -> None:
        """The CUSUM on disagreement: a spot check, or an explored request the table would have answered."""
        with self.lock:
            if self.live is not v:
                return
            self.counters["checks"] += 1
            self.counters["checks_disagreed"] += bool(disagree)
            self.cusum = max(0.0, self.cusum + float(bool(disagree)) - v.alpha)
            if self.cusum >= CUSUM_LIMIT:
                self.suspend(
                    "disagreement_cusum",
                    f"decisio's answers differ from the table's more often than the certificate's alpha "
                    f"({v.alpha:g}) on the spot checks and explored requests (CUSUM {self.cusum:.2f} reached "
                    f"{CUSUM_LIMIT:g})",
                )

    def suspend(self, alarm: str, reason: str) -> None:
        with self.lock:
            if self.live is None or self.suspended:
                return
            self.suspended = {"alarm": alarm, "reason": reason, "at": self.m.now(), "version": self.live.number}
            self.counters[f"suspended_{alarm}"] += 1
            self.low.clear()
            self.cusum = 0.0
        logger.warning("task %s: the learned fast table (version %s) is suspended: %s", self.task_id,
                       self.live.number, reason)  # fmt: skip
        self.persist()

    def record(self, text: str, feats: dict, p, top: int, ms_decisio: float, kind: int, table_conf=math.nan,
               would_answer: bool = False) -> None:  # fmt: skip
        """Log one request decisio read: `feats` {student: (vector, ms)} already computed for it (a live table's,
        under the same feature function); the static student's are computed here when missing, the encoder's after
        the response by the background queue."""
        row = {
            "t": self.m.now(),
            "kind": int(kind),
            "digest": self.digest(text),
            "stack": self.stack_sha(),
            "top": int(top),
            "dist": np.asarray(p, dtype=np.float16),
            "conf": float(np.max(p)),
            "ms_decisio": float(ms_decisio),
            "table_conf": float(table_conf),
            "would": bool(would_answer),
        }
        pending = None
        for s, f in self.students.items():
            got = feats.get(s)
            if got is not None and got[2] == self.ids[s]:
                row[f"f_{s}"] = np.asarray(got[0], dtype=np.float16)
                row[f"ms_{s}"] = float(got[1])
            elif s in self.unavailable:
                continue
            elif s == "static":
                try:
                    t0 = self.m.perf()
                    x = FT.featurize(f, [text])[0]
                    row["f_static"], row["ms_static"] = x.astype(np.float16), (self.m.perf() - t0) * 1000.0
                except FT.TableUnavailable as e:
                    self.unavailable[s] = str(e)
            else:
                pending = s
        if pending is not None:
            self.m.enqueue(self, row, text, pending)
        else:
            self.append(row)

    def append(self, row: dict) -> None:
        with self.lock:
            self.log.append(row)
            self.counters["logged"] += 1
            self.counters[ROW_KINDS[row["kind"]]] += 1
            if row["kind"] == EXPLORED and row["would"]:
                self.counters["explored_would_answer"] += 1
            self.since_refit += 1
            due = self.since_refit >= REFIT_ROWS
        if due:
            self.m.notify()

    # ---- refits ---------------------------------------------------------------------------------------------------

    def due(self) -> bool:
        if self.log.facts()["kept"] < FT.MIN_EXAMPLES:
            return False
        return self.since_refit >= REFIT_ROWS or (
            self.since_refit > 0 and self.m.now() - self.last_refit_at >= REFIT_SECONDS
        )

    def refit(self) -> dict:
        """Fit, certify and maybe promote one version from the kept rows; returns the outcome (also kept)."""
        started = self.m.now()
        with self.lock:
            self.since_refit = 0
            self.last_refit_at = started
        rows, n_all = self._kept_rows()
        try:
            out = self._refit(rows)
        except Exception as e:  # a refit never stops serving; it says what went wrong
            logger.exception("task %s: a fast-table refit failed", self.task_id)
            out = {"outcome": "error", "reason": f"{type(e).__name__}: {e}"}
        out = {"at": started, "rows": int(min(n_all, self.max_rows)), **out}
        with self.lock:
            self.last_refit = out
            self.counters["refits"] += 1
            self.counters[f"refit_{out['outcome']}"] += 1
        self.persist()
        return out

    def _kept_rows(self) -> tuple[dict, int]:
        """(the rows a refit fits and certifies on, and how many were kept before `max_rows`)."""
        self.log.flush()
        rows = self.log.read()
        n_all = len(rows["id"])
        if n_all > self.max_rows:  # the oldest rows past max_rows go
            self.log.drop_before(int(rows["id"][n_all - self.max_rows]))
            rows = {k: v[n_all - self.max_rows :] for k, v in rows.items()}
        return rows, n_all

    def _certificate_rows(self, rows: dict, ci, held_unbiased) -> dict:
        """More of what an agreement certificate records about the rows it drew on: nothing on one server (on a shared
        store, the replicas)."""
        return {}

    def _refit(self, rows: dict) -> dict:
        stack = self.stack_sha()
        on_stack = rows["stack"] == stack.encode()
        bucket = (rows["digest"] % np.uint64(10_000)).astype(np.int64)
        held = bucket < int(HOLDOUT * 10_000)
        unbiased = np.isin(rows["kind"], CERTIFY_KINDS)
        fit_mask = on_stack & ~held
        cert_mask = on_stack & held & unbiased
        # a request the live table would have answered was logged only when explored or spot-checked: its row stands
        # for 1 / that chance of them; every other row was logged whatever happened (weight 1)
        p_logged = self.explore + (1.0 - self.explore) / FT.SAMPLE_EVERY
        weight = np.ones(len(rows["id"]))
        weight[np.isin(rows["kind"], (EXPLORED, SPOT_CHECK)) & rows["would"]] = 1.0 / p_logged
        target = rows["top"].astype(int)
        lab = self.labels["arrays"] if self.labels else None
        lab_ids = self.labels["meta"].get("students", {}) if self.labels else {}
        n_usable = int(fit_mask.sum()) + (len(lab["fit_y"]) if lab is not None else 0)
        if n_usable < FT.MIN_EXAMPLES:
            return {
                "outcome": "too_few_rows",
                "reason": f"{n_usable} rows to fit on this serving stack, and at least {FT.MIN_EXAMPLES} are needed",
            }
        need = certify.minimum_examples(self.alpha, self.delta)
        candidates = []
        for s, feat in self.students.items():
            c = {"kind": s, "features": feat, "holds": False, "saving": -math.inf}
            candidates.append(c)
            why = FT.cannot_compute(feat) or self.unavailable.get(s)
            if why:
                c["why_not"] = why
                continue
            has = rows[f"has_{s}"] if f"has_{s}" in rows else np.zeros(len(rows["id"]), bool)
            fi, ci = fit_mask & has, cert_mask & has
            X = rows[f"f_{s}"][fi].astype(np.float32)
            y, w = target[fi], weight[fi]
            gold_ok = lab is not None and lab_ids.get(s) == self.ids[s] and f"cert_X_{s}" in lab
            if lab is not None and lab_ids.get(s) == self.ids[s] and f"fit_X_{s}" in lab:
                X = np.vstack([X, lab[f"fit_X_{s}"].astype(np.float32)])
                y = np.concatenate([y, lab["fit_y"].astype(int)])
                w = np.concatenate([w, np.full(len(lab["fit_y"]), LABEL_WEIGHT)])
            c["fit_rows"] = int(len(y))
            if len(y) < FT.MIN_EXAMPLES:
                c["why_not"] = f"{len(y)} rows with {s} features to fit on, and at least {FT.MIN_EXAMPLES} are needed"
                continue
            if len(set(y.tolist())) < 2:
                c["why_not"] = "decisio chose one option on every row: there is nothing for a table to tell apart"
                continue
            rec = _fit(X, y, self.k, feat, w)
            if gold_ok and len(lab["cert_y"]) >= need:
                Xc = lab[f"cert_X_{s}"].astype(np.float32)
                ref = lab["cert_y"].astype(int)
                ms_t, ms_d = lab[f"cert_ms_{s}"].astype(float), lab["cert_ms_decisio"].astype(float)
                ckind, rows_by, drawn = "gold", {"labelled": int(len(ref))}, {}
            else:
                Xc = rows[f"f_{s}"][ci].astype(np.float32)
                ref = target[ci]
                ms_t, ms_d = rows[f"ms_{s}"][ci].astype(float), rows["ms_decisio"][ci].astype(float)
                ckind = "agreement"
                rows_by = {ROW_KINDS[kk]: int((rows["kind"][ci] == kk).sum()) for kk in CERTIFY_KINDS}
                drawn = self._certificate_rows(rows, ci, held & unbiased & has)
            c["certificate_kind"], c["cert_rows"] = ckind, rows_by
            if len(ref) < need:
                c["why_not"] = (
                    f"{len(ref)} held-out rows logged independently of the table, and at least {need} are needed for "
                    f"alpha {self.alpha:g} and delta {self.delta:g}"
                )
                continue
            P = FT.predict(rec, Xc)
            if self.noul:  # its own P(yes), calibrated on the certificate's rows against their reference answers
                rec["calibration"] = FT.fit_temperature(P, ref == 0, reference=ckind)
                P = FT.predict(rec, Xc)
            cT, pT = FT.top_confidence(P, self.commit_band), _tops(P, self.options)
            correct = pT == ref
            thr = certify.threshold(cT, correct, self.alpha, self.delta)
            holds = bool(np.isfinite(thr))
            by = cT >= thr if holds else np.zeros(len(ref), bool)
            answered, wrong = int(by.sum()), int((by & ~correct).sum())
            share = float(by.mean()) if len(ref) else 0.0
            predict_ms = self._predict_ms(rec, Xc)
            # the expected saving and the share under the threshold are the traffic's: on the held-out rows logged
            # independently of the table, with the times this host took on them, where there are enough of them (a
            # gold certificate's labelled rows were timed at registration, on traffic that may not be like them)
            if ckind == "gold" and int(ci.sum()) >= need:
                cT_t = FT.top_confidence(FT.predict(rec, rows[f"f_{s}"][ci].astype(np.float32)), self.commit_band)
                ms_t, ms_d = rows[f"ms_{s}"][ci].astype(float), rows["ms_decisio"][ci].astype(float)
                timed_on = "traffic"
            else:
                cT_t, timed_on = cT, ("traffic" if ckind == "agreement" else "labelled")
            by_t = cT_t >= thr if holds else np.zeros(len(cT_t), bool)
            guard = FT.timing_guard(ms_t + predict_ms, ms_d, float(by_t.mean()) if len(by_t) else 0.0, self.min_saving,
                                    deferred=~by_t)  # fmt: skip
            guard["timed_on"] = timed_on
            low_share = float((cT_t < thr).mean()) if holds and len(cT_t) else 1.0
            c.update(
                record=rec,
                holds=holds,
                threshold=float(thr) if holds else None,
                saving=guard["saving"],
                guard=guard,
                low_share=low_share,
                certificate={
                    "kind": ckind,
                    "procedure": certify.PROCEDURE,
                    "promise": (
                        "at most alpha of the table's answers differ from decisio's"
                        if ckind == "agreement"
                        else "at most alpha of the table's answers are wrong against the labelled answers"
                    ),
                    "alpha": self.alpha,
                    "delta": self.delta,
                    "threshold": float(thr) if holds else None,
                    "n": int(len(ref)),
                    "answered": answered,
                    "errors": wrong,
                    "error_rate": wrong / answered if answered else None,
                    "upper_bound": certify.upper_bound(wrong, answered, self.delta) if answered else None,
                    "table_share": share,
                    "rows": rows_by,
                    **drawn,
                    "stack": stack,
                },
            )
            if not holds:
                c["why_not"] = (
                    f"on the {len(ref)} held-out rows no confidence threshold keeps the table's "
                    + ("disagreements with decisio" if ckind == "agreement" else "wrong answers")
                    + f" at {self.alpha:.0%} or fewer with probability {1 - self.delta:.0%}"
                )
            elif not guard["kept"]:
                c["why_not"] = (
                    f"it would save {guard['saving']:.0%} of the time per request, under the "
                    f"{self.min_saving:.0%} required"
                )
        tried = [_brief(c) for c in candidates]
        eligible = [c for c in candidates if "why_not" not in c]
        if self.request["kind"] == "auto":
            ok = [c for c in candidates if "record" in c]
            idx, why = FT.choose(ok, self.min_saving) if ok else (None, "no kind could be fitted")
            chosen = ok[idx] if idx is not None else None
        else:
            chosen = eligible[0] if eligible else None
            why = None if chosen else candidates[0].get("why_not")
        live = self.live
        if chosen is None:
            return {
                "outcome": "not_certified",
                "reason": why + ("; the live version stays live" if live is not None else ""),
                "kinds": tried,
            }
        ref = None  # what a new version must save more than: the live table's expected saving when it went live
        if live is not None and self.state() == "active":
            ref, name = live.saving, f"the live version {live.number}'s"
        elif live is None:  # a labelled task's own table, until a learned version replaces it
            ft = (self.m.task(self.task_id) or {}).get("fast_table")
            if ft and not ft.get("suspended"):
                ref, name = ft["timing"]["guard"]["saving"], "the labelled table's"
        if ref is not None and chosen["saving"] <= ref:
            return {
                "outcome": "not_promoted",
                "reason": (
                    f"version {self.next_version}'s {chosen['kind']} table would save {chosen['saving']:.1%}, not more "
                    f"than {name} {ref:.1%}: the live table stays live"
                ),
                "kinds": tried,
            }
        d = self._promote(chosen, stack)
        return {"outcome": "promoted", "version": d["version"], "kind": d["kind"], "kinds": tried,
                **({"why": why} if why and self.request["kind"] == "auto" else {})}  # fmt: skip

    def _predict_ms(self, rec, X) -> float:
        perf, ms = self.m.perf, []
        for i in range(min(TIMED_PREDICTS, len(X))):
            t0 = perf()
            FT.predict(rec, X[i : i + 1])
            ms.append((perf() - t0) * 1000.0)
        return float(np.mean(ms)) if ms else 0.0

    def _promote(self, c: dict, stack: str) -> dict:
        with self.lock:
            n = self.next_version
            self.next_version += 1
            ident = sha256([self.task_id, n, c["threshold"], stack])[:8]
            cert = {**c["certificate"], "id": f"{self.task_id}#learned-v{n}-{ident}"}
            d = {
                "version": n,
                "kind": c["kind"],
                "features": c["features"],
                "record": c["record"],
                "threshold": c["threshold"],
                "alpha": self.alpha,
                "certificate": cert,
                "guard": c["guard"],
                "low_share": c["low_share"],
                "stack": stack,
                "promoted_at": self.m.now(),
                "fit_rows": c["fit_rows"],
            }
            self.versions = (self.versions + [d])[-VERSIONS_KEPT:]
            self.counters["promotions"] += 1
            self._set_live(Version(d))
        logger.info("task %s: the learned fast table version %s (%s) is live", self.task_id, n, c["kind"])
        return d

    def rollback(self, number: int) -> Version:
        d = next((v for v in self.versions if v["version"] == number), None)
        if d is None:
            kept = [v["version"] for v in self.versions]
            raise ValueError(f"fast_table.rollback: no version {number} is kept (the kept versions are {kept})")
        why = FT.cannot_compute(d["features"])
        if why:
            raise ValueError(f"fast_table.rollback: version {number} was {why}")
        with self.lock:
            self._set_live(Version(d))
            self.counters["rollbacks"] += 1
        self.persist()
        return self.live

    # ---- status --------------------------------------------------------------------------------------------------

    def facts(self) -> dict:
        v = self.live
        log = self.log.facts()
        st = self.state()
        with self.lock:
            c = dict(self.counters)
        out = {"mode": self.mode, "state": st}
        if st == "off":
            out["reason"] = self.off_reason
        elif st == "suspended":
            out["reason"] = self.suspended["reason"]
            out["suspended"] = self.suspended
        elif st == "learning":
            out["reason"] = "decisio answers every request until a refit certifies a table on the logged rows"
        out["kind"] = {
            "asked": self.request["kind"],
            **({"static_model": self.request["static_model"]} if "static_model" in self.request else {}),
            "chosen": v.kind if v is not None else None,
        }
        if self.unavailable:
            out["kind"]["unavailable"] = dict(self.unavailable)
        out["rows"] = {
            "kept": log["kept"],
            "since_refit": self.since_refit,
            "bytes": log["bytes"],
            "logged": c.get("logged", 0),
            "by_kind": {k: c.get(k, 0) for k in ROW_KINDS},
            "max_rows": self.max_rows,
            "stored": (
                f"segment files under {self.place.describe()}/log"
                if self.place.persistent
                else "in memory: the rows are lost at restart (start the server with --tasks-dir to keep them)"
            ),
        }
        if "encoder" in self.students:
            out["rows"]["encoder_dropped"] = c.get("encoder_dropped", 0)
        out["version"] = {"live": v.number if v is not None else None, "kept": [_summary(d) for d in self.versions]}
        out["coverage"] = (
            {
                "certified": v.d["certificate"]["table_share"],
                "observed": self.would_answer / self.evaluated if self.evaluated else None,
                "requests_read": self.evaluated,
            }
            if v is not None
            else None
        )
        out["saving"] = v.saving if v is not None else None
        out["min_saving"] = self.min_saving  # what a version must save to be promoted (on a replica: to answer)
        out["certificate"] = dict(v.d["certificate"]) if v is not None else None
        cov = v.d["certificate"]["table_share"] if v is not None else 0.0
        out["explore"] = {
            "fraction": self.explore,
            "explored": c.get("explored", 0),
            "explored_would_answer": c.get("explored_would_answer", 0),
            "cost": round(self.explore * cov, 4),
            "rule": "a request the live table reads is explored with this chance; one the table would answer is then "
            "answered by decisio (the cost is this share of the table's answers)",
        }
        low = list(self.low)
        out["guards"] = {
            "low_confidence_share": {
                "window": LOWCONF_WINDOW,
                "requests": len(low),
                "share": sum(low) / len(low) if low else None,
                "certified_share": v.low_share if v is not None else None,
                "margin": LOWCONF_MARGIN,
                "alarm": bool(self.suspended and self.suspended.get("alarm") == "low_confidence_share"),
            },
            "disagreement_cusum": {
                "statistic": round(self.cusum, 4),
                "limit": CUSUM_LIMIT,
                "reference": v.alpha if v is not None else self.alpha,
                "checks": c.get("checks", 0),
                "disagreed": c.get("checks_disagreed", 0),
                "alarm": bool(self.suspended and self.suspended.get("alarm") == "disagreement_cusum"),
            },
        }
        out["refit"] = {
            "last": self.last_refit,
            "every_rows": REFIT_ROWS,
            "every_minutes": REFIT_SECONDS / 60,
            "min_rows": FT.MIN_EXAMPLES,
            "refits": c.get("refits", 0),
            "promotions": c.get("promotions", 0),
        }
        if self.labels is not None:
            a = self.labels["arrays"]
            out["labels"] = {"fit": int(len(a.get("fit_y", []))), "held_out": int(len(a.get("cert_y", [])))}
        return out


def _brief(c: dict) -> dict:
    out = {"kind": c["kind"], "eligible": "why_not" not in c, "fit_rows": c.get("fit_rows")}
    for k in ("certificate_kind", "threshold", "saving", "cert_rows"):
        if k in c and c[k] is not None and c[k] != -math.inf:
            out[k] = c[k]
    if "certificate" in c:
        out["table_share"] = round(c["certificate"]["table_share"], 4)
        out["upper_bound"] = c["certificate"]["upper_bound"]
    if "why_not" in c:
        out["why_not"] = c["why_not"]
    return out


def _tops(P: np.ndarray, keys: list[str]) -> np.ndarray:
    """The declared choice of each row (the most probable option; an exact tie to the key that sorts first)."""
    m = P.max(axis=1, keepdims=True)
    order = np.argsort(np.array(keys, dtype=object))  # positions by key
    rank = np.empty(len(keys), dtype=int)
    rank[order] = np.arange(len(keys))
    tied = np.where(P == m, rank[None, :], len(keys))
    return order[tied.min(axis=1)]


def _fit(X, y, k, feat, w) -> dict:
    """`fast_table.fit` with sample weights and one BLAS thread."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # scikit-learn depends on it; without it the fit runs as configured
        return FT.fit(X, y, k, features=feat, sample_weight=w)
    with threadpool_limits(limits=1):
        return FT.fit(X, y, k, features=feat, sample_weight=w)


# ---- the server's learning ------------------------------------------------------------------------------------


class Learning:
    """Every learning task's learner on this server, the refit worker and the encoder's background queue.

    `background`: run the refit worker and the encoder queue on their own threads (the server); when false, refits run
    when `refit_task` is called and queued encoder rows when `drain` is (tests). `shared`: learning shared across the
    replicas of a shared task store (`decisio.serve.fast_table_replicas.Shared`), or None: this server alone."""

    def __init__(self, root=None, stack_sha=None, perf=time.perf_counter, now=time.time, background=None, task=None,
                 shared=None, layout_of=None, commit_band=None):  # fmt: skip
        self.root = Path(root) if root is not None else None  # <tasks-dir>/learn, or None: in memory
        # `stack_sha()`: the server's serving stack, short; `stack_sha(layout)`: the stack a request of that layout is
        # answered on. `layout_of(task)`: the layout a task was fitted under, or None (the server's own)
        self._stack_sha = stack_sha or (lambda layout=None: "0" * 16)
        self._stack_values: dict = {}
        self.layout_of = layout_of or (lambda task: None)
        self.task = task or (lambda task_id: None)  # the task record of an id, as the server holds it
        # `commit_band()`: the server's --noul-commit band (lo, hi) when the rule is on, else None
        self.commit_band = commit_band or (lambda: None)
        self.perf, self.now = perf, now
        self.background = BACKGROUND if background is None else background
        self.seed: int | None = SEED
        self.learners: dict[str, Learner] = {}
        self._places: dict[str, object] = {}
        self._lock = threading.Lock()
        self._refit_lock = threading.Lock()
        self._event = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._encoder: threading.Thread | None = None
        self.encode_q: queue.Queue = queue.Queue(maxsize=ENCODE_QUEUE)
        self.encoder_dropped = 0
        self.worker = {"state": "not started", "refits": 0, "cpu_seconds": 0.0, "last": None}
        self._closed = False
        self.shared = shared
        if shared is not None:
            shared.attach(self)

    def stack_sha(self, layout: str | None = None) -> str:
        """The serving stack's fingerprint (short) under `layout` (None: the server's own), taken once per layout: the
        server's stacks do not change while it runs."""
        if layout not in self._stack_values:
            self._stack_values[layout] = self._stack_sha() if layout is None else self._stack_sha(layout)
        return self._stack_values[layout]

    def seed_for(self, task_id: str):
        if self.seed is None:
            return None
        return int.from_bytes(hashlib.sha256(f"{self.seed}:{task_id}".encode()).digest()[:8], "little")

    # learners

    def place_for(self, task_id: str):
        if self.root is None:
            return self._places.setdefault(task_id, MemoryPlace())
        return DirPlace(self.root / task_digest(task_id))

    def get(self, t) -> Learner | None:
        """The learner of task t (made, or resumed from its files, on first use), or None when t does not learn."""
        if not t or not t.get("learn"):
            return None
        L = self.learners.get(t["id"])
        if L is not None and self._same(L, t):
            return L
        with self._lock:
            L = self.learners.get(t["id"])
            if L is None or not self._same(L, t):
                L = self.learners[t["id"]] = self._make(t)
        self.start()
        return L

    @staticmethod
    def _same(L: Learner, t) -> bool:
        return L.key == t["key"] and L.registration == t["learn"].get("registration")

    def _make(self, t) -> Learner:
        if self.shared is not None:
            return self.shared.learner(t)
        return Learner(self, t, self.place_for(t["id"]))

    def new_registration(self) -> dict:
        """What a new registration adds to its `learn` settings: on a shared store an id of its own, under which its
        rows and versions are kept (a task registered again starts afresh); nothing on one server."""
        return {"registration": uuid.uuid4().hex[:16]} if self.shared is not None else {}

    def start_task(self, t, labels: dict | None = None) -> Learner:
        """A new learning task (registered, or registered again): earlier rows and versions of its id are dropped."""
        self.drop(t["id"])
        L = self.get(t)
        if labels:
            L.save_labels(labels)
        L.persist()
        if self.shared is not None:
            self.shared.started(L)
        return L

    def drop(self, task_id: str) -> None:
        with self._lock:
            L = self.learners.pop(task_id, None)
            place = self._places.pop(task_id, None) if self.root is None else self.place_for(task_id)
        if self.shared is not None:  # its rows and versions in the shared store go with it
            if L is not None:
                self.shared.remove(L)
            return
        if place is not None:
            place.remove_all()

    def forget(self, task_id: str) -> None:
        """A task another replica removed or registered again: this server's learner goes, the store is left alone."""
        with self._lock:
            self.learners.pop(task_id, None)

    def sync(self) -> None:
        """Follow the shared store (the task store's poll calls it): nothing on one server."""
        if self.shared is not None:
            self.shared.tick()

    # the worker

    def start(self) -> None:
        if not self.background or self._closed:
            return
        with self._lock:
            if self._worker is None:
                self._worker = threading.Thread(target=self._loop, name="decisio-fast-table-learner", daemon=True)
                self._worker.start()
                self.worker["state"] = "idle"
                ref = weakref.ref(self)
                atexit.register(lambda: (lambda m: m and m.close())(ref()))

    def notify(self) -> None:
        self._event.set()

    def _loop(self) -> None:
        lower_priority()
        while not self._stop.is_set():
            self._event.wait(WORKER_TICK)
            self._event.clear()
            if self._stop.is_set():
                break
            try:
                self.run_due()
            except Exception:  # the worker keeps going; the failure is logged
                logger.exception("the fast-table learner's pass failed")

    def run_due(self) -> None:
        for L in list(self.learners.values()):
            if L.log.tick():
                L.persist()  # the counters with the rows they count
            if L.due():
                self.refit(L)

    def refit(self, L: Learner) -> dict:
        with self._refit_lock:  # one refit at a time on this server
            self.worker["state"] = f"refitting {L.task_id}"
            c0 = time.thread_time()
            try:
                out = L.refit()
            finally:
                self.worker["cpu_seconds"] = round(self.worker["cpu_seconds"] + time.thread_time() - c0, 3)
                self.worker["refits"] += 1
                self.worker["state"] = "idle" if self._worker is not None else "not started"
            self.worker["last"] = {"task": L.task_id, "outcome": out["outcome"], "at": out["at"]}
            return out

    def refit_task(self, task_id: str) -> dict:
        L = self.learners.get(task_id)
        if L is None:
            raise KeyError(task_id)
        return self.refit(L)

    # the encoder's background queue

    def enqueue(self, L: Learner, row: dict, text: str, student: str) -> None:
        """Queue a row for its encoder features, computed after the response; a full queue logs it without them."""
        try:
            self.encode_q.put_nowait((L, row, text, student))
        except queue.Full:
            self.encoder_dropped += 1
            with L.lock:
                L.counters["encoder_dropped"] += 1
            L.append(row)
            return
        if self.background and not self._closed:
            with self._lock:
                if self._encoder is None:
                    self._encoder = threading.Thread(target=self._encode_loop, name="decisio-fast-table-encoder",
                                                     daemon=True)  # fmt: skip
                    self._encoder.start()

    def _encode_one(self, item) -> None:
        L, row, text, s = item
        try:
            t0 = self.perf()
            x = FT.featurize(L.students[s], [text])[0]
            row[f"f_{s}"], row[f"ms_{s}"] = np.asarray(x, dtype=np.float16), (self.perf() - t0) * 1000.0
        except FT.TableUnavailable as e:
            L.unavailable[s] = str(e)
        except Exception as e:  # the row is logged without the features
            logger.warning("task %s: the encoder features of a logged row failed: %s", L.task_id, e)
        L.append(row)

    def _encode_loop(self) -> None:
        lower_priority()
        while not self._stop.is_set():
            try:
                item = self.encode_q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._encode_one(item)
            finally:
                self.encode_q.task_done()

    def drain(self) -> None:
        """Wait until every queued row has its encoder features (computed here when there is no encoder thread)."""
        if self._encoder is not None and self._encoder.is_alive():
            self.encode_q.join()
            return
        while True:
            try:
                item = self.encode_q.get_nowait()
            except queue.Empty:
                return
            try:
                self._encode_one(item)
            finally:
                self.encode_q.task_done()

    def close(self) -> None:
        """Stop the threads and write every buffered row and every learner's state (the server is stopping)."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._event.set()
        while True:  # rows still waiting for their encoder features are logged without them
            try:
                L, row, _, _ = self.encode_q.get_nowait()
            except queue.Empty:
                break
            with L.lock:
                L.counters["encoder_dropped"] += 1
            L.append(row)
            self.encode_q.task_done()
        for L in list(self.learners.values()):
            try:
                L.log.flush()
                L.persist()
            except OSError as e:
                logger.warning("task %s: the fast table's rows could not be written at stop: %s", L.task_id, e)
        if self.shared is not None:
            self.shared.close()

    def health(self) -> dict:
        learners = list(self.learners.values())
        return {
            "tasks": len(learners),
            "rows_kept": sum(L.log.facts()["kept"] for L in learners),
            "rows_logged": sum(L.counters.get("logged", 0) for L in learners),
            "stored": "disk" if self.root is not None else "memory (lost at restart)",
            "states": dict(collections.Counter(L.state() for L in learners)),
            "worker": dict(self.worker),
            "encoder_queue": {
                "queued": self.encode_q.qsize(),
                "capacity": ENCODE_QUEUE,
                "dropped": self.encoder_dropped,
            },
            **({"stored": "the shared task store", "shared": self.shared.health()} if self.shared is not None else {}),
        }


def lower_priority() -> None:
    """Run the calling thread at nice NICE where a thread can have its own (Linux); elsewhere leave it."""
    if not sys.platform.startswith("linux"):
        return
    with contextlib.suppress(OSError, AttributeError):
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), NICE)
