# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Labels from traffic: a request id on every answer of a task that asks for labels, a probability draw of the answers
to ask a person about, a log of the drawn answers, and the labels posted back (docs/tasks.md, "Labels from traffic").

What a label is good for depends on why that answer was labelled, so the server, not the client, chooses which
answers to ask about. Each served question of such a task gets one uniform per slice, `u = H(salt, request id,
question, slice)` (a keyed hash, so any replica draws the same and the draw cannot depend on the right answer), and is
in the slice when `u < pi`, where `pi` depends only on what is known when the answer is served:

  audit    a constant `pi` over all traffic: accuracy and calibration estimates, certificates and drift checks
  cert     a constant `pi` where the served confidence is at or above `tau_min`, else 0: labels where a certificate
           needs them (the region above the lowest threshold a certificate could pick)
  margin   `pi` rising as the top two options get closer (between `margin_pi_min` and `margin_pi_max`): labels where a
           refit learns most; weighted by 1 / `pi`, never used for a certificate

A `pi` is 0 or at least `PI_MIN`, so a weight is at most 1 / `PI_MIN`. The settings in force are an epoch; a row
records its epoch, and its `pi` is that epoch's. Labels a client posts for answers the server did not draw are kept as
`self` (or `import` when no logged answer matches): label review and error analysis, never a certificate, a
calibration or a monitor.

A label is an event, appended and flushed before the write is acknowledged; rows and events are joined when they are
read, by request id and question, the latest label winning. A drawn row whose label does not come by its due time is
`expired`, and a certificate's strict rule counts it as an error. Fits record which labels they used (`fit_use`); a
certificate never uses a label its version, or a version it inherits from, was fitted on (`assert_disjoint`).

No request text is kept: a row holds the served distribution and the plain readout's label log-probabilities (float32,
the numbers a new version re-scores), the confidence, the margin, two digests of the input (a salted one, and `hk1`, an
HMAC under the task's digest key, which a client that kept the text but not the request id can compute), the language
tag the request gave, the script class of the text, and for a task with a head the drawn rows' hidden state (with
`features: "full"`, the default; `"minimal"` keeps neither the distributions nor h).

Standard library and numpy; the storage is the learner's (`fast_table_learn.DirPlace`, `MemoryPlace`)."""

from __future__ import annotations

import base64
import collections
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import unicodedata

import numpy as np

SLICES = ("audit", "cert", "margin")
EVAL_SLICES = ("audit", "cert")  # the slices certificates and drift checks read; never used to fit
FIT_SLICES = ("margin",)  # probability-sampled slices a refit may use (weighted by 1 / pi)
PI_MIN = 0.02  # a slice's pi is 0 or at least this: inverse-probability weights stay at most 50
DEFAULTS = {
    "features": "full",
    "audit_pi": 0.02,
    "cert_pi": 0.05,
    "margin_pi_min": 0.0,
    "margin_pi_max": 0.2,
    "margin_scale": 0.1,  # the margin (top-two gap) at which the margin pi is halfway between its min and max
    "region": 0.5,  # the share of traffic the cert slice's region should cover once tau_min is set
    "tau_after": 200,  # rows seen before tau_min is set from their confidences (all traffic until then)
    "due_days": 14,
    "max_rows": 50_000,
}
FEATURES = ("full", "minimal")
SEGMENT_ROWS = 2_000  # rows buffered before a segment is written
SEGMENT_SECONDS = 60  # and at most this old when written
FLUSH_EVERY = 10  # seconds between the server's checks for a segment that is due
MAX_BATCH = 1_000  # labels per write
REFUSED_TEXT = (
    "the labels log keeps no request text: give the request id, the digest, or the state or text to be hashed"
)


class LabelsError(ValueError):
    """A labels request decisio refuses (the message says why)."""


# ---- settings ------------------------------------------------------------------------------------------------------


def parse_settings(block) -> dict | None:
    """A registration's `labels` block as the stored settings (defaults filled in), or None without one. `true` asks
    for the defaults; `{"enabled": false}` is the same as no block. Raises LabelsError on an unknown or invalid
    setting."""
    if block is None or block is False:
        return None
    if block is True:
        block = {}
    if not isinstance(block, dict):
        raise LabelsError('labels is true or an object of settings, e.g. {"cert_pi": 0.1}')
    if not isinstance(block.get("enabled", True), bool):
        raise LabelsError("labels.enabled is true or false")
    if block.get("enabled") is False:
        return None
    unknown = sorted(set(block) - set(DEFAULTS) - {"enabled"})
    if unknown:
        raise LabelsError(f"labels: unknown settings {unknown}; the settings are {sorted(DEFAULTS)}")
    s = {**DEFAULTS, **{k: v for k, v in block.items() if k != "enabled"}}
    if s["features"] not in FEATURES:
        raise LabelsError(f"labels.features is one of {list(FEATURES)}")
    for k in ("audit_pi", "cert_pi", "margin_pi_min", "margin_pi_max"):
        v = s[k]
        if not isinstance(v, (int, float)) or not (v == 0 or PI_MIN <= v <= 1):
            raise LabelsError(f"labels.{k} is 0 or between {PI_MIN} and 1, got {v!r}")
    if s["margin_pi_min"] > s["margin_pi_max"]:
        raise LabelsError("labels.margin_pi_min is at most labels.margin_pi_max")
    if not 0 < float(s["margin_scale"]) <= 1 or not 0 < float(s["region"]) <= 1:
        raise LabelsError("labels.margin_scale and labels.region are fractions above 0 and at most 1")
    for k in ("tau_after", "due_days", "max_rows"):
        if not isinstance(s[k], int) or s[k] < 1:
            raise LabelsError(f"labels.{k} is a positive whole number")
    return s


STORES = (
    "for every answered question of this task: a request id, the time, the answer, its confidence and the gap to the "
    "next option, the language tag the request gave (meta.lang or x-decisio-lang) and the script class of its text, "
    "two digests of the input (salted; not reversible without the task's secrets), the certificate and the serving "
    "stack it was answered under, and whether it was drawn to be labelled",
    'with features "full" (the default): also the served distribution and the plain readout\'s label '
    "log-probabilities, and for drawn questions of a task with an intent head the hidden state at the answer position "
    "(a model feature that is not anonymous; it stays in this server's tasks directory)",
    "never the request's text or state",
    "the labels you post, with who chose the item (the draw, or you)",
)


ROW_FIELDS = ("request_id", "question", "t", "epoch", "digest", "hk1", "lang", "script", "keys", "top", "conf",
              "margin", "served_certified", "via", "version", "stack", "draws", "drawn", "replica")  # fmt: skip
FULL_FIELDS = ("ps", "lp", "h")  # with features "full": the served distribution, the plain readout, the hidden state


def stored_fields(settings: dict) -> list[str]:
    """The fields a row of this task's log has, as the mode in force stores them (`lp`, `h` null on undrawn rows)."""
    return list(ROW_FIELDS) + (list(FULL_FIELDS) if settings["features"] == "full" else [])


def registration_block(rec: dict, durable: bool) -> dict:
    """What a registration's response says of the task's labels: the settings, the digest key (shown here only), what
    the log stores and where, and the queue's first view (empty until the task has answered)."""
    s = rec["settings"]
    return {
        "enabled": True,
        "text_stored": False,
        "features": s["features"],
        "stored_fields": stored_fields(s),
        "retention": {"max_rows": s["max_rows"], "due_days": s["due_days"], "deleted_with_task": True},
        "location": "tasks_dir" if durable else "memory",
        "settings": s,
        "digest_key": rec["secret"]["digest_key"],
        "digest": "hk1: then the hex HMAC-SHA256, under digest_key, of the request's state as text (a string as it "
        "is, anything else as JSON with sorted keys, no spaces and unescaped Unicode), in NFC with whitespace runs as "
        "one space and the ends stripped",
        "stores": list(STORES),
        "next_to_label": {"items": [], "pending": 0, "available": False, "reason": "no traffic yet"},
    }


def new_secrets() -> dict:
    """A task's two label secrets: the draw's salt (never shown) and the digest key (returned once, at registration)."""
    return {"salt": secrets.token_hex(32), "digest_key": secrets.token_hex(32)}


# ---- ids, digests, the draw ----------------------------------------------------------------------------------------

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid(now: float | None = None) -> str:
    """A 26-character ULID: 48 bits of milliseconds then 80 random bits, Crockford base32 (sorts by time)."""
    ms = int((time.time() if now is None else now) * 1000)
    n = (ms << 80) | int.from_bytes(os.urandom(10), "big")
    return "".join(_CROCKFORD[(n >> (5 * i)) & 31] for i in range(25, -1, -1))


def request_id(replica: str, now: float | None = None) -> str:
    return f"{replica}-{ulid(now)}"


def state_text(state) -> str:
    """A request's state as the text its digest is made of: a string as it is, anything else as compact JSON with
    sorted keys and unescaped Unicode (`json.dumps(state, sort_keys=True, separators=(",", ":"),
    ensure_ascii=False)`), so a client that kept the state can compute the same digest."""
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_text(text: str) -> str:
    """The input text a digest is made of: NFC, whitespace runs collapsed, ends stripped (a client can repeat it)."""
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def hk1(digest_key: str, text: str) -> str:
    """The client-computable digest of an input: `hk1:` and HMAC-SHA256 of its canonical text under the task's key."""
    mac = hmac.new(bytes.fromhex(digest_key), canonical_text(text).encode(), hashlib.sha256).hexdigest()
    return "hk1:" + mac


def uniform(salt: str, rid: str, question: str, slice_name: str) -> float:
    """The draw's uniform for one slice of one served question: a keyed 64-bit hash mapped to [0, 1)."""
    h = hashlib.blake2b(f"{rid}\x00{question}\x00{slice_name}".encode(), key=bytes.fromhex(salt)[:64], digest_size=8)
    return int.from_bytes(h.digest(), "big") / 2.0**64


def floor_pi(pi: float) -> float:
    """A slice's pi with the inclusion floor applied: below `PI_MIN` the row is not drawn at all."""
    return 0.0 if pi < PI_MIN else min(1.0, pi)


def margin_pi(margin: float, settings: dict) -> float:
    lo, hi, scale = settings["margin_pi_min"], settings["margin_pi_max"], settings["margin_scale"]
    w = 1.0 / (1.0 + (max(0.0, float(margin)) / scale) ** 2)  # 1 at a tie, 1/2 at the scale, falling after
    return floor_pi(lo + (hi - lo) * w)


def draw(salt: str, rid: str, question: str, conf: float, margin: float, epoch: dict) -> dict:
    """{slice: {"in", "pi", "u"}} for one served question under `epoch` (its settings and `tau_min`)."""
    s = epoch["settings"]
    pis = {
        "audit": floor_pi(s["audit_pi"]),
        "cert": floor_pi(s["cert_pi"]) if conf >= epoch["tau_min"] else 0.0,
        "margin": margin_pi(margin, s),
    }
    out = {}
    for name in SLICES:
        u = uniform(salt, rid, question, name)
        out[name] = {"in": u < pis[name], "pi": pis[name], "u": u}
    return out


def script_class(text: str) -> str:
    """The Unicode block of most of the letters: latin, cyrillic, greek, arabic, hebrew, devanagari, cjk or other."""
    counts: dict[str, int] = {}
    for ch in str(text)[:4000]:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        for key, word in (
            ("latin", "LATIN"),
            ("cyrillic", "CYRILLIC"),
            ("greek", "GREEK"),
            ("arabic", "ARABIC"),
            ("hebrew", "HEBREW"),
            ("devanagari", "DEVANAGARI"),
            ("cjk", "CJK"),
            ("cjk", "HIRAGANA"),
            ("cjk", "KATAKANA"),
            ("cjk", "HANGUL"),
        ):
            if word in name:
                counts[key] = counts.get(key, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1
    return max(counts, key=counts.get) if counts else "other"


def _arr(v) -> str | None:
    return None if v is None else base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode()


def unarr(s: str | None):
    return None if s is None else np.frombuffer(base64.b64decode(s), dtype="<f4").copy()


# ---- the log of one task -------------------------------------------------------------------------------------------


class TaskLabels:
    """One task's rows and label events, in memory and on its place (`DirPlace` under --tasks-dir, else
    `MemoryPlace`): `rows/<ms>-<seq>.jsonl` written in segments off the request path, `events/<ms>-<seq>.jsonl`
    written before a label write is acknowledged, `EPOCHS.json` the settings in force. Thread-safe."""

    def __init__(self, task_id: str, settings: dict, secret: dict, place, replica: str, clock=time.time):
        self.task_id, self.settings, self.secret, self.place, self.replica = task_id, settings, secret, place, replica
        self.clock = clock
        self._lock = threading.RLock()  # the in-memory log; held briefly, never across a write to the place
        self._write = threading.Lock()  # one event writer at a time (labels, declines, fit use)
        self.rows: dict[tuple[str, str], dict] = {}  # (request id, question) -> row
        self.by_digest: dict[str, list[tuple[str, str]]] = {}
        self.by_rid: dict[str, list[tuple[str, str]]] = {}
        self._undrawn: collections.deque = collections.deque()  # undrawn keys, oldest first: what max_rows drops
        self.events: list[dict] = []
        self.latest: dict[tuple[str, str], dict] = {}  # (request id, question) -> its latest label or decline event
        self.revisions = 0  # labels that replaced a different answer (a sign of label noise)
        self._pending_rows: list[dict] = []
        self._last_flush = clock()
        self._seq = 0
        self.epochs: list[dict] = []
        self._load()
        if not self.epochs:
            self._new_epoch(tau_min=0.0, why="registration")

    # ---- persistence

    def _name(self, kind: str) -> str:
        self._seq += 1
        return f"{kind}/{int(self.clock() * 1000):013d}-{self._seq:06d}.jsonl"

    def _load(self) -> None:
        raw = self.place.get("EPOCHS.json")
        if raw:
            self.epochs = json.loads(raw)
        for name in sorted(self.place.names("rows/")):
            for line in (self.place.get(name) or b"").decode().splitlines():
                if line.strip():
                    self._index(json.loads(line))
        for name in sorted(self.place.names("events/")):
            for line in (self.place.get(name) or b"").decode().splitlines():
                if line.strip():
                    self._apply(json.loads(line))
        self._seq = len(self.place.names("rows/")) + len(self.place.names("events/"))
        while len(self.rows) > self.settings["max_rows"] and self._evict():
            pass

    def _index(self, row: dict) -> None:
        key = (row["request_id"], row["question"])
        if key not in self.rows:
            self.by_rid.setdefault(key[0], []).append(key)
            self.by_digest.setdefault(row["hk1"], []).append(key)
            if not row["drawn"]:
                self._undrawn.append(key)
        self.rows[key] = row

    def _new_epoch(self, tau_min: float, why: str) -> dict:
        e = {"epoch": len(self.epochs), "start": self.clock(), "tau_min": float(tau_min), "why": why,
             "settings": dict(self.settings)}  # fmt: skip
        self.epochs.append(e)
        self.place.put("EPOCHS.json", json.dumps(self.epochs).encode())
        return e

    def _apply(self, ev: dict) -> None:
        """Add one event to the in-memory log (under `_lock`, or while loading)."""
        self.events.append(ev)
        if ev["type"] in ("label", "decline") and ev.get("request_id") is not None:
            key = (ev["request_id"], ev["question"])
            prev = self.latest.get(key)
            if ev["type"] == "label" and prev is not None and prev["type"] == "label":
                self.revisions += prev["answer"] != ev["answer"]
            self.latest[key] = ev

    def _write_events(self, events: list[dict]) -> None:
        """Write events to the place, then add them to the log: a write that fails raises and leaves the log as it
        was, so nothing unsaved is ever acknowledged. The caller holds `_write`, never `_lock`, so the request path
        (`observe`) does not wait on the disk."""
        if not events:
            return
        with self._lock:
            name = self._name("events")
        self.place.put(name, "".join(json.dumps(e) + "\n" for e in events).encode())
        with self._lock:
            for ev in events:
                self._apply(ev)

    def flush(self, force: bool = False) -> int:
        """Write the buffered rows as one segment when there are `SEGMENT_ROWS` or the oldest is `SEGMENT_SECONDS`
        old (or `force`); returns how many were written. Called off the request path (the task store's poll, or a
        timer), and at shutdown. The write happens outside the log's lock; a failed write puts the rows back."""
        with self._write:
            with self._lock:
                due = (
                    force
                    or len(self._pending_rows) >= SEGMENT_ROWS
                    or (self._pending_rows and self.clock() - self._last_flush >= SEGMENT_SECONDS)
                )
                if not due or not self._pending_rows:
                    return 0
                rows, self._pending_rows = self._pending_rows, []
                name = self._name("rows")
            try:
                self.place.put(name, "".join(json.dumps(r) + "\n" for r in rows).encode())
            except Exception:
                with self._lock:
                    self._pending_rows[:0] = rows
                raise
            with self._lock:
                self._last_flush = self.clock()
            return len(rows)

    # ---- the request path

    @property
    def epoch(self) -> dict:
        return self.epochs[-1]

    def observe(self, rid: str, question: str, *, text: str, keys, served, lp, h, certified, version, stack,
                lang: str | None, via: str = "model") -> dict:  # fmt: skip
        """Log one served question and draw it. Returns the answer's additions: {"request_id"} and, when the row was
        drawn into any slice, "label_requested": True. Cheap: a few hashes and an append under the lock."""
        p = np.asarray(served, dtype=np.float64)
        order = np.argsort(-p)
        top = int(order[0])
        conf = float(p[top])
        margin = float(p[top] - p[order[1]]) if len(p) > 1 else 1.0
        with self._lock:
            e = self.epoch
            d = draw(self.secret["salt"], rid, question, conf, margin, e)
            drawn = any(v["in"] for v in d.values())
            full = self.settings["features"] == "full"
            canonical = canonical_text(text)
            row = {
                "request_id": rid,
                "question": question,
                "t": self.clock(),
                "epoch": e["epoch"],
                "digest": hashlib.blake2b(canonical.encode(), key=bytes.fromhex(self.secret["salt"])[:64]).hexdigest(),
                "hk1": hk1(self.secret["digest_key"], text),
                "lang": lang or "unknown",
                "script": script_class(text),
                "keys": list(keys),
                "top": keys[top],
                "conf": conf,
                "margin": margin,
                "served_certified": certified,
                "via": via,
                "version": version,
                "stack": stack,
                "draws": d,
                "drawn": drawn,
                "replica": self.replica,
                **({"ps": _arr(p), "lp": _arr(lp) if drawn else None, "h": _arr(h) if drawn else None} if full else {}),
            }
            if len(self.rows) >= self.settings["max_rows"]:
                self._evict()
            self._index(row)
            self._pending_rows.append(row)
            if e["epoch"] == 0 and e["tau_min"] == 0.0 and len(self.rows) >= self.settings["tau_after"]:
                self._set_tau_min()
        return {"request_id": rid, **({"label_requested": True} if drawn else {})}

    def _evict(self) -> bool:
        """Over `max_rows`: drop the oldest undrawn row (a drawn row stays: its label may still come). Constant time on
        the request path; False when every row left is drawn."""
        while self._undrawn:
            key = self._undrawn.popleft()
            row = self.rows.pop(key, None)
            if row is None:
                continue
            for index, k in ((self.by_digest, row["hk1"]), (self.by_rid, key[0])):
                keys = index.get(k, [])
                if key in keys:
                    keys.remove(key)
                if not keys:
                    index.pop(k, None)
            return True
        return False

    def _set_tau_min(self) -> None:
        """A new epoch whose cert region is the top `region` share of the confidences seen so far."""
        confs = np.sort([r["conf"] for r in self.rows.values()])
        tau = float(np.quantile(confs, 1.0 - self.settings["region"])) if len(confs) else 0.0
        self._new_epoch(tau_min=tau, why=f"tau_min from the first {len(confs)} answers")

    # ---- labels

    def _latest(self) -> dict:
        """{(request id, question): the latest label-side event} (labels, declines), kept as events are added."""
        return self.latest

    def post(self, items) -> dict:
        """Store a batch of labels (each with exactly one of request_id, digest, state, text, and an answer). The
        events are written before this returns; a failure to write raises and nothing is reported accepted."""
        if not isinstance(items, list) or not items:
            raise LabelsError('labels: give "labels": [{"request_id": ..., "answer": ...}, ...]')
        if len(items) > MAX_BATCH:
            raise LabelsError(f"labels: at most {MAX_BATCH} labels per write")
        now = self.clock()
        results, events = [], []
        with self._write:
            with self._lock:
                latest = dict(self.latest)  # this batch's view: earlier labels in the batch count for later ones
                for it in items:
                    results.append(self._one(it, now, latest, events))
            self._write_events(events)
        return {"results": results, **self.counts()}

    def _one(self, it, now, latest, events) -> dict:
        if not isinstance(it, dict):
            return {"status": "invalid", "why": "each label is an object"}
        refs = [k for k in ("request_id", "digest", "state", "text") if it.get(k) is not None]
        if len(refs) != 1:
            return {"status": "invalid", "why": "give exactly one of request_id, digest, state, text"}
        answer = it.get("answer")
        source = it.get("source", "user")
        if source not in ("user", "queue", "import"):
            return {"status": "invalid", "why": "source is user, queue or import"}
        key, matched = None, 0
        if "request_id" in refs:
            rid = str(it["request_id"])
            q = it.get("question")
            cands = [k for k in self.by_rid.get(rid, []) if q is None or k[1] == q]
            if len(cands) > 1:
                return {"status": "invalid", "why": "this request has several questions: name the question"}
            key = cands[0] if cands else None
            if key is None:
                return {"status": "unknown_ref", "request_id": rid}
        else:
            if "digest" in refs:
                dg = it["digest"]
            else:
                dg = hk1(self.secret["digest_key"], state_text(it["state"]) if "state" in refs else str(it["text"]))
            cands = sorted(self.by_digest.get(dg, []), key=lambda k: self.rows[k]["t"], reverse=True)
            matched = len(cands)
            fresh = [k for k in cands if k not in latest]
            key = (fresh or cands or [None])[0]
            if key is None:
                ev = {"type": "label", "request_id": None, "question": None, "digest": dg, "answer": answer,
                      "source": "import", "selection": "import", "labelled_at": it.get("labelled_at") or now,
                      "received": now, "replica": self.replica}  # fmt: skip
                events.append(ev)
                return {"status": "accepted", "source": "import", "matched": 0}
        row = self.rows[key]
        if answer not in row["keys"]:
            return {"status": "invalid_answer", "options": row["keys"], "request_id": key[0]}
        prev = latest.get(key)
        if prev is not None and prev["type"] == "label" and prev["answer"] == answer:
            return {"status": "duplicate", "request_id": key[0], "question": key[1]}
        selection = "slice" if row["drawn"] else "self"
        ev = {"type": "label", "request_id": key[0], "question": key[1], "answer": answer,
              "source": source, "selection": selection,
              "labelled_at": it.get("labelled_at") or now, "received": now, "replica": self.replica}  # fmt: skip
        events.append(ev)
        latest[key] = ev
        status = "revised" if prev is not None and prev["type"] == "label" else "accepted"
        return {"status": status, "request_id": key[0], "question": key[1], **({"matched": matched} if matched else {})}

    def decline(self, refs) -> dict:
        """The client will not label these drawn requests: they leave the queue and count as missing."""
        now = self.clock()
        events = []
        with self._lock:
            for r in refs:
                rid, q = (r.get("request_id"), r.get("question")) if isinstance(r, dict) else (str(r), None)
                for key in [k for k in self.by_rid.get(rid, []) if q is None or k[1] == q]:
                    events.append({"type": "decline", "request_id": key[0], "question": key[1], "received": now,
                                   "replica": self.replica})  # fmt: skip
        with self._write:
            self._write_events(events)
        return {"declined": len(events), **self.counts()}

    def record_fit_use(self, version: str, label_ids) -> None:
        """A fit of `version` is about to use these labels ((request id, question) pairs): recorded first."""
        ev = {"type": "fit_use", "version": version, "labels": [list(x) for x in label_ids], "received": self.clock()}
        with self._write:
            self._write_events([ev])

    def fit_use(self, versions) -> set:
        vs = set(versions)
        return {tuple(x) for ev in self.events if ev["type"] == "fit_use" and ev["version"] in vs for x in ev["labels"]}

    # ---- reads

    def state_of(self, key, latest, now) -> str:
        row = self.rows[key]
        ev = latest.get(key)
        if ev is not None:
            return "labelled" if ev["type"] == "label" else "declined"
        if not row["drawn"]:
            return "undrawn"
        return "expired" if now - row["t"] > self.settings["due_days"] * 86400 else "pending"

    def table(self) -> list[dict]:
        """The joined label table: one record per logged question with every stored field, its label side and its
        state (the export route's JSON lines)."""
        now = self.clock()
        with self._lock:
            latest = self._latest()
            out = []
            for key, row in self.rows.items():
                ev = latest.get(key)
                rec = dict(row)  # every stored field (ps, lp and h as base64 little-endian float32; unarr reads them)
                rec["state"] = self.state_of(key, latest, now)
                if ev is not None and ev["type"] == "label":
                    rec.update(answer=ev["answer"], correct=ev["answer"] == row["top"], selection=ev["selection"],
                               source=ev["source"], labelled_at=ev["labelled_at"])  # fmt: skip
                out.append(rec)
            return sorted(out, key=lambda r: r["t"])

    def queue(self, limit: int = 20) -> dict:
        """The drawn questions still to label, oldest first: which request, never its text, the served answer, its
        confidence, the slice or the probability (a labeller who sees the answer tends to agree with it, and one who
        knows the slice could label it differently)."""
        now = self.clock()
        with self._lock:
            latest = self._latest()
            pend = sorted(
                (k for k in self.rows if self.state_of(k, latest, now) == "pending"), key=lambda k: self.rows[k]["t"]
            )
            items = []
            for key in pend[: max(0, int(limit))]:
                row = self.rows[key]
                item = {"request_id": key[0], "question": key[1], "asked_at": row["t"],
                        "due_by": row["t"] + self.settings["due_days"] * 86400}  # fmt: skip
                items.append(item)
            return {"items": items, "pending": len(pend), "epoch": self.epoch["epoch"], "as_of": now}

    def counts(self) -> dict:
        now = self.clock()
        with self._lock:
            latest = self._latest()
            states: dict[str, int] = {}
            by_slice = {s: {"drawn": 0, "labelled": 0} for s in SLICES}
            for key, row in self.rows.items():
                st = self.state_of(key, latest, now)
                states[st] = states.get(st, 0) + 1
                for s in SLICES:
                    if row["draws"][s]["in"]:
                        by_slice[s]["drawn"] += 1
                        by_slice[s]["labelled"] += st == "labelled"
            imports = sum(1 for ev in self.events if ev["type"] == "label" and ev.get("source") == "import")
            return {"rows": len(self.rows), "states": states, "by_slice": by_slice, "imports": imports,
                    "revisions": self.revisions}  # fmt: skip

    def need(self, alphas=(0.05, 0.10, 0.01, 0.005), delta: float = 0.1) -> list[dict]:
        """Per alpha: the cert slice's labelled answers in the current region and the errors among them (expired and
        declined rows counted as errors, the strict rule), the Clopper-Pearson upper bound, and how many more labels
        the region needs if no further error comes and if one more does.

        The region is the current epoch's: served confidence at or above its `tau_min`. A row of an earlier epoch
        counts when its confidence is in that region and that epoch's region contained the current one (its `tau_min`
        at most the current one), whatever its cert `pi`: each epoch's drawn rows are then a uniform sample of the
        current region's traffic in that epoch, and `pi` only changes how many there are, so the Clopper-Pearson bound
        is exact (`pooling: "exact"`). An epoch whose `tau_min` was higher drew nothing in the band between the two
        thresholds, and its rows are left out. Pending rows are shown apart (`region_pending`), never counted."""
        from decisio.readout import certify

        now = self.clock()
        with self._lock:
            e = self.epoch
            tau = e["tau_min"]
            covers = {x["epoch"] for x in self.epochs if x["tau_min"] <= tau}  # epochs whose region held the current
            latest = self.latest
            n = err = pending = 0
            for key, row in self.rows.items():
                c = row["draws"]["cert"]
                if not c["in"] or row["conf"] < tau or row["epoch"] not in covers:
                    continue
                st = self.state_of(key, latest, now)
                if st == "pending":
                    pending += 1
                elif st == "labelled":
                    n += 1
                    err += latest[key]["answer"] != row["top"]
                elif st in ("expired", "declined"):
                    n += 1
                    err += 1

        def more(k: int, a: float) -> int | None:
            """The fewest further labels (all right) after which k errors certify at a, or None past 100,000."""
            lo, hi = max(n, k + 1), n + 100_000
            if certify.upper_bound(k, hi, delta) > a:
                return None
            while lo < hi:  # the bound falls as labels are added: the first m whose bound is at most a
                mid = (lo + hi) // 2
                if certify.upper_bound(k, mid, delta) <= a:
                    hi = mid
                else:
                    lo = mid + 1
            return lo - n

        out = []
        for a in alphas:
            rec = {"alpha": a, "delta": delta, "region_labels": n, "region_errors": err, "region_pending": pending,
                   "pooling": "exact",
                   "upper_bound": None if n == 0 else round(float(certify.upper_bound(err, n, delta)), 5),
                   "more_if_no_error": more(err, a), "more_if_one_more_error": more(err + 1, a)}  # fmt: skip
            if n and err / n >= a:
                rec["reason"] = "the region's observed error is at or above alpha: it will not certify at this alpha"
            out.append(rec)
        return out


def assert_disjoint(fit_ids, cert_ids) -> None:
    """A certificate must use no label its version was fitted on: raise when the two sets share one."""
    shared = set(map(tuple, fit_ids)) & set(map(tuple, cert_ids))
    if shared:
        raise LabelsError(f"a certificate would use {len(shared)} label(s) its version was fitted on: refused")
