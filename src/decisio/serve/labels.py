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
import re
import secrets
import threading
import time
import unicodedata

import numpy as np

SLICES = ("audit", "cert", "margin", "typical")
EVAL_SLICES = ("audit", "cert")  # the slices certificates and drift checks read; never used to fit
FIT_SLICES = ("margin", "typical")  # probability-sampled slices a refit may use (weighted by 1 / pi)
TYPICAL_RESERVOIR = 512  # answers' embeddings the typicality of a new one is measured against
TYPICAL_K = 10  # nearest neighbours whose mean similarity is an answer's typicality
TYPICAL_MIN = 64  # embeddings held before typicality levels exist (the typical slice draws nothing before)
TYPICAL_LEVELS = 4  # typicality is bucketed into this many levels, by the quantiles of the reservoir's own
TYPICAL_TEXT = 1_000  # characters of an input the student embeds for its typicality: a bounded cost per answer
SCRIPT_TEXT = 500  # characters read for the script class: enough to tell the script, a bounded cost per answer
PI_MIN = 0.02  # a slice's pi is 0 or at least this: inverse-probability weights stay at most 50
DEFAULTS = {
    "features": "full",
    "audit_pi": 0.02,
    "cert_pi": 0.05,
    "margin_pi_min": 0.0,
    "margin_pi_max": 0.2,
    "typical_pi_max": 0.0,  # the typical slice's chance at the most typical level (cold start; fitting only)
    "typical_labels": 50,  # the typical slice ends once this many of its answers are labelled
    "margin_scale": 0.1,  # the margin (top-two gap) at which the margin pi is halfway between its min and max
    "region": 0.8,  # the share of traffic the cert region covers at least once tau_min is set (lenient alphas)
    "tau_after": 200,  # rows seen before tau_min is set from their confidences (all traffic until then)
    "due_days": 14,
    "ttl_days": None,  # answers and labels older than this are erased (None: kept until erased or dropped by max_rows)
    "max_rows": 50_000,
    "reservoir": 2_000,  # undrawn answers that keep their served distribution: a uniform sample, about this many
    "per_day": 20,  # labels a day the chances aim at (None: the chances above as given); a place to start
    "shares": {"audit": 0.15, "cert": 0.45, "margin": 0.30, "typical": 0.10},  # of per_day, by slice
}
STRICT_ALPHA = 0.01  # a task certified at this alpha or below has its labels drawn mostly where its certificate is
STRICT_SHARES = {"audit": 0.10, "cert": 0.80, "margin": 0.10, "typical": 0.0}
STRICT_MIN_SHARE = 0.10  # such a task needs labels on at least this share of its answers (its drift checks fail below)
PI_KEYS = ("audit_pi", "cert_pi", "margin_pi_min", "margin_pi_max", "typical_pi_max")
REBUDGET_SECONDS = 86_400  # a budget epoch lasts at least this long
REBUDGET_BAND = (0.5, 2.0)  # and ends when the measured rate leaves this band around the rate it assumed
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
    explicit = any(k in block for k in PI_KEYS)
    if explicit and block.get("per_day") is not None:
        raise LabelsError("labels: give per_day (a budget the chances follow) or the chances themselves, not both")
    s = {**DEFAULTS, **{k: v for k, v in block.items() if k != "enabled"}}
    if explicit:
        s["per_day"] = None
    if s["per_day"] is not None and (isinstance(s["per_day"], bool) or not isinstance(s["per_day"], (int, float))
                                     or s["per_day"] <= 0):  # fmt: skip
        raise LabelsError("labels.per_day is a number of labels a day above 0, or null for the chances as given")
    sh = s["shares"]
    if (not isinstance(sh, dict) or set(sh) - set(DEFAULTS["shares"])
            or any(not isinstance(v, (int, float)) or v < 0 for v in sh.values())
            or abs(sum(sh.values()) - 1.0) > 1e-6):  # fmt: skip
        raise LabelsError(f"labels.shares: fractions of per_day for {sorted(DEFAULTS['shares'])}, summing to 1")
    s["shares"] = {k: float(sh.get(k, 0.0)) for k in DEFAULTS["shares"]}
    if s["per_day"] is not None:  # the registration's epoch: tau_after answers a day assumed, all traffic the region
        s.update(budget_pis(s, rate=float(s["tau_after"]), region_share=1.0, margin_weight=0.5, typical=False))
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
    if s["ttl_days"] is not None and (isinstance(s["ttl_days"], bool) or not isinstance(s["ttl_days"], (int, float))
                                      or s["ttl_days"] < s["due_days"]):  # fmt: skip
        raise LabelsError("labels.ttl_days is a number of days at least due_days, or null to keep answers")
    for k in ("tau_after", "due_days", "max_rows", "reservoir", "typical_labels"):
        if not isinstance(s[k], int) or s[k] < 1:
            raise LabelsError(f"labels.{k} is a positive whole number")
    return s


def margin_weight(margin: float, scale: float) -> float:
    """The margin slice's shape: 1 at a tie, 1/2 at `scale`, falling after."""
    return 1.0 / (1.0 + (max(0.0, float(margin)) / scale) ** 2)


def budget_pis(s: dict, rate: float, region_share: float, margin_weight: float, typical: bool = False) -> dict:
    """The chances that spend `per_day` labels a day in `shares` at `rate` answers a day: audit over all traffic,
    cert over the region (`region_share` of traffic), margin scaled by the mean margin weight, and with `typical`
    the typical slice (its chance at the top level, the levels' mean weight being 5/8). Audit, cert and a running
    typical slice are raised to `PI_MIN` when the budget is too small (they spend more than their share then; at the
    floor only the most typical level is drawn); margin below it is off.
    Without the typical slice its share goes to margin."""
    per, sh = float(s["per_day"]), s["shares"]
    rate = max(float(rate), 1e-9)
    audit = min(1.0, max(PI_MIN, per * sh["audit"] / rate))
    cert = min(1.0, max(PI_MIN, per * sh["cert"] / (rate * max(region_share, 1e-9))))
    m_share = sh["margin"] + (0.0 if typical else sh.get("typical", 0.0))
    m = per * m_share / (rate * max(margin_weight, 1e-9))
    t = per * sh.get("typical", 0.0) / (rate * 0.625) if typical else 0.0
    return {"audit_pi": round(audit, 6), "cert_pi": round(cert, 6), "margin_pi_min": 0.0,
            "margin_pi_max": 0.0 if m < PI_MIN else round(min(1.0, m), 6),
            "typical_pi_max": round(min(1.0, max(PI_MIN, t)), 6) if typical else 0.0}  # fmt: skip


class Typicality:
    """How typical an answer's input is, from the static student's embedding (unit vectors) against a reservoir of
    earlier ones: the mean similarity to its `TYPICAL_K` nearest, bucketed into `TYPICAL_LEVELS` levels by the
    quantiles of the reservoir's own scores. `level` is the request path's (a matrix product); `refresh`, which sets
    the levels' cut points, is the labels thread's. The reservoir is a uniform sample of what was seen (reservoir
    sampling), so a level depends only on earlier traffic, never on a label."""

    def __init__(self, seed: int = 0):
        import random

        self._lock = threading.Lock()
        self.vecs: np.ndarray | None = None
        self.n = 0
        self.seen = 0
        self.cuts: np.ndarray | None = None
        self._changed = 0
        self._rng = random.Random(seed)

    def _score(self, vecs: np.ndarray, v: np.ndarray, skip: int | None = None) -> float:
        sims = vecs @ v
        if skip is not None:
            sims = np.delete(sims, skip)
        k = min(TYPICAL_K, len(sims))
        return float(np.mean(np.partition(sims, len(sims) - k)[-k:])) if k else 0.0

    def level(self, v: np.ndarray) -> int | None:
        """1 (least typical) to `TYPICAL_LEVELS`, or None before the levels exist."""
        with self._lock:
            vecs, cuts, n = self.vecs, self.cuts, self.n
        if cuts is None or vecs is None:
            return None
        return 1 + int(np.searchsorted(cuts, self._score(vecs[:n], v)))

    def add(self, v: np.ndarray) -> None:
        with self._lock:
            if self.vecs is None:
                self.vecs = np.zeros((TYPICAL_RESERVOIR, len(v)), dtype=np.float32)
            self.seen += 1
            if self.n < TYPICAL_RESERVOIR:
                self.vecs[self.n] = v
                self.n += 1
                self._changed += 1
            else:
                j = self._rng.randrange(self.seen)
                if j < TYPICAL_RESERVOIR:
                    self.vecs[j] = v
                    self._changed += 1

    def refresh(self) -> None:
        """The levels' cut points from the reservoir's own scores (each against the others), once it holds
        `TYPICAL_MIN` and has changed by an eighth since the last time."""
        with self._lock:
            if self.n < TYPICAL_MIN or (self.cuts is not None and self._changed < max(8, self.n // 8)):
                return
            vecs = self.vecs[: self.n].copy()
            self._changed = 0
        scores = np.array([self._score(vecs, vecs[i], skip=i) for i in range(len(vecs))])
        cuts = np.quantile(scores, [i / TYPICAL_LEVELS for i in range(1, TYPICAL_LEVELS)])
        with self._lock:
            self.cuts = cuts


STORES = (
    "for every answered question of this task: a request id, the time, the answer, its confidence and the gap to the "
    "next option, the language tag the request gave (meta.lang or x-decisio-lang) and the script class of its text, "
    "two digests of the input (salted; not reversible without the task's secrets), the certificate and the serving "
    "stack it was answered under, and whether it was drawn to be labelled",
    'with features "full" (the default): for drawn questions also the served distribution, the plain readout\'s label '
    "log-probabilities and, for a task with an intent head, the hidden state at the answer position (a model feature "
    "that is not anonymous; it stays where the task's log is kept); and the served distribution of a uniform sample "
    "of the other questions (about `reservoir` of them)",
    "for a task whose options change with every request: its options and answers only as keyed digests, and one "
    "keyed digest of the request's option texts",
    "never the request's text or state",
    "the labels you post, with who chose the item (the draw, or you)",
)


ROW_FIELDS = ("request_id", "question", "t", "epoch", "digest", "hk1", "lang", "script", "keys", "top", "conf",
              "margin", "served_certified", "via", "version", "stack", "draws", "drawn", "sample",
              "options_digest", "replica")  # fmt: skip
FULL_FIELDS = ("ps", "lp", "h")  # with features "full": the served distribution, the plain readout, the hidden state


MINIMAL_STORES = 'with features "minimal": no distribution, no plain readout and no hidden state'


def stored_fields(settings: dict) -> list[str]:
    """The fields a row of this task's log has, as the mode in force stores them (`lp`, `h` null on undrawn rows,
    `ps` too outside the `sample`)."""
    return list(ROW_FIELDS) + (list(FULL_FIELDS) if settings["features"] == "full" else [])


def stores(settings: dict) -> list[str]:
    """What the log stores under `settings`, in words (the `features` setting decides the distributions' line)."""
    full = settings["features"] == "full"
    return [STORES[0]] + ([STORES[1]] if full else [MINIMAL_STORES]) + list(STORES[2:])


def disclosure(settings: dict, location: str) -> dict:
    """What the log keeps under `settings`, exactly: the registration's response says it, and so does every change of
    the settings (`PUT .../labels/settings`), since a change can widen or narrow it. Never a secret."""
    s = settings
    return {
        "enabled": True,
        "text_stored": False,
        "features": s["features"],
        "stored_fields": stored_fields(s),
        "retention": {
            "max_rows": s["max_rows"],
            "due_days": s["due_days"],
            "ttl_days": s.get("ttl_days"),
            "deleted_with_task": True,
        },
        "location": location,
        "settings": s,
        "stores": stores(s),
    }


def most_labelled(s: dict) -> float:
    """The largest share of answers explicit chances can draw (each slice at its highest chance, the cert region at
    `region`): a bound, so a check against it refuses only what cannot reach a share."""
    return min(1.0, s["audit_pi"] + s["cert_pi"] * s["region"] + s["margin_pi_max"] + s.get("typical_pi_max", 0.0))


def strict_check(s: dict, alpha: float | None) -> tuple[str | None, list[str]]:
    """For a task certified at `STRICT_ALPHA` or below, which needs labels on at least `STRICT_MIN_SHARE` of its
    answers: (an error for explicit chances that cannot reach it, warnings for a budget that must keep up with
    traffic)."""
    if alpha is None or float(alpha) > STRICT_ALPHA:
        return None, []
    need = f"a task certified at alpha {alpha} needs labels on at least {STRICT_MIN_SHARE:.0%} of its answers"
    if s.get("per_day") is None:
        most = most_labelled(s)
        if most < STRICT_MIN_SHARE:
            cert = (STRICT_MIN_SHARE - s["audit_pi"] - s["margin_pi_max"] - s.get("typical_pi_max", 0.0)) / s["region"]
            return (f"labels: {need}; these chances label at most {most:.1%} (raise cert_pi to at least "
                    f"{min(1.0, cert):.3f}, or give per_day)"), []  # fmt: skip
        return None, []
    return None, [f"{need}: per_day ({s['per_day']}) covers up to {s['per_day'] / STRICT_MIN_SHARE:,.0f} answers a "
                  "day; status reports the share labelled and warns below it"]  # fmt: skip


def not_enabled_block(alpha: float | None) -> dict | None:
    """What a registration says of labels when the task did not opt in: nothing, except for a task certified at
    `STRICT_ALPHA` or below, which needs labels on a share of its answers; it is told so, and nothing is logged."""
    if alpha is None or float(alpha) > STRICT_ALPHA:
        return None
    return {"enabled": False, "warnings": [
        f"a task certified at alpha {alpha} needs labels on at least {STRICT_MIN_SHARE:.0%} of its answers for its "
        "drift checks to catch a rising error in time; this task takes no labels and logs nothing: register it with "
        'a labels block ("labels": true, or a per_day budget) to opt in'
    ]}  # fmt: skip


def registration_block(rec: dict, location: str, alpha: float | None = None) -> dict:
    """What a registration's response says of the task's labels: the disclosure (`disclosure`), the digest key (shown
    here only), and the queue's first view (empty until the task has answered)."""
    s = rec["settings"]
    warnings = strict_check(s, alpha)[1]
    return {
        **disclosure(s, location),
        **({"warnings": warnings} if warnings else {}),
        "digest_key": rec["secret"]["digest_key"],
        "digest": "hk1: then the hex HMAC-SHA256, under digest_key, of the request's state as text (a string as it "
        "is, anything else as JSON with sorted keys, no spaces and unescaped Unicode), in NFC with whitespace runs as "
        "one space and the ends stripped",
        "next_to_label": {
            "items": [],
            "pending": 0,
            "available": False,
            "reason": "no traffic yet",
            "budget": {"per_day": s.get("per_day")},
        },
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


def rid_time(rid: str) -> float | None:
    """The time a request id was issued (its ULID's milliseconds), or None when it is not one of ours."""
    tail = str(rid).rsplit("-", 1)[-1]
    if "-" not in str(rid) or len(tail) != 26 or any(c not in _CROCKFORD for c in tail):
        return None
    ms = 0
    for c in tail[:10]:
        ms = ms * 32 + _CROCKFORD.index(c)
    return ms / 1000.0


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


def hk1(digest_key: str, text: str, canonical: str | None = None) -> str:
    """The client-computable digest of an input: `hk1:` and HMAC-SHA256 of its canonical text under the task's key
    (`canonical`: that text, when the caller has it already)."""
    canon = canonical_text(text) if canonical is None else canonical
    return "hk1:" + hmac.new(bytes.fromhex(digest_key), canon.encode(), hashlib.sha256).hexdigest()


def uniform(salt: str, rid: str, question: str, slice_name: str) -> float:
    """The draw's uniform for one slice of one served question: a keyed 64-bit hash mapped to [0, 1)."""
    h = hashlib.blake2b(f"{rid}\x00{question}\x00{slice_name}".encode(), key=bytes.fromhex(salt)[:64], digest_size=8)
    return int.from_bytes(h.digest(), "big") / 2.0**64


def floor_pi(pi: float) -> float:
    """A slice's pi with the inclusion floor applied: below `PI_MIN` the row is not drawn at all."""
    return 0.0 if pi < PI_MIN else min(1.0, pi)


def margin_pi(margin: float, settings: dict) -> float:
    lo, hi, scale = settings["margin_pi_min"], settings["margin_pi_max"], settings["margin_scale"]
    w = margin_weight(margin, scale)
    return floor_pi(lo + (hi - lo) * w)


def draw(
    salt: str, rid: str, question: str, conf: float, margin: float, epoch: dict, typical: int | None = None
) -> dict:
    """{slice: {"in", "pi", "u"}} for one served question under `epoch` (its settings and `tau_min`); `typical`: its
    typicality level (None: the typical slice draws nothing, as before the levels exist or without the student)."""
    s = epoch["settings"]
    pis = {
        "audit": floor_pi(s["audit_pi"]),
        "cert": floor_pi(s["cert_pi"]) if conf >= epoch["tau_min"] else 0.0,
        "margin": margin_pi(margin, s),
        "typical": floor_pi(s.get("typical_pi_max", 0.0) * typical / TYPICAL_LEVELS) if typical else 0.0,
    }
    out = {}
    for name in SLICES:
        u = uniform(salt, rid, question, name)
        out[name] = {"in": u < pis[name], "pi": pis[name], "u": u}
    return out


def script_class(text: str) -> str:
    """The Unicode block of most of the letters: latin, cyrillic, greek, arabic, hebrew, devanagari, cjk or other."""
    counts: dict[str, int] = {}
    for ch in str(text)[:SCRIPT_TEXT]:
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


class LabelsReadOnly(LabelsError):
    """This replica does not write to the shared task store (`--task-store-readonly`): it takes no labels."""


READ_ONLY = (
    "this replica does not write to the task store (--task-store-readonly): it logs no answers and takes no labels; "
    "post labels to a replica that writes"
)
PENDING_MAX = 50_000  # rows a task holds for writing while its place cannot be reached; past it, new rows go unsaved
COMPACT_FILES = 64  # files read before the merger rewrites a log as a few merged ones
MERGED_ROWS = 5_000  # rows per merged file
MERGED_EVENTS = 20_000  # events per merged file
MERGED = "merged"  # the writer name of merged files
PURGE_AFTER = 3 * FLUSH_EVERY  # an erasure this old is read by every replica that writes (each reads before it writes)


class TaskLabels:
    """One task's rows and label events, in memory and on its place: `DirPlace` under --tasks-dir, `MemoryPlace`
    without it, or `SharedPlace` on a shared task store. Files are written once and never rewritten:
    `rows/<ms>-<seq>-<replica>.jsonl` in segments off the request path, `events/<ms>-<seq>-<replica>.jsonl` before a
    label write is acknowledged; `EPOCHS.json` holds the epochs, moved only by `_swap_epochs` (a compare-and-swap on a
    shared store).

    Nothing here touches the place on the request path (`observe`). `tick`, on the server's labels thread, reads what
    other replicas wrote (`sync`), starts a due epoch and writes a due segment; the label routes `sync` before they
    read. A label event is joined to its row when both are known, whichever comes first, so a label posted to one
    replica for an answer another replica served is kept at once and counted once that replica's segment is read.
    Thread-safe."""

    def __init__(self, task_id: str, settings: dict, secret: dict, place, replica: str, clock=time.time,
                 dynamic: bool = False):  # fmt: skip
        self.task_id, self.secret, self.place, self.replica = task_id, secret, place, replica
        self.dynamic = dynamic  # options change with every request: options and answers are kept as digests only
        self.registered = dict(settings)  # the registration's settings: epoch 0's
        self.clock = clock
        self.shared = bool(getattr(place, "shared", False))
        self.writes = bool(getattr(place, "writes", True))
        self._lock = threading.RLock()  # the in-memory log; held briefly, never across a call to the place
        self._write = threading.Lock()  # one writer to the place at a time (labels, declines, fit use, segments)
        self._sync = threading.Lock()  # one reader of the place at a time
        self.rows: dict[tuple[str, str], dict] = {}  # (request id, question) -> row
        self.by_digest: dict[str, list[tuple[str, str]]] = {}
        self.by_rid: dict[str, list[tuple[str, str]]] = {}
        self._undrawn: collections.deque = collections.deque()  # undrawn keys, oldest first: what max_rows drops
        self.events: list[dict] = []
        self.latest: dict[tuple[str, str], dict] = {}  # (request id, question) -> its latest label or decline event
        self._history: dict[tuple[str, str], list[dict]] = {}  # its label and decline events, oldest first
        self._changes: dict[tuple[str, str], int] = {}
        self._parked: dict[str, list[dict]] = {}  # request id -> label events whose row is not known here yet
        self.revisions = 0  # labels that replaced a different answer (a sign of label noise)
        self.invalid_on_join = 0  # labels joined to a row whose options do not include their answer: ignored
        self._ids: set[str] = set()  # event ids seen: a merged file repeats events, each counts once
        self._erased_rids: set[str] = set()
        self._erased_before = float("-inf")  # rows served and events received before this are erased
        self._tombs: dict[str, dict] = {}  # erase events by id
        self._purged: set[str] = set()  # erase ids whose bytes the merger has removed from the place
        self.merges = 0
        self.cert_threshold: float | None = None  # the task's certified threshold, when it has one (the server sets it)
        self.cert_alpha: float | None = None  # the alpha the task is certified at (the server sets it)
        self.embed = None  # the static student's embedding of a text (the server sets it once the student is here)
        self.embed_off: str | None = "no static student on this server"
        self.typicality = Typicality(int(secret["salt"][:8], 16))
        self._hours: dict[int, int] = {}  # answers known here (every replica's) by the hour they were served
        self._first_t: float | None = None
        self.unsaved = 0  # rows not written because the place could not be reached for too long (`PENDING_MAX`)
        self.unjoined_expired = 0  # labels for ids no replica's row ever joined, dropped after their due window
        self._evicted_upto = float("-inf")  # the newest serve time of an answer `max_rows` dropped
        self._pending_rows: list[dict] = []
        self._last_flush = clock()
        self._seq = 0
        self._last_ms = 0
        self._read: set[str] = set()  # files of the place already in memory (this replica's own as it writes them)
        self.loaded = False  # the place has been read once (`sync`)
        self.last_sync: float | None = None
        self.epochs: list[dict] = [self._epoch_zero()]

    # ---- the place

    @property
    def settings(self) -> dict:
        """The settings in force: the current epoch's."""
        return self.epochs[-1]["settings"]

    def _epoch_zero(self) -> dict:
        """The registration's epoch: its settings, the region all traffic. The same on every replica without a write."""
        return {"epoch": 0, "start": None, "tau_min": 0.0, "tau_set": False, "why": "registration",
                "settings": dict(self.registered)}  # fmt: skip

    def _name(self, kind: str, writer: str | None = None) -> str:
        """A new file's name: time-ordered for its writer (the milliseconds never go back), and the writer's own."""
        self._seq += 1
        self._last_ms = max(self._last_ms, int(self.clock() * 1000))
        return f"{kind}/{self._last_ms:013d}-{self._seq:09d}-{writer or self.replica}.jsonl"

    def sync(self) -> None:
        """Read what the place holds that this log has not: the epochs, and every replica's new row and event files
        (written once, so each is read once). Raises when the place cannot be read; what is in memory stays."""
        with self._sync:
            self._read_place()

    def _read_place(self) -> None:
        """`sync`'s work; the caller holds `_sync`."""
        raw = self.place.get("EPOCHS.json")
        if raw:
            epochs = json.loads(raw)
            with self._lock:
                if len(epochs) > len(self.epochs):
                    self.epochs = epochs
        for kind in ("rows/", "events/"):
            advance = getattr(self.place, "advance", None)  # a shared store's listing resumes after what was read
            for name in self.place.names(kind):
                if name in self._read:
                    if advance:
                        advance(name)
                    continue
                data = self.place.get(name)  # a failure raises here: the listing stays before this file
                if data is None:  # gone since the listing (merged elsewhere): nothing to read
                    if advance:
                        advance(name)
                    continue
                lines = [x for x in data.decode().splitlines() if x.strip()]
                items = [json.loads(x) for x in lines]
                if (
                    kind == "events/"
                ):  # an event written without an id gets one from its bytes: read twice, counted once
                    for line, it in zip(lines, items):
                        it.setdefault("id", "h" + hashlib.blake2b(line.encode(), digest_size=12).hexdigest())
                with self._lock:
                    if name in self._read:
                        continue
                    late = 0
                    for it in items:
                        if kind == "rows/":
                            late += self._index(it) is False
                        else:
                            self._apply(it)
                    if late and not name.endswith(f"-{MERGED}.jsonl"):
                        # an erased row in a file the purge did not see: the next merge purges again
                        self._purged.difference_update(self._tombs)
                    self._read.add(name)
                if advance:
                    advance(name)
        with self._lock:
            self._join_parked()
            while len(self.rows) > self.settings["max_rows"] and self._evict():
                pass
            self.loaded = True
            self.last_sync = self.clock()

    def ensure_loaded(self) -> None:
        if not self.loaded:
            self.sync()

    def read_epochs(self) -> None:
        """Only the epochs from the place (one small read), for a view that needs the settings in force but not the
        rows: the task listing."""
        raw = self.place.get("EPOCHS.json")
        if raw:
            self._adopt(json.loads(raw))

    def tick(self, force: bool = False, merge: bool = True) -> None:
        """The labels thread's turn: read the place, start an epoch that is due, write a segment that is due, and,
        for the merger (`merge`: one server, or the holder of the labels lease on a shared store), merge the log when
        it has many files or an erasure waits for its purge."""
        self.sync()
        self._maybe_tau()
        self._maybe_rebudget()
        self._maybe_typical()
        self._maybe_strict()
        self._maybe_expire()
        self.flush(force)
        if merge and self.writes and (len(self._read) >= COMPACT_FILES or self.unpurged(ripe=True)):
            self.compact()

    def unpurged(self, ripe: bool = False) -> list[str]:
        """The erase events whose bytes may still be on the place; `ripe`: only those a merge may purge now. On a
        shared store that is `PURGE_AFTER` after the erasure: another replica may write a file holding an erased row
        until its labels thread has read the erasure, and that thread reads the place before it writes."""
        with self._lock:
            now = self.clock()
            return [i for i, t in self._tombs.items() if i not in self._purged
                    and (not ripe or not self.shared or now - float(t["received"]) >= PURGE_AFTER)]  # fmt: skip

    def compact(self) -> int:
        """Rewrite the files this log has read as a few merged ones holding what it keeps (no erased row or label,
        no row dropped by `max_rows`), then delete the files it read, then record the erasures as purged. Every
        write is a new file and only files read and merged are deleted, so a second merger racing this one, or a
        replica reading meanwhile, loses nothing: a row read twice is one row, an event read twice is one event.
        Returns how many files it replaced."""
        with self._sync, self._write:
            self._read_place()  # every file there is now, so a ripe erasure's rows are in no file left unread
            ripe = set(self.unpurged(ripe=True))
            with self._lock:
                files = sorted(self._read)
                if not files:
                    return 0
                waiting = {(r["request_id"], r["question"]) for r in self._pending_rows}
                rows = [r for k, r in self.rows.items() if k not in waiting]
                events = [e for e in self.events if e["type"] != "purged"]
                tombs = sorted(ripe)
            with self._lock:  # name after every merged file read, whoever's clock wrote it (a lease handover)
                for name in self._read:
                    m = _FILE_NAME.match(name)
                    if m and m.group(3) == MERGED:
                        self._last_ms = max(self._last_ms, int(m.group(2).split("-")[0]) + 1)
            written = []
            for kind, items, size in (("rows", rows, MERGED_ROWS), ("events", events, MERGED_EVENTS)):
                for i in range(0, len(items), size):
                    with self._lock:
                        name = self._name(kind, MERGED)
                    self.place.put(name, "".join(json.dumps(x) + "\n" for x in items[i : i + size]).encode())
                    written.append(name)
            for name in files:
                if name not in written:
                    self.place.delete(name)
            with self._lock:
                self._read = (self._read - set(files)) | set(written)
                self.merges += 1
            if tombs:
                done = {"type": "purged", "of": tombs, "received": self.clock(), "replica": self.replica,
                        "id": ulid(self.clock())}  # fmt: skip
                with self._lock:
                    name = self._name("events", MERGED)
                self.place.put(name, (json.dumps(done) + "\n").encode())
                with self._lock:
                    self._read.add(name)
                    self._apply(done)
            return len(files)

    def erase(self, request_id: str | None = None, before: float | None = None) -> dict:
        """Erase one request's answers and labels (`request_id`), or everything served or labelled before a time
        (`before`, in seconds since the epoch). An erase event is written first and the rows leave memory at once
        (`erased`, with how many logged answers and labels it matched); the bytes leave the place at the merger's next
        turn (`purged`: within `FLUSH_EVERY` seconds on one server, `PURGE_AFTER` and a turn on a shared store), so
        a thousand erasures are one rewrite, not a thousand."""
        if not self.writes:
            raise LabelsReadOnly(READ_ONLY)
        if (request_id is None) == (before is None):
            raise LabelsError("erase: give a request id or a time, not both")
        if request_id is not None and rid_time(request_id) is None:
            raise LabelsError(f"erase: {request_id!r} is not a request id")
        now = self.clock()
        with self._lock:
            bound = float("-inf") if before is None else float(before)
            keys = [k for k, r in self.rows.items() if k[0] == request_id or r["t"] < bound]
            rids = {k[0] for k in keys}
            n_labels = sum(
                1
                for e in self.events
                if e["type"] in ("label", "decline")
                and (e.get("request_id") in rids or float(e.get("received") or 0) < bound)
            )
            matched = {"answers": len(keys), "labels": n_labels}
        tomb = {"type": "erase", "request_id": request_id, "before": None if before is None else float(before),
                "received": now, "replica": self.replica}  # fmt: skip
        with self._write:
            self._write_events([tomb])
        return {"state": "erased", "erase_id": tomb["id"], "matched": matched,
                **({"request_id": request_id} if request_id else {"before": tomb["before"]})}  # fmt: skip

    def _swap_epochs(self, fn) -> list:
        """Move `EPOCHS.json` by `fn(current) -> new list, or None to keep it`: a compare-and-swap on a shared store
        (`SharedPlace.swap`), else a read and a write under this log's writer lock. Returns the epochs now stored."""
        if hasattr(self.place, "swap"):
            return self.place.swap("EPOCHS.json", fn)
        with self._write:
            raw = self.place.get("EPOCHS.json")
            cur = json.loads(raw) if raw else None
            new = fn(cur)
            if new is None:
                return cur or []
            self.place.put("EPOCHS.json", json.dumps(new).encode())
            return new

    def _maybe_tau(self) -> None:
        """Once `tau_after` answers are known, a new epoch whose cert region is the top `region` share of their
        confidences. Every replica may propose it; the first stored wins, and the others adopt it."""
        if not self.writes:
            return
        with self._lock:
            if _tau_set(self.epoch) or len(self.rows) < self.settings["tau_after"]:
                return
            tau, n = self._tau_now()
            why = f"tau_min from the first {n} answers"
            b = self._budget(self.settings, tau) if self.settings.get("per_day") is not None else None
        zero = self._epoch_zero()

        def add(cur):
            cur = cur or [zero]
            if _tau_set(cur[-1]):
                return None  # another replica set it first: adopt theirs
            last = cur[-1]
            new = {"epoch": len(cur), "start": self.clock(), "tau_min": tau, "tau_set": True,
                   "settings": last["settings"], "why": why, "by": self.replica}  # fmt: skip
            if b is not None:  # the chances from the measured rate, from now on
                new.update(settings=b["settings"], budget=b["budget"])
            return cur + [new]

        self._adopt(self._swap_epochs(add))

    def rate(self) -> float:
        """Answers a day over the last 7 days (or since the first answer, at least an hour), every replica's that
        this log has read (under `_lock`)."""
        now = self.clock()
        lo = now - 7 * 86400
        for h in [h for h in self._hours if (h + 1) * 3600 < lo]:
            del self._hours[h]
        n = sum(c for h, c in self._hours.items() if (h + 1) * 3600 >= lo)
        span = max(3600.0, now - max(lo, self._first_t if self._first_t is not None else now))
        return n * 86400.0 / span

    def typical_wanted(self) -> bool:
        """Whether the typical slice should run: the student is here and the slice has not ended."""
        return self.embed is not None and not any(e.get("why") == "the typical slice ended" for e in self.epochs)

    def _budget(self, settings: dict, tau: float, typical: bool | None = None, rate: float | None = None) -> dict:
        """The chances for `settings`' per_day at the measured rate, over the region above `tau` (under `_lock`):
        {"settings": with the chances, "budget": what they assumed}."""
        rows = list(self.rows.values())
        rate = self.rate() if rate is None else float(rate)
        share = (sum(r["conf"] >= tau for r in rows) / len(rows)) if rows else 1.0
        w = float(np.mean([margin_weight(r["margin"], settings["margin_scale"]) for r in rows])) if rows else 0.5
        pis = budget_pis(settings, rate, share, w, typical=self.typical_wanted() if typical is None else typical)
        new = {**settings, **pis}
        per_row = [  # the chance each held answer would have of being drawn into each slice, floors and caps applied
            floor_pi(new["audit_pi"])
            + (floor_pi(new["cert_pi"]) if r["conf"] >= tau else 0.0)
            + margin_pi(r["margin"], new)
            for r in rows
        ]
        expected = rate * float(np.mean(per_row)) if rows else 0.0  # below per_day when a slice is floored or full
        budget = {"per_day": settings["per_day"], "rate_per_day": round(rate, 3), "region_share": round(share, 4),
                  "margin_weight": round(w, 4), "expected_per_day": round(expected, 2)}  # fmt: skip
        return {"settings": new, "budget": budget}

    def budget(self) -> dict:
        """The budget in force, for the queue and status: the labels a day asked for (`per_day`, None: the chances as
        given), and the labels a day expected at the measured rate from the chances the last 7 days' answers were
        drawn at (`expected_per_day`); `over_budget` when that is more than a quarter above per_day, as when the
        0.02 floor of the audit and cert chances spends more than a small budget on heavy traffic."""
        now = self.clock()
        with self._lock:
            e = self.epoch
            s = e["settings"]
            rate = self.rate()
            week = [r for r in self.rows.values() if now - r["t"] <= 7 * 86400]
            share = (
                float(np.mean([1.0 - np.prod([1.0 - d["pi"] for d in r["draws"].values()]) for r in week]))
                if week else 0.0
            )  # fmt: skip
            expected = round(rate * share, 2)
            out = {"per_day": s.get("per_day"), "rate_per_day": round(rate, 3), "since": e.get("start"),
                   "expected_per_day": expected,
                   "over_budget": s.get("per_day") is not None and expected > 1.25 * s["per_day"]}  # fmt: skip
            if e.get("budget"):
                out["assumed_rate_per_day"] = e["budget"]["rate_per_day"]
            out["chances"] = {k: s[k] for k in PI_KEYS}
            return out

    def _maybe_rebudget(self) -> None:
        """A budget epoch lasts at least `REBUDGET_SECONDS`; after that, a new one when the measured rate has left
        `REBUDGET_BAND` around the rate it assumed (the region kept)."""
        if not self.writes:
            return
        with self._lock:
            e = self.epoch
            s = e["settings"]
            if s.get("per_day") is None or not _tau_set(e) or e.get("start") is None:
                return
            if self.clock() - e["start"] < REBUDGET_SECONDS:
                return
            assumed = (e.get("budget") or {}).get("rate_per_day") or float(s["tau_after"])
            ratio = self.rate() / max(assumed, 1e-9)
            if REBUDGET_BAND[0] <= ratio <= REBUDGET_BAND[1]:
                return
            b = self._budget(s, e["tau_min"])
            n = e["epoch"]

        def add(cur):
            cur = cur or [self._epoch_zero()]
            if cur[-1]["epoch"] != n:
                return None  # another replica moved the epochs first
            why = f"the rate moved {ratio:.2f}x"
            return cur + [{**cur[-1], "epoch": len(cur), "start": self.clock(), "settings": b["settings"],
                           "budget": b["budget"], "why": why, "by": self.replica}]  # fmt: skip

        self._adopt(self._swap_epochs(add))

    def _maybe_strict(self) -> None:
        """A task certified at `STRICT_ALPHA` or below, in budget mode with the default shares: the shares move to
        `STRICT_SHARES` (an epoch), so its labels reach the hundreds a strict certificate needs about 1.8x sooner;
        the audit share stays above 30 labels in the first 1,000. Shares given at registration are kept."""
        if not self.writes or self.cert_alpha is None or float(self.cert_alpha) > STRICT_ALPHA:
            return
        with self._lock:
            s, e = self.settings, self.epoch
            if s.get("per_day") is None or s["shares"] != DEFAULTS["shares"]:
                return
            strict = {**s, "shares": dict(STRICT_SHARES)}
            assumed = None if _tau_set(e) else float(s["tau_after"])
            b = self._budget(strict, e["tau_min"], typical=False, rate=assumed)
            n = e["epoch"]
            why = f"shares for a certificate at alpha {self.cert_alpha}"

        def add(cur):
            cur = cur or [self._epoch_zero()]
            if cur[-1]["epoch"] != n:
                return None
            return cur + [{**cur[-1], "epoch": len(cur), "start": self.clock(), "settings": b["settings"],
                           "budget": b["budget"], "why": why, "by": self.replica}]  # fmt: skip

        self._adopt(self._swap_epochs(add))

    def _maybe_expire(self) -> None:
        """With `ttl_days`: erase what is older, at most once an hour, through the erase path (an erase event, then
        the purge at the merge), so expiry and erasure leave the same trace."""
        ttl = self.settings.get("ttl_days")
        if ttl is None or not self.writes:
            return
        now = self.clock()
        if now - getattr(self, "_last_expiry", float("-inf")) < 3600:
            return
        self._last_expiry = now
        cutoff = now - float(ttl) * 86400
        with self._lock:
            old = any(r["t"] < cutoff for r in self.rows.values()) and cutoff > self._erased_before
        if old:
            self.erase(before=cutoff)

    def typical_labelled(self) -> int:
        with self._lock:
            return sum(
                1
                for k, r in self.rows.items()
                if r["draws"].get("typical", {}).get("in") and (self.latest.get(k) or {}).get("type") == "label"
            )

    def _maybe_typical(self) -> None:
        """The typical slice's turn: refresh its levels; in budget mode, start it (a new epoch) once the student is
        here; end it (a new epoch, its share back to margin) once `typical_labels` of its answers are labelled."""
        self.typicality.refresh()
        if not self.writes:
            return
        with self._lock:
            s, e = self.settings, self.epoch
            on = s.get("typical_pi_max", 0.0) > 0
        if on and self.typical_labelled() >= s["typical_labels"]:
            why, change = "the typical slice ended", {"typical_pi_max": 0.0}
        elif (not on and s.get("per_day") is not None and s["shares"].get("typical", 0) > 0 and self.typical_wanted()
              and not any(x.get("why") == "the typical slice started" for x in self.epochs)):  # fmt: skip
            why, change = "the typical slice started", None
        else:
            return
        with self._lock:
            ended = why == "the typical slice ended"
            if s.get("per_day") is not None:  # recomputed: ended, its share goes back to margin
                assumed = None if _tau_set(e) else float(s["tau_after"])  # before the region: the assumed rate
                b = self._budget(s, e["tau_min"], typical=not ended, rate=assumed)
                new_settings, budget = b["settings"], b["budget"]
            else:
                new_settings, budget = {**s, **change}, None
            n = e["epoch"]

        def add(cur):
            cur = cur or [self._epoch_zero()]
            if cur[-1]["epoch"] != n:
                return None
            return cur + [{**cur[-1], "epoch": len(cur), "start": self.clock(), "settings": new_settings,
                           "why": why, "by": self.replica, **({"budget": budget} if budget else {})}]  # fmt: skip

        self._adopt(self._swap_epochs(add))

    def typical_status(self) -> dict:
        s = self.settings
        state = (
            "ended" if any(e.get("why") == "the typical slice ended" for e in self.epochs)
            else f"off: {self.embed_off}" if self.embed is None
            else "on" if s.get("typical_pi_max", 0.0) > 0 and self.typicality.cuts is not None
            else f"waiting for {TYPICAL_MIN} answers" if s.get("typical_pi_max", 0.0) > 0
            else "off"
        )  # fmt: skip
        return {"state": state, "labelled": self.typical_labelled(), "until": s["typical_labels"],
                "held": self.typicality.n}  # fmt: skip

    def _tau_now(self, region: float | None = None) -> tuple[float, int]:
        """The cert region's lowest confidence for the top `region` share (default the settings') of the answers held
        (under `_lock`)."""
        region = self.settings["region"] if region is None else region
        confs = np.sort([r["conf"] for r in self.rows.values()])
        tau = float(np.quantile(confs, 1.0 - region)) if len(confs) else 0.0
        if self.cert_threshold is not None:  # the region reaches down to the task's certified threshold at least
            tau = min(tau, float(self.cert_threshold))
        return tau, len(confs)

    def _adopt(self, stored: list) -> None:
        with self._lock:
            if stored and len(stored) > len(self.epochs):
                self.epochs = stored

    def set_settings(self, block) -> dict:
        """Change the settings after registration: a new epoch, stored by compare-and-swap, so every replica draws
        under it once it has read it (each row keeps the epoch it was drawn under). Returns the new epoch."""
        if not self.writes:
            raise LabelsReadOnly(READ_ONLY)
        if not isinstance(block, dict) or not block or "enabled" in block:
            raise LabelsError("labels settings: an object of the settings to change (remove the task to stop)")
        with self._lock:
            base = dict(self.settings)
            if any(k in block for k in PI_KEYS):
                base["per_day"] = None  # chances given: the budget is off
            elif {**base, **block}.get("per_day") is not None:
                for k in PI_KEYS:
                    base.pop(k, None)  # a budget: the chances follow it
            new_settings = parse_settings({**base, **block})
            region_moved = new_settings["region"] != self.settings["region"]
            tau, n = self._tau_now(new_settings["region"])
            if new_settings["per_day"] is not None and _tau_set(self.epoch):
                b = self._budget(new_settings, tau if region_moved else self.epoch["tau_min"])
                new_settings, budget = b["settings"], b["budget"]
            else:
                budget = None
        zero = self._epoch_zero()

        def add(cur):
            cur = cur or [zero]
            last = cur[-1]
            keep = _tau_set(last) and not region_moved
            return cur + [{"epoch": len(cur), "start": self.clock(), "settings": new_settings, "by": self.replica,
                           "tau_min": last["tau_min"] if keep or not _tau_set(last) else tau,
                           "tau_set": _tau_set(last), "why": "settings changed",
                           **({"budget": budget} if budget else {})}]  # fmt: skip

        stored = self._swap_epochs(add)
        self._adopt(stored)
        return stored[-1]

    def _index(self, row: dict) -> bool | None:
        """Add a row; False when it is erased (and so dropped)."""
        if row["request_id"] in self._erased_rids or row["t"] < self._erased_before:
            return False
        key = (row["request_id"], row["question"])
        if key not in self.rows:
            if row["t"] > self._evicted_upto:  # an answer max_rows dropped and read again (a merged file) is not new
                h = int(row["t"] // 3600)
                self._hours[h] = self._hours.get(h, 0) + 1
            self._first_t = row["t"] if self._first_t is None else min(self._first_t, row["t"])
            self.by_rid.setdefault(key[0], []).append(key)
            self.by_digest.setdefault(row["hk1"], []).append(key)
            if not row["drawn"]:
                self._undrawn.append(key)
        self.rows[key] = row

    def _apply(self, ev: dict) -> None:
        """Add one event to the in-memory log (under `_lock`): a label or decline is joined to its row, or parked
        until the row is known. An event seen before (a merged file repeats them) is skipped; an erased one too."""
        eid = ev.get("id")
        if eid is not None:
            if eid in self._ids:
                return
            self._ids.add(eid)
        if ev["type"] == "erase":
            self.events.append(ev)
            self._tombs[eid] = ev
            self._erase_in_memory(ev)
            return
        if ev["type"] == "purged":
            self.events.append(ev)
            self._purged.update(ev["of"])
            return
        if self._is_erased(ev):
            return
        self.events.append(ev)
        if ev["type"] in ("label", "decline") and ev.get("request_id") is not None:
            if not self._attach(ev):
                self._parked.setdefault(ev["request_id"], []).append(ev)

    def _is_erased(self, ev: dict) -> bool:
        if ev["type"] not in ("label", "decline"):
            return False
        return ev.get("request_id") in self._erased_rids or float(ev.get("received") or 0) < self._erased_before

    def _erase_in_memory(self, tomb: dict) -> None:
        """Drop what an erase event names, at once: the rows, their labels and declines (under `_lock`)."""
        if tomb.get("request_id") is not None:
            self._erased_rids.add(tomb["request_id"])
        if tomb.get("before") is not None:
            self._erased_before = max(self._erased_before, float(tomb["before"]))
        gone = [k for k, r in self.rows.items() if k[0] in self._erased_rids or r["t"] < self._erased_before]
        self._erased_rids.update(k[0] for k in gone)  # an erased answer's labels go with it, whenever they came
        for key in gone:
            row = self.rows.pop(key)
            for index, k in ((self.by_digest, row["hk1"]), (self.by_rid, key[0])):
                keys = index.get(k, [])
                if key in keys:
                    keys.remove(key)
                if not keys:
                    index.pop(k, None)
            self._history.pop(key, None)
            self.latest.pop(key, None)
            self.revisions -= self._changes.pop(key, 0)
        for rid in [r for r in self._parked if r in self._erased_rids]:
            self._parked.pop(rid)
        self.events = [e for e in self.events if not self._is_erased(e)]
        self._pending_rows = [r for r in self._pending_rows if (r["request_id"], r["question"]) in self.rows]

    @staticmethod
    def _order(ev: dict) -> tuple:
        return (float(ev.get("received") or 0.0), str(ev.get("replica") or ""))  # ties: one replica's write order

    def _attach(self, ev: dict) -> bool:
        """Join a label or decline to its row; False when the row is not known here (yet). The latest event by
        receipt time wins, whichever replica took it and in whatever order the replicas' files are read."""
        rid, q = ev["request_id"], ev.get("question")
        keys = [k for k in self.by_rid.get(rid, []) if q is None or k[1] == q]
        if len(keys) != 1:
            return False
        key = keys[0]
        if ev["type"] == "label" and ev["answer"] not in self.rows[key]["keys"]:
            self.invalid_on_join += 1
            return True
        hist = self._history.setdefault(key, [])
        hist.append(ev)
        hist.sort(key=self._order)
        self.latest[key] = hist[-1]
        answers = [e["answer"] for e in hist if e["type"] == "label"]
        changes = sum(a != b for a, b in zip(answers, answers[1:]))
        self.revisions += changes - self._changes.get(key, 0)
        self._changes[key] = changes
        return True

    def _join_parked(self) -> None:
        """Join the parked labels whose rows are known now; drop, and count as `unjoined_expired`, those whose
        request is older than `due_days` and a few listing turns: no replica will write that row any more, so the
        id was mistyped or made up."""
        for rid in [r for r in self._parked if r in self.by_rid]:
            waiting = self._parked.pop(rid)
            left = [ev for ev in waiting if not self._attach(ev)]
            if left:
                self._parked[rid] = left
        horizon = self.clock() - self.settings["due_days"] * 86400 - 3 * (SEGMENT_SECONDS + FLUSH_EVERY)
        for rid in [r for r in self._parked if (rid_time(r) or 0.0) < horizon]:
            self.unjoined_expired += len(self._parked.pop(rid))

    def _write_events(self, events: list[dict]) -> None:
        """Write events to the place, then add them to the log: a write that fails raises and leaves the log as it
        was, so nothing unsaved is ever acknowledged. The caller holds `_write`, never `_lock`, so the request path
        (`observe`) does not wait on the place."""
        if not events:
            return
        if not self.writes:
            raise LabelsReadOnly(READ_ONLY)
        for ev in events:
            ev.setdefault("id", ulid(self.clock()))
        with self._lock:
            name = self._name("events")
        self.place.put(name, "".join(json.dumps(e) + "\n" for e in events).encode())
        with self._lock:
            self._read.add(name)
            for ev in events:
                self._apply(ev)

    def flush(self, force: bool = False) -> int:
        """Write the buffered rows as one segment when there are `SEGMENT_ROWS` or the oldest is `SEGMENT_SECONDS`
        old (or `force`); returns how many were written. Called off the request path (the labels thread), and at
        shutdown. The write happens outside the log's lock; a failed write puts the rows back for the next try."""
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
                self._read.add(name)
                self._last_flush = self.clock()
            return len(rows)

    # ---- the request path

    @property
    def epoch(self) -> dict:
        return self.epochs[-1]

    def option_key(self, key) -> str:
        """How an option's key is kept: as given, or for a task whose options change with every request a keyed
        digest (`ok1:`), so no option of a request is stored; a label's answer is kept the same way."""
        if not self.dynamic:
            return key
        mac = hmac.new(bytes.fromhex(self.secret["digest_key"]), b"option\x00" + str(key).encode(), hashlib.sha256)
        return "ok1:" + mac.hexdigest()[:32]

    def observe(self, rid: str, question: str, *, text: str, keys, served, lp, h, certified, version, stack,
                lang: str | None, via: str = "model", option_texts=None,
                embed_text: str | None = None) -> dict:  # fmt: skip
        """Log one served question and draw it. Returns the answer's additions: {"request_id"} and, when the row was
        drawn into any slice, "label_requested": True; nothing on a replica that does not write. Cheap: a few hashes
        and an append under the lock, never a call to the place."""
        if not self.writes:
            return {}
        level = None
        if self.embed is not None and self.settings.get("typical_pi_max", 0.0) > 0:
            try:  # cold start only; a failing student never fails an answer
                src = text if embed_text is None else embed_text
                v = np.asarray(self.embed(src[:TYPICAL_TEXT]), dtype=np.float32).reshape(-1)  # a bounded cost
                level = self.typicality.level(v)
                self.typicality.add(v)
            except Exception:
                level = None
        p = np.asarray(served, dtype=np.float64)
        order = np.argsort(-p)
        top = int(order[0])
        conf = float(p[top])
        margin = float(p[top] - p[order[1]]) if len(p) > 1 else 1.0
        with self._lock:
            e = self.epoch
            d = draw(self.secret["salt"], rid, question, conf, margin, e, typical=level)
            drawn = any(v["in"] for v in d.values())
            sample = self._sample(rid, question)
            full = self.settings["features"] == "full"
            canonical = canonical_text(text)
            row = {
                "request_id": rid,
                "question": question,
                "t": self.clock(),
                "epoch": e["epoch"],
                "digest": hashlib.blake2b(canonical.encode(), key=bytes.fromhex(self.secret["salt"])[:64]).hexdigest(),
                "hk1": hk1(self.secret["digest_key"], text, canonical),
                "lang": lang or "unknown",
                "script": script_class(text),
                "keys": [self.option_key(k) for k in keys],
                "top": self.option_key(keys[top]),
                "options_digest": None
                if option_texts is None
                else hk1(self.secret["digest_key"], "\x1f".join(canonical_text(x) for x in option_texts)),
                "conf": conf,
                "margin": margin,
                "served_certified": certified,
                "via": via,
                "version": version,
                "stack": stack,
                "draws": d,
                "sample": sample,
                "drawn": drawn,
                "replica": self.replica,
                **(
                    {
                        "ps": _arr(p) if drawn or sample["in"] else None,
                        "lp": _arr(lp) if drawn else None,
                        "h": _arr(h) if drawn else None,
                    }
                    if full
                    else {}
                ),  # fmt: skip
            }
            if len(self.rows) >= self.settings["max_rows"]:
                self._evict()
            self._index(row)
            if len(self._pending_rows) < PENDING_MAX:
                self._pending_rows.append(row)
            else:  # the place has been out of reach for long: the row serves here but is not written
                self.unsaved += 1
        return {"request_id": rid, **({"label_requested": True} if drawn else {})}

    def _sample(self, rid: str, question: str) -> dict:
        """Whether an undrawn answer keeps its served distribution: a keyed draw at `reservoir / max_rows`, so the
        undrawn answers that keep it are a uniform sample of those held (about `reservoir` at the cap), the same on
        every replica: the unlabelled pool for monitoring. Every other undrawn answer keeps no distribution."""
        pi = min(1.0, self.settings["reservoir"] / self.settings["max_rows"])
        u = uniform(self.secret["salt"], rid, question, "reservoir")
        return {"in": u < pi, "pi": pi, "u": u}

    def _evict(self) -> bool:
        """Over `max_rows`: drop the oldest undrawn row (a drawn row stays: its label may still come). Constant time on
        the request path; False when every row left is drawn."""
        while self._undrawn:
            key = self._undrawn.popleft()
            row = self.rows.pop(key, None)
            if row is None:
                continue
            self._evicted_upto = max(self._evicted_upto, float(row["t"]))
            for index, k in ((self.by_digest, row["hk1"]), (self.by_rid, key[0])):
                keys = index.get(k, [])
                if key in keys:
                    keys.remove(key)
                if not keys:
                    index.pop(k, None)
            return True
        return False

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
        if not self.writes:
            raise LabelsReadOnly(READ_ONLY)
        now = self.clock()
        results, events = [], []
        with self._write:
            with self._lock:
                latest = dict(self.latest)  # this batch's view: earlier labels in the batch count for later ones
                for it in items:
                    results.append(self._one(it, now, latest, events))
            self._write_events(events)
        return {"results": results, **self.counts(), "need": {n["alpha"]: n["more_if_no_error"] for n in self.need()}}

    def _one(self, it, now, latest, events) -> dict:
        if not isinstance(it, dict):
            return {"status": "invalid", "why": "each label is an object"}
        refs = [k for k in ("request_id", "digest", "state", "text") if it.get(k) is not None]
        if len(refs) != 1:
            return {"status": "invalid", "why": "give exactly one of request_id, digest, state, text"}
        answer = it.get("answer")
        if self.dynamic and answer is not None:
            answer = self.option_key(str(answer))
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
            if key is None and self._elsewhere(rid):
                # served by another replica whose segment this one has not read yet: kept now, joined when it is
                ev = {"type": "label", "request_id": rid, "question": q, "answer": answer, "source": source,
                      "selection": None, "labelled_at": it.get("labelled_at") or now, "received": now,
                      "replica": self.replica}  # fmt: skip
                events.append(ev)
                return {"status": "not_joined", "request_id": rid, **({"question": q} if q is not None else {})}
            if key is None:
                t = rid_time(rid)
                if t is not None and t <= self._evicted_upto:  # served, then dropped past max_rows
                    return {"status": "expired_ref", "request_id": rid}
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
            if self.dynamic:  # the options are kept as digests: not shown
                return {"status": "invalid_answer", "why": "not one of this request's options", "request_id": key[0]}
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

    def _elsewhere(self, rid: str) -> bool:
        """A request id another replica of a shared store issued, within the time its row could still be kept."""
        t = rid_time(rid)
        if not self.shared or t is None or rid.rsplit("-", 1)[0] == self.replica:
            return False
        return -86400 < self.clock() - t < self.settings["due_days"] * 86400

    def decline(self, refs) -> dict:
        """The client will not label these drawn requests: they leave the queue and count as missing."""
        if not self.writes:
            raise LabelsReadOnly(READ_ONLY)
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
                    rec.update(answer=ev["answer"], correct=ev["answer"] == row["top"], source=ev["source"],
                               selection="slice" if row["drawn"] else "self", labelled_at=ev["labelled_at"],
                               replica_labelled=ev.get("replica"))  # fmt: skip
                out.append(rec)
            return sorted(out, key=lambda r: (r["t"], r["request_id"], r["question"]))

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
            return {"items": items, "pending": len(pend), "epoch": self.epoch["epoch"], "as_of": now,
                    "budget": {k: v for k, v in self.budget().items() if k in ("per_day", "rate_per_day")}}  # fmt: skip

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
            out = {"rows": len(self.rows), "states": states, "by_slice": by_slice, "imports": imports,
                   "revisions": self.revisions, "not_joined": sum(len(v) for v in self._parked.values())}  # fmt: skip
            if self.invalid_on_join:
                out["invalid_on_join"] = self.invalid_on_join
            if self.unsaved:
                out["unsaved_rows"] = self.unsaved
            if self.unjoined_expired:
                out["unjoined_expired"] = self.unjoined_expired
            if self._tombs:
                out["erasures"] = {"erased": len(self._tombs), "purged": len(self._purged & set(self._tombs))}
            return out

    @staticmethod
    def p_eval(row: dict) -> float:
        """The chance a row had of being drawn by an evaluation slice (audit or cert): 1 - (1 - pi_a)(1 - pi_c)."""
        d = row["draws"]
        return 1.0 - (1.0 - d["audit"]["pi"]) * (1.0 - d["cert"]["pi"])

    def eval_rows(self) -> list[dict]:
        """The rows an evaluation slice drew (audit or cert), whatever else drew them: certificates, accuracy,
        drift and challenger tests read these, and no fit may (an evaluation draw makes a row evaluation-only).
        Each: request_id, question, slices, pi per slice, p_eval, epoch, lang, conf, top, t, state, answer, correct
        (None while pending; expired and declined rows have no answer)."""
        now = self.clock()
        out = []
        with self._lock:
            for key, row in self.rows.items():
                d = row["draws"]
                if not (d["audit"]["in"] or d["cert"]["in"]):
                    continue
                out.append(self._role_rec(key, row, now, [s for s in EVAL_SLICES if d[s]["in"]]))
        return sorted(out, key=lambda r: (r["t"], r["request_id"], r["question"]))

    def fit_rows(self) -> list[dict]:
        """The labelled rows a fit may use: those drawn by a fit slice (margin, typical) and by no evaluation slice,
        weighted `1 / (p_fit x (1 - p_eval))` with `p_fit = 1 - (1 - pi_margin)(1 - pi_typical)` (the chance of being
        drawn by a fit slice and not by audit or cert), and the
        labels a client chose itself on rows no slice drew (`selection: "self"`, weight None: not probability-sampled,
        never for calibration). A label on a row an evaluation slice drew is never here."""
        now = self.clock()
        out = []
        with self._lock:
            for key, row in self.rows.items():
                ev = self.latest.get(key)
                if ev is None or ev["type"] != "label":
                    continue
                d = row["draws"]
                if d["audit"]["in"] or d["cert"]["in"]:
                    continue
                rec = self._role_rec(key, row, now, [s for s in FIT_SLICES if d.get(s, {}).get("in")])
                if rec["slices"]:  # drawn by margin or typical: the chance of either, and of no evaluation draw
                    p_fit = 1.0 - float(np.prod([1.0 - d[s]["pi"] for s in FIT_SLICES if s in d]))
                    rec["weight"] = 1.0 / (p_fit * (1.0 - self.p_eval(row)))
                    rec["selection"] = "slice"
                else:
                    rec["weight"], rec["selection"] = None, "self"
                out.append(rec)
        return sorted(out, key=lambda r: (r["t"], r["request_id"], r["question"]))

    def _role_rec(self, key, row, now, slices) -> dict:
        ev = self.latest.get(key)
        labelled = ev is not None and ev["type"] == "label"
        return {"request_id": key[0], "question": key[1], "slices": slices, "t": row["t"], "epoch": row["epoch"],
                "lang": row["lang"], "conf": row["conf"], "top": row["top"], "version": row["version"],
                "pi": {s: row["draws"][s]["pi"] for s in SLICES if s in row["draws"]}, "p_eval": self.p_eval(row),
                "state": self.state_of(key, self.latest, now), "answer": ev["answer"] if labelled else None,
                "correct": (ev["answer"] == row["top"]) if labelled else None}  # fmt: skip

    def cohort(self, cutoff: float | None = None) -> list[dict]:
        """A certificate's input: the evaluation rows drawn at or before `cutoff` (default now - due_days), so every
        one is labelled, expired or declined and none is left out for being slow to label (labelling delay can
        depend on whether the answer was right). Under the strict rule an expired or declined row is an error."""
        cutoff = self.clock() - self.settings["due_days"] * 86400 if cutoff is None else float(cutoff)
        out = [r for r in self.eval_rows() if r["t"] <= cutoff]
        for r in out:
            if r["state"] == "pending":  # only with a cutoff later than the default: not closed yet
                raise LabelsError(f"cohort: row {r['request_id']} is still pending at this cutoff; use an earlier one")
            r["strict_error"] = r["state"] != "labelled" or not r["correct"]
        return out

    def overview(self, top_langs: int = 20) -> dict:
        """What status adds to the counts: the response rate of drawn answers (labelled among those labelled,
        expired or declined), the audit slice's accuracy with a 90% Clopper-Pearson interval (on labelled answers, and
        strict, with every expired or declined one counted wrong), the time of the last label, and per language tag
        (at most `top_langs`, the busiest; "unknown" its own) the answers, drawn and labelled, and the labels still
        needed per alpha. Aggregates only: no row, text or chance per row."""
        now = self.clock()
        with self._lock:
            closed = labelled = 0
            a_n = a_right = a_closed = a_closed_right = 0
            langs: dict[str, dict] = {}
            last = None
            for key, row in self.rows.items():
                st = self.state_of(key, self.latest, now)
                ev = self.latest.get(key)
                lg = langs.setdefault(row["lang"], {"answers": 0, "drawn": 0, "labelled": 0})
                lg["answers"] += 1
                lg["drawn"] += bool(row["drawn"])
                lg["labelled"] += st == "labelled"
                if st == "labelled":
                    last = max(last or 0.0, float(ev["received"]))
                if row["drawn"] and st in ("labelled", "expired", "declined"):
                    closed += 1
                    labelled += st == "labelled"
                if row["draws"]["audit"]["in"] and st in ("labelled", "expired", "declined"):
                    right = st == "labelled" and ev["answer"] == row["top"]
                    a_closed += 1
                    a_closed_right += right
                    if st == "labelled":
                        a_n += 1
                        a_right += right
        with self._lock:
            week = [r for r in self.rows.values() if now - r["t"] <= 7 * 86400]
            share = (
                float(np.mean([1.0 - np.prod([1.0 - d["pi"] for d in r["draws"].values()]) for r in week]))
                if week else None
            )  # fmt: skip
        warnings = []
        if share is not None and len(week) >= 100 and self.cert_alpha is not None and self.cert_alpha <= STRICT_ALPHA:
            if share < STRICT_MIN_SHARE:
                warnings.append(f"certified at alpha {self.cert_alpha}: {share:.1%} of answers are drawn for labels, "
                                f"under the {STRICT_MIN_SHARE:.0%} it needs; raise per_day or cert_pi")  # fmt: skip
        busiest = sorted(langs, key=lambda k: -langs[k]["answers"])[:top_langs]
        for k in busiest:
            langs[k]["need"] = {n["alpha"]: n["more_if_no_error"] for n in self.need(lang=k)}
        return {
            "response_rate": round(labelled / closed, 4) if closed else None,
            "audit": {
                "labelled": a_n,
                "accuracy": round(a_right / a_n, 4) if a_n else None,
                "interval_90": accuracy_interval(a_right, a_n),
                "strict_accuracy": round(a_closed_right / a_closed, 4) if a_closed else None,
                "strict_interval_90": accuracy_interval(a_closed_right, a_closed),
            },  # fmt: skip
            "last_label_at": last,
            "since_last_label_s": None if last is None else round(now - last, 1),
            "languages": {k: langs[k] for k in busiest},
            "label_share": None if share is None else round(share, 4),
            **({"warnings": warnings} if warnings else {}),
        }

    def need(self, alphas=(0.05, 0.10, 0.01, 0.005), delta: float = 0.1, lang: str | None = None) -> list[dict]:
        """Per alpha: the cert slice's labelled answers in the current region and the errors among them (expired and
        declined rows counted as errors, the strict rule), the Clopper-Pearson upper bound, and how many more labels
        the region needs if no further error comes and if one more does.

        A planning figure, not a certificate: it counts the labels received so far and leaves pending rows out, and
        labelling delay can depend on whether the answer was right. A certificate reads a closed cohort (`cohort`).

        The region is the current epoch's: served confidence at or above its `tau_min`. A row of an earlier epoch
        counts when its confidence is in that region and that epoch's region contained the current one (its `tau_min`
        at most the current one), whatever its cert `pi`: each epoch's drawn rows are then a uniform sample of the
        current region's traffic in that epoch, and `pi` only changes how many there are, so the Clopper-Pearson bound
        is exact (`pooling: "exact"`). An epoch whose `tau_min` was higher drew nothing in the band between the two
        thresholds, and its rows are left out. Pending rows are shown apart (`region_pending`), never counted. Only cert
        rows are counted, which is conservative: audit rows of the region could be pooled under the same rule.
        `lang`: only that language tag's answers ("unknown" for untagged ones). `days_if_no_error` is the time to the
        labels still needed at the region's label rate over the last 7 days."""
        from decisio.readout import certify

        now = self.clock()
        with self._lock:
            e = self.epoch
            tau = e["tau_min"]
            covers = {x["epoch"] for x in self.epochs if x["tau_min"] <= tau}  # epochs whose region held the current
            latest = self.latest
            n = err = pending = recent = 0
            for key, row in self.rows.items():
                c = row["draws"]["cert"]
                if not c["in"] or row["conf"] < tau or row["epoch"] not in covers:
                    continue
                if lang is not None and row["lang"] != lang:
                    continue
                st = self.state_of(key, latest, now)
                if st == "labelled" and now - float(latest[key]["received"]) <= 7 * 86400:
                    recent += 1
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

        per_day = recent / 7.0
        out = []
        for a in alphas:
            m = more(err, a)
            days = None if m is None or per_day == 0 else round(m / per_day, 1)
            rec = {"alpha": a, "delta": delta, "region_labels": n, "region_errors": err, "region_pending": pending,
                   "region_labels_per_day": round(per_day, 2), "days_if_no_error": days,
                   "pooling": "exact",
                   "upper_bound": None if n == 0 else round(float(certify.upper_bound(err, n, delta)), 5),
                   "more_if_no_error": m, "more_if_one_more_error": more(err + 1, a)}  # fmt: skip
            if n and err / n >= a:
                rec["reason"] = (
                    "the errors so far are at or above alpha: it certifies only after more right labels bring the "
                    "error down"
                )
            out.append(rec)
        return out


def accuracy_interval(right: int, n: int, level: float = 0.9) -> list[float] | None:
    """The two-sided Clopper-Pearson interval on an accuracy seen as `right` of `n`, at `level`."""
    if n == 0:
        return None
    from scipy.stats import beta

    a = (1 - level) / 2
    lo = 0.0 if right == 0 else float(beta.ppf(a, right, n - right + 1))
    hi = 1.0 if right == n else float(beta.ppf(1 - a, right + 1, n - right))
    return [round(lo, 4), round(hi, 4)]


def _tau_set(epoch: dict) -> bool:
    """Whether an epoch's cert region has been set from traffic (epoch 0's has not: it is all traffic)."""
    return bool(epoch.get("tau_set", epoch["epoch"] > 0))


def assert_disjoint(fit_ids, cert_ids) -> None:
    """A certificate must use no label its version was fitted on: raise when the two sets share one."""
    shared = set(map(tuple, fit_ids)) & set(map(tuple, cert_ids))
    if shared:
        raise LabelsError(f"a certificate would use {len(shared)} label(s) its version was fitted on: refused")


# ---- a shared task store -------------------------------------------------------------------------------------------


_FILE_NAME = re.compile(r"^(rows|events)/(\d{13}-\d+)-(.+)\.jsonl$")


def shared_prefix(task_digest: str, salt: str) -> str:
    """A registration's label log on a shared task store: `labels/<task digest>/<first 12 of its salt>/`."""
    return f"labels/{task_digest}/{salt[:12]}/"


class SharedPlace:
    """A task's label log on a shared task store (`fast_table_replicas.Objects`), as `TaskLabels` reads and writes it:

        labels/<task digest>/<registration>/EPOCHS.json                        the epochs, by compare-and-swap
        labels/<task digest>/<registration>/<replica>/rows/<ms>-<seq>.jsonl    one replica's rows, written once
        labels/<task digest>/<registration>/<replica>/events/<ms>-<seq>.jsonl  its label events, written once

    A replica writes only under its own name and only with create-only writes, so replicas never overwrite each
    other. `names` lists only what is new since its last call (each replica's files are listed after the last one
    seen), so a poll costs one listing of the replicas and one short listing per replica and kind. A replica that
    does not write (`writes=False`) reads everything and writes nothing."""

    persistent = True
    shared = True

    def __init__(self, objects, prefix: str, replica: str, writes: bool = True):
        self.objects, self.prefix, self.replica, self.writes = objects, prefix, replica, writes
        self._after: dict[str, str] = {}  # "<replica>/<kind>" -> the last path listed there

    def describe(self) -> str:
        return f"task store {self.prefix}"

    @staticmethod
    def _split(name: str) -> tuple[str, str, str]:
        """`rows/<ms>-<seq>-<replica>.jsonl` -> ("rows", "<ms>-<seq>", "<replica>"), whatever the sequence's width."""
        m = _FILE_NAME.match(name)
        if m is None:
            raise LabelsError(f"not a label log file name: {name!r}")
        return m.group(1), m.group(2), m.group(3)

    def _path(self, name: str) -> str:
        if name == "EPOCHS.json":
            return self.prefix + name
        kind, stem, replica = self._split(name)
        return f"{self.prefix}{replica}/{kind}/{stem}.jsonl"

    def put(self, name: str, data: bytes) -> None:
        if not self.writes:
            raise LabelsReadOnly(READ_ONLY)
        self.objects.create(self._path(name), data)

    def get(self, name: str) -> bytes | None:
        got = self.objects.get(self._path(name))
        return None if got is None else got[0]

    def names(self, prefix: str) -> list[str]:
        """The files of one kind (`rows/` or `events/`) after each replica's cursor, as `TaskLabels` names them. The
        cursor does not move here: `advance` moves it past a file once it has been read, so a read that fails leaves
        that file and every later one to be listed again."""
        kind = prefix.rstrip("/")
        out = []
        for rdir in self.objects.prefixes(self.prefix):
            replica = rdir[len(self.prefix) :].strip("/")
            at = f"{self.prefix}{replica}/{kind}/"
            for path in self.objects.list(at, offset=self._after.get(f"{replica}/{kind}")):
                stem = path[len(at) :].removesuffix(".jsonl")
                out.append(f"{kind}/{stem}-{replica}.jsonl")
        return sorted(out)

    def advance(self, name: str) -> None:
        """The file `name` has been read (or is gone): its replica's listing resumes after it. Files of one replica
        are read in name order and a failed read stops the reading, so nothing before the cursor is left unread."""
        kind, stem, replica = self._split(name)
        path = f"{self.prefix}{replica}/{kind}/{stem}.jsonl"
        key = f"{replica}/{kind}"
        if path > self._after.get(key, ""):
            self._after[key] = path

    def swap(self, name: str, fn, races: int = 8) -> list:
        """Move a shared object by compare-and-swap: `fn(current or None)` gives the new value, or None to keep it.
        Retried when another replica wrote first; returns the value stored."""
        from decisio.serve.fast_table_replicas import Exists, Raced

        path = self._path(name)
        for _ in range(races):
            got = self.objects.get(path)
            cur, tag = (None, None) if got is None else (json.loads(got[0]), got[1])
            new = fn(cur)
            if new is None or not self.writes:
                return cur or []
            data = json.dumps(new).encode()
            try:
                if tag is None:
                    self.objects.create(path, data)
                else:
                    self.objects.update(path, data, tag)
                return new
            except (Exists, Raced):
                continue
        raise LabelsError(f"{path}: another replica kept writing it ({races} tries)")

    def delete(self, name: str) -> None:
        if not self.writes:
            raise LabelsReadOnly(READ_ONLY)
        self.objects.delete(self._path(name))

    def remove_all(self) -> None:
        """Delete the whole log (the task was removed on this replica)."""
        if not self.writes:
            return
        for path in self.objects.list(self.prefix):
            self.objects.delete(path)
