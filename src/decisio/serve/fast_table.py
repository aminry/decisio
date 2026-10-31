# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""An opt-in fast table for a fixed-option task: a sentence encoder and a per-task classifier answer first.

For short texts that arrive in volume (an assistant's commands, spam, a fixed set of intents) a small classifier trained
on the task's own labelled examples answers most requests faster than decisio's own path, on the host's CPU. The table
is the encoder's embedding of the request's text, a multinomial logistic regression on the task's examples, and a
confidence threshold; a request whose confidence reaches the threshold is answered by the table and decisio's model is
not asked. Every other request goes to decisio as it would without the table.

The table is off unless the registration asks for it (`"fast_table": true`), and it keeps itself only where it pays:
at registration the server times the table and decisio's own path on the task's certification examples and keeps the
table only if the expected mean time per request falls by at least a quarter (`MIN_SAVING`). Where it does not, the
registration says so and the task is served as it would be without a table. It does not pay on long texts: the encoder
runs in front of every request, and on a text of several hundred tokens it costs more than decisio's own path.

The table and decisio form a two-stage cascade with one certificate (the way the certificate of a task is made, on
examples held out of the fit): the table's threshold is the lowest at which at most `alpha` of its answers are wrong,
with probability at least 1 - delta/2; decisio's threshold is made the same way on the requests the table defers. An
answer of either stage whose confidence reaches its threshold is certified, and the promise is the cascade's: at most
`alpha` wrong among the certified answers, with probability at least 1 - delta.

Needs the `table` extra (sentence-transformers, which brings scikit-learn). The encoder is `BAAI/bge-base-en-v1.5` at a
pinned revision. Standard library, numpy, and the extra.
"""

from __future__ import annotations

import base64
import collections
import math
import os
import platform
import threading
import warnings

import numpy as np

from decisio.names import record_format
from decisio.readout import certify
from decisio.serve import monitor
from decisio.serve.stack import sha256

FORMAT = record_format("fast-table")
ENCODER = "BAAI/bge-base-en-v1.5"
REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
MAX_SEQ_LENGTH = 512
CS = (0.1, 1.0, 10.0, 100.0)  # the L2 strength of the classifier, chosen by 5-fold cross-validated log loss
MIN_SAVING = 0.25  # the table is kept when it cuts the expected mean time per request by at least this share
MIN_EXAMPLES = 100  # fewer fit examples than this fit no table
FOLDS = 5
WINDOW = 50  # table-path latencies kept for the rolling median
MIN_LATENCY_SAMPLES = 20  # a rolling median is judged only from this many table answers
SHADOW_EVERY = 10  # while the table is off, every 10th request also times the table (after decisio has answered)
SHADOW_N = 30  # and the table is put back when 30 such pairs show it pays (`timing_guard`)
SAMPLE_EVERY = 50  # while on, every 50th table answer is also read by decisio and compared

_lock = threading.Lock()
_run = threading.Lock()  # one encode at a time: the encoder is shared by every table task
_encoder = None


class TableUnavailable(ImportError):
    """The `table` extra is not installed."""


def encoder():
    """The sentence encoder, loaded once on the CPU."""
    global _encoder
    with _lock:
        if _encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise TableUnavailable(
                    "the fast table needs the `table` extra: pip install 'decisio[table]' (sentence-transformers)"
                ) from e
            m = SentenceTransformer(ENCODER, device="cpu", revision=REVISION)
            m.max_seq_length = MAX_SEQ_LENGTH
            _encoder = m
    return _encoder


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings of the texts, float32, one row each."""
    enc = encoder()
    with _run:
        out = enc.encode(
            list(texts), batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
    return np.asarray(out, dtype=np.float32)


def _pack(a) -> dict:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return {"dtype": "float32", "shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def unpack(d) -> np.ndarray:
    if isinstance(d, np.ndarray):
        return d
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.float32).reshape(d["shape"]).copy()


def fit(X: np.ndarray, y, k: int) -> dict:
    """The stored table from embeddings X (n x d) and gold option indices y (below k): a multinomial logistic regression
    with L2, the strength chosen from `CS` by 5-fold stratified cross-validated log loss. Needs scikit-learn."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except ImportError as e:
        raise TableUnavailable("the fast table needs the `table` extra: pip install 'decisio[table]'") from e
    y = np.asarray(y, dtype=int)

    def fitted(c, Xa, ya):
        m = LogisticRegression(C=c, max_iter=2000)
        m.fit(Xa, ya)
        return m

    def probs(m, Xb):
        P = np.zeros((len(Xb), k))
        P[:, m.classes_] = m.predict_proba(Xb)
        return P

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # classes with fewer than 5 members in a fold; convergence notes
        folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(X, y))
        loss = {}
        for c in CS:
            ll = []
            for tr, te in folds:
                P = probs(fitted(c, X[tr], y[tr]), X[te])
                ll.append(-np.log(np.clip(P[np.arange(len(te)), y[te]], 1e-12, 1)).mean())
            loss[c] = float(np.mean(ll))
        best = min(CS, key=lambda c: (loss[c], c))
        m = fitted(best, X, y)
    return {
        "format": FORMAT,
        "encoder": ENCODER,
        "revision": REVISION,
        "k": k,
        "classes": [int(c) for c in m.classes_],
        "binary": bool(m.coef_.shape[0] == 1),
        "coef": _pack(m.coef_),
        "intercept": _pack(m.intercept_),
        "C": best,
        "cv_logloss": {str(c): round(v, 5) for c, v in loss.items()},
        "n_examples": int(len(y)),
    }


def predict(rec: dict, X: np.ndarray) -> np.ndarray:
    """(n x k) probabilities of the stored table for embeddings X; an option the table never saw gets 0."""
    coef, b = unpack(rec["coef"]).astype(np.float64), unpack(rec["intercept"]).astype(np.float64)
    z = np.asarray(X, dtype=np.float64) @ coef.T + b
    if rec["binary"]:
        p1 = 1.0 / (1.0 + np.exp(-z[:, 0]))
        q = np.stack([1.0 - p1, p1], axis=1)
    else:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        q = e / e.sum(axis=1, keepdims=True)
    P = np.zeros((len(q), rec["k"]))
    P[:, rec["classes"]] = q
    return P


def confidence(P: np.ndarray):
    """(confidence, prediction) of each row: the top probability and its option."""
    return P.max(axis=1), P.argmax(axis=1)


# ---- the two-stage certificate ---------------------------------------------------------------------------------------


def cascade(cT, pT, y, cD, pD, alpha: float, delta: float) -> dict:
    """The thresholds of the cascade on the held-out examples: the table's `(cT, pT)` and decisio's `(cD, pD)`
    confidences and predictions against the gold options y. The table's threshold at (alpha, delta/2); decisio's on the
    requests the table defers, at (alpha, delta/2). A stage with no threshold that holds is None."""
    cT, pT, y, cD, pD = (np.asarray(a) for a in (cT, pT, y, cD, pD))
    tau_t = certify.threshold(cT, pT == y, alpha, delta / 2)
    table = cT >= tau_t if np.isfinite(tau_t) else np.zeros(len(y), bool)
    deferred = ~table
    tau_d = certify.threshold(cD[deferred], pD[deferred] == y[deferred], alpha, delta / 2) if deferred.any() else np.inf
    return {
        "table_threshold": float(tau_t) if np.isfinite(tau_t) else None,
        "decisio_threshold": float(tau_d) if np.isfinite(tau_d) else None,
        "table_share": float(table.mean()) if len(y) else 0.0,
        "n": int(len(y)),
    }


def served(cT, pT, cD, pD, thresholds: dict):
    """(answered by the table, certified, predicted option) for held-out or live confidences under the thresholds of
    `cascade`."""
    cT, pT, cD, pD = (np.asarray(a) for a in (cT, pT, cD, pD))
    tt, td = thresholds["table_threshold"], thresholds["decisio_threshold"]
    by_table = cT >= tt if tt is not None else np.zeros(len(cT), bool)
    certified = by_table | (~by_table & (cD >= td if td is not None else np.zeros(len(cT), bool)))
    return by_table, certified, np.where(by_table, pT, pD)


def certificate(alpha: float, delta: float, thresholds: dict, by_table, certified, correct) -> dict:
    """The numbers of the cascade's certification on the held-out examples, in the fields of a single-stage certificate:
    `threshold` is decisio's (on the requests the table defers), `table` the table's, the counts are the cascade's."""
    by_table, certified, correct = (np.asarray(a, dtype=bool) for a in (by_table, certified, correct))
    n = len(correct)
    stages = {}
    for name, mask, thr in (
        ("table", by_table, thresholds["table_threshold"]),
        ("decisio", certified & ~by_table, thresholds["decisio_threshold"]),
    ):
        answered = int(mask.sum())
        wrong = int((mask & ~correct).sum())
        stages[name] = {
            "threshold": thr,
            "certified_n": answered,
            "errors": wrong,
            "upper_bound": certify.upper_bound(wrong, answered, delta / 2) if answered else None,
        }
    cert_n, wrong = int(certified.sum()), int((certified & ~correct).sum())
    return {
        "procedure": certify.PROCEDURE
        + ", two stages (the table's threshold, then decisio's on what the table defers)",
        "alpha": alpha,
        "delta": delta,
        "n": n,
        "accuracy": float(correct.mean()) if n else None,
        "threshold": thresholds["decisio_threshold"],
        "certified": thresholds["table_threshold"] is not None,
        "table_threshold": thresholds["table_threshold"],
        "certified_share": cert_n / n if n else 0.0,
        "certified_n": cert_n,
        "errors_at_threshold": wrong,
        "error_rate_at_threshold": wrong / cert_n if cert_n else None,
        "upper_bound_at_threshold": None,
        "empirical_share_at_alpha": cert_n / n if n else 0.0,
        "stages": stages,
    }


def summary(c: dict, table_share: float) -> str:
    """The cascade's certificate in words."""
    st = c["stages"]
    return (
        f"Answers of the fast table whose top probability is at least {c['table_threshold']:.4f}, and answers of "
        f"decisio "
        f"(on the requests the table defers) whose top probability is at least "
        + (f"{c['threshold']:.4f}" if c["threshold"] is not None else "(no threshold holds: none)")
        + f", are certified: on the {c['n']} held-out examples that is {c['certified_share']:.0%} of the answers, "
        f"{c['errors_at_threshold']} of {c['certified_n']} wrong ({st['table']['certified_n']} answered by the table, "
        f"{st['table']['errors']} wrong; {st['decisio']['certified_n']} by decisio, {st['decisio']['errors']} wrong). "
        "The table's stage keeps its promise by the procedure's proof: with probability at least "
        f"{1 - c['delta'] / 2:.0%} over the draw of such examples, at most {c['alpha']:.0%} of its certified answers "
        "are wrong. Decisio's stage is certified on the requests the table's chosen threshold defers, from the same "
        "held-out examples, so its promise is supported by simulation (about 1 draw in 100 broke it), not proved. "
        "Both are for traffic like the examples and on this serving stack. "
        f"The table answers {table_share:.0%} of requests without asking the model."
    )


def timing_guard(table_ms, decisio_ms, table_share: float, min_saving: float = MIN_SAVING, deferred=None) -> dict:
    """Whether the table pays: the expected mean time per request with the table in front against decisio's own, both
    measured on the same requests one at a time. `table_ms` and `decisio_ms` are per-request milliseconds. With
    `deferred` (which requests the table defers) the expected time is the mean of the table's time plus decisio's on
    each deferred request, so a deferral of the slow requests is counted as it is; without it, the table's mean time
    plus decisio's mean on the share `table_share` it defers."""
    t, d = float(np.mean(table_ms)), float(np.mean(decisio_ms))
    if deferred is not None:
        expected = float(
            np.mean(np.asarray(table_ms, dtype=float) + np.asarray(deferred) * np.asarray(decisio_ms, dtype=float))
        )
    else:
        expected = t + (1.0 - table_share) * d
    saving = 1.0 - expected / d if d > 0 else 0.0
    return {
        "kept": bool(saving >= min_saving),
        "saving": round(saving, 4),
        "required": min_saving,
        "table_ms_mean": round(t, 2),
        "decisio_ms_mean": round(d, 2),
        "table_share": round(table_share, 4),
        "expected_ms_mean": round(expected, 2),
    }


# ---- one replica's use of the table -----------------------------------------------------------------------------


def cpu_model() -> str:
    """The host CPU's model name, as the host gives it."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip()
        if out:
            return out
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def timing_stack(engine_facts: dict) -> dict:
    """What a timing of the table against decisio is true for: the host CPU, the engine and its card, and the encoder
    with its revision and libraries. A replica whose own differs does not trust the timing."""
    from importlib.metadata import PackageNotFoundError, version

    def lib(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    keep = ("engine", "vllm", "mlx", "mlx_lm", "gpu", "device", "machine", "quantization")
    return {
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "machine": platform.machine(),
        "engine": {k: engine_facts[k] for k in keep if k in engine_facts},
        "encoder": ENCODER,
        "revision": REVISION,
        "torch": lib("torch"),
        "sentence_transformers": lib("sentence-transformers"),
    }


def stack_differences(a: dict, b: dict) -> list[str]:
    """The dotted names of the facts on which two timing stacks differ."""
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) and isinstance(y, dict):
            out += [f"{k}.{d}" for d in stack_differences(x, y)]
        elif x != y:
            out.append(k)
    return out


def agreement_baseline(disagree, seed: str) -> dict | None:
    """The record of a CUSUM on how often decisio's answer differs from the table's, for the held-out examples the table
    answered (`disagree` flags each one); None when there are too few of them, or the rate is too high to tell a rise
    from normal. The same rule as the monitor's other rate signals."""
    d = np.asarray(disagree, dtype=bool)
    if len(d) < 20:
        return None
    p0 = max(float(d.mean()), 1.0 / len(d))
    p_ok, p_bad = monitor.OK_FACTOR * p0, monitor.BAD_FACTOR * p0
    if p_bad > 0.5:
        return None
    return {
        "share": p0,
        "p_ok": p_ok,
        "p_bad": p_bad,
        "limit": monitor.cusum_limit(p_ok, p_bad, "agreement:" + seed),
        "false_alarm": monitor.FALSE_ALARM,
        "horizon": monitor.HORIZON,
    }


class Agreement:
    """A CUSUM on sampled table answers that decisio would have answered otherwise."""

    def __init__(self, base: dict):
        self.base = base
        self._up = math.log(base["p_bad"] / base["p_ok"])
        self._down = math.log((1 - base["p_bad"]) / (1 - base["p_ok"]))
        self.statistic, self.alerting, self.n, self.disagreed = 0.0, False, 0, 0

    def observe(self, disagree: bool) -> None:
        self.n += 1
        self.disagreed += bool(disagree)
        lim = self.base["limit"]
        self.statistic = min(8 * lim, max(0.0, self.statistic + (self._up if disagree else self._down)))
        if self.statistic > lim:
            self.alerting = True
        elif self.statistic < lim / 2:
            self.alerting = False

    def facts(self) -> dict:
        return {
            "statistic": self.statistic,
            "limit": self.base["limit"],
            "registered_disagreement": self.base["share"],
            "sampled": self.n,
            "disagreed": self.disagreed,
            "alerting": self.alerting,
        }


class Runtime:
    """One replica's use of one task's table: whether it answers, and why not when it does not.

    `active`: the table answers the requests it is sure of. `timing`: the table is not used and is being timed against
    decisio on this replica's own requests (the timing at registration was made on another stack, or the table stopped
    beating decisio under this replica's load); it comes back when 30 paired timings show the saving. `suspended`: the
    monitor's alarm fired (on this replica or another that shares the task store). `unavailable`: this server cannot run
    the table (another encoder revision, or the `table` extra is missing). Not thread-safe by itself: the server
    calls it from the request path, one task at a time."""

    def __init__(self, rec: dict, stack: dict):
        self.rec = rec
        self.stack = stack
        self.stack_sha = sha256(stack)
        self.share = rec["thresholds"]["table_share"]
        self.registered_ms = rec["timing"]["guard"]["decisio_ms_mean"]
        self.table_ms: collections.deque = collections.deque(maxlen=WINDOW)
        self.pairs: list[tuple[float, float]] = []
        self.requests = 0
        self.table_answers = 0
        self.last_guard = None
        self.retimed = None
        self.agreement = Agreement(rec["agreement"]) if rec.get("agreement") else None
        self.state, self.reason = "active", None
        if rec["encoder"] != ENCODER or rec["revision"] != REVISION:
            self.state = "unavailable"
            self.reason = (
                f"fitted with {rec['encoder']} at revision {rec['revision'][:8]}; this server has {ENCODER} at "
                f"{REVISION[:8]}: register the task again"
            )
        elif rec["timing"]["stack_sha256"] != self.stack_sha:
            self.state = "timing"
            diff = stack_differences(rec["timing"]["stack"], stack)
            self.reason = (
                f"the table was timed on another stack ({', '.join(diff)}): serving through decisio while this "
                "replica times the table on its own requests"
            )
        sus = rec.get("suspended")
        if sus:
            self.state, self.reason = "suspended", f"suspended by {sus.get('by', 'a replica')}: {sus.get('reason')}"

    # state changes

    def suspend(self, reason: str) -> None:
        self.state, self.reason = "suspended", reason

    def unavailable(self, reason: str) -> None:
        self.state, self.reason = "unavailable", reason

    @property
    def uses_table(self) -> bool:
        return self.state == "active"

    # the table's own latency, while it answers

    def table_answered(self, ms: float) -> None:
        """A request answered by the table in `ms`. Falls back to decisio when the rolling median no longer pays: the
        expected time per request with the table (its median on every request, decisio's registered mean on the share it
        defers) must still be `MIN_SAVING` below decisio's registered mean."""
        self.table_answers += 1
        self.table_ms.append(ms)
        if len(self.table_ms) < MIN_LATENCY_SAMPLES:
            return
        p50 = float(np.median(self.table_ms))
        expected = p50 + (1.0 - self.share) * self.registered_ms
        if expected > (1.0 - MIN_SAVING) * self.registered_ms:
            self.state = "timing"
            self.reason = (
                f"under this replica's load the table's median is {p50:.0f} ms against decisio's registered "
                f"{self.registered_ms:.0f} ms, so the expected time per request is {expected:.0f} ms, not "
                f"{(1.0 - MIN_SAVING) * self.registered_ms:.0f} ms or less: serving through decisio while the table is "
                "timed again on this replica's requests"
            )
            self.table_ms.clear()
            self.pairs.clear()

    # re-timing, while the table is off for speed

    def shadow_due(self) -> bool:
        """Counts a request answered by decisio while the table is being timed; True when this one is to time the table
        too."""
        if self.state != "timing":
            return False
        self.requests += 1
        return self.requests % SHADOW_EVERY == 0

    def shadowed(self, table_ms: float, decisio_ms: float) -> None:
        """One request timed both ways: the table's time (embedding and classifier) and decisio's for the request."""
        self.pairs.append((table_ms, decisio_ms))
        if len(self.pairs) < SHADOW_N:
            return
        g = timing_guard([a for a, _ in self.pairs], [b for _, b in self.pairs], self.share)
        self.last_guard = g
        self.pairs.clear()
        if g["kept"]:
            self.state, self.reason, self.retimed = "active", None, g
            self.table_ms.clear()

    # sampled agreement, while the table answers

    def sample_due(self) -> bool:
        return self.state == "active" and self.agreement is not None and self.table_answers % SAMPLE_EVERY == 0

    def facts(self) -> dict:
        out = {
            "state": self.state,
            "uses_table": self.uses_table,
            "table_share_registered": self.share,
            "timed_on": self.rec["timing"]["stack_sha256"][:12],
            "this_stack": self.stack_sha[:12],
        }
        if self.reason:
            out["reason"] = self.reason
        if self.table_ms:
            out["table_ms_p50"] = round(float(np.median(self.table_ms)), 2)
            out["table_ms_samples"] = len(self.table_ms)
        if self.retimed:
            out["retimed"] = self.retimed
        if self.last_guard and not self.retimed:
            out["last_timing"] = self.last_guard
        if self.state == "timing":
            out["timing_progress"] = f"{len(self.pairs)} of {SHADOW_N} paired timings"
        if self.agreement:
            out["agreement"] = self.agreement.facts()
        else:
            out["agreement"] = {"armed": False, "why": "too few held-out table answers to tell a rise in disagreement"}
        return out
