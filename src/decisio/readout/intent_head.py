# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The intent head: a linear correction to the letter readout, fitted per task on labelled examples.

Intent tasks with many options (Banking77's 77, CLINC's 150) are where the one-pass readout is weakest. A head on
the final hidden state h at the readout position (the vector the letter logits are read from) closes most of that
gap when the customer labels 5 to 10 examples per intent:

    logits = letter log-scores (K-way log-softmax, in the task's option order) + W f(h) + b

f standardises h and projects it on the labelled examples' top principal directions (at most 512). The L2 penalty
is chosen by 5-fold stratified cross-validated log loss on the customer's own examples, among LAMS and "no head";
if "no head" wins, the task stores the head as off and serves exactly as the letter readout. For serving the head
folds into one d x K matrix and a K-vector: logits = letter log-scores + h @ A + c.

The head is served as softmax(logits / T), with one temperature T per task. The head fits its examples tightly and
keeps the scale of the letter log-scores, which are overconfident by the factor the served global temperature removes
from the untaught readout, so without T it is more confident than it is right (on the 31B, ECE 0.12 to 0.19 against
0.046 for the untaught readout). T is fitted on the head's out-of-fold logits, the five folds of the penalty's
cross-validation, each fold refitted at the chosen penalty: it never sees an example the head was fitted on, nor any
held-out evidence. The arg-max does not change. A record without a temperature serves as T = 1.

The head is bound to the option list it was fitted on, in that order: it learns which answer slot to trust, so a
reordered, extended or shortened list needs a re-fit (under reordered options the head fell below the plain
readout). For the same reason it is not combined with cyclic-shift averaging.

    rec = fit_intent_head(logps, H, labels, options, fingerprint=..., task_id=...)
    p = apply_intent_head(logp, h, rec, options)     # at serving
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from decisio.names import record_format

FORMAT = record_format("intent-head")
# Labelled examples every option needs before a head is fitted. 5 and 10 per intent passed the acceptance gate on both
# intent sets, and 2 passed on one. 2 was measured offline on ten decision sets and two bases at 50, 200 and 1,000
# examples (runs/2026-10-30_intent-head-floor): pooled macro-F1 +1.08 points over a floor of 5, no set worse by more
# than 0.1 point, the certificate kept; the head's cross-validation still declines it where the plain readout wins.
MIN_PER_OPTION = 2
SHORT_LISTED = 12  # the decline reason names at most this many short options


def _fit_linear(Z, base, y, K, lam, w0=None):
    """Multinomial logistic regression on features Z (n x m) with offset `base` (n x K log-scores; zeros for a
    classifier on the hidden state alone), L2 penalty lam on W and b. Returns the flat parameter vector."""
    from scipy.optimize import minimize

    n, m = Z.shape

    def f(w):
        W = w[: K * m].reshape(K, m)
        b = w[K * m :]
        s = base + Z @ W.T + b
        s = s - s.max(1, keepdims=True)
        lse = np.log(np.exp(s).sum(1))
        loss = (lse - s[np.arange(n), y]).mean() + lam * (w**2).sum()
        P = np.exp(s - lse[:, None])
        P[np.arange(n), y] -= 1
        g = np.concatenate([(P.T @ Z / n).ravel(), P.mean(0)]) + 2 * lam * w
        return loss, g

    x0 = np.zeros(K * m + K) if w0 is None else w0
    return minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 300}).x


LAMS = [100.0, 10.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]  # strongest first: each fit warm-starts the next
MAX_DIMS = 512
T_RANGE = (0.2, 20.0)  # the head's temperature is searched within this range, on log T


def _log_loss(S, y):
    """Mean negative log-likelihood of the gold option under softmax(S), row by row."""
    S = S - S.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(S).sum(1)) - S[np.arange(len(y)), y]))


def _ece(S, y, bins=10):
    """Expected calibration error of softmax(S)'s top probability: `bins` bins of equal item count
    (decisio.bench.di_cal.ece_equal_mass)."""
    S = S - S.max(1, keepdims=True)
    P = np.exp(S) / np.exp(S).sum(1, keepdims=True)
    conf, ok = P.max(1), (P.argmax(1) == y).astype(float)
    e = 0.0
    for chunk in np.array_split(np.argsort(conf), bins):
        if len(chunk):
            e += len(chunk) / len(conf) * abs(ok[chunk].mean() - conf[chunk].mean())
    return float(e)


def fit_temperature(S, y):
    """The T in T_RANGE that minimises the log loss of softmax(S / T) on the rows S (n x K) with gold options y."""
    from scipy.optimize import minimize_scalar

    S = np.asarray(S, dtype=np.float64)
    r = minimize_scalar(
        lambda u: _log_loss(S / np.exp(u), y),
        bounds=(np.log(T_RANGE[0]), np.log(T_RANGE[1])),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(np.exp(r.x))


def _features(H):
    """Standardise, then project on the examples' top principal directions (at most n - 1 and MAX_DIMS of them),
    scaled to unit variance. Fitted on the training examples only."""
    mu, sd = H.mean(0), H.std(0) + 1e-6
    X = (H - mu) / sd
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(int((S > 1e-6 * S[0]).sum()), MAX_DIMS, len(H) - 1)
    V = Vt[:r].T / S[:r] * np.sqrt(len(H))

    def f(H2):
        return ((H2 - mu) / sd) @ V

    f.params = (mu, sd, V)
    return f


def _folds(y, k, rng):
    """Stratified fold labels: each label's examples spread over the folds, from a random fold onwards."""
    f = np.empty(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        f[idx] = (rng.integers(k) + np.arange(len(idx))) % k
    return f


def fit_linear(H, base, y, K, offset, seed=0):
    """A linear head fitted on labelled examples.
    offset=True:  the letter-score head, logits = letter log-scores + W h + b.
    offset=False: the classifier, logits = W h + b, on the hidden state alone.
    The penalty is chosen by 5-fold stratified cross-validated log loss among LAMS and "no head" (the letter
    readout); returns a function (H, base) -> log-scores and the choice."""
    rng = np.random.default_rng(seed)
    folds = _folds(y, 5, rng)

    def ll(s, yy):
        return float(np.mean(np.log(np.exp(s - s.max(1, keepdims=True)).sum(1)) + s.max(1) - s[np.arange(len(yy)), yy]))

    cv = {"none": [], **{lam: [] for lam in LAMS}}
    oof = (
        {lam: np.zeros((len(y), K)) for lam in LAMS} if offset else {}
    )  # each example's score from the folds that did not see it
    seen = np.zeros(len(y), bool)
    for k in range(5):
        tr, te = folds != k, folds == k
        if not te.any():
            continue
        feat = _features(H[tr])
        Ztr, Zte = feat(H[tr]), feat(H[te])
        off_tr = base[tr] if offset else np.zeros((tr.sum(), K))
        off_te = base[te] if offset else np.zeros((te.sum(), K))
        cv["none"].append(ll(base[te], y[te]) * te.sum())
        seen |= te
        w = None
        for lam in LAMS:
            w = _fit_linear(Ztr, off_tr, y[tr], K, lam, w)
            m = Ztr.shape[1]
            s_te = off_te + Zte @ w[: K * m].reshape(K, m).T + w[K * m :]
            cv[lam].append(ll(s_te, y[te]) * te.sum())
            if offset:
                oof[lam][te] = s_te
    scores = {k: float(np.sum(v) / len(y)) for k, v in cv.items()}
    best = min(scores, key=lambda k: scores[k])
    if best == "none":
        return (lambda H2, base2: base2), {"lambda": None, "cv": scores}
    feat = _features(H)
    Z = feat(H)
    w = None
    for lam in LAMS[: LAMS.index(best) + 1]:
        w = _fit_linear(Z, base if offset else np.zeros((len(y), K)), y, K, lam, w)
    m = Z.shape[1]
    W, b = w[: K * m].reshape(K, m), w[K * m :]
    return (lambda H2, base2: (base2 if offset else 0) + feat(H2) @ W.T + b), {
        "lambda": best,
        "cv": scores,
        "params": (feat.params, W, b),
        "oof": oof[best][seen] if offset else None,  # the chosen penalty's out-of-fold scores
        "oof_y": y[seen],
    }


def short_options(counts, options, floor=None):
    """The options with fewer than `floor` (MIN_PER_OPTION) labelled examples, in option order, with their counts."""
    floor = MIN_PER_OPTION if floor is None else floor
    return [(str(options[i]), int(c)) for i, c in enumerate(counts) if c < floor]


def floor_reason(counts, options, floor=None):
    """Why the head was declined for short options, naming them, so the labels that would change the model show."""
    floor = MIN_PER_OPTION if floor is None else floor
    short = short_options(counts, options, floor)
    listed = ", ".join(f"{name} ({n})" for name, n in short[:SHORT_LISTED])
    more = f" and {len(short) - SHORT_LISTED} more" if len(short) > SHORT_LISTED else ""
    return (
        f"every option needs at least {floor} labelled examples; {len(short)} of {len(counts)} have fewer: "
        f"{listed}{more} (option, examples)"
    )


def options_digest(options):
    """sha256 of the option list in order: the head is valid only for exactly this list."""
    return hashlib.sha256(json.dumps(list(options), ensure_ascii=False).encode()).hexdigest()


def fit_intent_head(logps, H, labels, options, fingerprint="", task_id="", seed=0):
    """The stored head for one task. `logps`: per example, the K letter log-scores in option order (the live
    readout's K-way log-softmax); `H`: per example, the final hidden state at the readout position; `labels`: the
    gold option index; `options`: the task's option list, in the order it is served. Returns a record whose
    "applied" says whether serving uses the head; A (d x K) and c (K) are float32 arrays when it does."""
    y = np.asarray(labels)
    K = len(options)
    logps, H = np.asarray(logps, dtype=np.float64), np.asarray(H, dtype=np.float64)
    counts = np.bincount(y, minlength=K)
    rec = {
        "format": FORMAT,
        "task_id": task_id,
        "fingerprint": fingerprint,
        "K": K,
        "options": list(options),
        "options_sha256": options_digest(options),
        "n_examples": int(len(y)),
        "min_per_option": int(counts.min()),
        "applied": False,
    }
    if counts.min() < MIN_PER_OPTION:
        return {
            **rec,
            "reason": floor_reason(counts, options),
            "short_options": [n for n, _ in short_options(counts, options)],
        }
    _, meta = fit_linear(H, logps, y, K, offset=True, seed=seed)
    rec["cv_logloss"] = {str(k): round(v, 5) for k, v in meta["cv"].items()}
    if meta["lambda"] is None:
        return {**rec, "reason": "cross-validation on the labelled examples prefers the plain readout"}
    (mu, sd, V), W, b = meta["params"]
    S, yo = meta["oof"], meta["oof_y"]
    T = fit_temperature(S, yo)
    rec["temperature"] = round(T, 6)
    rec["temperature_cv"] = {  # the head on its out-of-fold examples, as served (with T) and without T
        "n": int(len(yo)),
        "log_loss": {"without": round(_log_loss(S, yo), 5), "with": round(_log_loss(S / T, yo), 5)},
        "ece": {"without": round(_ece(S, yo), 5), "with": round(_ece(S / T, yo), 5)},
    }
    VW = V @ W.T  # d x K
    A = VW / sd[:, None]
    c = b - (mu / sd) @ VW
    return {
        **rec,
        "lambda": meta["lambda"],
        "applied": True,
        "reason": "cross-validated gain",
        "A": A.astype(np.float32),
        "c": c.astype(np.float32),
    }


def apply_intent_head(logp, h, rec, options):
    """Serving: probabilities over the options for one question, from the K letter log-scores (option order) and h.
    Falls back to the plain readout when the head is off or was fitted on a different option list. The head's logits
    are divided by the record's temperature; a record without one serves as before (T = 1)."""
    logp = np.asarray(logp, dtype=np.float64)
    s = logp
    if rec and rec.get("applied") and rec["options_sha256"] == options_digest(options) and len(logp) == rec["K"]:
        s = logp + np.asarray(h, dtype=np.float64) @ rec["A"].astype(np.float64) + rec["c"].astype(np.float64)
        s = s / float(rec.get("temperature") or 1.0)
    s = s - s.max()
    return np.exp(s) / np.exp(s).sum()
