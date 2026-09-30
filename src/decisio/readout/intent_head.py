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
MIN_PER_OPTION = 5         # 5 and 10 per intent passed the acceptance gate on both intent sets; 2 passed on one


def _fit_linear(Z, base, y, K, lam, w0=None):
    """Multinomial logistic regression on features Z (n x m) with offset `base` (n x K log-scores; zeros for a
    classifier on the hidden state alone), L2 penalty lam on W and b. Returns the flat parameter vector."""
    from scipy.optimize import minimize
    n, m = Z.shape

    def f(w):
        W = w[:K * m].reshape(K, m)
        b = w[K * m:]
        s = base + Z @ W.T + b
        s = s - s.max(1, keepdims=True)
        lse = np.log(np.exp(s).sum(1))
        loss = (lse - s[np.arange(n), y]).mean() + lam * (w ** 2).sum()
        P = np.exp(s - lse[:, None])
        P[np.arange(n), y] -= 1
        g = np.concatenate([(P.T @ Z / n).ravel(), P.mean(0)]) + 2 * lam * w
        return loss, g
    x0 = np.zeros(K * m + K) if w0 is None else w0
    return minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 300}).x


LAMS = [100.0, 10.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]     # strongest first: each fit warm-starts the next
MAX_DIMS = 512


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
        return float(np.mean(np.log(np.exp(s - s.max(1, keepdims=True)).sum(1)) + s.max(1)
                             - s[np.arange(len(yy)), yy]))
    cv = {"none": [], **{lam: [] for lam in LAMS}}
    for k in range(5):
        tr, te = folds != k, folds == k
        if not te.any():
            continue
        feat = _features(H[tr])
        Ztr, Zte = feat(H[tr]), feat(H[te])
        off_tr = base[tr] if offset else np.zeros((tr.sum(), K))
        off_te = base[te] if offset else np.zeros((te.sum(), K))
        cv["none"].append(ll(base[te], y[te]) * te.sum())
        w = None
        for lam in LAMS:
            w = _fit_linear(Ztr, off_tr, y[tr], K, lam, w)
            m = Ztr.shape[1]
            cv[lam].append(ll(off_te + Zte @ w[:K * m].reshape(K, m).T + w[K * m:], y[te]) * te.sum())
    scores = {k: float(np.sum(v) / len(y)) for k, v in cv.items()}
    best = min(scores, key=lambda k: scores[k])
    if best == "none":
        return (lambda H2, base2: base2), {"lambda": None, "cv": scores}
    feat = _features(H)
    Z = feat(H)
    w = None
    for lam in LAMS[:LAMS.index(best) + 1]:
        w = _fit_linear(Z, base if offset else np.zeros((len(y), K)), y, K, lam, w)
    m = Z.shape[1]
    W, b = w[:K * m].reshape(K, m), w[K * m:]
    return (lambda H2, base2: (base2 if offset else 0) + feat(H2) @ W.T + b), {"lambda": best, "cv": scores,
                                                                              "params": (feat.params, W, b)}


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
    rec = {"format": FORMAT, "task_id": task_id, "fingerprint": fingerprint, "K": K, "options": list(options),
           "options_sha256": options_digest(options), "n_examples": int(len(y)),
           "min_per_option": int(counts.min()), "applied": False}
    if counts.min() < MIN_PER_OPTION:
        return {**rec, "reason": f"every option needs at least {MIN_PER_OPTION} labelled examples; "
                                 f"{int((counts < MIN_PER_OPTION).sum())} of {K} have fewer"}
    _, meta = fit_linear(H, logps, y, K, offset=True, seed=seed)
    rec["cv_logloss"] = {str(k): round(v, 5) for k, v in meta["cv"].items()}
    if meta["lambda"] is None:
        return {**rec, "reason": "cross-validation on the labelled examples prefers the plain readout"}
    (mu, sd, V), W, b = meta["params"]
    VW = V @ W.T                                     # d x K
    A = VW / sd[:, None]
    c = b - (mu / sd) @ VW
    return {**rec, "lambda": meta["lambda"], "applied": True, "reason": "cross-validated gain",
            "A": A.astype(np.float32), "c": c.astype(np.float32)}


def apply_intent_head(logp, h, rec, options):
    """Serving: probabilities over the options for one question, from the K letter log-scores (option order) and h.
    Falls back to the plain readout when the head is off or was fitted on a different option list."""
    logp = np.asarray(logp, dtype=np.float64)
    s = logp
    if rec and rec.get("applied") and rec["options_sha256"] == options_digest(options) and len(logp) == rec["K"]:
        s = logp + np.asarray(h, dtype=np.float64) @ rec["A"].astype(np.float64) + rec["c"].astype(np.float64)
    s = s - s.max()
    return np.exp(s) / np.exp(s).sum()
