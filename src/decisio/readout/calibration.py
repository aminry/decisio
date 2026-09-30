# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Per-task calibration from labelled examples.

A customer supplies about 20 labelled examples of a task (the same question, asked of different states). From
the letter readout's label scores on those examples we fit one bias per answer slot and store it with the task.
At serving time the bias is subtracted from the label scores before the softmax: one K-vector, no extra
forward.

What makes it safe to switch on for any task is the acceptance rule. The bias is fitted with a fixed L2 penalty
(LAMBDA) and kept only if 5-fold cross-validation on the customer's own labelled examples shows a clear gain:
cross-validated log loss lower by at least MIN_GAIN nats per item, a one-sided 95% test on the per-item
differences, and no lower cross-validated accuracy. Otherwise the task stores a zero bias ("off") and serves
exactly as before. The decision uses only the customer's examples, never evaluation data.
The rule was developed on 16 internal tasks and frozen before it was validated on others.

A stored prior is valid only for the model and adapter it was fitted under; `fingerprint` names them, and the
serving side must refuse a prior whose fingerprint differs from the live model's.

    prior = fit_task_prior(logps, labels)            # logps: per example, the K label log-scores in slot order
    p = apply_task_prior(logp, prior)                # at serving
"""
from __future__ import annotations

import numpy as np

from decisio.names import record_format
from decisio.readout import debias

MIN_EXAMPLES = 10          # below this the fit is noise (5 per task gave +0.5 points, 95% CI [-2.5, +2.9])
RECOMMENDED = 20           # where the measured effect was full
MIN_GAIN = 0.005           # nats per item of cross-validated log loss the bias must save to be switched on
T_ACCEPT = 1.645           # and the saving must clear a one-sided 95% test on the per-item differences
LAMBDA = 0.1               # L2 penalty on the bias, fixed: with ~20 examples a cross-validated choice often picked no
                           # penalty and overfit (in development, 12 of 16 tasks passed with it, 16 of 16 at 0.1)
FORMAT = record_format("task-prior")


def _fit(logps, y, K, idx):
    sub = [logps[i] for i in idx]
    if LAMBDA is None:
        return debias.fit_bias(sub, y[idx], K, folds=min(5, len(idx)))[0]
    return debias.fit_bias(sub, y[idx], K, lambdas=(LAMBDA,), folds=2)[0]


def _cv(logps, y, K, folds):
    """Cross-validated per-item (log loss, correct) of the plain readout and of the fitted bias, same folds."""
    n = len(y)
    idx = np.random.default_rng(0).permutation(n)
    parts = np.array_split(idx, folds)
    L = np.full((n, K), -1e9)
    for i, lp in enumerate(logps):
        L[i, :len(lp)] = lp
    ll0, ll1, ok0, ok1 = (np.zeros(n) for _ in range(4))
    for f in range(folds):
        te, tr = parts[f], np.concatenate([parts[g] for g in range(folds) if g != f])
        b = _fit(logps, y, K, tr)
        for i in te:
            p0 = debias.apply_bias(L[i], np.zeros(K))
            p1 = debias.apply_bias(L[i], b)
            ll0[i], ll1[i] = -np.log(max(p0[y[i]], 1e-12)), -np.log(max(p1[y[i]], 1e-12))
            ok0[i], ok1[i] = float(np.argmax(p0) == y[i]), float(np.argmax(p1) == y[i])
    return ll0, ll1, ok0, ok1


def fit_task_prior(logps, labels, fingerprint="", task_id="", K=None):
    """The stored prior for one task, from its labelled examples.

    `logps`: per example, the label log-scores in slot order (the readout's K-way log-softmax). `labels`: the
    gold slot per example. K defaults to the widest example. Returns a JSON-serialisable dict; "applied" says
    whether serving should use the bias (if False the bias is all zeros and serving is unchanged).
    """
    y = np.asarray(labels)
    n = len(y)
    K = K or max(len(lp) for lp in logps)
    rec = {"format": FORMAT, "task_id": task_id, "fingerprint": fingerprint, "K": int(K), "n_examples": int(n),
           "bias": [0.0] * int(K), "applied": False}
    if n < MIN_EXAMPLES:
        return {**rec, "reason": f"{n} labelled examples, at least {MIN_EXAMPLES} needed"}
    if len(set(y.tolist())) < 2:
        return {**rec, "reason": "all labelled examples have the same answer; the prior cannot be told from the task"}
    folds = min(5, n)
    ll0, ll1, ok0, ok1 = _cv(logps, y, K, folds)
    d = ll1 - ll0                                      # per-item change in log loss; negative is better
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
    t = d.mean() / se if se > 0 else (-np.inf if d.mean() < 0 else np.inf)
    rec.update(cv_logloss_plain=round(float(ll0.mean()), 5), cv_logloss_fitted=round(float(ll1.mean()), 5),
               cv_acc_plain=round(float(ok0.mean()), 4), cv_acc_fitted=round(float(ok1.mean()), 4),
               cv_t=round(float(t), 3))
    # switched on only when the gain is both material and unlikely to be noise on these few examples
    if d.mean() > -MIN_GAIN or t > -T_ACCEPT or ok1.mean() < ok0.mean():
        return {**rec, "reason": "cross-validation on the labelled examples does not show a clear gain"}
    b = _fit(logps, y, K, np.arange(n))
    return {**rec, "bias": [round(float(x), 6) for x in b], "lambda": LAMBDA, "applied": True,
            "reason": "cross-validated gain"}


def apply_task_prior(logp, prior):
    """Serving: corrected probabilities for one question's K label log-scores (slot order)."""
    logp = np.asarray(logp, dtype=np.float64)
    if not prior or not prior.get("applied"):
        return debias.apply_bias(logp, np.zeros(len(logp)))
    if len(logp) > prior["K"]:
        # more options than the prior was fitted for: its slots do not cover the question, so it is not applied
        return debias.apply_bias(logp, np.zeros(len(logp)))
    return debias.apply_bias(logp, np.asarray(prior["bias"][:len(logp)]))
