# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Per-task calibration from labelled examples.

A customer supplies about 20 labelled examples of a task (the same question, asked of different states). From
the letter readout's label scores on those examples we fit one bias per answer slot and store it with the task.
At serving time the bias is subtracted from the label scores before the softmax: one K-vector, no extra
forward.

The bias works on top of the server's temperature for the question, not instead of it: it is fitted on the label
log-scores divided by that temperature (what the question is served at without a task), the acceptance rule compares
it against that tempered readout, and serving is softmax(log p / T - bias). The record stores T; a record fitted at
another temperature (every record before 0.52.0 was fitted at 1) is never applied with it (`is_current`).

What makes it safe to switch on for any task is the acceptance rule. The bias is fitted with a fixed L2 penalty
(LAMBDA) and kept only if 5-fold cross-validation on the customer's own labelled examples shows a clear gain:
cross-validated log loss lower by at least MIN_GAIN nats per item, a one-sided 95% test on the per-item
differences, and no lower cross-validated accuracy. Otherwise the task stores a zero bias ("off") and serves
exactly as before. The decision uses only the customer's examples, never evaluation data.
The rule was developed on 16 internal tasks and frozen before it was validated on others.

A stored prior is valid only for the model and adapter it was fitted under; `fingerprint` names them, and the
serving side must refuse a prior whose fingerprint differs from the live model's.

From `tb_min_examples(K)` labelled examples (counted in the registration request, before a certificate's held-out
split: 100 for a yes/no question, 50 for three or more options) the calibration is instead a temperature and a
per-option bias fitted together, by maximum a posteriori with no keep-or-drop gate (`fit_tb`): served
p = softmax(log p / T - b), with the priors log T ~ N(log T_q, TB_TAU^2) around the question's own temperature T_q
and b_k ~ N(0, TB_SIGMA_B^2). A task that serves a head is unchanged.

    prior = fit_task_prior(logps, labels, temperature=T)   # logps: per example, the K label log-scores in slot order
    p = apply_task_prior(logp, prior, temperature=T)       # at serving, at the same T
    T, b = fit_tb(probs, labels, T_q)                      # from 50 examples: the plain readout's probabilities
    p = apply_tb(probs, T, b)
"""

from __future__ import annotations

import numpy as np

from decisio.names import record_format
from decisio.readout import debias

# from this many labelled examples in the request: the temperature and bias fitted together (`fit_tb`)
TB_MIN_EXAMPLES = 50
TB_METHOD = "map_temperature_bias"
TB_TAU = 0.1  # the prior's spread of log T around the question's own temperature
TB_SIGMA_B = 0.8496  # the prior's spread of each option's bias (from the per-set fits of the study that chose it)
TB_MAXITER = 500
TB_EPS = 1e-12  # the probability floor of the log-scores the fit and the serving read
TB_MIN_EXAMPLES_YES_NO = 100  # a two-option (yes/no) task: from this many labelled examples


def tb_min_examples(K: int) -> int:
    """From how many labelled examples in the request a task of K options is calibrated by the temperature-and-bias
    fit (below it, the gated bias): 100 for two options (a yes/no question), 50 for three or more."""
    return TB_MIN_EXAMPLES_YES_NO if int(K) == 2 else TB_MIN_EXAMPLES


MIN_EXAMPLES = 10  # below this the fit is noise (5 per task gave +0.5 points, 95% CI [-2.5, +2.9])
RECOMMENDED = 20  # where the measured effect was full
MIN_GAIN = 0.005  # nats per item of cross-validated log loss the bias must save to be switched on
T_ACCEPT = 1.645  # and the saving must clear a one-sided 95% test on the per-item differences
LAMBDA = 0.1  # L2 penalty on the bias, fixed: with ~20 examples a cross-validated choice often picked no
# penalty and overfit (in development, 12 of 16 tasks passed with it, 16 of 16 at 0.1)
FORMAT = record_format("task-prior")


def tempered(logp, temperature: float = 1.0):
    """The label log-scores at a temperature: softmax of them is the question's distribution served at it."""
    return np.asarray(logp, dtype=np.float64) / float(temperature)


def fitted_temperature(prior) -> float:
    """The temperature a prior was fitted at (a record without one was fitted at 1)."""
    return float((prior or {}).get("temperature", 1.0))


def is_current(prior, temperature: float) -> bool:
    """Whether a prior may be applied at this serving temperature: it was fitted at the same one."""
    return abs(fitted_temperature(prior) - float(temperature)) <= 1e-9


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
        L[i, : len(lp)] = lp
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


def fit_task_prior(logps, labels, fingerprint="", task_id="", K=None, temperature: float = 1.0):
    """The stored prior for one task, from its labelled examples.

    `logps`: per example, the label log-scores in slot order (the readout's K-way log-softmax). `labels`: the
    gold slot per example. K defaults to the widest example. `temperature`: the one the question is served at without
    a task; the bias is fitted, and accepted or declined, on the log-scores at it. Returns a JSON-serialisable dict;
    "applied" says whether serving should use the bias (if False the bias is all zeros and serving is unchanged).
    """
    logps = [tempered(lp, temperature) for lp in logps]
    y = np.asarray(labels)
    n = len(y)
    K = K or max(len(lp) for lp in logps)
    rec = {
        "format": FORMAT,
        "task_id": task_id,
        "fingerprint": fingerprint,
        "K": int(K),
        "n_examples": int(n),
        "bias": [0.0] * int(K),
        "applied": False,
        "temperature": float(temperature),
    }
    if n < MIN_EXAMPLES:
        return {**rec, "reason": f"{n} labelled examples, at least {MIN_EXAMPLES} needed"}
    if len(set(y.tolist())) < 2:
        return {**rec, "reason": "all labelled examples have the same answer; the prior cannot be told from the task"}
    folds = min(5, n)
    ll0, ll1, ok0, ok1 = _cv(logps, y, K, folds)
    d = ll1 - ll0  # per-item change in log loss; negative is better
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
    if se > 0:
        t = d.mean() / se
    elif d.mean() == 0:
        # every fold's change is exactly zero: no evidence either way (a readout certain on every example, so the
        # fitted prior moves nothing; the infinite t this used to give broke the JSON response of POST /v1/tasks)
        t = 0.0
    else:
        t = -np.inf if d.mean() < 0 else np.inf
    rec.update(
        cv_logloss_plain=round(float(ll0.mean()), 5),
        cv_logloss_fitted=round(float(ll1.mean()), 5),
        cv_acc_plain=round(float(ok0.mean()), 4),
        cv_acc_fitted=round(float(ok1.mean()), 4),
        cv_t=round(float(t), 3) if np.isfinite(t) else None,  # JSON has no infinity
    )
    # switched on only when the gain is both material and unlikely to be noise on these few examples
    if d.mean() > -MIN_GAIN or t > -T_ACCEPT or ok1.mean() < ok0.mean():
        return {**rec, "reason": "cross-validation on the labelled examples does not show a clear gain"}
    b = _fit(logps, y, K, np.arange(n))
    return {
        **rec,
        "bias": [round(float(x), 6) for x in b],
        "lambda": LAMBDA,
        "applied": True,
        "reason": "cross-validated gain",
    }


def apply_task_prior(logp, prior, temperature: float = 1.0):
    """Serving: corrected probabilities for one question's K label log-scores (slot order) at the question's
    temperature, softmax(log p / T - bias). A prior fitted at another temperature is not applied: the question is
    served as without a task."""
    if prior and prior.get("method") == TB_METHOD and prior.get("applied") and is_current(prior, temperature):
        if len(logp) > prior["K"]:
            return debias.apply_bias(tempered(logp, temperature), np.zeros(len(logp)))
        return apply_tb_logscores(np.asarray(logp, dtype=np.float64)[None, :], prior["T"], prior["bias"][: len(logp)])[
            0
        ]
    logp = tempered(logp, temperature)
    if not prior or not prior.get("applied") or not is_current(prior, temperature):
        return debias.apply_bias(logp, np.zeros(len(logp)))
    if len(logp) > prior["K"]:
        # more options than the prior was fitted for: its slots do not cover the question, so it is not applied
        return debias.apply_bias(logp, np.zeros(len(logp)))
    return debias.apply_bias(logp, np.asarray(prior["bias"][: len(logp)]))


def tb_logscores(LP):
    """The log-scores the temperature-and-bias fit reads: log p floored at TB_EPS and capped at 0 (from log-scores of
    any floor: the same as log(clip(p, TB_EPS, 1)), bit for bit)."""
    return np.clip(np.asarray(LP, dtype=np.float64), np.log(TB_EPS), 0.0)


def fit_tb_logscores(LP, y, T_q, tau=TB_TAU, sigma_b=TB_SIGMA_B, maxiter=TB_MAXITER):
    """(T, b): the MAP of softmax(LP / T - b) on labels y (LP: n x K log-scores), with log T ~ N(log T_q, tau^2) and
    b_k ~ N(0, sigma_b^2); L-BFGS-B with analytic gradients from log T = log T_q, b = 0."""
    import math

    from scipy.optimize import minimize

    LP = tb_logscores(LP)
    y = np.asarray(y)
    n, K = LP.shape
    mu = math.log(float(T_q))
    tw = 1.0 / (2 * tau * tau)
    bw = 1.0 / (2 * sigma_b * sigma_b)

    def f(w):
        lt, b = w[0], w[1:]
        T = math.exp(lt)
        Z = LP / T - b
        Z = Z - Z.max(1, keepdims=True)
        lse = np.log(np.exp(Z).sum(1))
        nll = (lse - Z[np.arange(n), y]).sum()
        Pm = np.exp(Z - lse[:, None])
        Pm[np.arange(n), y] -= 1.0
        g_b = -Pm.sum(0)
        g_lt = (Pm * (-LP / T)).sum()
        loss = nll + tw * (lt - mu) ** 2 + bw * (b**2).sum()
        g = np.concatenate([[g_lt + 2 * tw * (lt - mu)], g_b + 2 * bw * b])
        return loss, g

    w0 = np.concatenate([[mu], np.zeros(K)])
    r = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return math.exp(r.x[0]), r.x[1:]


def apply_tb_logscores(LP, T, b):
    """softmax(LP / T - b) per row (LP: n x K log-scores of any floor, read as `tb_logscores`)."""
    Z = tb_logscores(LP) / float(T) - np.asarray(b, dtype=np.float64)
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit_tb(probs, labels, T_q):
    """(T, b) from the plain readout's probabilities of the fit examples (n x K), their labels and the question's
    temperature T_q: the fit a task of at least TB_MIN_EXAMPLES labelled examples stores."""
    return fit_tb_logscores(np.log(np.clip(np.asarray(probs, dtype=np.float64), TB_EPS, 1)), labels, T_q)


def apply_tb(probs, T, b):
    """The served probabilities (n x K) of plain-readout probabilities under a fit of `fit_tb`."""
    return apply_tb_logscores(np.log(np.clip(np.asarray(probs, dtype=np.float64), TB_EPS, 1)), T, b)


def fit_tb_prior(logps, labels, fingerprint="", task_id="", K=None, temperature: float = 1.0, n_request=None):
    """The stored record of the temperature-and-bias fit (`fit_tb_logscores` on the examples' K label log-scores at
    the question's temperature `temperature`), for a task of at least TB_MIN_EXAMPLES labelled examples. Always
    applied: there is no keep-or-drop gate."""
    LP = np.asarray(logps, dtype=np.float64)
    K = int(K if K is not None else LP.shape[1])
    T, b = fit_tb_logscores(LP, labels, temperature)
    return {
        "format": FORMAT,
        "method": TB_METHOD,
        "fingerprint": fingerprint,
        "task_id": task_id,
        "K": K,
        "n": int(len(labels)),
        **({"n_request": int(n_request)} if n_request is not None else {}),
        "temperature": float(temperature),
        "T": float(T),
        "bias": [float(x) for x in b],
        "tau": TB_TAU,
        "sigma_b": TB_SIGMA_B,
        "applied": True,
        "reason": (
            f"{n_request if n_request is not None else len(labels)} labelled examples: the temperature and a bias per "
            f"option fitted together (from {TB_MIN_EXAMPLES} examples, with no keep-or-drop gate)"
        ),
    }
