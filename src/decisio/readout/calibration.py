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

From `min_examples_for_fit(K)` (`TB_MIN_EXAMPLES`, 50) labelled examples (counted in the registration request, before a
certificate's held-out split), for the questions in their base's `Family.large_calibration_types` (`in_large_scope`),
the calibration is instead the
evidence-averaged temperature and bias (`fit_c_prior`, `CURRENT_METHOD`, candidate C): the temperature and a per-option
bias fitted together by maximum a posteriori at each prior spread in `C_GRID` (`fit_tb_logscores` is one such fit; the
priors are log T ~ N(log T_q, TB_TAU^2) around the question's own temperature T_q and b_k ~ N(0, s^2)), the fits
weighted by their Laplace evidence and mixed: served p = sum_j w_j softmax(log p / T_j - b_j). There is no keep-or-drop
gate: it is always applied. A task that serves a head is unchanged. `fit_tb` and `fit_tb_prior` (one spread, TB_SIGMA_B)
are the earlier form of this fit and stay for records stored under `TB_METHOD`.

    prior = fit_task_prior(logps, labels, temperature=T)   # below 50 examples, the gated bias; logps: per example,
                                                           # the K label log-scores in slot order
    p = apply_task_prior(logp, prior, temperature=T)       # at serving, at the same T
    prior = fit_c_prior(logps, labels, temperature=T)      # from 50 examples: the evidence-averaged fit, always applied
    p = apply_c(probs, prior)                              # probs: the plain readout's probabilities (n x K)
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
# from TB_MIN_EXAMPLES: the temperature-and-bias fit at each of these bias-prior spreads (0: the temperature alone),
# each weighted by its Laplace evidence, and their mixture served (`fit_c`)
C_METHOD = "evidence_averaged_temperature_bias"
C_GRID = (0.0, 0.2, 0.4, 0.8496)
SB_FLOOR = 1e-4  # the smallest prior spread of a bias (the temperature-only fit's)
CURRENT_METHOD = C_METHOD  # the method this release fits from min_examples_for_fit(K) examples


def in_large_scope(scope, qtype: str, k: int) -> bool:
    """Whether a question of type `qtype` with `k` options (levels, for a score question) is in a base's large
    calibration scope (`Family.large_calibration_types`): an entry is a question type, every option count of it, or a
    `(type, lowest, highest)` range of option counts (`highest` None: no upper end), as the score cells were measured
    one by one."""
    for entry in scope or ():
        if isinstance(entry, str):
            if entry == qtype:
                return True
        else:
            t, lo, hi = entry
            if t == qtype and k >= lo and (hi is None or k <= hi):
                return True
    return False


def min_examples_for_fit(K: int) -> int:
    """From how many labelled examples in the request a task of K options is calibrated by `fit_large_prior` (below
    it, the gated bias)."""
    return TB_MIN_EXAMPLES


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
    if prior and prior.get("method") == C_METHOD and prior.get("applied") and is_current(prior, temperature):
        if len(logp) > prior["K"]:
            return debias.apply_bias(tempered(logp, temperature), np.zeros(len(logp)))
        LP = np.asarray(logp, dtype=np.float64)[None, :]
        pts = [(q["w"], q["T"], np.asarray(q["bias"][: len(logp)], dtype=np.float64)) for q in prior["points"]]
        return sum(w * _apply_rows(tb_logscores(LP), T, b) for w, T, b in pts if w > 0)[0]
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


# ---- the evidence-averaged fit (C_METHOD) --------------------------------------------------------------------------


def _apply_rows(LP, T, b):
    """softmax(LP / T - b) per row."""
    Z = LP / T - b
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit_tbk_logscores(LP, y, mu, tau, s):
    """(T, b): the MAP of softmax(LP / T - b), log T ~ N(mu, tau^2), b_k ~ N(0, s_k^2) (s floored at SB_FLOOR);
    L-BFGS-B with analytic gradients from log T = mu, b = 0, at most TB_MAXITER iterations."""
    import math

    from scipy.optimize import minimize

    n, K = LP.shape
    tw = 1.0 / (2 * tau * tau)
    bw = 1.0 / (2 * np.maximum(np.asarray(s, np.float64), SB_FLOOR) ** 2)

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
        loss = nll + tw * (lt - mu) ** 2 + (bw * b * b).sum()
        g = np.concatenate([[g_lt + 2 * tw * (lt - mu)], g_b + 2 * bw * b])
        return loss, g

    w0 = np.concatenate([[mu], np.zeros(K)])
    r = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": TB_MAXITER})
    return math.exp(r.x[0]), r.x[1:]


def fit_t_logscores(LP, y, mu, tau):
    """The temperature alone (b = 0): the MAP of log T under log T ~ N(mu, tau^2), L-BFGS-B from log T = mu."""
    import math

    from scipy.optimize import minimize

    n = len(y)

    def f(w):
        T = math.exp(w[0])
        Z = LP / T
        Z = Z - Z.max(1, keepdims=True)
        lse = np.log(np.exp(Z).sum(1))
        Pm = np.exp(Z - lse[:, None])
        Pm[np.arange(n), y] -= 1.0
        nll = float((lse - Z[np.arange(n), y]).sum())
        return nll + (w[0] - mu) ** 2 / (2 * tau**2), np.array([(Pm * (-LP / T)).sum() + (w[0] - mu) / tau**2])

    r = minimize(f, np.array([mu]), jac=True, method="L-BFGS-B", options={"maxiter": TB_MAXITER})
    return math.exp(r.x[0])


def hessian(LP, y, T, b, tau, sigma):
    """The analytic Hessian of the negative log posterior in (log T, b) at (T, b); sigma None: the temperature alone
    (1 x 1). z_i = LP_i / T - b, u_i = -LP_i / T, p_i = softmax(z_i), g_i = p_i - e_y, S_i = diag(p_i) - p_i p_i^T:
    H_tt = sum(u^T S u - g^T u) + 1 / tau^2, H_tb = -sum S u, H_bb = sum S + I / sigma^2."""
    n, K = LP.shape
    U = -LP / T
    P = _apply_rows(LP, T, b)
    G = P.copy()
    G[np.arange(n), y] -= 1.0
    PU = (P * U).sum(1)
    SU = P * U - P * PU[:, None]
    h_tt = float((U * SU).sum() - (G * U).sum()) + 1.0 / tau**2
    if sigma is None:
        return np.array([[h_tt]])
    H = np.empty((K + 1, K + 1))
    H[0, 0] = h_tt
    H[0, 1:] = H[1:, 0] = -SU.sum(0)
    H[1:, 1:] = np.diag(P.sum(0)) - P.T @ P + np.eye(K) / sigma**2
    return H


def log_evidence(LP, y, mu, tau, sigma):
    """(Laplace log evidence, T, b) of the fit at bias-prior spread sigma (0: the temperature alone, b = 0); a fit
    whose Hessian is not positive definite has evidence -inf. Terms common to every sigma are left out."""
    import math

    n, K = LP.shape
    if sigma <= 0:
        T = fit_t_logscores(LP, y, mu, tau)
        b = np.zeros(K)
        nll = float(-np.log(np.clip(_apply_rows(LP, T, b)[np.arange(n), y], 1e-300, None)).sum())
        H = hessian(LP, y, T, b, tau, None)
        ev = -(nll + (math.log(T) - mu) ** 2 / (2 * tau**2)) - 0.5 * math.log(H[0, 0]) if H[0, 0] > 0 else -math.inf
        return ev, T, b
    T, b = fit_tbk_logscores(LP, y, mu, tau, np.full(K, sigma))
    nll = float(-np.log(np.clip(_apply_rows(LP, T, b)[np.arange(n), y], 1e-300, None)).sum())
    sign, logdet = np.linalg.slogdet(hessian(LP, y, T, b, tau, sigma))
    if sign <= 0 or not np.isfinite(logdet):
        return -math.inf, T, b
    U_ = nll + (math.log(T) - mu) ** 2 / (2 * tau**2) + float((b * b).sum()) / (2 * sigma**2) + K * math.log(sigma)
    return -U_ - 0.5 * logdet, T, b


def fit_c_logscores(LP, y, T_q, tau=TB_TAU, grid=C_GRID):
    """[(weight, T, b)] in grid order: the fit at each bias-prior spread, weighted by its Laplace evidence (a uniform
    prior on the grid; the weights sum to 1)."""
    import math

    LP = tb_logscores(LP)
    y = np.asarray(y)
    fits = [log_evidence(LP, y, math.log(float(T_q)), tau, sg) for sg in grid]
    ev = np.array([f[0] for f in fits])
    w = np.exp(ev - ev.max())
    w /= w.sum()
    return [(float(wi), float(f[1]), np.asarray(f[2], np.float64)) for wi, f in zip(w, fits)]


def fit_c(p_fit, y, T_q):
    """The evidence-averaged fit from the plain readout's probabilities of the fit examples (n x K), their labels and
    the question's temperature: [(weight, T, b)] in C_GRID order."""
    return fit_c_logscores(np.log(np.clip(np.asarray(p_fit, dtype=np.float64), TB_EPS, 1)), y, T_q)


def apply_c(p, record):
    """The served probabilities (n x K) of plain-readout probabilities under `fit_c`'s record (or a stored record of
    C_METHOD): sum_j w_j softmax(log p / T_j - b_j)."""
    LP = np.log(np.clip(np.asarray(p, dtype=np.float64), TB_EPS, 1))
    if isinstance(record, dict):
        record = [(q["w"], q["T"], np.asarray(q["bias"], dtype=np.float64)) for q in record["points"]]
    return sum(w * _apply_rows(LP, T, b) for w, T, b in record if w > 0)


def fit_c_prior(logps, labels, fingerprint="", task_id="", K=None, temperature: float = 1.0, n_request=None):
    """The stored record of the evidence-averaged fit, for a task of at least min_examples_for_fit(K) labelled
    examples: per point of C_GRID its spread, weight, T and bias. Always applied: there is no keep-or-drop gate."""
    LP = np.asarray(logps, dtype=np.float64)
    K = int(K if K is not None else LP.shape[1])
    points = fit_c_logscores(LP, labels, temperature)
    n = int(len(labels)) if n_request is None else int(n_request)
    return {
        "format": FORMAT,
        "method": C_METHOD,
        "fingerprint": fingerprint,
        "task_id": task_id,
        "K": K,
        "n": int(len(labels)),
        **({"n_request": int(n_request)} if n_request is not None else {}),
        "temperature": float(temperature),
        "tau": TB_TAU,
        "grid": list(C_GRID),
        "points": [
            {"sigma": float(sg), "w": w, "T": T, "bias": [float(x) for x in b]} for sg, (w, T, b) in zip(C_GRID, points)
        ],
        "applied": True,
        "reason": (
            f"{n} labelled examples: the temperature and a bias per option fitted at {len(C_GRID)} prior spreads and "
            f"averaged by their evidence (from {TB_MIN_EXAMPLES} examples, with no keep-or-drop gate)"
        ),
    }


fit_large_prior = fit_c_prior  # what registration fits from min_examples_for_fit(K) examples
