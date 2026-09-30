# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Readout debiasing for the letter readout, without training.

The letter readout reads one position and renormalises over the K label tokens. Whatever the model
prefers about a slot regardless of content (" A" over " J", " yes" over " no") is added to every
answer. Three corrections, all acting on the K label log-scores of one forward:

  * **fitted prior** (`fit_bias`): a per-slot bias b, subtracted before the softmax, fitted by maximum
    likelihood on held-out probe items (never on evaluated items), with an L2 penalty chosen by
    5-fold cross-validation on the probe. For items whose options change per item the slot is the
    only thing a bias can attach to; for fixed label sets (Banking77, yes/no) slot and label coincide,
    so this is the per-label prior. Uses the probe's gold labels.
  * **permutation prior** (`pride_prior`, PriDe, Zheng et al., ICLR 2024): the same bias estimated
    without labels, by scoring probe items under cyclic shifts of their options: averaged over
    shifts, every option visits every slot, so what remains of the slot scores is the slot's prior.
  * **content-free PMI** (Zhao et al., ICML 2021): the slot scores of a prompt whose question and
    options are all "N/A", subtracted. Computed at serving time, one forward per (instruction, K),
    cached; the arithmetic is `apply_bias` with b = cf.

And one that removes position from the answer instead of correcting it:

  * **cyclic averaging**: the item is scored under k cyclic shifts of its option order and the
    probabilities, mapped back to the original options, are averaged. Costs k forwards.

Everything here is CPU arithmetic on saved scores; the server applies the same functions online.
"""

from __future__ import annotations

import numpy as np

FLOOR = 1e-300


# ---- item permutation ---------------------------------------------------------------------------


def cyclic_order(K, j, k):
    """order[position] = original option index, for shift j of k: offset j*K//k (j = 0 is identity)."""
    off = j * K // k
    return [(i + off) % K for i in range(K)]


def unpermute(p_pos, order):
    """Position-space values back to original option order."""
    back = np.empty(len(order))
    back[list(order)] = p_pos
    return back


# ---- applying a bias --------------------------------------------------------------------------


def slot_bias(bias_by_k, K):
    """The bias for a K-option item: the fitted vector for K if there is one, else the first K
    entries of the widest fitted vector (a slot's bias does not depend on how many slots follow)."""
    if bias_by_k is None:
        return np.zeros(K)
    if str(K) in bias_by_k:
        return np.asarray(bias_by_k[str(K)], dtype=np.float64)
    widest = max(bias_by_k, key=int)
    if int(widest) < K:
        raise ValueError(f"no prior covers {K} options (widest fitted: {widest})")
    return np.asarray(bias_by_k[widest][:K], dtype=np.float64)


def apply_bias(logp, b):
    """softmax(logp - b): position-space log-scores (any constant offset) to corrected probabilities."""
    z = np.asarray(logp, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def log_probs(p):
    return np.log(np.clip(np.asarray(p, dtype=np.float64), FLOOR, None))


# ---- fitting ------------------------------------------------------------------------------------


def _nll_grad(b, L, y, lam):
    """Mean NLL of softmax(L - b) at y, plus lam*|b|^2, and its gradient. L: [n, K] log-scores."""
    Z = L - b
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    P /= P.sum(1, keepdims=True)
    n = len(y)
    nll = -np.log(np.clip(P[np.arange(n), y], FLOOR, None)).mean() + lam * (b @ b)
    onehot = np.zeros_like(P)
    onehot[np.arange(n), y] = 1.0
    # z = L - b, d(-log softmax(z)_y)/dz = P - onehot, dz/db = -1
    grad = (onehot - P).mean(0) + 2 * lam * b
    return nll, grad


def _fit(L, y, lam):
    from scipy.optimize import minimize

    K = L.shape[1]
    res = minimize(_nll_grad, np.zeros(K), args=(L, y, lam), jac=True, method="L-BFGS-B")
    b = res.x - res.x.mean()  # softmax is shift-invariant; centre for readability
    return b


LAMBDAS = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def fit_bias(logps, y, K, groups=None, lambdas=LAMBDAS, folds=5, seed=0):
    """Per-slot bias for K-option items, by penalised maximum likelihood; the penalty by K-fold CV.

    `logps` are position-space log-scores of probe items with at most K options; an item with k < K
    options is fitted on its first k slots (the missing slots get -inf scores, so they take no mass).
    `y` is the gold slot. `groups` keeps rows of one item (the same item under several shifts) in the
    same fold, so cross-validation never scores an item it was fitted on.
    Returns (b, chosen lambda, CV NLL per lambda).
    """
    L = np.full((len(logps), K), -1e9)
    for i, lp in enumerate(logps):
        L[i, : len(lp)] = lp
    y = np.asarray(y)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    uniq = np.random.default_rng(seed).permutation(np.unique(groups))
    parts = [np.flatnonzero(np.isin(groups, g)) for g in np.array_split(uniq, folds)]
    cv = {}
    for lam in lambdas:
        tot = 0.0
        for f in range(folds):
            te = parts[f]
            tr = np.concatenate([parts[g] for g in range(folds) if g != f])
            b = _fit(L[tr], y[tr], lam)
            tot += _nll_grad(b, L[te], y[te], 0.0)[0] * len(te)
        cv[lam] = tot / len(y)
    lam = min(cv, key=lambda k: (round(cv[k], 6), -k))  # ties go to the stronger penalty
    return _fit(L, y, lam), lam, cv


def pride_prior(shift_logps):
    """PriDe's prior from permuted scoring, no labels needed.

    `shift_logps` is a list over items of [k_shifts, K] position-space log-probabilities, each item
    scored under k cyclic shifts. Per item, the average over shifts of a slot's log-probability is
    content-free up to a constant (every option has visited every slot, for k = K; approximately for
    k < K). The prior is that, softmax-normalised per item and averaged over items; the bias is its log.
    """
    K = shift_logps[0].shape[1]
    acc = np.zeros(K)
    for S in shift_logps:
        acc += apply_bias(np.asarray(S).mean(0), np.zeros(K))
    prior = acc / len(shift_logps)
    b = np.log(prior)
    return b - b.mean()


def cyclic_average(probs_by_shift):
    """Mean of the shifts' probabilities, each already in original option order."""
    return np.mean(np.asarray(probs_by_shift, dtype=np.float64), axis=0)
