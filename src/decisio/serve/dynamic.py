# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Teaching a task whose options change with every request (`"options": "dynamic"`).

A fixed-option task (decisio.serve.tasks) is one question with one list of options: brand detection from a product
title, entity linking, "which of these tools", "which of these products" are not. Their options are different in every
request, so there is no option list to calibrate, no intent head to fit and no answer mix to watch. What carries over
from request to request is the question's instructions, and the way the right option relates to the text and to what
decisio already thinks of it. A dynamic task is therefore matched on the instructions, and it holds:

  the certificate        on decisio's own top probability, whatever the options (the same Learn-then-Test as any
                         certified task: the examples may carry different option lists). Every dynamic task gets it.
  the re-ranker          one weight vector over a few features of an option computed from the request itself (below),
                         a conditional logit (a softmax over the request's options), kept only when its 5-fold
                         cross-validation on the examples beats decisio's own distribution, like the intent head. The
                         certificate is then made on the re-ranker's top probability.
  the option prior       for each option text seen in the examples, how often it was the answer and how often it was
                         offered. Only option texts the user's own examples contained; no request text is stored.

The features of an option (all but the first two from the request's text and the option's text; `none` is the request's
"none of these" option and has only the first two and its own indicator):

   0 log of decisio's probability for the option (at the task's temperature)     1 it is decisio's top option
   2 share of the option's tokens that occur in the state                        3 Jaccard overlap of the two token sets
   4 where the option (or its first matching token) sits in the state, 0 to 1
   5 none of its tokens occurs in the state
   6 the whole option occurs in the state                                        7 only some of its tokens occur
   8 log(1 + tokens in the option)                                               9 log(1 + characters in the option)
  10 log odds that the option was the answer among the examples (smoothed)      11 it never appeared in the examples
  12 the none option

The prior in a training row is fitted on the other folds of the training examples, so an option's own label never
informs its own row; the features are standardised on the training rows; the L2 strength of the conditional logit is
chosen by 5-fold cross-validation. No encoder is involved: a request costs well under a millisecond.

Standard library and numpy/scipy.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict

import numpy as np

from decisio.names import record_format

FORMAT = record_format("dynamic")
NONE_KEY = "none"  # the key of a request's "none of these" option, unless the task says another
N_STATIC = 8  # features 2 to 9, computed from the texts alone
N_COLS = 13  # the columns of a row (the list above)
NONE_COL = N_COLS - 1
CS = (0.01, 0.1, 1.0, 10.0)  # the L2 strength of the conditional logit, chosen by cross-validation
FOLDS = 5
T_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))
MIN_GAIN = 0.005  # a re-ranker is kept when its cross-validated log loss beats decisio's by this much per example
MIN_EXAMPLES = 20  # fewer examples than this fit nothing: the certificate alone


def norm(text: str) -> str:
    """The comparison form of a text: NFKC, lower case, punctuation as a space, single spaces."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def option_key(text: str) -> str:
    """An option text's identity in the prior table: its comparison form, hashed (the table holds no text)."""
    return hashlib.sha256(norm(text).encode()).hexdigest()[:16]


def static_features(state_text: str, options: list[str]) -> np.ndarray:
    """Features 2 to 9 of every option (rows) from the request's state text and the options' texts, in order."""
    st = norm(state_text)
    stoks = st.split()
    sset = set(stoks)
    out = np.zeros((len(options), N_STATIC))
    for j, c in enumerate(options):
        ct = norm(c).split()
        cn = " ".join(ct)
        inter = len(set(ct) & sset)
        share = inter / max(1, len(set(ct)))
        jac = inter / max(1, len(set(ct) | sset))
        pos = st.find(cn) if cn else -1
        exact = pos >= 0 and bool(cn)
        if not exact:
            cset = set(ct)
            first = next((i for i, w in enumerate(stoks) if w in cset), -1)
            rel = (sum(len(w) + 1 for w in stoks[:first]) / max(1, len(st))) if first >= 0 else 0.0
        else:
            rel = pos / max(1, len(st))
        absent = not exact and inter == 0
        out[j] = [
            share,
            jac,
            rel,
            float(absent),
            float(exact),
            float(not exact and inter > 0),
            np.log1p(len(ct)),
            np.log1p(len(c)),
        ]
    return out


def temper(p, T: float) -> np.ndarray:
    """softmax(log p / T) of a probability vector."""
    z = np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1.0)) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def rows_of(static: np.ndarray, keys: list[str], p, none_index: int | None, T: float, prior: dict) -> np.ndarray:
    """The feature matrix of one request, one row per option in the options' order (13 columns). `static` is
    `static_features` of its real options (the none option, if any, has no row there: `keys` and `static` cover the
    real options in order); `p` is decisio's distribution over all the request's options, the none option included."""
    n = len(p)
    p_t = temper(p, T)
    lp = np.log(np.clip(p_t, 1e-300, 1.0))
    top = int(np.argmax(p_t))
    X = np.zeros((n, N_COLS))
    real = [i for i in range(n) if i != none_index]
    for r, i in enumerate(real):
        g, a = prior.get(keys[r], (0, 0))
        X[i, 0], X[i, 1] = lp[i], float(i == top)
        X[i, 2 : 2 + N_STATIC] = static[r]
        X[i, 10] = np.log((g + 1) / (a + 2))
        X[i, 11] = float(a == 0)
    if none_index is not None:
        X[none_index, 0], X[none_index, 1], X[none_index, NONE_COL] = lp[none_index], float(none_index == top), 1.0
    return X


def prior_counts(examples: list[dict], idx) -> dict:
    """Option key -> (times it was the answer, times it was offered) over the examples `idx`."""
    pr = defaultdict(lambda: [0, 0])
    for i in idx:
        ex = examples[i]
        for r, k in enumerate(ex["keys"]):
            pr[k][0] += int(ex["gold_real"] == r)
            pr[k][1] += 1
    return {k: (v[0], v[1]) for k, v in pr.items()}


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fit_task_temperature(ps, golds) -> float:
    """The temperature (0.25 to 4) that minimises the log loss of decisio's plain distributions on the examples."""
    lps = [np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1.0)) for p in ps]
    nll = [-np.mean([np.log(max(_softmax(lp / T)[g], 1e-300)) for lp, g in zip(lps, golds)]) for T in T_GRID]
    return float(T_GRID[int(np.argmin(nll))])


def _fit_logit(Xs, ys, C, d):
    from scipy.optimize import minimize

    def f(w):
        val, grad = 0.5 * w @ w, w.copy()
        for X, y in zip(Xs, ys):
            p = _softmax(X @ w)
            val -= C * np.log(max(p[y], 1e-300))
            grad -= C * (X[y] - p @ X)
        return val, grad

    return minimize(f, np.zeros(d), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x


def fit(examples: list[dict], seed: int = 0, gate: bool = True) -> dict:
    """The dynamic task's stored record from labelled examples. Each example is a dict with
        static   (K, 8) the real options' static features              keys       the K real options' keys
        p        decisio's plain distribution over all the options      none_index the none option's index, or None
        gold     the answer's index among all the options
        gold_real the answer's index among the real options, or -1 when it is the none option
        served   decisio's distribution as it would answer without the re-ranker (its profile temperature applied);
                 the baseline the re-ranker must beat. `p` when absent.
    `gate=False` keeps the re-ranker whatever its cross-validation says (to evaluate the method without the gate).
    Returns {"applied": bool, "reason": ..., and, when applied: temperature, C, weights, mean, scale, prior, cv}."""
    n = len(examples)
    rec = {"format": FORMAT, "n_examples": n, "applied": False}
    if n < MIN_EXAMPLES:
        return {
            **rec,
            "reason": f"{n} examples; the re-ranker needs at least {MIN_EXAMPLES} (the certificate alone is made)",
        }
    ps = [np.asarray(e["p"], dtype=np.float64) for e in examples]
    golds = [int(e["gold"]) for e in examples]
    T = fit_task_temperature(ps, golds)
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, FOLDS)
    # the training features use a prior fitted on the other folds (an option's own label never informs its own row)
    X = [None] * n
    for k, fo in enumerate(folds):
        rest = [i for kk, f2 in enumerate(folds) if kk != k for i in f2]
        pr = prior_counts(examples, rest)
        for i in fo:
            e = examples[i]
            X[i] = rows_of(e["static"], e["keys"], e["p"], e["none_index"], T, pr)
    allrows = np.vstack(X)
    mu, sd = allrows.mean(0), allrows.std(0)
    sd[sd == 0] = 1.0
    Z = [(x - mu) / sd for x in X]
    cv = {}
    for C in CS:
        ll = []
        for k, fo in enumerate(folds):
            tr = [i for kk, f2 in enumerate(folds) if kk != k for i in f2]
            w = _fit_logit([Z[i] for i in tr], [golds[i] for i in tr], C, N_COLS)
            ll += [-np.log(max(_softmax(Z[i] @ w)[golds[i]], 1e-300)) for i in fo]
        cv[C] = float(np.mean(ll))
    C = min(CS, key=lambda c: (cv[c], c))
    served = [temper_free(e.get("served", e["p"])) for e in examples]  # what decisio alone would answer with
    plain_ll = float(np.mean([-np.log(max(q[g], 1e-300)) for q, g in zip(served, golds)]))
    plain_acc = float(np.mean([int(np.argmax(q)) == g for q, g in zip(served, golds)]))
    # cross-validated accuracy of the chosen C, for the report
    hits = []
    for k, fo in enumerate(folds):
        tr = [i for kk, f2 in enumerate(folds) if kk != k for i in f2]
        w = _fit_logit([Z[i] for i in tr], [golds[i] for i in tr], C, N_COLS)
        hits += [int(np.argmax(Z[i] @ w)) == golds[i] for i in fo]
    cv_acc = float(np.mean(hits))
    rec["cv"] = {
        "logloss_reranker": round(cv[C], 5),
        "logloss_decisio": round(plain_ll, 5),
        "accuracy_reranker": round(cv_acc, 4),
        "accuracy_decisio": round(plain_acc, 4),
        "C": C,
    }
    if gate and not (plain_ll - cv[C] >= MIN_GAIN and cv_acc >= plain_acc):
        return {**rec, "reason": "cross-validation on the labelled examples prefers decisio's own distribution"}
    w = _fit_logit(Z, golds, C, N_COLS)
    prior = prior_counts(examples, range(n))
    return {
        **rec,
        "applied": True,
        "reason": "cross-validated gain",
        "temperature": round(T, 6),
        "weights": [float(x) for x in w],
        "mean": [float(x) for x in mu],
        "scale": [float(x) for x in sd],
        "prior": {k: [int(a), int(b)] for k, (a, b) in prior.items()},
    }


def temper_free(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1.0)
    return p / p.sum()


def apply(rec: dict, static: np.ndarray, keys: list[str], p, none_index: int | None, served=None) -> np.ndarray:
    """The re-ranked distribution over a request's options. Where the record is not applied, decisio's own distribution
    as the server serves it: `served`, its readout with the server's temperature, which the caller must give."""
    if not rec or not rec.get("applied"):
        if served is None:
            raise ValueError(
                "a task whose re-ranker is not applied answers with decisio's served distribution: pass it"
            )
        return temper_free(served)
    prior = {k: tuple(v) for k, v in rec["prior"].items()}
    X = rows_of(static, keys, p, none_index, rec["temperature"], prior)
    Z = (X - np.asarray(rec["mean"])) / np.asarray(rec["scale"])
    return _softmax(Z @ np.asarray(rec["weights"]))


def never_seen_share(rec: dict, keys: list[str]) -> float:
    """The share of a request's real options that never appeared in the task's examples (the monitor's signal)."""
    if not keys:
        return 0.0
    prior = (rec or {}).get("prior") or {}
    return sum(k not in prior for k in keys) / len(keys)


# ---- the task's identity and the wire question's parts ---------------------------------------------------------------


def task_key(instructions) -> str:
    """A dynamic task's identity: its question's instructions in their comparison form; its options are not part."""
    return "dynamic:" + hashlib.sha256(norm(instructions or "").encode()).hexdigest()


def state_text(state) -> str:
    """The text a request's state is read as: a string as it is; an object the string values of its fields, in order."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(state_text(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return "\n".join(state_text(v) for v in state)
    return str(state)


def question_parts(criteria: dict, render, none_key: str = NONE_KEY):
    """(option keys on the wire, the real options' texts, their hashes, the none option's index or None) of a choice
    question whose options are `criteria` (key -> description); `render` turns a description into text."""
    wire_keys = list(criteria)
    none_index = wire_keys.index(none_key) if none_key in wire_keys else None
    texts = [render(criteria[k]) for i, k in enumerate(wire_keys) if i != none_index]
    return wire_keys, texts, [option_key(t) for t in texts], none_index


def all_unseen(rec: dict, keys: list[str]) -> bool:
    """True when none of a request's real options appeared in the task's examples (the monitor's event)."""
    prior = (rec or {}).get("prior") or {}
    return bool(keys) and all(k not in prior for k in keys)
