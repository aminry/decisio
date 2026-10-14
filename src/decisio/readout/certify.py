# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Certified answers: a confidence threshold with a bound on the rate of wrong answers above it, from labelled examples.

The promise, in the words of the docs: among the answers a task marks certified, at most `alpha` are wrong, with
probability at least `1 - delta` over the draw of the labelled examples, for traffic like those examples and under the
serving stack they were read through.

The procedure is Learn-then-Test with fixed-sequence testing over thresholds and a Clopper-Pearson upper bound on the
error
of the answers above each threshold (Angelopoulos, Bates, Candes, Jordan, Lei 2021, "Learn then Test"). It is the
procedure of the shad0w package's `SelectiveRiskController(alpha, delta, "auto")`, written out here so that decisio
takes
no new dependency (scipy, already a dependency, gives the Beta quantile): two arms, each given half of `delta`,

  fixed sequence  test, from the strictest candidate to the most lenient, "the error above this threshold is at most
                  alpha" with the bound at delta / 2, and stop at the first failure; the last threshold that held is
                  kept;
  Bonferroni      test twelve fixed served shares (5 percent to 100 percent of the examples) with the bound at
                  delta / 2 / (number tested) and keep the most lenient that holds;

and the answer is the stricter (higher) of the two thresholds. A threshold of `inf` means nothing could be certified.
`tests/unit/test_certify.py` pins it to reference values, and `runs/2026-10-14_certified-answers/` re-runs the check
that it
equals the package's procedure on every split of a public record.

Ties: every example with the k-th confidence is served with it, so a threshold never separates equal confidences.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from scipy.stats import beta as _beta

PROCEDURE = "learn-then-test: fixed sequence and Bonferroni arms, each at delta/2, Clopper-Pearson upper bounds"
LEVELS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)  # the Bonferroni arm's served shares
GRID = 50  # candidate thresholds of the fixed sequence


@functools.lru_cache(maxsize=None)
def upper_bound(errors: int, n: int, delta: float) -> float:
    """The one-sided Clopper-Pearson upper bound, at level 1 - delta, on an error rate seen as `errors` in `n`."""
    if n == 0 or errors >= n:
        return 1.0
    return float(_beta.ppf(1 - delta, errors + 1, n - errors))


@functools.lru_cache(maxsize=None)
def _tolerant_start(alpha: float, delta: float, n: int) -> int:
    """The smallest served count at which a run with alpha / 2 of its answers wrong could still pass."""
    for k in range(1, n + 1):
        if upper_bound(int(alpha * k / 2), k, delta) <= alpha:
            return k
    return n


def _served_k(c_sorted: np.ndarray, k: int) -> int:
    return int(np.searchsorted(-c_sorted, -c_sorted[k - 1], side="right"))  # rows tied with the k-th are served too


def _fixed_sequence(c_sorted: np.ndarray, errs: np.ndarray, alpha: float, delta: float) -> float:
    n = len(c_sorted)
    if n == 0:
        return math.inf
    k0 = max(
        math.ceil(math.log(delta) / math.log(1 - alpha)),
        math.ceil(0.05 * n),
        min(50, n),
        min(_tolerant_start(alpha, delta, n), n),
    )
    thr = math.inf
    for k in np.unique(np.linspace(min(k0, n), n, GRID).astype(int)):
        k = _served_k(c_sorted, int(k))
        if upper_bound(int(errs[k - 1]), k, delta) > alpha:
            break
        thr = float(c_sorted[k - 1])
    return thr


def _bonferroni(c_sorted: np.ndarray, errs: np.ndarray, alpha: float, delta: float) -> float:
    n = len(c_sorted)
    ks = sorted({max(1, int(round(lv * n))) for lv in LEVELS}) if n else []
    thr = math.inf
    for k in ks:
        k = _served_k(c_sorted, k)
        if upper_bound(int(errs[k - 1]), k, delta / len(ks)) <= alpha:
            thr = min(thr, float(c_sorted[k - 1]))
    return thr


def threshold(conf, correct, alpha: float, delta: float) -> float:
    """The certified confidence threshold for `conf` (the confidence of each answer) and `correct` (whether
    it was right), or `inf` when no threshold can be certified."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=bool)
    order = np.argsort(-conf)
    c_sorted, errs = conf[order], np.cumsum(~correct[order])
    return min(_fixed_sequence(c_sorted, errs, alpha, delta / 2), _bonferroni(c_sorted, errs, alpha, delta / 2))


def empirical_threshold(conf, correct, alpha: float) -> float:
    """The most lenient threshold whose error among the examples themselves is at most alpha: no bound, so
    not a promise; kept as the context for what a certificate gives up (`inf` when there is none)."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=bool)
    order = np.argsort(-conf)
    c_sorted, errs = conf[order], np.cumsum(~correct[order])
    n = len(c_sorted)
    k = np.arange(1, n + 1)
    ends = np.append(c_sorted[:-1] != c_sorted[1:], True)  # only thresholds that end a group of equal confidences
    ok_k = ends & (errs / k <= alpha + 1e-12)
    return float(c_sorted[np.nonzero(ok_k)[0][-1]]) if ok_k.any() else math.inf


@functools.lru_cache(maxsize=None)
def minimum_examples(alpha: float, delta: float) -> int:
    """The fewest labelled examples at which the procedure could certify anything, which is when every one of them is
    answered right: with fewer, no threshold can be certified whatever the model does."""
    n = 1
    while n < 100_000:
        if math.isfinite(threshold(np.ones(n), np.ones(n, dtype=bool), alpha, delta)):
            return n
        n += 1
    return n


def share_and_errors(conf, correct, thr: float) -> tuple[int, int]:
    """(answers at or above `thr`, wrong among them)."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=bool)
    m = conf >= thr
    return int(m.sum()), int((~correct[m]).sum())


def certificate(conf, correct, alpha: float, delta: float) -> dict:
    """The numbers of a certification on `conf` and `correct` (the held-out examples' served confidence and
    correctness)."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=bool)
    n = len(conf)
    thr = threshold(conf, correct, alpha, delta)
    emp = empirical_threshold(conf, correct, alpha)
    out = {
        "procedure": PROCEDURE,
        "alpha": alpha,
        "delta": delta,
        "n": n,
        "accuracy": float(correct.mean()) if n else None,
        "threshold": thr if math.isfinite(thr) else None,
        "certified": math.isfinite(thr),
        "empirical_share_at_alpha": float((conf >= emp).mean()) if n and math.isfinite(emp) else 0.0,
    }
    if math.isfinite(thr):
        served, wrong = share_and_errors(conf, correct, thr)
        out.update(
            {
                "certified_share": served / n,
                "certified_n": served,
                "errors_at_threshold": wrong,
                "error_rate_at_threshold": wrong / served,
                "upper_bound_at_threshold": upper_bound(wrong, served, delta / 2),
            }
        )
    else:
        out.update(
            {
                "certified_share": 0.0,
                "certified_n": 0,
                "errors_at_threshold": None,
                "error_rate_at_threshold": None,
                "upper_bound_at_threshold": None,
            }
        )
    return out
