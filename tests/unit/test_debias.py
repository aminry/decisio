# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The readout's debiasing arithmetic (`decisio.readout.debias`), on synthetic scores.

D1  the analytic gradient of the penalised log loss equals finite differences
D2  fit_bias recovers a planted per-slot bias from labelled scores
D3  pride_prior recovers a planted slot prior from cyclically shifted scores, without labels
D4  cyclic shifts map back: a scorer that always picks one option's text gives that option's original index under
    every shift, and the cyclic average of such readouts is that option with probability one

  uv run pytest -q tests/unit/test_debias.py
"""

import numpy as np
import pytest

from decisio.readout import debias


@pytest.fixture(scope="module")
def planted():
    rng = np.random.default_rng(0)
    K, n = 10, 3000
    b_true = rng.normal(0, 0.6, K)
    b_true -= b_true.mean()
    content = rng.normal(0, 1.5, (n, K))
    y = np.array([rng.choice(K, p=debias.apply_bias(c, np.zeros(K))) for c in content])
    return K, b_true, content, y, content + b_true  # the last: what a biased readout reports


def test_d1_gradient(planted):
    K, _, _, y, L = planted
    b0 = np.random.default_rng(1).normal(0, 0.3, K)
    f0, g = debias._nll_grad(b0, L[:200], y[:200], 0.01)
    fd = np.array(
        [(debias._nll_grad(b0 + 1e-6 * np.eye(K)[k], L[:200], y[:200], 0.01)[0] - f0) / 1e-6 for k in range(K)]
    )
    assert np.abs(fd - g).max() < 1e-4


def test_d2_fit_bias_recovers_a_planted_bias(planted):
    K, b_true, _, y, L = planted
    b, lam, _ = debias.fit_bias(list(L), y, K)
    assert np.abs(b - b_true).max() < 0.15, (b, lam)


def test_d3_pride_prior_recovers_a_planted_prior(planted):
    K, b_true, content, _, _ = planted
    shifts = []
    for c in content[:400]:
        shifts.append(
            np.array(
                [np.log(debias.apply_bias(c[debias.cyclic_order(K, j, K)] + b_true, np.zeros(K))) for j in range(K)]
            )
        )  # all K shifts: the content cancels exactly
    assert np.corrcoef(debias.pride_prior(shifts), b_true)[0, 1] > 0.95


@pytest.mark.parametrize("K,k", [(4, 4), (10, 4), (77, 2), (2, 2)])
def test_d4_cyclic_shifts_map_back(K, k):
    texts = [f"option {i}" for i in range(K)]
    for target in (0, K - 1, K // 2):
        by_shift = []
        for j in range(k):
            order = debias.cyclic_order(K, j, k)
            shown = [texts[i] for i in order]  # position -> the option shown there
            p_pos = np.eye(K)[shown.index(texts[target])]  # the scorer picks the target's text
            back = debias.unpermute(p_pos, order)
            assert int(back.argmax()) == target
            by_shift.append(back)
        assert debias.cyclic_order(K, 0, k) == list(range(K))  # shift 0 is the plain order
        assert np.array_equal(debias.cyclic_average(by_shift), np.eye(K)[target])
