# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The intent head (`decisio.readout.intent_head`): the folded serving form equals the fitted head; the guards hold.

N1  the stored head (A, c) served with apply_intent_head equals the fitted classifier, to float32 storage
N2  a different option list (here: rotated) serves the plain readout
N3  fewer than MIN_PER_OPTION (2) examples per option: the head is off, and the reason names the short options
N5  the floor is exact: every option at MIN_PER_OPTION examples fits a head, one option one short declines it
N4  a hidden state that carries nothing about the labels does not make an informative readout worse on new items

  uv run pytest -q tests/unit/test_intent_head.py
"""

import numpy as np
import pytest

from decisio.readout.intent_head import (
    MIN_PER_OPTION,
    apply_intent_head,
    fit_intent_head,
    fit_linear,
    floor_reason,
    options_digest,
    short_options,
)


def data(K=12, per=8, d=64, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(K), per)
    proto = rng.normal(size=(K, d))
    H = proto[y] + rng.normal(size=(len(y), d))
    logp = rng.normal(size=(len(y), K))
    logp -= np.log(np.exp(logp).sum(1, keepdims=True))
    return H, logp, y, [f"intent {k}" for k in range(K)]


@pytest.fixture(scope="module")
def fitted():
    H, logp, y, opts = data()
    rec = fit_intent_head(logp, H, y, opts, fingerprint="m", task_id="t")
    assert rec["applied"], rec["reason"]
    return H, logp, y, opts, rec


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def test_n1_folded_form_equals_the_fitted_head(fitted):
    H, logp, y, opts, rec = fitted
    fn, _ = fit_linear(H, logp, y, len(opts), offset=True, seed=0)
    Ht, lpt, _, _ = data(seed=1)
    got = np.array([apply_intent_head(lpt[i], Ht[i], rec, opts) for i in range(len(Ht))])
    assert (
        np.abs(got - softmax(fn(Ht, lpt) / rec["temperature"])).max() < 1e-4
    )  # float32 storage of A and c, served at T


def test_n2_another_option_list_serves_the_plain_readout(fitted):
    _, _, _, opts, rec = fitted
    Ht, lpt, _, _ = data(seed=1)
    swapped = opts[1:] + opts[:1]
    assert options_digest(swapped) != rec["options_sha256"]
    assert np.allclose(apply_intent_head(lpt[0], Ht[0], rec, swapped), softmax(lpt[0]))


def test_n3_too_few_examples_per_option(fitted):
    H, logp, y, opts, _ = fitted
    step = len(y) // len(opts)  # one example of every option
    few = fit_intent_head(logp[::step], H[::step], y[::step], opts)
    assert not few["applied"] and f"at least {MIN_PER_OPTION}" in few["reason"]
    assert "intent 0 (1)" in few["reason"] and few["short_options"] == opts


def test_n5_the_floor_is_exact_and_names_the_short_options():
    H, logp, y, opts = data(per=MIN_PER_OPTION + 1)
    at_floor = np.concatenate([np.where(y == k)[0][:MIN_PER_OPTION] for k in range(len(opts))])
    ok = fit_intent_head(logp[at_floor], H[at_floor], y[at_floor], opts)
    assert ok["min_per_option"] == MIN_PER_OPTION and "every option needs" not in (ok.get("reason") or "")
    one_short = np.delete(at_floor, np.where(y[at_floor] == 3)[0][0])  # option 3 now has MIN_PER_OPTION - 1 examples
    short = fit_intent_head(logp[one_short], H[one_short], y[one_short], opts)
    assert not short["applied"] and short["short_options"] == ["intent 3"]
    assert f"1 of {len(opts)} have fewer: intent 3 ({MIN_PER_OPTION - 1})" in short["reason"]
    counts = np.bincount(y[one_short], minlength=len(opts))
    assert short_options(counts, opts) == [("intent 3", MIN_PER_OPTION - 1)]


def test_n6_a_long_list_of_short_options_is_cut_in_the_reason():
    opts = [f"o{k}" for k in range(40)]
    counts = np.zeros(40, int)
    reason = floor_reason(counts, opts)
    assert (
        "40 of 40 have fewer" in reason and "and 28 more" in reason and "o11 (0)" in reason and "o12 (0)" not in reason
    )


def test_n4_uninformative_hidden_state_does_no_harm(fitted):
    H, _, y, opts, _ = fitted
    rng = np.random.default_rng(3)
    K = len(opts)

    def informative(yy):
        s = rng.normal(size=(len(yy), K))
        s[np.arange(len(yy)), yy] += 2.0
        return s - np.log(np.exp(s).sum(1, keepdims=True))

    lp_tr, lp_te = informative(y), informative(y)
    rec = fit_intent_head(lp_tr, rng.normal(size=H.shape), y, opts)
    Hte = rng.normal(size=H.shape)
    acc_plain = float(np.mean(lp_te.argmax(1) == y))
    acc_head = float(np.mean([apply_intent_head(lp_te[i], Hte[i], rec, opts).argmax() == y[i] for i in range(len(y))]))
    assert acc_head >= acc_plain - 0.03, (acc_head, acc_plain)
