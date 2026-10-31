# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Dynamic-option teaching (`decisio.serve.dynamic`): the re-ranker of a task whose options change with every request.

D1  the features of an option: hand-computed rows for a state and its options (token share and overlap, where it sits,
    exact or partial or absent, lengths), the none option, the prior's log odds and the never-seen flag
D2  the evaluation reproduced: on the stored items of five recurring questions (tests/data/dynamic_options: numbers and
    hashes of the items, no text), the 200-label draw 0 of each question and base gives the same certified share, error
    and accuracy as an independent offline implementation of the same method, and the same numbers for decisio's own
    distribution alone; pooled, the re-ranker gains accuracy over decisio alone on both bases
D3  the gate: a re-ranker is kept only when its cross-validation beats decisio's own distribution, and is declined
    otherwise
D4  apply: the same request gives the same distribution; a record that is not applied returns decisio's distribution; a
    request with no none option and the never-seen share work

  uv run pytest -q tests/unit/test_dynamic.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from decisio.readout import certify
from decisio.serve import dynamic as Dy

DATA = Path(__file__).resolve().parents[1] / "data" / "dynamic_options"
FAMILIES = ("brand", "entity", "tool", "product", "route")
EXPECTED = json.loads((DATA / "expected.json").read_text())
ALPHA, DELTA = 0.05, 0.1


def examples_of(z, pool, base, with_served=False):
    k = z[f"{pool}_k"]  # the real options of each request; the none option is the last of the p vector
    off, offp = np.concatenate([[0], np.cumsum(k)]), np.concatenate([[0], np.cumsum(k + 1)])
    out = []
    for i, n in enumerate(k):
        n = int(n)
        p = z[f"{pool}_p_{base}"][offp[i] : offp[i + 1]]
        keys = [f"{int(v):016x}" for v in z[f"{pool}_keys"][off[i] : off[i] + n]]
        gold = int(z[f"{pool}_gold"][i])
        e = {
            "static": z[f"{pool}_static"][off[i] : off[i] + n],
            "keys": keys,
            "p": p,
            "none_index": n,
            "gold": gold,
            "gold_real": gold if gold < n else -1,
        }
        e["served"] = z[f"{pool}_served_{base}"][offp[i] : offp[i + 1]] if with_served else p
        out.append(e)
    return out


def certified(conf, correct_test_conf, y_cert_correct, y_test_correct):
    thr = certify.threshold(np.asarray(conf), np.asarray(y_cert_correct), ALPHA, DELTA)
    ok = correct_test_conf >= thr if np.isfinite(thr) else np.zeros(len(correct_test_conf), bool)
    share = float(ok.mean())
    err = float((~np.asarray(y_test_correct)[ok]).mean()) if ok.any() else None
    return share, err


def top_conf_correct(dists, golds):
    return np.array([float(np.max(d)) for d in dists]), np.array([int(np.argmax(d)) == g for d, g in zip(dists, golds)])


# ---- D1 ----------------------------------------------------------------------------------------------------


def test_d1_static_features_by_hand():
    state = "Apple iPhone 13 Pro 256GB, graphite, unlocked"
    f = Dy.static_features(state, ["Apple", "Samsung", "iPhone 13 Pro", "Apple Watch", "graphite blue"])
    st = "apple iphone 13 pro 256gb graphite unlocked"
    # Apple: token share 1, Jaccard 1/7, the whole option occurs at position 0
    assert (
        f[0, 0] == 1.0
        and f[0, 1] == pytest.approx(1 / 7)
        and f[0, 2] == 0.0
        and f[0, 4] == 1.0
        and f[0, 3] == f[0, 5] == 0.0
    )
    assert f[0, 6] == pytest.approx(np.log1p(1)) and f[0, 7] == pytest.approx(np.log1p(5))
    # Samsung: nothing occurs
    assert f[1, 0] == f[1, 1] == f[1, 2] == 0.0 and f[1, 3] == 1.0 and f[1, 4] == f[1, 5] == 0.0
    # iPhone 13 Pro occurs as a phrase, starting after "apple "
    assert f[2, 4] == 1.0 and f[2, 2] == pytest.approx(st.find("iphone 13 pro") / len(st))
    # Apple Watch: only one of two tokens occurs, at the front of the state
    assert f[3, 0] == 0.5 and f[3, 5] == 1.0 and f[3, 4] == 0.0 and f[3, 2] == 0.0
    # graphite blue: partial, "graphite" sits after "apple iphone 13 pro 256gb "
    assert f[4, 5] == 1.0 and f[4, 2] == pytest.approx(len("apple iphone 13 pro 256gb ") / len(st))


def test_d1_rows_carry_decisios_distribution_the_prior_and_the_none_option():
    static = Dy.static_features("red shoes", ["red", "blue"])
    keys = [Dy.option_key("red"), Dy.option_key("blue")]
    p = np.array([0.6, 0.1, 0.3])  # red, blue, none
    prior = {keys[0]: (3, 4)}  # red was the answer 3 times in 4
    X = Dy.rows_of(static, keys, p, 2, 1.0, prior)
    assert X.shape == (3, Dy.N_COLS)
    assert X[0, 0] == pytest.approx(np.log(0.6)) and X[0, 1] == 1.0 and X[1, 1] == X[2, 1] == 0.0
    assert X[0, 10] == pytest.approx(np.log(4 / 6)) and X[0, 11] == 0.0
    assert X[1, 10] == pytest.approx(np.log(1 / 2)) and X[1, 11] == 1.0  # blue never appeared
    assert X[2, Dy.NONE_COL] == 1.0 and X[2, 2:12].tolist() == [0.0] * 10 and X[2, 0] == pytest.approx(np.log(0.3))
    assert X[:2, Dy.NONE_COL].tolist() == [0.0, 0.0]


def test_d1_the_task_temperature_changes_the_log_probability_column_not_the_top_flag():
    static = Dy.static_features("x y", ["x", "y"])
    keys = ["a", "b"]
    p = np.array([0.7, 0.2, 0.1])
    cold, hot = Dy.rows_of(static, keys, p, 2, 0.5, {}), Dy.rows_of(static, keys, p, 2, 2.0, {})
    assert cold[0, 1] == hot[0, 1] == 1.0
    assert cold[0, 0] > hot[0, 0]  # sharper at T < 1: more mass on the top option
    assert Dy.fit_task_temperature([np.array([0.9, 0.05, 0.05])] * 40, [0] * 40) < 1.0  # right and confident: sharper
    assert Dy.fit_task_temperature([np.array([0.9, 0.05, 0.05])] * 40, [1] * 40) > 1.0  # wrong and confident: flatter


# ---- D2 ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("base", ["12b", "31b"])
@pytest.mark.parametrize("family", FAMILIES)
def test_d2_the_evaluation_cells_are_reproduced(family, base):
    z = np.load(DATA / f"{family}.npz")
    teach, cert, test = (examples_of(z, w, base, with_served=(w != "teach")) for w in ("teach", "cert", "test"))
    ref = EXPECTED[f"{family}|{base}"]
    # A0: the certificate on decisio's own served distribution
    sc, st_ = [e["served"] for e in cert], [e["served"] for e in test]
    cc, ok_c = top_conf_correct(sc, [e["gold"] for e in cert])
    ct, ok_t = top_conf_correct(st_, [e["gold"] for e in test])
    share, err = certified(cc, ct, ok_c, ok_t)
    assert share == pytest.approx(ref["A0"]["share"], abs=1e-9)
    assert (err is None and ref["A0"]["error"] is None) or err == pytest.approx(ref["A0"]["error"], abs=1e-9)
    assert float(ok_t.mean()) == pytest.approx(ref["A0"]["accuracy"], abs=1e-9)
    # the re-ranker fitted on the 200 labels (kept whatever its cross-validation says: the method without the gate),
    # certified on the top probability of its distribution
    rec = Dy.fit(teach, gate=False)
    assert rec["applied"] and len(rec["weights"]) == Dy.N_COLS and len(rec["mean"]) == len(rec["scale"]) == Dy.N_COLS
    pc = [Dy.apply(rec, e["static"], e["keys"], e["p"], e["none_index"]) for e in cert]
    pt = [Dy.apply(rec, e["static"], e["keys"], e["p"], e["none_index"]) for e in test]
    cc, ok_c = top_conf_correct(pc, [e["gold"] for e in cert])
    ct, ok_t = top_conf_correct(pt, [e["gold"] for e in test])
    share, err = certified(cc, ct, ok_c, ok_t)
    assert float(ok_t.mean()) == pytest.approx(ref["B"]["accuracy"], abs=1e-9)
    assert share == pytest.approx(ref["B"]["share"], abs=1e-9)
    assert (err is None and ref["B"]["error"] is None) or err == pytest.approx(ref["B"]["error"], abs=1e-9)


@pytest.mark.parametrize("base", ["12b", "31b"])
def test_d2_pooled_over_the_five_families_b_gains_accuracy_and_a0_certifies_about_half(base):
    a0 = np.mean([EXPECTED[f"{f}|{base}"]["A0"]["accuracy"] for f in FAMILIES])
    b = np.mean([EXPECTED[f"{f}|{base}"]["B"]["accuracy"] for f in FAMILIES])
    share = np.mean([EXPECTED[f"{f}|{base}"]["A0"]["share"] for f in FAMILIES])
    errs = [
        EXPECTED[f"{f}|{base}"]["A0"]["error"] for f in FAMILIES if EXPECTED[f"{f}|{base}"]["A0"]["error"] is not None
    ]
    # draw 0 cells: the medians over three draws are 80.3 and 83.2 (decisio alone), 84.0 and 85.2 (re-ranker)
    assert b - a0 >= {"12b": 0.03, "31b": 0.015}[base]
    assert 0.45 <= share <= 0.6 and max(errs) <= 0.05


# ---- D3 and D4 ------------------------------------------------------------------------------------------


def synthetic(n, informative_p, seed=0, text_says=True):
    """Requests of four options and no none option, whose answer is one of them; the state names it when `text_says`.
    decisio's p is uninformative, or calibrated: 0.8 on its top option, which is the answer four times in five."""
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(40)]
    out = []
    for _ in range(n):
        opts = list(rng.choice(words, 4, replace=False))
        g = int(rng.integers(0, 4))
        state = f"text about {opts[g]} and more" if text_says else "text about nothing in particular"
        keys = [Dy.option_key(o) for o in opts]
        if informative_p:
            top = g if rng.random() < 0.8 else int(rng.choice([j for j in range(4) if j != g]))
            p = np.full(4, 0.2 / 3)
            p[top] = 0.8
        else:
            p = rng.dirichlet(np.ones(4))
        out.append(
            {
                "static": Dy.static_features(state, opts),
                "keys": keys,
                "p": p,
                "served": p,
                "none_index": None,
                "gold": g,
                "gold_real": g,
            }
        )
    return out


def test_d3_the_gate_keeps_a_reranker_that_helps_and_declines_one_that_does_not():
    helps = Dy.fit(synthetic(120, informative_p=False))  # decisio knows nothing; the text says it all
    assert helps["applied"] and helps["reason"] == "cross-validated gain"
    assert helps["cv"]["accuracy_reranker"] > helps["cv"]["accuracy_decisio"] + 0.3
    no_gain = Dy.fit(
        synthetic(200, informative_p=True, text_says=False)
    )  # decisio is calibrated, the text says nothing
    assert not no_gain["applied"] and "prefers decisio's own distribution" in no_gain["reason"]
    assert Dy.fit(synthetic(200, informative_p=True, text_says=False), gate=False)[
        "applied"
    ]  # without the gate it is kept
    few = Dy.fit(synthetic(Dy.MIN_EXAMPLES - 1, informative_p=False))
    assert not few["applied"] and "at least" in few["reason"]


def test_d4_apply_is_deterministic_and_a_declined_record_returns_decisios_distribution():
    ex = synthetic(120, informative_p=False)
    rec = Dy.fit(ex)
    e = synthetic(1, informative_p=False, seed=9)[0]
    a = Dy.apply(rec, e["static"], e["keys"], e["p"], e["none_index"])
    b = Dy.apply(rec, e["static"], e["keys"], e["p"], e["none_index"])
    assert np.array_equal(a, b) and a.sum() == pytest.approx(1.0) and len(a) == 4
    declined = {"format": Dy.FORMAT, "applied": False}
    assert np.allclose(Dy.apply(declined, e["static"], e["keys"], e["p"], e["none_index"]), e["p"] / e["p"].sum())
    assert np.allclose(Dy.apply(None, e["static"], e["keys"], e["p"], e["none_index"]), e["p"] / e["p"].sum())
    # a request with a none option: five rows, five probabilities
    p5 = np.append(e["p"] * 0.9, 0.1)
    assert len(Dy.apply(rec, e["static"], e["keys"], p5, 4)) == 5


def test_d4_never_seen_share_and_the_stored_record_holds_hashes_only():
    ex = synthetic(60, informative_p=False)
    rec = Dy.fit(ex, gate=False)
    seen = ex[0]["keys"][0]
    assert Dy.never_seen_share(rec, [seen]) == 0.0
    assert Dy.never_seen_share(rec, [seen, "ffffffffffffffff"]) == 0.5
    assert Dy.never_seen_share(rec, []) == 0.0
    assert Dy.never_seen_share({"applied": False}, ["x"]) == 1.0
    blob = json.dumps(rec)
    assert (
        all(len(k) == 16 for k in rec["prior"]) and "w1" not in blob and "text about" not in blob
    )  # no option or request text
    json.loads(blob)  # serialisable as is
