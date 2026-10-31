# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The opt-in fast table (`decisio.serve.fast_table`): the classifier, the two-stage certificate and the timing guard.
No encoder: embeddings are synthetic.

F1  the stored classifier gives the probabilities of the fitted logistic regression (multiclass and binary), maps
    options the training never saw to zero, and survives the JSON round trip
F2  the regularisation is chosen by cross-validated log loss and the fit is deterministic
F3  the cascade's thresholds equal a direct reimplementation (the table's at delta/2, decisio's on what the table
    defers, also at delta/2), `served` applies them, and a stage with no threshold that holds is None
F4  the timing guard keeps the table only when the expected mean time falls by a quarter
F5  without the extra, fitting and embedding say what to install
F6  the cascade's promise on simulated populations: over many draws of the held-out examples, the share of draws in
    which the wrong share among the certified answers (both stages) exceeds alpha stays within delta
F7  the study's CLINC150 cell (1,000 labels, draw 0) is reproduced on stored embeddings: table share, certified error,
    thresholds and accuracy, on both bases' decisio side
F8  the docs' table and counts (4 of 10 sets, 1 of 7 without the assistant-command sets) equal the run's summary.json

  uv run pytest -q tests/unit/test_fast_table.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from decisio.readout import certify
from decisio.serve import fast_table as FT

sklearn = pytest.importorskip("sklearn")


def blobs(n, k, d=24, seed=0, sep=3.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * sep
    y = rng.integers(0, k, n)
    X = centers[y] + rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32), y


@pytest.mark.parametrize("k", [2, 5])
def test_f1_the_stored_classifier_matches_the_fitted_one(k):
    from sklearn.linear_model import LogisticRegression

    X, y = blobs(400, k)
    rec = FT.fit(X, y, k)
    ref = LogisticRegression(C=rec["C"], max_iter=2000).fit(X, y)
    Xt, _ = blobs(100, k, seed=9)
    assert np.allclose(FT.predict(rec, Xt), ref.predict_proba(Xt), atol=1e-5)
    assert rec["binary"] == (k == 2) and rec["format"] == FT.FORMAT and rec["encoder"] == FT.ENCODER
    back = json.loads(json.dumps(rec))  # arrays are packed: JSON round trip
    assert np.array_equal(FT.predict(back, Xt), FT.predict(rec, Xt))


def test_f1_an_option_the_training_never_saw_has_probability_zero():
    X, y = blobs(300, 4)
    y = np.where(y == 3, 0, y)  # option 3 never appears
    rec = FT.fit(X, y, 5)
    P = FT.predict(rec, X[:20])
    assert P.shape == (20, 5) and (P[:, 3:] == 0).all() and np.allclose(P.sum(1), 1)
    assert rec["classes"] == [0, 1, 2]


def test_f2_c_is_chosen_by_cross_validated_log_loss_and_the_fit_is_deterministic():
    X, y = blobs(300, 3, sep=0.4)  # overlapping: a strong penalty cannot be the best, the choice is made not defaulted
    a, b = FT.fit(X, y, 3), FT.fit(X, y, 3)
    assert a["C"] == min(FT.CS, key=lambda c: (a["cv_logloss"][str(c)], c)) and a["C"] in FT.CS
    assert a == b
    assert set(a["cv_logloss"]) == {str(c) for c in FT.CS}


def reference_cascade(cT, pT, y, cD, pD, alpha, delta):
    """Two stages: the table's threshold at delta/2; decisio's on what the table defers, at delta/2."""
    tau_t = certify.threshold(cT, pT == y, alpha, delta / 2)
    by_table = cT >= tau_t if np.isfinite(tau_t) else np.zeros(len(y), bool)
    dfr = ~by_table
    tau_d = certify.threshold(cD[dfr], pD[dfr] == y[dfr], alpha, delta / 2) if dfr.any() else np.inf
    return tau_t, tau_d, by_table


def stage_data(n, seed, table_acc=0.97, decisio_acc=0.9):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 6, n)
    cT = np.clip(rng.beta(8, 1.2, n), 0, 1)
    pT = np.where(rng.random(n) < np.clip(table_acc + (cT - 0.85) * 0.5, 0, 1), y, (y + 1) % 6)
    cD = np.clip(rng.beta(6, 1.5, n), 0, 1)
    pD = np.where(rng.random(n) < np.clip(decisio_acc + (cD - 0.8) * 0.5, 0, 1), y, (y + 2) % 6)
    return cT, pT, y, cD, pD


@pytest.mark.parametrize("seed", range(6))
def test_f3_the_cascade_thresholds_equal_the_reference(seed):
    cT, pT, y, cD, pD = stage_data(900, seed)
    got = FT.cascade(cT, pT, y, cD, pD, 0.05, 0.1)
    tau_t, tau_d, by_table = reference_cascade(cT, pT, y, cD, pD, 0.05, 0.1)
    assert (got["table_threshold"] is None) == (not np.isfinite(tau_t))
    if np.isfinite(tau_t):
        assert got["table_threshold"] == tau_t
    assert (got["decisio_threshold"] is None) == (not np.isfinite(tau_d))
    if np.isfinite(tau_d):
        assert got["decisio_threshold"] == tau_d
    assert got["table_share"] == pytest.approx(by_table.mean()) and got["n"] == 900
    sh, cert, pred = FT.served(cT, pT, cD, pD, got)
    assert (sh == by_table).all()
    assert (pred[sh] == pT[sh]).all() and (pred[~sh] == pD[~sh]).all()
    wrong_certified = (pred != y)[cert].mean() if cert.any() else 0.0
    assert wrong_certified <= 0.08  # the promise (0.05) holds with room on the sample it was made on


def test_f3_a_stage_without_a_threshold_is_none():
    n = 200
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, n)
    flat = np.full(n, 0.6)
    right = np.where(rng.random(n) < 0.5, y, (y + 1) % 4)
    c = FT.cascade(flat, right, y, flat, right, 0.05, 0.1)
    assert c["table_threshold"] is None and c["decisio_threshold"] is None and c["table_share"] == 0.0
    sh, cert, _ = FT.served(flat, right, flat, right, c)
    assert not sh.any() and not cert.any()


def test_f4_the_guard_keeps_the_table_only_when_it_saves_a_quarter():
    # the table takes 25 ms, decisio 60 ms: with 60% answered by the table, expected 25 + 0.4 * 60 = 49 (18% saved)
    g = FT.timing_guard([25.0] * 50, [60.0] * 50, 0.6)
    assert not g["kept"] and g["saving"] == pytest.approx(1 - 49 / 60, abs=1e-3) and g["expected_ms_mean"] == 49.0
    g = FT.timing_guard([25.0] * 50, [60.0] * 50, 0.8)
    assert g["kept"] and g["saving"] == pytest.approx(1 - 37 / 60, abs=1e-3)
    assert not FT.timing_guard([70.0], [60.0], 0.99)["kept"]  # slower than decisio even at 99%
    assert FT.timing_guard([10.0], [60.0], 0.0)["kept"] is False
    assert FT.timing_guard([20.0, 30.0], [50.0, 70.0], 0.75, min_saving=0.5)["kept"] is False
    # the table defers the two slow requests: counted as it happened the mean is 100 ms, as slow as decisio alone,
    # where the share-based estimate (10 + 0.5 * 100 = 60 ms) would keep it
    g = FT.timing_guard([10.0] * 4, [20.0, 20.0, 180.0, 180.0], 0.5, deferred=[False, False, True, True])
    assert g["expected_ms_mean"] == 100.0 and g["saving"] == 0.0 and not g["kept"]
    assert FT.timing_guard([10.0] * 4, [20.0, 20.0, 180.0, 180.0], 0.5)["kept"]  # the share-based estimate: 60 of 100


def test_f5_without_the_extra_it_says_what_to_install(monkeypatch):
    import builtins

    real = builtins.__import__

    def refuse(name, *a, **k):
        if name.startswith("sentence_transformers") or name.startswith("sklearn"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.setattr(FT, "_encoder", None)
    with pytest.raises(FT.TableUnavailable, match=r"decisio\[table\]"):
        FT.embed(["a"])
    with pytest.raises(FT.TableUnavailable, match=r"decisio\[table\]"):
        FT.fit(np.zeros((4, 3), np.float32), [0, 1, 0, 1], 2)


def population(n, seed):
    """A population of requests: the table's and decisio's top confidence and whether each is right. Right answers are
    more confident; a request that is hard for one is somewhat likelier hard for the other."""
    rng = np.random.default_rng(seed)
    hard = rng.random(n) < 0.3
    ct = np.clip(np.where(hard, rng.beta(3, 3, n), rng.beta(9, 1.2, n)), 0, 1)
    cd = np.clip(np.where(hard, rng.beta(4, 2.5, n), rng.beta(8, 1.2, n)), 0, 1)
    ok_t = rng.random(n) < np.clip(1 - 0.5 * (1 - ct) - np.where(hard, 0.02, 0.0), 0, 1)
    ok_d = rng.random(n) < np.clip(1 - 0.45 * (1 - cd) - np.where(hard, 0.02, 0.0), 0, 1)
    return ct, ok_t, cd, ok_d


def test_f6_the_cascades_promise_holds_over_draws():
    alpha, delta, n_cert, draws = 0.05, 0.1, 1000, 150
    ct, ok_t, cd, ok_d = population(100_000, seed=11)
    y = np.zeros(len(ct), dtype=int)
    pT = np.where(ok_t, y, 1)  # the predictions are 0 when right, 1 when wrong: equality with y is the correctness
    pD = np.where(ok_d, y, 1)
    rng = np.random.default_rng(5)
    broken, certified_any = 0, 0
    for _ in range(draws):
        idx = rng.permutation(len(ct))
        cal, test = idx[:n_cert], idx[n_cert:]
        th = FT.cascade(ct[cal], pT[cal], y[cal], cd[cal], pD[cal], alpha, delta)
        by_table, certified, pred = FT.served(ct[test], pT[test], cd[test], pD[test], th)
        if not certified.any():
            continue
        certified_any += 1
        wrong = float((pred[certified] != y[test][certified]).mean())
        broken += wrong > alpha
    assert certified_any >= draws * 0.9  # the population has a certifiable share
    # delta allows 10%; the two arms at delta/2 each are conservative (about 2 to 3% of draws broke it when measured)
    assert broken / certified_any <= delta, (broken, certified_any)


# ---- F7: Lab 3's clinc150 cell -------------------------------------------------------------------------------------

DATA = Path(__file__).resolve().parents[1] / "data" / "fast_table"


@pytest.mark.parametrize("base", ["31b", "12b"])
def test_f7_the_clinc150_cell_of_the_study_is_reproduced(base):
    """CLINC150, 1,000 labels (draw 0), the stored decisio side of the study's cell: the table fitted by `fit` on the
    float16 embeddings, the cascade made on the 800 certification examples and applied to the 1,000 test examples,
    against the study's own numbers on the float32 embeddings (hand-rolled in the study, same procedure)."""
    z = np.load(DATA / "clinc150.npz")
    ref = json.loads((DATA / "expected.json").read_text())[base]
    k = int(z["k"])
    rec = FT.fit(z["X_teach"].astype(np.float32), z["y_teach"], k)
    assert rec["C"] == ref["table_fit"]["C"]
    Pc, Pt = FT.predict(rec, z["X_cert"].astype(np.float32)), FT.predict(rec, z["X_test"].astype(np.float32))
    cTc, pTc = FT.confidence(Pc)
    cTt, pTt = FT.confidence(Pt)
    th = FT.cascade(cTc, pTc, z["y_cert"], z[f"{base}_c_cert"], z[f"{base}_p_cert"], 0.05, 0.1)
    by_table, certified, pred = FT.served(cTt, pTt, z[f"{base}_c_test"], z[f"{base}_p_test"], th)
    want = ref["cascade"]
    assert th["table_threshold"] == pytest.approx(want["tau_T"], abs=1e-3)
    assert (th["decisio_threshold"] is None) == (want["tau_D"] is None)
    assert float(by_table.mean()) == pytest.approx(want["table_share"], abs=1e-3)
    assert float(certified.mean()) == pytest.approx(want["coverage"], abs=1e-3)
    wrong = float((pred[certified] != z["y_test"][certified]).mean())
    assert wrong == pytest.approx(want["error_certified"], abs=1e-3) and wrong <= 0.05
    assert float((pred == z["y_test"]).mean()) == pytest.approx(want["accuracy_all"], abs=1e-3)


def test_f8_the_docs_figures_are_the_summary_json():
    root = Path(__file__).resolve().parents[2]
    summ = json.loads((root / "runs" / "2026-10-30_fast-table" / "summary.json").read_text())
    text = (root / "docs" / "tasks.md").read_text()
    rows = summ["sets"]
    assert summ["passed"] == sum(r["passes"] for r in rows) == 4 and summ["of"] == len(rows) == 10
    outside = [r for r in rows if r["set"] not in summ["assistant_command_sets"]]
    assert summ["passed_without_assistant_command_sets"] == sum(r["passes"] for r in outside) == 1
    assert summ["of_without_assistant_command_sets"] == len(outside) == 7
    assert "passed the 25% guard on 4" in text and "it passed on 1 of 7" in text
    table = [line for line in text.splitlines() if line.startswith("| ") and " ms (" in line]
    assert len(table) == len(rows)
    for line, r in zip(table, rows):
        cells = [c.strip() for c in line.split("|")[1:-1]]
        assert cells[1] == f"{r['table_share']}%" and cells[4] == ("yes" if r["passes"] else "no"), line
        assert cells[3] == f"{r['decisio_ms']} / {r['cascade_ms']} ms ({r['saving']}%)", line
        assert cells[2] == ("-" if r["table_error"] is None else f"{r['table_error']}%"), line
