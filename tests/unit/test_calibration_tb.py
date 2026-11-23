# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""From 50 labelled examples a task's calibration is its temperature and a per-option bias fitted together at four
bias-prior spreads, averaged by their Laplace evidence (decisio.readout.calibration.fit_c; docs/tasks.md).

E1  with b = 0 and T = T_q the method serves the untaught distribution at the question's temperature; the weights sum to
    1; with every weight on the temperature-only fit it serves that fit
E2  a registration of 49 examples is the gated bias exactly; one of 50 stores and serves the evidence-averaged fit,
    counted on the request's examples before a held-out split
E3  a task that serves its head is unchanged: its calibration is never stale and the head answers
E4  a record of the gated bias for a task of 50 or more examples (made before) is stale: answered as without the task,
    with the reason and the register command, and not certified again; one of fewer examples is not
E5  serving a stored fit is apply_c on the plain readout's probabilities, exactly; the fit is deterministic and its
    record is JSON
E7  label review calibrates its folds by the registration's rule
E6  the scope: the 12B profiles (bf16, FP8) fit it for choice and yes/no, and for score questions of 3 to 4 levels and
    of 10 or more; a score question of 5 to 9 levels, and every task on a base without large_calibration_types, keeps
    the gated bias at any count, and its records are never stale for this rule
E8  in_large_scope: a type covers every option count, a (type, lowest, highest) range its own counts

  uv run pytest -q tests/unit/test_calibration_tb.py
"""

import json
import types

import numpy as np
import pytest

from decisio.readout import calibration as C
from decisio.readout.debias import log_probs
from decisio.serve.temperature import apply_temperature

TQ = 3.592
TWELVE_B = ("choice", "noul", ("score", 3, 4), ("score", 10, None))  # Family.large_calibration_types of the 12B


def _readout(n, K=4, seed=0):
    """Overconfident plain probabilities with a per-option shift the model does not see, and labels from the truth."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.2, (n, K))
    truth = np.exp(z + np.linspace(-0.6, 0.6, K))
    truth /= truth.sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=p) for p in truth])
    p = np.exp(TQ * z)
    return p / p.sum(1, keepdims=True), y


def test_e1_no_bias_at_the_question_temperature_is_the_untaught_answer():
    p, _ = _readout(40)
    got = C.apply_tb(p, TQ, np.zeros(p.shape[1]))
    want = np.asarray([apply_temperature(r, TQ) for r in p])
    assert np.allclose(got, want, rtol=0, atol=1e-12)
    p, y = _readout(80)
    rec = C.fit_c(p, y, TQ)
    assert len(rec) == len(C.C_GRID) and abs(sum(w for w, _, _ in rec) - 1.0) < 1e-12
    assert np.array_equal(rec[0][2], np.zeros(4))  # the grid's first point is the temperature alone
    only_t = [(1.0, rec[0][1], rec[0][2])] + [(0.0, T, b) for _, T, b in rec[1:]]
    assert np.allclose(C.apply_c(p, only_t), C.apply_tb(p, rec[0][1], np.zeros(4)), rtol=0, atol=1e-15)


def _store_register(n, **kw):
    from decisio.serve.tasks import TaskStore

    p, y = _readout(n)
    keys = ["a", "b", "c", "d"]
    q = types.SimpleNamespace(type="choice", criteria={k: None for k in keys})
    from decisio.serve.systemone import SystemOneRequest

    q = SystemOneRequest.model_validate(
        {"state": "", "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": dict.fromkeys(keys)}}}
    ).questions["q"]
    examples = [(q, f"s{i}", keys[y[i]]) for i in range(n)]
    store = TaskStore("fp")
    kw = {"large_types": TWELVE_B, **kw}
    task, fitted = store.register("t", examples, lambda pairs: list(p), None, temperature=TQ, **kw)
    return task, p, y


def test_e2_49_examples_are_the_gated_bias_and_50_the_new_fit():
    t49, p, y = _store_register(49)
    ref = C.fit_task_prior([log_probs(r) for r in p], y.tolist(), fingerprint="fp", task_id="t", K=4, temperature=TQ)
    assert "method" not in t49["calibration"]
    assert {k: v for k, v in t49["calibration"].items() if k != "inputs_sha256"} == ref
    t50, p, y = _store_register(50)
    cal = t50["calibration"]
    assert cal["method"] == C.C_METHOD and cal["applied"] and cal["temperature"] == TQ and t50["n_request"] == 50
    rec = C.fit_c(p, y, TQ)
    assert [q["sigma"] for q in cal["points"]] == list(C.C_GRID)
    assert [(q["w"], q["T"], q["bias"]) for q in cal["points"]] == [(w, T, list(b)) for w, T, b in rec]
    # the count is the request's: 40 fit examples of a request of 80 (a held-out split) are the new fit
    t40, _, _ = _store_register(40, n_request=80)
    assert t40["calibration"]["method"] == C.C_METHOD and t40["n_request"] == 80


def test_e5_serving_a_stored_fit_is_apply_c_exactly():
    t50, p, y = _store_register(50)
    cal = t50["calibration"]
    for r in p[:10]:
        assert np.array_equal(C.apply_task_prior(log_probs(r), cal, TQ), C.apply_c(r[None, :], cal)[0])
    # at another served temperature the record is not applied (as every calibration record)
    assert np.allclose(C.apply_task_prior(log_probs(p[0]), cal, 1.5), apply_temperature(p[0], 1.5))
    assert C.fit_c(p, y, TQ)[1][1] == C.fit_c(p, y, TQ)[1][1]
    json.dumps(cal, allow_nan=False)


def _so(hidden=None, family=None):
    from decisio.families import GEMMA4
    from decisio.serve.systemone import SystemOne

    so = SystemOne(None, "m", temperature=TQ)
    so.engine = types.SimpleNamespace(family=GEMMA4 if family is None else family)
    so.hidden_engine = hidden
    return so


def _task(cal, n_examples, head_applied=False, **extra):
    return {
        "id": "t",
        "type": "choice",
        "options": ["a", "b", "c", "d"],
        "calibration": {"applied": True, "K": 4, "bias": [0.1, 0.0, -0.1, 0.0], **cal},
        "head": {"applied": head_applied},
        "n_examples": n_examples,
        **extra,
    }


def test_e4_an_old_rule_record_of_50_examples_is_stale():
    so = _so()
    T = so.temperature_of("choice", None, 4)
    old = _task({"temperature": T}, 60)  # the gated bias, fitted at this temperature, for 60 examples
    assert so.calibration_stale(old, "choice", 4)
    note = so.calibration_note(old)
    assert "earlier rule" in note["why"] and "60 labelled examples" in note["why"]
    assert "/v1/tasks" in note["register_again"]
    so.task_precheck = lambda kind, task_id: None
    with pytest.raises(ValueError, match="register the task again"):
        so._recertify({**old, "certificate": {"id": "c1", "alpha": 0.05, "delta": 0.1}}, [], None)
    # a held-out split counts: 30 fit examples of a request of 60
    split = {"certificate": {"split": {"method": "holdout", "certify_examples": 30}}}
    assert so.calibration_stale(_task({"temperature": T}, 30, **split), "choice", 4)
    # fewer than 50 in the request: current, as before
    assert not so.calibration_stale(_task({"temperature": T}, 49), "choice", 4)
    # a record of the new fit is current
    new = _task({"temperature": T, "method": C.CURRENT_METHOD, "n_request": 60}, 60, n_request=60)
    assert not so.calibration_stale(new, "choice", 4) and so.calibration_note(new) is None


def test_e3_a_task_that_serves_its_head_is_unchanged():
    so = _so(hidden=object())
    T = so.temperature_of("choice", None, 4)
    head_task = _task({"temperature": T}, 200, head_applied=True)
    assert not so.calibration_stale(head_task, "choice", 4)  # the head answers; the calibration is not used
    # without a hidden-state reader the head is not served, and the old record is stale
    assert _so().calibration_stale(head_task, "choice", 4)


def test_e6_the_12b_scope_is_choice_yes_no_and_the_measured_score_cells():
    from decisio.families import BASES, GEMMA4, GEMMA4_FP8
    from decisio.serve.systemone import SystemOneRequest
    from decisio.serve.tasks import TaskStore

    assert GEMMA4.large_calibration_types == TWELVE_B and GEMMA4_FP8.large_calibration_types == TWELVE_B
    assert all(not f.large_calibration_types for k, f in BASES.items() if k not in ("gemma-4-12b", "gemma-4-12b-fp8"))
    other, p, y = _store_register(60, large_types=())  # another base: the gated bias at 60
    assert "method" not in other["calibration"]

    def score_task(levels: int):
        p, y = _readout(60, K=levels)
        q = SystemOneRequest.model_validate(
            {"state": "", "questions": {"q": {"type": "score", "instructions": "How?",
                                              "criteria": [f"l{i}" for i in range(levels)]}}}
        ).questions["q"]  # fmt: skip
        task, _ = TaskStore("fp").register(
            "s", [(q, f"s{i}", int(y[i])) for i in range(60)], lambda pairs: list(p), None, temperature=TQ,
            large_types=TWELVE_B,
        )  # fmt: skip
        return task

    for levels in (3, 4, 10, 11):  # the measured cells that passed: the evidence-averaged fit
        assert score_task(levels)["calibration"]["method"] == C.CURRENT_METHOD, levels
    for levels in (5, 6, 7, 9):  # 5 to 6 failed its check and 7 to 9 was not measured: the gated bias
        assert "method" not in score_task(levels)["calibration"], levels
    # the stale rule follows the scope: a gated record of 60 examples is stale for a score task of 4 levels on the 12B,
    # current on another base and for a score task of 6 levels
    T = _so().temperature_of("choice", None, 4)
    old = _task({"temperature": T}, 60)
    assert _so().calibration_stale(old, "choice", 4)
    assert not _so(family=BASES["gemma-4-31b"]).calibration_stale(old, "choice", 4)
    four = {**old, "type": "score"}
    assert _so().calibration_stale(four, "score", 4)
    six = {
        **old,
        "type": "score",
        "options": [f"l{i}" for i in range(6)],
        "calibration": {**old["calibration"], "K": 6},
    }
    T6 = _so().temperature_of("score", None, 6)
    six["calibration"]["temperature"] = T6
    assert not _so().calibration_stale(six, "score", 6)


def test_e8_in_large_scope():
    assert C.in_large_scope(TWELVE_B, "choice", 2) and C.in_large_scope(TWELVE_B, "choice", 77)
    assert C.in_large_scope(TWELVE_B, "noul", 2)
    assert [k for k in range(2, 14) if C.in_large_scope(TWELVE_B, "score", k)] == [3, 4, 10, 11, 12, 13]
    assert not C.in_large_scope((), "choice", 4) and not C.in_large_scope(None, "choice", 4)
    assert not C.in_large_scope((("score", 3, 4),), "choice", 3)


def test_e7_label_review_calibrates_its_folds_as_registration_does(monkeypatch):
    """E7 the label review's folds use the registration's rule: the evidence-averaged fit where the registration fits
    it, the gated bias otherwise."""
    from decisio.readout import label_review

    p, y = _readout(60)
    lps = [log_probs(r) for r in p]
    seen = []
    for name in ("fit_large_prior", "fit_task_prior"):
        real = getattr(C, name)
        monkeypatch.setattr(C, name, lambda *a, _n=name, _f=real, **k: seen.append(_n) or _f(*a, **k))
    train, serve = np.arange(40), np.arange(40, 60)
    label_review.fit_serve(lps, y.tolist(), ["a", "b", "c", "d"], TQ, None, train, serve, 10, large=True)
    label_review.fit_serve(lps, y.tolist(), ["a", "b", "c", "d"], TQ, None, train, serve, 10)
    assert seen == ["fit_large_prior", "fit_task_prior"]
