# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""From 50 labelled examples a task's calibration is its temperature and a per-option bias fitted together
(decisio.readout.calibration.fit_tb; docs/tasks.md).

E1  with b = 0 and T = T_q the method serves the untaught distribution at the question's temperature
E2  a registration of 49 examples is the gated bias exactly; one of 50 stores and serves the temperature-and-bias fit,
    counted on the request's examples before a held-out split
E3  a task that serves its head is unchanged: its calibration is never stale and the head answers
E4  a record of the gated bias for a task of 50 or more examples (made before) is stale: answered as without the task,
    with the reason and the register command, and not certified again; one of fewer examples is not
E5  serving a stored fit is apply_tb on the plain readout's probabilities, exactly; the fit is deterministic and its
    record is JSON

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
    task, fitted = store.register("t", examples, lambda pairs: list(p), None, temperature=TQ, **kw)
    return task, p, y


def test_e2_49_examples_are_the_gated_bias_and_50_the_new_fit():
    t49, p, y = _store_register(49)
    ref = C.fit_task_prior([log_probs(r) for r in p], y.tolist(), fingerprint="fp", task_id="t", K=4, temperature=TQ)
    assert "method" not in t49["calibration"]
    assert {k: v for k, v in t49["calibration"].items() if k != "inputs_sha256"} == ref
    t50, p, y = _store_register(50)
    cal = t50["calibration"]
    assert cal["method"] == C.TB_METHOD and cal["applied"] and cal["temperature"] == TQ and t50["n_request"] == 50
    T, b = C.fit_tb(p, y, TQ)
    assert cal["T"] == T and cal["bias"] == list(b)
    # the count is the request's: 40 fit examples of a request of 80 (a held-out split) are the new fit
    t40, _, _ = _store_register(40, n_request=80)
    assert t40["calibration"]["method"] == C.TB_METHOD and t40["n_request"] == 80


def test_e5_serving_a_stored_fit_is_apply_tb_exactly():
    t50, p, y = _store_register(50)
    cal = t50["calibration"]
    for r in p[:10]:
        assert np.array_equal(
            C.apply_task_prior(log_probs(r), cal, TQ), C.apply_tb(r[None, :], cal["T"], cal["bias"])[0]
        )
    # at another served temperature the record is not applied (as every calibration record)
    assert np.allclose(C.apply_task_prior(log_probs(p[0]), cal, 1.5), apply_temperature(p[0], 1.5))
    assert C.fit_tb(p, y, TQ)[0] == C.fit_tb(p, y, TQ)[0]
    json.dumps(cal, allow_nan=False)


def _so(hidden=None):
    from decisio.serve.systemone import SystemOne

    so = SystemOne(None, "m", temperature=TQ)
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
    assert (
        "gated bias" in note["why"] and "60 labelled examples" in note["why"] and "/v1/tasks" in note["register_again"]
    )
    so.task_precheck = lambda kind, task_id: None
    with pytest.raises(ValueError, match="register the task again"):
        so._recertify({**old, "certificate": {"id": "c1", "alpha": 0.05, "delta": 0.1}}, [], None)
    # a held-out split counts: 30 fit examples of a request of 60
    split = {"certificate": {"split": {"method": "holdout", "certify_examples": 30}}}
    assert so.calibration_stale(_task({"temperature": T}, 30, **split), "choice", 4)
    # fewer than 50 in the request: current, as before
    assert not so.calibration_stale(_task({"temperature": T}, 49), "choice", 4)
    # a record of the new fit is current
    new = _task({"temperature": T, "method": C.TB_METHOD, "T": T, "n_request": 60}, 60, n_request=60)
    assert not so.calibration_stale(new, "choice", 4) and so.calibration_note(new) is None


def test_e3_a_task_that_serves_its_head_is_unchanged():
    so = _so(hidden=object())
    T = so.temperature_of("choice", None, 4)
    head_task = _task({"temperature": T}, 200, head_applied=True)
    assert not so.calibration_stale(head_task, "choice", 4)  # the head answers; the calibration is not used
    # without a hidden-state reader the head is not served, and the old record is stale
    assert _so().calibration_stale(head_task, "choice", 4)
