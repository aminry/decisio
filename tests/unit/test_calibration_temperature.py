# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A task's calibration works on top of the server's temperature (decisio.readout.calibration; docs/tasks.md).

C1  the fit at a temperature is the fit on the tempered log-scores, and the record stores the temperature
C2  serving: softmax(log p / T - bias); a zero bias serves exactly the untaught question at T; a record fitted at
    another temperature (before 0.52.0, at 1) is never applied
C3  on an overconfident readout that the temperature calibrates, a kept calibration fitted at the temperature is not
    worse calibrated than the untaught question, where the bias fitted at temperature 1 was
C4  the server: such a record is stale, the task listing says why and how to register it again, and the task is not
    certified again on it

  uv run pytest -q tests/unit/test_calibration_temperature.py
"""

import numpy as np
import pytest

from decisio.readout import calibration
from decisio.serve.temperature import apply_temperature


def _readout(n, K=5, T0=3.0, seed=0):
    """Overconfident label log-scores (the true distribution sharpened by T0) with a per-option prior shift the model
    does not see, and labels drawn from the truth."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.2, (n, K))
    shift = np.linspace(-0.6, 0.6, K)
    truth = np.exp(z + shift)
    truth /= truth.sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=p) for p in truth])
    lp = T0 * z
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    return list(lp), y


def _ece(P, y, bins=10):
    conf, top = P.max(1), P.argmax(1)
    edges = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(
        abs((top[edges == b] == y[edges == b]).mean() - conf[edges == b].mean()) * (edges == b).mean()
        for b in range(bins)
        if (edges == b).any()
    )


def test_c1_the_fit_at_a_temperature_is_the_fit_on_the_tempered_scores():
    lp, y = _readout(200)
    at = calibration.fit_task_prior(lp, y, K=5, temperature=3.0)
    ref = calibration.fit_task_prior([x / 3.0 for x in lp], y, K=5)
    assert at["temperature"] == 3.0 and ref["temperature"] == 1.0
    assert {k: v for k, v in at.items() if k != "temperature"} == {k: v for k, v in ref.items() if k != "temperature"}


def test_c2_serving_on_top_of_the_temperature():
    lp = np.log(np.array([0.7, 0.2, 0.1]))
    zero = {"applied": True, "K": 3, "bias": [0.0, 0.0, 0.0], "temperature": 2.5}
    assert np.allclose(calibration.apply_task_prior(lp, zero, 2.5), apply_temperature(np.exp(lp), 2.5), atol=1e-12)
    biased = {**zero, "bias": [0.4, 0.0, -0.4]}
    z = lp / 2.5 - np.array([0.4, 0.0, -0.4])
    assert np.allclose(calibration.apply_task_prior(lp, biased, 2.5), np.exp(z) / np.exp(z).sum(), atol=1e-12)
    # fitted at 1 (a record made before 0.52.0, which has no temperature): not applied at 2.5
    old = {k: v for k, v in biased.items() if k != "temperature"}
    assert not calibration.is_current(old, 2.5) and calibration.is_current(old, 1.0)
    assert np.allclose(calibration.apply_task_prior(lp, old, 2.5), apply_temperature(np.exp(lp), 2.5), atol=1e-12)


def test_c3_a_kept_calibration_is_not_worse_calibrated_than_the_untaught_question():
    lp_fit, y_fit = _readout(300, seed=1)
    lp_test, y_test = _readout(4000, seed=2)
    T = 3.0
    untaught = np.array([apply_temperature(np.exp(x), T) for x in lp_test])
    now = calibration.fit_task_prior(lp_fit, y_fit, K=5, temperature=T)
    assert now["applied"]
    served_now = np.array([calibration.apply_task_prior(x, now, T) for x in lp_test])
    # what 0.51 served once it kept a bias: the bias fitted on the untempered scores, served at temperature 1
    b1 = calibration._fit(lp_fit, np.asarray(y_fit), 5, np.arange(len(y_fit)))
    before = {"applied": True, "K": 5, "bias": list(b1), "temperature": 1.0}
    served_before = np.array([calibration.apply_task_prior(x, before, 1.0) for x in lp_test])
    assert _ece(served_now, y_test) <= _ece(untaught, y_test) + 0.01
    assert _ece(served_before, y_test) > _ece(untaught, y_test) + 0.05  # what 0.51 served: overconfident


def test_c4_the_server_names_a_stale_calibration_and_does_not_certify_on_it():
    from decisio.serve.systemone import SystemOne

    so = SystemOne(None, "m", temperature=3.592)
    t = {
        "id": "intents",
        "type": "choice",
        "options": ["a", "b", "c"],
        "calibration": {"applied": True, "K": 3, "bias": [0.1, 0.0, -0.1]},  # no temperature: fitted at 1
        "certificate": {"id": "c1", "alpha": 0.05, "delta": 0.1},
    }
    assert so.calibration_stale(t, "choice", 3)
    note = so.calibration_note(t)
    assert note["stale"] and "registered again" in note["why"] and "/v1/tasks" in note["register_again"]
    assert not so.calibration_stale({**t, "calibration": {**t["calibration"], "temperature": 3.592}}, "choice", 3)
    assert so.calibration_note({**t, "calibration": {**t["calibration"], "applied": False}}) is None
    so.task_precheck = lambda kind, task_id: None
    with pytest.raises(ValueError, match="register the task again"):
        so._recertify(t, [], None)
