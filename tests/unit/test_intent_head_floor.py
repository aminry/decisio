# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The intent head's floor (`intent_head.MIN_PER_OPTION`): the acceptance gate at the floor, on the stored readouts.

F1  at MIN_PER_OPTION examples per intent on the stored BANKING77 and CLINC150 readouts (tests/data/intent_readouts),
    the head is kept and beats the plain readout on the intent's other, held-out examples by at least 2 points, in two
    draws each (the gate that chose 5 and 10 per intent, run again at the floor)

  uv run pytest -q tests/unit/test_intent_head_floor.py
"""

from pathlib import Path

import numpy as np
import pytest

from decisio.readout.intent_head import MIN_PER_OPTION, apply_intent_head, fit_intent_head

READOUTS = Path(__file__).resolve().parents[1] / "data" / "intent_readouts"


@pytest.mark.parametrize("name", ["banking77", "clinc150"])
@pytest.mark.parametrize("draw", [0, 1])
def test_f1_the_head_at_the_floor_beats_the_plain_readout_on_held_out_examples(name, draw):
    z = np.load(READOUTS / f"{name}_examples.npz")
    h, lp, y = z["h"], z["lp"], z["y"]
    options = [str(k) for k in range(lp.shape[1])]
    rng = np.random.default_rng(draw)
    fit, held = [], []
    for k in range(len(options)):
        idx = rng.permutation(np.where(y == k)[0])
        fit += list(idx[:MIN_PER_OPTION])
        held += list(idx[MIN_PER_OPTION:])
    fit, held = np.array(fit), np.array(held)
    rec = fit_intent_head(lp[fit], h[fit], y[fit], options)
    assert rec["applied"], rec["reason"]
    plain = float((lp[held].argmax(1) == y[held]).mean())
    head = float(np.mean([apply_intent_head(lp[i], h[i], rec, options).argmax() == y[i] for i in held]))
    assert head >= plain + 0.02, (name, draw, plain, head)
