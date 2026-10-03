# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The served default's temperatures (docs/handoffs/tasks.md; EVAL_CARD.md, "What was fitted on what").

One scalar T rescales a question's label log-probabilities, `softmax(log p / T)`: fitted by minimum log loss on the
served prompt's plain readouts of a private 1,400-item decision suite, never on a benchmark's items. Choice questions
have their own T, fitted the same way on the suite's 800 choice items; yes/no and score questions take the global one.
It never changes which option is the most probable. A registered task's own correction (per-task calibration or a
head) replaces it.

    p = apply_temperature(p, T)
"""

from __future__ import annotations

import numpy as np

SERVED_TEMPERATURE = 1.506  # the fit on all 1,400 suite items (served prompt since 2026-10-03; 1.307 before)
SERVED_CHOICE_TEMPERATURE = 1.370  # the fit on the suite's 800 choice items


def apply_temperature(p, T: float):
    """softmax(log p / T) in float64; T == 1 returns p unchanged (the same array values, bit for bit). The most
    probable option stays the most probable, and exactly tied options stay tied."""
    p = np.asarray(p, dtype=np.float64)
    if T == 1.0:
        return p
    with np.errstate(divide="ignore"):
        z = np.log(p) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
