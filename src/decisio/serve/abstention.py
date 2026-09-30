# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Abstention by a per-task threshold on the abstain option's probability (docs/handoffs/tasks.md).

A task's questions carry one abstain option: declared by the customer among their own options (the Decision Index's
CLINC150+OOS "out of scope" key), or appended by the server (imajev's "can't tell"; answered through imajev's
`unknown_probability` / `abstained` fields). Instead of taking the argmax, the server abstains when that option's
probability exceeds a threshold fitted from the customer's labelled examples, and otherwise answers the best of the
other options. The probabilities themselves are reported unchanged; only the declared choice follows the threshold.

The threshold is fitted like per-task calibration (decisio.readout.calibration): from about 20
labelled examples that include some unanswerable ones, with leave-one-out cross-validation and an acceptance rule that
keeps the plain behaviour when the examples do not justify a threshold.

    cfg = fit(p_abs, unanswerable, answer_ok, plain_ok)          # per example; stored with the task
    abstain, answer_index = decide(p, abstain_index, cfg)          # per question
"""

from __future__ import annotations

import math

import numpy as np

from decisio.names import record_format

FORMAT = record_format("abstention")
MIN_EXAMPLES = 10  # fewer examples: no threshold
MIN_EACH = 2  # at least this many unanswerable and answerable examples
EPS = 1e-9


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def candidates(p_abs) -> list[float]:
    """Thresholds between consecutive distinct probabilities, at the midpoint in log-odds; below the smallest (abstain
    on every example) and 1.0 (never abstain)."""
    v = sorted(set(float(x) for x in p_abs))
    out = [_sigmoid((_logit(a) + _logit(b)) / 2) for a, b in zip(v, v[1:])]
    if v:
        out.append(_sigmoid(_logit(v[0]) - 1.0))
    out.append(1.0)
    return sorted(out)


def outcome(p_abs: float, unanswerable: bool, answer_ok: bool, t: float) -> tuple[bool, bool]:
    """(right, false abstention) for one example under threshold t: abstaining is right on an unanswerable example; an
    answer is right when the best of the other options is the gold one."""
    abstain = p_abs > t
    if unanswerable:
        return abstain, False
    return (not abstain) and answer_ok, abstain


def best_threshold(p_abs, unanswerable, answer_ok) -> float:
    """The candidate that gets the most examples right; ties go to the largest threshold (the fewest abstentions)."""
    best, best_t = -1, 1.0
    for t in candidates(p_abs):
        right = sum(outcome(p, u, ok, t)[0] for p, u, ok in zip(p_abs, unanswerable, answer_ok))
        if right > best or (right == best and t > best_t):
            best, best_t = right, t
    return best_t


def fit(p_abs, unanswerable, answer_ok, plain_right, plain_false) -> dict:
    """A task's abstention config from its labelled examples.

    p_abs: the abstain option's probability per example; unanswerable: the example's reference is "abstain";
    answer_ok: the best of the other options is the gold answer (answerable examples); plain_right / plain_false: the
    plain behaviour's outcome per example (the argmax over the task's options when the option is the customer's; the
    question without the option when the server appends it).

    Acceptance rule (fixed in the experiment's PREREG.md before any measurement): at least MIN_EXAMPLES examples with at
    least MIN_EACH unanswerable and MIN_EACH answerable ones; leave-one-out, the threshold rule gets at least one more
    example right than the plain behaviour and makes no more false abstentions. Otherwise the task keeps the plain
    behaviour (`applied` false, no threshold)."""
    n = len(p_abs)
    n_u = int(sum(bool(u) for u in unanswerable))
    base = {"format": FORMAT, "n": n, "n_unanswerable": n_u, "applied": False, "threshold": None}
    if n < MIN_EXAMPLES or n_u < MIN_EACH or n - n_u < MIN_EACH:
        return {
            **base,
            "reason": f"needs at least {MIN_EXAMPLES} examples with {MIN_EACH} unanswerable and "
            f"{MIN_EACH} answerable ones",
        }
    loo_right = loo_false = 0
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        t = best_threshold([p_abs[j] for j in keep], [unanswerable[j] for j in keep], [answer_ok[j] for j in keep])
        r, f = outcome(p_abs[i], unanswerable[i], answer_ok[i], t)
        loo_right += r
        loo_false += f
    plain_r, plain_f = int(sum(plain_right)), int(sum(plain_false))
    t = best_threshold(p_abs, unanswerable, answer_ok)
    evidence = {
        "loo_right_threshold": loo_right,
        "loo_false_threshold": loo_false,
        "right_plain": plain_r,
        "false_plain": plain_f,
    }
    if loo_right >= plain_r + 1 and loo_false <= plain_f:
        return {
            **base,
            **evidence,
            "applied": True,
            "threshold": t,
            "reason": f"leave-one-out {loo_right} of {n} right against {plain_r} for the plain behaviour",
        }
    return {
        **base,
        **evidence,
        "fitted_threshold": t,
        "reason": f"leave-one-out {loo_right} right, {loo_false} false abstentions against the plain behaviour's "
        f"{plain_r}, {plain_f}: kept plain",
    }


def decide(p, abstain_index: int, cfg: dict | None, keys=None) -> tuple[bool, int]:
    """(abstain, answer index) for one question's distribution `p` (the abstain option at `abstain_index`).

    With an applied config: abstain when p[abstain_index] > threshold; otherwise the best of the other options.
    Without: the argmax over every option (abstaining when that is the abstain option).
    Exact ties go to the option whose key sorts first when `keys` is given (the served rule,
    `decisio.serve.systemone.top_index`), else to the first in order."""
    p = np.asarray(p, dtype=np.float64)
    others = [i for i in range(len(p)) if i != abstain_index]
    if keys is None:
        best_other = max(others, key=lambda i: p[i])
        top = int(p.argmax())
    else:
        from decisio.serve.systemone import top_index

        best_other = top_index(keys, p, others)
        top = top_index(keys, p)
    if cfg and cfg.get("applied"):
        return bool(p[abstain_index] > cfg["threshold"]), best_other
    return top == abstain_index, best_other
