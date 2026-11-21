# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Label review: the labelled examples of a task whose label looks wrong, from out-of-fold answers of the task's own
pipeline (docs/tasks.md, "Label review").

All the task's labelled examples are split into `FOLDS` folds, deterministically from the task id. Each fold is answered
by the pipeline fitted on the other folds exactly as the task serves: its intent head where one is fitted and kept,
else its calibration where kept (on top of the temperature, decisio.readout.calibration), else the question as it is
served without the task (at its temperature). An example is flagged when its out-of-fold probability for its given
label is at most `P_LABEL_MAX` and another option's is at least `P_OTHER_MIN`.

A review aid only: no label is changed or dropped, and the task is fitted and certified on the labels as given.

    review(lps, labels, options, temperature, readout, task_id)  ->  {"reviewed", "examples", "flags"}
"""

from __future__ import annotations

import hashlib

import numpy as np

MIN_EXAMPLES = 20
FOLDS = 5
P_LABEL_MAX = 0.02
P_OTHER_MIN = 0.9
# the bases the review runs on: Gemma 4 12B in every format (bf16, FP8, and on a Mac, whose builds load under the bf16
# profile), the base and the tuned 12B repositories, one served pipeline; a registration on any other base says the
# review is not available there
CHECKED_BASES = frozenset({"gemma-4-12b", "gemma-4-12b-fp8"})


def fold_seed(task_id: str) -> int:
    """The folds' seed: the first 8 bytes of sha256(task id), so the same request gives the same folds."""
    return int.from_bytes(hashlib.sha256(str(task_id).encode()).digest()[:8], "big")


def folds(n: int, rng) -> list[np.ndarray]:
    """`FOLDS` parts (fewer for fewer examples) of a random permutation of range(n), as numpy's array_split cuts it."""
    return np.array_split(rng.permutation(n), min(FOLDS, n))


def fit_serve(lps, labels, options, temperature, readout, train, serve, min_options_for_head):
    """The served distributions of the `serve` examples under the task's pipeline fitted on the `train` examples.
    `lps`: per example, the K label log-scores of the plain readout (temperature 1); `readout`: per example, the
    hidden-state readout (lp, h), or None when no head can be fitted (no reader, too few options or examples)."""
    from decisio.readout import calibration, intent_head
    from decisio.serve.temperature import apply_temperature

    K = len(options)
    y = np.asarray(labels)
    cal = calibration.fit_task_prior([lps[i] for i in train], y[train].tolist(), K=K, temperature=temperature)
    head = None
    if (
        readout is not None
        and K >= min_options_for_head
        and np.bincount(y[train], minlength=K).min() >= intent_head.MIN_PER_OPTION
    ):
        head = intent_head.fit_intent_head(
            [readout[i][0] for i in train], [readout[i][1] for i in train], y[train].tolist(), options
        )
    if head is not None and head.get("applied"):
        return np.asarray([intent_head.apply_intent_head(readout[i][0], readout[i][1], head, options) for i in serve])
    if cal.get("applied"):
        return np.asarray([calibration.apply_task_prior(lps[i], cal, temperature) for i in serve])
    return np.asarray([apply_temperature(np.exp(np.asarray(lps[i], dtype=np.float64)), temperature) for i in serve])


def out_of_fold(lps, labels, options, temperature, readout, parts, min_options_for_head):
    """n x K: each example's distribution from the pipeline fitted on the folds it is not in."""
    P = np.zeros((len(labels), len(options)))
    for k, test in enumerate(parts):
        train = np.concatenate([parts[j] for j in range(len(parts)) if j != k])
        P[test] = fit_serve(lps, labels, options, temperature, readout, train, test, min_options_for_head)
    return P


def flag_mask(P, labels, p_label_max=P_LABEL_MAX, p_other_min=P_OTHER_MIN):
    """(flagged, P(label), the most likely other option, its P) per example."""
    y = np.asarray(labels)
    rows = np.arange(len(y))
    p_label = P[rows, y]
    Q = P.copy()
    Q[rows, y] = -1
    other = Q.argmax(1)
    p_other = P[rows, other]
    return (p_label <= p_label_max) & (p_other >= p_other_min), p_label, other, p_other


def too_few(n: int) -> dict:
    return {
        "reviewed": False,
        "examples": n,
        "reason": f"{n} labelled examples; the review needs at least {MIN_EXAMPLES}",
    }


def review(lps, labels, options, temperature, readout, task_id, positions=None, min_options_for_head=10, rng=None):
    """The label review of a task's labelled examples. `positions`: per example, where it was in the request (a
    dict, e.g. {"list": "examples", "index": 3}); its index in `labels` when absent. `rng`: the folds' generator
    (numpy default_rng of `fold_seed(task_id)` when absent)."""
    n = len(labels)
    if n < MIN_EXAMPLES:
        return too_few(n)
    rng = np.random.default_rng(fold_seed(task_id)) if rng is None else rng
    parts = folds(n, rng)
    P = out_of_fold(lps, labels, options, temperature, readout, parts, min_options_for_head)
    flagged, p_label, other, p_other = flag_mask(P, labels)
    flags = []
    for i in np.flatnonzero(flagged):
        where = positions[i] if positions is not None else {"index": int(i)}
        flags.append(
            {
                **where,
                "label": options[int(labels[i])],
                "suggested": options[int(other[i])],
                "p_label": round(float(p_label[i]), 4),
                "p_suggested": round(float(p_other[i]), 4),
            }
        )
    return {
        "reviewed": True,
        "examples": n,
        "folds": len(parts),
        "rule": {"p_label_max": P_LABEL_MAX, "p_suggested_min": P_OTHER_MIN},
        "flags": flags,
    }
