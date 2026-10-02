# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Per-task calibration and the per-intent head on `/v1/systemone`: one registration per task, fitted from the
customer's labelled examples exactly as served, applied at serving time with the reference implementations unchanged
(docs/handoffs/tasks.md).

A task is identified by its question's option list: the question type and its options (keys and descriptions) in the
served order, plus the instructions for yes/no and score questions, whose options do not identify a task. Any change
to the list is a new task, which is re-registered.

  calibration   `decisio.readout.calibration`: `fit_task_prior` on the examples' K label log-scores from the served
                readout; served as `apply_task_prior` (softmax(lp - bias)).
  intent head   `decisio.readout.intent_head`: `fit_intent_head` on the examples' (lp, h) from the hidden-state readout
                (decisio.serve.hidden_engine), for a question of at least 10 options (an intent set, not a yes/no or
                multiple-choice question) with at least 5 examples of every option; served as `apply_intent_head`
                (softmax(lp + h @ A + c)). Where a head applies, calibration is not stacked on it (its c covers what the
                prior does); the head is refused with two-order averaging (it is bound to one option order).

    store = TaskStore(fingerprint="...")
    task = store.register(task_id, examples, score_fn, hidden_fn)     # examples: [(question, state, gold key)]
"""

from __future__ import annotations

import base64
import hashlib
import json

import numpy as np

from decisio.names import check_format, record_format, same_fingerprint
from decisio.readout import calibration, intent_head
from decisio.readout.debias import log_probs

FORMAT = record_format("task")
MIN_OPTIONS_FOR_HEAD = 10  # heads on intent sets only: smaller option lists get calibration alone


def task_key(q, render_text, described: bool = False) -> str:
    """The identity of a wire question's task: type and options in order, plus instructions for yes/no and score.
    `described`: the server's --describe-options rule changes how this question is shown (its options have
    descriptions), so a task fitted under the other rendering has another key and is not applied. A question the rule
    leaves alone keeps the same key under either setting."""
    if q.type == "choice":
        spec = {"type": "choice", "options": [[k, render_text(d)] for k, d in q.criteria.items()]}
    elif q.type == "score":
        spec = {
            "type": "score",
            "options": [render_text(c) for c in q.criteria],
            "instructions": render_text(q.instructions),
        }
    else:
        crit = None if q.criteria is None else [render_text(q.criteria.true), render_text(q.criteria.false)]
        spec = {"type": "noul", "criteria": crit, "instructions": render_text(q.instructions)}
    if described:
        spec["rendering"] = "describe_options"
    return hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def answer_keys(q) -> list[str]:
    """The task's answers in served order, as a customer names them."""
    if q.type == "choice":
        return list(q.criteria)
    if q.type == "score":
        return [str(i) for i in range(len(q.criteria))]
    return ["yes", "no"]


def gold_index(q, gold) -> int:
    keys = answer_keys(q)
    g = {True: "yes", False: "no"}.get(gold, gold) if q.type == "noul" else gold
    g = str(g)
    if g not in keys:
        raise ValueError(f"answer {gold!r} is not one of the task's answers {keys[:8]}...")
    return keys.index(g)


def _pack(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return {"dtype": "float32", "shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def _unpack(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.float32).reshape(d["shape"]).copy()


class TaskStore:
    """Registered tasks keyed by `task_key`; each holds its calibration prior and its head record."""

    def __init__(self, fingerprint: str):
        self.fingerprint = fingerprint
        self.by_key: dict[str, dict] = {}

    def lookup(self, key):
        t = self.by_key.get(key)
        return t if t is not None and same_fingerprint(t["fingerprint"], self.fingerprint) else None

    def register(self, task_id, examples, score_fn, hidden_fn=None, key_fn=None):
        """examples: [(wire question, state, gold)], all of one task; score_fn([(state, q)]) -> served label
        probabilities per example; hidden_fn([(state, q)]) -> [(lp, h)] from the hidden-state engine, or None when the
        server has none. Returns the stored task."""
        if not examples:
            raise ValueError("no examples")
        key_fn = key_fn or (lambda q: task_key(q, _render))  # the server passes its rendering's key
        keys = {key_fn(q) for q, _, _ in examples}
        if len(keys) != 1:
            raise ValueError("the examples must share one question: the same type and option list in the same order")
        q0 = examples[0][0]
        options = answer_keys(q0)
        labels = [gold_index(q, g) for q, _, g in examples]
        pairs = [(s, q) for q, s, _ in examples]
        lps = [log_probs(p) for p in score_fn(pairs)]
        prior = calibration.fit_task_prior(lps, labels, fingerprint=self.fingerprint, task_id=task_id, K=len(options))
        counts = np.bincount(labels, minlength=len(options))
        if hidden_fn is None:
            head = {
                "format": intent_head.FORMAT,
                "applied": False,
                "K": len(options),
                "options": options,
                "options_sha256": intent_head.options_digest(options),
                "reason": "this server has no hidden-state engine (--head-engine)",
            }
            readout = None
        elif len(options) < MIN_OPTIONS_FOR_HEAD:
            head = {
                "format": intent_head.FORMAT,
                "applied": False,
                "K": len(options),
                "options": options,
                "options_sha256": intent_head.options_digest(options),
                "reason": f"the head is fitted for questions of at least {MIN_OPTIONS_FOR_HEAD} options "
                f"(intent sets); this one has {len(options)}: calibration only",
            }
            readout = None
        elif counts.min() < intent_head.MIN_PER_OPTION:
            head = {
                "format": intent_head.FORMAT,
                "applied": False,
                "K": len(options),
                "options": options,
                "options_sha256": intent_head.options_digest(options),
                "min_per_option": int(counts.min()),
                "reason": f"every option needs at least {intent_head.MIN_PER_OPTION} labelled examples; "
                f"{int((counts < intent_head.MIN_PER_OPTION).sum())} of {len(options)} have fewer",
            }
            readout = None
        else:
            readout = hidden_fn(pairs)
            head = intent_head.fit_intent_head(
                [lp for lp, _ in readout],
                [h for _, h in readout],
                labels,
                options,
                fingerprint=self.fingerprint,
                task_id=task_id,
            )
        task = {
            "format": FORMAT,
            "id": task_id,
            "key": keys.pop(),
            "type": q0.type,
            "options": options,
            "fingerprint": self.fingerprint,
            "calibration": prior,
            "head": head,
            "n_examples": len(examples),
            "per_option_min": int(counts.min()),
        }
        self.remove(task_id)  # a re-registration replaces the task, whatever its list
        self.by_key[task["key"]] = task
        return task, {"lps": lps, "labels": labels, "readout": readout}

    def remove(self, task_id) -> bool:
        keys = [k for k, t in self.by_key.items() if t["id"] == task_id]
        for k in keys:
            del self.by_key[k]
        return bool(keys)

    # ---- persistence: JSON with the head's arrays as base64 float32 ---------------------------------------------

    @staticmethod
    def public(task, full=False):
        head = dict(task["head"])
        for k in ("A", "c"):
            if k in head:
                head[k] = _pack(head[k]) if full else {"shape": list(np.asarray(head[k]).shape)}
        return {**task, "head": head}

    def load(self, tasks):
        """Load task records as `public(task, full=True)` writes them; records written before the rename (the `rlcd-*/1`
        formats) load unchanged. A record of another kind is refused before anything is stored."""
        for t in tasks:
            what = f"task {t.get('id')!r}" if isinstance(t, dict) else "task"
            check_format(t, "task", what)
            check_format(t.get("calibration"), "task-prior", f"{what}, calibration")
            check_format(t.get("head"), "intent-head", f"{what}, head")
        for t in tasks:
            head = dict(t["head"])
            for k in ("A", "c"):
                if isinstance(head.get(k), dict) and "b64" in head[k]:
                    head[k] = _unpack(head[k])
            if head.get("applied") and not all(isinstance(head.get(k), np.ndarray) for k in ("A", "c")):
                raise ValueError(f"task {t['id']!r}: an applied head without its arrays (export with ?full=1)")
            self.remove(t["id"])
            self.by_key[t["key"]] = {**t, "head": head}


def _render(x):
    from decisio.serve.systemone import render_text

    return render_text(x)
