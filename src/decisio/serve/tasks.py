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

A task may also carry a certificate (decisio.readout.certify, made by `SystemOne.register_readout_task` on examples held
out of the fit) and the baseline of its answer-mix monitor (decisio.serve.monitor); both are stored with the task and
travel with its export.
"""

from __future__ import annotations

import base64
import hashlib
import json

import numpy as np

from decisio.names import check_format, record_format, same_fingerprint
from decisio.readout import calibration, intent_head
from decisio.readout.debias import log_probs
from decisio.serve.monitor import MixMonitor

FORMAT = record_format("task")
CERTIFICATE_FORMAT = record_format("certificate")
MIN_OPTIONS_FOR_HEAD = 10  # heads on intent sets only: smaller option lists get calibration alone


def task_key(q, render_text, described: bool = False, noul_rendering: str | None = None) -> str:
    """The identity of a wire question's task: type and options in order, plus instructions for yes/no and score.
    `described`: the server's --describe-options rule changes how this question is shown (its options have
    descriptions), so a task fitted under the other rendering has another key and is not applied. A question the rule
    leaves alone keeps the same key under either setting. `noul_rendering`: a yes/no question asked as letters
    (--noul-rendering) has another key than in words."""
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
    if noul_rendering:
        spec["rendering"] = noul_rendering
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


def example_digest(state, q) -> str:
    """The identity of an example's input, whatever its label: its state and its question."""
    body = {"state": state, "question": q.model_dump(mode="json")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def split_holdout(task_id: str, examples, fraction: float):
    """(fit indices, held-out indices) of `examples` [(question, state, gold)], deterministic and independent
    of their order: examples with the same input go to the same side, groups are ordered by the sha256 of the task id
    and the input, and the first groups, up to `fraction` of the examples, are held out."""
    groups: dict[str, list[int]] = {}
    for i, (q, s, _) in enumerate(examples):
        groups.setdefault(example_digest(s, q), []).append(i)
    order = sorted(groups, key=lambda d: hashlib.sha256((task_id + d).encode()).hexdigest())
    target, held = round(fraction * len(examples)), []
    for d in order:
        if len(held) >= target:
            break
        held += groups[d]
    held_set = set(held)
    return [i for i in range(len(examples)) if i not in held_set], sorted(held)


def fit_digest(task) -> str:
    """A digest of what the task applies to an answer: the calibration prior's record and the head's, arrays
    included."""
    h = hashlib.sha256()
    for part in ("calibration", "head"):
        for k, v in sorted(task[part].items()):
            h.update(k.encode())
            if isinstance(v, np.ndarray):
                h.update(np.ascontiguousarray(v, dtype=np.float32).tobytes())
            elif isinstance(v, dict) and "b64" in v:
                h.update(base64.b64decode(v["b64"]))
            else:
                h.update(json.dumps(v, sort_keys=True, default=float).encode())
    return h.hexdigest()


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
        self.monitors: dict[str, MixMonitor] = {}  # the answer-mix window of each certified task, by task id

    def lookup(self, key):
        t = self.by_key.get(key)
        return t if t is not None and same_fingerprint(t["fingerprint"], self.fingerprint) else None

    def register(self, task_id, examples, score_fn, hidden_fn=None, key_fn=None, commit=True):
        """examples: [(wire question, state, gold)], all of one task; score_fn([(state, q)]) -> served label
        probabilities per example; hidden_fn([(state, q)]) -> [(lp, h)] from the hidden-state engine, or None when the
        server has none. Returns the stored task (`commit=False`: the fitted task, not yet stored: the caller adds a
        certificate and calls `put`, and a failure leaves the previous task in place)."""
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
        if commit:
            self.put(task)
        return task, {"lps": lps, "labels": labels, "readout": readout}

    def put(self, task) -> None:
        """Store a task, replacing the task of the same id whatever its list; start its monitor when it has a
        baseline."""
        self.remove(task["id"])
        if task.get("certificate"):  # the certificate names the fit it was made for; recomputed here, never trusted
            task["fit_sha256"] = fit_digest(task)
        self.by_key[task["key"]] = task
        if task.get("monitor"):
            self.monitors[task["id"]] = MixMonitor(task["monitor"])

    def remove(self, task_id) -> bool:
        keys = [k for k, t in self.by_key.items() if t["id"] == task_id]
        for k in keys:
            del self.by_key[k]
        self.monitors.pop(task_id, None)
        return bool(keys)

    # ---- persistence: JSON with the head's arrays as base64 float32 ---------------------------------------------

    @staticmethod
    def public(task, full=False):
        head = dict(task["head"])
        for k in ("A", "c"):
            if k in head:
                head[k] = _pack(head[k]) if full else {"shape": list(np.asarray(head[k]).shape)}
        out = {**task, "head": head}
        cert = task.get("certificate")
        if cert and not full:  # the held-in digests are for re-certification; the listing says how many there are
            split = dict(cert["split"])
            split["fit_digests"] = len(split.get("fit_digests", []))
            out["certificate"] = {**cert, "split": split}
        return out

    def load(self, tasks):
        """Load task records as `public(task, full=True)` writes them; records written before the rename (the `rlcd-*/1`
        formats) load unchanged. A record of another kind is refused before anything is stored."""
        for t in tasks:
            what = f"task {t.get('id')!r}" if isinstance(t, dict) else "task"
            check_format(t, "task", what)
            check_format(t.get("calibration"), "task-prior", f"{what}, calibration")
            check_format(t.get("head"), "intent-head", f"{what}, head")
            if t.get("certificate") is not None:
                check_format(t["certificate"], "certificate", f"{what}, certificate")
        for t in tasks:
            head = dict(t["head"])
            for k in ("A", "c"):
                if isinstance(head.get(k), dict) and "b64" in head[k]:
                    head[k] = _unpack(head[k])
            if head.get("applied") and not all(isinstance(head.get(k), np.ndarray) for k in ("A", "c")):
                raise ValueError(f"task {t['id']!r}: an applied head without its arrays (export with ?full=1)")
            task = {**t, "head": head}
            task.pop("fit_sha256", None)  # recomputed in `put` from what was loaded
            self.put(task)


def _render(x):
    from decisio.serve.systemone import render_text

    return render_text(x)
