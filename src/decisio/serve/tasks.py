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
                multiple-choice question) with at least `intent_head.MIN_PER_OPTION` (2) examples of every option;
                served as `apply_intent_head` (softmax(lp + h @ A + c)). Where a head applies, calibration is not
                stacked on it (its c covers what the prior does); the head is refused with two-order averaging (it is
                bound to one option order).

    store = TaskStore(fingerprint="...")
    task = store.register(task_id, examples, score_fn, hidden_fn)     # examples: [(question, state, gold key)]

A task whose options change with every request (`"options": "dynamic"`, decisio.serve.dynamic) is identified by its
question's instructions instead; it holds a re-ranker and an option prior in `dynamic`, and no calibration or head (the
two records are kept, not applied, so every reader of a task finds them).

A task may also carry a certificate (decisio.readout.certify, made by `SystemOne.register_readout_task` on examples held
out of the fit) and the baseline of its answer-mix monitor (decisio.serve.monitor); both are stored with the task and
travel with its export.

A task may carry a context (`context_record`): a long, fixed text read before every request's state, so requests send
only the part that changes. Its sha256 is part of the task's identity: the same question with another context is
another task, kept beside it (`TaskStore.slot_of`), and the context enters the fit digest, so a certificate made under
one context does not hold under another. A task without one is stored, keyed and digested exactly as before.
"""

from __future__ import annotations

import base64
import hashlib
import json

import numpy as np

from decisio.names import check_format, record_format, same_fingerprint
from decisio.readout import calibration, intent_head
from decisio.readout.debias import log_probs
from decisio.serve import layout
from decisio.serve.monitor import MixMonitor

FORMAT = record_format("task")
# a task with a context: a server that does not know contexts refuses the record (its format check) rather than load it
# into the question's plain slot and serve it without its context; a task without one keeps FORMAT
CONTEXT_FORMAT = "decisio-task/2"
CERTIFICATE_FORMAT = record_format("certificate")
MIN_OPTIONS_FOR_HEAD = 10  # heads on intent sets only: smaller option lists get calibration alone


def check_table_settings(rec: dict, what: str) -> None:
    """A stored fast table's settings, checked where a record is loaded or imported: its `min_saving` (when it has one)
    in `fast_table.MIN_SAVING_RANGE`, and a yes/no table's calibration temperature in `fast_table.TEMPERATURES`, so a
    bad record is refused at load and never fails every request of its task. ValueError says which."""
    from decisio.serve import fast_table as FT

    if rec.get("min_saving") is not None:
        try:
            FT.parse_min_saving(rec["min_saving"])
        except ValueError as e:
            raise ValueError(f"{what}: {e}") from None
    cal = rec.get("calibration")
    if cal is not None:
        t = cal.get("temperature") if isinstance(cal, dict) else None
        lo, hi = FT.TEMPERATURES
        ok = isinstance(t, (int, float)) and not isinstance(t, bool) and lo <= t <= hi
        if not ok:
            raise ValueError(f"{what}: a calibration temperature from {lo:g} to {hi:g}, not {t!r}")


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


class TaskSelectionError(ValueError):
    """A request's task cannot be found as asked (an unknown or inapplicable `"task"`, several tasks that ask the
    question with no selector, questions of one request with different contexts): a 422."""


def serialise_context(value) -> str:
    """A registration's `context` as the text read before the state: a string as given; an object or an array as
    compact JSON. Raises ValueError for anything else, and for an empty text."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    else:
        raise ValueError("context: a string, an object or an array (objects and arrays are read as compact JSON)")
    text = text.strip()  # the outer whitespace is not part of the context: a chat template may trim it
    if not text:
        raise ValueError("context: an empty text; register the task without one")
    return text


def context_record(text: str, tokens: int | None = None) -> dict:
    """A task's context as stored: its text, sha256 and length in characters and in tokens (None: not counted)."""
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
        "tokens": tokens,
        "text": text,
    }


def context_text(task) -> str | None:
    """The text read before the state for a task's requests, or None."""
    ctx = task.get("context") if isinstance(task, dict) else None
    return ctx["text"] if ctx else None


def context_sha(task) -> str | None:
    ctx = task.get("context") if isinstance(task, dict) else None
    return ctx["sha256"] if ctx else None


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


def inputs_digest(*arrays) -> str:
    """A digest of what a registration read from the engine (the examples' label log-scores, and for a head their hidden
    states), as float64 bytes. Two registrations of the same examples have the same digest exactly when the engine read
    them identically, so a head or a threshold that differs between them can be told apart from a fit that differs."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def fit_digest(task) -> str:
    """A digest of what the task applies to an answer: the calibration prior's record and the head's, arrays
    included."""
    h = hashlib.sha256()
    for part in ("calibration", "head", "dynamic"):
        if part == "dynamic" and not task.get("dynamic"):
            continue  # a fixed-option task's digest is what it was
        for k, v in sorted(task[part].items()):
            h.update(k.encode())
            if isinstance(v, np.ndarray):
                h.update(np.ascontiguousarray(v, dtype=np.float32).tobytes())
            elif isinstance(v, dict) and "b64" in v:
                h.update(base64.b64decode(v["b64"]))
            else:
                h.update(json.dumps(v, sort_keys=True, default=float).encode())
    ctx = context_sha(task)
    if ctx:  # the context read before every state; absent for a task without one, whose digest is what it was
        h.update(b"context")
        h.update(ctx.encode())
    ft = task.get("fast_table")
    if ft:  # the table's fit and thresholds; not its timing, agreement baseline or suspension, which are not the fit
        h.update(b"fast_table")
        h.update(json.dumps({k: v for k, v in ft.items() if k not in FAST_TABLE_VOLATILE}, sort_keys=True).encode())
    return h.hexdigest()


def dynamic_placeholders(reason: str = "this task's options change with every request (dynamic options)"):
    """The calibration and head records of a dynamic task: present and not applied, so no reader needs a special case"""
    return (
        {"format": calibration.FORMAT, "applied": False, "reason": reason},
        {"format": intent_head.FORMAT, "applied": False, "reason": reason},
    )


# the parts of a task's fast-table record that are not its fit: where and how fast it was timed, what the monitor
# compares sampled answers with, and whether a replica suspended it
FAST_TABLE_VOLATILE = ("timing", "agreement", "suspended")


def _pack(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return {"dtype": "float32", "shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def _unpack(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.float32).reshape(d["shape"]).copy()


class TaskStore:
    """Registered tasks keyed by `task_key`; each holds its calibration prior and its head record.

    `fingerprint`: the one a task is fitted under on this server, for its own layout. `layouts`: {layout: fingerprint}
    for every layout a request may ask for (decisio.serve.layout); inside `layout.using(L)` the store's fingerprint is
    L's, so a task is registered under the layout in effect and applies to the requests of that layout only. Without
    `layouts` the store has the one fingerprint (a library's or a test's store)."""

    def __init__(self, fingerprint: str, layouts: dict | None = None):
        self._fingerprint = fingerprint
        self.layouts = dict(layouts or {})
        self.by_key: dict[str, dict] = {}
        self.monitors: dict[str, MixMonitor] = {}  # the answer-mix window of each certified task, by task id
        # a task with a fast table also has the monitor of decisio's own answers, for the requests it answers without
        # the table (the table off): their mix and confidences are not the combined stream's
        self.monitors_alone: dict[str, MixMonitor] = {}
        # where tasks are kept across restarts (decisio.serve.task_backends); None: in memory only. A task is written
        # before it is stored in memory and removed from the backend before it is dropped, so a write that fails leaves
        # the store as it was and the caller learns it (TaskPersistError)
        self.backend = None

    @property
    def fingerprint(self) -> str:
        """The fingerprint a task is fitted, and looked up, under: the layout in effect's, else the server's own."""
        return self.layouts.get(layout.current(), self._fingerprint)

    @fingerprint.setter
    def fingerprint(self, value: str) -> None:
        self._fingerprint = value

    def layout_of(self, task) -> str | None:
        """The layout task was fitted under, among the ones this server serves; None for a task of another model or
        rendering, or for every task of a store without `layouts`."""
        fp = task.get("fingerprint") if isinstance(task, dict) else None
        return next((name for name, f in self.layouts.items() if same_fingerprint(fp, f)), None)

    def applies(self, task) -> bool:
        """Whether the task applies to this server, to the requests of one of its layouts."""
        fp = task.get("fingerprint") if isinstance(task, dict) else None
        return same_fingerprint(fp, self._fingerprint) or self.layout_of(task) is not None

    def _own_layout(self) -> str | None:
        return next((name for name, f in self.layouts.items() if same_fingerprint(self._fingerprint, f)), None)

    def slot_of(self, key, task_layout: str | None, context: str | None = None) -> str:
        """Where a task of question `key` fitted under `task_layout` with the context of sha256 `context` is kept in
        `by_key`: its key for the server's own layout (and for a task of no layout this server serves), `key@layout`
        for the other one; `key~<context sha256>` (and `key~<sha256>@layout`) for a task with a context. The same
        question fitted under each layout, or with each context, is two tasks, and neither replaces the other."""
        base = f"{key}~{context}" if context else key
        return base if task_layout is None or task_layout == self._own_layout() else f"{base}@{task_layout}"

    def slot(self, task) -> str:
        return self.slot_of(task["key"], self.layout_of(task), context_sha(task))

    def matches(self, key) -> list:
        """Every task of question `key` that applies to the requests of the layout in effect: one per context."""
        return [t for t in self.of_question(key) if same_fingerprint(t["fingerprint"], self.fingerprint)]

    def lookup(self, key):
        """The task of question `key` that a request naming no task gets under the layout in effect, or None: the one
        task that asks it; of several (the question registered with different contexts), the one without a context,
        which is what such a request got before contexts existed. Several with a context and none without: no task can
        be chosen for the request (TaskSelectionError, a 422 that asks for `"task"`)."""
        found = self.matches(key)
        if len(found) <= 1:
            return found[0] if found else None
        plain = [t for t in found if not context_sha(t)]
        if plain:
            return plain[0]
        raise TaskSelectionError(
            f"the tasks {sorted(t['id'] for t in found)} all ask this question, each with its own context; name the "
            'one this request is for with "task": "<id>"'
        )

    def by_id(self, task_id):
        """The stored task of this id, or None."""
        return next((t for t in self.by_key.values() if t["id"] == task_id), None)

    def of_question(self, key) -> list:
        """Every task kept for the question `key`, whatever it was fitted under or with: the one at its key first."""
        return [t for k, t in self.by_key.items() if k == key or k.startswith((key + "@", key + "~"))]

    def register(self, task_id, examples, score_fn, hidden_fn=None, key_fn=None, commit=True, context=None):
        """examples: [(wire question, state, gold)], all of one task; score_fn([(state, q)]) -> served label
        probabilities per example; hidden_fn([(state, q)]) -> [(lp, h)] from the hidden-state engine, or None when the
        server has none. Returns the stored task (`commit=False`: the fitted task, not yet stored: the caller adds a
        certificate and calls `put`, and a failure leaves the previous task in place). `context`: the task's
        `context_record`, stored with it; the two functions read each state with it (the caller's concern)."""
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
        prior = {**prior, "inputs_sha256": inputs_digest(lps)}
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
                "reason": intent_head.floor_reason(counts, options),
                "short_options": [n for n, _ in intent_head.short_options(counts, options)],
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
            head = {
                **head,
                "inputs_sha256": inputs_digest([lp for lp, _ in readout], [h for _, h in readout]),
            }
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
        if context is not None:
            task["context"] = context
        if commit:
            self.put(task)
        return task, {"lps": lps, "labels": labels, "readout": readout}

    def put(self, task, persist: bool = True) -> None:
        """Store a task, replacing the task of the same id whatever its list; start its monitor when it has a
        baseline. With a backend the task is written first (`persist=False`: the caller writes it with `persist_task`
        once it has finished with the record, and rolls back with `rollback` if that fails)."""
        if task.get("certificate"):  # the certificate names the fit it was made for; recomputed here, never trusted
            task["fit_sha256"] = fit_digest(task)
        if task.get("context") is not None:
            task["format"] = CONTEXT_FORMAT
        if persist:
            self.persist_task(task)
        # the new record goes in before the old one goes out, so a reader never finds the task missing (a replica
        # applying another replica's version while it serves)
        slot = self.slot(task)
        old = [k for k, t in self.by_key.items() if t["id"] == task["id"] and k != slot]
        self.by_key[slot] = task
        for k in old:
            self.by_key.pop(k, None)
        self.monitors.pop(task["id"], None)
        self.monitors_alone.pop(task["id"], None)
        if task.get("monitor"):
            self.monitors[task["id"]] = MixMonitor(task["monitor"])
        if task.get("monitor_alone"):
            self.monitors_alone[task["id"]] = MixMonitor(task["monitor_alone"])

    def persist_task(self, task) -> None:
        """Write the task's record to the backend, if there is one. Raises TaskPersistError."""
        if self.backend is not None:
            self.backend.save("readout", self.public(task, full=True))

    def snapshot(self, task_id):
        """The stored task of this id (or None), for `rollback`."""
        return next((t for t in self.by_key.values() if t["id"] == task_id), None)

    def rollback(self, task_id, previous) -> None:
        """Put memory back as it was before a `put(..., persist=False)` whose write failed."""
        self._drop(task_id)
        if previous is not None:
            self.by_key[self.slot(previous)] = previous
            if previous.get("monitor"):
                self.monitors[task_id] = MixMonitor(previous["monitor"])
            if previous.get("monitor_alone"):
                self.monitors_alone[task_id] = MixMonitor(previous["monitor_alone"])

    def _drop(self, task_id) -> bool:
        keys = [k for k, t in self.by_key.items() if t["id"] == task_id]
        for k in keys:
            del self.by_key[k]
        self.monitors.pop(task_id, None)
        self.monitors_alone.pop(task_id, None)
        return bool(keys)

    def remove(self, task_id, persist: bool = True) -> bool:
        """Remove the task (and its saved record, first). False when there is no such task."""
        if not any(t["id"] == task_id for t in self.by_key.values()):
            return False
        if persist and self.backend is not None:
            self.backend.delete("readout", task_id)
        return self._drop(task_id)

    # ---- persistence: JSON with the head's arrays as base64 float32 ---------------------------------------------

    @staticmethod
    def public(task, full=False):
        head = dict(task["head"])
        for k in ("A", "c"):
            if k in head:
                head[k] = _pack(head[k]) if full else {"shape": list(np.asarray(head[k]).shape)}
        out = {**task, "head": head}
        dyn = task.get("dynamic")
        if dyn and not full and "prior" in dyn:  # the listing says how big the prior is; the export holds the table
            out["dynamic"] = {**dyn, "prior": {"options": len(dyn["prior"])}}
        if task.get("fast_table") and not full:
            ft = dict(task["fast_table"])
            for k in ("coef", "intercept"):
                if isinstance(ft.get(k), dict) and "b64" in ft[k]:
                    ft[k] = {"shape": ft[k]["shape"]}
            out["fast_table"] = ft
        if task.get("context") and not full:  # the listing says how long the context is and its digest, never its text
            out["context"] = {k: v for k, v in task["context"].items() if k != "text"}
        cert = task.get("certificate")
        if cert and not full:  # the held-in digests are for re-certification; the listing says how many there are
            split = dict(cert["split"])
            split["fit_digests"] = len(split.get("fit_digests", []))
            out["certificate"] = {**cert, "split": split}
        return out

    def load(self, tasks, persist: bool = True):
        """Load task records as `public(task, full=True)` writes them; records written before the rename (the `rlcd-*/1`
        formats) load unchanged; a task with a context has CONTEXT_FORMAT, and only it. A record of another kind is
        refused before anything is stored. With a backend they are saved as they load (`persist=False`: they came from
        it); one that cannot be saved stops the load, the ones before it stay loaded and saved."""
        for t in tasks:
            what = f"task {t.get('id')!r}" if isinstance(t, dict) else "task"
            has_context = isinstance(t, dict) and t.get("context") is not None
            got = t.get("format") if isinstance(t, dict) else None
            if has_context and got != CONTEXT_FORMAT:
                raise ValueError(f"{what}: a task with a context has the format {CONTEXT_FORMAT!r}, not {got!r}")
            if not has_context and got == CONTEXT_FORMAT:
                raise ValueError(f"{what}: the format {CONTEXT_FORMAT!r} is a task with a context, and it has none")
            if not has_context:
                check_format(t, "task", what)
            check_format(t.get("calibration"), "task-prior", f"{what}, calibration")
            check_format(t.get("head"), "intent-head", f"{what}, head")
            if t.get("dynamic") is not None:
                check_format(t["dynamic"], "dynamic", f"{what}, dynamic")
                if t["dynamic"].get("applied") and not isinstance(t["dynamic"].get("prior"), dict):
                    raise ValueError(f"task {t['id']!r}: a re-ranker without its prior table (export with ?full=1)")
            if t.get("certificate") is not None:
                check_format(t["certificate"], "certificate", f"{what}, certificate")
            if t.get("fast_table") is not None:
                check_format(t["fast_table"], "fast-table", f"{what}, fast_table")
                if not isinstance(t["fast_table"].get("coef"), dict):
                    raise ValueError(f"task {t['id']!r}: a fast table without its arrays (export with ?full=1)")
                check_table_settings(t["fast_table"], f"task {t['id']!r}, fast_table")
            learn_req = (t.get("learn") or {}).get("request") if isinstance(t.get("learn"), dict) else None
            if isinstance(learn_req, dict) and learn_req.get("min_saving") is not None:
                check_table_settings({"min_saving": learn_req["min_saving"]}, f"task {t['id']!r}, learn.request")
            if t.get("context") is not None:
                ctx = t["context"]
                if not isinstance(ctx, dict) or not isinstance(ctx.get("text"), str):
                    raise ValueError(f"task {t['id']!r}: a context without its text (export with ?full=1)")
                if context_record(ctx["text"])["sha256"] != ctx.get("sha256"):
                    raise ValueError(f"task {t['id']!r}: the context's sha256 is not its text's")
        for t in tasks:
            head = dict(t["head"])
            for k in ("A", "c"):
                if isinstance(head.get(k), dict) and "b64" in head[k]:
                    head[k] = _unpack(head[k])
            if head.get("applied") and not all(isinstance(head.get(k), np.ndarray) for k in ("A", "c")):
                raise ValueError(f"task {t['id']!r}: an applied head without its arrays (export with ?full=1)")
            task = {**t, "head": head}
            task.pop("fit_sha256", None)  # recomputed in `put` from what was loaded
            self.put(task, persist=persist)


def _render(x):
    from decisio.serve.systemone import render_text

    return render_text(x)
