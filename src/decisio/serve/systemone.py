# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""TypeSafe's System One wire format (`POST /v1/systemone`, `GET /v1/models`) on the letters engine.

The request and response schemas are TypeSafe's own (typesafe-sdk 0.7.0, `_schemas/models.py`, generated from
api.typesafe.ai/openapi.json; MIT). Each question becomes one
question of the state-plus-questions engine and is rendered by the same letters readout as `/v1/answer`, so a request
whose criteria carry no descriptions scores exactly the token rows `/v1/answer` scores (conformance gate C3).
The mapping was fixed before any benchmark item was sent (runs/2026-09-27_boards-baseline):

  state         passed to the engine unchanged (a string as is; an object as `key: value` lines, like evaluation)
  instructions  a string as is; an object or array as compact JSON; absent or blank: a fixed default per type
  noul          instructions, then "Yes means: <true>" / "No means: <false>" lines when criteria are given; P(yes)
  choice        options in the request's key order, shown "<key>: <description>" or "<key>"; letters readout
  score         levels in criteria order as the letters legend; score = probability-weighted mean level
  confidence    the System One adapter's rules (choice: max rescaled from uniform; score: concentration at the mode)
  usage         input_tokens = state + each question's own text in this tokenizer; output_tokens = one per question

Two-order mode (after Reflex, github.com/kshetrajna12/reflex, MIT, `prompt.distinct_orders` and
`ensemble.disagreement`): each question is also read in one distinct second order (seeded by its name; the swap for two
options; "Answer no or yes." for yes/no), the two distributions are averaged per label, and the disagreement (mean
total-variation distance of each branch from their average) goes to a branch log, never into the response.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import re
import threading
import time
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from decisio.names import DEBUG_KEY, check_format, header, request_header, same_fingerprint
from decisio.serve.temperature import apply_temperature

JSONContent = Any  # str | dict | list, as the SDK's schema allows


# ---- wire schema (TypeSafe's SystemOneRequest and its questions) -----------------------------------------------------


class NoulCriteria(BaseModel):
    true: JSONContent | None = None
    false: JSONContent | None = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None] = Field(..., min_length=1)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JSONContent | None = None
    criteria: list[JSONContent] = Field(..., min_length=1)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # extensions (`orders`) and fields TypeSafe may add later are ignored
    state: str | dict[str, Any] | list[Any]
    # TypeSafe's API requires `model`; imajev's requests (github.com/mohit67890/imajev, Apache-2.0) omit it, so it is
    # optional here and the response names the served model when it is absent
    model: str | None = None
    questions: dict[str, Question] = Field(..., min_length=1)


# ---- mapping --------------------------------------------------------------------------------------------------------


def render_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


# absent instructions (the wire format allows them): the letters prompt needs a question line, or the question's text
# would start with whitespace and merge with the state's last token (PREREG amendment 1, before any measurement)
DEFAULT_INSTRUCTIONS = {"noul": "Is this true?", "choice": "Which option applies?", "score": "Which level applies?"}


INDEX_KEY = re.compile(r"^([A-Za-z]+[_\- ]?)?(\d+)$")


def index_keys(criteria: dict) -> bool:
    """True when a choice question's keys are a pure enumeration carrying no meaning of their own: every key is one
    shared word stem plus a number (or a bare number), the numbers are exactly 0..n-1 or 1..n (in any order: a shuffled
    enumeration is still no more than an index), and every option has a non-empty description to show instead (the
    Decision Index's intent format: `option_0` ... `option_76`, each described by its intent name). Keys such as
    `sku_12`, `room 101` or `2xl` are meaning, not an index."""
    keys = list(criteria)
    matches = [INDEX_KEY.match(k) for k in keys]
    if len(keys) < 2 or not all(matches) or len({(m.group(1) or "").lower() for m in matches}) != 1:
        return False
    nums = sorted(int(m.group(2)) for m in matches)
    if nums not in (list(range(len(nums))), list(range(1, len(nums) + 1))):
        return False
    return all(d is not None and render_text(d).strip() != "" for d in criteria.values())


SNAKE_LABEL = re.compile(r"^[a-z]+(?:_[a-z]+)*$")


def snake_label(label: str) -> bool:
    """True for a snake_case label: lowercase words joined by underscores, at least two words (no digits, capitals,
    spaces or punctuation): the Decision Index's BANKING77 names such as `card_arrival`. Any other label is left as sent
    (two of BANKING77's 77: `Refund_not_showing_up`, `reverted_card_payment?`)."""
    return "_" in label and SNAKE_LABEL.match(label) is not None


def to_engine_question(
    q, hide_index_keys: bool = True, desnake_labels: bool = True, describe_options: bool = True
) -> tuple[dict, list[str]]:
    """(engine question dict, the answer keys in engine option order) for one wire question.

    `hide_index_keys` (the served default): a choice question whose keys are a pure enumeration (`index_keys`) shows
    each option's description alone; the keys stay the answer's keys. Otherwise an option shows `key: description`, or
    the key alone. Rendering `B. option_1: card_linking` put a second index beside our letters, which the single-order
    readout confused with its own (runs/2026-09-27_boards-baseline, EVAL_CARD.md).

    `desnake_labels` (the served default): when the options are shown as bare labels (index keys hidden, or keys without
    descriptions), each snake_case label (`snake_label`) is shown with spaces (`card arrival`) and every other label as
    sent; the keys stay the answer's keys. `key: description` renderings are never changed.

    `describe_options` (the served default; `--no-describe-options` turns it off): an option with a description is
    shown as its description alone, with no key; an option without one is shown as its key (de-snaked as above).
    The keys stay the answer's keys. If two options would then read the same, the question is rendered as without
    the rule."""
    instr = render_text(q.instructions).strip() or DEFAULT_INSTRUCTIONS[q.type]
    if q.type == "noul":
        extra = []
        if q.criteria is not None:
            if q.criteria.true is not None:
                extra.append(f"Yes means: {render_text(q.criteria.true)}")
            if q.criteria.false is not None:
                extra.append(f"No means: {render_text(q.criteria.false)}")
        text = "\n".join([instr, *extra]) if extra else instr
        return {"kind": "noul", "instructions": text}, ["yes", "no"]
    if q.type == "choice":
        keys = list(q.criteria)
        bare = None  # options shown as bare labels, if they are
        if hide_index_keys and index_keys(q.criteria):
            bare = [render_text(d) for d in q.criteria.values()]
        elif all(d is None or render_text(d) == "" for d in q.criteria.values()):
            bare = list(keys)
        if bare is None and describe_options:
            has = [d is not None and render_text(d) != "" for d in q.criteria.values()]
            shown = [
                render_text(d) if h else (k.replace("_", " ") if desnake_labels and snake_label(k) else k)
                for (k, d), h in zip(q.criteria.items(), has)
            ]
            if len(set(shown)) == len(shown):
                return {"kind": "choice", "instructions": instr, "options": shown}, keys
        if bare is not None:
            if desnake_labels:
                bare = [x.replace("_", " ") if snake_label(x) else x for x in bare]
            return {"kind": "choice", "instructions": instr, "options": bare}, keys
        shown = [
            f"{k}: {render_text(d)}" if d is not None and render_text(d) != "" else k for k, d in q.criteria.items()
        ]
        return {"kind": "choice", "instructions": instr, "options": shown}, keys
    levels = [render_text(c) for c in q.criteria]
    return {"kind": "score", "instructions": instr, "options": levels}, [str(i) for i in range(len(levels))]


UNKNOWN_KEY = "__unknown__"


NOUL_RENDERINGS = ("words", "letters", "letters-keys")


def noul_as_letters(q, keys_shown: bool = False) -> tuple[dict, list[str]] | None:
    """A yes/no question as a two-option choice read from the option letters (--noul-rendering letters, Cygnet's
    form): the false side first, each side shown as its criteria description, "No" and "Yes" for a side without one;
    `keys_shown` (letters-keys) names the sides, "No: <false>" and "Yes: <true>". Returns the engine question and its
    keys in engine order (["no", "yes"]), or None when the two sides would read the same (the question is then asked
    in words)."""
    instr = render_text(q.instructions).strip() or DEFAULT_INSTRUCTIONS["noul"]
    crit = q.criteria

    def side(desc, word):
        text = None if desc is None else render_text(desc).strip()
        if not text:
            return word
        return f"{word}: {text}" if keys_shown else text

    options = [side(crit.false if crit else None, "No"), side(crit.true if crit else None, "Yes")]
    if options[0] == options[1]:
        return None
    return {"kind": "choice", "instructions": instr, "options": options}, ["no", "yes"]


def with_abstain(eq: dict, keys: list[str], option: str) -> tuple[dict, list[str]]:
    """The engine question with one more option, `option` (e.g. "can't tell"), listed last; yes/no questions become a
    three-option choice (yes, no, option). Its key is UNKNOWN_KEY. No training: the option is only offered."""
    if eq["kind"] == "noul":
        return {"kind": "choice", "instructions": eq["instructions"], "options": ["yes", "no", option]}, [
            "yes",
            "no",
            UNKNOWN_KEY,
        ]
    return {**eq, "options": [*eq["options"], option]}, [*keys, UNKNOWN_KEY]


def abstain_answer(q, keys: list[str], p: np.ndarray, abstained: bool | None = None) -> dict:
    """imajev's extension from a distribution whose last entry is the offered "can't tell" option: `unknown_probability`
    is that entry, `abstained` is whether it is the most likely outcome (or the task's threshold decision, when given),
    and the answer's own probabilities are over the real options, conditional on answering (imajev's decoder multiplies
    them by 1 - unknown_probability); a yes/no answer's `noul` is P(yes) plus half the unknown mass, imajev's
    convention."""
    unknown, real = float(p[-1]), np.asarray(p[:-1], dtype=np.float64)
    if abstained is None:
        abstained = bool(unknown > real.max())
    if q.type == "noul":
        out = {"type": "noul", "noul": float(real[0] + unknown / 2)}
    else:
        out = to_answer(q, keys[:-1], real / real.sum())
    return {**out, "unknown_probability": unknown, "abstained": abstained}


def second_order(name: str, q: dict, keys: list[str]) -> tuple[dict, list[int]]:
    """The second branch of two-order mode and its permutation (position j shows original option perm[j])."""
    if q["kind"] == "noul":
        return dict(q, noul_order="no_yes"), [0, 1]
    n = len(q["options"])
    rng = random.Random(f"0:{name}")  # Reflex's seeding: one question's order never depends on the others
    perm = list(range(n))
    if n == 2:
        perm = [1, 0]
    else:
        for _ in range(50):
            rng.shuffle(perm)
            if perm != list(range(n)):
                break
    return dict(q, options=[q["options"][j] for j in perm]), perm


def choice_confidence(p) -> float:
    k = len(p)
    if k == 1:
        return 1.0
    return float((max(p) - 1.0 / k) / (1.0 - 1.0 / k))


def score_confidence(p) -> float:
    k = len(p)
    if k == 1:
        return 1.0
    mode = int(np.argmax(p))
    dist = float(sum(pi * abs(i - mode) for i, pi in enumerate(p)))
    centre = (k - 1) / 2
    uniform_mad = sum(abs(i - centre) for i in range(k)) / k
    return max(0.0, 1.0 - dist / uniform_mad)


def top_index(keys, p, among=None) -> int:
    """The declared choice: the most probable option; among options whose probabilities are exactly equal, the one whose
    key sorts first (Unicode code-point order), so the choice never depends on the order of the keys on the wire (JSON
    objects are unordered, and clients that serialise with sorted keys reorder them; docs/handoffs/tasks.md,
    "Exact ties"). `among`: the candidate indices (default: all)."""
    idx = list(range(len(p))) if among is None else list(among)
    m = max(p[i] for i in idx)
    return min((i for i in idx if p[i] == m), key=lambda i: keys[i])


def to_answer(q, keys, p) -> dict:
    p = [float(x) for x in p]
    if q.type == "noul":
        return {"type": "noul", "noul": p[0]}
    if q.type == "choice":
        top = keys[top_index(keys, p)]
        return {
            "type": "choice",
            "choice": top,
            "confidence": choice_confidence(p),
            "probabilities": dict(zip(keys, p)),
        }
    return {
        "type": "score",
        "score": float(sum(i * pi for i, pi in enumerate(p))),
        "confidence": score_confidence(p),
        "legend": {str(i): c for i, c in enumerate(q.criteria)},
        "probabilities": dict(zip(keys, p)),
    }


def disagreement(branches: list[np.ndarray]) -> float:
    m = np.stack(branches)
    return float(0.5 * np.abs(m - m.mean(0)).sum(1).mean())


IMAGE_DATA_URL = re.compile(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+")


def decode_data_url(value: str) -> bytes:
    m = (
        re.fullmatch(r"data:image/[\w.+-]+;base64,([A-Za-z0-9+/=\s]+)", value.strip())
        if isinstance(value, str)
        else None
    )
    if m is None:
        raise ValueError("images must be data:image/...;base64 URLs")
    return base64.b64decode(m.group(1), validate=False)


def extract_state_images(state):
    """imajev's rule: data:image URIs anywhere in the state become images, in reading order, replaced by "[image N]"."""
    found = []

    def walk(node):
        if isinstance(node, str):
            return (
                IMAGE_DATA_URL.sub(lambda m: (found.append(m.group(0)), f"[image {len(found)}]")[1], node)
                if "data:image/" in node
                else node
            )
        if isinstance(node, list):
            return [walk(x) for x in node]
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        return node

    return walk(state), found


def option_set(q, described: bool = False) -> str:
    """A choice question's task identity for abstention: its options (keys and descriptions, in any order). The same
    question asked of different states keeps its options, whatever its instructions (the Decision Index puts the
    utterance there). `described`: as `decisio.serve.tasks.task_key`, the --describe-options rule changes how this
    question is shown, so a threshold fitted under the other rendering is not applied."""
    crit = sorted([k, render_text(d)] for k, d in q.criteria.items()) if q.type == "choice" else None
    spec = {"type": q.type, "criteria": crit, **({"rendering": "describe_options"} if described else {})}
    return hashlib.sha256(json.dumps(spec, ensure_ascii=False).encode()).hexdigest()


class SystemOne:
    """Serves the wire format on a text engine and, optionally, an image engine (decisio.serve.image_engine): requests
    with images go to the image engine, every other request to the text engine."""

    def __init__(
        self,
        engine,
        served_name: str,
        orders: int = 1,
        branch_log: str | None = None,
        release_date: str = "2026-09-27",
        description: str = "",
        image_engine=None,
        abstain_option: str | None = None,
        hide_index_keys: bool = True,
        desnake_labels: bool = True,
        describe_options: bool = True,
        noul_rendering: str = "words",
        abstention: bool = True,
        abstention_tasks: list[dict] | None = None,
        tasks_enabled: bool = True,
        task_store=None,
        hidden_engine=None,
        debug_readout: bool = False,
        temperature: float = 1.0,
        temperatures: dict | None = None,
    ):
        self.engine, self.image_engine, self.name, self.orders = engine, image_engine, served_name, orders
        self.hide_index_keys = hide_index_keys  # to_engine_question; the served default is on
        self.desnake_labels = desnake_labels  # likewise
        self.describe_options = describe_options  # likewise
        if noul_rendering not in NOUL_RENDERINGS:
            raise ValueError(f"noul_rendering must be one of {NOUL_RENDERINGS}")
        self.noul_rendering = noul_rendering  # how a yes/no question is asked (noul_as_letters); words by default
        # opt-in: requests using imajev's extension are offered one more option, whose probability is reported as
        # imajev's unknown_probability
        self.abstain_option = abstain_option
        # per-task abstention thresholds (decisio.serve.abstention; docs/handoffs/tasks.md): tasks registered
        # from labelled examples; `abstention=False` ignores them all
        self.abstention = abstention
        for t in abstention_tasks or []:  # records written before the rename load unchanged
            check_format(t.get("config"), "abstention", f"abstention task {t.get('id')!r}")
        self.tasks: dict[str, dict] = {t["id"]: t for t in (abstention_tasks or [])}
        # per-task calibration and the intent head (decisio.serve.tasks; docs/handoffs/tasks.md): registered by
        # POST /v1/tasks, applied to text-route questions whose option list matches; the head needs the hidden-state
        # engine (decisio.serve.hidden_engine); `debug_readout` lets a request ask for its raw readout (verification)
        self.tasks_enabled, self.task_store, self.hidden_engine = tasks_enabled, task_store, hidden_engine
        self.debug_readout = debug_readout
        # the global temperature on the text route's plain readout (decisio.serve.temperature); 1.0 is off, bit for bit;
        # a registered task's own correction replaces it
        self.temperature = float(temperature)
        # per question type ("choice", "noul", "score"), a temperature replacing the global one; unset types keep it
        self.temperatures = {k: float(v) for k, v in (temperatures or {}).items() if v is not None}
        self.release_date, self.description = release_date, description
        self.branch_log = branch_log
        self._log_lock = threading.Lock()

    def models(self) -> dict:
        return {
            "models": [
                {
                    "name": self.name,
                    "description": self.description or "letters readout, served default",
                    "release_date": self.release_date,
                }
            ]
        }

    def usage_tokens(self, state, qs: list[dict]) -> int:
        from decisio.readout.letters import DEFAULT_FORMAT, fmt_state
        from decisio.serve.vllm_engine import question_text

        tok, fmt = self.engine.tok, getattr(self.engine, "fmt", DEFAULT_FORMAT)
        n = len(tok.encode(fmt_state(state), add_special_tokens=False))
        return n + sum(len(tok.encode(question_text(tok, q, fmt)[0], add_special_tokens=False)) for q in qs)

    def answer(
        self, req: SystemOneRequest, adapter=None, images=None, imajev_ext=False, route=None, debug=None
    ) -> dict:
        """`images`: decoded PIL images (the request is then served by the image engine); `imajev_ext`: the request used
        imajev's `images` extension, so every answer also carries `unknown_probability` and `abstained` (0.0 and false:
        the letters readout has no unknown outcome); `route="image"` sends a text request to the image engine (the
        paired check)."""
        images = list(images or [])
        use_image = bool(images) or route == "image"
        if use_image and self.image_engine is None:
            raise ValueError("this server has no image engine (start it with --image-model)")
        engine = self.image_engine if use_image else self.engine
        orders = int((req.model_extra or {}).get("orders") or self.orders)
        names = list(req.questions)
        wire = [req.questions[n] for n in names]
        # a yes/no question asked as letters is read false first and reversed to (yes, no) below; abstention and
        # two-order mode keep the words form, whose arithmetic they assume; the image route keeps it too, the prompt
        # its ImajevBench figures were measured with (its engine builds photo prompts without --prompt-tail)
        words = orders == 2 or (bool(self.abstain_option) and imajev_ext) or use_image
        asked = [self.engine_question(q, words=words) for q in wire]
        mapped = [(eq, keys) for eq, keys, _ in asked]
        swapped = [sw for _, _, sw in asked]
        # per-task abstention: (abstain index, config, appended?) per question, or None
        plans = [self.abstention_plan(q, keys, imajev_ext) for q, (_, keys) in zip(wire, mapped)]
        for i, pl in enumerate(plans):  # an abstention task asks its yes/no question in words
            if pl and swapped[i]:
                eq, keys, _ = self.engine_question(wire[i], words=True)
                mapped[i], swapped[i] = (eq, keys), False
                plans[i] = self.abstention_plan(wire[i], keys, imajev_ext)
        for i, pl in enumerate(plans):
            if pl and pl[2]:
                mapped[i] = with_abstain(*mapped[i], pl[3])
        abstain = bool(self.abstain_option) and imajev_ext
        if abstain:
            if orders == 2:
                raise ValueError("the abstain option is served with one order only")
            mapped = [m if pl and pl[2] else with_abstain(*m, self.abstain_option) for m, pl in zip(mapped, plans)]
        qs = [m[0] for m in mapped]
        # per-task calibration and head: the registered task of each text-route question (None when there is none)
        rts = [
            self.readout_task(q) if not use_image and not (plans[i] and plans[i][2]) and not abstain else None
            for i, q in enumerate(wire)
        ]
        head = [i for i, t in enumerate(rts) if t is not None and t["head"].get("applied") and self.hidden_engine]
        if head and orders == 2:
            raise ValueError(
                "the intent head is not combined with two-order averaging: a question of a task with a "
                "head was asked with orders=2 (the head is bound to one option order)"
            )
        gen = [i for i in range(len(names)) if i not in head]
        engine_qs, plan = [qs[i] for i in gen], {}
        if orders == 2:
            for i in gen:
                q2, perm = second_order(names[i], *mapped[i])
                plan[i] = (len(engine_qs), perm)
                engine_qs.append(q2)
        t0 = time.perf_counter()
        probs, info = [], {}
        if engine_qs:
            if images:
                probs, info = engine.answer(req.state, engine_qs, adapter, images=images)
            else:
                probs, info = engine.answer(req.state, engine_qs, adapter)
        at = {i: n for n, i in enumerate(gen)}
        hidden = dict(zip(head, self.hidden_engine.readout(req.state, [qs[i] for i in head]))) if head else {}
        dbg = {}
        if debug == "hidden" and self.hidden_engine is not None:  # verification: every question's hidden readout
            for i, (lp, h) in enumerate(self.hidden_engine.readout(req.state, qs)):
                dbg[names[i]] = {"hidden_lp": lp.tolist(), "h": h.tolist()}
        answers, log = {}, []
        for i, (name, q, (eq, keys)) in enumerate(zip(names, wire, mapped)):
            t = rts[i]
            if i in hidden:  # the intent head (reference arithmetic)
                from decisio.readout.intent_head import apply_intent_head

                lp, h = hidden[i]
                p = apply_intent_head(lp, h, t["head"], t["options"])
                if debug:
                    dbg.setdefault(name, {}).update(path="head", task=t["id"], hidden_lp=lp.tolist(), h=h.tolist())
            else:
                p = np.asarray(probs[at[i]], dtype=np.float64)
                if swapped[i]:  # read false first; from here on (yes, no), as every yes/no readout
                    p, keys = p[::-1].copy(), ["yes", "no"]
                p2, corrected = None, False
                if orders == 2:
                    j, perm = plan[i]
                    p2 = np.zeros_like(p)
                    for pos, orig in enumerate(perm):  # branch 2 position pos shows original option perm[pos]
                        p2[orig] = probs[j][pos]
                    d = disagreement([p, p2])
                    log.append(
                        {
                            "name": name,
                            "kind": eq["kind"],
                            "k": len(keys),
                            "p1": p.tolist(),
                            "p2": p2.tolist(),
                            "perm": perm,
                            "disagreement": d,
                            "images": len(images),
                        }
                    )
                if debug:  # the served readout before any task (order 1)
                    dbg.setdefault(name, {}).update(path="plain", p=p.tolist())
                    forms = info.get("label_token_logprobs")
                    if forms:  # --label-variants summed or cygnet: each form's log-probability, in engine order
                        dbg[name]["form_logprobs"] = forms[at[i]]
                if t is not None and t["calibration"].get("applied"):  # per-task calibration (reference arithmetic)
                    from decisio.readout.calibration import apply_task_prior
                    from decisio.readout.debias import log_probs

                    pc = apply_task_prior(log_probs(p), t["calibration"])
                    if p2 is not None:  # the bias is per displayed slot: apply it in branch 2's order
                        j, perm = plan[i]
                        shown = apply_task_prior(log_probs(np.asarray(probs[j], dtype=np.float64)), t["calibration"])
                        p2 = np.zeros_like(p)
                        for pos, orig in enumerate(perm):
                            p2[orig] = shown[pos]
                    p = pc
                    if debug:
                        dbg[name].update(path="calibration", task=t["id"])
                    corrected = True
                if p2 is not None:
                    p = (p + p2) / 2
                # the global temperature, where no task correction applied; after two-order averaging, so the most
                # probable option is the same with and without it
                t_q = self.temperature_of(q.type)
                if not corrected and t_q != 1.0 and not use_image:
                    p = apply_temperature(p, t_q)
                    if debug:
                        dbg[name].update(path="temperature", temperature=t_q)
            if plans[i]:
                answers[name] = self.abstention_answer(q, keys, p, plans[i], imajev_ext)
                continue
            if abstain:
                answers[name] = abstain_answer(q, keys, p)
                continue
            answers[name] = to_answer(q, keys, p)
            if imajev_ext:
                answers[name].update(unknown_probability=0.0, abstained=False)
        if log and self.branch_log:
            body = json.dumps(
                {"state": req.state, "questions": {n: q.model_dump() for n, q in req.questions.items()}},
                sort_keys=True,
                ensure_ascii=False,
            )
            h = hashlib.sha256(body.encode()).hexdigest()
            with self._log_lock, open(self.branch_log, "a") as f:
                for rec in log:
                    f.write(json.dumps({"t": time.time(), "request_sha256": h, **rec}) + "\n")
        usage = {
            "input_tokens": self.usage_tokens(req.state, qs) + sum(info.get("image_tokens") or []),
            "output_tokens": len(names),
        }
        out = {
            "model": req.model or self.name,
            "answers": answers,
            "usage": usage,
            "_timing": {
                "server_ms": (time.perf_counter() - t0) * 1000,
                "route": "image" if use_image else "text",
                "tasks": sorted({t["id"] for t in rts if t is not None}),
                **{k: v for k, v in info.items() if k in ("cached_tokens_mean", "image_tokens")},
                # where the engine's time went, for the x-decisio-stages header (milliseconds)
                "stages": {
                    {"server_ms": "engine"}.get(k, k[:-3]): info[k]
                    for k in ("prepare_ms", "warm_ms", "questions_ms", "readout_ms", "server_ms")
                    if isinstance(info.get(k), (int, float))
                },
            },
        }
        if debug and self.debug_readout:
            out[DEBUG_KEY] = dbg
        return out

    # ---- per-task calibration and the intent head ----------------------------------------------------------------

    def temperature_of(self, qtype: str) -> float:
        """The temperature a question of this type is served at: its own (--temperature-<type>) or the global one."""
        return self.temperatures.get(qtype, self.temperature)

    def engine_question(self, q, words: bool = False) -> tuple[dict, list[str], bool]:
        """(engine question, keys in engine order, swapped) for one wire question as this server asks it. A yes/no
        question under --noul-rendering letters or letters-keys is asked as a two-option choice whose readout comes
        back false first: `swapped` says to reverse it to (yes, no) before anything else reads it. `words` asks yes/no
        questions in words whatever the setting (abstention and two-order mode, whose arithmetic assumes it)."""
        if q.type == "noul" and self.noul_rendering != "words" and not words:
            got = noul_as_letters(q, keys_shown=self.noul_rendering == "letters-keys")
            if got is not None:
                return got[0], got[1], True
        eq, keys = to_engine_question(q, self.hide_index_keys, self.desnake_labels, self.describe_options)
        return eq, keys, False

    def noul_rendered(self, q) -> str | None:
        """The yes/no rendering that enters question q's task key, or None (words, or not a yes/no question)."""
        if q.type != "noul" or self.noul_rendering == "words" or noul_as_letters(q) is None:
            return None
        return self.noul_rendering

    def described(self, q) -> bool:
        """Whether this server's --describe-options rule changes how question q is shown (part of its task keys)."""
        if not self.describe_options or q.type != "choice":
            return False
        on = to_engine_question(q, self.hide_index_keys, self.desnake_labels, True)[0]
        return on != to_engine_question(q, self.hide_index_keys, self.desnake_labels, False)[0]

    def readout_task(self, q):
        if not self.tasks_enabled or self.task_store is None or not self.task_store.by_key:
            return None
        from decisio.serve.tasks import task_key

        return self.task_store.lookup(task_key(q, render_text, self.described(q), self.noul_rendered(q)))

    def register_readout_task(self, task_id: str, examples: list[tuple]):
        """Fit and store a task from labelled examples [(state, wire question, gold)] (decisio.serve.tasks)."""
        if self.task_store is None:
            raise ValueError("this server has no task store (--no-tasks)")

        def score(pairs):
            out = []
            for state, q in pairs:
                eq, _, swapped = self.engine_question(q)
                probs, _ = self.engine.answer(state, [eq])
                out.append(probs[0][::-1] if swapped else probs[0])
            return out

        def hidden(pairs):
            out = []
            for state, q in pairs:
                eq, _ = to_engine_question(q, self.hide_index_keys, self.desnake_labels, self.describe_options)
                out.append(self.hidden_engine.readout(state, [eq])[0])
            return out

        from decisio.serve.tasks import task_key

        return self.task_store.register(
            task_id,
            [(q, s, g) for s, q, g in examples],
            score,
            hidden if self.hidden_engine is not None else None,
            key_fn=lambda q: task_key(q, render_text, self.described(q), self.noul_rendered(q)),
        )

    def with_readout_task(self, state, q, eq, p):
        """A text-route question's order-1 distribution as served: its registered task's head or calibration applied to
        the plain readout p (for abstention examples, so a threshold is fitted on what it will decide)."""
        t = self.readout_task(q)
        if t is None:
            return apply_temperature(p, self.temperature_of(q.type))
        if t["head"].get("applied") and self.hidden_engine is not None:
            from decisio.readout.intent_head import apply_intent_head

            lp, h = self.hidden_engine.readout(state, [eq])[0]
            return apply_intent_head(lp, h, t["head"], t["options"])
        if t["calibration"].get("applied"):
            from decisio.readout.calibration import apply_task_prior
            from decisio.readout.debias import log_probs

            return apply_task_prior(log_probs(p), t["calibration"])
        return apply_temperature(p, self.temperature_of(q.type))

    # ---- per-task abstention -------------------------------------------------------------------------------------

    def task_for(self, q, imajev_ext: bool) -> dict | None:
        if not self.abstention or not self.tasks:
            return None
        fp = option_set(q, self.described(q)) if q.type == "choice" else None
        for t in self.tasks.values():
            m = t["match"]
            if (m.get("option_set") and m["option_set"] == fp) or (m.get("imajev_extension") and imajev_ext):
                return t
        return None

    def abstention_plan(self, q, keys: list[str], imajev_ext: bool):
        """(abstain index, config, appended, append text) for a question whose task has an abstain option, else None.
        A declared option is the customer's own key; an appended one is served only with imajev's extension (a plain
        TypeSafe answer has no key for it) and only when the task's threshold was accepted (otherwise the question is
        served exactly as without the feature)."""
        t = self.task_for(q, imajev_ext)
        if t is None:
            return None
        opt, cfg = t["option"], t.get("config") or {}
        if opt.get("key") is not None:
            return (keys.index(opt["key"]), cfg, False, None) if opt["key"] in keys and q.type == "choice" else None
        if opt.get("append") and imajev_ext and cfg.get("applied"):
            n = 3 if q.type == "noul" else len(keys) + 1
            return (n - 1, cfg, True, opt["append"])
        return None

    def abstention_answer(self, q, keys: list[str], p, plan, imajev_ext: bool) -> dict:
        """The answer under the task's decision: `choice` (or imajev's `abstained`) follows the threshold; the
        probabilities are reported unchanged."""
        from decisio.serve.abstention import decide

        idx, cfg, appended, _ = plan
        abstain, best = decide(p, idx, cfg, keys=keys)
        if appended:
            return abstain_answer(q, keys, p, abstained=abstain)
        out = to_answer(q, keys, p)
        c = idx if abstain else best
        k = len(p)
        out["choice"] = keys[c]
        out["confidence"] = float(max(0.0, (p[c] - 1.0 / k) / (1.0 - 1.0 / k))) if k > 1 else 1.0
        if imajev_ext:
            out.update(unknown_probability=float(p[idx]), abstained=bool(abstain))
        return out

    def score_example(self, req: SystemOneRequest, images, option: dict, imajev_ext: bool):
        """One labelled example of a task, scored exactly as served: (p with the abstain option, its index, keys with
        it, the plain behaviour's distribution and keys). Declared option: one scoring, plain is the argmax over the
        customer's options. Appended option: scored with and without it (plain is the question without it)."""
        if len(req.questions) != 1:
            raise ValueError("an abstention example carries exactly one question")
        ((name, q),) = req.questions.items()
        eq, keys = to_engine_question(q, self.hide_index_keys, self.desnake_labels, self.describe_options)
        engine = self.image_engine if images else self.engine
        run = (
            (lambda qs: engine.answer(req.state, qs, None, images=images))
            if images
            else (lambda qs: engine.answer(req.state, qs, None))
        )
        if option.get("key") is not None:
            if q.type != "choice" or option["key"] not in keys:
                raise ValueError(f"the declared abstain option {option['key']!r} is not one of the question's options")
            probs, _ = run([eq])
            p = np.asarray(probs[0], dtype=np.float64)
            if not images:  # the threshold composes after calibration and the head
                p = self.with_readout_task(req.state, q, eq, p)
            return q, p, keys.index(option["key"]), keys, p, keys
        eq2, keys2 = with_abstain(eq, keys, option["append"])
        probs, _ = run([eq, eq2])
        T = 1.0 if images else self.temperature
        return (q, apply_temperature(probs[1], T), len(keys2) - 1, keys2, apply_temperature(probs[0], T), keys)

    def register_task(self, task_id: str, option: dict, match: str, examples: list[tuple]) -> dict:
        """Fit and store a task from labelled examples [(request, images, imajev_ext, gold)], gold None for
        unanswerable (decisio.serve.abstention.fit, its acceptance rule)."""
        from decisio.serve.abstention import decide, fit

        if bool(option.get("key") is not None) == bool(option.get("append")):
            raise ValueError("option: exactly one of {'key': <declared key>} or {'append': <text>}")
        if match not in ("option_set", "imajev_extension"):
            raise ValueError("match: 'option_set' or 'imajev_extension'")
        if option.get("append") and match != "imajev_extension" and not all(e[2] for e in examples):
            raise ValueError("an appended abstain option is answered only through imajev's extension")
        p_abs, unans, ok, plain_right, plain_false, fps = [], [], [], [], [], set()
        for req, images, ext, gold in examples:
            q, p, idx, keys, p0, keys0 = self.score_example(req, images, option, ext)
            fps.add(option_set(q, self.described(q)))
            gold_key = gold_key_of(q, gold)
            _, best = decide(p, idx, None, keys=keys)
            u = gold_key is None
            p_abs.append(float(p[idx]))
            unans.append(u)
            ok.append((not u) and keys[best] == gold_key)
            if option.get("key") is not None:
                plain_abs, _ = decide(p, idx, None, keys=keys)
                plain_right.append(plain_abs if u else (not plain_abs and keys[best] == gold_key))
                plain_false.append((not u) and plain_abs)
            else:
                plain_right.append((not u) and keys0[top_index(keys0, p0)] == gold_key)
                plain_false.append(False)
        if match == "option_set" and len(fps) != 1:
            raise ValueError("the examples of an option-set task must share one option set")
        cfg = fit(p_abs, unans, ok, plain_right, plain_false)
        task = {
            "id": task_id,
            "option": option,
            "match": {"option_set": fps.pop()} if match == "option_set" else {"imajev_extension": True},
            "config": cfg,
            "model": self.name,
        }
        self.tasks[task_id] = task
        return task


def gold_key_of(q, gold):
    """A labelled example's answer as an engine key: None (unanswerable), a choice key, "yes"/"no" for yes/no
    (booleans accepted), a level index for score."""
    if gold is None:
        return None
    if q.type == "noul":
        return {True: "yes", False: "no"}.get(gold, gold)
    return str(gold)


def add_routes(app, systemone: SystemOne):
    """Mount POST /v1/systemone and GET /v1/models on a FastAPI app. The body is exactly the wire object (plus imajev's
    two answer fields when a request uses its `images` extension); the server time and the route taken go in
    `x-decisio-server-ms` and `x-decisio-route` headers, never in the body.

    Accepted encodings (imajev's server accepts the same three): JSON with an optional `images` list of data URLs;
    multipart/form-data with the JSON request in the `request` field and up to two files in `image` (or `images`,
    `image[]`); data:image URIs anywhere in the state, which become images and are replaced by "[image N]"."""
    from fastapi import HTTPException, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse
    from pydantic import ValidationError

    @app.get("/v1/models")
    def models():
        return systemone.models()

    async def read(request: Request):
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        blobs = []
        if ctype == "multipart/form-data":
            form = await request.form()
            raw = form.get("request")
            if raw is None:
                raise HTTPException(422, "multipart requests need the 'request' form field")
            try:
                payload = json.loads(raw if isinstance(raw, str) else (await raw.read()).decode())
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise HTTPException(422, f"the 'request' form field is not valid JSON: {e}")
            if not isinstance(payload, dict):
                raise HTTPException(422, "the 'request' form field must be a JSON object")
            for key in ("image", "images", "image[]"):
                for f in form.getlist(key):
                    if isinstance(f, str):
                        raise HTTPException(422, f"form field {key!r} must be an uploaded file")
                    blobs.append(await f.read())
            ext = True
        else:
            try:
                payload = await request.json()
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise HTTPException(422, f"request body is not valid JSON: {e}")
            if not isinstance(payload, dict):
                raise HTTPException(422, "request body must be a JSON object")
            ext = "images" in payload
            imgs = payload.pop("images", None) or []
            if not isinstance(imgs, list):
                raise HTTPException(422, "'images' must be a list of data URLs")
            try:
                blobs += [decode_data_url(u) for u in imgs]
            except (ValueError, TypeError) as e:
                raise HTTPException(422, str(e))
        if payload.get("state") is not None:
            payload["state"], found = extract_state_images(payload["state"])
            if found:
                ext = True
                try:
                    blobs += [decode_data_url(u) for u in found]
                except (ValueError, TypeError) as e:
                    raise HTTPException(422, f"image in the state: {e}")
        return payload, blobs, ext

    async def system_one(request: Request):
        from decisio.serve.image_engine import MAX_IMAGES, load_image

        payload, blobs, ext = await read(request)
        try:
            req = SystemOneRequest.model_validate(payload)
        except ValidationError as e:
            detail = [
                {**{k: v for k, v in err.items() if k in ("type", "msg", "input", "ctx")}, "loc": ["body", *err["loc"]]}
                for err in e.errors(include_url=False)
            ]
            return JSONResponse({"detail": json.loads(json.dumps(detail, default=str))}, status_code=422)
        if len(blobs) > MAX_IMAGES:
            raise HTTPException(422, f"at most {MAX_IMAGES} images per request; got {len(blobs)}")
        try:
            images = [load_image(b) for b in blobs]
        except (ValueError, OSError) as e:
            raise HTTPException(422, f"bad image: {e}")
        route = request_header(request.headers, "route")
        if route not in (None, "text", "image"):
            raise HTTPException(422, "x-decisio-route must be 'text' or 'image'")
        debug = debug_of(request)
        try:
            out = await run_in_threadpool(systemone.answer, req, None, images, ext, route, debug)
        except (KeyError, ValueError, AssertionError) as e:
            raise HTTPException(400, str(e))
        timing = out.pop("_timing")
        headers = {header("server-ms"): f"{timing['server_ms']:.1f}", header("route"): timing["route"]}
        if timing.get("tasks"):
            headers[header("tasks")] = ",".join(timing["tasks"])
        # where the engine's time went, e.g. "prepare=1.2;warm=0.0;questions=52.1;readout=0.1;engine=53.6"
        if timing.get("stages"):
            headers[header("stages")] = ";".join(f"{k}={v:.1f}" for k, v in timing["stages"].items())
        return JSONResponse(out, headers=headers)

    def debug_of(request):
        """`x-decisio-debug: readout` (the served readout before any task, the path taken) or `hidden` (also the hidden
        readout of every question); only on a server started with --debug-readout (verification, never production)."""
        debug = request_header(request.headers, "debug")
        if debug is None:
            return None
        if not systemone.debug_readout:
            raise HTTPException(422, "x-decisio-debug needs a server started with --debug-readout")
        if debug not in ("readout", "hidden"):
            raise HTTPException(422, "x-decisio-debug must be 'readout' or 'hidden'")
        return debug

    # this module has `from __future__ import annotations`: give FastAPI the Request class itself, or it reads the
    # parameter as a missing query field (the same defect `/v1/answer` had)
    system_one.__annotations__["request"] = Request
    app.post("/v1/systemone")(system_one)

    # per-task abstention (decisio.serve.abstention; docs/handoffs/tasks.md): register a task from labelled
    # examples, each a /v1/systemone request (JSON, images as data URLs) with its answer (null: unanswerable)
    async def register(request: Request):
        from decisio.serve.image_engine import MAX_IMAGES, load_image

        body = await request.json()
        if not isinstance(body, dict) or not body.get("id") or not isinstance(body.get("examples"), list):
            raise HTTPException(422, "body: {'id', 'option', 'match', 'examples': [{'request', 'answer'}]}")
        examples = []
        for n, ex in enumerate(body["examples"]):
            payload = dict(ex.get("request") or {})
            ext = "images" in payload
            imgs = payload.pop("images", None) or []
            try:
                blobs = [decode_data_url(u) for u in imgs]
                if payload.get("state") is not None:
                    payload["state"], found = extract_state_images(payload["state"])
                    ext = ext or bool(found)
                    blobs += [decode_data_url(u) for u in found]
                if len(blobs) > MAX_IMAGES:
                    raise ValueError(f"at most {MAX_IMAGES} images per request")
                req = SystemOneRequest.model_validate(payload)
                images = [load_image(b) for b in blobs]
            except (ValueError, TypeError, OSError, ValidationError) as e:
                raise HTTPException(422, f"example {n}: {e}")
            examples.append((req, images, ext, ex.get("answer")))
        try:
            task = await run_in_threadpool(
                systemone.register_task,
                str(body["id"]),
                body.get("option") or {},
                body.get("match", "option_set"),
                examples,
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(422, str(e))
        return task

    register.__annotations__["request"] = Request
    app.post("/v1/abstention/tasks")(register)

    @app.get("/v1/abstention/tasks")
    def tasks():
        return {"enabled": systemone.abstention, "tasks": list(systemone.tasks.values())}

    # per-task calibration and the intent head (decisio.serve.tasks; docs/handoffs/tasks.md): register a task from
    # labelled examples, each a text /v1/systemone request with one question and its answer
    async def register_readout(request: Request):
        from decisio.serve.tasks import TaskStore

        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise HTTPException(422, f"request body is not valid JSON: {e}")
        if not isinstance(body, dict) or not body.get("id") or not isinstance(body.get("examples"), list):
            raise HTTPException(
                422, "body: {'id', 'examples': [{'request': <a /v1/systemone request, one question>, 'answer': <key>}]}"
            )
        debug = debug_of(request)
        examples = []
        for n, ex in enumerate(body["examples"]):
            if not isinstance(ex, dict):
                raise HTTPException(422, f"example {n}: must be an object {{'request', 'answer'}}")
            payload = dict(ex.get("request") or {})
            if "images" in payload or (payload.get("state") is not None and extract_state_images(payload["state"])[1]):
                raise HTTPException(422, f"example {n}: calibration and the intent head serve text requests only")
            try:
                req = SystemOneRequest.model_validate(payload)
            except ValidationError as e:
                raise HTTPException(422, f"example {n}: {e}")
            if len(req.questions) != 1:
                raise HTTPException(422, f"example {n}: an example carries exactly one question")
            examples.append((req.state, next(iter(req.questions.values())), ex.get("answer")))
        try:
            task, fitted = await run_in_threadpool(systemone.register_readout_task, str(body["id"]), examples)
        except (KeyError, ValueError) as e:
            raise HTTPException(422, str(e))
        out = TaskStore.public(task)
        if debug:
            out[DEBUG_KEY] = {"lps": [np.asarray(x).tolist() for x in fitted["lps"]], "labels": fitted["labels"]}
            if debug == "hidden" and fitted["readout"] is not None:
                out[DEBUG_KEY]["readout"] = [[lp.tolist(), h.tolist()] for lp, h in fitted["readout"]]
        return JSONResponse(json.loads(json.dumps(out, default=float)))

    register_readout.__annotations__["request"] = Request
    app.post("/v1/tasks")(register_readout)

    @app.get("/v1/tasks")
    def readout_tasks(full: int = 0):
        """Every registered task; `?full=1` includes the heads' arrays, the form `--tasks-file` loads."""
        from decisio.serve.tasks import TaskStore

        store = systemone.task_store
        return {
            "enabled": systemone.tasks_enabled,
            "head_engine": systemone.hidden_engine is not None,
            "fingerprint": store.fingerprint if store else None,
            "tasks": [TaskStore.public(t, bool(full)) for t in (store.by_key.values() if store else [])],
        }

    @app.post("/v1/tasks/import")
    def import_readout_tasks(body: dict):
        """Load tasks as `GET /v1/tasks?full=1` returns them (no re-scoring); refused when fitted under another model or
        rendering (re-register those from their examples)."""
        store = systemone.task_store
        if store is None:
            raise HTTPException(422, "this server has no task store")
        tasks = body.get("tasks") if isinstance(body, dict) else None
        if not isinstance(tasks, list):
            raise HTTPException(422, "body: {'tasks': [<task as GET /v1/tasks?full=1 returns it>]}")
        stale = [t.get("id") for t in tasks if not same_fingerprint(t.get("fingerprint"), store.fingerprint)]
        if stale:
            raise HTTPException(422, f"fitted under another model or rendering: {stale}")
        try:
            store.load(tasks)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(422, f"not a task record: {e}")
        return {"loaded": [t["id"] for t in tasks]}

    @app.delete("/v1/tasks/{task_id}")
    def remove_readout_task(task_id: str):
        if systemone.task_store is None or not systemone.task_store.remove(task_id):
            raise HTTPException(404, f"no task {task_id!r}")
        return {"removed": task_id}

    return app
