# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""TypeSafe's System One wire format (`POST /v1/systemone`, `GET /v1/models`) on the letters engine.

The request and response schemas are TypeSafe's own (typesafe-sdk 0.7.0, `_schemas/models.py`, generated from
api.typesafe.ai/openapi.json; MIT). Each question becomes one
question of the state-plus-questions engine and is rendered by the same letters readout as `/v1/answer`, so a request
whose criteria carry no descriptions scores exactly the token rows `/v1/answer` scores (a conformance check).
The mapping was fixed before any benchmark item was sent (runs/2026-09-27_boards-baseline):

  state         passed to the engine unchanged (a string as is; an object as `key: value` lines, like evaluation)
  instructions  a string as is; an object or array as compact JSON; absent or blank: a fixed default per type
  noul          instructions, then "Yes means: <true>" / "No means: <false>" lines when criteria are given; P(yes)
  choice        options in the request's key order, shown "<key>: <description>" or "<key>"; letters readout
  score         levels in criteria order as the letters legend; score = probability-weighted mean level
  confidence    the System One adapter's rules (choice: max rescaled from uniform; score: concentration at the mode)
  usage         input_tokens = state + each question's own text in this tokenizer (plus the header the layout lists, and
                the second copy of every row under read-twice: what is read); output_tokens = one per question

Two-order mode (after Reflex, github.com/kshetrajna12/reflex, MIT, `prompt.distinct_orders` and
`ensemble.disagreement`): each question is also read in one distinct second order (seeded by its name; the swap for two
options; "Answer no or yes." for yes/no), the two distributions are averaged per label, and the disagreement (mean
total-variation distance of each branch from their average) goes to a branch log, never into the response.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import random
import re
import threading
import time
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from decisio import flavors
from decisio.names import DEBUG_KEY, check_format, header, request_header, same_fingerprint
from decisio.serve import header_limit
from decisio.serve import over_limit as over
from decisio.serve import think as thinking
from decisio.serve.adapter_temperatures import band_table
from decisio.serve.engine_health import client_error
from decisio.serve.task_backends import (
    WRITE_INTENT,
    TaskConflict,
    TaskPersistError,
    TaskStoreReadOnly,
    TaskStoreUnavailable,
    WriteIntent,
)
from decisio.serve.temperature import apply_temperature

JSONContent = Any  # str | dict | list, as the SDK's schema allows

HOLDOUT_RULE = (
    "examples with the same input go to the same side; groups are ordered by the sha256 of the task id and the input "
    "and the first ones, up to the held-out share, certify; the fit sees the rest"
)
SUPPLIED_RULE = "the certification examples were supplied separately and share no input with the fit examples"


# ---- wire schema (TypeSafe's SystemOneRequest and its questions) -----------------------------------------------------


logger = logging.getLogger(__name__)


class NoulCriteria(BaseModel):
    true: JSONContent | None = None
    false: JSONContent | None = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


# System One's cap on the options of a choice (`maxProperties: 255` in its OpenAPI): more is a 422 `too_long`, as there
MAX_CHOICE_OPTIONS = 255


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None] = Field(..., min_length=1, max_length=MAX_CHOICE_OPTIONS)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JSONContent | None = None
    criteria: list[JSONContent] = Field(..., min_length=1)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class Asset(BaseModel):
    """System One's image asset: an image that replaces its placeholder, with its bytes in `data_base64` (and their
    `format`), or, in a multipart request, the name of the image part that holds them."""

    type: Literal["image"] = "image"  # System One's schema requires nothing of an asset
    format: Literal["png", "jpg", "jpeg"] | None = None
    data_base64: str | None = None
    part: str | None = None


class TaskApplier:
    """What a shared task store applies other replicas' tasks through (decisio.serve.task_store_shared): as an import
    does, nothing is fitted again; a record of another stack is kept and not applied, a certificate made under another
    serving stack is suspended when it is served."""

    def __init__(self, systemone):
        self.so = systemone

    def apply(self, kind: str, records: list) -> None:
        if kind == "readout":
            if self.so.task_store is not None:
                self.so.task_store.load(records, persist=False)
        else:
            self.so.load_abstention(records, persist=False)

    def drop(self, kind: str, task_id: str) -> None:
        if kind == "readout":
            if self.so.task_store is not None:
                self.so.task_store.remove(task_id, persist=False)
        else:
            self.so.tasks.pop(task_id, None)


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # extensions (`orders`) and fields TypeSafe may add later are ignored
    state: str | dict[str, Any] | list[Any]
    # TypeSafe's API requires `model`; imajev's requests (github.com/mohit67890/imajev, Apache-2.0) omit it, so it is
    # optional here and the response names the served model when it is absent
    model: str | None = None
    questions: dict[str, Question] = Field(..., min_length=1)
    # System One's images: placeholders in the state and the questions' texts, replaced by these (`bind_assets` does it
    # before the request is validated, so a validated request never carries them)
    assets: dict[str, Asset] | None = Field(None, min_length=1)


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
    """A yes/no question as a two-option choice read from the option letters (--noul-rendering letters): the false
    side first, each side shown as its criteria description, "No" and "Yes" for a side without one;
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


COMMIT_LO, COMMIT_HI = 0.20, 0.80


# A certified task whose traffic alert is on stops certifying its answers (`certified_reason: traffic_changed`) until
# the alert clears, unless its certify block says `"suspend_on_alert": false`. The promise is for traffic like the
# examples.
SUSPEND_ON_ALERT = True

# a catch-all option: the words of an option's key that say "none of the above"
CATCH_ALL = re.compile(
    r"\b(other|others|none|neither|unknown|unclear|irrelevant|miscellaneous|misc|else|nonsense|oos|"
    r"out[ _-]?of[ _-]?(scope|domain)|not[ _-]?applicable|n/?a|no[ _-]?intent|can'?t[ _-]?tell|cannot[ _-]?tell)\b"
)


def has_catch_all(keys) -> bool:
    return any(CATCH_ALL.search(str(k).lower().replace("_", " ")) for k in keys)


def commit_noul(p: np.ndarray) -> np.ndarray:
    """--noul-commit: a yes/no answer with 0.20 < P(yes) < 0.80 is reported at the band's edge on its own side, 0.80 if
    P(yes) > 0.5 and 0.20 otherwise (exactly 0.5 stays no, the served tie-break); outside the band it is unchanged. An
    output transform: the answer never changes, its probability does."""
    py = float(p[0])
    if COMMIT_LO < py < COMMIT_HI:
        py = COMMIT_HI if py > 0.5 else COMMIT_LO
    return np.array([py, 1.0 - py])


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


class UnknownAdapter(LookupError):
    """A request named an adapter this server did not load (a 404 that names it and the ones that are loaded)."""


class NonFiniteReadout(RuntimeError):
    """An engine returned NaN or infinity as a probability. It is a fault in the model's arithmetic on this server (a
    dtype or matmul precision the checkpoint does not survive, a checkpoint that does not belong to the base's
    profile), not in the request, so the route answers 500 with a message that says so (`add_routes`), never 400."""


def require_finite(p, what: str = "an engine probability") -> None:
    if not np.isfinite(np.asarray(p, dtype=np.float64)).all():
        raise NonFiniteReadout(f"{what} is not finite (NaN or infinity)")


def top_index(keys, p, among=None) -> int:
    """The declared choice: the most probable option; among options whose probabilities are exactly equal, the one whose
    key sorts first (Unicode code-point order), so the choice never depends on the order of the keys on the wire (JSON
    objects are unordered, and clients that serialise with sorted keys reorder them; docs/handoffs/tasks.md,
    "Exact ties"). `among`: the candidate indices (default: all)."""
    idx = list(range(len(p))) if among is None else list(among)
    require_finite([p[i] for i in idx], "a choice question's probability")
    m = max(p[i] for i in idx)
    return min((i for i in idx if p[i] == m), key=lambda i: keys[i])


def to_answer(q, keys, p) -> dict:
    p = [float(x) for x in p]
    require_finite(p, f"a {q.type} question's probability")
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


class ImageRequestError(ValueError):
    """An image problem in a request that uses `assets`. System One answers these with 400 and
    `{"detail": {"error_type": "api_usage_error", "message": ...}}` (the route's handler does the same)."""


def bind_assets(payload: dict, parts: dict[str, bytes] | None = None):
    """System One's `assets`: every placeholder (an asset's name, found as or inside a string of the state, a question's
    instructions or its criteria's descriptions, longest name first) becomes the marker "[image N]", N counting the
    occurrences in reading order (the state, then each question: instructions, then criteria), and the asset's bytes
    become image N, once per occurrence, exactly as a data:image URI in the state does (`extract_state_images`). Option
    keys are not searched. Returns (the payload without `assets`, the images' bytes in order, the score questions'
    original levels by name, so the legend can show the text the caller wrote). `parts`: a multipart request's image
    parts by name. Raises ImageRequestError for an asset that is not usable or never used, an image part that no asset
    uses or two assets share."""
    assets = payload.pop("assets", None)
    if assets is None:
        return payload, [], {}
    if not isinstance(assets, dict) or not assets:
        raise ImageRequestError("assets must be a non-empty object of placeholder names to images.")
    data: dict[str, bytes] = {}
    used_parts: set[str] = set()
    for name, raw in assets.items():
        if not isinstance(name, str) or not name:
            raise ImageRequestError("Each asset needs a nonempty placeholder name.")
        try:
            asset = Asset.model_validate(raw)
        except ValidationError:
            raise ImageRequestError(f'Asset "{name}": an image asset has type "image" and format png, jpg or jpeg.')
        if asset.part is not None:
            if not parts or asset.part not in parts:
                raise ImageRequestError(f'Asset "{name}": no image part named "{asset.part}" in the request.')
            if asset.part in used_parts:
                raise ImageRequestError(f'Image part "{asset.part}" is used by more than one asset.')
            used_parts.add(asset.part)
            data[name] = parts[asset.part]
            continue
        if asset.format is None:
            raise ImageRequestError(f'Asset "{name}": format must be png, jpg or jpeg.')
        try:
            blob = base64.b64decode(asset.data_base64 or "", validate=True)
        except ValueError:
            raise ImageRequestError(f'Asset "{name}": data_base64 is not valid base64.')
        if not blob:
            raise ImageRequestError(f'Asset "{name}": data_base64 is empty.')
        data[name] = blob
    unused_parts = sorted(set(parts or ()) - used_parts)
    if unused_parts:
        raise ImageRequestError(f'Image part "{unused_parts[0]}" is not used by any asset.')

    pattern = re.compile("|".join(re.escape(n) for n in sorted(data, key=len, reverse=True)))
    blobs: list[bytes] = []
    used: set[str] = set()

    def sub(text: str) -> str:
        def one(m):
            blobs.append(data[m.group(0)])
            used.add(m.group(0))
            return f"[image {len(blobs)}]"

        return pattern.sub(one, text)

    def walk(node):
        if isinstance(node, str):
            return sub(node)
        if isinstance(node, list):
            return [walk(x) for x in node]
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        return node

    legends: dict[str, list] = {}
    out = dict(payload)
    if "state" in out:
        out["state"] = walk(out["state"])
    if isinstance(out.get("questions"), dict):
        qs = {}
        for qname, q in out["questions"].items():
            if not isinstance(q, dict):
                qs[qname] = q
                continue
            q = dict(q)
            if "instructions" in q:
                q["instructions"] = walk(q["instructions"])
            crit = q.get("criteria")
            if isinstance(crit, dict):  # option keys are answers' names: only the descriptions are searched
                q["criteria"] = {k: walk(v) for k, v in crit.items()}
            elif isinstance(crit, list):
                if q.get("type") == "score":
                    legends[qname] = list(crit)
                q["criteria"] = walk(crit)
            qs[qname] = q
        out["questions"] = qs
    unused = [n for n in data if n not in used]
    if unused:
        raise ImageRequestError(f'Asset "{unused[0]}" is not used: its name appears in no text of the request.')
    return out, blobs, legends


def option_set(q, described: bool = False) -> str:
    """A choice question's task identity for abstention: its options (keys and descriptions, in any order). The same
    question asked of different states keeps its options, whatever its instructions (the Decision Index puts the
    utterance there). `described`: as `decisio.serve.tasks.task_key`, the --describe-options rule changes how this
    question is shown, so a threshold fitted under the other rendering is not applied."""
    crit = sorted([k, render_text(d)] for k, d in q.criteria.items()) if q.type == "choice" else None
    spec = {"type": q.type, "criteria": crit, **({"rendering": "describe_options"} if described else {})}
    return hashlib.sha256(json.dumps(spec, ensure_ascii=False).encode()).hexdigest()


def option_keys(q) -> str:
    """A choice question's task identity by its option keys alone: the set of keys, whatever their descriptions, their
    order and the question's instructions. For a family of questions that share their labels but not their texts (a
    multiple-choice exam: the keys A to D and "idk", each item with its own four answers), where `option_set` would see
    a different question every time."""
    spec = {"type": q.type, "keys": sorted(q.criteria) if q.type == "choice" else None}
    return hashlib.sha256(json.dumps(spec, ensure_ascii=False).encode()).hexdigest()


def _serial(method):
    """A method whose engine reads take the serial path on a batched server (decisio.serve.combine.serial_reads): a
    certified task's answers and a task's registration and certification reads stay a function of the request alone,
    so a certificate is made and served on reads no other traffic shares."""
    import functools

    from decisio.serve.combine import serial_reads

    @functools.wraps(method)
    def run(*args, **kwargs):
        with serial_reads():
            return method(*args, **kwargs)

    return run


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
        noul_commit: bool = False,
        abstention: bool = True,
        abstention_tasks: list[dict] | None = None,
        task_backend=None,
        task_report: dict | None = None,
        tasks_enabled: bool = True,
        task_store=None,
        hidden_engine=None,
        debug_readout: bool = False,
        temperature: float = 1.0,
        temperatures: dict | None = None,
        adapter_temperatures: dict | None = None,
        think_temperatures: dict | None = None,
        choice_bands: dict | None = None,
        over_limit: str = "refuse",
        think: bool = False,
        think_factor: float = thinking.DEFAULT_FACTOR,
        think_budget: int = thinking.DEFAULT_BUDGET,
        flavor: dict | None = None,
        default_adapter: str | None = None,
        batch_invariant: dict | None = None,
        precision_casts: dict | None = None,
    ):
        self.engine, self.image_engine, self.name, self.orders = engine, image_engine, served_name, orders
        # a named configuration (decisio.flavors): its name, overrides and pending values; None for a server with no
        # flavor, which is then exactly what it was (the model it echoes, the fingerprint, the headers)
        self.flavor = dict(flavor) if flavor else None
        # the loaded adapter that answers a request naming none (--default-adapter); "none" asks for the base
        self.default_adapter = default_adapter
        # vLLM's batch-invariant kernels (decisio.serve.batch_invariant): {"on", "source", "confirmed"}; None where the
        # server does not run them, which is then exactly what it was
        self.batch_invariant = dict(batch_invariant) if batch_invariant and batch_invariant.get("on") else None
        # inductor's emulate_precision_casts (decisio.serve.precision_casts): {"on", "source", "confirmed"}; None where
        # the server does not run it, which is then exactly what it was
        self.precision_casts = dict(precision_casts) if precision_casts and precision_casts.get("on") else None
        self.hide_index_keys = hide_index_keys  # to_engine_question; the served default is on
        self.desnake_labels = desnake_labels  # likewise
        self.describe_options = describe_options  # likewise
        if noul_rendering not in NOUL_RENDERINGS:
            raise ValueError(f"noul_rendering must be one of {NOUL_RENDERINGS}")
        self.noul_rendering = noul_rendering  # how a yes/no question is asked (noul_as_letters); words by default
        self.noul_commit = bool(noul_commit)  # an output transform on yes/no answers (commit_noul); off by default
        # a state over the engine's prompt ceiling: refused (the engine's 422) or cut to its start (decisio.serve.
        # over_limit); a request's "over_limit" field decides for that request. Counted in /health.
        self.over_limit = over.check_mode(over_limit, "over_limit")
        self.truncated = {"requests": 0, "largest_state_tokens": 0}
        # requests whose header listed only the questions that fit the context (decisio.serve.header_limit), those of
        # them that fell back to the plain layout (no line fit), those refused because even that did not fit
        self.header_limited = {"requests": 0, "plain": 0, "refused": 0, "lines_dropped": 0}
        # think when unsure (decisio.serve.think): the server's default (a request's "think" field decides for that
        # request), the factor on the type thresholds and the token budget; counted in /health
        self.think = bool(think)
        self.think_factor = thinking.check_factor(think_factor, "think_factor")
        self.think_budget = thinking.check_budget(think_budget, "think_budget")
        self.thought = {"requests": 0, "questions": 0, "unavailable": 0, "tokens": 0, "ms": 0.0}
        # opt-in: requests using imajev's extension are offered one more option, whose probability is reported as
        # imajev's unknown_probability
        self.abstain_option = abstain_option
        # per-task abstention thresholds (decisio.serve.abstention; docs/handoffs/tasks.md): tasks registered
        # from labelled examples; `abstention=False` ignores them all
        self.abstention = abstention
        for t in abstention_tasks or []:  # records written before the rename load unchanged
            check_format(t.get("config"), "abstention", f"abstention task {t.get('id')!r}")
        self.tasks: dict[str, dict] = {t["id"]: t for t in (abstention_tasks or [])}
        # where taught tasks survive a restart (decisio.serve.task_backends; docs/tasks.md): both kinds are written
        # through before they are stored in memory; None keeps them in memory only. `task_report`: what the start loaded
        self.task_backend = task_backend
        if task_store is not None:
            task_store.backend = task_backend
        self.task_report = task_report or {}
        if task_backend is not None and hasattr(task_backend, "applier"):
            task_backend.applier = TaskApplier(self)  # a shared store applies other replicas' tasks through this
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
        # the choice temperature by the question's number of options, {"lo-hi": temperature}, replacing the choice
        # temperature above for a question whose option count falls in a band (a tuned repository's defaults,
        # decisio.hub); the adapter's own bands (below) come first
        self.choice_bands = {b: float(t) for b, t in (choice_bands or {}).items()}
        self._base_bands = band_table(self.choice_bands) if self.choice_bands else []
        # per question type, the temperature of an answer that thought (the mix of the plain read and the read after the
        # reasoning, decisio.serve.think), replacing the type's single-pass temperature for those answers only; a type
        # without one keeps the single-pass temperature (a tuned repository's `think_temperatures`)
        self.think_temperatures = {k: float(v) for k, v in (think_temperatures or {}).items() if v is not None}
        # per adapter, {question type: temperature} replacing the base's for a request that names the adapter
        # (decisio.serve.adapter_temperatures); a type the adapter has none for keeps the base's. A "choice_bands"
        # entry, {"lo-hi": temperature}, is the adapter's choice temperature by the question's number of options; a
        # "think" entry, {question type: temperature}, is its temperature for the answers that thought
        self.adapter_temperatures = {
            n: {
                k: ({b: float(x) for b, x in v.items()} if k in ("choice_bands", "think") else float(v))
                for k, v in t.items()
            }
            for n, t in (adapter_temperatures or {}).items()
            if t
        }
        self._adapter_bands = {
            n: band_table(b) for n, t in self.adapter_temperatures.items() if (b := t.get("choice_bands"))
        }
        self.release_date, self.description = release_date, description
        self.branch_log = branch_log
        self._log_lock = threading.Lock()
        self._probe = threading.local()  # set while a registration re-reads its sample through the serving path

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

    def usage_tokens(self, state, qs: list[dict], header_lines=None) -> int:
        from decisio.readout.letters import DEFAULT_FORMAT, READ_AGAIN
        from decisio.serve.vllm_engine import counted_tokens

        tok, fmt = self.engine.tok, getattr(self.engine, "fmt", DEFAULT_FORMAT)
        c = counted_tokens(tok, fmt, state, qs, header_lines)
        total = c["total"]
        if fmt.read_twice_under and self.engine.read_plan(state, qs)["twice"]:
            # what is read: every question's row reads its whole user turn again (the header, the state, its own
            # text, and the line that joins them), none of it a cached prefix
            again = len(tok.encode(READ_AGAIN, add_special_tokens=False))
            total += sum(c["header"] + c["state"] + a + again for a in c["asked"])
        return total

    def think_pass(
        self,
        state,
        names,
        wire,
        mapped,
        swapped,
        qs,
        at,
        probs,
        plans,
        rts,
        orders,
        abstain,
        use_image,
        adapter,
        engine,
    ):
        """({question index: log-probabilities of the second read in the question's own option order}, the report, ms):
        every question that is unsure at its served confidence (decisio.serve.think) is read again after the served
        model has written its reasoning, in one call of the engine. A question a registered task answers, an abstention
        or an image request, a request of two orders and a question of more than 16 options stay at one read."""
        if not (use_image or orders != 1 or abstain) and not getattr(engine, "can_think", lambda: False)():
            raise thinking.ThinkUnavailable(
                "this server's engine cannot write reasoning (think needs the vLLM engine); send the request without "
                '"think": true'
            )
        why, cand = {}, []
        for i, name in enumerate(names):
            if use_image or orders != 1 or abstain or plans[i]:
                why[name] = thinking.NOT_COMBINED
            elif rts[i] is not None or i not in at:
                why[name] = thinking.REGISTERED_TASK
            elif len(mapped[i][1]) > thinking.MAX_OPTIONS:
                why[name] = thinking.TOO_MANY_OPTIONS
            else:
                p = np.asarray(probs[at[i]], dtype=np.float64)
                p = p[::-1].copy() if swapped[i] else p
                t_q = self.temperature_of(wire[i].type, adapter, len(p))
                served = apply_temperature(p, t_q) if t_q != 1.0 else p
                if thinking.is_unsure(served, wire[i].type, self.think_factor):
                    cand.append(i)
                else:
                    why[name] = thinking.SURE
        second, per_question, tokens, ms, unavailable = {}, {}, 0, 0.0, 0
        if cand:
            t = time.perf_counter()
            asked = [qs[i] for i in sorted(at, key=at.get)]  # the questions as the first read had them
            lps, counts, parts = engine.think(state, asked, adapter, self.think_budget, [at[i] for i in cand])
            wall = (time.perf_counter() - t) * 1000
            ms = float(parts.get("generate_ms", 0.0) + parts.get("read_ms", 0.0)) or wall  # the engine's own timing
            tokens = sum(counts)
            bad = parts.get("unavailable") or {}
            for k, (i, lp, n) in enumerate(zip(cand, lps, counts)):
                if lp is None or k in bad:  # no usable reasoning: this question keeps its single-pass answer
                    cause = bad.get(k, "no second read")
                    per_question[names[i]] = {"thought": False, "thinking": "unavailable", "why": cause}
                    unavailable += 1
                    logger.warning("think unavailable for %r: %s; its single-pass answer is served", names[i], cause)
                    continue
                second[i] = np.asarray(lp)[::-1].copy() if swapped[i] else np.asarray(lp)
                per_question[names[i]] = {"thought": True, "tokens": n}
        for name, reason in why.items():
            per_question[name] = {"thought": False, "why": reason}
        self.thought["requests"] += 1
        self.thought["questions"] += len(cand) - unavailable
        self.thought["unavailable"] += unavailable
        self.thought["tokens"] += tokens
        self.thought["ms"] += ms
        report = {
            "questions": len(cand) - unavailable,
            **({"unavailable": unavailable} if unavailable else {}),
            "of": len(names),
            "tokens": tokens,
            "ms": round(ms, 1),
            "factor": self.think_factor,
            "budget": self.think_budget,
            "per_question": {n: per_question[n] for n in names},
        }
        return second, report, ms

    def header_plan(self, state, qs: list[dict], engine) -> dict | None:
        """The engine's header plan for the request (decisio.serve.header_limit): None where the header layout is not in
        use or nothing was over; a request that cannot be answered even from the plain layout, or with the limit off
        and its header over the context, is refused here with where its tokens are (a 422)."""
        plan = getattr(engine, "header_plan", None)
        hp = plan(state, qs) if plan is not None else None
        if hp is None:
            return None
        if not hp["fits"]:
            self.header_limited["refused"] += 1
            off = bool(hp.get("off"))
            raise over.OverLimit(header_limit.refusal(hp["limit"], hp["pieces"], off))
        if hp["limited"]:
            self.header_limited["requests"] += 1
            self.header_limited["plain"] += 1 if hp["listed"] == 0 else 0
            self.header_limited["lines_dropped"] += hp["of"] - hp["listed"]
        return hp

    def truncated_request(self, req: SystemOneRequest, qs: list[dict], engine) -> tuple[SystemOneRequest, dict | None]:
        """(the request, the report): the request with its state cut to its start where a row would be over the
        engine's prompt ceiling, and what was cut; the request itself and None where the whole state fits.
        decisio.serve.over_limit says how the cut is made: once, to what the longest question leaves."""
        from decisio.readout.letters import DEFAULT_FORMAT, fmt_state, layout_header
        from decisio.serve.vllm_engine import question_text, user_turn

        limit = getattr(engine, "prompt_limit", lambda: None)()
        if limit is None:
            raise over.OverLimit(
                "this engine has no prompt ceiling to cut to; start the server with --over-limit-tokens N, or send "
                'the request without "over_limit": "truncate"'
            )
        tok, fmt = engine.tok, getattr(engine, "fmt", DEFAULT_FORMAT)
        enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
        # a row is the user turn of the header (under the header layout: every question of the request, listed before
        # the state), the state, a blank line and the question; the budget the state may take is what is left of the
        # ceiling beside the longest such row, so the header counts as much as the question
        # (with the header limit on, the header gives way before the state does: the state is cut for the plain row)
        limited = hasattr(engine, "header_plan") and getattr(engine, "header_limit_setting", ("auto", None))[0] != "off"
        keep_header = fmt.layout == "header" and not limited
        head = layout_header([q["instructions"] for q in qs]) if keep_header else ""
        overhead = max(len(enc(user_turn(tok, f"{head}\n\n{question_text(tok, q, fmt)[0]}", fmt))) for q in qs)
        text, report = over.fit_state(
            tok, fmt_state(req.state), overhead, limit, reserve=int(getattr(engine, "pad_unit", 0) or 0)
        )
        if report is None:
            return req, None
        self.truncated["requests"] += 1
        self.truncated["largest_state_tokens"] = max(self.truncated["largest_state_tokens"], report["of_tokens"])
        return req.model_copy(update={"state": text}), report

    def answer(
        self,
        req: SystemOneRequest,
        adapter=None,
        images=None,
        imajev_ext=False,
        route=None,
        debug=None,
        over_limit=None,
        think=None,
    ) -> dict:
        """`images`: decoded PIL images (the request is then served by the image engine); `imajev_ext`: the request used
        imajev's `images` extension, so every answer also carries `unknown_probability` and `abstained` (0.0 and false:
        the letters readout has no unknown outcome); `route="image"` sends a text request to the image engine (the
        paired check). `adapter`: the name of a loaded LoRA adapter (a request's `"adapter"` field), served on the
        text route; the registered tasks were fitted on the base's readout, so a question one matches is refused.
        `over_limit`: this request's choice for a state over the engine's prompt ceiling ("refuse" or "truncate"; the
        server's own when absent). A truncated state is cut to its start, reported as `state_truncated`, and none of
        the request's answers is certified. `think`: this request's choice for think when unsure (True or False; the
        server's own when None): an unsure question is read again after the served model has written its reasoning,
        and the response's `thinking` says which questions did and what it cost."""
        images = list(images or [])
        use_image = bool(images) or route == "image"
        if use_image and self.image_engine is None:
            raise ValueError("this server has no image engine (start it with --image-model)")
        adapter = adapter or None
        if adapter is not None:
            if use_image:
                raise ValueError("an adapter is served on the text route only")
            loaded = getattr(self.engine, "adapters", None) or {}
            if adapter not in loaded:
                raise UnknownAdapter(f"unknown adapter {adapter!r}; loaded: {sorted(loaded)}")
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
        cut = None
        if (over_limit or self.over_limit) == "truncate" and not use_image:
            req, cut = self.truncated_request(req, qs, engine)
        hp = None if use_image else self.header_plan(req.state, qs, engine)
        header_lines = hp["listed"] if hp is not None and hp["limited"] else None
        hl = hp["report"] if hp is not None and hp["limited"] else None
        if adapter is not None:
            # a task (calibration, intent head, certificate, abstention threshold) was fitted on the base's readout and
            # would be applied to another one: refuse rather than answer with a correction that does not fit
            clash = sorted(
                {t["id"] for t in (self.readout_task(q) for q in wire) if t is not None}
                | {t["id"] for t in (self.task_for(q, imajev_ext) for q in wire) if t is not None}
            )
            if clash:
                raise ValueError(
                    f"adapter {adapter!r}: the question matches the registered task(s) {clash}, which were fitted on "
                    "the base's readout; ask without the adapter, or delete the task"
                )
        # per-task calibration and head: the registered task of each text-route question (None when there is none)
        rts = [
            self.readout_task(q) if not use_image and not (plans[i] and plans[i][2]) and not abstain else None
            for i, q in enumerate(wire)
        ]
        # a question of a certified task is scored by the function its certificate was made with (`served_rows`): one
        # question at a time, in request order, never batched with the request's other questions (docs/tasks.md)
        own = [
            i
            for i, t in enumerate(rts)
            if t is not None
            and t.get("certificate")
            and wire[i].type != "score"
            and orders == 1
            and not abstain
            and not use_image
        ]
        served = {i: self.served_rows(rts[i], [(req.state, wire[i])])[0] for i in own}
        head = [
            i
            for i, t in enumerate(rts)
            if t is not None and t["head"].get("applied") and self.hidden_engine and i not in served
        ]
        if head and orders == 2:
            raise ValueError(
                "the intent head is not combined with two-order averaging: a question of a task with a "
                "head was asked with orders=2 (the head is bound to one option order)"
            )
        gen = [i for i in range(len(names)) if i not in head and i not in served]
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
        effective_think = self.think if think is None else bool(think)
        second, think_report, think_ms = {}, None, 0.0
        if effective_think:
            second, think_report, think_ms = self.think_pass(
                req.state, names, wire, mapped, swapped, qs, at, probs, plans, rts, orders, abstain, use_image,
                adapter, engine,
            )  # fmt: skip
        hidden = dict(zip(head, self.hidden_engine.readout(req.state, [qs[i] for i in head]))) if head else {}
        dbg = {}
        if debug == "hidden" and self.hidden_engine is not None:  # verification: every question's hidden readout
            for i, (lp, h) in enumerate(self.hidden_engine.readout(req.state, qs)):
                dbg[names[i]] = {"hidden_lp": lp.tolist(), "h": h.tolist()}
        answers, log = {}, []
        for i, (name, q, (eq, keys)) in enumerate(zip(names, wire, mapped)):
            t = rts[i]
            if i in served:  # a certified task's question: the registration's own function, already applied
                p = served[i]
                if swapped[i]:
                    keys = ["yes", "no"]
                if debug:
                    dbg.setdefault(name, {}).update(path="served", task=t["id"], p=p.tolist())
            elif i in hidden:  # the intent head (reference arithmetic)
                from decisio.readout.intent_head import apply_intent_head

                lp, h = hidden[i]
                p = apply_intent_head(lp, h, t["head"], t["options"])
                if debug:
                    dbg.setdefault(name, {}).update(path="head", task=t["id"], hidden_lp=lp.tolist(), h=h.tolist())
            else:
                p = np.asarray(probs[at[i]], dtype=np.float64)
                if swapped[i]:  # read false first; from here on (yes, no), as every yes/no readout
                    p, keys = p[::-1].copy(), ["yes", "no"]
                if i in second:  # an unsure question: the plain read and the read after reasoning, mixed in log space
                    if debug:
                        dbg.setdefault(name, {}).update(think_p1=p.tolist(), think_lp2=second[i].tolist())
                    p = thinking.mix(p, second[i])
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
                    if forms:  # --label-variants summed: each form's log-probability, in engine order
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
                t_q = (self.think_temperature_of if i in second else self.temperature_of)(q.type, adapter, len(p))
                if not corrected and t_q != 1.0 and not use_image:
                    p = apply_temperature(p, t_q)
                    if debug:
                        dbg[name].update(path="temperature", temperature=t_q, adapter=adapter)
            if plans[i]:
                answers[name] = self.abstention_answer(q, keys, p, plans[i], imajev_ext)
                continue
            if abstain:
                answers[name] = abstain_answer(q, keys, p)
                continue
            # --noul-commit, the output transform, last (abstention answers above keep theirs)
            if self.noul_commit and q.type == "noul" and i not in served:  # served rows are committed already
                p = commit_noul(p)
            answers[name] = to_answer(q, keys, p)
            if cut is not None and t is not None and t.get("certificate") and q.type != "score":
                # the promise is for whole inputs like the examples; the monitor does not see a cut state's answer
                answers[name].update(
                    {"certified": False, "certificate": t["certificate"]["id"], "certified_reason": "state_truncated"}
                )
            elif hl is not None and t is not None and t.get("certificate") and q.type != "score":
                # the certificate was made with the whole header; this request's lists fewer of its questions
                answers[name].update(
                    {"certified": False, "certificate": t["certificate"]["id"], "certified_reason": "header_limited"}
                )
            elif t is not None and t.get("certificate") and q.type != "score":
                # an answer scored in an engine call shared with other requests (--concurrency batched; never a
                # certified task's own read, which is serial) is not observed against the certificate's baseline
                shared = i not in served and bool(info.get("batch"))
                answers[name].update(self.certified_fields(t, q, keys, p, orders, top_index(keys, p), shared))
            elif t is None and not use_image and adapter is None and q.type != "score":
                stale = self.stale_certified_task(q)
                if stale is not None:  # fitted under another model or rendering: not applied, so not certified
                    answers[name].update(
                        {
                            "certified": False,
                            "certificate": stale["certificate"]["id"],
                            "certified_reason": "serving_stack_changed",
                        }
                    )
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
            "input_tokens": self.usage_tokens(req.state, qs, header_lines) + sum(info.get("image_tokens") or []),
            "output_tokens": len(names),
        }
        if think_report is not None:  # the generated tokens are output tokens, and said so
            usage["output_tokens"] += think_report["tokens"]
            usage["thinking_tokens"] = think_report["tokens"]
        out = {
            "model": self.name if self.flavor else req.model or self.name,
            "answers": answers,
            "usage": usage,
            "_timing": {
                "server_ms": (time.perf_counter() - t0) * 1000,
                "route": "image" if use_image else "text",
                "layout": "plain" if hl is not None and hl["listed"] == 0 else self.layout,
                "tasks": sorted({t["id"] for t in rts if t is not None}),
                # the mean prompt the engine computed or read per question, for the x-decisio-prompt header
                **(
                    {"prompt_tokens_mean": float(np.mean(info["engine_prompt_tokens"]))}
                    if info.get("engine_prompt_tokens")
                    else {}
                ),
                **{
                    k: v
                    for k, v in info.items()
                    if k in ("cached_tokens_mean", "image_tokens", "waited_behind_registration", "batch")
                },
                # where the engine's time went, for the x-decisio-stages header (milliseconds)
                "stages": {
                    {"server_ms": "engine", "waited_ms": "wait"}.get(k, k[:-3]): info[k]
                    for k in ("prepare_ms", "warm_ms", "questions_ms", "readout_ms", "server_ms", "waited_ms")
                    if isinstance(info.get(k), (int, float))
                },
            },
        }
        if adapter is not None:
            out["adapter"] = adapter  # which adapter answered; absent when the base did
        if cut is not None:
            out["state_truncated"] = cut  # absent when the whole state was read
        if hl is not None:
            out["header_limited"] = hl  # absent when the header listed every question
        if think_report is not None:
            out["thinking"] = think_report  # present when thinking was on for the request
            out["_timing"]["think_ms"] = think_ms
            out["_timing"]["think_tokens"] = think_report["tokens"]
            out["_timing"]["stages"]["think"] = think_ms
        if debug and self.debug_readout:
            out[DEBUG_KEY] = dbg
        return out

    # ---- per-task calibration and the intent head ----------------------------------------------------------------

    def temperature_of(self, qtype: str, adapter: str | None = None, options: int | None = None) -> float:
        """The temperature a question of this type is served at: the named adapter's own for the type (for a choice
        question of `options` options, its temperature for the band that count falls in, if it has bands), else, for a
        choice question, the band of the server's own bands (a tuned repository's), else the type's
        (--temperature-<type>), else the global one."""
        if adapter is not None:
            if qtype == "choice" and options is not None:
                for lo, hi, t in self._adapter_bands.get(adapter, ()):
                    if lo <= options <= hi:
                        return t
            own = self.adapter_temperatures.get(adapter, {}).get(qtype)
            if own is not None:
                return own
        if qtype == "choice" and options is not None:
            for lo, hi, t in self._base_bands:
                if lo <= options <= hi:
                    return t
        return self.temperatures.get(qtype, self.temperature)

    def think_temperature_of(self, qtype: str, adapter: str | None = None, options: int | None = None) -> float:
        """The temperature an answer that thought is served at: the named adapter's think temperature for the type, else
        (an adapter that has its own single-pass temperature for the type keeps it) the server's think temperature for
        the type (--think-temperature-<type>, a tuned repository's `think_temperatures`), else the single-pass
        temperature of `temperature_of`. Only the answer after thinking uses it; whether a question thinks is decided at
        the single-pass temperature."""
        if adapter is not None:
            own = self.adapter_temperatures.get(adapter, {})
            if qtype in own.get("think", {}):
                return own["think"][qtype]
            single = self.temperature_of(qtype, adapter, options)
            if single != self.temperature_of(qtype, None, options):
                return (
                    single  # the adapter's own single-pass temperature, which the base's think temperature does not fit
                )
        if qtype in self.think_temperatures:
            return self.think_temperatures[qtype]
        return self.temperature_of(qtype, adapter, options)

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

    def stale_certified_task(self, q):
        """A stored task of this question that has a certificate but is not applied because the store's fingerprint
        (the model and rendering it was fitted under) is not this server's: its answers are plain, so not certified."""
        store = self.task_store
        if not self.tasks_enabled or store is None or not store.by_key:
            return None
        from decisio.serve.tasks import task_key

        t = store.by_key.get(task_key(q, render_text, self.described(q), self.noul_rendered(q)))
        return t if t is not None and t.get("certificate") and store.lookup(t["key"]) is None else None

    @_serial
    def register_readout_task(
        self, task_id: str, examples: list[tuple], certify: dict | None = None, certify_examples=None
    ):
        """Fit and store a task from labelled examples [(state, wire question, gold)] (decisio.serve.tasks).
        With `certify` (`certify_settings`) the task is also certified on examples held out of the fit:
        `certify_examples` when given, else a deterministic share of `examples` (`certify["holdout"]`); the fit then
        uses the rest only.
        Nothing is stored when the request is refused or the certification fails."""
        if self.task_store is None:
            raise ValueError("this server has no task store (--no-tasks)")
        self.task_precheck("readout", task_id)
        from decisio.serve.tasks import task_key

        full = [(q, s, g) for s, q, g in examples]
        cert_ex, split = [], None
        if certify is not None:
            fit_ex, cert_ex, split = self.certification_split(task_id, full, certify, certify_examples)
        else:
            fit_ex = full

        def score(pairs):
            out = []
            for state, q in pairs:
                eq, _, swapped = self.engine_question(q)
                probs, _ = self.engine.answer(state, [eq])
                require_finite(probs[0], "a registration example's probability")
                out.append(probs[0][::-1] if swapped else probs[0])
            return out

        def hidden(pairs):
            out = []
            for state, q in pairs:
                eq, _ = to_engine_question(q, self.hide_index_keys, self.desnake_labels, self.describe_options)
                lp, h = self.hidden_engine.readout(state, [eq])[0]
                require_finite(lp, "a registration example's label log-probability")
                require_finite(h, "a registration example's hidden state")
                out.append((lp, h))
            return out

        result = self.task_store.register(
            task_id,
            fit_ex,
            score,
            hidden if self.hidden_engine is not None else None,
            key_fn=lambda q: task_key(q, render_text, self.described(q), self.noul_rendered(q)),
            **({"commit": False} if certify is not None else {}),
        )
        if certify is None:
            return result
        task, fitted = result
        registered = self.attach_certificate(task, cert_ex, certify, split)  # stored only once it is made
        previous = self.task_store.snapshot(task["id"])
        self.task_store.put(task, persist=False)  # served from memory for the check below, written once it has run
        self.check_served(task, cert_ex, *registered)
        try:
            self.task_store.persist_task(task)
        except Exception:
            self.task_store.rollback(task["id"], previous)  # not saved: not registered
            raise
        return task, fitted

    def with_readout_task(self, state, q, eq, p):
        """A text-route question's order-1 distribution as served: its registered task's head or calibration applied to
        the plain readout p (for abstention examples, so a threshold is fitted on what it will decide)."""
        t = self.readout_task(q)
        if t is None:
            return apply_temperature(p, self.temperature_of(q.type, None, len(p)))
        if t["head"].get("applied") and self.hidden_engine is not None:
            from decisio.readout.intent_head import apply_intent_head

            lp, h = self.hidden_engine.readout(state, [eq])[0]
            return apply_intent_head(lp, h, t["head"], t["options"])
        if t["calibration"].get("applied"):
            from decisio.readout.calibration import apply_task_prior
            from decisio.readout.debias import log_probs

            return apply_task_prior(log_probs(p), t["calibration"])
        return apply_temperature(p, self.temperature_of(q.type, None, len(p)))

    # ---- certified answers and their monitor (decisio.readout.certify, decisio.serve.monitor, decisio.serve.stack) --

    @staticmethod
    def certify_settings(spec: dict | None) -> dict:
        """The settings of a certification, from the request's `certify` block: alpha (the most wrong answers
        tolerated among the certified ones, default 0.05), delta (the chance allowed that the promise fails, default
        0.1), min_examples (at least this many held-out examples, never fewer than the procedure needs at all) and
        holdout (the share of `examples` held out when no separate list is given, default 0.5)."""
        from decisio.readout import certify

        spec = dict(spec or {})
        unknown = sorted(set(spec) - {"alpha", "delta", "min_examples", "holdout", "examples", "suspend_on_alert"})
        if unknown:
            raise ValueError(
                f"certify: unknown settings {unknown}; the settings are alpha, delta, min_examples, holdout, examples, "
                "suspend_on_alert"
            )
        suspend = spec.get("suspend_on_alert", SUSPEND_ON_ALERT)
        if not isinstance(suspend, bool):
            raise ValueError("certify: suspend_on_alert is true or false")
        try:
            alpha, delta = float(spec.get("alpha", 0.05)), float(spec.get("delta", 0.1))
            holdout = float(spec.get("holdout", 0.5))
            asked = int(spec.get("min_examples") or 0)
        except (TypeError, ValueError) as e:
            raise ValueError(f"certify: {e}")
        if not 0 < alpha < 1 or not 0 < delta < 1:
            raise ValueError("certify: alpha and delta are fractions between 0 and 1 (defaults 0.05 and 0.1)")
        if not 0 < holdout < 1:
            raise ValueError("certify: holdout is a fraction between 0 and 1 (default 0.5)")
        floor = certify.minimum_examples(alpha, delta)
        return {
            "alpha": alpha,
            "delta": delta,
            "holdout": holdout,
            "suspend_on_alert": suspend,
            "min_examples": max(asked, floor),
            "floor": floor,
            "asked": asked,
        }

    def certification_split(self, task_id, examples, settings, certify_examples):
        """(fit examples, certification examples, the split's evidence); refuses a certification that cannot be made."""
        from decisio.serve.tasks import example_digest, split_holdout

        q0 = examples[0][0] if examples else None
        if q0 is not None and q0.type == "score":
            raise ValueError(
                "certify: score questions are not certified (an ordinal answer has no single wrong answer)"
            )
        if certify_examples is not None:
            cert_ex = [(q, s, g) for s, q, g in certify_examples]
            fit_ex = examples
            fit_keys = {example_digest(s, q) for q, s, _ in fit_ex}
            overlap = sum(example_digest(s, q) in fit_keys for q, s, _ in cert_ex)
            if overlap:
                raise ValueError(
                    f"certify: {overlap} of the {len(cert_ex)} certification examples are also fit examples (same "
                    "input); certification runs on examples the fit has not seen"
                )
            method = "supplied"
        else:
            fit_i, cert_i = split_holdout(task_id, examples, settings["holdout"])
            fit_ex, cert_ex = [examples[i] for i in fit_i], [examples[i] for i in cert_i]
            method = "holdout"
        if len(cert_ex) < settings["min_examples"]:
            if settings["asked"] > settings["floor"]:
                why = f"you asked for at least {settings['asked']}"
            else:
                why = (
                    f"with alpha {settings['alpha']} and delta {settings['delta']} even a task answered right every "
                    f"time needs {settings['floor']}"
                )
            share = f", {settings['holdout']:g} of {len(examples)}" if method == "holdout" else ""
            raise ValueError(
                f"certify: {len(cert_ex)} held-out labelled examples ({method}{share}), and at least "
                f"{settings['min_examples']} are needed: {why}. Give more examples, or a separate list in "
                "certify.examples (docs/tasks.md says how many a task of a given accuracy needs)"
            )
        split = {
            "method": method,
            "holdout": settings["holdout"] if method == "holdout" else None,
            "fit_examples": len(fit_ex),
            "certify_examples": len(cert_ex),
            "disjoint": True,
            "rule": HOLDOUT_RULE if method == "holdout" else SUPPLIED_RULE,
            "fit_digests": sorted({example_digest(s, q)[:16] for q, s, _ in fit_ex}),
        }
        return fit_ex, cert_ex, split

    @_serial
    def served_rows(self, t, pairs) -> list[np.ndarray]:
        """The distribution this server serves for each (state, question), through the task `t`: what a
        certification is made on and an answer is checked against."""
        from decisio.readout.calibration import apply_task_prior
        from decisio.readout.debias import log_probs
        from decisio.readout.intent_head import apply_intent_head

        out = []
        with_head = t["head"].get("applied") and self.hidden_engine is not None
        for state, q in pairs:
            if with_head:
                # the head's answer comes from the hidden-state read alone, as serving always read it. Registration
                # used to make a plain read first and discard it, but that read ran before the hidden-state one, and on
                # vLLM the hidden-state read then differs slightly from the same read made without it (the card session
                # of 2026-10-16: 154 served above the threshold against 153 certified, whether read one at a time or
                # 16 in flight).
                eqh, _ = to_engine_question(q, self.hide_index_keys, self.desnake_labels, self.describe_options)
                lp, h = self.hidden_engine.readout(state, [eqh])[0]
                p = apply_intent_head(lp, h, t["head"], t["options"])
                if self.noul_commit and q.type == "noul":
                    p = commit_noul(p)
                out.append(np.asarray(p, dtype=np.float64))
                continue
            eq, _, swapped = self.engine_question(q)
            probs, _ = self.engine.answer(state, [eq])
            p = np.asarray(probs[0], dtype=np.float64)
            if swapped:
                p = p[::-1].copy()
            if t["calibration"].get("applied"):
                p = apply_task_prior(log_probs(p), t["calibration"])
            else:
                p = apply_temperature(p, self.temperature_of(q.type, None, len(p)))
            if self.noul_commit and q.type == "noul":
                p = commit_noul(p)
            out.append(np.asarray(p, dtype=np.float64))
        return out

    def stack_facts(self) -> dict:
        """The live serving stack's fingerprint (decisio.serve.stack), taken once."""
        if getattr(self, "_stack", None) is None:
            from importlib.metadata import PackageNotFoundError, version

            from decisio.serve import stack

            try:
                ver = version("decisio")
            except PackageNotFoundError:
                ver = "unknown"
            serving = {
                "orders": self.orders,
                "multi_question": getattr(self.engine, "multi_question", None),
                "pad_policy": getattr(self.engine, "pad_policy", None),
                "hide_index_keys": self.hide_index_keys,
                "desnake_labels": self.desnake_labels,
                "describe_options": self.describe_options,
                "noul_rendering": self.noul_rendering,
                "noul_commit": self.noul_commit,
                "abstention": self.abstention,
                "abstain_option": self.abstain_option,
                "head_engine": self.hidden_engine is not None,
                # the server's think default and procedure, only when on, so that every other server's fingerprint is
                # what it was; a certificate made with thinking off is suspended by a server that thinks, and the
                # other way round (`think_mode_changed`)
                **({"think": self.think_facts()} if self.think else {}),
                # the flavor, only when one is set: a certificate made under another flavor, other overrides or
                # placeholder values is suspended
                **({"flavor": flavors.fingerprint_facts(self.flavor)} if self.flavor else {}),
                # the default adapter, only when one is set: a certificate made when requests were answered by the base
                # (or by another adapter) is suspended
                **({"default_adapter": self.default_adapter} if self.default_adapter else {}),
                # batch-invariant kernels, only when on: other kernels, so other numerics; a certificate made without
                # them is suspended
                **({"batch_invariant": True} if self.batch_invariant else {}),
                # emulated precision casts, only when on: other arithmetic in the compiled kernels, so other numerics;
                # a certificate made without them is suspended
                **({"emulate_precision_casts": True} if self.precision_casts else {}),
                # --concurrency, only when batched, so that a serial server's fingerprint is what it was; a certificate
                # made one request at a time is suspended by a batching server, and the other way round
                # (`concurrency_changed`): a batched answer depends on the traffic it was scored with
                **({"concurrency": self.concurrency_facts()} if self.batched else {}),
            }
            self._stack = stack.stack_facts(
                self.engine,
                decisio_version=ver,
                serving=serving,
                temperatures={
                    "global": self.temperature,
                    **self.temperatures,
                    **({"choice_bands": self.choice_bands} if self.choice_bands else {}),
                    **({"adapters": self.adapter_temperatures} if self.adapter_temperatures else {}),
                },
                task_fingerprint=self.task_store.fingerprint if self.task_store else None,
            )
        return self._stack

    @property
    def batched(self) -> bool:
        return getattr(self.engine, "concurrency", "serial") == "batched"

    def concurrency_facts(self) -> dict:
        """What identifies how this server batches requests: the mode and the most requests in one engine call."""
        return {"mode": "batched", "max_requests": getattr(self.engine, "batch_max_requests", None)}

    def think_facts(self) -> dict:
        """What identifies how this server thinks: the factor, the budget and the procedure (prefix, suffix, thresholds,
        option limit), the last as a digest."""
        proc = json.dumps(
            [thinking.PREFIX, thinking.SUFFIX, thinking.TAU, thinking.MAX_OPTIONS], sort_keys=True, ensure_ascii=False
        )
        adapters = {n: t["think"] for n, t in self.adapter_temperatures.items() if t.get("think", None)}
        return {
            "factor": self.think_factor,
            "budget": self.think_budget,
            "procedure_sha256": hashlib.sha256(proc.encode()).hexdigest()[:16],
            # the temperatures of the answers that thought, only when there are any, so that a server without them keeps
            # its fingerprint; they change what a thinking answer is, so a certificate made under others is suspended
            **(
                {"temperatures": {**self.think_temperatures, **({"adapters": adapters} if adapters else {})}}
                if self.think_temperatures or adapters
                else {}
            ),
        }

    def stack_with_fit(self, t) -> dict:
        s = self.stack_facts()
        return {**s, "calibration": {**s["calibration"], "task_fit": t.get("fit_sha256")}}

    def stack_sha(self, t) -> str:
        """The live stack with the task's fit, hashed; cached per fit."""
        from decisio.serve import stack

        cache = self.__dict__.setdefault("_stack_sha", {})
        fit = t.get("fit_sha256")
        if fit not in cache:
            cache[fit] = stack.sha256(self.stack_with_fit(t))
        return cache[fit]

    def attach_certificate(self, t, cert_ex, settings, split):
        """Certify task `t` (fitted, not yet stored) on the held-out examples [(question, state, gold)] and attach the
        certificate and the monitor's baseline to it. Returns the held-out examples' registered confidences and chosen
        options, for `check_served`."""
        from decisio.readout import certify
        from decisio.serve import monitor
        from decisio.serve.stack import sha256
        from decisio.serve.tasks import CERTIFICATE_FORMAT, answer_keys, fit_digest, gold_index

        t["fit_sha256"] = fit_digest(t)
        ps = self.served_rows(t, [(s, q) for q, s, _ in cert_ex])
        keys = answer_keys(cert_ex[0][0])
        chosen = [top_index(keys, p) for p in ps]
        correct = np.array([c == gold_index(q, g) for c, (q, _, g) in zip(chosen, cert_ex)])
        conf = np.array([float(max(p)) for p in ps])
        cert = certify.certificate(conf, correct, settings["alpha"], settings["delta"])
        stack = self.stack_with_fit(t)
        stack_sha = sha256(stack)
        ident = sha256([t["id"], cert["threshold"], cert["n"], split["fit_digests"], stack_sha, split["method"]])[:12]
        cert.update(
            {
                "format": CERTIFICATE_FORMAT,
                "id": f"{t['id']}#{ident}",
                "min_examples": settings["min_examples"],
                "split": split,
                "fit_sha256": t["fit_sha256"],
                "stack": stack,
                "stack_sha256": stack_sha,
                "summary": self.certificate_summary(cert),
            }
        )
        cert["suspend_on_alert"] = settings["suspend_on_alert"]
        if t.get("type", "choice") == "choice" and not has_catch_all(keys):
            cert["warnings"] = [
                {
                    "code": "no_catch_all_option",
                    "text": "This task has no option for a message that fits none of its labels (such as 'other' or "
                    "'out of scope'), so such a message gets a wrong answer that can be certified. Add one, or rely on "
                    "the traffic monitor, which watches the answer mix and the confidences.",
                }
            ]
            cert["summary"] += " Warning: " + cert["warnings"][0]["text"]
        t["certificate"] = cert
        t["monitor"] = monitor.baseline(chosen, len(keys), cert["id"], conf)
        return conf, chosen

    SERVED_CHECK_SHARE, SERVED_CHECK_MIN, SERVED_CHECK_MAX = 0.1, 20, 200

    def check_served(self, t, cert_ex, conf, chosen) -> None:
        """Re-read a sample of the held-out examples through the serving path (`answer`, one question per request) and
        record how far the confidences it serves are from the ones the certificate was made on: the largest and the
        99th-percentile difference, how many items changed sides of the threshold and how many changed their top
        answer. A certified question is scored by the function that registered it, so on one stack this reads zero;
        a nonzero value is the certificate's exposure, stated in the certificate and its summary. Not made when
        nothing is certified (no answer would carry the promise); the answer-mix monitor does not see these reads."""
        cert = t["certificate"]
        if cert["threshold"] is None or not cert_ex:
            return
        import math

        from decisio.serve.tasks import answer_keys

        n = len(cert_ex)
        k = min(n, max(self.SERVED_CHECK_MIN, math.ceil(self.SERVED_CHECK_SHARE * n)), self.SERVED_CHECK_MAX)
        idx = sorted({int(i * n / k) for i in range(k)})
        thr = cert["threshold"]
        keys = answer_keys(cert_ex[0][0])
        shifts, crossed, top_changed, applied = [], 0, 0, 0
        self._probe.on = True
        try:
            for i in idx:
                q, state, _ = cert_ex[i]
                out = self.answer(SystemOneRequest(state=state, questions={"q": q}))
                a = out["answers"]["q"]
                applied += t["id"] in out["_timing"]["tasks"]
                if a["type"] == "noul":
                    p = np.array([a["noul"], 1.0 - a["noul"]])
                else:
                    p = np.array(list(a["probabilities"].values()), dtype=np.float64)
                shifts.append(abs(float(max(p)) - float(conf[i])))
                crossed += (float(max(p)) >= thr) != (float(conf[i]) >= thr)
                top_changed += top_index(keys, p) != chosen[i]
        except Exception as e:  # the check never blocks a registration; its failure is recorded
            cert["served_check"] = {"error": f"{type(e).__name__}: {e}"}
            return
        finally:
            self._probe.on = False
        sh = np.asarray(shifts)
        cert["served_check"] = {
            "n": len(idx),
            "task_applied": applied,
            "max_shift": float(sh.max()),
            "p99_shift": float(np.percentile(sh, 99)),
            "crossed_threshold": int(crossed),
            "top_answer_changed": int(top_changed),
            "rule": "every 1/10th held-out example (20 to 200), each as a request of its own, through the serving path",
        }
        if crossed or sh.max() > 0:
            cert["summary"] += (
                f" Re-read through the serving path, {len(idx)} of the held-out examples moved by at most "
                f"{sh.max():.2g} in confidence and {int(crossed)} changed sides of the threshold: served confidences "
                "can differ from the ones this certificate was made on."
            )

    @staticmethod
    def certificate_summary(c: dict) -> str:
        """The certificate in words: what is certified and the promise, or that nothing is."""
        if c["threshold"] is None:
            return (
                f"Nothing is certified: on the {c['n']} held-out examples no confidence threshold keeps the wrong "
                f"answers above it at {c['alpha']:.0%} or fewer with probability {1 - c['delta']:.0%}. "
                "The task works as before, and every answer carries certified: false. "
                f"(Without the bound, {c['empirical_share_at_alpha']:.0%} of the examples would have met "
                f"{c['alpha']:.0%} on these examples themselves; that is not a promise.)"
            )
        return (
            f"Answers whose top probability is at least {c['threshold']:.4f} are certified: on the {c['n']} held-out "
            f"examples that is {c['certified_share']:.0%} of the answers, {c['errors_at_threshold']} of "
            f"{c['certified_n']} wrong. With probability at least {1 - c['delta']:.0%} over the draw of such examples, "
            f"at most {c['alpha']:.0%} of the certified answers are wrong, for traffic like them and on this "
            "serving stack."
        )

    @_serial
    def recertify_task(self, task_id: str, examples: list[tuple], spec: dict | None):
        """Make the certificate of a stored task again, under this server's stack, on new labelled examples (the fit is
        kept; the examples must not share an input with the fit's)."""
        store = self.task_store
        t = next((x for x in (store.by_key.values() if store else []) if x["id"] == task_id), None)
        if t is None:
            raise KeyError(task_id)
        self.task_precheck("readout", task_id)
        if not t.get("certificate"):
            raise ValueError(
                f"task {task_id!r} was registered without certification: register it again with a certify block"
            )
        from decisio.readout import certify
        from decisio.serve.tasks import example_digest

        old = t["certificate"]
        # min_examples is carried over only if the original call set it: a stored value above what the old alpha and
        # delta need can only have been asked for; a value at that floor was derived, and the new alpha and delta
        # derive their own (a looser target needs fewer examples than the 59 of the defaults)
        asked_before = old.get("min_examples") or 0
        if asked_before <= certify.minimum_examples(old["alpha"], old["delta"]):
            asked_before = None
        spec = {
            **{k: old[k] for k in ("alpha", "delta")},
            "min_examples": asked_before,
            "suspend_on_alert": old.get("suspend_on_alert", SUSPEND_ON_ALERT),
            **(spec or {}),
        }
        settings = self.certify_settings(spec)
        cert_ex = [(q, s, g) for s, q, g in examples]
        fit_digests = set(old["split"]["fit_digests"])
        overlap = sum(example_digest(s, q)[:16] in fit_digests for q, s, _ in cert_ex)
        if overlap:
            raise ValueError(
                f"certify: {overlap} of the {len(cert_ex)} examples share an input with the fit; use new examples"
            )
        if len(cert_ex) < settings["min_examples"]:
            raise ValueError(
                f"certify: {len(cert_ex)} labelled examples, and at least {settings['min_examples']} are needed "
                f"for alpha {settings['alpha']} and delta {settings['delta']}"
            )
        split = {
            **old["split"],
            "method": "supplied",
            "holdout": None,
            "certify_examples": len(cert_ex),
            "disjoint": True,
            "rule": "re-certified on new examples that share no input with the fit",
        }
        new = {k: v for k, v in t.items() if k not in ("certificate", "monitor")}
        registered = self.attach_certificate(new, cert_ex, settings, split)
        previous = store.snapshot(new["id"])
        store.put(new, persist=False)  # served from memory for the check below, written once it has run
        self.check_served(new, cert_ex, *registered)
        try:
            store.persist_task(new)
        except Exception:
            store.rollback(new["id"], previous)  # not saved: the previous certificate stays
            raise
        return new

    def certified_fields(self, t, q, keys, p, orders: int, chosen: int, shared: bool = False) -> dict:
        """The fields an answer of a certified task carries: whether it is certified, the certificate, and why not."""
        cert = t["certificate"]
        if self.stack_sha(t) != cert["stack_sha256"]:
            from decisio.serve import stack

            # a stack that differs only in how the server thinks is its own reason: a task certified one-pass is not
            # served as certified by a server that thinks, and the other way round; the same for how it batches
            diff = set(stack.differences(cert["stack"], self.stack_with_fit(t)))
            if all(d == "serving.think" or d.startswith("serving.think.") for d in diff):
                reason = "think_mode_changed"
            elif diff <= {"serving.concurrency"} | {f"serving.concurrency.{k}" for k in ("mode", "max_requests")}:
                reason = "concurrency_changed"
            else:
                reason = "serving_stack_changed"
        elif orders != cert["stack"]["serving"]["orders"]:
            reason = "serving_stack_changed"  # this request asked for another number of orders than the certificate's
        elif cert["threshold"] is None:
            reason = "nothing_certified"
        elif float(max(p)) >= cert["threshold"]:
            reason = None
        else:
            reason = "below_threshold"
        m = self.task_store.monitors.get(t["id"])
        if reason is None and m is not None and m.conf_alerting and cert.get("suspend_on_alert", SUSPEND_ON_ALERT):
            reason = "traffic_changed"  # the promise is for traffic like the examples, and the monitor says it is not
        if m is not None and not getattr(self._probe, "on", False):
            m.observe(chosen, reason is None, float(max(p)), **({"shared": True} if shared else {}))
        out = {"certified": reason is None, "certificate": cert["id"]}
        if reason is not None:
            out["certified_reason"] = reason
        return out

    def task_status(self, t) -> dict | None:
        """The state of a certified task: ok, traffic_changed (the answer-mix alert) or suspended (the stack
        changed)."""
        cert = t.get("certificate")
        if not cert:
            return None
        from decisio.serve import stack

        if self.stack_sha(t) != cert["stack_sha256"]:
            return {
                "state": "suspended",
                "text": "suspended: serving stack changed",
                "changed": stack.differences(cert["stack"], self.stack_with_fit(t)),
                "recertify": f"POST /v1/tasks/{t['id']}/certify with new labelled examples",
            }
        m = self.task_store.monitors.get(t["id"])
        f = m.facts() if m is not None else None
        state = f["state"] if f else "ok"
        if state == "traffic_changed":
            what = " and ".join(
                {"answer_mix": "the mix of chosen options", "confidence": "the confidences"}[x] for x in f["signals"]
            )
            if cert.get("suspend_on_alert", SUSPEND_ON_ALERT) and "confidence" in f["signals"]:
                text = (
                    f"traffic changed: {what} moved away from the certification examples'; certification is "
                    "suspended (every answer is certified: false) until the alert clears"
                )
            else:
                text = (
                    f"traffic changed: {what} moved away from the certification examples' (the certificate is not "
                    "suspended; this alert says the traffic is not like the examples)"
                )
        else:
            text = "ok"
        if f and self.batched:  # which reads the monitor counts on a batched server
            f = {**f, "concurrency": "batched: answers scored in a shared engine call are not observed"}
        return {"state": state, "text": text, **({"monitor": f} if f else {})}

    def metrics_text(self) -> str:
        """Per certified task, in Prometheus text format: its state, answers, certified answers and the answer-mix
        distance."""
        rows = []
        for t in self.task_store.by_key.values() if self.task_store else []:
            st = self.task_status(t)
            if st is None:
                continue
            tid = str(t["id"]).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            f = st.get("monitor") or {}
            for s in ("ok", "traffic_changed", "suspended"):
                rows.append(("decisio_task_state", f'task="{tid}",state="{s}"', int(st["state"] == s)))
            rows += [
                ("decisio_task_answers_total", f'task="{tid}"', f.get("answers", 0)),
                ("decisio_task_certified_answers_total", f'task="{tid}"', f.get("certified_answers", 0)),
                ("decisio_task_mix_alerts_total", f'task="{tid}"', f.get("alerts", 0)),
                ("decisio_task_window_answers", f'task="{tid}"', f.get("window_answers", 0)),
                ("decisio_task_mix_line", f'task="{tid}"', f.get("line", 0.0)),
            ]
            if f.get("distance") is not None:
                rows.append(("decisio_task_mix_distance", f'task="{tid}"', f["distance"]))
            if f.get("confidence"):
                rows += [
                    ("decisio_task_confidence_statistic", f'task="{tid}"', f["confidence"]["statistic"]),
                    ("decisio_task_confidence_limit", f'task="{tid}"', f["confidence"]["limit"]),
                    ("decisio_task_confidence_alerts_total", f'task="{tid}"', f["confidence"]["alerts"]),
                ]
        helps = {
            "decisio_task_state": "gauge",
            "decisio_task_answers_total": "counter",
            "decisio_task_certified_answers_total": "counter",
            "decisio_task_mix_alerts_total": "counter",
            "decisio_task_window_answers": "gauge",
            "decisio_task_mix_line": "gauge",
            "decisio_task_mix_distance": "gauge",
            "decisio_task_confidence_statistic": "gauge",
            "decisio_task_confidence_limit": "gauge",
            "decisio_task_confidence_alerts_total": "counter",
        }
        out, seen = [], set()
        for name, labels, value in rows:
            if name not in seen:
                seen.add(name)
                out.append(f"# TYPE {name} {helps[name]}")
            out.append(f"{name}{{{labels}}} {value}")
        metrics = getattr(self.task_backend, "metrics", None)
        if metrics is not None:  # a shared store: is it reachable, how stale is this replica (per replica, as the rest)
            out += metrics()
        return "\n".join(out) + ("\n" if out else "")

    # ---- per-task abstention -------------------------------------------------------------------------------------

    def task_for(self, q, imajev_ext: bool) -> dict | None:
        if not self.abstention or not self.tasks:
            return None
        fp = option_set(q, self.described(q)) if q.type == "choice" else None
        kp = option_keys(q) if q.type == "choice" else None
        for t in self.tasks.values():  # the exact option set first, then the imajev extension
            if not self.abstention_applies(t):
                continue
            m = t["match"]
            if (m.get("option_set") and m["option_set"] == fp) or (m.get("imajev_extension") and imajev_ext):
                return t
        for t in self.tasks.values():  # then the key set alone: the most general match, so the last
            if not self.abstention_applies(t):
                continue
            if kp is not None and t["match"].get("option_keys") == kp:
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

    @_serial
    def register_task(self, task_id: str, option: dict, match: str, examples: list[tuple]) -> dict:
        """Fit and store a task from labelled examples [(request, images, imajev_ext, gold)], gold None for
        unanswerable (decisio.serve.abstention.fit, its acceptance rule)."""
        from decisio.serve.abstention import decide, fit

        self.task_precheck("abstention", task_id)
        if bool(option.get("key") is not None) == bool(option.get("append")):
            raise ValueError("option: exactly one of {'key': <declared key>} or {'append': <text>}")
        if match not in ("option_set", "option_keys", "imajev_extension"):
            raise ValueError("match: 'option_set', 'option_keys' or 'imajev_extension'")
        if option.get("append") and match != "imajev_extension" and not all(e[2] for e in examples):
            raise ValueError("an appended abstain option is answered only through imajev's extension")
        p_abs, unans, ok, plain_right, plain_false, fps, kps = [], [], [], [], [], set(), set()
        for req, images, ext, gold in examples:
            q, p, idx, keys, p0, keys0 = self.score_example(req, images, option, ext)
            fps.add(option_set(q, self.described(q)))
            kps.add(option_keys(q))
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
            raise ValueError(
                "the examples of an option-set task must share one option set (the same keys and the same "
                "descriptions); examples that share their keys but not their texts take match 'option_keys'"
            )
        if match == "option_keys" and len(kps) != 1:
            raise ValueError("the examples of an option-keys task must share one set of option keys")
        cfg = fit(p_abs, unans, ok, plain_right, plain_false)
        task = {
            "id": task_id,
            "option": option,
            "match": (
                {"option_set": fps.pop()}
                if match == "option_set"
                else {"option_keys": kps.pop()}
                if match == "option_keys"
                else {"imajev_extension": True}
            ),
            "config": cfg,
            "model": self.name,
            # the layout is written only when it is not plain, so a plain server's records are what they were
            **({"layout": self.layout} if self.layout != "plain" else {}),
        }
        self.save_abstention(task)
        self.tasks[task_id] = task
        return task

    def task_precheck(self, kind: str, task_id: str) -> None:
        """Before an expensive registration: a shared store refuses now what it would refuse at the end (a replica that
        does not teach, a task someone else changed)."""
        if self.task_backend is not None:
            self.task_backend.precheck(kind, task_id)

    def task_version(self, kind: str, task_id: str):
        """The shared store's version of this task as this replica holds it, or None (no shared store, or unknown)."""
        return self.task_backend.version_of(kind, task_id) if self.task_backend is not None else None

    def save_abstention(self, task: dict) -> None:
        """Write an abstention task through to the backend, if there is one (TaskPersistError when it fails)."""
        if self.task_backend is not None:
            self.task_backend.save("abstention", task)

    def remove_abstention(self, task_id: str) -> bool:
        if task_id not in self.tasks:
            return False
        if self.task_backend is not None:
            self.task_backend.delete("abstention", task_id)
        del self.tasks[task_id]
        return True

    def load_abstention(self, records: list, persist: bool = True) -> list[str]:
        """Load abstention task records as `GET /v1/abstention/tasks` returns them. Validated first, so a bad record
        stores nothing; saved as they load (`persist=False`: they came from the backend)."""
        for t in records:
            if not isinstance(t, dict) or not t.get("id") or not isinstance(t.get("option"), dict):
                raise ValueError("an abstention task record has an id and an option")
            if not isinstance(t.get("match"), dict):
                raise ValueError(f"abstention task {t['id']!r}: no match")
            check_format(t.get("config"), "abstention", f"abstention task {t.get('id')!r}")
        for t in records:
            if persist:
                self.save_abstention(t)
            self.tasks[t["id"]] = t
        return [t["id"] for t in records]

    @property
    def layout(self) -> str:
        """The layout of the user turn this server serves (decisio.serve.layout)."""
        return getattr(getattr(self.engine, "fmt", None), "layout", "plain")

    def abstention_applies(self, t: dict) -> bool:
        """A task fitted under another served model, or under another layout of the user turn, is kept but not applied
        (its threshold is for that model's probabilities under that prompt); records written before the fields carry no
        model and apply, and no layout means plain."""
        return t.get("model") in (None, self.name) and (t.get("layout") or "plain") == self.layout

    def task_store_report(self) -> dict:
        """What /health and GET /v1/tasks say about the saved tasks: where they are kept, how many are loaded, how
        many of those apply to this server's stack, and what the start loaded from where."""
        store = self.task_store
        readout = list(store.by_key.values()) if store else []
        applied = sum(1 for t in readout if same_fingerprint(t["fingerprint"], store.fingerprint))
        a_applied = sum(1 for t in self.tasks.values() if self.abstention_applies(t))
        a_model = sum(1 for t in self.tasks.values() if t.get("model") not in (None, self.name))
        return {
            "saved_to": self.task_backend.describe() if self.task_backend is not None else None,
            "readout": {"loaded": len(readout), "applied": applied, "other_stack": len(readout) - applied},
            "abstention": {
                "loaded": len(self.tasks),
                "applied": a_applied,
                "other_model": a_model,
                "other_layout": len(self.tasks) - a_applied - a_model,
            },
            "at_start": self.task_report,
        }


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

    def raise_client_error(e):
        """An error vLLM says the request caused (a prompt over the context) is a 4xx, not a 500 (engine_health)."""
        hit = client_error(e)
        if hit is not None:
            raise HTTPException(*hit) from e

    @app.exception_handler(ImageRequestError)
    def image_request_error(request, exc):
        return JSONResponse({"detail": {"error_type": "api_usage_error", "message": str(exc)}}, status_code=400)

    @app.exception_handler(NonFiniteReadout)
    async def non_finite_readout(request, exc):
        facts = systemone.engine.facts()
        said = {k: facts[k] for k in ("base", "engine", "model", "quantization", "vllm", "mode") if k in facts}
        return JSONResponse(
            {
                "detail": f"{exc}. The request is not at fault: the model's arithmetic on this server produced it. "
                f"Served model {systemone.name!r}, engine {json.dumps(said, default=str)}. Check the device's dtype "
                "and matmul precision and that the checkpoint belongs to the base given with --base."
            },
            status_code=500,
        )

    @app.get("/v1/models")
    def models():
        return systemone.models()

    async def read(request: Request):
        """The body as (payload, image bytes, imajev's extension used, the score levels as written, whether `assets`
        bound the images). `assets` (System One's images) replaces its placeholders by "[image N]" markers and takes the
        images from the asset's `data_base64`, or from the multipart part its `part` names."""
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        blobs = []
        parts = None
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
            if "assets" in payload:  # System One's multipart: the request first, then every image as a part of its own
                items = form.multi_items()
                if not items or items[0][0] != "request":
                    raise ImageRequestError(
                        'A multipart request must start with a "request" part holding the JSON request.'
                    )
                parts = {}
                for key, f in items[1:]:
                    if not key or key in parts:
                        raise ImageRequestError("Each part needs a unique, nonempty name.")
                    if isinstance(f, str):
                        raise ImageRequestError(f'Part "{key}" must be an uploaded image.')
                    parts[key] = await f.read()
            else:
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
            if "assets" in payload and "images" in payload:
                raise ImageRequestError('A request carries its images in "assets" or in "images", not both.')
            ext = "images" in payload
            imgs = payload.pop("images", None) or []
            if not isinstance(imgs, list):
                raise HTTPException(422, "'images' must be a list of data URLs")
            try:
                blobs += [decode_data_url(u) for u in imgs]
            except (ValueError, TypeError) as e:
                raise HTTPException(422, str(e))
        if "assets" in payload:  # System One's images; the imajev encodings are not read beside them
            payload, blobs, legends = bind_assets(payload, parts)
            return payload, blobs, False, legends, True
        if payload.get("state") is not None:
            payload["state"], found = extract_state_images(payload["state"])
            if found:
                ext = True
                try:
                    blobs += [decode_data_url(u) for u in found]
                except (ValueError, TypeError) as e:
                    raise HTTPException(422, f"image in the state: {e}")
        return payload, blobs, ext, {}, False

    async def system_one(request: Request):
        from decisio.serve.image_engine import DEFAULT_MAX_IMAGES, load_image

        payload, blobs, ext, legends, from_assets = await read(request)
        try:
            req = SystemOneRequest.model_validate(payload)
        except ValidationError as e:
            detail = [
                {**{k: v for k, v in err.items() if k in ("type", "msg", "input", "ctx")}, "loc": ["body", *err["loc"]]}
                for err in e.errors(include_url=False)
            ]
            return JSONResponse({"detail": json.loads(json.dumps(detail, default=str))}, status_code=422)
        limit = getattr(systemone.image_engine, "max_images", DEFAULT_MAX_IMAGES)
        if len(blobs) > limit:
            if from_assets:
                raise ImageRequestError(f"at most {limit} images per request; the placeholders make {len(blobs)}.")
            raise HTTPException(422, f"at most {limit} images per request; got {len(blobs)}")
        try:
            images = [load_image(b) for b in blobs]
        except (ValueError, OSError) as e:
            if from_assets:
                raise ImageRequestError(f"Invalid image: {e}")
            raise HTTPException(422, f"bad image: {e}")
        route = request_header(request.headers, "route")
        if route not in (None, "text", "image"):
            raise HTTPException(422, "x-decisio-route must be 'text' or 'image'")
        if systemone.flavor and req.model:
            others = {n.lower() for n in flavors.siblings(systemone.flavor)}
            if req.model.lower() in others:
                raise HTTPException(
                    422,
                    f'"model" {req.model!r} is another flavor: this server serves {systemone.name} '
                    f"(case does not matter)",
                )
        debug = debug_of(request)
        adapter = (req.model_extra or {}).get("adapter")
        if adapter is not None and not isinstance(adapter, str):
            raise HTTPException(422, '"adapter" must be the name of a loaded adapter, "none" or null')
        if adapter == "none":
            adapter = None  # the base, even on a server with a default adapter
        elif adapter is None:
            adapter = systemone.default_adapter  # None when the server has none
        over_limit = (req.model_extra or {}).get("over_limit")
        if over_limit is not None and over_limit not in over.MODES:
            raise HTTPException(422, f'"over_limit" must be one of {list(over.MODES)}, or null')
        think = (req.model_extra or {}).get("think")
        if think is not None and not isinstance(think, bool):
            raise HTTPException(422, '"think" must be true, false or null')
        try:
            out = await run_in_threadpool(systemone.answer, req, adapter, images, ext, route, debug, over_limit, think)
        except UnknownAdapter as e:
            raise HTTPException(404, str(e))
        except (over.OverLimit, thinking.ThinkUnavailable) as e:
            raise HTTPException(422, str(e))
        except (KeyError, ValueError, AssertionError) as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise_client_error(e)
            raise
        timing = out.pop("_timing")
        for qname, levels in legends.items():  # a score's legend shows the levels as the caller wrote them
            ans = out["answers"].get(qname)
            if ans and ans.get("type") == "score":
                ans["legend"] = {str(i): c for i, c in enumerate(levels)}
        headers = {header("server-ms"): f"{timing['server_ms']:.1f}", header("route"): timing["route"]}
        if systemone.flavor:
            headers[header("flavor")] = flavors.header_value(systemone.flavor)
        if timing.get("tasks"):
            headers[header("tasks")] = ",".join(timing["tasks"])
        if out.get("adapter"):
            headers[header("adapter")] = out["adapter"]
        # what the prompt cost: the layout, the mean prompt tokens per question and how many of them the cache served
        # (under the header layout a state cached for another set of questions is read again)
        if timing.get("prompt_tokens_mean") is not None:
            headers[header("prompt")] = (
                f"layout={timing['layout']}; tokens={timing['prompt_tokens_mean']:.0f}; "
                f"cached={timing.get('cached_tokens_mean', 0.0):.0f}"
            )
        if timing.get("think_tokens") is not None and "thinking" in out:
            headers[header("think-ms")] = f"{timing['think_ms']:.1f}"
            headers[header("think-tokens")] = str(timing["think_tokens"])
        if out.get("header_limited"):
            headers[header("layout")] = header_limit.header_value(out["header_limited"])
        if out.get("state_truncated"):
            cut = out["state_truncated"]
            headers[header("state-truncated")] = f"kept={cut['kept_tokens']}; of={cut['of_tokens']}; part={cut['part']}"
        # where the engine's time went, e.g. "prepare=1.2;warm=0.0;questions=52.1;readout=0.1;engine=53.6"
        if timing.get("stages"):
            headers[header("stages")] = ";".join(f"{k}={v:.1f}" for k, v in timing["stages"].items())
        if timing.get("waited_behind_registration"):  # the wait in `stages` was behind a boundary registration (#132)
            headers[header("wait-behind")] = "registration"
        if timing.get("batch"):  # --concurrency batched: the engine call this request shared with others
            b = timing["batch"]
            headers[header("batch")] = f"requests={b['requests']}; rows={b['rows']}"
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

    def not_saved(e: Exception) -> HTTPException:
        """A task that could not be written: nothing was registered (or removed), and the caller is told so. A shared
        store adds three answers: 409 (someone changed the task since, or this replica does not teach), 503 (the store
        cannot be reached)."""
        if isinstance(e, TaskConflict):
            return HTTPException(
                409,
                {
                    "message": str(e),
                    "kind": e.kind,
                    "task": e.task_id,
                    "store_version": e.store_version,
                    "your_version": e.your_version,
                },
            )
        if isinstance(e, TaskStoreReadOnly):
            return HTTPException(409, str(e))
        if isinstance(e, TaskStoreUnavailable):
            return HTTPException(503, f"{e}; the change was not made")
        return HTTPException(500, f"{e}; the change was not made")

    def set_intent(request: Request) -> None:
        """What the caller says about the shared store's versions: `?replace=1` replaces whatever the store holds,
        `x-decisio-replaces: N` names the version the caller was shown (a task's `store_version`). Set for the
        request's own context, which the worker thread that registers inherits."""
        q = request.query_params.get("replace", "").lower() in ("1", "true", "yes")
        named = request.headers.get("x-decisio-replaces")
        try:
            replaces = int(named) if named is not None else None
        except ValueError:
            raise HTTPException(422, "x-decisio-replaces must be a whole number (a task's store_version)")
        WRITE_INTENT.set(WriteIntent(replace=q, replaces=replaces))

    def with_version(out: dict, kind: str, task_id: str) -> dict:
        v = systemone.task_version(kind, task_id)
        if v is not None:
            out["store_version"] = v
        return out

    # per-task abstention (decisio.serve.abstention; docs/handoffs/tasks.md): register a task from labelled
    # examples, each a /v1/systemone request (JSON, images as data URLs) with its answer (null: unanswerable)
    async def register(request: Request):
        from decisio.serve.image_engine import DEFAULT_MAX_IMAGES, load_image

        body = await request.json()
        if not isinstance(body, dict) or not body.get("id") or not isinstance(body.get("examples"), list):
            raise HTTPException(422, "body: {'id', 'option', 'match', 'examples': [{'request', 'answer'}]}")
        set_intent(request)
        refuse_adapter(body)
        examples = []
        for n, ex in enumerate(body["examples"]):
            payload = dict(ex.get("request") or {})
            refuse_adapter(payload, f"example {n}")
            ext = "images" in payload
            imgs = payload.pop("images", None) or []
            try:
                if "assets" in payload:
                    if imgs:
                        raise ValueError('a request carries its images in "assets" or in "images", not both')
                    payload, blobs, _ = bind_assets(payload)
                    ext = False
                else:
                    blobs = [decode_data_url(u) for u in imgs]
                if "assets" not in ex.get("request", {}) and payload.get("state") is not None:
                    payload["state"], found = extract_state_images(payload["state"])
                    ext = ext or bool(found)
                    blobs += [decode_data_url(u) for u in found]
                limit = getattr(systemone.image_engine, "max_images", DEFAULT_MAX_IMAGES)
                if len(blobs) > limit:
                    raise ValueError(f"at most {limit} images per request")
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
        except TaskPersistError as e:
            raise not_saved(e)
        except Exception as e:
            raise_client_error(e)
            raise
        return with_version(dict(task), "abstention", task["id"])

    register.__annotations__["request"] = Request
    app.post("/v1/abstention/tasks")(register)

    @app.get("/v1/abstention/tasks")
    def tasks():
        """Every abstention task, in the form `POST /v1/abstention/tasks/import` and `--abstention-tasks` load."""
        return {
            "enabled": systemone.abstention,
            "tasks": list(systemone.tasks.values()),
            "store": systemone.task_store_report(),
        }

    def import_abstention_tasks(body: dict, request: Request):
        """Load abstention tasks as `GET /v1/abstention/tasks` returns them (no re-scoring); refused when fitted for
        another served model."""
        set_intent(request)
        records = body.get("tasks") if isinstance(body, dict) else None
        if not isinstance(records, list):
            raise HTTPException(422, "body: {'tasks': [<task as GET /v1/abstention/tasks returns it>]}")
        other = [t.get("id") for t in records if isinstance(t, dict) and t.get("model") not in (None, systemone.name)]
        if other:
            raise HTTPException(422, f"fitted for another served model than {systemone.name!r}: {other}")
        other = [
            t.get("id") for t in records if isinstance(t, dict) and (t.get("layout") or "plain") != systemone.layout
        ]
        if other:
            raise HTTPException(422, f"fitted under another layout than {systemone.layout!r}: {other}")
        try:
            loaded = systemone.load_abstention(records)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(422, f"not an abstention task record: {e}")
        except TaskPersistError as e:
            raise not_saved(e)
        return {"loaded": loaded}

    import_abstention_tasks.__annotations__["request"] = Request
    app.post("/v1/abstention/tasks/import")(import_abstention_tasks)

    def remove_abstention_task(task_id: str, request: Request):
        set_intent(request)
        try:
            removed = systemone.remove_abstention(task_id)
        except TaskPersistError as e:
            raise not_saved(e)
        if not removed:
            raise HTTPException(404, f"no abstention task {task_id!r}")
        return {"removed": task_id}

    remove_abstention_task.__annotations__["request"] = Request
    app.delete("/v1/abstention/tasks/{task_id}")(remove_abstention_task)

    # per-task calibration and the intent head (decisio.serve.tasks; docs/handoffs/tasks.md): register a task from
    # labelled examples, each a text /v1/systemone request with one question and its answer
    NO_ADAPTER = (
        '"adapter" is not accepted when registering or certifying a task: a task is fitted on the base\'s readout, '
        "and the field would be ignored"
    )

    NO_OVER_LIMIT = (
        '"over_limit" is not accepted when registering or certifying a task: registration never truncates a state, '
        "a certificate is made on whole inputs only"
    )

    def refuse_adapter(body, where="body"):
        if isinstance(body, dict) and body.get("adapter") is not None:
            raise HTTPException(422, f"{where}: {NO_ADAPTER}")
        if isinstance(body, dict) and body.get("over_limit") is not None:
            raise HTTPException(422, f"{where}: {NO_OVER_LIMIT}")

    def parse_examples(items, where="example"):
        examples = []
        for n, ex in enumerate(items):
            if not isinstance(ex, dict):
                raise HTTPException(422, f"{where} {n}: must be an object {{'request', 'answer'}}")
            payload = dict(ex.get("request") or {})
            refuse_adapter(payload, f"{where} {n}")
            if (
                "images" in payload
                or "assets" in payload
                or (payload.get("state") is not None and extract_state_images(payload["state"])[1])
            ):
                raise HTTPException(422, f"{where} {n}: calibration and the intent head serve text requests only")
            try:
                req = SystemOneRequest.model_validate(payload)
            except ValidationError as e:
                raise HTTPException(422, f"{where} {n}: {e}")
            if len(req.questions) != 1:
                raise HTTPException(422, f"{where} {n}: an example carries exactly one question")
            examples.append((req.state, next(iter(req.questions.values())), ex.get("answer")))
        return examples

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
        set_intent(request)
        refuse_adapter(body)
        debug = debug_of(request)
        examples = parse_examples(body["examples"])
        certify, held_out = None, None
        if body.get("certify") is not None:
            if not isinstance(body["certify"], dict):
                raise HTTPException(422, "certify: an object {'alpha', 'delta', 'min_examples', 'holdout', 'examples'}")
            try:
                certify = systemone.certify_settings(body["certify"])
            except ValueError as e:
                raise HTTPException(422, str(e))
            if body["certify"].get("examples") is not None:
                if not isinstance(body["certify"]["examples"], list):
                    raise HTTPException(422, "certify.examples: a list of {'request', 'answer'} like examples")
                held_out = parse_examples(body["certify"]["examples"], "certify.examples")
        try:
            task, fitted = await run_in_threadpool(
                systemone.register_readout_task, str(body["id"]), examples, certify, held_out
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(422, str(e))
        except TaskPersistError as e:
            raise not_saved(e)
        except Exception as e:
            raise_client_error(e)
            raise
        out = with_version(TaskStore.public(task), "readout", task["id"])
        if task.get("certificate"):
            out["status"] = systemone.task_status(task)
        if debug:
            out[DEBUG_KEY] = {"lps": [np.asarray(x).tolist() for x in fitted["lps"]], "labels": fitted["labels"]}
            if debug == "hidden" and fitted["readout"] is not None:
                out[DEBUG_KEY]["readout"] = [[lp.tolist(), h.tolist()] for lp, h in fitted["readout"]]
        return JSONResponse(json.loads(json.dumps(out, default=float)))

    register_readout.__annotations__["request"] = Request
    app.post("/v1/tasks")(register_readout)

    @app.get("/v1/tasks")
    def readout_tasks(full: int = 0):
        """Every registered task; `?full=1` includes the heads' arrays, the form `--tasks-file` loads. A certified task
        also carries its `status` (ok, traffic_changed or suspended)."""
        from decisio.serve.tasks import TaskStore

        store = systemone.task_store
        listed = []
        for t in store.by_key.values() if store else []:
            item = (
                with_version(TaskStore.public(t, bool(full)), "readout", t["id"])
                if not full
                else TaskStore.public(t, True)
            )
            if t.get("certificate") and not full:
                item["status"] = systemone.task_status(t)
            listed.append(item)
        return {
            "enabled": systemone.tasks_enabled,
            "head_engine": systemone.hidden_engine is not None,
            "fingerprint": store.fingerprint if store else None,
            "store": systemone.task_store_report(),
            "tasks": listed,
        }

    @app.get("/v1/tasks/{task_id}")
    def readout_task_one(task_id: str):
        """One task, with its status and the answer-mix monitor's facts when it is certified."""
        from decisio.serve.tasks import TaskStore

        store = systemone.task_store
        t = next((x for x in (store.by_key.values() if store else []) if x["id"] == task_id), None)
        if t is None:
            raise HTTPException(404, f"no task {task_id!r}")
        out = with_version(TaskStore.public(t), "readout", t["id"])
        if t.get("certificate"):
            out["status"] = systemone.task_status(t)
        return JSONResponse(json.loads(json.dumps(out, default=float)))

    async def certify_readout(task_id: str, request: Request):
        from decisio.serve.tasks import TaskStore

        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise HTTPException(422, f"request body is not valid JSON: {e}")
        if not isinstance(body, dict) or not isinstance(body.get("examples"), list):
            raise HTTPException(
                422, "body: {'examples': [{'request', 'answer'}], optional 'alpha', 'delta', 'min_examples'}"
            )
        set_intent(request)
        refuse_adapter(body)
        examples = parse_examples(body["examples"])
        spec = {k: body[k] for k in ("alpha", "delta", "min_examples", "suspend_on_alert") if k in body}
        try:
            task = await run_in_threadpool(systemone.recertify_task, task_id, examples, spec)
        except KeyError:
            raise HTTPException(404, f"no task {task_id!r}")
        except ValueError as e:
            raise HTTPException(422, str(e))
        except TaskPersistError as e:
            raise not_saved(e)
        except Exception as e:
            raise_client_error(e)
            raise
        out = with_version(TaskStore.public(task), "readout", task["id"])
        out["status"] = systemone.task_status(task)
        return JSONResponse(json.loads(json.dumps(out, default=float)))

    certify_readout.__annotations__["request"] = Request
    app.post("/v1/tasks/{task_id}/certify")(certify_readout)

    @app.get("/metrics")
    def metrics():
        """Per certified task, in Prometheus text format: state, answers, certified answers, answer-mix distance."""
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(systemone.metrics_text(), media_type="text/plain; version=0.0.4")

    def import_readout_tasks(body: dict, request: Request):
        """Load tasks as `GET /v1/tasks?full=1` returns them (no re-scoring); refused when fitted under another model or
        rendering (re-register those from their examples)."""
        set_intent(request)
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
        except TaskPersistError as e:
            raise not_saved(e)
        return {"loaded": [t["id"] for t in tasks]}

    import_readout_tasks.__annotations__["request"] = Request
    app.post("/v1/tasks/import")(import_readout_tasks)

    def remove_readout_task(task_id: str, request: Request):
        set_intent(request)
        try:
            removed = systemone.task_store is not None and systemone.task_store.remove(task_id)
        except TaskPersistError as e:
            raise not_saved(e)
        if not removed:
            raise HTTPException(404, f"no task {task_id!r}")
        return {"removed": task_id}

    remove_readout_task.__annotations__["request"] = Request
    app.delete("/v1/tasks/{task_id}")(remove_readout_task)

    return app
