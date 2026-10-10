# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# SPDX-FileCopyrightText: Copyright (c) 2026 Nood Co (github.com/blockbrain-ai) and contributors
"""A plain-transformers reference for a decisio model repository: one decision per call.

It reads `decision_config.json` from the repository, builds the prompt the decisio server builds, runs one forward pass
of the model through Hugging Face transformers, reads the option letters at the last position, applies the repository's
temperatures and returns the System One answer. It needs torch and transformers (5.10.4 or later) and the standard
library; it imports nothing from decisio, so that it can check a repository independently of the code it checks.

    python decisio_reference.py tachara-ai/decisio-gemma-4-12b --state "The parcel is late." \\
        --question '{"type": "noul", "instructions": "Is the customer unhappy?"}'
    python decisio_reference.py ./my-repository --request request.json    # a /v1/systemone body, one call per question

    ref = Reference.load("tachara-ai/decisio-gemma-4-12b")
    ref.decide("The parcel is late.", {"type": "choice", "criteria": {"delivery": "Where is it", "billing": None}})

What it is: the server's prompt and readout without the server. One question per call, no prefix cache, no batching and
no engine; the two MIT-licensed prompt texts (SYSTEM_PROMPT, ANSWER_LINE) are the server's, verbatim (THIRD-PARTY.md).
What it is not: registered tasks (a task's calibration or intent head), abstention, two-order averaging and
`--noul-commit` are not here, so it reproduces the server's plain readout, the one a task replaces; a state that carries
an image is refused.

The padding a base's profile asks for (`serving.pad_to`) is applied as the vLLM server applies it: the Qwen base pads
the state to end on the engine's 1,056-token block (`block_size`), the Gemma bases do not pad.
Run it in the dtype the checkpoint was served in: the server's numbers come from a bfloat16 or FP8 engine, and a
float32 forward pass differs from them in the last digits. `tests/unit/test_reference.py` pins the reference to the
server's own path.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCHEMA = "decisio.decision_config/1"
FILE = "decision_config.json"

NOUL_RENDERINGS = ("words", "letters", "letters-keys")
TAILS = ("compact", "spaced")
SLOTS = ("prefill", "template")
VARIANTS = ("single", "summed")
PAD_PLACES = ("between", "front", "user")
QUESTION_TYPES = ("noul", "choice", "score")

PAD_TOKEN = 198  # "\n" in the Qwen tokenizer
USER_HEAD = "<|im_start|>user\n"
QWEN_BLOCK = 1056  # the vLLM engine's block on the Qwen base, `serving.pad_to: block` (docs/running.md)
MAX_LABELS = 255
THINK_MARKERS = (("<think>", "</think>"), ("<|channel>", "<channel|>"))
DEFAULT_INSTRUCTIONS = {"noul": "Is this true?", "choice": "Which option applies?", "score": "Which level applies?"}

# MIT, Copyright (c) 2026 Nood Co and contributors: the Cygnet recipe's texts, verbatim (THIRD-PARTY.md)
ANSWER_LINE = "Answer with the letter of exactly one option, and nothing else:"
SYSTEM_PROMPT = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else — a single character, no words, no punctuation, no explanation."
)
LETTERS_INSTRUCTION = "Answer with the letter only."  # the compact tail's

IMAGE_DATA_URL = re.compile(r"data:image/[\w.+-]+;base64,")
INDEX_KEY = re.compile(r"^([A-Za-z]+[_\- ]?)?(\d+)$")
SNAKE_LABEL = re.compile(r"^[a-z]+(?:_[a-z]+)*$")


# ---- the repository's file -------------------------------------------------------------------------------------------


def read_config(repository, revision=None) -> dict:
    """`decision_config.json` of a repository: a directory, or `owner/name[@revision]` on the Hugging Face Hub."""
    if Path(str(repository)).is_dir():
        path = Path(str(repository)) / FILE
    else:
        from huggingface_hub import hf_hub_download

        name, _, at = str(repository).partition("@")
        try:
            path = Path(hf_hub_download(name, FILE, revision=at or revision))
        except Exception as e:  # noqa: BLE001  (any failure to fetch it is "no readable file")
            raise ValueError(f"{repository} has no readable {FILE} ({e})") from e
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{repository} has no readable {FILE} ({e})") from e
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        got = config.get("schema") if isinstance(config, dict) else None
        raise ValueError(f"{repository}: {FILE} is schema {got!r}; this reference reads {SCHEMA}")
    return config


@dataclass(frozen=True)
class Format:
    """The profile's prompt format (`prompt.format`): see decisio.readout.letters.PromptFormat."""

    tail: str
    slot: str
    variants: str
    system_prompt: bool

    @classmethod
    def of(cls, d: dict) -> Format:
        for key, allowed in (("tail", TAILS), ("slot", SLOTS), ("variants", VARIANTS)):
            if d.get(key) not in allowed:
                raise ValueError(f"prompt.format.{key} is {d.get(key)!r}; this reference reads {allowed}")
        return cls(d["tail"], d["slot"], d["variants"], bool(d.get("system_prompt", False)))

    def label_form(self) -> str:
        """The letter codes the format needs: "bare" when it reads the bare form only, else "spaced"."""
        return "bare" if (self.variants, self.slot) == ("single", "template") else "spaced"


def model_softcap(model):
    """The final-logit softcap the transformers model applies itself, or None."""
    config = model.config
    for c in (config, getattr(config, "text_config", None)):
        value = getattr(c, "final_logit_softcapping", None)
        if value is not None:
            return float(value)
    return None


def pad_unit_of(pad_to, block_size):
    if pad_to == "none":
        return None
    if pad_to == "block":
        return block_size or QWEN_BLOCK
    if isinstance(pad_to, int) and not isinstance(pad_to, bool) and pad_to > 0:
        return pad_to
    if isinstance(pad_to, str) and pad_to.isdigit() and int(pad_to) > 0:
        return int(pad_to)
    raise ValueError(f"serving.pad_to is {pad_to!r}; this reference reads 'none', 'block' or a token count")


# ---- how a wire question becomes a prompt ----------------------------------------------------------------------------


def render_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def fmt_state(state) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):  # "_" keys are item metadata, never shown
        return "\n".join(f"{k}: {v}" for k, v in state.items() if not k.startswith("_"))
    return str(state)


@dataclass(frozen=True)
class Question:
    type: str
    instructions: object
    criteria: object  # noul: {"true": ..., "false": ...} or None; choice: {key: description}; score: [level, ...]

    @classmethod
    def of(cls, q: dict) -> Question:
        if not isinstance(q, dict) or q.get("type") not in QUESTION_TYPES:
            raise ValueError(f"a question is an object whose type is one of {QUESTION_TYPES}")
        kind, criteria = q["type"], q.get("criteria")
        if kind == "noul":
            if criteria is not None and not (isinstance(criteria, dict) and set(criteria) <= {"true", "false"}):
                raise ValueError("a noul question's criteria are {'true': ..., 'false': ...}")
        elif kind == "choice":
            if not (isinstance(criteria, dict) and criteria):
                raise ValueError("a choice question's criteria are a non-empty object, key to description")
        elif not (isinstance(criteria, list) and criteria):
            raise ValueError("a score question's criteria are a non-empty list of levels")
        return cls(kind, q.get("instructions"), criteria)

    def noul_side(self, which):
        return None if self.criteria is None else self.criteria.get(which)


def index_keys(criteria: dict) -> bool:
    """A choice question whose keys are a pure enumeration (`option_0` ... `option_76`, or 1..n), each described."""
    keys = list(criteria)
    matches = [INDEX_KEY.match(k) for k in keys]
    if len(keys) < 2 or not all(matches) or len({(m.group(1) or "").lower() for m in matches}) != 1:
        return False
    nums = sorted(int(m.group(2)) for m in matches)
    if nums not in (list(range(len(nums))), list(range(1, len(nums) + 1))):
        return False
    return all(d is not None and render_text(d).strip() != "" for d in criteria.values())


def snake_label(label: str) -> bool:
    return "_" in label and SNAKE_LABEL.match(label) is not None


def to_engine_question(q: Question, hide_index_keys=True, desnake_labels=True):
    """(engine question, the answer keys in engine option order), as decisio.serve.systemone.to_engine_question."""
    instr = render_text(q.instructions).strip() or DEFAULT_INSTRUCTIONS[q.type]
    if q.type == "noul":
        extra = []
        if q.noul_side("true") is not None:
            extra.append(f"Yes means: {render_text(q.noul_side('true'))}")
        if q.noul_side("false") is not None:
            extra.append(f"No means: {render_text(q.noul_side('false'))}")
        return {"kind": "noul", "instructions": "\n".join([instr, *extra]) if extra else instr}, ["yes", "no"]
    if q.type == "choice":
        keys = list(q.criteria)
        bare = None  # options shown as bare labels, if they are
        if hide_index_keys and index_keys(q.criteria):
            bare = [render_text(d) for d in q.criteria.values()]
        elif all(d is None or render_text(d) == "" for d in q.criteria.values()):
            bare = list(keys)
        if bare is None:  # an option with a description shows it alone, one without shows its key
            has = [d is not None and render_text(d) != "" for d in q.criteria.values()]
            shown = [
                render_text(d) if h else (k.replace("_", " ") if desnake_labels and snake_label(k) else k)
                for (k, d), h in zip(q.criteria.items(), has, strict=True)
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


def noul_as_letters(q: Question, keys_shown: bool):
    """A yes/no question asked as a two-option choice, the false side first; None when the two sides read the same."""
    instr = render_text(q.instructions).strip() or DEFAULT_INSTRUCTIONS["noul"]

    def side(desc, word):
        text = None if desc is None else render_text(desc).strip()
        if not text:
            return word
        return f"{word}: {text}" if keys_shown else text

    options = [side(q.noul_side("false"), "No"), side(q.noul_side("true"), "Yes")]
    if options[0] == options[1]:
        return None
    return {"kind": "choice", "instructions": instr, "options": options}, ["no", "yes"]


def letter_labels(tok, k, form):
    """The first k letter codes ("A" ... "Z", "AA", ...) whose spaced (" A") or bare ("A") form is one token."""
    if not 1 <= k <= MAX_LABELS:
        raise ValueError(f"the letter readout supports 1 to {MAX_LABELS} options, got {k}")
    out, seen = [], set()
    codes = [chr(65 + i) for i in range(26)] + [chr(65 + i) + chr(65 + j) for i in range(26) for j in range(26)]
    for code in codes:
        ids = tok.encode((" " + code) if form == "spaced" else code, add_special_tokens=False)
        if len(ids) != 1:
            continue
        if ids[0] in seen:
            raise ValueError(f"label {code!r} collides with an earlier label's token {ids[0]}")
        seen.add(ids[0])
        out.append(code)
        if len(out) == k:
            return out
    raise ValueError(f"only {len(out)} single-token letter labels exist in this tokenizer, {k} were asked for")


def question_body(tok, eq: dict, fmt: Format):
    """The question's own text (the part after "<state>\\n\\n") and its label strings (" A", ...; " yes", " no")."""
    kind, instructions = eq["kind"], eq["instructions"]
    options = ["yes", "no"] if kind == "noul" else eq["options"]
    if fmt.tail == "spaced":
        if kind == "noul":
            return f"{instructions}\n\nAnswer with yes or no, and nothing else:", [" yes", " no"]
        labs = letter_labels(tok, len(options), fmt.label_form())
        listing = "\n".join(f"{lab}. {o}" for lab, o in zip(labs, options, strict=True))
        return f"{instructions}\n\nOptions:\n{listing}\n\n{ANSWER_LINE}", [" " + lab for lab in labs]
    if kind == "noul":
        return f"{instructions}\nAnswer yes or no.", [" yes", " no"]
    labs = letter_labels(tok, len(options), fmt.label_form())
    if kind == "score":
        legend = "; ".join(f"{lab} = {o}" for lab, o in zip(labs, options, strict=True))
        return f"{instructions}\nRate on this scale: {legend}.\n{LETTERS_INSTRUCTION}", [" " + lab for lab in labs]
    listing = "\n".join(f"{lab}. {o}" for lab, o in zip(labs, options, strict=True))
    return f"{instructions}\nOptions:\n{listing}\n{LETTERS_INSTRUCTION}", [" " + lab for lab in labs]


def chat_turn(tok, content: str, fmt: Format) -> str:
    """The chat prompt: the system turn if the format has one, `content` as the user turn, the template's generation
    prompt with thinking disabled, then "Answer:" under the prefill slot."""
    messages = [{"role": "user", "content": content}]
    if fmt.system_prompt:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    for open_, close in THINK_MARKERS:
        if out.count(open_) != out.count(close):
            raise ValueError(f"thinking left open in the chat prompt: {out[-80:]!r}")
    return out + ("Answer:" if fmt.slot == "prefill" else "")


def pad_prompt(ids, n, k, token, where, head_len):
    """Insert k pad tokens so a prompt whose shared prefix is ids[:n] gets a prefix of n + k tokens."""
    if not k:
        return ids
    at = {"between": n, "front": 0, "user": head_len}[where]
    return ids[:at] + [token] * k + ids[at:]


# ---- labels and the readout ------------------------------------------------------------------------------------------


def one_token(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 and tok.decode(ids) == text else None


def byte_token(tok, ch):
    """The id of a byte-fallback token for one ASCII character ("<0x41>" for "A"), or None."""
    if len(ch) != 1 or ord(ch) > 127:
        return None
    name = f"<0x{ord(ch):02X}>"
    tid = tok.convert_tokens_to_ids(name)
    unk = getattr(tok, "unk_token_id", None)
    return tid if isinstance(tid, int) and tid >= 0 and tid != unk and tok.convert_ids_to_tokens(tid) == name else None


def label_forms(tok, cand, fmt: Format):
    """The token ids read for one label string: one token under `single`, every single-token form under `summed`."""
    lab = cand.strip()
    if fmt.variants == "single":
        text = (" " + lab) if fmt.slot == "prefill" else lab
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"label {text!r} is {len(ids)} tokens; the letter readout needs one")
        return (ids[0],)
    texts = [" " + lab, lab]
    if lab.isalpha() and lab.islower():  # yes / no
        texts += [" " + lab.capitalize(), lab.capitalize()]
    forms = [one_token(tok, t) for t in texts] + [byte_token(tok, lab)]
    out = tuple(dict.fromkeys(t for t in forms if t is not None))
    if not out:
        raise ValueError(f"label {lab!r} has no single-token form in this tokenizer")
    return out


def label_groups(tok, cands, fmt: Format):
    """Per label, the token ids it is read from (a tuple each); two labels never share a token."""
    groups = [label_forms(tok, c, fmt) for c in cands]
    flat = [t for g in groups for t in g]
    if len(set(flat)) != len(flat):
        raise ValueError(f"label forms collide: {cands}")
    return groups


def label_probabilities(logits, groups, variants):
    """One distribution over the labels from the last position's logits (float64): a softmax over the label tokens
    under `single`; under `summed` the log-softmax over every allowed token, each label's probability the sum over its
    forms, then normalised."""
    if variants == "single":
        z = logits[[g[0] for g in groups]].numpy()
    else:
        flat = [t for g in groups for t in g]
        a = logits[flat].numpy()
        a = a - a.max()
        lp = a - math.log(sum(math.exp(x) for x in a))
        at = {t: i for i, t in enumerate(flat)}
        out = []
        for g in groups:
            v = [float(lp[at[t]]) for t in g]
            m = max(v)
            out.append(m + math.log(sum(math.exp(x - m) for x in v)))
        z = out
    z = [float(x) for x in z]
    top = max(z)
    e = [math.exp(x - top) for x in z]
    s = sum(e)
    return [x / s for x in e]


def apply_temperature(p, T: float):
    """softmax(log p / T) in float64; T == 1 returns p unchanged."""
    p = [float(x) for x in p]
    if T == 1.0:
        return p
    z = [(math.log(x) if x > 0 else -math.inf) / T for x in p]
    top = max(z)
    e = [math.exp(x - top) for x in z]
    s = sum(e)
    return [x / s for x in e]


def choice_confidence(p) -> float:
    k = len(p)
    return 1.0 if k == 1 else float((max(p) - 1.0 / k) / (1.0 - 1.0 / k))


def score_confidence(p) -> float:
    k = len(p)
    if k == 1:
        return 1.0
    mode = max(range(k), key=lambda i: (p[i], -i))  # the first of equal maxima, as numpy's argmax
    dist = float(sum(pi * abs(i - mode) for i, pi in enumerate(p)))
    centre = (k - 1) / 2
    uniform_mad = sum(abs(i - centre) for i in range(k)) / k
    return max(0.0, 1.0 - dist / uniform_mad)


def top_index(keys, p) -> int:
    """The most probable option; among options whose probabilities are exactly equal, the one whose key sorts first
    (Unicode code-point order), so the choice never depends on the order of the keys on the wire."""
    m = max(p)
    return min((i for i in range(len(p)) if p[i] == m), key=lambda i: keys[i])


def to_answer(q: Question, keys, p) -> dict:
    p = [float(x) for x in p]
    if q.type == "noul":
        return {"type": "noul", "noul": p[0]}
    if q.type == "choice":
        return {
            "type": "choice",
            "choice": keys[top_index(keys, p)],
            "confidence": choice_confidence(p),
            "probabilities": dict(zip(keys, p, strict=True)),
        }
    return {
        "type": "score",
        "score": float(sum(i * pi for i, pi in enumerate(p))),
        "confidence": score_confidence(p),
        "legend": {str(i): c for i, c in enumerate(q.criteria)},
        "probabilities": dict(zip(keys, p, strict=True)),
    }


# ---- the reference ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prepared:
    """What one call feeds the model: the token ids (padding included), the label token ids per label, the answer keys
    in the order the labels are read, whether the readout is reversed to (yes, no), and the padded prefix length."""

    ids: list
    labels: list
    keys: list
    swapped: bool
    prefix_tokens: int


class Reference:
    def __init__(self, model, tokenizer, config: dict, block_size: int | None = None):
        """`model` and `tokenizer` of the repository's weights, `config` its decision_config.json (a dict)."""
        self.model, self.tok, self.config = model, tokenizer, config
        if config.get("schema") != SCHEMA:
            raise ValueError(f"decision_config.json is schema {config.get('schema')!r}; this reference reads {SCHEMA}")
        prompt, serving = config.get("prompt") or {}, config.get("serving") or {}
        self.fmt = Format.of(prompt.get("format") or {})
        if prompt.get("label_form") != self.fmt.label_form():
            raise ValueError(
                f"prompt.label_form is {prompt.get('label_form')!r}; the format needs {self.fmt.label_form()!r}"
            )
        self.noul_rendering = prompt.get("noul_rendering")
        if self.noul_rendering not in NOUL_RENDERINGS:
            raise ValueError(
                f"prompt.noul_rendering is {self.noul_rendering!r}; this reference reads {NOUL_RENDERINGS}"
            )
        fp = prompt.get("fingerprint") or {}
        self.hide_index_keys, self.desnake_labels = (
            bool(fp.get("hide_index_keys", True)),
            bool(fp.get("desnake_labels", True)),
        )
        self.pad_unit = pad_unit_of(serving.get("pad_to", "none"), block_size)
        self.pad_where = fp.get("pad_where", "front")
        if self.pad_where not in PAD_PLACES:
            raise ValueError(f"prompt.fingerprint.pad_where is {self.pad_where!r}; this reference reads {PAD_PLACES}")
        temps = config.get("temperatures") or {}
        self.temperatures = {k: float(v) for k, v in temps.items() if v is not None}
        for k, v in self.temperatures.items():
            if not v > 0:
                raise ValueError(f"temperatures.{k} is {v}; a temperature is positive")
        self.softcap = (config.get("readout") or {}).get("softcap")
        have = model_softcap(model)
        if (self.softcap is None) != (have is None) or (have is not None and abs(float(self.softcap) - have) > 1e-9):
            raise ValueError(f"readout.softcap is {self.softcap!r} but the model applies {have!r}")

    @classmethod
    def load(cls, repository, revision=None, dtype=None, device=None, block_size=None) -> Reference:
        """The model, tokenizer and decision_config.json of a repository: a directory, or `owner/name[@revision]`.
        `dtype`: "float32", "bfloat16", "float16", or None for the checkpoint's own."""
        config = read_config(repository, revision)
        name, at = str(repository), revision
        if not Path(name).is_dir():
            name, _, pinned = name.partition("@")
            at = pinned or revision
        kw = {"dtype": getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else (dtype or "auto")}
        tok = AutoTokenizer.from_pretrained(name, revision=at)
        model = AutoModelForCausalLM.from_pretrained(name, revision=at, **kw).eval()
        if device:
            model = model.to(device)
        return cls(model, tok, config, block_size=block_size)

    def temperature_of(self, qtype: str) -> float:
        """The temperature a question of this type is served at: its own, or the global one."""
        return self.temperatures.get(qtype, self.temperatures.get("global", 1.0))

    def engine_question(self, q: Question):
        """(engine question, keys in engine order, swapped): a yes/no question under the letters renderings is asked as
        a two-option choice read false side first and reversed to (yes, no) afterwards."""
        if q.type == "noul" and self.noul_rendering != "words":
            got = noul_as_letters(q, keys_shown=self.noul_rendering == "letters-keys")
            if got is not None:
                return got[0], got[1], True
        eq, keys = to_engine_question(q, self.hide_index_keys, self.desnake_labels)
        return eq, keys, False

    def prepare(self, state, question: dict) -> Prepared:
        """The model's input for one question about a state."""
        text_of_state = fmt_state(state)
        if IMAGE_DATA_URL.search(text_of_state):
            raise ValueError("the state carries an image; this reference serves text")
        q = Question.of(question)
        eq, keys, swapped = self.engine_question(q)
        text, cands = question_body(self.tok, eq, self.fmt)
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)  # noqa: E731
        full = enc(chat_turn(self.tok, f"{text_of_state}\n\n{text}", self.fmt))
        # where the state ends: the tokens before the question's own, or, when the question's tokens do not end the
        # prompt, where two prompts that differ in the question's first letter part
        mark = ""
        tail = chat_turn(self.tok, mark, self.fmt).split(mark)[1]
        suffix = enc(text + tail)
        n = len(full) - len(suffix)
        if n > 0 and full[n:] == suffix:
            prefix = full[:n]
        else:
            a = enc(chat_turn(self.tok, f"{text_of_state}\n\nA", self.fmt))
            b = enc(chat_turn(self.tok, f"{text_of_state}\n\nB", self.fmt))
            n = 0
            while a[n] == b[n]:
                n += 1
            prefix = a[:n]
        if full[:n] != prefix:
            raise ValueError(
                "the question's text merges with the state's last token; a question must not start with whitespace"
            )
        k = (-n % self.pad_unit) if self.pad_unit else 0
        ids = pad_prompt(full, n, k, PAD_TOKEN, self.pad_where, len(enc(USER_HEAD)))
        return Prepared(ids, label_groups(self.tok, cands, self.fmt), keys, swapped, n + k)

    def readout(self, prepared: Prepared) -> list:
        """The label distribution, in the order of `prepared.labels`: one forward pass, the last position's logits."""
        ids = torch.tensor([prepared.ids], device=self.model.device)
        with torch.no_grad():
            logits = self.model(ids, logits_to_keep=1).logits[0, -1].double().cpu()
        return label_probabilities(logits, prepared.labels, self.fmt.variants)

    def decide(self, state, question: dict) -> dict:
        """One decision: the System One answer to one question about a state (the value under its name in `answers`)."""
        q = Question.of(question)
        prepared = self.prepare(state, question)
        p, keys = self.readout(prepared), prepared.keys
        if prepared.swapped:  # read false first; from here on (yes, no)
            p, keys = p[::-1], ["yes", "no"]
        return to_answer(q, keys, apply_temperature(p, self.temperature_of(q.type)))


# ---- command line ----------------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One decisio decision per call, through plain transformers.")
    ap.add_argument("repository", help="a directory, or owner/name[@revision] on the Hugging Face Hub")
    ap.add_argument("--revision", help="the repository's revision (when not given as owner/name@revision)")
    ap.add_argument("--state", help="the state, as text")
    ap.add_argument("--question", help="one question, a JSON object ({'type': 'noul' | 'choice' | 'score', ...})")
    ap.add_argument("--request", help="a /v1/systemone body (state and questions): each question is one call")
    ap.add_argument("--dtype", default=None, help="float32, bfloat16 or float16 (default: the checkpoint's own)")
    ap.add_argument("--device", default=None, help="where to run the model (default: transformers' own)")
    ap.add_argument(
        "--block-size", type=int, default=None, help=f"the engine's block for `pad_to: block` (default {QWEN_BLOCK})"
    )
    args = ap.parse_args(argv)
    if bool(args.request) == bool(args.state is not None and args.question):
        ap.error("give --request, or --state with --question")
    ref = Reference.load(args.repository, args.revision, args.dtype, args.device, args.block_size)
    if args.request:
        body = json.loads(Path(args.request).read_text())
        out = {"answers": {name: ref.decide(body["state"], q) for name, q in body["questions"].items()}}
    else:
        out = ref.decide(args.state, json.loads(args.question))
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
