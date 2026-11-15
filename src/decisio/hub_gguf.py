# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""GGUF files that llama.cpp's and Ollama's `/v1/systemone` route serves with decisio's own prompt.

llama.cpp's server (and Ollama 0.40.4 and later, which hands every GGUF decision request to it) answers `/v1/systemone`
for a GGUF that says how to read it: `<arch>.decision.type`, a chat template named `systemone` and, optionally, the
temperature of each question type. For the letter readout no tensor is added: the server renders the template once per
question, reads the logits of one label token per option at the prompt's last position (`A` .. `Z`, then `AA`, `AB`,
.. whose bare form is one token) and takes the softmax at the type's temperature. That is decisio's readout.

This module writes those keys for a Gemma base or a tuned repository of one:

    systemone_template(tok, settings)   the Jinja template: decisio's prompt (system turn, header layout, spaced tail,
                                        template slot, yes/no as letters), built from decisio's own pieces
    decision_keys(tok, settings)        the keys and values to add, and notes on what the file cannot carry
    write_gguf(src, dst, keys)          a copy of `src` with the keys added; tensors are copied byte for byte

    python -m decisio.hub_gguf template --base gemma-4-12b
    python -m decisio.hub_gguf keys --base tachara-ai/Decisio-12b-tuned-calibrated@baa731aa
    python -m decisio.hub_gguf write --base gemma-4-12b gemma-4-12B-it-Q4_K_M.gguf decisio-gemma-4-12b-Q4_K_M.gguf
    python -m decisio.hub_gguf check --base gemma-4-12b decisio-gemma-4-12b-Q4_K_M.gguf

`write` and `check` need the `gguf` extra (the `gguf` package, llama.cpp's own reader and writer).

The type is `pplx-decider` by default: the letter readout with label codes of one or two letters, and the letter type
llama.cpp lets read images. It does not give the template the request's other questions, so the header lists only
the question being asked: a request of one question gets decisio's prompt byte for byte, a request of several gets
a header of one line per prompt instead of all of them. `nimble` gives the template every question (the full header)
but llama.cpp refuses images for it.

What the file cannot carry (each is reported in `decision_keys`'s notes):
  - Gemma's summed label variants: decisio adds the probability of " A" to that of "A"; llama.cpp reads "A" only.
  - Label codes past the 68th option: decisio keeps codes whose spaced form (" BR") is one token, llama.cpp codes whose
    bare form is; the lists part at the 69th code in Gemma's tokenizer, and a question with more options is shown
    llama.cpp's codes, which it reads consistently but which are not decisio's prompt.
  - Choice temperature bands: llama.cpp's buckets are 2, 3-5, 6-10 and 11 or more options; a band boundary inside one
    of them takes the value of the band that holds the bucket's smallest count.
  - The commit rule, thinking and read twice: a repository that sets one is refused.
  - Images: llama.cpp puts a request's images before the state, not at their places in it, and has no per-question
    images and no Box.
  - A list state: llama.cpp reads it as chat messages.

The template's own text comes from decisio's prompt code: the chat template's wrapper around the user turn
(`decisio.readout.letters.chat_turn`), the header (`QUESTIONS_HEAD`), the answer line (`ANSWER_LINE`), the default
instructions and the option rules of `decisio.serve.systemone` (index keys hidden, snake_case labels shown with
spaces, an option shown as its description, the `key: description` form when two would read the same).
tests/unit/test_hub_gguf.py renders the template and compares it with the engine's prompt on every rule.
"""

from __future__ import annotations

import argparse
import json
import os
import string
import sys
from dataclasses import dataclass, field, replace

from decisio.families import BASES, Family
from decisio.readout.letters import QUESTIONS_HEAD, PromptFormat, chat_turn
from decisio.readout.spaced import ANSWER_LINE
from decisio.serve.systemone import DEFAULT_INSTRUCTIONS

TYPES = ("pplx-decider", "nimble")
DEFAULT_TYPE = "pplx-decider"
TEMPLATE_NAME = "systemone"
TEMPLATE_KEY = f"tokenizer.chat_template.{TEMPLATE_NAME}"
TEMPLATE_NAMES = "tokenizer.chat_templates"  # the names of a GGUF's named chat templates
# llama.cpp's option-count buckets for a choice temperature (server-decision.cpp, get_temperature), smallest and largest
BUCKETS = {"2": (2, 2), "3_5": (3, 5), "6_10": (6, 10), "11": (11, 255)}
MARK = "\ue000"  # stands for the user turn while the chat template's wrapper is cut out


@dataclass(frozen=True)
class Settings:
    """What a GGUF carries for one base or tuned repository: its prompt format, yes/no rendering and temperatures."""

    name: str
    fmt: PromptFormat
    noul_rendering: str
    temperatures: dict  # question type -> T
    choice_bands: dict = field(default_factory=dict)  # "lo-hi" -> T
    tokenizer: tuple[str, str | None] = ("", None)  # repository and revision of the tokenizer


def settings_of_base(fam: Family) -> Settings:
    t = fam.temperature
    return Settings(
        name=fam.key,
        fmt=PromptFormat(
            tail=fam.prompt_tail,
            slot=fam.answer_slot,
            variants=fam.label_variants,
            system_prompt=fam.system_prompt,
            layout=fam.layout,
        ),
        noul_rendering=fam.noul_rendering,
        temperatures={"choice": fam.choice_temperature or t, "noul": t, "score": t},
        tokenizer=(fam.model, fam.revision),
    )


def settings_of_config(cfg: dict, name: str, revision: str | None = None) -> Settings:
    """A repository's settings, from its decision_config.json: its base's, changed by its `defaults` (a tuned model's
    temperatures, choice bands and layout); a repository that thinks, reads twice or commits is refused."""
    fam = BASES[cfg["base"]]
    d = cfg.get("defaults")
    if d is None:
        return replace(settings_of_base(fam), name=name, tokenizer=(name, revision))
    for key, what in (("noul_commit", "the commit rule"), ("think", "thinking"), ("read_twice_under", "read twice")):
        if d.get(key):
            raise ValueError(f"{name} sets {what} ({key}), which a GGUF's decision route cannot apply")
    p = cfg["prompt"]["format"]
    fmt = PromptFormat(
        tail=p["tail"],
        slot=p["slot"],
        variants=p["variants"],
        system_prompt=p.get("system_prompt", False),
        layout=d.get("layout", "plain"),
    )
    return Settings(
        name=(cfg.get("flavor") or {}).get("served_name") or name,
        fmt=fmt,
        noul_rendering=cfg["prompt"]["noul_rendering"],
        temperatures=dict(d["temperatures"]),
        choice_bands=dict(d.get("choice_bands") or {}),
        tokenizer=(name, revision),
    )


def settings_of_repository(spec: str) -> Settings:
    """A repository's settings (owner/name[@revision] or a directory; decisio.hub.open_repository checks its file)."""
    from decisio.hub import open_repository

    repo = open_repository(spec)
    return settings_of_config(repo.config, repo.model, repo.revision)


def settings_of(spec: str) -> Settings:
    return settings_of_base(BASES[spec]) if spec in BASES else settings_of_repository(spec)


def supported(s: Settings) -> list[str]:
    """Why a format cannot be written as a template (empty when it can)."""
    f, why = s.fmt, []
    if f.tail != "spaced":
        why.append(f"the {f.tail!r} tail (the template writes the spaced tail)")
    if f.slot != "template":
        why.append(
            f"the {f.slot!r} answer slot: llama.cpp reads bare labels at the prompt's end, decisio reads ' A' after "
            "'Answer:'"
        )
    if f.layout != "header":
        why.append(f"the {f.layout!r} layout (the template writes the header layout)")
    if s.noul_rendering != "letters":
        why.append(f"yes/no rendered as {s.noul_rendering!r} (the template writes 'letters')")
    if f.read_twice_under:
        why.append("read twice")
    return why


# ---- the template ------------------------------------------------------------------------------------------------


def _lit(text: str) -> str:
    """A Jinja output of fixed text: a string literal, so that no character of it is read as template syntax."""
    esc = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    return "{{ '" + esc + "' }}"


def _q(text: str) -> str:
    """A double-quoted Jinja string literal with the escapes both jinja2 and llama.cpp's lexer read (\\n, \\t, \\r,
    \\b, \\f, \\xNN, \\\\, \\"); other characters, non-ASCII ones included, are written as they are."""
    named = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r", "\b": "\\b", "\f": "\\f"}
    return (
        '"' + "".join(named.get(c) or (f"\\x{ord(c):02x}" if ord(c) < 32 or ord(c) == 127 else c) for c in text) + '"'
    )


def _replace_chain(var: str, pairs) -> str:
    return var + "".join(f" | replace({_q(a)}, {_q(b)})" for a, b in pairs)


# JSON string escapes (json.dumps, ensure_ascii=False) and Python repr escapes (str.__repr__), as replace chains
_CONTROL = [chr(c) for c in range(1, 32) if chr(c) not in "\n\r\t\b\f"] + ["\x7f"]
_JSON_ESC = [
    ("\\", "\\\\"),
    ('"', '\\"'),
    ("\n", "\\n"),
    ("\r", "\\r"),
    ("\t", "\\t"),
    ("\b", "\\b"),
    ("\f", "\\f"),
] + [(c, f"\\u{ord(c):04x}") for c in _CONTROL if c != "\x7f"]
_REPR_ESC = [("\\", "\\\\"), ("\n", "\\n"), ("\r", "\\r"), ("\t", "\\t"), ("\b", "\\x08"), ("\f", "\\x0c")] + [
    (c, f"\\x{ord(c):02x}") for c in _CONTROL
]

_MACROS = (
    # compact JSON (decisio.serve.systemone.render_text: json.dumps(ensure_ascii=False, separators=(",", ":")))
    "{%- macro cj(v) -%}"
    "{%- if v is string -%}" + "{{ '\"' }}{{ " + _replace_chain("v", _JSON_ESC) + " }}{{ '\"' }}"
    "{%- elif v is none -%}null{%- elif v is sameas true -%}true{%- elif v is sameas false -%}false"
    "{%- elif v is mapping -%}{{ '{' }}{%- for k, x in v.items() -%}{{ cj(k) }}:{{ cj(x) }}"
    "{%- if not loop.last -%},{%- endif -%}{%- endfor -%}{{ '}' }}"
    "{%- elif v is iterable -%}[{%- for x in v -%}{{ cj(x) }}{%- if not loop.last -%},{%- endif -%}{%- endfor -%}]"
    "{%- else -%}{{ v }}{%- endif -%}{%- endmacro -%}"
    # render_text: a string as it is, null as nothing, anything else as compact JSON
    "{%- macro txt(v) -%}{%- if v is string -%}{{ v }}{%- elif v is none -%}{%- else -%}{{ cj(v) }}{%- endif -%}"
    "{%- endmacro -%}"
    # Python's repr, for the values of an object state (decisio.readout.letters.fmt_state: f"{k}: {v}")
    "{%- macro pyrepr(v) -%}"
    "{%- if v is string -%}"
    "{%- if \"'\" in v and '\"' not in v -%}{{ '\"' }}{{ " + _replace_chain("v", _REPR_ESC) + " }}{{ '\"' }}"
    '{%- else -%}{{ "\'" }}{{ ' + _replace_chain("v", _REPR_ESC + [("'", "\\'")]) + ' }}{{ "\'" }}{%- endif -%}'
    "{%- elif v is none -%}None{%- elif v is sameas true -%}True{%- elif v is sameas false -%}False"
    "{%- elif v is mapping -%}{{ '{' }}{%- for k, x in v.items() -%}{{ pyrepr(k) }}: {{ pyrepr(x) }}"
    "{%- if not loop.last -%}, {% endif -%}{%- endfor -%}{{ '}' }}"
    "{%- elif v is iterable -%}[{%- for x in v -%}{{ pyrepr(x) }}{%- if not loop.last -%}, {% endif -%}{%- endfor -%}]"
    "{%- else -%}{{ v }}{%- endif -%}{%- endmacro -%}"
    "{%- macro pystr(v) -%}{%- if v is string -%}{{ v }}{%- else -%}{{ pyrepr(v) }}{%- endif -%}{%- endmacro -%}"
    # the question's text, or the type's default (render_text(...).strip() or DEFAULT_INSTRUCTIONS)
    "{%- macro instr(t, x) -%}{%- set s = txt(x) | trim -%}{%- if s -%}{{ s }}"
    + "".join(f"{{%- elif t == {_q(k)} -%}}{_lit(v)}" for k, v in DEFAULT_INSTRUCTIONS.items())
    + "{%- endif -%}{%- endmacro -%}"
    # snake_label: lowercase words joined by single underscores, at least two words, shown with spaces
    "{%- macro desnake(k) -%}{%- if '_' in k and not k.startswith('_') and not k.endswith('_') and '__' not in k and ("
    + _replace_chain("k", [(c, "") for c in string.ascii_lowercase + "_"])
    + ") == '' -%}{{ k | replace('_', ' ') }}{%- else -%}{{ k }}{%- endif -%}{%- endmacro -%}"
    # a word stem of index_keys' INDEX_KEY, ([A-Za-z]+[_\- ]?)?, given as the key without its number
    "{%- macro stem_ok(s) -%}{%- set b = s[:-1] if (s.endswith('_') or s.endswith('-') or s.endswith(' ')) else s -%}"
    "{%- if s == '' or (b != '' and ("
    + _replace_chain("b", [(c, "") for c in string.ascii_letters])
    + ") == '') -%}1{%- endif -%}{%- endmacro -%}"
    # the stem of a key numbered n (written with up to two leading zeros), or nothing
    "{%- macro stem_of(k, n) -%}{%- set ns = namespace(out='') -%}"
    "{%- for d in [n | string, '0' ~ (n | string), '00' ~ (n | string)] -%}"
    "{%- if ns.out == '' and k.endswith(d) and stem_ok(k[:(k | length) - (d | length)]) == '1' -%}"
    "{%- set ns.out = '=' ~ (k[:(k | length) - (d | length)] | lower) -%}{%- endif -%}{%- endfor -%}"
    "{{ ns.out }}{%- endmacro -%}"
    # index_keys: one shared stem plus a number, the numbers 0..n-1 or 1..n, every option described
    "{%- macro is_index(opts) -%}{%- set n = opts | length -%}{%- set r = namespace(ok=false) -%}"
    "{%- if n >= 2 -%}{%- for base in [0, 1] -%}{%- set ns = namespace(ok=true, stem=none, seen='|') -%}"
    "{%- for o in opts -%}{%- set f = namespace(stem='', num=-1) -%}"
    "{%- for i in range(base, base + n) -%}{%- if f.stem == '' -%}{%- set s = stem_of(o.key, i) -%}"
    "{%- if s != '' -%}{%- set f.stem = s -%}{%- set f.num = i -%}{%- endif -%}{%- endif -%}{%- endfor -%}"
    "{%- if f.stem == '' or ('|' ~ f.num ~ '|') in ns.seen or (ns.stem is not none and ns.stem != f.stem) "
    "or (txt(o.description) | trim) == '' -%}{%- set ns.ok = false -%}"
    "{%- else -%}{%- set ns.stem = f.stem -%}{%- set ns.seen = ns.seen ~ f.num ~ '|' -%}{%- endif -%}{%- endfor -%}"
    "{%- if ns.ok -%}{%- set r.ok = true -%}{%- endif -%}{%- endfor -%}{%- endif -%}"
    "{%- if r.ok -%}1{%- endif -%}{%- endmacro -%}"
    # a choice option described (decisio's describe_options) or shown as its key
    "{%- macro described(o) -%}{%- set d = txt(o.description) -%}{%- if d != '' -%}{{ d }}{%- else -%}"
    "{{ desnake(o.key) }}{%- endif -%}{%- endmacro -%}"
    # how each option of a choice reads: 'index' (descriptions), 'bare' (keys), 'described' or 'full' (key: description)
    "{%- macro choice_mode(opts) -%}{%- set e = namespace(empty=true, dup=false) -%}"
    "{%- for o in opts -%}{%- if txt(o.description) != '' -%}{%- set e.empty = false -%}{%- endif -%}{%- endfor -%}"
    "{%- if is_index(opts) == '1' -%}index{%- elif e.empty -%}bare{%- else -%}"
    "{%- for a in opts -%}{%- set i = loop.index0 -%}{%- for b in opts -%}"
    "{%- if loop.index0 > i and not e.dup and described(a) == described(b) -%}{%- set e.dup = true -%}{%- endif -%}"
    "{%- endfor -%}{%- endfor -%}"
    "{%- if e.dup -%}full{%- else -%}described{%- endif -%}{%- endif -%}{%- endmacro -%}"
)


def _choice_shown() -> str:
    """The text of each option of a choice question in its mode (decisio.serve.systemone.to_engine_question)."""
    return (
        "{%- if mode == 'index' -%}{{ desnake(txt(o.description)) }}"
        "{%- elif mode == 'bare' -%}{{ desnake(o.key) }}"
        "{%- elif mode == 'described' -%}{{ described(o) }}"
        "{%- else -%}{%- set d = txt(o.description) -%}{%- if d != '' -%}{{ o.key }}: {{ d }}{%- else -%}{{ o.key }}"
        "{%- endif -%}{%- endif -%}"
    )


def _wrapper(tok, fmt: PromptFormat) -> tuple[str, str]:
    """The chat template's text before and after a question's user turn, as decisio renders it."""
    out = chat_turn(tok, MARK, fmt.core())
    if out.count(MARK) != 1:
        raise ValueError("the chat template does not keep the user turn as written")
    before, after = out.split(MARK)
    return before, after


def systemone_template(tok, s: Settings, decision_type: str = DEFAULT_TYPE) -> str:
    """The `systemone` template for llama.cpp's letter types: decisio's prompt for one question."""
    why = supported(s)
    if why:
        raise ValueError(f"{s.name}: a GGUF template cannot reproduce " + "; ".join(why))
    if decision_type not in TYPES:
        raise ValueError(f"decision type must be one of {TYPES}, not {decision_type!r}")
    before, after = _wrapper(tok, s.fmt)
    for text in (before, after, QUESTIONS_HEAD, ANSWER_LINE):
        if "{{" in text or "{%" in text or "{#" in text:
            raise ValueError(f"template text looks like Jinja syntax: {text!r}")
    header_lines = (
        "{%- for q in questions -%}{{ '- ' }}{{ instr(q.type, q.instructions) }}{{ '\\n' }}{%- endfor -%}"
        if decision_type == "nimble"
        else "{{ '- ' }}{{ instr(type, instructions) }}{{ '\\n' }}"
    )
    return (
        _MACROS
        + _lit(before + QUESTIONS_HEAD)
        + header_lines
        + _lit("\n")
        + "{%- for m in images -%}{{ m }}{%- endfor -%}"
        # the state (decisio.readout.letters.fmt_state): a string as it is; an object as "key: value" lines, keys
        # starting with "_" left out
        + "{%- if state is string -%}{{ state }}"
        "{%- elif state is mapping -%}{%- set ns = namespace(first=true) -%}{%- for k, v in state.items() -%}"
        "{%- if not k.startswith('_') -%}{%- if not ns.first -%}{{ '\\n' }}{%- endif -%}{%- set ns.first = false -%}"
        "{{ k }}: {{ pystr(v) }}{%- endif -%}{%- endfor -%}"
        "{%- else -%}{{ pystr(state) }}{%- endif -%}"
        + _lit("\n\n")
        + "{{ instr(type, instructions) }}"
        + _lit("\n\nOptions:\n")
        + "{%- set mode = choice_mode(options) if type == 'choice' else '' -%}"
        "{%- for o in options -%}{{ o.label }}{{ '. ' }}"
        "{%- if type == 'noul' -%}{%- set d = txt(o.description) | trim -%}"
        "{%- if d -%}{{ d }}{%- elif o.key == 'true' -%}Yes{%- else -%}No{%- endif -%}"
        "{%- elif type == 'score' -%}{{ txt(o.description) }}"
        "{%- else -%}" + _choice_shown() + "{%- endif -%}{{ '\\n' }}{%- endfor -%}" + _lit("\n" + ANSWER_LINE + after)
    )


# ---- the keys ----------------------------------------------------------------------------------------------------


def choice_buckets(s: Settings) -> tuple[dict, list[str]]:
    """llama.cpp's choice temperature per option-count bucket, from the repository's bands, and the notes."""
    if not s.choice_bands:
        return {}, []
    bands = []
    for name, t in s.choice_bands.items():
        lo, hi = (int(x) for x in name.split("-"))
        bands.append((lo, hi, float(t)))
    out, notes = {}, []
    for bucket, (lo, hi) in BUCKETS.items():
        hit = [b for b in bands if b[0] <= hi and lo <= b[1]]
        if not hit:
            continue
        at_lo = next((b for b in hit if b[0] <= lo <= b[1]), hit[0])
        out[bucket] = at_lo[2]
        if len({b[2] for b in hit}) > 1:
            parts = ", ".join(f"{b[0]}-{b[1]}: {b[2]}" for b in hit)
            notes.append(
                f"choice bucket {lo}-{hi} options spans bands with different temperatures ({parts}); "
                f"it takes {at_lo[2]} ({at_lo[0]}-{at_lo[1]})"
            )
    return out, notes


def decision_keys(tok, s: Settings, arch: str = "gemma4", decision_type: str = DEFAULT_TYPE) -> tuple[dict, list[str]]:
    """{key: value} to add to the GGUF, and the notes on what it does not reproduce."""
    keys = {f"{arch}.decision.type": decision_type, TEMPLATE_KEY: systemone_template(tok, s, decision_type)}
    for qtype in ("choice", "noul", "score"):
        keys[f"{arch}.decision.temperature.{qtype}"] = float(s.temperatures[qtype])
    buckets, notes = choice_buckets(s)
    for bucket, t in buckets.items():
        keys[f"{arch}.decision.temperature.choice.{bucket}"] = t
    if s.fmt.variants == "summed":
        notes.append("labels are read in their bare form only (decisio sums ' A' and 'A')")
    notes.append(
        "a choice of more than 68 options is shown llama.cpp's label codes, which part from decisio's at the 69th"
    )
    if decision_type == "pplx-decider":
        notes.append("a request of several questions gets a header of one line (the question asked)")
    else:
        notes.append("llama.cpp refuses images for nimble")
    return keys, notes


# ---- the file ----------------------------------------------------------------------------------------------------


def _gguf():
    try:
        import gguf
    except ImportError as e:  # pragma: no cover - the extra is optional
        raise SystemExit("this needs the gguf extra: pip install 'decisio[gguf]'") from e
    return gguf


def write_gguf(src: str, dst: str, keys: dict) -> None:
    """Copy `src` to `dst` with `keys` added; an existing decision key or systemone template is replaced.

    Every other key is kept, the other named chat templates (`tokenizer.chat_template.<name>`) among them, and the list
    of names (`tokenizer.chat_templates`) keeps its names and gains "systemone". A value is a str (a GGUF string) or a
    float (a GGUF float32); any other type is refused, so a key cannot be written in a type its reader does not expect.
    `dst` must not be `src`: the reader maps `src` into memory while the writer truncates `dst`."""
    gguf = _gguf()
    if os.path.realpath(src) == os.path.realpath(dst) or (os.path.exists(dst) and os.path.samefile(src, dst)):
        raise ValueError(f"the output {dst} is the input file; write the copy to another path")
    for k, v in keys.items():
        if not isinstance(v, (str, float)):
            raise TypeError(f"{k}: a value must be a str or a float, not {type(v).__name__}")
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    prefix = f"{arch}.decision."
    if any(not k.startswith(prefix) and k != TEMPLATE_KEY for k in keys):
        raise ValueError(f"every key must be {prefix}* or {TEMPLATE_KEY}")
    names = list(r.fields[TEMPLATE_NAMES].contents()) if TEMPLATE_NAMES in r.fields else []
    if TEMPLATE_KEY in keys and TEMPLATE_NAME not in names:
        names.append(TEMPLATE_NAME)
    w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
    for f in r.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."):
            continue
        if f.name.startswith(prefix) or f.name == TEMPLATE_KEY or f.name == TEMPLATE_NAMES:
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    if names:
        w.add_array(TEMPLATE_NAMES, names)
    for k, v in keys.items():
        if isinstance(v, str):
            w.add_string(k, v)
        else:
            w.add_float32(k, v)
    for t in r.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()


def read_keys(path: str) -> tuple[str, dict]:
    """The architecture and the decision keys (and systemone template) a GGUF carries."""
    gguf = _gguf()
    r = gguf.GGUFReader(path)
    arch = r.fields["general.architecture"].contents()
    return arch, {
        f.name: f.contents()
        for f in r.fields.values()
        if f.name.startswith(f"{arch}.decision.") or f.name == TEMPLATE_KEY
    }


def compare_keys(got: dict, want: dict) -> list[str]:
    diff = []
    for k in sorted(set(got) | set(want)):
        a, b = got.get(k), want.get(k)
        same = a == b if not isinstance(b, float) else isinstance(a, float) and abs(a - b) < 1e-6
        if not same:
            diff.append(k if k == TEMPLATE_KEY else f"{k}: {a!r} (want {b!r})")
    return diff


def _tokenizer(s: Settings):
    from transformers import AutoTokenizer

    name, rev = s.tokenizer
    return AutoTokenizer.from_pretrained(name, revision=rev)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m decisio.hub_gguf", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (
        ("template", "print the systemone template"),
        ("keys", "print the keys a GGUF gets, and the notes"),
        ("write", "copy a GGUF with the keys added"),
        ("check", "compare a GGUF's keys with the ones this decisio writes"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--base", required=True, help="a base's key, or a tuned repository (owner/name[@rev] or a dir)")
        p.add_argument("--type", default=DEFAULT_TYPE, choices=TYPES)
        if name == "write":
            p.add_argument("src")
            p.add_argument("dst")
        if name == "check":
            p.add_argument("file")
    a = ap.parse_args(argv)
    s = settings_of(a.base)
    tok = _tokenizer(s)
    if a.cmd == "template":
        sys.stdout.write(systemone_template(tok, s, a.type))
        return 0
    if a.cmd in ("keys", "write"):
        arch = "gemma4"
        if a.cmd == "write":
            arch = _gguf().GGUFReader(a.src).fields["general.architecture"].contents()
        keys, notes = decision_keys(tok, s, arch, a.type)
        if a.cmd == "keys":
            shown = {k: (f"<{len(v)} characters>" if k == TEMPLATE_KEY else v) for k, v in keys.items()}
            print(json.dumps({"keys": shown, "notes": notes}, indent=1))
            return 0
        write_gguf(a.src, a.dst, keys)
        print(f"wrote {a.dst}: {', '.join(sorted(keys))}")
        for n in notes:
            print(f"NOTE {n}")
        return 0
    arch, got = read_keys(a.file)
    want, _ = decision_keys(tok, s, arch, a.type)
    diff = compare_keys(got, want)
    if diff:
        print(f"{a.file} differs from {s.name}'s keys at: " + "; ".join(diff))
        return 1
    print(f"{a.file}: the decision keys of {s.name} ({len(want)} keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
