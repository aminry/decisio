# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The GGUF keys for llama.cpp's and Ollama's `/v1/systemone` route (decisio.hub_gguf), on the Gemma tokenizer:

G1  the systemone template, given the inputs llama.cpp's server gives it (pplx-decider: one question, its options in
    request order with llama.cpp's label codes, yes/no false first), renders the token row decisio's engine builds for
    a request of that one question, for every option rule (index keys, snake_case labels, descriptions, duplicates),
    question type, default instructions, object and JSON values, and text that looks like template syntax
G2  under nimble the header lists every question of the request, as decisio's does
G3  past 68 options the prompt differs only in its label codes (llama.cpp's own, read consistently)
G4  the keys: type, template and temperatures for a base; a tuned repository's temperatures and choice bands, mapped
    to llama.cpp's buckets with a note where two bands share one; the commit rule, thinking and the Qwen prompt refused
G5  write_gguf adds the keys to a copy whose tensors are byte for byte the source's, keeps the source's other named
    chat templates and their list (adding systemone), and check reads them back; it refuses to write over its input
    (the same path, a symlink or a hard link) and values that are neither str nor float

  uv run pytest -q tests/unit/test_hub_gguf.py      (downloads the Gemma tokenizer once)
"""

import string

import jinja2
import numpy as np
import pytest

from decisio import hub_gguf as hg
from decisio.families import GEMMA4, QWEN
from decisio.serve import systemone as so
from decisio.serve import vllm_engine as sv

STATE = "Hi, since this morning none of our 40 staff can log in to the dashboard. Payroll is due today."
OBJECT_STATE = {
    "name": "O'Brien",
    "quote": 'say "hi"',
    "both": 'it\'s "x"',
    "multi": "line1\nline2\ttab",
    "n": 3,
    "f": 2.5,
    "t": True,
    "z": None,
    "nested": {"a": [1, "b", {"c": False}]},
    "_hidden": "never shown",
    "empty": "",
    "uni": "café — 日本",
}
QUESTIONS = {
    "index": {
        "type": "choice",
        "instructions": "Which intent?",
        "criteria": {"option_0": "card_arrival", "option_1": "lost card"},
    },
    "one_based": {"type": "choice", "instructions": "x", "criteria": {"label 1": "first", "label 2": "second"}},
    "padded": {"type": "choice", "instructions": "x", "criteria": {"q01": "alpha", "q02": "beta", "q03": "gamma"}},
    "numbers": {"type": "choice", "instructions": "x", "criteria": {"0": "zero", "1": "one"}},
    "sku": {"type": "choice", "instructions": "x", "criteria": {"sku_12": "a widget", "sku_13": "a gadget"}},
    "index_gap": {"type": "choice", "instructions": "x", "criteria": {"option_0": "a", "option_1": None}},
    "index_dup": {"type": "choice", "instructions": "x", "criteria": {"option_0": "same", "option_1": "same"}},
    "snake": {
        "type": "choice",
        "instructions": "x",
        "criteria": {"card_arrival": None, "Refund_pending": None, "one": None},
    },
    "mixed": {
        "type": "choice",
        "instructions": "x",
        "criteria": {"billing": "Invoices", "wrong_item": None, "other": ""},
    },
    "dup": {"type": "choice", "instructions": "x", "criteria": {"a": "Same text", "b": "Same text", "c": None}},
    "json": {
        "type": "choice",
        "instructions": "x",
        "criteria": {"a": {"x": 1, "y": [True, None, 'q"t']}, "b": ["l", 2.5]},
    },
    "obj_instr": {"type": "choice", "instructions": {"task": "an object", "n": 3}, "criteria": {"a": None, "b": None}},
    "blank": {"type": "choice", "instructions": "   ", "criteria": {"a": None, "b": None}},
    "syntax": {
        "type": "choice",
        "instructions": "Specials {{x}} {% y %}",
        "criteria": {"{a}": "opt {{ b }}", "c": "d"},
    },
    "noul": {"type": "noul", "instructions": "  Is it urgent?  "},
    "noul_sides": {"type": "noul", "instructions": "x", "criteria": {"true": "It rains now", "false": "  "}},
    "noul_default": {"type": "noul"},
    "score": {"type": "score", "instructions": "How severe?", "criteria": ["low", None, "high"]},
    "score_json": {"type": "score", "criteria": [{"level": 0}, "one"]},
}


@pytest.fixture(scope="module")
def gemma():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(GEMMA4.model, revision=GEMMA4.revision)


@pytest.fixture(scope="module")
def settings():
    return hg.settings_of_base(GEMMA4)


def llama_codes(tok):
    """llama.cpp's label codes (server-decision.cpp, init): A..Z, then AA..ZZ whose bare form is one token, at most
    255."""
    every = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    return [c for c in every if len(tok.encode(c, add_special_tokens=False)) == 1][:255]


def llama_question(qid, q, codes):
    """The template inputs llama.cpp's server builds for one question (parse_questions, render_options)."""
    crit = q.get("criteria")
    if q["type"] == "choice":
        opts = [{"key": k, "description": d} for k, d in crit.items()]
    elif q["type"] == "score":
        opts = [{"key": str(i), "description": d} for i, d in enumerate(crit)]
    else:
        opts = [{"key": k, "description": (crit or {}).get(k)} for k in ("false", "true")]
    for o, code in zip(opts, codes):
        o["label"] = code
    return {"id": qid, "type": q["type"], "instructions": q.get("instructions"), "options": opts}


def render(template, state, qid, q, codes, all_questions=None):
    inp = llama_question(qid, q, codes)
    return (
        jinja2.Environment()
        .from_string(template)
        .render(
            **inp,
            state=state,
            images=[],
            questions=[llama_question(i, x, codes) for i, x in (all_questions or {}).items()],
        )
    )


class Tokonly(sv.LettersEngine):
    def __init__(self, tok, fmt):
        self.tok, self.fmt, self.pad_unit, self.pad_token, self.pad_where = tok, fmt, None, sv.PAD_TOKEN, "front"


def engine_rows(tok, settings, state, questions):
    """The token rows decisio builds for a request of `questions`, one per question in request order (the server's
    question rules, yes/no as letters, then the engine's rows)."""
    req = so.SystemOneRequest.model_validate({"model": "m", "state": state, "questions": questions})
    eqs = [
        ((so.noul_as_letters(q) if q.type == "noul" else None) or so.to_engine_question(q))[0]
        for q in req.questions.values()
    ]
    rows, _, _ = Tokonly(tok, settings.fmt)._rows(req.state, eqs)
    return [ids for ids, _ in rows]


@pytest.mark.parametrize("state", [STATE, OBJECT_STATE], ids=["text", "object"])
@pytest.mark.parametrize("qid", list(QUESTIONS))
def test_g1_template_renders_the_engine_row(gemma, settings, state, qid):
    template = hg.systemone_template(gemma, settings)
    codes = llama_codes(gemma)
    text = render(template, state, qid, QUESTIONS[qid], codes)
    assert gemma.encode(text, add_special_tokens=False) == engine_rows(gemma, settings, state, {qid: QUESTIONS[qid]})[0]


def test_g2_nimble_header_lists_every_question(gemma, settings):
    template = hg.systemone_template(gemma, settings, "nimble")
    codes = llama_codes(gemma)
    qs = {k: QUESTIONS[k] for k in ("noul", "mixed", "score", "obj_instr")}
    want = engine_rows(gemma, settings, STATE, qs)
    for (qid, q), row in zip(qs.items(), want):
        assert gemma.encode(render(template, STATE, qid, q, codes, all_questions=qs), add_special_tokens=False) == row


def test_g3_past_68_options_only_the_codes_differ(gemma, settings):
    template = hg.systemone_template(gemma, settings)
    codes = llama_codes(gemma)
    q = {"type": "choice", "instructions": "Which?", "criteria": {f"thing {i}": None for i in range(80)}}
    text = render(template, STATE, "q", q, codes)
    want = gemma.decode(engine_rows(gemma, settings, STATE, {"q": q})[0])
    got_lines, want_lines = text.split("\n"), want.split("\n")
    assert len(got_lines) == len(want_lines)
    differ = [i for i, (a, b) in enumerate(zip(got_lines, want_lines)) if a != b]
    assert differ and all(
        a.split(". ", 1)[1] == b.split(". ", 1)[1] for a, b in ((got_lines[i], want_lines[i]) for i in differ)
    )
    assert got_lines[want_lines.index(next(x for x in want_lines if x.endswith("thing 67")))].startswith(codes[67])


def test_g4_base_keys(gemma, settings):
    keys, notes = hg.decision_keys(gemma, settings)
    assert keys["gemma4.decision.type"] == "pplx-decider"
    assert keys[hg.TEMPLATE_KEY] == hg.systemone_template(gemma, settings)
    for t in ("choice", "noul", "score"):
        assert keys[f"gemma4.decision.temperature.{t}"] == GEMMA4.temperature
    assert not any(k.startswith("gemma4.decision.temperature.choice.") for k in keys)
    assert any("bare form" in n for n in notes) and any("69th" in n for n in notes)


TUNED = {
    "base": "gemma-4-12b",
    "prompt": {
        "format": {"tail": "spaced", "slot": "template", "variants": "summed", "system_prompt": True},
        "noul_rendering": "letters",
    },
    "defaults": {
        "temperatures": {"choice": 1.8222, "noul": 3.314, "score": 2.1109},
        "choice_bands": {"2-10": 2.1317, "11-26": 1.8766, "27-255": 1.7181},
        "noul_commit": False,
        "layout": "header",
    },
    "flavor": {"name": "calibrated", "served_name": "Decisio-12b-tuned-calibrated"},
}


def test_g4_tuned_keys_and_bands(gemma):
    s = hg.settings_of_config(TUNED, "owner/tuned", "abc")
    keys, notes = hg.decision_keys(gemma, s)
    t = "gemma4.decision.temperature."
    assert (keys[t + "choice"], keys[t + "noul"], keys[t + "score"]) == (1.8222, 3.314, 2.1109)
    assert [keys[t + f"choice.{b}"] for b in ("2", "3_5", "6_10", "11")] == [2.1317, 2.1317, 2.1317, 1.8766]
    assert any("11-255" in n and "1.7181" in n for n in notes)
    assert s.name == "Decisio-12b-tuned-calibrated"


@pytest.mark.parametrize("key,value", [("noul_commit", True), ("think", {"factor": 1}), ("read_twice_under", 2000)])
def test_g4_refuses_what_a_gguf_cannot_apply(key, value):
    cfg = {**TUNED, "defaults": {**TUNED["defaults"], key: value}}
    with pytest.raises(ValueError, match=key):
        hg.settings_of_config(cfg, "owner/tuned")


def test_g4_refuses_the_qwen_prompt(gemma):
    with pytest.raises(ValueError, match="answer slot"):
        hg.systemone_template(gemma, hg.settings_of_base(QWEN))


def tiny_gguf(gguf, path, templates=True):
    """A small GGUF with a decision key to replace and, optionally, a named chat template (tool_use) of its own."""
    w = gguf.GGUFWriter(str(path), arch="gemma4")
    w.add_string("general.name", "tiny")
    w.add_uint32("gemma4.context_length", 8192)
    w.add_string("gemma4.decision.type", "nimble")  # replaced, not kept twice
    if templates:
        w.add_string("tokenizer.chat_template", "{{ default }}")
        w.add_string("tokenizer.chat_template.tool_use", "{{ tools }}")
        w.add_array("tokenizer.chat_templates", ["tool_use"])
    weights = np.arange(12, dtype=np.float32).reshape(3, 4)
    w.add_tensor("token_embd.weight", weights)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return weights


def test_g5_write_and_check_round_trip(tmp_path, gemma, settings):
    gguf = pytest.importorskip("gguf")
    src, dst = tmp_path / "src.gguf", tmp_path / "dst.gguf"
    weights = tiny_gguf(gguf, src)
    keys, _ = hg.decision_keys(gemma, settings)
    hg.write_gguf(str(src), str(dst), keys)
    arch, got = hg.read_keys(str(dst))
    assert arch == "gemma4" and not hg.compare_keys(got, keys)
    r = gguf.GGUFReader(str(dst))
    names = [f.name for f in r.fields.values()]
    assert r.fields["general.name"].contents() == "tiny"
    assert r.fields["gemma4.context_length"].contents() == 8192
    assert names.count("gemma4.decision.type") == 1
    assert np.array_equal(np.asarray(r.tensors[0].data), weights)
    # the source's other chat templates stay, and the list of names gains systemone once
    assert r.fields["tokenizer.chat_template"].contents() == "{{ default }}"
    assert r.fields["tokenizer.chat_template.tool_use"].contents() == "{{ tools }}"
    assert r.fields["tokenizer.chat_templates"].contents() == ["tool_use", "systemone"]
    again = tmp_path / "again.gguf"
    hg.write_gguf(str(dst), str(again), keys)
    assert gguf.GGUFReader(str(again)).fields["tokenizer.chat_templates"].contents() == ["tool_use", "systemone"]
    # a source without named templates gets the list ["systemone"]
    bare, out = tmp_path / "bare.gguf", tmp_path / "bare_out.gguf"
    tiny_gguf(gguf, bare, templates=False)
    hg.write_gguf(str(bare), str(out), keys)
    assert gguf.GGUFReader(str(out)).fields["tokenizer.chat_templates"].contents() == ["systemone"]
    with pytest.raises(ValueError, match="every key"):
        hg.write_gguf(str(src), str(tmp_path / "x.gguf"), {"general.name": "other"})


def test_g5_refuses_writing_in_place(tmp_path, gemma, settings):
    gguf = pytest.importorskip("gguf")
    src = tmp_path / "src.gguf"
    tiny_gguf(gguf, src)
    before = src.read_bytes()
    keys, _ = hg.decision_keys(gemma, settings)
    link = tmp_path / "link.gguf"
    link.symlink_to(src)
    for dst in (src, tmp_path / "." / "src.gguf", link):
        with pytest.raises(ValueError, match="is the input file"):
            hg.write_gguf(str(src), str(dst), keys)
    hard = tmp_path / "hard.gguf"
    hard.hardlink_to(src)
    with pytest.raises(ValueError, match="is the input file"):
        hg.write_gguf(str(src), str(hard), keys)
    assert src.read_bytes() == before


@pytest.mark.parametrize("value", [3, True, ["A", "B"], None])
def test_g5_refuses_values_of_other_types(tmp_path, value):
    gguf = pytest.importorskip("gguf")
    src = tmp_path / "src.gguf"
    tiny_gguf(gguf, src)
    with pytest.raises(TypeError, match="str or a float"):
        hg.write_gguf(str(src), str(tmp_path / "dst.gguf"), {"gemma4.decision.labels": value})
