# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Prompt layout (`--layout header`) and the base's `--temperatures-file`.

The oracle is an independent renderer of the plain and header layouts, copied below, so the served flag
and the A/B read the same tokens.

L1  the format: the default's facts and rows are unchanged; the flags enter the facts (hence the fingerprint) only when
    on; bad values are refused
L2  the header layout equals the A/B's A1 token for token, lists every question of a request once, keeps the shared
    prefix (header and state) for every question, and the rows with the plain layout are what they were
L4  usage.input_tokens counts what is read: the header
L5  a task fitted, or a certificate made, under one layout does not apply under another
L6  the flag is in the parser and reaches the format; the base's temperatures file and its precedence

  uv run pytest -q tests/unit/test_layout_flags.py
"""

import json
import re
import threading

import pytest

from decisio.names import task_fingerprint
from decisio.readout import letters as L
from decisio.serve import vllm_engine
from decisio.serve.vllm_engine import PAD_TOKEN, LettersEngine, question_text

BASE = dict(tail="spaced", slot="template", variants="summed", system_prompt=True)


class StubTok:
    """Words with their leading space, whitespace runs and punctuation-free pieces are tokens; a chat template that
    wraps roles. Concatenation-consistent at whitespace, as the engine's row builder needs."""

    def __init__(self):
        self.vocab, self.rev = {}, {}

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = 1000 + len(self.vocab)
            self.rev[self.vocab[piece]] = piece
        return self.vocab[piece]

    def encode(self, text, add_special_tokens=False):
        return [self._id(p) for p in re.findall(r"\s?\S+|\s+", str(text))]

    def decode(self, ids):
        return "".join(self.rev[i] for i in ids)

    def convert_tokens_to_ids(self, name):
        return -1

    unk_token_id = None

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, enable_thinking=None):
        out = "".join(f"<{m['role']}>\n{m['content']}<end>\n" for m in messages)
        return out + ("<model>\n" if add_generation_prompt else "")


def engine(fmt, tok=None, pad_policy="none"):
    eng = LettersEngine.__new__(LettersEngine)
    eng.tok = tok or StubTok()
    eng.fmt, eng.mode, eng.pad_token, eng.pad_where = fmt, "separate", PAD_TOKEN, "front"
    eng.adapters, eng.block_size, eng.match_unit, eng.pad_unit = {}, 16, 16, None
    eng.pad_policy = pad_policy
    eng._lock = threading.Lock()
    return eng


def q(text, options=("billing", "shipping", "access"), kind="choice"):
    return {"kind": kind, "instructions": text, "options": list(options) if kind == "choice" else None}


# the A/B's renderer (card/code/layouts.py), on engine questions
HEAD = "Questions that will be asked about the context:\n"


def oracle(tok, arm, state, questions, fmt):
    body = L.fmt_state(state)
    bodies = [L.question_body(tok, e["kind"], e["instructions"], e.get("options"), fmt)[0] for e in questions]
    header = HEAD + "".join(f"- {e['instructions']}\n" for e in questions) + "\n" if arm == "A1" else ""
    turns = [header + body + "\n\n" + t for t in bodies]
    ids = [tok.encode(L.chat_turn(tok, t, fmt), add_special_tokens=False) for t in turns]

    return ids


STATE = {"message": "my invoice was charged twice and nobody answered my first mail"}
ONE = [q("Which queue should this go to?")]
TWO = [q("Which queue should this go to?"), q("Is the customer angry?", kind="noul")]


# ---- L1 -------------------------------------------------------------------------------------------------------------


def test_l1_the_default_facts_are_unchanged_and_the_flags_enter_only_when_on():
    base = L.PromptFormat(**BASE)
    assert base.facts() == {"tail": "spaced", "slot": "template", "variants": "summed", "system_prompt": True}
    assert L.PromptFormat().facts() == {"tail": "compact", "slot": "prefill", "variants": "single"}
    h = L.PromptFormat(**BASE, layout="header")
    assert h.facts() == {**base.facts(), "layout": "header"}
    assert L.PromptFormat(**BASE, layout="plain").facts() == base.facts()
    assert base.core() == h.core() == base and not h.is_default()
    with pytest.raises(ValueError):
        L.PromptFormat(**BASE, layout="footer")


def test_l1_the_default_layout_rows_are_the_plain_prompt():
    tok = StubTok()
    fmt = L.PromptFormat(**BASE)
    rows, n = engine(fmt, tok)._prepare_separate(STATE, TWO)
    for (ids, _), e in zip(rows, TWO):
        text = question_text(tok, e, fmt)[0]
        assert ids == tok.encode(L.chat_turn(tok, f"{L.fmt_state(STATE)}\n\n{text}", fmt), add_special_tokens=False)
    ids = oracle(tok, "A0", STATE, TWO, fmt)
    assert [r[0] for r in rows] == ids and n > 0
    # the default format of the Qwen base (compact, prefill) is the evaluation prompt, with or without the flags' fields
    plain, flagged = L.PromptFormat(), L.PromptFormat(layout="header")
    assert L.chat_turn(tok, "x", flagged) == L.chat_turn(tok, "x", plain)  # the row's own text is not changed by it


# ---- L2 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("questions", [ONE, TWO])
def test_l2_the_header_layout_is_the_ab_arm_a1_token_for_token(questions):
    tok, fmt = StubTok(), L.PromptFormat(**BASE, layout="header")
    rows, n = engine(fmt, tok)._prepare_separate(STATE, questions)
    ids = oracle(tok, "A1", STATE, questions, L.PromptFormat(**BASE))
    assert [r[0] for r in rows] == ids


def test_l2_the_header_lists_every_question_once_and_every_row_keeps_the_prefix():
    tok, fmt = StubTok(), L.PromptFormat(**BASE, layout="header")
    eng = engine(fmt, tok)
    rows, n = eng._prepare_separate(STATE, TWO)
    text = tok.decode(rows[0][0])
    assert text.count("Questions that will be asked about the context:") == 1
    assert "- Which queue should this go to?\n- Is the customer angry?\n\n" in text
    prefix = rows[0][0][:n]
    assert all(r[0][:n] == prefix for r in rows) and tok.decode(prefix).endswith(L.fmt_state(STATE) + "\n\n")
    # the shared prefix is the header and the state, and it differs with the set of questions asked
    other, n2 = eng._prepare_separate(STATE, [q("Is the customer angry?", kind="noul")])
    assert other[0][0][:n2] != prefix
    assert layout_header_text(TWO) == HEAD + "- Which queue should this go to?\n- Is the customer angry?\n\n"


def layout_header_text(questions):
    return L.layout_header([e["instructions"] for e in questions])


def test_l2_padding_and_the_boundary_follow_the_header():
    tok, fmt = StubTok(), L.PromptFormat(**BASE, layout="header")
    eng = engine(fmt, tok, pad_policy="always")
    eng.pad_unit = 16
    rows, n = eng._prepare_separate(STATE, TWO)
    assert n % 16 == 0 and all(len(r[0]) > n for r in rows)  # the padded prefix still ends on a block boundary


# ---- L4 -------------------------------------------------------------------------------------------------------------


class UsageEngine:
    def __init__(self, fmt):
        self.tok, self.fmt = StubTok(), fmt


def usage(fmt, questions=TWO):
    from decisio.serve.systemone import SystemOne

    so = SystemOne.__new__(SystemOne)
    so.engine = UsageEngine(fmt)
    return so.usage_tokens(STATE, questions), so.engine.tok


def test_l4_usage_counts_the_state_and_the_questions_as_before_under_the_default_layout():
    n, tok = usage(L.PromptFormat(**BASE))
    fmt = L.PromptFormat(**BASE)
    want = len(tok.encode(L.fmt_state(STATE))) + sum(len(tok.encode(question_text(tok, e, fmt)[0])) for e in TWO)
    assert n == want


def test_l4_usage_counts_the_header():
    plain, tok = usage(L.PromptFormat(**BASE))
    header_n = len(tok.encode(layout_header_text(TWO)))
    header, _ = usage(L.PromptFormat(**BASE, layout="header"))
    assert header == plain + header_n and header_n > 0
    one, _ = usage(L.PromptFormat(**BASE, layout="header"), ONE)
    assert one == usage(L.PromptFormat(**BASE), ONE)[0] + len(tok.encode(layout_header_text(ONE)))


# ---- L5 -------------------------------------------------------------------------------------------------------------


def fingerprint(fmt):
    return task_fingerprint(
        served_name="decisio-test",
        model="m",
        pad_to="block",
        pad_where="front",
        hide_index_keys=False,
        desnake_labels=False,
        pad_policy="always",
        prompt_format=None if fmt.is_default() else fmt.facts(),
        base="gemma-4-12b",
    )


def test_l5_tasks_and_certificates_are_per_layout():
    plain = fingerprint(L.PromptFormat(**BASE))
    header = fingerprint(L.PromptFormat(**BASE, layout="header"))
    assert plain != header
    assert plain == fingerprint(L.PromptFormat(**BASE, layout="plain"))  # the default, unchanged
    from decisio.names import same_fingerprint

    assert not same_fingerprint(plain, header)


def test_l5_a_certificate_made_under_another_layout_is_suspended():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from test_certified_answers import FakeEngine, examples, post_task, wire

    from decisio.serve.systemone import SystemOne, add_routes
    from decisio.serve.tasks import TaskStore

    def server(fmt):
        eng = FakeEngine()
        eng.fmt = fmt
        so = SystemOne(eng, "decisio-test", task_store=TaskStore(fingerprint(fmt)))
        app = FastAPI()
        add_routes(app, so)
        return TestClient(app), so

    a, so_a = server(L.PromptFormat(**BASE))
    assert post_task(a, {"id": "c", "examples": examples(400, 0.96), "certify": {}}).status_code == 200
    exported = a.get("/v1/tasks?full=1").json()["tasks"]
    b, so_b = server(L.PromptFormat(**BASE, layout="header"))
    assert b.post("/v1/tasks/import", json={"tasks": exported}).status_code == 422  # fitted under another layout
    so_b.task_store.load(exported)  # as a file load does: kept, not applied
    assert so_b.task_store.lookup(next(iter(so_b.task_store.by_key))) is None
    ans = b.post("/v1/systemone", json=wire("0:0.99:x")).json()["answers"]["q1"]
    assert ans.get("certified") is not True
    from decisio.serve import stack as stack_mod

    # the serving stack a certificate is made under names the layout, so it is suspended where the layout differs
    assert any(d.endswith("layout") for d in stack_mod.differences(so_a.stack_facts(), so_b.stack_facts()))


# ---- L6 -------------------------------------------------------------------------------------------------------------


def test_l6_the_flag_reaches_the_format():
    from test_hub import args_for

    parser = vllm_engine.build_parser()
    a = parser.parse_args(["--layout", "header", "--temperatures-file", "t.json"])
    assert (a.layout, a.temperatures_file) == ("header", "t.json")
    d = parser.parse_args([])
    assert (d.layout, d.temperatures_file) == (None, None)
    args = args_for("gemma-4-12b", layout="header")
    fam, fmt = vllm_engine.resolve_base(args)
    assert fmt.layout == "header" and fmt.tail == fam.prompt_tail
    plain_args = args_for("gemma-4-12b")
    _, plain = vllm_engine.resolve_base(plain_args)
    assert plain.facts() == {"tail": "spaced", "slot": "template", "variants": "summed", "system_prompt": True}


def temps_file(tmp_path, **doc):
    p = tmp_path / "t.json"
    p.write_text(
        json.dumps(
            {
                "schema": "decisio.adapter/1",
                "temperatures": {"choice": 4.5, "noul": 6.25, "score": 3.5},
                "choice_bands": {"2-10": 4.0, "11-26": 5.5},
                **doc,
            }
        )
    )
    return str(p)


def test_l6_the_temperatures_file_sets_the_base_and_a_flag_beats_it(tmp_path):
    from test_hub import args_for

    f = temps_file(tmp_path)
    args = args_for("gemma-4-12b", temperatures_file=f)
    vllm_engine.resolve_base(args)
    assert (args.temperature_choice, args.temperature_noul, args.temperature_score) == (4.5, 6.25, 3.5)
    assert args.choice_bands == {"2-10": 4.0, "11-26": 5.5}
    facts = args.temperatures_file_facts
    assert facts["file"] == f and len(facts["sha256"]) == 64 and facts["applied"]["noul"] == 6.25
    flagged = args_for("gemma-4-12b", temperatures_file=f, temperature_noul=2.0, temperature_choice=3.0)
    vllm_engine.resolve_base(flagged)
    assert flagged.temperature_noul == 2.0 and flagged.temperature_choice == 3.0 and flagged.temperature_score == 3.5
    assert not flagged.choice_bands  # --temperature-choice is one number for every choice question
    assert flagged.temperatures_file_facts["applied"]["choice_bands_dropped_by"] == "--temperature-choice"
    none = args_for("gemma-4-12b")
    vllm_engine.resolve_base(none)
    assert none.temperatures_file_facts is None


def test_l6_a_bad_temperatures_file_stops_the_start(tmp_path):
    from test_hub import args_for

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "decisio.adapter/1", "temperatures": {"choice": -1}}))
    with pytest.raises(SystemExit, match="temperatures-file"):
        vllm_engine.resolve_base(args_for("gemma-4-12b", temperatures_file=str(bad)))
    with pytest.raises(SystemExit, match="temperatures-file"):
        vllm_engine.resolve_base(args_for("gemma-4-12b", temperatures_file=str(tmp_path / "missing.json")))


def test_l6_the_file_wins_over_a_tuned_repositorys_defaults(tmp_path):
    from test_hub import args_for
    from test_hub_tuned import build, write_repo

    d = write_repo(tmp_path, build(), name="t")  # a tuned 31B repository with its own temperatures and bands
    args = args_for(str(d), temperatures_file=temps_file(tmp_path))
    vllm_engine.resolve_base(args)
    assert args.temperature_choice == 4.5 and args.choice_bands == {"2-10": 4.0, "11-26": 5.5}
    plain = args_for(str(d))
    vllm_engine.resolve_base(plain)
    assert plain.temperature_choice != 4.5


# ---- L7: the real Gemma 4 tokenizer, when it is cached -------------------------------------------------------------


def gemma_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("google/gemma-4-12B-it", local_files_only=True)
    except Exception as e:  # not in the local cache
        pytest.skip(f"the Gemma 4 tokenizer is not cached: {e}")


def test_l7_the_real_tokenizer_reads_the_same_tokens_as_the_ab_arms():
    tok = gemma_tokenizer()
    plain = L.PromptFormat(**BASE)
    for arm, kw in (("A0", {}), ("A1", {"layout": "header"})):
        fmt = L.PromptFormat(**BASE, **kw)
        for questions in (ONE, TWO):
            rows, n = engine(fmt, tok)._prepare_separate(STATE, questions)
            assert [r[0] for r in rows] == oracle(tok, arm, STATE, questions, plain), (arm, len(questions))
