# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The prompt-format flags (`--system-prompt`, `--prompt-tail`, `--answer-slot`, `--label-variants`) on the served
model's tokenizer, no model:

  P1  the defaults are the served prompt as before, token for token, and the server starts with them
  P2  byte stability: every flag combination builds the same token rows and label ids as when this test was written
      (sha256 of the rows, for a fixed state and four questions)
  P3  the template slot is the position after Qwen3.6's empty, closed think block; the system turn and the Cygnet
      recipe's tail render as written
  P4  label forms: summed reads " A" and "A" (yes / no: four forms); the template slot with one form reads the bare
      code, and its code list skips the codes whose bare form is not one token (CJ, CZ, EJ, GK, JL)
  P5  the state prefix is the same for every question mix in every format
  P6  conformance C3 builds rows in the server's format and asks /v1/answer the question /v1/systemone asks

    uv run pytest -q tests/unit/test_prompt_format.py      (downloads the tokenizer of Qwen/Qwen3.6-35B-A3B once)
"""

import hashlib
import itertools
import json
import os
import sys

import pytest

from decisio.readout.letters import (
    DEFAULT_FORMAT,
    PromptFormat,
    allowed_ids,
    chat_turn,
    chat_wrap,
    fmt_state,
    label_forms,
    label_groups,
    letter_labels,
    letters_prompt,
)
from decisio.serve import vllm_engine as sv

TOKENIZER = os.environ.get("DECISIO_TOKENIZER", "Qwen/Qwen3.6-35B-A3B")
STATE = (
    "Ticket 8842. Customer writes: our API integration started returning 500 errors on every request about 20 "
    "minutes ago. Account tier: enterprise."
)
QUESTIONS = [
    {"kind": "choice", "instructions": "Which team should handle this?", "options": ["billing", "technical", "sales"]},
    {"kind": "noul", "instructions": "Is the customer reporting an outage?"},
    {"kind": "score", "instructions": "How urgent is this?", "options": ["low", "medium", "high", "critical"]},
    {"kind": "choice", "instructions": "Which intent?", "options": [f"intent {i}" for i in range(100)]},
]
FORMATS = [
    PromptFormat(system=s, tail=t, slot=a, variants=v)
    for s, t, a, v in itertools.product(
        ("none", "cygnet"), ("decisio", "cygnet"), ("prefill", "template"), ("single", "summed")
    )
]
# sha256 of json.dumps([P, rows]) per format, written with this test (P2); a change here is a change of served prompts
GOLDEN = {
    "none-decisio-prefill-single": "a4c7266b75766664fbec28fff759d778360cd9fcdb9f0521def4e485f9a883c0",
    "none-decisio-prefill-summed": "29e515796af1df8d4d354846f60ef48c6663dcdb0db36ebd484db4dc3a10a552",
    "none-decisio-template-single": "798d18e548e905b1327869fa82013a9a79f6b619a56a2a1297e701b09bb45a13",
    "none-decisio-template-summed": "a2ac8b0d1a57f78f85ebb1ae9fdf8c83249d86e2285a7e465232a60efa561f33",
    "none-cygnet-prefill-single": "501d57e32f5107129d7b26b0c3e98b191a96ac84f78453c11218ef395a3376b8",
    "none-cygnet-prefill-summed": "aae75c5cd33ca0bb9939e87199036886945a0f6fce92fc93c4250fec9cbf1501",
    "none-cygnet-template-single": "c9c358ae2b3f6f4a5e7f1642949e6d7c0d359e0489140cd1fdbe6f1a32b8f385",
    "none-cygnet-template-summed": "7fbcc614c73ac8816b6696608ebb2decc605ff695296d736ba92dcbb25ac31ea",
    "cygnet-decisio-prefill-single": "896a7f3fff8e2e16c201d9cc096247c83c9300193f76d5e1dafcdb1d8923c9df",
    "cygnet-decisio-prefill-summed": "908aaa03836de756958645b20d07eaea7d3f165f7106ab485a163097a6550cd1",
    "cygnet-decisio-template-single": "618453bfecd13d33970f3f7cb5c713b7ac29017b3af212a7a38b0b598fcc0106",
    "cygnet-decisio-template-summed": "95c0ef181fdae100052e3f3bf360f1eccc0cd0af218a3ec87993660d0f51ee40",
    "cygnet-cygnet-prefill-single": "ba432bacf026db58437101fea2e84b36df7468cadb74321125a807a32680a87b",
    "cygnet-cygnet-prefill-summed": "df6c21627a9ddbabc9c3e3c38b312f859870908edc0a7646f7a16b2b296591a5",
    "cygnet-cygnet-template-single": "a6bb42d2b80d7dabc8a4a9feef41eaba425766088b447835ee810954931c5d92",
    "cygnet-cygnet-template-summed": "794b6b84130141c871ae6ad9d97d63b30cc1abaa98a3b86565887bb86308c2c2",
}
# the default format's digest computed with main's code before these flags (8ccf143), unpadded and padded to 1,056
MAIN_DEFAULT = {
    None: "a4c7266b75766664fbec28fff759d778360cd9fcdb9f0521def4e485f9a883c0",
    1056: "e3c059e51e90fcd335dc6838fcff5f2e6ec2e7bc092d88cd56b929617da1b63f",
}


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


class Tokonly(sv.LettersEngine):
    """The engine's prompt logic with a tokenizer and no model."""

    def __init__(self, tok, fmt=DEFAULT_FORMAT, pad_unit=None):
        self.tok, self.fmt, self.pad_unit, self.pad_token, self.pad_where = tok, fmt, pad_unit, sv.PAD_TOKEN, "front"


def name(fmt):
    return f"{fmt.system}-{fmt.tail}-{fmt.slot}-{fmt.variants}"


def rows_digest(tok, fmt, pad_unit=None):
    rows, P = Tokonly(tok, fmt, pad_unit)._prepare_separate(STATE, QUESTIONS)
    data = [[ids, [list(g) if isinstance(g, tuple) else g for g in lab]] for ids, lab in rows]
    return hashlib.sha256(json.dumps([P, data]).encode()).hexdigest()


def test_p1_defaults_are_todays_prompt(tok):
    assert PromptFormat(system="none", tail="decisio", slot="prefill", variants="single") == DEFAULT_FORMAT
    # the rows main's code (before these flags, 8ccf143) builds, unpadded and front-padded to the block
    for pad_unit, want in MAIN_DEFAULT.items():
        assert rows_digest(tok, DEFAULT_FORMAT, pad_unit) == want
    rows, _ = Tokonly(tok)._prepare_separate(STATE, QUESTIONS)
    for q, (ids, lab) in zip(QUESTIONS, rows):
        opts = ["yes", "no"] if q["kind"] == "noul" else q["options"]
        prompt, cands = letters_prompt(tok, q["kind"], fmt_state(STATE), q["instructions"], opts)
        assert ids == tok.encode(chat_wrap(tok, prompt, "chat"), add_special_tokens=False)
        assert lab == [tok.encode(c, add_special_tokens=False)[0] for c in cands]  # one id per label, as before


def test_p1_server_starts_with_the_defaults(monkeypatch, tmp_path):
    """main() on a stand-in engine: no prompt flag given, the engine gets the default format and the task
    fingerprint carries no prompt_format entry (so tasks registered before these flags still apply)."""
    import uvicorn

    import decisio.serve.hf_letters as hf

    made = {}

    class Engine:
        adapters, pad_unit = {}, 64

        def __init__(self, *a, **kw):
            made["engine"] = self

        def facts(self):
            return {}

    monkeypatch.setattr(hf, "HFLettersEngine", Engine)
    monkeypatch.setattr(sv, "make_app", lambda engine, so: made.update(so=so) or object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    base = ["decisio", "--backend", "hf", "--model", "/models/moe-fp8", "--model-class", "view"]
    monkeypatch.setattr(sys, "argv", base)
    sv.main()
    assert made["engine"].fmt == DEFAULT_FORMAT and "prompt_format" not in made["so"].task_store.fingerprint
    monkeypatch.setattr(sys, "argv", [*base, "--system-prompt", "cygnet", "--answer-slot", "template"])
    sv.main()
    assert made["engine"].fmt == PromptFormat(system="cygnet", slot="template")
    assert '"prompt_format"' in made["so"].task_store.fingerprint


@pytest.mark.parametrize("fmt", FORMATS, ids=name)
def test_p2_byte_stable(tok, fmt):
    got = rows_digest(tok, fmt)
    assert GOLDEN, "GOLDEN is empty: fill it from rows_digest"
    assert got == GOLDEN[name(fmt)], f"{name(fmt)}: the served prompt changed ({got})"


def test_p3_slots_system_and_tail(tok):
    template = chat_turn(tok, "Q?", PromptFormat(slot="template"))
    assert template.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") and "system" not in template
    assert chat_turn(tok, "Q?", PromptFormat(slot="prefill", variants="summed")) == template + "Answer:"
    from decisio.readout.cygnet import SYSTEM, TAIL

    rows, _ = Tokonly(tok, PromptFormat(system="cygnet", tail="cygnet", slot="template"))._prepare_separate(
        STATE, QUESTIONS[:1]
    )
    want = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{STATE}\n\nWhich team should handle this?\n\n"
        f"Options:\nA. billing\nB. technical\nC. sales\n\n{TAIL}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert tok.decode(rows[0][0]) == want


def test_p4_label_forms(tok):
    names = lambda ids: [tok.decode([t]) for t in ids]  # noqa: E731
    summed = PromptFormat(variants="summed")
    assert names(label_forms(tok, " A", summed)) == [" A", "A"]  # no byte-fallback token in a byte-level BPE
    assert names(label_forms(tok, " yes", summed)) == [" yes", "yes", " Yes", "Yes"]
    assert names(label_forms(tok, " A", PromptFormat(slot="template"))) == ["A"]
    bare, spaced = letter_labels(tok, 255, "bare"), letter_labels(tok, 255)
    assert bare[:85] == spaced[:85] and spaced[85] == "CJ" and "CJ" not in bare
    assert not {"CJ", "CZ", "EJ", "GK", "JL"} & set(bare)
    for fmt in FORMATS:
        cands = [" " + c for c in letter_labels(tok, 255, fmt.label_form())]
        flat = allowed_ids(label_groups(tok, cands, fmt))
        assert len(flat) == len(set(flat)) and len(flat) <= 1024  # vLLM's allowed-id limit per request


@pytest.mark.parametrize("fmt", FORMATS, ids=name)
def test_p5_state_prefix_shared(tok, fmt):
    eng = Tokonly(tok, fmt)
    rows, P = eng._prepare_separate(STATE, QUESTIONS)
    _, P2 = eng._prepare_separate(STATE, QUESTIONS[::-1][:2])
    assert P == P2 and all(ids[:P] == rows[0][0][:P] for ids, _ in rows)
    assert tok.decode(rows[0][0][:P]).endswith(STATE + "\n\n")
    for (ids, _), q in zip(rows, QUESTIONS):  # every row is its whole prompt, tokenized at once
        text = sv.question_text(tok, q, fmt)[0]
        assert ids == tok.encode(chat_turn(tok, f"{STATE}\n\n{text}", fmt), add_special_tokens=False)


def test_p6_conformance_rows_in_the_servers_format(tok, monkeypatch):
    from decisio.serve import systemone_conformance as sc

    items = [
        {
            "task": "x",
            "i": 0,
            "label": 0,
            "answer": {"state": STATE, "questions": [{"kind": "noul", "instructions": "Is this an outage?"}]},
            "systemone": {"state": STATE, "questions": {"q": {"type": "noul", "instructions": "Is this an outage?"}}},
        },
        {
            "task": "x",
            "i": 1,
            "label": 0,
            "answer": {
                "state": STATE,
                "questions": [{"kind": "choice", "instructions": "Which team?", "options": ["billing", "technical"]}],
            },
            "systemone": {
                "state": STATE,
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {"billing": None, "technical": None},
                    }
                },
            },
        },
    ]
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: tok)
    for fmt in (DEFAULT_FORMAT, PromptFormat(system="cygnet", tail="cygnet", slot="template", variants="summed")):
        for noul in ("words", "letters", "letters-keys"):
            ok, detail = sc.gate_c3(items, TOKENIZER, 1056, {}, fmt, noul)
            assert ok, (fmt, noul, detail)
    body, yes_at = sc.answer_counterpart(items[0], "letters")
    assert body["questions"][0]["kind"] == "choice" and body["questions"][0]["options"] == ["No", "Yes"] and yes_at == 1
