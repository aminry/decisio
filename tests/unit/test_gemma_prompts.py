# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The prompt formats and label forms on Gemma 4 12B's tokenizer (no model), and the default format on Qwen's:

F1  the default format is today's prompt, token for token, on Qwen's tokenizer (rows and labels)
F2  Gemma's three forms of a letter (spaced, bare, byte-fallback) and four of yes / no; the cygnet rule keeps the bare
    and byte forms (what xgrammar allows for Cygnet's choice "A"); no two labels share a token, up to 255 options
F3  the template slot is the position after Gemma 4's empty, closed thought channel; the prefill slot adds "Answer:"
F4  the system prompt and the tail render as Cygnet's prompt (decisio.readout.cygnet; layout checked literally)
F5  the state prefix is the same for every question mix in every format, with no padding (the family's default)
F6  per-label probabilities from several forms: the summed probability of each label's forms
F7  the head's soft cap follows vLLM's LogitsProcessor (bf16, step by step)

  uv run pytest -q tests/unit/test_gemma_prompts.py      (downloads both tokenizers once)
"""

import os

import numpy as np
import pytest

from decisio.readout.letters import (
    DEFAULT_FORMAT,
    PromptFormat,
    allowed_ids,
    chat_turn,
    label_forms,
    label_groups,
    label_log_softmax,
    letter_labels,
)
from decisio.serve import vllm_engine as sv

GEMMA = os.environ.get("DECISIO_GEMMA_TOKENIZER", "google/gemma-4-12B-it")
GEMMA_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"  # Cygnet's pinned revision
QWEN = os.environ.get("DECISIO_TOKENIZER", "Qwen/Qwen3.6-35B-A3B")

STATE = "Order 5521 arrived with a cracked screen. The customer wants it fixed before Friday."
QUESTIONS = [
    {"kind": "choice", "instructions": "Which team handles this?", "options": ["billing", "repairs", "sales"]},
    {"kind": "noul", "instructions": "Is the device damaged?"},
    {"kind": "score", "instructions": "How urgent is it?", "options": ["low", "medium", "high"]},
]
CYGNET = PromptFormat(system="cygnet", tail="cygnet", slot="template", variants="summed")


@pytest.fixture(scope="module")
def gemma():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(GEMMA, revision=GEMMA_REVISION)


@pytest.fixture(scope="module")
def qwen():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN)


class Tokonly(sv.LettersEngine):
    """The engine's prompt logic with a tokenizer and no model."""

    def __init__(self, tok, fmt=DEFAULT_FORMAT, pad_unit=None):
        self.tok, self.fmt, self.pad_unit, self.pad_token, self.pad_where = tok, fmt, pad_unit, sv.PAD_TOKEN, "front"


def test_f1_default_format_is_todays_prompt(qwen):
    from decisio.readout.letters import chat_wrap, fmt_state, letters_prompt

    rows, _ = Tokonly(qwen)._prepare_separate(STATE, QUESTIONS)
    for q, (ids, lab) in zip(QUESTIONS, rows):
        opts = ["yes", "no"] if q["kind"] == "noul" else q["options"]
        prompt, cands = letters_prompt(qwen, q["kind"], fmt_state(STATE), q["instructions"], opts)
        assert ids == qwen.encode(chat_wrap(qwen, prompt, "chat"), add_special_tokens=False)
        assert lab == [qwen.encode(c, add_special_tokens=False)[0] for c in cands]  # a list of ints, as before


def test_f2_label_forms(gemma):
    names = lambda ids: [gemma.convert_ids_to_tokens(t) for t in ids]  # noqa: E731
    summed, cygnet = PromptFormat(variants="summed"), PromptFormat(variants="cygnet", slot="template")
    assert names(label_forms(gemma, " A", summed)) == ["▁A", "A", "<0x41>"]
    assert names(label_forms(gemma, " A", cygnet)) == ["A", "<0x41>"]
    assert [gemma.decode([t]) for t in label_forms(gemma, " yes", summed)] == [" yes", "yes", " Yes", "Yes"]
    assert names(label_forms(gemma, " AB", summed)) == ["▁AB", "AB"]  # two letters: no byte form
    # single: the form the slot reads
    assert names(label_forms(gemma, " A", PromptFormat())) == ["▁A"]
    assert names(label_forms(gemma, " A", PromptFormat(slot="template"))) == ["A"]
    for fmt in (summed, cygnet, PromptFormat(slot="template"), PromptFormat()):
        groups = label_groups(gemma, [" " + c for c in letter_labels(gemma, 255)], fmt)
        flat = allowed_ids(groups)
        assert len(flat) == len(set(flat)) and len(flat) <= 1024  # vLLM's allowed-id limit per request


def test_f3_answer_slots(gemma):
    template = chat_turn(gemma, "Q?", PromptFormat(slot="template"))
    assert template.endswith("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    assert chat_turn(gemma, "Q?", PromptFormat(slot="prefill", variants="summed")) == template + "Answer:"
    assert "<|turn>system" not in template
    assert template.startswith("<bos>") and template.count("<bos>") == 1


def test_f4_cygnet_layout(gemma):
    from decisio.readout.cygnet import SYSTEM, TAIL

    rows, _ = Tokonly(gemma, CYGNET)._prepare_separate(STATE, QUESTIONS[:1])
    text = gemma.decode(rows[0][0])
    want = (
        f"<bos><|turn>system\n{SYSTEM}<turn|>\n<|turn>user\n{STATE}\n\nWhich team handles this?\n\nOptions:\n"
        f"A. billing\nB. repairs\nC. sales\n\n{TAIL}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )
    assert text == want
    # Cygnet's shim, read from its source (cygnet_shim.build_prompt): state, "", instructions, "", "Options:", the
    # lettered lines, "", the instruction; joined by newlines
    lines = [STATE, "", "Which team handles this?", "", "Options:", "A. billing", "B. repairs", "C. sales", "", TAIL]
    assert "\n".join(lines) in text


@pytest.mark.parametrize(
    "fmt",
    [CYGNET, PromptFormat(variants="summed"), PromptFormat(system="cygnet", slot="template", variants="summed")],
)
def test_f5_state_prefix_shared(gemma, fmt):
    eng = Tokonly(gemma, fmt)
    rows, P = eng._prepare_separate(STATE, QUESTIONS)
    _, P2 = eng._prepare_separate(STATE, QUESTIONS[::-1][:2])
    assert P == P2 and all(ids[:P] == rows[0][0][:P] for ids, _ in rows)
    assert gemma.decode(rows[0][0][:P]).endswith(STATE + "\n\n")
    for (ids, _), q in zip(rows, QUESTIONS):  # every row is its whole prompt, tokenized at once
        text = sv.question_text(gemma, q, fmt)[0]
        assert ids == gemma.encode(chat_turn(gemma, f"{STATE}\n\n{text}", fmt), add_special_tokens=False)


def test_f6_summed_label_probabilities():
    lab = ((10, 11, 12), (20, 21))
    z = np.array([2.0, 0.5, -1.0, 1.0, 1.5])  # logits of allowed_ids(lab), in that order
    p = np.exp(z) / np.exp(z).sum()
    want = np.log([p[:3].sum(), p[3:].sum()])
    assert np.allclose(label_log_softmax(z, lab), want, atol=1e-12)
    # one token per label: the plain log-softmax
    flat = [10, 20]
    assert np.allclose(label_log_softmax(z[[0, 3]], flat), z[[0, 3]] - np.log(np.exp(z[[0, 3]]).sum()))


def test_f7_softcap_follows_vllm():
    import torch

    from decisio.serve.hidden_engine import softcap_bf16

    z = (torch.randn(1000, dtype=torch.float64) * 40).float().to(torch.bfloat16)
    ref = z / 30.0  # vLLM's LogitsProcessor.forward, in the logits' dtype
    ref = torch.tanh(ref)
    ref = ref * 30.0
    assert torch.equal(softcap_bf16(z, 30.0), ref) and softcap_bf16(z, None) is z


# ---- F8: the family and the server's flags ---------------------------------------------------------------------------


def _args(model, **kw):
    import types

    base = dict(
        model=model,
        backend="vllm",
        system_prompt=None,
        prompt_tail=None,
        answer_slot=None,
        label_variants=None,
        state_rendering=None,
        prompt_preset=None,
        noul_rendering=None,
        pad_to=None,
        served_name=None,
        temperature=None,
    )
    return types.SimpleNamespace(**{**base, **kw})


def _checkpoint(tmp_path, model_type):
    import json

    d = tmp_path / model_type
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(d)


def test_f8_family_detection(tmp_path, monkeypatch):
    from decisio.families import GEMMA4, QWEN, family_of

    monkeypatch.delenv("DECISIO_FAMILY", raising=False)
    assert family_of(_checkpoint(tmp_path, "gemma4_unified")) is GEMMA4
    assert family_of(_checkpoint(tmp_path, "qwen3_5_moe")) is QWEN
    assert family_of(_checkpoint(tmp_path, "qwen3")) is QWEN  # the CPU stand-in's small models: served as before


def test_f8_flags_resolve(tmp_path, monkeypatch):
    from decisio.families import GEMMA4, QWEN

    monkeypatch.delenv("DECISIO_FAMILY", raising=False)
    qwen, gemma = _checkpoint(tmp_path, "qwen3_5_moe"), _checkpoint(tmp_path, "gemma4_unified")
    a = _args(qwen)
    fam, fmt = sv.resolve_family_and_format(a)
    assert fam is QWEN and fmt == DEFAULT_FORMAT and fmt.is_default()
    assert (a.pad_to, a.temperature, a.noul_rendering, a.served_name) == ("block", 1.307, "words", QWEN.served_name)
    with pytest.raises(ValueError, match="no fitted temperature"):
        sv.resolve_family_and_format(_args(gemma))
    a = _args(gemma, temperature=1.0)
    fam, fmt = sv.resolve_family_and_format(a)
    assert fam is GEMMA4 and fmt == PromptFormat(system="cygnet", tail="cygnet", slot="template", variants="summed")
    assert (a.pad_to, a.noul_rendering, a.served_name) == ("none", "letters", GEMMA4.served_name)
    # one factor at a time (the ablations)
    assert sv.resolve_family_and_format(_args(gemma, temperature=1.0, answer_slot="prefill"))[1].slot == "prefill"
    a = _args(gemma, prompt_preset="cygnet-identity")
    _, fmt = sv.resolve_family_and_format(a)
    assert fmt == PromptFormat(system="cygnet", tail="cygnet", slot="template", variants="cygnet", state="cygnet")
    assert (a.temperature, a.noul_rendering) == (3.4, "letters")
    with pytest.raises(ValueError, match="fixes"):
        sv.resolve_family_and_format(_args(gemma, prompt_preset="cygnet-identity", label_variants="summed"))
    # the Qwen path with a Phase 2 arm's flags
    _, fmt = sv.resolve_family_and_format(_args(qwen, system_prompt="cygnet", answer_slot="template"))
    assert fmt == PromptFormat(system="cygnet", slot="template")
