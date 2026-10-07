# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The Gemma 4 12B base (`--base gemma-4-12b`, decisio.families) on its tokenizer and without a model:

B1  label forms: a capital letter has three single tokens (spaced, bare, byte-fallback), yes / no four; summed reads
    them all, single the slot's own; no two labels share a token up to 255 options
B2  the template slot is the position after Gemma 4's empty, closed thought channel; the system turn and the spaced
    layout render as written
B3  the state prefix is the same for every question mix, with no padding
B4  the bases resolve: the Qwen base keeps the served defaults; the 31B brings its pinned FP8 repository and settings;
    a checkpoint of the other base is refused; MLX takes either base from an MLX conversion (--model)
B5  the hidden-state readout's reserved ids hold no label form, for each base on its own tokenizer
B6  the head's numbers: vLLM's bf16 soft cap; under the Gemma base the head's label log-probabilities are the
    engine's own readout of the row, not a recomputation
B8  every base pins a checkpoint revision (a full commit hash), applied whenever the checkpoint is the base's own
    (--base alone, --model by hand, the container's explicit --model, the hf stand-in); --revision overrides it; any
    other checkpoint gets none

  uv run pytest -q tests/unit/test_gemma_base.py      (downloads both tokenizers once)
"""

import os
import re
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from decisio import hub
from decisio.families import BASES, FAMILIES, GEMMA4, GEMMA4_31B, QWEN, family_of
from decisio.readout.letters import (
    PromptFormat,
    allowed_ids,
    chat_turn,
    label_forms,
    label_groups,
    letter_labels,
)
from decisio.serve import vllm_engine as sv

GEMMA_FORMAT = PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True)
STATE = "Order 5521 arrived with a cracked screen. The customer wants it fixed before Friday."
QUESTIONS = [
    {"kind": "choice", "instructions": "Which team handles this?", "options": ["billing", "repairs", "sales"]},
    {"kind": "noul", "instructions": "Is the device damaged?"},
    {"kind": "score", "instructions": "How urgent is it?", "options": ["low", "medium", "high"]},
]


@pytest.fixture(scope="module")
def gemma():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(GEMMA4.model, revision=GEMMA4.revision)


@pytest.fixture(scope="module")
def qwen():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.environ.get("DECISIO_TOKENIZER", "Qwen/Qwen3.6-35B-A3B"))


class Tokonly(sv.LettersEngine):
    def __init__(self, tok, fmt):
        self.tok, self.fmt, self.pad_unit, self.pad_token, self.pad_where = tok, fmt, None, sv.PAD_TOKEN, "front"


def test_b1_label_forms(gemma):
    names = lambda ids: [gemma.convert_ids_to_tokens(t) for t in ids]  # noqa: E731
    summed = PromptFormat(variants="summed")
    assert names(label_forms(gemma, " A", summed)) == ["▁A", "A", "<0x41>"]
    assert [gemma.decode([t]) for t in label_forms(gemma, " yes", summed)] == [" yes", "yes", " Yes", "Yes"]
    assert names(label_forms(gemma, " AB", summed)) == ["▁AB", "AB"]  # two letters: no byte form
    assert names(label_forms(gemma, " A", PromptFormat(slot="template"))) == ["A"]
    for fmt in (GEMMA_FORMAT, PromptFormat(slot="template"), PromptFormat()):
        flat = allowed_ids(label_groups(gemma, [" " + c for c in letter_labels(gemma, 255, fmt.label_form())], fmt))
        assert len(flat) == len(set(flat)) and len(flat) <= 1024


def test_b2_slot_system_and_layout(gemma):
    template = chat_turn(gemma, "Q?", PromptFormat(slot="template"))
    assert template.endswith("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    assert template.startswith("<bos>") and template.count("<bos>") == 1 and "<|turn>system" not in template
    from decisio.readout.spaced import ANSWER_LINE
    from decisio.readout.system_prompt import SYSTEM_PROMPT

    rows, _ = Tokonly(gemma, GEMMA_FORMAT)._prepare_separate(STATE, QUESTIONS[:1])
    want = (
        f"<bos><|turn>system\n{SYSTEM_PROMPT}<turn|>\n<|turn>user\n{STATE}\n\nWhich team handles this?\n\nOptions:\n"
        f"A. billing\nB. repairs\nC. sales\n\n{ANSWER_LINE}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )
    assert gemma.decode(rows[0][0]) == want
    assert GEMMA_FORMAT.facts()["system_prompt"] is True and "system_prompt" not in PromptFormat().facts()


def test_b3_state_prefix_shared(gemma):
    eng = Tokonly(gemma, GEMMA_FORMAT)
    rows, P = eng._prepare_separate(STATE, QUESTIONS)
    _, P2 = eng._prepare_separate(STATE, QUESTIONS[::-1][:2])
    assert P == P2 and all(ids[:P] == rows[0][0][:P] for ids, _ in rows)
    assert gemma.decode(rows[0][0][:P]).endswith(STATE + "\n\n")
    for (ids, _), q in zip(rows, QUESTIONS):
        text = sv.question_text(gemma, q, GEMMA_FORMAT)[0]
        assert ids == gemma.encode(chat_turn(gemma, f"{STATE}\n\n{text}", GEMMA_FORMAT), add_special_tokens=False)


def _args(**kw):
    base = dict(
        base=None,
        model=None,
        revision=None,
        system_prompt=None,
        backend="vllm",
        prompt_tail=None,
        answer_slot=None,
        label_variants=None,
        noul_rendering=None,
        pad_to=None,
        served_name=None,
        temperature=None,
        temperature_choice=None,
    )
    return types.SimpleNamespace(**{**base, **kw})


def _checkpoint(tmp_path, model_type):
    import json

    d = tmp_path / model_type
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(d)


def test_b4_bases_resolve(tmp_path, monkeypatch):
    from decisio.serve.temperature import SERVED_CHOICE_TEMPERATURE, SERVED_TEMPERATURE

    monkeypatch.delenv("DECISIO_BASE", raising=False)
    assert (QWEN.temperature, QWEN.choice_temperature) == (SERVED_TEMPERATURE, SERVED_CHOICE_TEMPERATURE)
    qwen_dir, gemma_dir = _checkpoint(tmp_path, "qwen3_5_moe"), _checkpoint(tmp_path, "gemma4_unified")
    assert family_of(qwen_dir) is QWEN and family_of(gemma_dir) is GEMMA4
    assert family_of(_checkpoint(tmp_path, "qwen3")) is QWEN  # the CPU stand-in's small models
    # the Qwen base: the served defaults, unchanged
    a = _args(model=qwen_dir)
    fam, fmt = sv.resolve_base(a)
    assert fam is QWEN and fmt == PromptFormat(tail="spaced") and not fmt.system_prompt
    assert (a.pad_to, a.noul_rendering, a.temperature, a.temperature_choice) == ("block", "letters-keys", 1.506, 1.37)
    assert a.served_name == QWEN.served_name
    # the Gemma base by --base alone: its checkpoint at the pinned revision, and its settings
    a = _args(base="gemma-4-12b")
    fam, fmt = sv.resolve_base(a)
    assert fam is GEMMA4 and (a.model, a.revision) == (GEMMA4.model, GEMMA4.revision) and fmt == GEMMA_FORMAT
    assert (a.pad_to, a.noul_rendering, a.temperature, a.temperature_choice) == ("none", "letters", 3.592, None)
    # an explicit setting wins over the base's, the system turn included
    a = _args(base="gemma-4-12b", answer_slot="prefill", temperature=1.0)
    assert sv.resolve_base(a)[1].slot == "prefill" and a.temperature == 1.0
    assert not sv.resolve_base(_args(base="gemma-4-12b", system_prompt=False))[1].system_prompt
    assert sv.resolve_base(_args(model=qwen_dir, system_prompt=True))[1].system_prompt
    with pytest.raises(ValueError, match="is a qwen3.6-35b-a3b checkpoint"):
        sv.resolve_base(_args(base="gemma-4-12b", model=qwen_dir))
    with pytest.raises(ValueError, match="--backend mlx needs --model"):
        sv.resolve_base(_args(base="gemma-4-12b", backend="mlx"))
    a = _args(model=gemma_dir, backend="mlx")  # an MLX conversion names the base's model type: its settings follow
    fam, fmt = sv.resolve_base(a)
    assert fam.key == "gemma-4-12b" and fmt.system_prompt and (a.temperature, a.noul_rendering) == (3.592, "letters")
    # neither flag: the vLLM backend serves the default base (0.10.0: gemma-4-31b); the stand-in must name a checkpoint
    _fake_31b_repository(tmp_path, monkeypatch)  # no unit test reads the Hub
    assert sv.resolve_base(_args())[0].key == "gemma-4-31b"
    with pytest.raises(ValueError, match="--model or --base"):
        sv.resolve_base(_args(backend="hf"))
    assert sorted(BASES) == ["gemma-4-12b", "gemma-4-31b", "qwen3.6-35b-a3b"]


def _gemma4_checkpoint(tmp_path, name, moe):
    import json

    d = tmp_path / name
    d.mkdir(exist_ok=True)
    cfg = {"model_type": "gemma4", "text_config": {"model_type": "gemma4_text", "enable_moe_block": moe}}
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


def _fake_31b_repository(tmp_path, monkeypatch):
    import json

    d = tmp_path / "repo"
    d.mkdir(exist_ok=True)
    (d / hub.FILE).write_text(json.dumps(hub.export_decision_config(GEMMA4_31B, "0.10.0")))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda name, filename, revision=None: str(d / filename))


def test_b4_the_31b_base(tmp_path, monkeypatch):
    """The 31B: a dense gemma4 checkpoint is detected as it (a MoE one is not), --base brings its pinned FP8
    repository, the 12B's prompt and its own temperatures (on vLLM; MLX from a conversion, test_mlx_engine)."""
    monkeypatch.delenv("DECISIO_BASE", raising=False)
    _fake_31b_repository(tmp_path, monkeypatch)
    assert family_of(_gemma4_checkpoint(tmp_path, "dense", False)) is GEMMA4_31B
    assert family_of(_gemma4_checkpoint(tmp_path, "moe", True)) is QWEN  # not a base: the old default
    a = _args(base="gemma-4-31b")
    fam, fmt = sv.resolve_base(a)
    repo = hub.REPOSITORIES["gemma-4-31b"]
    assert fam.key == GEMMA4_31B.key and fmt == GEMMA_FORMAT
    assert (a.model, a.revision) == (repo, hub.PINNED_REVISIONS[repo]) and a.repository["name"] == repo
    assert (a.pad_to, a.noul_rendering) == ("none", "letters")
    assert (a.temperature, a.temperature_choice) == (fam.temperature, fam.choice_temperature)
    assert fam.quantization is None and fam.classes["hidden-readout"] == "DecisioGemma4HiddenReadout"


@pytest.mark.parametrize("base,hidden", [("gemma-4-12b", 3840), ("qwen3.6-35b-a3b", 2048), ("gemma-4-31b", 5376)])
def test_b5_reserved_range_clear_of_labels(base, hidden, gemma, qwen):
    """No label form of any prompt format is among a base's reserved ids (the 12B and the 31B share one tokenizer)."""
    from decisio.readout.letters import MAX_LABELS
    from decisio.vllm_plugin.hidden import check_reserved

    fam = BASES[base]
    tok = qwen if fam is QWEN else gemma
    labels = set()
    for variants in ("single", "summed"):
        for slot in ("prefill", "template"):
            fmt = PromptFormat(variants=variants, slot=slot)
            cands = [" yes", " no"] + [" " + c for c in letter_labels(tok, MAX_LABELS, fmt.label_form())]
            labels |= set(allowed_ids(label_groups(tok, cands, fmt)))
    ids = check_reserved(hidden, len(tok), sorted(labels), fam.hidden_start)
    assert ids[0] == fam.hidden_start and len(ids) == hidden + 1
    assert not set(tok.convert_ids_to_tokens(ids)) & set(tok.all_special_tokens)


def test_b6_softcap_follows_vllm():
    import torch

    from decisio.serve.hidden_engine import softcap_bf16

    z = (torch.randn(1000, dtype=torch.float64) * 40).float().to(torch.bfloat16)
    ref = torch.tanh(z / 30.0) * 30.0  # vLLM's LogitsProcessor.forward, in the logits' dtype
    assert torch.equal(softcap_bf16(z, 30.0), ref) and softcap_bf16(z, None) is z


def test_b6_head_reads_the_engines_readout():
    """Under the Gemma base, SingleEngineHidden.readout returns the engine's own label probabilities for the row
    (score_prompts, the plain path), and the hidden state from the reserved columns; no recomputation."""
    from decisio.serve.hidden_engine import SingleEngineHidden

    rows = [([1, 2, 3], ((10, 11), (20, 21))), ([1, 2, 4], ((10, 11), (20, 21)))]
    engine_p = [np.array([0.7, 0.3]), np.array([0.2, 0.8])]
    calls = []

    def score_prompts(rs):
        calls.append(("label", rs[0][0]))
        return [engine_p[[r for r, _ in rows].index(rs[0][0])]], {}

    eng = types.SimpleNamespace(score_prompts=score_prompts, _prepare_separate=lambda s, q: (rows, 2))
    head = SingleEngineHidden.__new__(SingleEngineHidden)
    head.engine, head._lock, head.label_source = eng, threading.Lock(), "engine"
    head.hidden_rows = lambda token_lists: calls.append(("hidden", token_lists[0])) or [np.ones(4)]
    head.label_logits = lambda h, lab: pytest.fail("the Gemma head must not recompute label logits")
    out = head.readout("state", ["q1", "q2"])
    assert [np.exp(lp).round(12).tolist() for lp, _ in out] == [p.tolist() for p in engine_p]
    # each row: its label readout, then its hidden-state chunks, one row after the other
    assert calls == [("label", [1, 2, 3]), ("hidden", [1, 2, 3]), ("label", [1, 2, 4]), ("hidden", [1, 2, 4])]


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.key)
def test_b8_every_family_pins_a_revision(fam, tmp_path, monkeypatch):
    """A run record names the bytes it was measured on: no base serves the repository's head."""
    assert isinstance(fam.revision, str) and re.fullmatch(r"[0-9a-f]{40}", fam.revision), fam.key
    # --base alone serves the checkpoint at the pin; an explicit --revision wins
    if fam is GEMMA4_31B:
        _fake_31b_repository(tmp_path, monkeypatch)
    a = _args(base=fam.key)
    sv.resolve_base(a)
    if fam is GEMMA4_31B:
        repo = hub.REPOSITORIES[fam.key]
        assert (a.model, a.revision) == (repo, hub.PINNED_REVISIONS[repo])
    else:
        assert (a.model, a.revision) == (fam.model, fam.revision)
    a = _args(base=fam.key, revision="refs/pr/1")
    sv.resolve_base(a)
    assert (a.model, a.revision) == (fam.model, "refs/pr/1")


def _config_reader(monkeypatch):
    """read_config without the network: the model type each family's checkpoint declares; records what it was asked."""
    calls = []
    types_ = {f.model: f.model_types[0] for f in FAMILIES}

    def read(model, revision=None):
        calls.append((model, revision))
        mt = "qwen3" if Path(model).exists() else types_.get(model, "qwen3")  # a local one: unknown
        return {"model_type": mt, "text_config": {"enable_moe_block": False}}

    monkeypatch.setattr("decisio.families.read_config", read)
    return calls


@pytest.mark.parametrize("backend", ["vllm", "hf"])
@pytest.mark.parametrize("base", [None, "own"])
@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.key)
def test_b8_the_pin_follows_the_checkpoint(fam, base, backend, monkeypatch):
    """--model naming the base's own checkpoint (a user by hand, docker/entrypoint.sh, the hf stand-in) is served at the
    pin, with or without --base; the config is read at the pin too."""
    calls = _config_reader(monkeypatch)
    a = _args(base=fam.key if base else None, model=fam.model, backend=backend)
    got, _ = sv.resolve_base(a)
    assert got is fam and (a.model, a.revision) == (fam.model, fam.revision)
    assert all(c == (fam.model, fam.revision) for c in calls)
    # an explicit --revision wins, and is what the config is read at
    calls.clear()
    a = _args(base=fam.key if base else None, model=fam.model, revision="refs/pr/1", backend=backend)
    sv.resolve_base(a)
    assert a.revision == "refs/pr/1" and all(c == (fam.model, "refs/pr/1") for c in calls)


@pytest.mark.parametrize("base", [None, "qwen3.6-35b-a3b", "gemma-4-12b"])
def test_b8_another_checkpoint_gets_no_pin(base, tmp_path, monkeypatch):
    calls = _config_reader(monkeypatch)
    # a different hub repository, even one of the same family or one a base would otherwise be served from
    for model in ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3.6-35B-A3B", "google/gemma-4-12B-it-other"):
        a = _args(base=base, model=model)
        sv.resolve_base(a)
        assert a.revision is None, model
    # a revision given for it is kept
    a = _args(base=base, model="Qwen/Qwen3-0.6B-Base", revision="abc123")
    sv.resolve_base(a)
    assert a.revision == "abc123"
    calls.clear()
    # a local directory, including one named like a base's repository (a relative path that exists is not the hub's)
    local = tmp_path / "Qwen" / "Qwen3.6-35B-A3B-FP8"
    local.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    for model in (str(local), QWEN.model):
        a = _args(base=base, model=model)
        sv.resolve_base(a)
        assert a.revision is None, model
    assert calls  # the configs were read, at no revision
    assert all(rev is None for _, rev in calls)


def test_b8_the_pins_are_the_recorded_revisions():
    assert QWEN.revision.startswith("95a723d0") and GEMMA4.revision.startswith("707f0a3b")


def test_b7_health_reports_the_profile():
    """/health quotes what a run record needs: the base, the temperature each question type is served at, and the
    prompt (layout, system turn, yes/no and option rendering, multi-question scoring, padding)."""
    from fastapi.testclient import TestClient

    from decisio.serve.systemone import SystemOne
    from decisio.serve.vllm_engine import make_app

    for fam, temps, fmt, noul in (
        (
            GEMMA4,
            {"choice": None},
            PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True),
            "letters",
        ),
        (QWEN, {"choice": 1.37}, PromptFormat(tail="spaced"), "letters-keys"),
    ):
        engine = types.SimpleNamespace(
            adapters={},
            family=fam,
            fmt=fmt,
            model_name=fam.model,
            revision=fam.revision,
            multi_question="sequential",
            pad_policy="always",
            pad_unit=None if fam is GEMMA4 else 1056,
            facts=lambda: {},
        )
        T = fam.temperature
        so = SystemOne(engine, fam.served_name, noul_rendering=noul, temperature=T, temperatures=temps)
        h = TestClient(make_app(engine, so)).get("/health").json()
        p = h["profile"]
        assert h["base"] == p["base"] == fam.key and (p["checkpoint"], p["revision"]) == (fam.model, fam.revision)
        want_choice = T if temps["choice"] is None else temps["choice"]
        assert p["temperatures"] == {"choice": want_choice, "noul": T, "score": T}
        assert p["prompt"]["noul_rendering"] == noul and p["prompt"]["multi_question"] == "sequential"
        assert p["prompt"]["system_prompt"] is (fam is GEMMA4) and p["prompt"]["tail"] == "spaced"
