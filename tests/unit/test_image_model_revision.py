# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The --image-model engine loads at a pinned revision, and a bad image model stops the start before any engine.

CPU only: the config read and the engines are stubbed, nothing is downloaded or loaded.

  R1  no revision given: the image model that is a base's checkpoint takes that base's pinned revision
  R2  --image-model-revision wins over the pin
  R3  an image model of no known base is read at the hub's head (no revision)
  R4  the config is read at the revision (the lookup an offline cache answers), and an unreadable one is refused
  R5  an unreadable image model config stops main before any engine is built
  R6  the vLLM image engine passes the revision to the engine, the tokenizer's reader and the processor
"""

import json
import sys
import types

import pytest

from decisio import families
from decisio.families import QWEN
from decisio.serve import image_engine
from decisio.serve import vllm_engine as sv

CFG = json.dumps({"model_type": "qwen3_5_moe"})


def args(model=QWEN.model, revision=None, backend="vllm"):
    return types.SimpleNamespace(image_model=model, image_model_revision=revision, backend=backend)


@pytest.fixture
def reads(monkeypatch, tmp_path):
    """Every read_config call as (model, revision); the hub answers with a Qwen config at any revision."""
    seen = []

    def read_config(model, revision=None):
        seen.append((model, revision))
        return json.loads(CFG)

    monkeypatch.setattr(families, "read_config", read_config)
    monkeypatch.delenv("DECISIO_BASE", raising=False)
    return seen


def test_r1_the_bases_pinned_revision_is_taken_when_none_is_given(reads):
    assert sv.resolve_image_model(args()) == QWEN.revision and QWEN.revision
    assert reads == [(QWEN.model, QWEN.revision)]


def test_r2_an_explicit_revision_wins(reads):
    assert sv.resolve_image_model(args(revision="abc123")) == "abc123"
    assert reads == [(QWEN.model, "abc123")]


def test_r3_an_unknown_checkpoint_is_read_at_the_hub_head(reads):
    assert sv.resolve_image_model(args(model="someone/qwen-vl")) is None
    assert reads == [("someone/qwen-vl", None)]


def test_r4_the_hub_lookup_gets_the_revision_and_an_unreadable_config_is_refused(monkeypatch):
    calls = []

    def fail(model, filename, revision=None):
        calls.append((model, filename, revision))
        raise OSError("not in the offline cache")

    monkeypatch.delenv("DECISIO_BASE", raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)
    with pytest.raises(ValueError, match="config.json could not be read"):
        sv.resolve_image_model(args())
    assert calls == [(QWEN.model, "config.json", QWEN.revision)]


def test_r5_a_bad_image_model_stops_the_start_before_any_engine(monkeypatch, capsys):
    order = []

    def resolve(a):
        a.repository, a.repository_defaults, a.flavor_report = None, None, None
        a.profile_temperatures, a.temperature = None, 1.0
        return QWEN, sv.DEFAULT_FORMAT

    def unreadable(model, revision=None, lenient=False):
        order.append(("family_of", model, revision))
        raise ValueError(f"--model {model}: its config.json could not be read, which no supported base serves")

    def engine(*a, **k):
        order.append("engine")
        raise AssertionError("an engine was built")

    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    monkeypatch.delenv("DECISIO_BASE", raising=False)
    monkeypatch.setattr(sv, "resolve_base", resolve)
    monkeypatch.setattr(families, "family_of", unreadable)
    monkeypatch.setattr(sv, "LettersEngine", engine)
    monkeypatch.setattr(image_engine, "ImageLettersEngine", engine)
    monkeypatch.setattr("sys.argv", ["decisio", "--model", "x", "--image-model", QWEN.model])
    with pytest.raises(SystemExit) as e:
        sv.main()
    assert e.value.code == 2 and "config.json could not be read" in capsys.readouterr().err
    assert order == [("family_of", QWEN.model, QWEN.revision)]


def test_r6_the_vllm_image_engine_hands_the_revision_on(monkeypatch):
    got = {}

    def base_init(self, model, engine_kw=None, **kw):
        got["engine"] = (model, kw.get("revision"))

    class Processor:
        @staticmethod
        def from_pretrained(model, revision=None):
            got["processor"] = (model, revision)
            return types.SimpleNamespace(image_processor=types.SimpleNamespace(merge_size=2))

    monkeypatch.setattr(sv.LettersEngine, "__init__", base_init)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoProcessor=Processor))
    monkeypatch.setattr(image_engine.ImageLettersEngine, "answer", lambda *a, **k: None)
    eng = types.SimpleNamespace(tok=types.SimpleNamespace(convert_tokens_to_ids=lambda t: 1))
    image_engine.ImageRequests._image_setup(eng, "m", "rev1")
    assert got["processor"] == ("m", "rev1")
    obj = image_engine.ImageLettersEngine.__new__(image_engine.ImageLettersEngine)
    obj.tok = eng.tok
    image_engine.ImageLettersEngine.__init__(obj, "m", revision="rev1")
    assert got["engine"] == ("m", "rev1") and got["processor"] == ("m", "rev1")
