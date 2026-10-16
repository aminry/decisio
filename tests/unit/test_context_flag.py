# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A longer context as an opt-in (`--max-model-len`, `--kv-cache-dtype`; decisio.serve.vllm_engine.resolve_context).

L1  without --max-model-len: the Gemma bases on the vLLM server ask for the largest length that fits (an AutoContext
    up to their ceiling); the Qwen base, the image engine and the other backends stay at 32,768; vLLM's own KV dtype
L2  a longer context is accepted up to the base's ceiling (the 12B 65,536, the 31B 98,304) and refused beyond it, on
    the Qwen base at all, off the vLLM server and with the image engine, each with its reason
L3  `--kv-cache-dtype fp8` and `--max-model-len` reach the text engine, and a refusal ends start-up with its reason
L4  the engines (the text engine and `--head-engine`'s second one) put them in vLLM's arguments

  uv run pytest -q tests/unit/test_context_flag.py
"""

import sys
import types

import pytest

from decisio.families import BASES
from decisio.serve import vllm_engine
from decisio.serve.context import AutoContext
from decisio.serve.vllm_engine import resolve_context


def test_l1_the_default_is_the_largest_that_fits_up_to_the_ceiling():
    for fam in BASES.values():
        want = AutoContext(fam.context_ceiling) if fam.context_ceiling > 32768 else 32768
        assert resolve_context(fam, None, None) == (want, None)
        assert resolve_context(fam, None, "auto") == (want, None)
        # off the vLLM server, and with the image engine, 32,768 as before
        assert resolve_context(fam, None, None, backend="mlx") == (32768, None)
        assert resolve_context(fam, None, None, backend="hf") == (32768, None)
        assert resolve_context(fam, None, None, image_model="x") == (32768, None)
    assert {k: f.context_ceiling for k, f in BASES.items()} == {
        "qwen3.6-35b-a3b": 32768,
        "gemma-4-12b": 65536,
        "gemma-4-31b": 98304,
    }


def test_l2_up_to_the_ceiling_and_refused_beyond_with_the_reason():
    g12, g31, qwen = BASES["gemma-4-12b"], BASES["gemma-4-31b"], BASES["qwen3.6-35b-a3b"]
    assert resolve_context(g12, 65536, None) == (65536, None) and resolve_context(g12, 65536, "fp8") == (65536, "fp8")
    assert resolve_context(g31, 98304, "fp8") == (98304, "fp8") and resolve_context(g31, 65536, None) == (65536, None)
    with pytest.raises(ValueError, match=r"gemma-4-12b base is documented to 65,536"):
        resolve_context(g12, 65537, None)
    with pytest.raises(ValueError, match=r"gemma-4-31b base is documented to 98,304"):
        resolve_context(g31, 98305, "fp8")
    with pytest.raises(ValueError, match=r"qwen3.6-35b-a3b base is documented to 32,768"):
        resolve_context(qwen, 40000, None)
    with pytest.raises(ValueError, match="for the vLLM server, not --backend mlx"):
        resolve_context(g12, 65536, None, backend="mlx")
    with pytest.raises(ValueError, match="does not combine with --image-model"):
        resolve_context(g12, 65536, None, image_model="x")
    with pytest.raises(ValueError, match="at least 1,024"):
        resolve_context(g12, 100, None)
    with pytest.raises(ValueError, match="--kv-cache-dtype fp8 is for the vLLM server"):
        resolve_context(g12, None, "fp8", backend="hf")
    # 32,768 and below are never refused, whatever the base and the backend (they are the measured length)
    assert resolve_context(qwen, 32768, None, backend="hf", image_model="x") == (32768, None)


class Built(Exception):
    pass


def run_main(monkeypatch, argv):
    """main() until the text engine would be built: the keyword arguments it was given."""
    seen = {}

    def first_engine(*a, **k):
        seen.update(k)
        raise Built

    monkeypatch.setattr(vllm_engine, "LettersEngine", first_engine)
    monkeypatch.setattr(vllm_engine, "engine_kwargs", lambda args: {})
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    monkeypatch.setattr(sys, "argv", ["decisio", "--model", "stand-in", *argv])
    with pytest.raises(Built):
        vllm_engine.main()
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    return seen


def test_l3_the_flags_reach_the_text_engine(monkeypatch):
    k = run_main(monkeypatch, ["--base", "gemma-4-12b"])
    assert k["max_model_len"] == AutoContext(65536) and k["kv_cache_dtype"] is None
    k = run_main(monkeypatch, ["--base", "qwen3.6-35b-a3b"])
    assert k["max_model_len"] == 32768 and "ceiling" in k["context_source"]
    k = run_main(monkeypatch, ["--base", "gemma-4-31b", "--engine-process", "separate"])
    assert k["max_model_len"] == 32768 and "one engine in the server's process" in k["context_source"]
    k = run_main(monkeypatch, ["--base", "gemma-4-12b", "--max-model-len", "65536"])
    assert k["max_model_len"] == 65536 and k["kv_cache_dtype"] is None
    k = run_main(
        monkeypatch,
        [
            "--base",
            "gemma-4-31b",
            "--max-model-len",
            "98304",
            "--kv-cache-dtype",
            "fp8",
            "--gpu-memory-utilization",
            "0.95",
        ],
    )
    assert (k["max_model_len"], k["kv_cache_dtype"], k["gpu_memory_utilization"]) == (98304, "fp8", 0.95)
    assert k["context_source"] == "set by --max-model-len"
    k = run_main(monkeypatch, ["--base", "gemma-4-31b", "--kv-cache-dtype", "auto"])
    assert k["kv_cache_dtype"] is None


@pytest.mark.parametrize(
    "argv,reason",
    [
        (["--base", "gemma-4-31b", "--max-model-len", "131072"], "documented to 98,304"),
        (["--base", "qwen3.6-35b-a3b", "--max-model-len", "65536"], "documented to 32,768"),
        (["--base", "gemma-4-12b", "--max-model-len", "65536", "--image-model", "x"], "image engine serves 32,768"),
    ],
)
def test_l3_a_refusal_ends_start_up_before_any_engine(monkeypatch, capsys, argv, reason):
    def no_engine(*a, **k):
        raise AssertionError("an engine was built")

    monkeypatch.setattr(vllm_engine, "LettersEngine", no_engine)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    monkeypatch.setattr(sys, "argv", ["decisio", "--model", "stand-in", *argv])
    with pytest.raises(SystemExit):
        vllm_engine.main()
    assert reason in capsys.readouterr().err


def test_l4_the_engine_gives_vllm_the_length_and_the_dtype(monkeypatch):
    captured = {}

    class Stop(Exception):
        pass

    def fake_llm(**kw):
        captured.update(kw)
        raise Stop

    mod = types.ModuleType("vllm")
    mod.LLM = fake_llm
    monkeypatch.setitem(sys.modules, "vllm", mod)
    fam = BASES["gemma-4-31b"]
    with pytest.raises(Stop):
        vllm_engine.LettersEngine("stand-in", family=fam, max_model_len=65536, kv_cache_dtype="fp8")
    assert captured["max_model_len"] == 65536 and captured["kv_cache_dtype"] == "fp8"
    captured.clear()
    with pytest.raises(Stop):
        vllm_engine.LettersEngine("stand-in", family=fam)  # a plain length, as the library's default
    assert captured["max_model_len"] == 32768 and "kv_cache_dtype" not in captured


def test_l4_the_second_engine_of_head_engine_takes_them_too(monkeypatch):
    from decisio.serve.hidden_engine import HiddenEngine

    captured = {}

    class Stop(Exception):
        pass

    def fake_llm(**kw):
        captured.update(kw)
        raise Stop

    mod = types.ModuleType("vllm")
    mod.LLM = fake_llm
    cfg = types.ModuleType("vllm.config")
    cfg.PoolerConfig = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setitem(sys.modules, "vllm.config", cfg)
    with pytest.raises(Stop):
        HiddenEngine("stand-in", max_model_len=65536, kv_cache_dtype="fp8")
    assert captured["max_model_len"] == 65536 and captured["kv_cache_dtype"] == "fp8"
