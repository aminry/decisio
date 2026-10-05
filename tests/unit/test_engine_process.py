# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""--engine-process: where vLLM's engine runs (decisio.serve.vllm_engine.engine_process_guard).

E1  separate (the default) leaves the environment alone and refuses to start when VLLM_ENABLE_V1_MULTIPROCESSING=0
    is already set (the engine would run in the server's process while the server says otherwise); in sets it to 0
E2  in is refused with a second engine (--image-model, --head-engine, --one-engine) and off vLLM
E3  the server's start-up applies it before any engine is built: the refusals end start-up with the reason

The GPU tier checks the effect: with the engine in-process, a --multi-question warm request repeats exactly
(tests/gpu/test_warm_in_process.py).

  uv run pytest -q tests/unit/test_engine_process.py
"""

import sys

import pytest

from decisio.serve import vllm_engine
from decisio.serve.vllm_engine import ENGINE_PROCESSES, engine_process_guard

VAR = "VLLM_ENABLE_V1_MULTIPROCESSING"


def test_e1_separate_is_the_default_and_in_sets_the_variable():
    assert ENGINE_PROCESSES[0] == "separate"
    env = {}
    assert engine_process_guard("vllm", "separate", env) == "separate" and env == {}
    assert engine_process_guard("vllm", "separate", {VAR: "1"}) == "separate"
    with pytest.raises(SystemExit, match="--engine-process in"):
        engine_process_guard("vllm", "separate", {VAR: "0"})
    env = {}
    assert engine_process_guard("vllm", "in", env) == "in" and env == {VAR: "0"}
    env = {VAR: "1"}
    assert engine_process_guard("vllm", "in", env) == "in" and env == {VAR: "0"}


@pytest.mark.parametrize("flags", [["--image-model"], ["--head-engine"], ["--image-model", "--one-engine"]])
def test_e2_in_is_refused_with_a_second_engine(flags):
    env = {}
    with pytest.raises(SystemExit, match="second one") as e:
        engine_process_guard("vllm", "in", env, flags)
    assert all(f in str(e.value) for f in flags) and env == {}


@pytest.mark.parametrize("backend", ["hf", "mlx"])
def test_e2_in_is_for_vllm_only(backend):
    with pytest.raises(SystemExit, match="--backend vllm"):
        engine_process_guard(backend, "in", {})
    env = {VAR: "0"}
    assert engine_process_guard(backend, "separate", env) is None and env == {VAR: "0"}  # not vLLM: untouched


@pytest.mark.parametrize(
    "argv,reason",
    [
        (["--engine-process", "in", "--image-model", "x"], "second one"),
        (["--engine-process", "in", "--head-engine"], "second one"),
        (["--engine-process", "in", "--backend", "hf"], "--backend vllm"),
    ],
)
def test_e3_start_up_refuses_before_any_engine_is_built(monkeypatch, argv, reason):
    def no_engine(*a, **k):
        raise AssertionError("an engine was built")

    monkeypatch.setattr(vllm_engine, "LettersEngine", no_engine)  # the image engines are imported later still
    monkeypatch.delenv(VAR, raising=False)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")  # start-up sets it; through monkeypatch it is restored afterwards
    monkeypatch.setattr(sys, "argv", ["decisio", "--base", "qwen3.6-35b-a3b", "--model", "stand-in", *argv])
    with pytest.raises(SystemExit, match=reason):
        vllm_engine.main()
