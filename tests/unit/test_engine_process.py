# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""--engine-process: where vLLM's engine runs, resolved once (decisio.serve.vllm_engine.engine_process_guard), and
--multi-question's default that follows it (resolve_multi_question).

E1  the default resolves to in for a single-engine server (and sets VLLM_ENABLE_V1_MULTIPROCESSING=0), to separate with
    a second engine (--image-model, --head-engine, --one-engine) or when the environment already sets the variable to
    1; an explicit choice is honoured; an explicit separate is refused when the variable already says 0 (the engine
    would run in the server's process while the server says otherwise)
E2  in is refused with a second engine and off vLLM
E3  the server's start-up applies it before any engine is built: the refusals end start-up with the reason; the
    resolved arrangement and multi-question scoring are printed
E4  --multi-question's default: gemma-4-31b's profile (warm) when the engine runs in the server's process, sequential
    when it runs separate and on the other bases; an explicit choice is honoured

Gates for the defaults: runs/2026-10-05_engine-death-gates. The GPU tier checks the effect: with the engine in-process,
a --multi-question warm request repeats exactly (tests/gpu/test_warm_in_process.py).

  uv run pytest -q tests/unit/test_engine_process.py
"""

import sys

import pytest

from decisio.families import BASES
from decisio.serve import vllm_engine
from decisio.serve.vllm_engine import ENGINE_PROCESSES, engine_process_guard, resolve_multi_question

VAR = "VLLM_ENABLE_V1_MULTIPROCESSING"


def test_e1_the_default_resolves_by_engine_count():
    assert ENGINE_PROCESSES[0] == "in"
    env = {}
    assert engine_process_guard("vllm", None, env) == ("in", "the default for a single-engine server")
    assert env == {VAR: "0"}
    for second in (["--image-model"], ["--head-engine"], ["--image-model", "--one-engine"]):
        env = {}
        arrangement, why = engine_process_guard("vllm", None, env, second)
        assert arrangement == "separate" and all(f in why for f in second) and env == {}
    env = {VAR: "1"}
    assert engine_process_guard("vllm", None, env)[0] == "separate" and env == {VAR: "1"}
    env = {VAR: "0"}
    assert engine_process_guard("vllm", None, env)[0] == "in" and env == {VAR: "0"}


def test_e1_an_explicit_choice_is_honoured():
    env = {}
    assert engine_process_guard("vllm", "separate", env) == ("separate", "--engine-process separate") and env == {}
    assert engine_process_guard("vllm", "separate", {VAR: "1"})[0] == "separate"
    with pytest.raises(SystemExit, match="--engine-process in"):
        engine_process_guard("vllm", "separate", {VAR: "0"})
    env = {VAR: "1"}
    assert engine_process_guard("vllm", "in", env) == ("in", "--engine-process in") and env == {VAR: "0"}


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
    for choice in (None, "separate"):
        env = {VAR: "0"}
        assert engine_process_guard(backend, choice, env)[0] is None and env == {VAR: "0"}  # not vLLM: untouched


class Built(Exception):
    pass


def start_up(monkeypatch, argv):
    """main() up to the first engine it would build; (VLLM_ENABLE_V1_MULTIPROCESSING then, the start-up output)."""
    seen = {}

    def first_engine(*a, **k):
        seen["var"] = vllm_engine.os.environ.get(VAR)
        raise Built

    monkeypatch.setattr(vllm_engine, "LettersEngine", first_engine)  # the image engines are imported later still
    monkeypatch.setattr(vllm_engine, "engine_kwargs", lambda args: {})  # the plugin's arguments need vLLM itself
    monkeypatch.delenv(VAR, raising=False)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")  # start-up sets it; through monkeypatch it is restored afterwards
    monkeypatch.setattr(sys, "argv", ["decisio", "--model", "stand-in", *argv])
    with pytest.raises(Built):
        vllm_engine.main()
    monkeypatch.delenv(VAR, raising=False)
    return seen["var"]


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

    monkeypatch.setattr(vllm_engine, "LettersEngine", no_engine)
    monkeypatch.delenv(VAR, raising=False)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    monkeypatch.setattr(sys, "argv", ["decisio", "--base", "qwen3.6-35b-a3b", "--model", "stand-in", *argv])
    with pytest.raises(SystemExit, match=reason):
        vllm_engine.main()


@pytest.mark.parametrize(
    "argv,var,line",
    [
        (
            ["--base", "qwen3.6-35b-a3b"],
            "0",
            "ENGINE PROCESS in (the default for a single-engine server); multi-question sequential",
        ),
        (
            ["--base", "gemma-4-12b"],
            "0",
            "ENGINE PROCESS in (the default for a single-engine server); multi-question sequential",
        ),
        (
            ["--base", "gemma-4-31b"],
            "0",
            "ENGINE PROCESS in (the default for a single-engine server); multi-question warm",
        ),
        (
            ["--base", "gemma-4-31b", "--engine-process", "separate"],
            None,
            "ENGINE PROCESS separate (--engine-process separate); multi-question sequential",
        ),
        (["--base", "gemma-4-31b", "--multi-question", "sequential"], "0", "multi-question sequential"),
        (
            ["--base", "qwen3.6-35b-a3b", "--head-engine"],
            None,
            "ENGINE PROCESS separate (the default with a second engine (--head-engine)); multi-question sequential",
        ),
    ],
)
def test_e3_start_up_applies_the_resolved_default(monkeypatch, capsys, argv, var, line):
    assert start_up(monkeypatch, argv) == var
    assert line in capsys.readouterr().out


def test_e4_multi_question_default_follows_the_profile_and_the_arrangement():
    g31, g12, qwen = BASES["gemma-4-31b"], BASES["gemma-4-12b"], BASES["qwen3.6-35b-a3b"]
    assert g31.multi_question == "warm" and g12.multi_question == qwen.multi_question == "sequential"
    assert resolve_multi_question(g31, None, "in") == "warm"
    assert resolve_multi_question(g31, None, None) == "warm"  # off vLLM the engine runs in the server's process
    assert resolve_multi_question(g31, None, "separate") == "sequential"  # warm does not repeat exactly there
    assert resolve_multi_question(g31, "batch", "in") == "batch"
    for fam in (g12, qwen, None):
        assert resolve_multi_question(fam, None, "in") == "sequential"
