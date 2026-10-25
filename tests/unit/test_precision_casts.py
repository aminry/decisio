# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Inductor's precision-cast emulation, on for FP8 bases (serve.precision_casts):

P1  which setting wins (flag, then the base's default for FP8, then the environment, then off), where it came from, and
    the variable set for the engine
P2  a variable that says the opposite is refused; a variable alone on a bf16 base is recorded as the environment's;
    other values and other backends are refused
P3  the flag is in the parser
P4  the server knows an FP8 base: the 12B FP8, the 31B (on load, stored, a tuned repository), and not bf16 or Qwen
P5  /health carries the setting and the fingerprint holds it only when on, so a certificate made without it is
    suspended
P6  the start line says where it came from and what torch reads; the setting reaches an inductor config already imported

  uv run pytest -q tests/unit/test_precision_casts.py
"""

import os
import sys
import types

import pytest
from test_hub import args_for
from test_hub_tuned import build, write_repo
from test_mlx_engine import FP8_CONFIGS, _checkpoint_with

from decisio.serve import precision_casts as pc
from decisio.serve import stack, vllm_engine
from decisio.serve.vllm_engine import resolve_base

VAR = pc.VARIABLE


@pytest.mark.parametrize(
    ("flag", "fp8", "want"),
    [
        (None, False, (False, "default")),
        (None, True, (True, "fp8 base")),
        ("auto", True, (True, "fp8 base")),
        ("auto", False, (False, "default")),
        ("on", False, (True, "flag")),
        ("on", True, (True, "flag")),
        ("off", True, (False, "flag")),  # the flag beats the base's default
        ("off", False, (False, "flag")),
    ],
)
def test_p1_the_flag_then_the_fp8_default_then_off(flag, fp8, want):
    env = {}
    r = pc.resolve(flag, fp8, env)
    assert (r["on"], r["source"]) == want
    assert env == ({VAR: "1"} if want[0] else {})  # set when on, never set (or removed) otherwise


def test_p1_the_variable_is_set_in_the_environment_the_engine_inherits():
    env = {"PATH": "/bin"}
    pc.resolve(None, True, env)
    assert env == {"PATH": "/bin", VAR: "1"}


@pytest.mark.parametrize(
    ("flag", "fp8", "env"),
    [("on", False, "0"), (None, True, "0"), ("off", False, "1"), ("off", True, "1")],
)
def test_p2_a_variable_that_says_the_opposite_is_refused(flag, fp8, env):
    with pytest.raises(SystemExit, match=rf"{VAR}={env} in the environment contradicts"):
        pc.resolve(flag, fp8, {VAR: env})


def test_p2_the_refusal_for_the_fp8_default_names_the_way_out():
    with pytest.raises(SystemExit, match="default for an FP8 base.*use --emulate-precision-casts off"):
        pc.resolve(None, True, {VAR: "0"})
    assert pc.resolve("off", True, {VAR: "0"}) == {"on": False, "source": "flag"}


def test_p2_the_variable_alone_is_recorded_and_agreeing_values_pass():
    assert pc.resolve(None, False, {VAR: "1"}) == {"on": True, "source": "environment"}
    assert pc.resolve(None, False, {VAR: "0"}) == {"on": False, "source": "environment"}
    assert pc.resolve(None, True, {VAR: "1"}) == {"on": True, "source": "fp8 base"}
    assert pc.resolve("on", False, {VAR: "1"})["on"] and pc.resolve("off", False, {VAR: "0"})["on"] is False
    assert pc.resolve(None, False, {VAR: ""}) == {"on": False, "source": "default"}  # empty is unset


def test_p2_other_values_and_other_backends_are_refused():
    with pytest.raises(SystemExit, match="torch reads 0 or 1"):
        pc.resolve(None, False, {VAR: "true"})
    with pytest.raises(SystemExit, match="not --backend mlx"):
        pc.resolve("on", False, {}, "mlx")
    assert pc.resolve(None, True, {}, "mlx") == {"on": False, "source": "default"}  # an FP8 base is not on elsewhere
    assert pc.resolve("off", False, {}, "hf") == {"on": False, "source": "default"}
    with pytest.raises(SystemExit, match="on, off or auto"):
        pc.resolve("yes", False, {})


def test_p3_the_flag_is_in_the_parser():
    p = vllm_engine.build_parser()
    assert p.parse_args([]).emulate_precision_casts is None
    for value in ("on", "off", "auto"):
        assert p.parse_args(["--emulate-precision-casts", value]).emulate_precision_casts == value
    with pytest.raises(SystemExit):
        p.parse_args(["--emulate-precision-casts", "maybe"])


@pytest.mark.parametrize(
    ("base", "fp8"),
    [
        ("gemma-4-12b-fp8", True),
        ("gemma-4-31b", True),  # quantized to FP8 on load, or the stored FP8 repository
        ("gemma-4-12b", False),
        ("gemma-4-31b-bf16", False),
        ("qwen3.6-35b-a3b", False),
    ],
)
def test_p4_which_bases_are_fp8(base, fp8):
    args = args_for(base)
    resolve_base(args)
    assert args.fp8_weights is fp8


def test_p4_a_tuned_repository_and_a_stored_checkpoint(tmp_path):
    d = write_repo(tmp_path, build(), name="t")  # a tuned 31B repository
    args = args_for(str(d))
    resolve_base(args)
    assert args.fp8_weights is True
    for name in sorted(FP8_CONFIGS):  # a checkpoint that declares FP8 weights, served as the 31B
        model = _checkpoint_with(tmp_path, name, **FP8_CONFIGS[name])
        args = args_for("gemma-4-31b")
        args.model = model
        resolve_base(args)
        assert args.fp8_weights is True, name


def server(report):
    from test_adapter_bands import make

    return make(temperatures={"choice": 5.0}, precision_casts=report)


def test_p5_the_fingerprint_holds_it_only_when_on():
    _, on = server({"on": True, "source": "fp8 base", "confirmed": True})
    _, off = server({"on": False, "source": "default"})
    _, none = server(None)
    assert on.stack_facts()["serving"]["emulate_precision_casts"] is True
    assert "emulate_precision_casts" not in off.stack_facts()["serving"]
    assert "emulate_precision_casts" not in none.stack_facts()["serving"]
    assert any(d.endswith("emulate_precision_casts") for d in stack.differences(on.stack_facts(), off.stack_facts()))
    assert stack.differences(off.stack_facts(), none.stack_facts()) == []  # off is exactly what a server always was


def test_p5_health_carries_it():
    from fastapi.testclient import TestClient

    from decisio.serve.reference_docs import _DocEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore
    from decisio.serve.vllm_engine import make_app

    engine = _DocEngine()
    rep = {"on": True, "source": "flag", "confirmed": None}
    so = SystemOne(engine, "decisio-test", task_store=TaskStore("fp"), precision_casts=rep)
    got = TestClient(make_app(engine, so)).get("/health").json()["systemone"]["emulate_precision_casts"]
    assert got == rep
    so2 = SystemOne(engine, "decisio-test", task_store=TaskStore("fp"), precision_casts={"on": False, "source": "x"})
    assert "emulate_precision_casts" not in TestClient(make_app(engine, so2)).get("/health").json()["systemone"]


def test_p6_the_start_line():
    assert pc.start_line({"on": False, "source": "default"}) == "PRECISION CASTS emulated off (the default)"
    assert pc.start_line({"on": True, "source": "fp8 base", "confirmed": True}) == (
        "PRECISION CASTS emulated on (the default for an FP8 base); torch reads it as on"
    )
    assert "torch reads it as off" in pc.start_line({"on": True, "source": "flag", "confirmed": False})
    assert pc.start_line({"on": True, "source": "environment", "confirmed": None}).endswith("in the environment)")
    assert pc.engine_state() in (None, True, False)  # never an exception


def test_p6_the_setting_reaches_an_inductor_config_that_was_imported_before(monkeypatch):
    config = types.SimpleNamespace(emulate_precision_casts=False)
    monkeypatch.setitem(sys.modules, "torch._inductor.config", config)
    pc.apply_in_process(False)
    assert config.emulate_precision_casts is False
    pc.apply_in_process(True)
    assert config.emulate_precision_casts is True
    monkeypatch.delitem(sys.modules, "torch._inductor.config")
    pc.apply_in_process(True)  # not imported: the variable is read when it is, nothing to write


def test_p6_a_real_inductor_config_reads_the_variable_at_import():
    """The premise of the module: torch reads the variable once, at import (a fresh interpreter, no engine)."""
    import subprocess

    torch = pytest.importorskip("torch")
    del torch
    code = "import torch._inductor.config as c; print(c.emulate_precision_casts)"
    for value, want in (("1", "True"), ("0", "False")):
        out = subprocess.run(
            [sys.executable, "-c", code], env={**os.environ, VAR: value}, capture_output=True, text=True
        )
        assert out.stdout.strip() == want, out.stderr[-300:]
