# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A compile with no timed choices, an opt-in (serve.compile_pin):

C1  which setting wins (the flag, then the environment, then off: no base turns it on), where it came from, and the
    variable set for the engine
C2  a variable that says the opposite of the flag is refused; a variable alone is recorded as the environment's;
    other values and other backends are refused
C3  the flag is in the parser and reaches the arguments
C4  the engine JSON: `deterministic: true` and `combo_kernels: false` in the inductor compile config when on (the keys
    vLLM hands to inductor's standalone compile, the route that reaches the code generation), what the engine is built
    with; an explicit value in --engine wins, other keys stay, off changes nothing
C5  /health carries the setting and the fingerprint holds it only when on
C6  the start line; the setting reaches an inductor config imported before
C7  the kernel report counts the generated kernels' metas: a pinned compile has deterministic True and no combo kernel

  uv run pytest -q tests/unit/test_compile_pin.py
"""

import json
import sys
import types

import pytest
from test_adapter_bands import make

from decisio.serve import compile_pin as cp
from decisio.serve import stack, vllm_engine

VAR = cp.VARIABLE


@pytest.mark.parametrize(
    ("flag", "want"),
    [
        (None, (False, "default")),  # no base turns it on, FP8 or not
        ("on", (True, "flag")),
        ("off", (False, "flag")),
    ],
)
def test_c1_the_flag_then_off(flag, want):
    env = {}
    r = cp.resolve(flag, env)
    assert (r["on"], r["source"]) == want
    assert env == ({VAR: "1"} if want[0] else {})


@pytest.mark.parametrize(("flag", "env"), [("on", "0"), ("off", "1")])
def test_c2_a_variable_that_says_the_opposite_is_refused(flag, env):
    with pytest.raises(SystemExit, match=rf"{VAR}={env} in the environment contradicts"):
        cp.resolve(flag, {VAR: env})


def test_c2_the_variable_alone_other_values_and_other_backends():
    assert cp.resolve(None, {VAR: "1"}) == {"on": True, "source": "environment"}
    assert cp.resolve(None, {VAR: "0"}) == {"on": False, "source": "environment"}
    assert cp.resolve("off", {VAR: "0"}) == {"on": False, "source": "flag"}
    assert cp.resolve(None, {VAR: ""}) == {"on": False, "source": "default"}
    with pytest.raises(SystemExit, match="torch reads 1"):
        cp.resolve(None, {VAR: "true"})
    with pytest.raises(SystemExit, match="not --backend mlx"):
        cp.resolve("on", {}, "mlx")
    assert cp.resolve(None, {}, "mlx") == {"on": False, "source": "default"}
    with pytest.raises(SystemExit, match="on or off"):
        cp.resolve("yes", {})
    with pytest.raises(SystemExit, match="on or off"):
        cp.resolve("auto", {})  # there is no auto: nothing turns it on by itself


def test_c3_the_flag_is_in_the_parser():
    p = vllm_engine.build_parser()
    assert p.parse_args([]).deterministic_compile is None
    for value in ("on", "off"):
        assert p.parse_args(["--deterministic-compile", value]).deterministic_compile == value
    for bad in ("maybe", "auto"):
        with pytest.raises(SystemExit):
            p.parse_args(["--deterministic-compile", bad])


def test_c4_the_engine_json():
    served = json.dumps(vllm_engine.SERVED_ENGINE)
    on = json.loads(cp.merge_engine(served, True))
    assert on["compilation_config"] == {
        "max_cudagraph_capture_size": 4096,
        "inductor_compile_config": {"combo_kernels": False, "deterministic": True},
    }
    # what the engine is built with: the keyword arguments of LLM(...) carry the keys
    from types import SimpleNamespace

    kw = vllm_engine.engine_kwargs(SimpleNamespace(engine=cp.merge_engine(served, True), model_class="view"))
    assert kw["compilation_config"]["inductor_compile_config"] == {"combo_kernels": False, "deterministic": True}
    assert vllm_engine.engine_kwargs(SimpleNamespace(engine=served, model_class="view")) == vllm_engine.SERVED_ENGINE
    assert cp.merge_engine(served, False) == served  # off is exactly what it was
    # a caller's own choice wins, and their other keys stay
    own = json.dumps({"compilation_config": {"inductor_compile_config": {"combo_kernels": True, "x": 1}}, "k": 2})
    got = json.loads(cp.merge_engine(own, True))
    assert got["compilation_config"]["inductor_compile_config"] == {
        "combo_kernels": True,
        "x": 1,
        "deterministic": True,
    }
    assert got["k"] == 2
    own = json.dumps({"compilation_config": {"inductor_compile_config": {"deterministic": False}}})
    assert json.loads(cp.merge_engine(own, True))["compilation_config"]["inductor_compile_config"] == {
        "deterministic": False,
        "combo_kernels": False,
    }
    assert json.loads(cp.merge_engine("{}", True)) == {
        "compilation_config": {"inductor_compile_config": {"combo_kernels": False, "deterministic": True}}
    }


def server(report):
    return make(temperatures={"choice": 5.0}, compile_pin=report)


def test_c5_the_fingerprint_holds_it_only_when_on():
    _, on = server({"on": True, "source": "flag", "confirmed": True})
    _, off = server({"on": False, "source": "default"})
    _, none = server(None)
    assert on.stack_facts()["serving"]["deterministic_compile"] is True
    assert "deterministic_compile" not in off.stack_facts()["serving"]
    assert any(d.endswith("deterministic_compile") for d in stack.differences(on.stack_facts(), off.stack_facts()))
    assert stack.differences(off.stack_facts(), none.stack_facts()) == []


def test_c5_health_carries_it():
    from fastapi.testclient import TestClient

    from decisio.serve.reference_docs import _DocEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore
    from decisio.serve.vllm_engine import make_app

    engine = _DocEngine()
    rep = {"on": True, "source": "flag", "confirmed": None}
    so = SystemOne(engine, "decisio-test", task_store=TaskStore("fp"), compile_pin=rep)
    assert TestClient(make_app(engine, so)).get("/health").json()["systemone"]["deterministic_compile"] == rep
    so2 = SystemOne(engine, "decisio-test", task_store=TaskStore("fp"), compile_pin={"on": False, "source": "x"})
    assert "deterministic_compile" not in TestClient(make_app(engine, so2)).get("/health").json()["systemone"]


def test_c6_the_start_line_and_the_imported_config(monkeypatch):
    assert cp.start_line({"on": False, "source": "default"}) == "DETERMINISTIC COMPILE off (the default)"
    assert cp.start_line({"on": True, "source": "flag", "confirmed": True}) == (
        "DETERMINISTIC COMPILE on (--deterministic-compile); torch reads it as on"
    )
    assert cp.engine_state() in (None, True, False)
    config = types.SimpleNamespace(deterministic=False)
    monkeypatch.setitem(sys.modules, "torch._inductor.config", config)
    cp.apply_in_process(False)
    assert config.deterministic is False
    cp.apply_in_process(True)
    assert config.deterministic is True


def test_c7_the_kernel_report_counts_deterministic_flags_and_combo_kernels(tmp_path):
    kernel = "@triton_heuristics.pointwise(inductor_meta={'deterministic': %s})\ndef triton_(): pass\n"
    combo = "@triton_heuristics.reduction(inductor_meta={'deterministic': %s, 'combo_grid_meta': {'num_kernels': 2}})\n"
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "k1.py").write_text(kernel % "True")
    (tmp_path / "a" / "k2.py").write_text(kernel % "True")
    (tmp_path / "a" / "not_a_kernel.py").write_text("deterministic = False\n")  # no triton_heuristics: not counted
    assert cp.kernel_meta_report(tmp_path) == {
        "files": 2,
        "deterministic_true": 2,
        "deterministic_false": 0,
        "combo": 0,
    }
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "k3.py").write_text(combo % "False")
    assert cp.kernel_meta_report(tmp_path / "a", tmp_path / "b") == {
        "files": 3,
        "deterministic_true": 2,
        "deterministic_false": 1,
        "combo": 1,
    }
    assert cp.kernel_meta_report(tmp_path / "missing") == {
        "files": 0,
        "deterministic_true": 0,
        "deterministic_false": 0,
        "combo": 0,
    }
