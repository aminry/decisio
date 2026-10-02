# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The checks of the vLLM packaging that need no GPU and no vLLM (docs/design/vllm-plugin.md, "Tests without a GPU");
vLLM 0.30.0 has no macOS wheels, so the real package is never imported here. The plugin's registration, version guard
and re-entrancy are in test_vllm_plugin.py, the model classes in test_vllm_plugin.py and test_hidden_readout.py, the
patch series in test_patches.py, the head modes in test_head_modes.py.

  V1  packaging: the built wheel, installed alone into a fresh environment, exposes the `vllm.general_plugins` entry
      point; loading and calling it without vLLM does nothing and imports nothing heavy; the `serve` extra pins vLLM
  V2  laziness: importing the plugin and registering on the stand-in imports neither torch nor the model module
  V3  the launcher: refuses DeepGEMM, sets it off when unset, leaves the CPU stand-in alone
  V4  the server's engine arguments: the view involves no plugin; decisio's classes need the installed entry point and
      the supported vLLM, and add the architecture override (and, for the hidden readout, the reserved-column start and
      the larger log-probability budget); the view's builder stays available as the fallback

    uv run pytest -q tests/unit/test_vllm_stub_suite.py
"""

import json
import os
import shutil
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from stub_vllm import stub_vllm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def wheel_env(tmp_path_factory):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    tmp = tmp_path_factory.mktemp("wheel")
    src = tmp / "src"  # build from a clean copy of what ships
    src.mkdir()
    for f in ("pyproject.toml", "LICENSE", "NOTICE"):
        shutil.copy(ROOT / f, src)
    shutil.copytree(ROOT / "src", src / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc"))
    b = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp / "dist"), str(src)], capture_output=True, text=True
    )
    if b.returncode != 0:
        pytest.skip(f"the wheel could not be built here (offline?): {b.stderr[-300:]}")
    wheel = next((tmp / "dist").glob("*.whl"))
    venv = tmp / "venv"
    subprocess.run([uv, "venv", "-q", "--python", "3.12", str(venv)], check=True)
    subprocess.run(
        [uv, "pip", "install", "-q", "--python", str(venv / "bin" / "python"), "--no-deps", str(wheel)], check=True
    )
    return venv / "bin" / "python"


def run(py, code):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, env=env, cwd="/")
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_v1_wheel_carries_every_package(wheel_env):
    import zipfile

    wheel = next(wheel_env.parents[2].glob("dist/*.whl"))
    names = set(zipfile.ZipFile(wheel).namelist())
    for pkg in ("decisio", "decisio/vllm_plugin", "decisio/readout", "decisio/serve", "decisio/bench"):
        assert f"{pkg}/__init__.py" in names, pkg
    assert "decisio/bench/banking77_labels.json" in names
    assert any(n.endswith("licenses/LICENSE") for n in names) and any(n.endswith("licenses/NOTICE") for n in names)


def test_v1_wheel_exposes_the_entry_point(wheel_env):
    out = json.loads(
        run(
            wheel_env,
            """
import json, sys
from importlib.metadata import entry_points
eps = entry_points(group="vllm.general_plugins")
fn = next(e for e in eps if e.name == "decisio").load()
import decisio.vllm_plugin as p
print(json.dumps({"eps": {e.name: e.value for e in eps}, "registered": fn(), "again": fn(),
                  "installed": p.installed_entry_point(), "vllm": p.vllm_version(),
                  "heavy": sorted(m for m in ("torch", "numpy", "vllm", "decisio.serve", "decisio.vllm_plugin.models")
                                  if m in sys.modules)}))
""",
        )
    )
    assert out["eps"] == {"decisio": "decisio.vllm_plugin:register"} and out["installed"] is True
    assert out["registered"] is False and out["again"] is False and out["vllm"] is None  # no vLLM: nothing, twice
    assert out["heavy"] == []


def test_v1_serve_extra_pins_vllm():
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert [d.split(";")[0].strip() for d in proj["optional-dependencies"]["serve"]] == ["vllm==0.30.0"]
    assert not any(d.startswith("vllm") for d in proj["dependencies"])  # vLLM stays optional (Linux, GPU)
    import decisio.vllm_plugin as p

    assert p.SUPPORTED_VLLM == "0.30.0"


def test_v2_registration_is_lazy():
    code = """
import sys, types
vllm = types.ModuleType("vllm"); vllm.__version__ = "0.30.0"
calls = []
models = types.ModuleType("vllm.model_executor.models")
models.ModelRegistry = types.SimpleNamespace(register_model=lambda a, c: calls.append((a, c)))
config = types.ModuleType("vllm.model_executor.models.config")
config.MODELS_CONFIG_MAP = {"Qwen3_5MoeForCausalLM": object, "Gemma4UnifiedForConditionalGeneration": object}
sys.modules.update({"vllm": vllm, "vllm.model_executor": types.ModuleType("vllm.model_executor"),
                    "vllm.model_executor.models": models, "vllm.model_executor.models.config": config})
import decisio.vllm_plugin as p
assert p.register() and len(calls) == len(p.MODELS) and all(isinstance(c, str) for _, c in calls)
print(sorted(m for m in ("torch", "numpy", "decisio.vllm_plugin.models", "decisio.vllm_plugin.gemma", "decisio.serve")
             if m in sys.modules))
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


def test_v3_launcher_deep_gemm_guard():
    from decisio.serve.vllm_engine import deep_gemm_guard

    env = {}
    deep_gemm_guard("vllm", env)
    assert env == {"VLLM_USE_DEEP_GEMM": "0"}  # unset: switched off before vLLM loads
    deep_gemm_guard("vllm", env)  # already off: fine
    for bad in ("1", "true", ""):
        with pytest.raises(SystemExit, match="VLLM_USE_DEEP_GEMM"):
            deep_gemm_guard("vllm", {"VLLM_USE_DEEP_GEMM": bad})
        deep_gemm_guard("vllm", {"VLLM_USE_DEEP_GEMM": bad}, allow=True)  # the operator's explicit override
    env = {"VLLM_USE_DEEP_GEMM": "1"}
    deep_gemm_guard("hf", env)  # the CPU stand-in: untouched
    assert env == {"VLLM_USE_DEEP_GEMM": "1"} and deep_gemm_guard("hf", {}) is None


def args(model_class, engine='{"compilation_config": {"max_cudagraph_capture_size": 4096}}'):
    return types.SimpleNamespace(model_class=model_class, engine=engine)


def test_v4_engine_arguments(monkeypatch):
    from decisio.serve import vllm_engine

    base = {"compilation_config": {"max_cudagraph_capture_size": 4096}}
    assert vllm_engine.engine_kwargs(args("view")) == base  # the fallback: no plugin involved
    monkeypatch.delenv("DECISIO_HIDDEN_READOUT_START", raising=False)
    with stub_vllm("0.30.0"):
        import decisio.vllm_plugin as p

        monkeypatch.setattr(p, "installed_entry_point", lambda: False)
        with pytest.raises(SystemExit, match="entry point"):
            vllm_engine.engine_kwargs(args("text-only"))
        monkeypatch.setattr(p, "installed_entry_point", lambda: True)
        assert vllm_engine.engine_kwargs(args("text-only")) == {
            **base,
            "hf_overrides": {"architectures": [p.TEXT_ONLY]},
        }
        kw = vllm_engine.engine_kwargs(args("hidden-readout"))
        assert kw == {**base, "max_logprobs": 1024, "hf_overrides": {"architectures": [p.HIDDEN_READOUT]}}
        assert os.environ["DECISIO_HIDDEN_READOUT_START"] == "100000"
        monkeypatch.delenv("DECISIO_HIDDEN_READOUT_START")
    with stub_vllm("0.31.0"):
        import decisio.vllm_plugin as p

        monkeypatch.setattr(p, "installed_entry_point", lambda: True)
        with pytest.raises(SystemExit, match="vllm==0.30.0"):
            vllm_engine.engine_kwargs(args("text-only"))


def test_v4_the_view_stays_available():
    import decisio.serve.make_text_only as m
    from decisio.vllm_plugin.weights import is_vision_weight

    assert callable(m.main)
    assert is_vision_weight("model.visual.blocks.0.attn.qkv.weight") and is_vision_weight("visual.merger.mlp.0.bias")
    assert not is_vision_weight("model.language_model.layers.0.mlp.gate.weight")
