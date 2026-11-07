# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""vLLM's batch-invariant kernels from a flag or a tuned repository's defaults (serve.batch_invariant):

B1  which setting wins (flag, then the repository, then off), where it came from, and the variable set for vLLM
B2  a variable that says the opposite is refused; a variable alone is recorded as the environment's; other values
    and other backends are refused
B3  the flag is in the parser
B4  the repository: `defaults.batch_invariant` is true or absent, `tuned-config --batch-invariant` writes it, the digest
    covers it, and a repository without it is what it was
B5  /health carries the setting and the fingerprint holds it only when on, so a certificate made without it is suspended
B6  the start line says where it came from and what vLLM reads
B7  the base's default: off for every base since 0.45.0 (on for the Gemma 4 12B from 0.39.0 to 0.44.0); the flag, the
    repository and the variable each beat a base default, so `--batch-invariant on` and a repository's
    defaults.batch_invariant turn it on, and `off` and VLLM_BATCH_INVARIANT=0 are opt-outs
B8  decisio sets no batch budget: --engine passes through untouched, and the start line reports vLLM's own budget when
    the mode is on
B9  no base runs the mode by default: it is opt-in
B10 the fingerprint holds vLLM's batch budget when the mode is on (a prompt longer than it is prefilled in steps,
    repeatable but other arithmetic), so a certificate made under another budget is suspended; off, nothing changes

  uv run pytest -q tests/unit/test_batch_invariant.py
"""

import json

import pytest
from test_hub_tuned import build as build31

from decisio import hub
from decisio.serve import batch_invariant as bi
from decisio.serve import stack, vllm_engine

VAR = bi.VARIABLE


@pytest.mark.parametrize(
    ("flag", "repo", "want"),
    [
        (None, None, (False, "default")),
        (None, True, (True, "repository")),
        ("on", None, (True, "flag")),
        ("on", True, (True, "flag")),
        ("off", True, (False, "flag")),  # the flag beats the repository
        ("off", None, (False, "flag")),
    ],
)
def test_b1_the_flag_then_the_repository_then_off(flag, repo, want):
    env = {}
    r = bi.resolve(flag, repo, env)
    assert (r["on"], r["source"]) == want
    assert env == ({VAR: "1"} if want[0] else {})  # set when on, never set (or removed) otherwise


def test_b1_the_variable_is_set_in_the_environment_vllm_will_read():
    env = {"PATH": "/bin"}
    bi.resolve("on", None, env)
    assert env == {"PATH": "/bin", VAR: "1"}


@pytest.mark.parametrize(
    ("flag", "repo", "env"),
    [("on", None, "0"), (None, True, "0"), ("off", None, "1"), ("off", True, "1")],
)
def test_b2_a_variable_that_says_the_opposite_is_refused(flag, repo, env):
    with pytest.raises(SystemExit, match=rf"{VAR}={env} in the environment contradicts"):
        bi.resolve(flag, repo, {VAR: env})


def test_b2_the_variable_alone_is_recorded_and_agreeing_values_pass():
    assert bi.resolve(None, None, {VAR: "1"}) == {"on": True, "source": "environment"}
    assert bi.resolve(None, None, {VAR: "0"}) == {"on": False, "source": "environment"}
    assert bi.resolve("on", None, {VAR: "1"})["on"] and bi.resolve(None, True, {VAR: "1"})["source"] == "repository"
    assert bi.resolve("off", None, {VAR: "0"})["on"] is False
    assert bi.resolve(None, None, {VAR: ""}) == {"on": False, "source": "default"}  # empty is unset


def test_b2_other_values_and_other_backends_are_refused():
    with pytest.raises(SystemExit, match="vLLM reads 0 or 1"):
        bi.resolve(None, None, {VAR: "true"})
    with pytest.raises(SystemExit, match="not --backend mlx"):
        bi.resolve("on", None, {}, "mlx")
    with pytest.raises(SystemExit, match="not --backend hf"):
        bi.resolve(None, True, {}, "hf")
    assert bi.resolve(None, None, {}, "mlx") == {"on": False, "source": "default"}
    with pytest.raises(SystemExit, match="on or off"):
        bi.resolve("yes", None, {})


def test_b3_the_flag_is_in_the_parser():
    p = vllm_engine.build_parser()
    assert p.parse_args([]).batch_invariant is None
    assert p.parse_args(["--batch-invariant", "on"]).batch_invariant == "on"
    assert p.parse_args(["--batch-invariant", "off"]).batch_invariant == "off"
    with pytest.raises(SystemExit):
        p.parse_args(["--batch-invariant", "maybe"])


def test_b4_the_repositorys_default():
    plain = build31()
    assert "batch_invariant" not in plain["defaults"]
    on = build31(batch_invariant=True)
    assert on["defaults"]["batch_invariant"] is True
    assert hub.defaults_sha256(on) != hub.defaults_sha256(plain)  # another stack
    body = json.dumps({"tuned": plain["tuned"], "defaults": plain["defaults"]}, sort_keys=True)
    import hashlib

    assert hub.defaults_sha256(plain) == hashlib.sha256(body.encode()).hexdigest()  # a repository without it: unchanged
    for bad in (False, "yes", 1, 0):
        c = build31(batch_invariant=True)
        c["defaults"]["batch_invariant"] = bad
        with pytest.raises(ValueError, match="batch_invariant must be true"):
            hub.check_sections(c)


def test_b4_the_command_writes_it(tmp_path):
    from test_hub_tuned import ADAPTER_SHA, TEMPS

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "model-00001-of-00004.safetensors").write_bytes(b"w")
    (merged / "config.json").write_text("{}")
    temps = tmp_path / "decisio_adapter.json"
    temps.write_text(json.dumps(TEMPS))
    report = tmp_path / "MERGE_FP8.json"
    report.write_text(json.dumps({"adapter_sha256": ADAPTER_SHA, "adapter_scale": 2.0, "adapted_tensors": 840}))
    argv = [
        "tuned-config", "--base", "gemma-4-31b", "--temperatures", str(temps), "--adapter-dir", str(adapter),
        "--adapter-repository", "tachara-ai/adapter", "--adapter-revision", "abc", "--merge-report", str(report),
        "--weights-dir", str(merged), "--decisio-version", "0.32.0", "--no-noul-commit",
    ]  # fmt: skip
    for flag, key in (([], False), (["--batch-invariant"], True)):
        out = tmp_path / f"out{key}.json"
        assert hub.main([*argv, *flag, "--out", str(out)]) == 0
        defaults = json.loads(out.read_text())["defaults"]
        assert ("batch_invariant" in defaults) is key and (defaults.get("batch_invariant") is True or not key)
        assert hub.main(["check", str(out), "--weights-dir", str(merged)]) == 0


def server(report):
    from test_adapter_bands import make

    return make(temperatures={"choice": 5.0}, batch_invariant=report)


def test_b5_the_fingerprint_holds_it_only_when_on():
    _, on = server({"on": True, "source": "repository", "confirmed": True})
    _, off = server({"on": False, "source": "default"})
    _, none = server(None)
    assert on.stack_facts()["serving"]["batch_invariant"] is True
    assert (
        "batch_invariant" not in off.stack_facts()["serving"] and "batch_invariant" not in none.stack_facts()["serving"]
    )
    assert any(d.endswith("batch_invariant") for d in stack.differences(on.stack_facts(), off.stack_facts()))
    assert stack.differences(off.stack_facts(), none.stack_facts()) == []  # off is exactly what a server always was


def test_b5_health_carries_it():
    from fastapi.testclient import TestClient

    from decisio.serve.reference_docs import _DocEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore
    from decisio.serve.vllm_engine import make_app

    engine = _DocEngine()
    rep = {"on": True, "source": "flag", "confirmed": None}
    so = SystemOne(engine, "decisio-test", task_store=TaskStore("fp"), batch_invariant=rep)
    got = TestClient(make_app(engine, so)).get("/health").json()["systemone"]["batch_invariant"]
    assert got == rep
    so2 = SystemOne(
        engine, "decisio-test", task_store=TaskStore("fp"), batch_invariant={"on": False, "source": "default"}
    )
    assert "batch_invariant" not in TestClient(make_app(engine, so2)).get("/health").json()["systemone"]


def test_b6_the_start_line():
    assert bi.start_line({"on": False, "source": "default"}) == "BATCH INVARIANT off (the default)"
    assert bi.start_line({"on": True, "source": "repository", "confirmed": True}) == (
        "BATCH INVARIANT on (the repository's defaults); vLLM reads it as on"
    )
    assert "reads it as off" in bi.start_line({"on": True, "source": "flag", "confirmed": False})
    assert bi.start_line({"on": True, "source": "environment", "confirmed": None}).endswith("in the environment)")
    assert bi.engine_state() in (None, True, False)  # no vLLM on a test machine: None, never an exception


# ---- B7 to B9: the default (decisio 0.39.0) --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "backend", "image", "mode", "want"),
    [
        ("gemma-4-12b", "vllm", None, "separate", False),  # opt-in again since 0.45.0 (on from 0.39.0 to 0.44.0)
        ("gemma-4-12b-fp8", "vllm", None, "separate", False),
        ("gemma-4-31b", "vllm", None, "separate", False),
        ("gemma-4-31b-bf16", "vllm", None, "separate", False),
        ("qwen3.6-35b-a3b", "vllm", None, "separate", False),
        ("gemma-4-12b", "vllm", "some/image-model", "separate", False),
        ("gemma-4-12b", "vllm", None, "packed", False),
        ("gemma-4-12b", "mlx", None, "separate", False),
        ("gemma-4-12b", "hf", None, "separate", False),
        (None, "vllm", None, "separate", False),
    ],
)
def test_b7_the_base_default(key, backend, image, mode, want):
    assert bi.base_default(key, backend, image, mode) is want


def test_b7_the_flag_the_repository_and_the_variable_beat_the_base():
    env = {}
    r = bi.resolve(None, None, env, "vllm", True)
    assert r == {"on": True, "source": "base"} and env == {VAR: "1"}
    env = {}
    assert bi.resolve("off", None, env, "vllm", True) == {"on": False, "source": "flag"} and env == {}
    assert bi.resolve(None, False, {}, "vllm", True) == {"on": False, "source": "repository"}
    assert bi.resolve(None, None, {VAR: "0"}, "vllm", True) == {"on": False, "source": "environment"}
    assert bi.resolve(None, None, {}, "vllm", False) == {"on": False, "source": "default"}
    assert bi.resolve(None, None, {}, "mlx", True) == {"on": False, "source": "default"}


def test_b8_decisio_sets_no_batch_budget_and_the_start_line_reports_vllms():
    assert not hasattr(bi, "merge_engine")
    line = bi.start_line({"on": True, "source": "base", "confirmed": True, "max_num_batched_tokens": 8192})
    assert (
        line == "BATCH INVARIANT on (the default for this base); vLLM reads it as on; vLLM's batch budget 8,192 tokens"
    )
    assert (
        bi.start_line({"on": False, "source": "default", "max_num_batched_tokens": 8192})
        == "BATCH INVARIANT off (the default)"
    )


def test_b9_no_base_runs_the_mode_by_default():
    from decisio.families import BASES

    # no base runs the mode by default: it is opt-in (--batch-invariant on, or a repository's defaults)
    assert bi.DEFAULT_ON == ()
    assert not any(bi.base_default(k, "vllm", None, "separate") for k in BASES)


def test_b10_the_fingerprint_holds_the_batch_budget_when_the_mode_is_on():
    _, a = server({"on": True, "source": "base", "confirmed": True, "max_num_batched_tokens": 8192})
    _, b = server({"on": True, "source": "base", "confirmed": True, "max_num_batched_tokens": 16384})
    _, off = server({"on": False, "source": "default", "max_num_batched_tokens": 8192})
    _, none = server(None)
    assert a.stack_facts()["serving"]["batch_budget"] == 8192
    assert any(d.endswith("batch_budget") for d in stack.differences(a.stack_facts(), b.stack_facts()))
    assert "batch_budget" not in off.stack_facts()["serving"]
    assert stack.differences(off.stack_facts(), none.stack_facts()) == []
