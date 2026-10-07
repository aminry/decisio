# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The default base (0.10.0): a vLLM server started with neither --base nor --model serves Gemma 4 31B, and the 31B is
served from its FP8 repository once that repository's revision is pinned (decisio.hub.DEFAULT_SOURCES). No model, no
network.

D1  neither flag on the vLLM backend: the 31B at its pinned revision, quantized on load while no repository is pinned
D2  what does not change: the other two bases by key, a checkpoint alone (the base it declares, and Qwen for a model it
    does not know, which the CPU stand-in relies on), and the backends that always name a checkpoint
D3  with the repository pinned: the 31B's key and the default are served from it, stored FP8 not quantized again, tasks
    named for Google's checkpoint; `--model google/gemma-4-31B-it` and `--revision` still serve Google's weights, the
    other bases keep their sources, and MLX is refused before anything is fetched
D4  the container: DECISIO_BASE, else the default, and a checkpoint alone naming its own base
D5  the benchmark helpers name the default base's served name

    uv run pytest -q tests/unit/test_default_base.py
"""

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from decisio import hub
from decisio.families import BASES, DEFAULT_BASE
from decisio.serve.vllm_engine import resolve_base

ROOT = Path(__file__).resolve().parents[2]
REPO = hub.REPOSITORIES["gemma-4-31b"]


def args_for(**kw):
    a = dict(
        backend="vllm",
        base=None,
        model=None,
        revision=None,
        prompt_tail=None,
        answer_slot=None,
        label_variants=None,
        system_prompt=None,
        pad_to=None,
        served_name=None,
        noul_rendering=None,
        temperature=None,
        temperature_choice=None,
    )
    return SimpleNamespace(**{**a, **kw})


def test_d1_the_default_base_is_the_31b():
    assert DEFAULT_BASE == "gemma-4-31b" and DEFAULT_BASE in BASES
    a = args_for()
    fam, _ = resolve_base(a)
    assert fam.key == "gemma-4-31b" and a.base == "gemma-4-31b"
    assert a.model == "google/gemma-4-31B-it" and a.revision == BASES["gemma-4-31b"].revision
    assert fam.quantization == "fp8" and a.repository is None  # no repository is pinned yet: Google's weights
    assert a.served_name == "decisio-gemma-4-31b-it-letters" and a.temperature == 5.252
    same = args_for(base="gemma-4-31b")
    resolve_base(same)
    assert (same.model, same.revision, same.served_name) == (a.model, a.revision, a.served_name)


def test_d2_what_does_not_change(tmp_path):
    for key in ("qwen3.6-35b-a3b", "gemma-4-12b"):
        a = args_for(base=key)
        fam, _ = resolve_base(a)
        assert fam.key == key and a.model == BASES[key].model and a.revision == BASES[key].revision
        assert a.repository is None
    # a checkpoint alone brings the base it declares; a model of a type decisio does not know is the Qwen base
    for model_type, key in (
        ("qwen3_5_moe", "qwen3.6-35b-a3b"),
        ("gemma4_unified", "gemma-4-12b"),
        ("gemma4", "gemma-4-31b"),
    ):
        d = tmp_path / model_type
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": model_type}))
        assert resolve_base(args_for(model=str(d)))[0].key == key
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert resolve_base(args_for(backend="hf", model=str(unknown)))[0].key == "qwen3.6-35b-a3b"
    # the backends that always name a checkpoint still must
    with pytest.raises(ValueError, match="--backend mlx needs --model"):
        resolve_base(args_for(backend="mlx"))
    with pytest.raises(ValueError, match="give --model or --base"):
        resolve_base(args_for(backend="hf"))


def fake_hub(monkeypatch, tmp_path, base="gemma-4-31b"):
    """A repository of the base in a directory, served as if fetched; the downloads recorded."""
    d = tmp_path / "repo"
    d.mkdir()
    (d / hub.FILE).write_text(json.dumps(hub.export_decision_config(BASES[base], "0.10.0")))
    model_type = BASES[base].model_types[0]
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    fetched = []

    def fake_download(name, filename, revision=None):
        fetched.append((name, filename, revision))
        return str(d / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    return fetched


def test_d3_the_31b_is_served_from_its_repository_once_pinned(monkeypatch, tmp_path):
    fetched = fake_hub(monkeypatch, tmp_path)
    monkeypatch.setitem(hub.PINNED_REVISIONS, REPO, "a" * 40)
    assert hub.default_source("gemma-4-31b") == f"{REPO}@{'a' * 40}"
    assert hub.default_source("gemma-4-12b") is None and hub.default_source("qwen3.6-35b-a3b") is None
    for a in (args_for(), args_for(base="gemma-4-31b")):  # the default, and the base by its key
        fam, _ = resolve_base(a)
        assert fam.key == "gemma-4-31b" and a.model == REPO and a.revision == "a" * 40
        assert fam.quantization is None  # stored FP8 is not quantized again on load
        assert a.repository["name"] == REPO and a.model_identity == "gemma-4-31B-it"  # tasks: Google's checkpoint's
    assert {(n, f) for n, f, _ in fetched} == {(REPO, hub.FILE), (REPO, "config.json")}


def test_d3_google_s_weights_stay_selectable_and_the_others_keep_their_sources(monkeypatch, tmp_path):
    fetched = fake_hub(monkeypatch, tmp_path)
    monkeypatch.setitem(hub.PINNED_REVISIONS, REPO, "a" * 40)

    def google(**kw):
        d = tmp_path / "google"
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda n, f, revision=None: str(d / f))
        return args_for(**kw)

    a = google(base="gemma-4-31b", model="google/gemma-4-31B-it")
    fam, _ = resolve_base(a)
    assert a.model == "google/gemma-4-31B-it" and a.revision == BASES["gemma-4-31b"].revision  # its pin
    assert fam.quantization == "fp8" and a.repository is None
    b = google(base="gemma-4-31b", revision="deadbeef")  # a revision names Google's checkpoint, not the repository
    resolve_base(b)
    assert b.model == "google/gemma-4-31B-it" and b.revision == "deadbeef" and b.repository is None
    for key in ("qwen3.6-35b-a3b", "gemma-4-12b"):
        c = google(base=key)
        resolve_base(c)
        assert c.model == BASES[key].model and c.repository is None
    # MLX: the base's key without a checkpoint is refused before anything is fetched
    n = len(fetched)
    with pytest.raises(ValueError, match="--backend mlx needs --model"):
        resolve_base(args_for(backend="mlx", base="gemma-4-31b"))
    assert len(fetched) == n
    # a repository named outright is still served as named, whatever the default
    (tmp_path / "again").mkdir()
    fake_hub(monkeypatch, tmp_path / "again")
    explicit = args_for(base=f"{REPO}@{'b' * 40}")
    resolve_base(explicit)
    assert explicit.revision == "b" * 40


def run_entrypoint(tmp_path, env=None, *extra):
    """docker/entrypoint.sh with a python3 that passes the GPU check and prints the server's arguments."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    py = fake / "python3"
    py.write_text('#!/bin/bash\nif [ "$1" = "-" ]; then cat > /dev/null; exit 0; fi\nprintf "%s\\n" "$@"\n')
    py.chmod(0o755)
    clean = {k: v for k, v in os.environ.items() if not k.startswith("DECISIO_")}
    r = subprocess.run(
        ["bash", str(ROOT / "docker" / "entrypoint.sh"), *extra],
        env={**clean, "PATH": f"{fake}:{clean['PATH']}", **(env or {})},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


def test_d4_the_container_serves_the_default_unless_told_otherwise(tmp_path):
    tail = ["--model-class", "hidden-readout", "--host", "0.0.0.0", "--port", "8000"]
    assert run_entrypoint(tmp_path) == ["-m", "decisio.serve.vllm_engine", "--base", "gemma-4-31b", *tail]
    other = run_entrypoint(tmp_path, {"DECISIO_BASE": "qwen3.6-35b-a3b"})
    assert other[2:4] == ["--base", "qwen3.6-35b-a3b"] and "--model" not in other
    # a checkpoint alone names its own base: a container configured as before keeps serving what it served
    old = run_entrypoint(tmp_path, {"DECISIO_MODEL": "Qwen/Qwen3.6-35B-A3B-FP8"})
    assert old[2:4] == ["--model", "Qwen/Qwen3.6-35B-A3B-FP8"] and "--base" not in old
    both = run_entrypoint(tmp_path, {"DECISIO_BASE": "gemma-4-31b", "DECISIO_MODEL": "/data/gemma"})
    assert both[2:6] == ["--base", "gemma-4-31b", "--model", "/data/gemma"]
    # the environment sets the host and port, and arguments after the image name come last, so they win
    custom = run_entrypoint(tmp_path, {"DECISIO_PORT": "9000"}, "--served-name", "mine")
    assert custom[custom.index("--port") + 1] == "9000" and custom[-2:] == ["--served-name", "mine"]


def test_d4_the_image_and_compose_do_not_pin_the_qwen_checkpoint():
    assert "Qwen" not in (ROOT / "Dockerfile").read_text().split("ENV HOME")[1].split("VOLUME")[0]
    compose = (ROOT / "compose.yaml").read_text()
    assert "DECISIO_BASE: gemma-4-31b" in compose and "DECISIO_MODEL" not in compose.replace("# ", "")


def test_d5_the_benchmark_helpers_name_the_default_base():
    path = ROOT / "benchmarks" / "make_conformance_items.py"
    spec = importlib.util.spec_from_file_location("make_conformance_items", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.served_name() == BASES[DEFAULT_BASE].served_name == "decisio-gemma-4-31b-it-letters"
    for script in ("jevbench.sh", "decision_index.sh"):
        assert "decisio-gemma-4-31b-it-letters" in (ROOT / "benchmarks" / script).read_text()


def test_d6_the_default_ships_with_its_repository_pinned():
    """The release that makes the 31B the default ships with its FP8 repository, so the default quickstart does not
    quantize Google's weights at every start (Amin, 2026-10-07). Red until the repository exists and its revision is
    pinned in decisio.hub.PINNED_REVISIONS: the guard against merging this change before the repositories are live."""
    assert REPO in hub.PINNED_REVISIONS, f"{REPO} is not pinned: the repository is not live, so this must not merge"
    assert len(hub.PINNED_REVISIONS[REPO]) == 40
    assert hub.default_source("gemma-4-31b") == f"{REPO}@{hub.PINNED_REVISIONS[REPO]}"
