# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A checkpoint of a model type no base serves is refused, not given the Qwen profile.

U1  `family_of` refuses a config.json naming a model type no base serves, and a checkpoint with no readable config.json,
    with a message naming the supported bases and `--base`; the known types, no checkpoint (the default base) and
    DECISIO_BASE are as before
U2  the server's argument resolution refuses `--model <unknown>` on the vLLM and MLX backends, and `--base <key>`
    with the same `--model` is the explicit way to serve it as that base's profile
U3  the CPU stand-in (`--backend hf`) keeps the Qwen profile for a small model of an unknown type

  uv run pytest -q tests/unit/test_unknown_model_type.py
"""

import json
from types import SimpleNamespace

import pytest

from decisio.families import BASES, QWEN, family_of
from decisio.serve.vllm_engine import resolve_base


def checkpoint(tmp_path, model_type, name="ckpt", **extra):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type, **extra}))
    return str(d)


def args_for(**kw):
    a = dict(
        backend="vllm", base=None, model=None, revision=None, prompt_tail=None, answer_slot=None,
        label_variants=None, system_prompt=None, pad_to=None, served_name=None, noul_rendering=None,
        temperature=None, temperature_choice=None,
    )  # fmt: skip
    return SimpleNamespace(**{**a, **kw})


def test_u1_family_of_refuses_what_no_base_serves(tmp_path, monkeypatch):
    monkeypatch.delenv("DECISIO_BASE", raising=False)
    unknown = checkpoint(tmp_path, "qwen3_8", name="qwen38")
    with pytest.raises(ValueError) as e:
        family_of(unknown)
    msg = str(e.value)
    assert "'qwen3_8'" in msg and "no supported base serves" in msg and "--base" in msg
    assert all(key in msg for key in BASES)  # the supported bases are named
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="could not be read"):
        family_of(str(empty))
    # as before: a known type, no checkpoint named, the forced base
    assert family_of(checkpoint(tmp_path, "gemma4", name="g")).key == "gemma-4-31b"
    assert family_of(checkpoint(tmp_path, "qwen3_5_moe", name="q")).key == "qwen3.6-35b-a3b"
    assert family_of(None) is QWEN
    monkeypatch.setenv("DECISIO_BASE", "gemma-4-12b")
    assert family_of(unknown).key == "gemma-4-12b"


def test_u2_the_server_refuses_it_and_base_is_the_explicit_override(tmp_path, monkeypatch):
    monkeypatch.delenv("DECISIO_BASE", raising=False)
    unknown = checkpoint(tmp_path, "qwen3_8", name="qwen38")
    for backend in ("vllm", "mlx"):
        with pytest.raises(ValueError, match="no supported base serves"):
            resolve_base(args_for(backend=backend, model=unknown))
    fam, _ = resolve_base(args_for(base="qwen3.6-35b-a3b", model=unknown))  # the operator says what it is
    assert fam.key == "qwen3.6-35b-a3b"


def test_u3_the_stand_in_keeps_the_qwen_profile_for_a_small_model(tmp_path, monkeypatch):
    monkeypatch.delenv("DECISIO_BASE", raising=False)
    small = checkpoint(tmp_path, "qwen3", name="tiny")
    fam, _ = resolve_base(args_for(backend="hf", model=small))
    assert fam.key == "qwen3.6-35b-a3b"
