# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The 31B in bf16 (`--base gemma-4-31b-bf16`, decisio.families), beside the default 31B (FP8), without a model:

Q1  the same profile as the default 31B: only the precision, the name and the context ceiling differ
Q2  the command line: --base gemma-4-31b-bf16 serves Google's checkpoint at its pin with no quantization; the default
    31B still quantizes the same checkpoint to FP8 on load; an FP8 checkpoint under the bf16 base is refused naming
    the other; MLX refuses it
Q3  detection without --base is unchanged: the 31B's checkpoint is the default 31B
Q4  the start-up line says bf16 and as the checkpoint stores it; the context is the largest that fits up to 65,536
Q5  it is its own stack (key, served name, task fingerprint); flavors and tuned repositories are not offered for it

  uv run pytest -q tests/unit/test_31b_bf16.py
"""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from decisio import flavors, hub
from decisio.families import BASES, FAMILIES, GEMMA4_31B, GEMMA4_31B_BF16, family_of
from decisio.serve import vllm_engine as sv
from decisio.serve.context import AutoContext

FP8_QUANT = {
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8, "strategy": "tensor"}}},
}


def checkpoint(tmp_path, name, fp8=False):
    d = tmp_path / name
    d.mkdir()
    cfg = {"model_type": "gemma4", "text_config": {"enable_moe_block": False}}
    if fp8:
        cfg["quantization_config"] = FP8_QUANT
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


def args(**kw):
    a = dict(
        backend="vllm", base=None, model=None, revision=None, prompt_tail=None, answer_slot=None,
        label_variants=None, system_prompt=None, pad_to=None, served_name=None, noul_rendering=None,
        temperature=None, temperature_choice=None,
    )  # fmt: skip
    return SimpleNamespace(**{**a, **kw})


@pytest.fixture(autouse=True)
def no_forced_base(monkeypatch):
    monkeypatch.delenv("DECISIO_BASE", raising=False)


@pytest.fixture
def google_config(monkeypatch):
    """read_config without the network: Google's 31B checkpoint declares model type gemma4 and no FP8; a local directory
    is read as it is."""
    from decisio import families

    real = families.read_config

    def read(model, revision=None):
        return real(model, revision) if Path(model).exists() else {"model_type": "gemma4", "text_config": {}}

    monkeypatch.setattr("decisio.families.read_config", read)


def test_q1_the_same_profile_as_the_default_31b():
    own = {"key", "served_name", "quantization", "precision", "context_ceiling", "detected", "vllm_only"}
    a, b = dataclasses.asdict(GEMMA4_31B), dataclasses.asdict(GEMMA4_31B_BF16)
    assert {k for k in a if a[k] != b[k]} == own
    assert (GEMMA4_31B_BF16.model, GEMMA4_31B_BF16.revision) == (GEMMA4_31B.model, GEMMA4_31B.revision)
    assert GEMMA4_31B.quantization == "fp8" and GEMMA4_31B_BF16.quantization is None
    assert GEMMA4_31B_BF16.precision == "bf16" and GEMMA4_31B_BF16.context_ceiling == 65536
    for k in ("temperature", "choice_temperature", "layout", "hidden_start", "softcap", "classes", "multi_question"):
        assert getattr(GEMMA4_31B_BF16, k) == getattr(GEMMA4_31B, k), k
    assert BASES["gemma-4-31b-bf16"] is GEMMA4_31B_BF16 and GEMMA4_31B_BF16 in FAMILIES


def test_q2_the_command_line(tmp_path, google_config):
    a = args(base="gemma-4-31b-bf16")  # alone: Google's checkpoint at its pin, no quantization
    fam, fmt = sv.resolve_base(a)
    assert fam is GEMMA4_31B_BF16 and fam.quantization is None
    assert (a.model, a.revision) == ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475")
    assert a.served_name == "decisio-gemma-4-31b-it-bf16-letters" and a.temperature == GEMMA4_31B.temperature
    other = args(base="gemma-4-31b", model="google/gemma-4-31B-it")  # the default 31B on the same checkpoint
    fam31, fmt31 = sv.resolve_base(other)
    assert fam31 is GEMMA4_31B and fam31.quantization == "fp8" and fmt == fmt31
    assert (other.model, other.revision) == (a.model, a.revision)
    stored, bf16 = checkpoint(tmp_path, "stored", fp8=True), checkpoint(tmp_path, "bf16")
    with pytest.raises(ValueError, match="declares FP8 weights: serve it with --base gemma-4-31b"):
        sv.resolve_base(args(base="gemma-4-31b-bf16", model=stored))
    assert sv.resolve_base(args(base="gemma-4-31b-bf16", model=bf16))[0] is GEMMA4_31B_BF16
    assert sv.resolve_base(args(base="gemma-4-31b", model=stored))[0].quantization is None  # stored FP8, as before
    with pytest.raises(ValueError, match="--backend mlx does not serve gemma-4-31b-bf16"):
        sv.resolve_base(args(base="gemma-4-31b-bf16", model=bf16, backend="mlx"))


def test_q3_detection_without_a_base_is_unchanged(tmp_path):
    bf16, stored = checkpoint(tmp_path, "bf16"), checkpoint(tmp_path, "stored", fp8=True)
    assert family_of(bf16) is GEMMA4_31B  # quantized to FP8 on load, as every release before this one
    assert family_of(stored).key == "gemma-4-31b" and family_of(stored).quantization is None
    assert sv.resolve_base(args(model=bf16))[0] is GEMMA4_31B
    assert sv.resolve_base(args(base="gemma-4-31b", model=bf16))[0] is GEMMA4_31B


def test_q4_the_start_up_line_and_the_context():
    line = sv.base_line(GEMMA4_31B_BF16, GEMMA4_31B_BF16.model)
    assert line == "BASE gemma-4-31b-bf16 (bf16, as the checkpoint stores it) from google/gemma-4-31B-it"
    assert sv.base_line(GEMMA4_31B, "x").startswith("BASE gemma-4-31b (quantized to FP8 on load)")
    assert sv.resolve_context(GEMMA4_31B_BF16, None, None) == (AutoContext(65536), None)
    assert sv.resolve_context(GEMMA4_31B_BF16, 65536, None) == (65536, None)
    with pytest.raises(ValueError, match=r"gemma-4-31b-bf16 base is documented to 65,536"):
        sv.resolve_context(GEMMA4_31B_BF16, 65537, None)


def test_q5_its_own_stack_and_no_repository_or_flavors():
    bf16, fp8 = hub.profile_of(GEMMA4_31B_BF16), hub.profile_of(GEMMA4_31B)
    assert bf16["prompt"]["fingerprint"]["base"] == "gemma-4-31b-bf16"  # tasks fitted on the FP8 31B are not applied
    assert bf16["prompt"]["fingerprint"] != fp8["prompt"]["fingerprint"]
    assert bf16["serving"]["served_name"] != fp8["serving"]["served_name"]
    assert "gemma-4-31b-bf16" not in hub.REPOSITORIES and "gemma-4-31b-bf16" not in hub.TUNED_BASES
    with pytest.raises(flavors.FlavorError, match="has no flavors"):
        flavors.flavor_of("gemma-4-31b-bf16", "calibrated")
