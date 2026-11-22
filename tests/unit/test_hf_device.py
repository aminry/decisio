# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""--backend hf on a chosen device and precision (`--hf-device`, `--hf-dtype`), and the thinking check.

D1  the transformers engines load the model in the precision asked for and move it to the device asked for, and say
    so in their facts; the stand-in's defaults (CPU, float32) are unchanged, the model never moved
D2  the server's flags reach both engines
D3  a template that closes a thinking block it never opened leaves none open; one that opens and does not close is
    refused

  uv run pytest -q tests/unit/test_hf_device.py
"""

import types

import pytest


def _fakes(monkeypatch, loads):
    import torch
    import transformers

    class Model(torch.nn.Module):
        moved = None

        def to(self, device):
            self.moved = torch.device(device)
            return self

    def model_from(name, **kw):
        loads.append(kw)
        return Model()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(model_from))
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: types.SimpleNamespace())
    )


def test_d1_the_engines_load_on_the_device_and_in_the_precision_asked(monkeypatch):
    import torch

    from decisio.serve.hf_letters import HFLettersEngine
    from decisio.serve.hidden_engine import HFHiddenEngine

    loads = []
    _fakes(monkeypatch, loads)
    eng = HFLettersEngine("m", warm_up=False, device="meta", dtype="bfloat16")
    assert eng.model.moved == torch.device("meta") and loads[-1]["torch_dtype"] == torch.bfloat16
    assert eng.facts()["engine"] == "hf_letters (transformers on meta, bfloat16)" and eng.facts()["device"] == "meta"
    hidden = HFHiddenEngine("m", device="meta", dtype="bfloat16")
    assert hidden.model.moved == torch.device("meta") and loads[-1]["torch_dtype"] == torch.bfloat16
    plain = HFLettersEngine("m", warm_up=False)
    assert plain.model.moved is None and loads[-1]["torch_dtype"] == torch.float32
    assert plain.facts()["engine"] == "hf_letters (CPU stand-in, not for measurement)" and plain.device == "cpu"


def test_d2_the_flags(monkeypatch):
    from decisio.serve import vllm_engine as sv

    args = sv.build_parser().parse_args(["--backend", "hf", "--hf-device", "cuda", "--hf-dtype", "bfloat16"])
    assert (args.hf_device, args.hf_dtype) == ("cuda", "bfloat16")
    plain = sv.build_parser().parse_args(["--backend", "hf"])
    assert (plain.hf_device, plain.hf_dtype) == ("cpu", "float32")
    with pytest.raises(SystemExit):
        sv.build_parser().parse_args(["--hf-dtype", "int8"])


def test_d3_a_block_closed_without_an_opener_is_not_open():
    from decisio.readout import letters as L

    L.check_thinking_closed("<|Bot|></think>")
    with pytest.raises(AssertionError, match="thinking left open"):
        L.check_thinking_closed("<|Bot|><think>")
