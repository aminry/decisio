# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A tuned repository whose checkpoint another tool made (`hub tuned-config --external-form TOOL@VERSION`):

X1  the record: tool, version, scheme, weights and activations granularity, merge precision, adapter digest and
    scale; the weights text is generated from it and does not say decisio made the merge
X2  what is refused: a record that is not a TOOL@VERSION, unknown keys (nothing about data), another adapter's digest, a
    scale of 0, values outside the lists, a non-FP8 base, decisio's own tool name
X3  decisio's own checks stay for decisio's own tools: the fp32 / adapted_tensors rules apply to its reports and not to
    an external one, and the older digest of a repository with no external form is unchanged
X4  the command writes a file `check` accepts with the whole file list hashed, and a changed file is found
X5  the repository serves: resolve_base takes it as a stored FP8 checkpoint with its defaults

  uv run pytest -q tests/unit/test_external_form.py
"""

import hashlib
import json

import pytest
from test_hub import args_for
from test_hub_tuned import ADAPTER_SHA, TEMPS, build

from decisio import hub
from decisio.families import BASES
from decisio.serve.vllm_engine import resolve_base

FAM = BASES["gemma-4-31b"]
REPORT = {
    "adapter_sha256": ADAPTER_SHA,
    "adapter_scale": 2.0,
    "merge_precision": "fp32",
    "scheme": "FP8_DYNAMIC",
    "weights": "per-channel",
    "activations": "per-token dynamic",
    "bf16_modules": ["lm_head", "re:.*vision_tower.*"],
}
FORM = "llm-compressor@0.8.1"


def merge_of(**over):
    return hub.external_facts(FORM, {**REPORT, **over}, "gemma-4-31b")


def test_x1_the_record_and_the_weights_text():
    m = merge_of()
    assert m["tool"] == "llm-compressor" and m["tool_version"] == "0.8.1" and m["external"] is True
    assert (m["scheme"], m["weights"], m["activations"]) == ("FP8_DYNAMIC", "per-channel", "per-token dynamic")
    assert m["precision"] == "fp32" and m["output"] == "fp8 compressed-tensors" and m["adapter_scale"] == 2.0
    c = build(merge=m)
    text = c["weights"]["method"]
    assert "the merged weights were held in fp32 when quantized" in text
    assert "llm-compressor 0.8.1" in text and "FP8_DYNAMIC: weights per-channel, activations per-token dynamic" in text
    assert "not by decisio's merge" in text and "modules left in bf16 are listed" in text
    assert c["weights"]["dtype"] == "fp8_e4m3" and c["weights"]["quantized_on_load"] is False
    assert hub.check_profile(c, FAM) == []
    assert "decisio.hub_fp8" not in json.dumps(c["tuned"]["merge"])
    assert "modules left in bf16" not in build(merge=merge_of(bf16_modules=None))["weights"]["method"]


@pytest.mark.parametrize(
    ("form", "over", "message"),
    [
        ("llm-compressor", {}, "expected TOOL@VERSION"),
        ("llm compressor@1", {}, "expected TOOL@VERSION"),
        (FORM, {"notes": "x"}, "unknown keys"),
        (FORM, {"adapter_sha256": "abc"}, "report.adapter_sha256 must be a sha256"),
        (FORM, {"adapter_scale": 0}, "positive number"),
        (FORM, {"merge_precision": "int8"}, "merge_precision must be one of"),
        (FORM, {"weights": "per-row"}, "weights must be one of"),
        (FORM, {"activations": "none"}, "activations must be one of"),
        (FORM, {"scheme": "FP8 DYNAMIC"}, "report.scheme"),
        (FORM, {"bf16_modules": "lm_head"}, "bf16_modules must be a list"),
        ("decisio-fp8@1", {}, "another tool's checkpoints"),
    ],
)
def test_x2_what_is_refused(form, over, message):
    with pytest.raises(ValueError, match=message):
        hub.external_facts(form, {**REPORT, **over}, "gemma-4-31b")
    with pytest.raises(ValueError, match="not an FP8 base"):
        hub.external_facts(FORM, REPORT, "gemma-4-12b")
    assert hub.external_facts(FORM, REPORT, "gemma-4-12b-fp8")["external"] is True


def test_x2_another_adapters_digest_is_refused_by_the_check():
    other = merge_of(adapter_sha256=hashlib.sha256(b"another").hexdigest())
    with pytest.raises(ValueError, match="merge report is of another adapter"):
        build(merge=other)


def test_x3_decisios_own_checks_stay_for_decisios_own_tools():
    c = build()  # decisio's own merge record, as before
    assert "external" not in c["tuned"]["merge"]
    for key, bad in (("precision", "bf16"), ("adapted_tensors", 0)):
        c2 = build()
        c2["tuned"]["merge"][key] = bad
        with pytest.raises(ValueError):
            hub.check_sections(c2)
    # the same two on an external record are not decisio's business
    ext = build(merge=merge_of(merge_precision="bf16"))
    assert ext["tuned"]["merge"]["precision"] == "bf16" and "adapted_tensors" not in ext["tuned"]["merge"]
    # a record that claims to be external but carries decisio's fields is refused
    bad = build(merge=merge_of())
    bad["tuned"]["merge"]["adapted_tensors"] = 840
    with pytest.raises(ValueError, match="beyond the ones an external form records"):
        hub.check_sections(bad)
    plain = build()
    body = json.dumps({"tuned": plain["tuned"], "defaults": plain["defaults"]}, sort_keys=True)
    assert hub.defaults_sha256(plain) == hashlib.sha256(body.encode()).hexdigest()


def make_files(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    merged = tmp_path / "ckpt"
    merged.mkdir()
    (merged / "model-00001-of-00002.safetensors").write_bytes(b"w1")
    (merged / "model-00002-of-00002.safetensors").write_bytes(b"w2")
    (merged / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8, "strategy": "channel"}}},
                },
            }
        )
    )
    (merged / "tokenizer.json").write_text("{}")
    temps = tmp_path / "A.json"
    temps.write_text(json.dumps(TEMPS))
    report = tmp_path / "external_report.json"
    report.write_text(json.dumps({**REPORT, "adapter_sha256": hashlib.sha256(b"adapter weights").hexdigest()}))
    return adapter, merged, temps, report


def command(tmp_path, adapter, merged, temps, report, extra=()):
    return [
        "tuned-config", "--base", "gemma-4-31b", "--temperatures", str(temps), "--adapter-dir", str(adapter),
        "--adapter-revision", "abc", "--merge-report", str(report), "--weights-dir", str(merged),
        "--out", str(merged / "decision_config.json"), "--decisio-version", "0.32.0", "--no-noul-commit",
        "--layout", "header", "--batch-invariant", *extra,
    ]  # fmt: skip


def test_x4_the_command_writes_a_file_check_accepts_and_hashes_the_whole_list(tmp_path, capsys):
    adapter, merged, temps, report = make_files(tmp_path)
    assert hub.main(command(tmp_path, adapter, merged, temps, report, ["--external-form", FORM])) == 0
    written = json.loads((merged / "decision_config.json").read_text())
    assert written["tuned"]["merge"]["tool"] == "llm-compressor" and written["defaults"]["batch_invariant"] is True
    assert set(written["tuned"]["files"]) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "config.json",
        "tokenizer.json",
    }
    capsys.readouterr()
    assert hub.main(["check", str(merged / "decision_config.json"), "--weights-dir", str(merged)]) == 0
    assert "agrees" in capsys.readouterr().out
    (merged / "model-00002-of-00002.safetensors").write_bytes(b"changed")
    assert hub.main(["check", str(merged / "decision_config.json"), "--weights-dir", str(merged)]) == 1
    assert "model-00002-of-00002.safetensors" in capsys.readouterr().out
    # without --external-form the same hand-written report is not decisio's report: its counts are missing
    capsys.readouterr()
    assert hub.main(command(tmp_path, adapter, merged, temps, report)) == 1
    assert "refused" in capsys.readouterr().out


def test_x5_the_repository_serves_as_a_stored_fp8_checkpoint(tmp_path):
    adapter, merged, temps, report = make_files(tmp_path)
    assert hub.main(command(tmp_path, adapter, merged, temps, report, ["--external-form", FORM])) == 0
    a = args_for(str(merged))
    fam, _ = resolve_base(a)
    assert fam.key == "gemma-4-31b" and fam.quantization is None  # declares FP8: loaded as stored, not quantized again
    assert a.repository["tuned"]["merge"]["external"] is True
    assert (a.temperature_noul, a.temperature_choice) == (
        TEMPS["temperatures"]["noul"],
        TEMPS["temperatures"]["choice"],
    )
    assert a.served_name == fam.served_name
