# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A tuned repository whose weights are an MLX conversion of the merged model (decisio.hub, `tuned.conversion`):

M1  `tuned_config` writes the MLX weights section and the conversion record; the file passes the checks, its defaults
    digest covers the conversion, and a repository without one is what it was
M2  every malformation is named: an unknown key, bits or group size outside the shipped ones, another format, an FP8 or
    unsupported base, a weights section that is not the conversion's
M3  the command: `tuned-config --mlx-conversion` writes a file `check` accepts, and refuses a conversion whose own
    config.json declares another quantization
M4  a repository read from a directory knows it is MLX; a server for vLLM refuses it and `--backend mlx` takes it, with
    the repository's defaults and revision

  uv run pytest -q tests/unit/test_hub_mlx.py
"""

import hashlib
import json

import pytest
from test_hub import args_for
from test_hub_tuned import ADAPTER_SHA, BASIS, CONFIG_SHA, TEMPS

from decisio import hub
from decisio.families import BASES
from decisio.serve.vllm_engine import resolve_base

FAM12 = BASES["gemma-4-12b"]
CONVERSION = {"tool": "mlx-lm", "tool_version": "0.32.0", "format": "mlx", "bits": 6, "group_size": 64}
MERGE = {
    "precision": "fp32",
    "output": "bf16",
    "rounding": "stochastic",
    "seed": 0,
    "adapter_sha256": ADAPTER_SHA,
    "adapted_tensors": 328,
}


def build(**over):
    kw = {
        "temperatures": dict(TEMPS["temperatures"]),
        "choice_bands": dict(TEMPS["choice_bands"]),
        "noul_commit": True,
        "noul_commit_basis": BASIS,
        "fit": {"method": "minimum log loss"},
        "adapter": {
            "revision": "abc",
            "weights_sha256": ADAPTER_SHA,
            "config_sha256": CONFIG_SHA,
            "rank": 16,
            "alpha": 32,
        },
        "merge": dict(MERGE),
        "files": {"model-00001-of-00002.safetensors": hashlib.sha256(b"w").hexdigest()},
        "decisio_version": "0.33.0",
        "layout": "header",
        "conversion": dict(CONVERSION),
    }
    kw.update(over)
    return hub.tuned_config(FAM12, **kw)


def test_m1_the_mlx_weights_section_and_the_conversion_record():
    c = build()
    assert c["weights"]["dtype"] == "mlx_affine" and c["weights"]["bits"] == 6 and c["weights"]["group_size"] == 64
    assert c["weights"]["quantization"] == "mlx-6bit" and c["weights"]["quantized_on_load"] is False
    assert "mlx-lm 0.32.0" in c["weights"]["method"] and "not by decisio" in c["weights"]["method"]
    assert c["tuned"]["conversion"] == CONVERSION
    assert hub.check_profile(c, FAM12) == []
    hub.check_sections(c)
    plain = build(conversion=None)
    assert plain["weights"]["dtype"] == "bfloat16" and "conversion" not in plain["tuned"]
    assert hub.defaults_sha256(c) != hub.defaults_sha256(plain)
    eight = build(conversion={**CONVERSION, "bits": 8})
    assert eight["weights"]["quantization"] == "mlx-8bit" and hub.defaults_sha256(eight) != hub.defaults_sha256(c)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"extra": 1}, "unknown keys"),
        ({"bits": 5}, "bits must be one of"),
        ({"group_size": 16}, "group_size must be one of"),
        ({"format": "gguf"}, "format must be 'mlx'"),
        ({"tool": "bad tool!"}, "tool must be a name"),
    ],
)
def test_m2_malformed_conversions_are_named(change, match):
    with pytest.raises(ValueError, match=match):
        build(conversion={**CONVERSION, **change})


def test_m2_bases_and_weights_sections():
    with pytest.raises(ValueError, match="made for gemma-4-31b, gemma-4-12b, not gemma-4-12b-fp8"):
        hub.check_conversion(CONVERSION, "gemma-4-12b-fp8")
    c = build()
    c["weights"]["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="weights must be the MLX conversion's section"):
        hub.check_sections(c)
    c = build()
    c["weights"]["bits"] = 8
    with pytest.raises(ValueError, match="differs at weights.bits"):
        hub.check_sections(c)
    c = build(conversion=None)
    c["weights"] = build()["weights"]  # an MLX weights section without a conversion record
    with pytest.raises(ValueError, match="weights.dtype"):
        hub.check_sections(c)


def write_mlx_dir(tmp_path, bits=6, group=64, name="mlx"):
    d = tmp_path / name
    d.mkdir()
    (d / "model-00001-of-00002.safetensors").write_bytes(b"w")
    (d / "config.json").write_text(
        json.dumps(
            {"model_type": "gemma4_unified", "quantization": {"bits": bits, "group_size": group, "mode": "affine"}}
        )
    )
    return d


def command(tmp_path, d, extra=()):
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    temps = tmp_path / "decisio_adapter.json"
    temps.write_text(json.dumps(TEMPS))
    report = tmp_path / "MERGE_BF16.json"
    report.write_text(json.dumps({**MERGE, "tool": "decisio.hub_bf16 merge"}))
    out = tmp_path / "decision_config.json"
    argv = [
        "tuned-config", "--base", "gemma-4-12b", "--temperatures", str(temps), "--adapter-dir", str(adapter),
        "--adapter-revision", "abc", "--merge-report", str(report), "--weights-dir", str(d), "--out", str(out),
        "--decisio-version", "0.33.0", "--no-noul-commit", "--layout", "header",
        "--mlx-conversion", "mlx-lm@0.32.0", "--mlx-bits", "6", "--mlx-group-size", "64", *extra,
    ]  # fmt: skip
    return argv, out


def test_m3_the_command_writes_a_file_check_accepts(tmp_path, capsys):
    d = write_mlx_dir(tmp_path)
    argv, out = command(tmp_path, d)
    assert hub.main(argv) == 0
    c = json.loads(out.read_text())
    assert c["weights"]["dtype"] == "mlx_affine" and c["tuned"]["conversion"] == CONVERSION
    assert set(c["tuned"]["files"]) == {"model-00001-of-00002.safetensors", "config.json"}
    (d / "decision_config.json").write_text(out.read_text())
    assert hub.main(["check", str(d / "decision_config.json"), "--weights-dir", str(d)]) == 0


def test_m3_a_conversion_that_declares_other_quantization_is_refused(tmp_path, capsys):
    d = write_mlx_dir(tmp_path, bits=8)
    argv, _ = command(tmp_path, d)
    assert hub.main(argv) == 1
    assert "not 6-bit affine with group size 64" in capsys.readouterr().out


def test_m3_the_flags_need_each_other(tmp_path, capsys):
    d = write_mlx_dir(tmp_path)
    argv, _ = command(tmp_path, d)
    i = argv.index("--mlx-bits")
    assert hub.main(argv[:i] + argv[i + 2 :]) == 1
    assert "needs --mlx-bits and --mlx-group-size" in capsys.readouterr().out


def repo_dir(tmp_path):
    d = write_mlx_dir(tmp_path, name="repo")
    (d / hub.FILE).write_text(json.dumps(build()))
    return d


def test_m4_vllm_refuses_an_mlx_repository_and_mlx_takes_it(tmp_path):
    d = repo_dir(tmp_path)
    assert hub.open_repository(str(d)).mlx is True
    with pytest.raises(ValueError, match="MLX conversion.*--backend mlx"):
        resolve_base(args_for(str(d)))
    args = args_for(str(d), backend="mlx")
    fam, _ = resolve_base(args)
    assert fam.key == "gemma-4-12b" and args.model == str(d)
    assert args.repository["defaults_sha256"] == hub.defaults_sha256(
        build()
    )  # the repository's defaults are the server's
    assert args.repository["tuned"]["conversion"] == CONVERSION


def test_m4_a_plain_tuned_repository_is_not_mlx(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    (d / hub.FILE).write_text(json.dumps(build(conversion=None)))
    assert hub.open_repository(str(d)).mlx is False
