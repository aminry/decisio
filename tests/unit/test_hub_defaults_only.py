# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A repository of the base's own weights with settings of their own (decisio.hub, a `/2` file without `tuned`):

D1  `defaults_config` writes the base's profile, source and weights section unchanged plus `defaults` (and a flavor),
    lists the weight files in `source.files`, passes the checks and has a defaults digest that follows every setting
D2  every malformation is named: weights that are not the base's own, no defaults, no source, bad file digests; a `/1`
    file with defaults is still refused, and a tuned file still needs its tuned section
D3  the command: `tuned-config --defaults-only` writes a file `check --weights-dir` accepts and finds a changed file in;
    the options about a merge are refused with it and required without it
D4  a repository read from a directory: a server resolves it with the settings as its defaults (temperatures, bands,
    commit rule, layout), the stored FP8 weights not quantized again, a flag beating them

  uv run pytest -q tests/unit/test_hub_defaults_only.py
"""

import hashlib
import json

import pytest
from test_hub import args_for
from test_hub_tuned import BASIS, TEMPS

from decisio import hub
from decisio.families import BASES
from decisio.serve.vllm_engine import resolve_base

FAM31 = BASES["gemma-4-31b"]
FILES = {"model-00001-of-00004.safetensors": hashlib.sha256(b"w").hexdigest()}


def build(**over):
    kw = {
        "temperatures": dict(TEMPS["temperatures"]),
        "choice_bands": dict(TEMPS["choice_bands"]),
        "noul_commit": True,
        "noul_commit_basis": BASIS,
        "fit": {"method": "minimum log loss per question type and choice band, on the plain readout"},
        "layout": "header",
        "files": dict(FILES),
        "decisio_version": "0.37.0",
    }
    kw.update(over)
    return hub.defaults_config(FAM31, **kw)


def test_d1_the_base_unchanged_plus_its_settings():
    c = build()
    plain = hub.export_decision_config(FAM31, "0.37.0")
    assert c["schema"] == hub.SCHEMA_TUNED and "tuned" not in c
    assert c["weights"] == plain["weights"] and hub.check_profile(c, FAM31) == []
    assert c["source"] == {**plain["source"], "files": FILES}
    assert c["defaults"]["temperatures"] == TEMPS["temperatures"] and c["defaults"]["layout"] == "header"
    assert c["defaults"]["noul_commit"] is True and c["defaults"]["noul_commit_basis"] == BASIS
    hub.check_sections(c)
    assert hub.defaults_sha256(c) != hub.defaults_sha256(build(noul_commit=False, noul_commit_basis=None))
    other = dict(TEMPS["temperatures"], noul=TEMPS["temperatures"]["noul"] + 0.1)
    assert hub.defaults_sha256(c) != hub.defaults_sha256(build(temperatures=other))
    assert hub.defaults_sha256(c) == hub.defaults_sha256(build())  # deterministic
    flav = build(flavor={"name": "commit", "served_name": "Decisio-31b-tuned-commit"})
    assert flav["flavor"]["name"] == "commit"
    with pytest.raises(TypeError, match="unexpected settings"):
        build(adapter={})


def test_d2_malformations_are_named():
    c = build()
    c["weights"] = {**c["weights"], "dtype": "bfloat16"}
    with pytest.raises(ValueError, match="serves the base's own weights.*differs at weights.dtype"):
        hub.check_sections(c)
    c = build()
    del c["defaults"]
    with pytest.raises(ValueError, match="defaults"):
        hub.check_sections(c)
    c = build()
    del c["source"]
    with pytest.raises(ValueError, match="source"):
        hub.check_sections(c)
    c = build()
    c["source"]["files"] = {"a.safetensors": "nothex"}
    with pytest.raises(ValueError, match=r"source.files\['a.safetensors'\] must be a sha256"):
        hub.check_sections(c)
    c = build()
    c["source"]["files"] = {}
    with pytest.raises(ValueError, match="source.files must list the weight files"):
        hub.check_sections(c)
    c = build()
    c["schema"] = hub.SCHEMA
    with pytest.raises(ValueError, match="but has defaults"):
        hub.check_sections(c)  # a /1 file carries no defaults
    with pytest.raises(ValueError, match="made for"):
        hub.defaults_config(BASES["qwen3.6-35b-a3b"], temperatures={}, noul_commit=False, fit={})


def write_weights(tmp_path, name="w"):
    d = tmp_path / name
    d.mkdir()
    (d / "model-00001-of-00004.safetensors").write_bytes(b"w")
    fp8 = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float"}}},
    }
    (d / "config.json").write_text(json.dumps({"model_type": "gemma4", "quantization_config": fp8}))
    return d


def command(tmp_path, weights, extra=()):
    temps = tmp_path / "decisio_adapter.json"
    temps.write_text(json.dumps(TEMPS))
    out = tmp_path / "decision_config.json"
    argv = [
        "tuned-config", "--base", "gemma-4-31b", "--temperatures", str(temps), "--weights-dir", str(weights),
        "--out", str(out), "--decisio-version", "0.37.0", "--noul-commit", "--noul-commit-basis", "because",
        "--layout", "header", *extra,
    ]  # fmt: skip
    return argv, out


def test_d3_the_command_writes_it_and_check_finds_a_changed_file(tmp_path, capsys):
    w = write_weights(tmp_path)
    argv, out = command(tmp_path, w, ["--defaults-only"])
    assert hub.main(argv) == 0
    c = json.loads(out.read_text())
    assert "tuned" not in c and set(c["source"]["files"]) == {"model-00001-of-00004.safetensors", "config.json"}
    assert hub.main(["check", str(out), "--weights-dir", str(w)]) == 0
    (w / "model-00001-of-00004.safetensors").write_bytes(b"changed")
    assert hub.main(["check", str(out), "--weights-dir", str(w)]) == 1
    assert "files differ from source.files: model-00001-of-00004.safetensors" in capsys.readouterr().out


def test_d3_the_merge_options_go_with_a_merge_only(tmp_path):
    w = write_weights(tmp_path)
    argv, _ = command(tmp_path, w, ["--defaults-only", "--adapter-revision", "abc"])
    with pytest.raises(SystemExit):
        hub.main(argv)
    argv, _ = command(tmp_path, w)  # no --defaults-only and no adapter
    with pytest.raises(SystemExit):
        hub.main(argv)


def test_d4_a_server_resolves_it_with_the_settings_as_its_defaults(tmp_path):
    w = write_weights(tmp_path)
    argv, out = command(tmp_path, w, ["--defaults-only"])
    assert hub.main(argv) == 0
    (w / hub.FILE).write_text(out.read_text())
    repo = hub.open_repository(str(w))
    assert repo.tuned and repo.defaults["noul_commit"] is True and "tuned" not in repo.facts()
    assert repo.facts()["defaults_sha256"] == hub.defaults_sha256(json.loads(out.read_text()))
    args = args_for(str(w))
    fam, _ = resolve_base(args)
    assert fam.key == "gemma-4-31b" and fam.quantization is None  # the stored FP8 is not quantized again
    assert args.noul_commit is True and args.temperature_noul == TEMPS["temperatures"]["noul"]
    assert args.choice_bands == TEMPS["choice_bands"] and args.layout_recorded == "header"
    assert args.repository["defaults_sha256"] == repo.defaults_sha256
    flag = args_for(str(w), noul_commit=False)  # a flag beats the repository
    resolve_base(flag)
    assert flag.noul_commit is False
