# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A tuned repository that is a flavor (`flavor` and the read-twice keys in decision_config.json /2, decisio.hub):

T1  the file: each flavor's `defaults` agree with its name, the digest covers the flavor, a repository with no flavor
    has the digest it had
T2  every malformation is named: the flavor on a /1 file, a name that is not a flavor, a served name that does not end
    in the name, defaults of another flavor, the layout, thinking, read twice's keys
T3  the server: a flavor repository is named by its file, reads twice where its defaults say so, reports the flavor and
    the flags that differ from it (overrides), and `--flavor` stays refused for a tuned repository
T4  the command: `tuned-config --flavor` writes a file `check` accepts; a flavor its defaults disagree with is refused

  uv run pytest -q tests/unit/test_tuned_flavor.py
"""

import hashlib
import json

import pytest
from test_hub import args_for

from decisio import flavors, hub
from decisio.families import BASES
from decisio.serve.vllm_engine import resolve_base

FAM = BASES["gemma-4-12b"]
ADAPTER_SHA = hashlib.sha256(b"adapter weights").hexdigest()
TEMPS = {"choice": 1.8222, "noul": 3.314, "score": 2.1109}
BANDS = {"2-10": 2.1317, "11-26": 1.8766, "27-255": 1.7181}
NAMES = {
    "calibrated": "Decisio-12b-tuned-calibrated",
    "commit": "Decisio-12b-tuned-commit",
    "read-enhanced": "Decisio-12b-tuned-read-enhanced",
}


def build(name="calibrated", **over):
    kw = {
        "temperatures": dict(TEMPS),
        "choice_bands": dict(BANDS),
        "noul_commit": name == "commit",
        "noul_commit_basis": {"accuracy": [0.875, 0.875]} if name == "commit" else None,
        "fit": {"method": "minimum log loss per question type and choice band, on the plain readout"},
        "adapter": {
            "repository": "tachara-ai/adapter",
            "revision": "abc",
            "weights_sha256": ADAPTER_SHA,
            "config_sha256": hashlib.sha256(b"c").hexdigest(),
            "rank": 16,
            "alpha": 32,
        },
        "merge": {
            "precision": "fp32",
            "output": "bf16",
            "rounding": "stochastic",
            "seed": 0,
            "adapter_sha256": ADAPTER_SHA,
            "adapted_tensors": 328,
        },
        "files": {"model-00001-of-00006.safetensors": hashlib.sha256(b"w").hexdigest()},
        "decisio_version": "0.30.0",
        "layout": "header",
        "flavor": {"name": name, "served_name": NAMES[name]},
        "read_twice_under": 2000 if name == "read-enhanced" else None,
    }
    kw.update(over)
    return hub.tuned_config(FAM, **kw)


def repo_dir(tmp_path, config, name="repo"):
    d = tmp_path / name
    d.mkdir()
    (d / hub.FILE).write_text(json.dumps(config))
    (d / "config.json").write_text(json.dumps({"model_type": "gemma4_unified"}))
    return d


@pytest.fixture(autouse=True)
def no_forced_base(monkeypatch):
    monkeypatch.delenv("DECISIO_BASE", raising=False)


def test_t1_each_flavors_defaults_agree_with_its_name():
    c = {n: build(n) for n in NAMES}
    assert c["calibrated"]["defaults"]["noul_commit"] is False and "read_twice_under" not in c["calibrated"]["defaults"]
    assert c["commit"]["defaults"]["noul_commit"] is True
    assert c["read-enhanced"]["defaults"]["read_twice_under"] == 2000
    assert all(x["flavor"]["served_name"] == NAMES[n] for n, x in c.items())
    assert len({hub.defaults_sha256(x) for x in c.values()}) == 3  # three repositories, three identities
    assert hub.check_profile(c["commit"], FAM) == []
    plain = build(flavor=None)
    assert "flavor" not in plain
    body = json.dumps({"tuned": plain["tuned"], "defaults": plain["defaults"]}, sort_keys=True)
    assert hub.defaults_sha256(plain) == hashlib.sha256(body.encode()).hexdigest()  # a repository with no flavor
    capped = build("read-enhanced", read_twice_max_questions=8)
    assert capped["defaults"]["read_twice_max_questions"] == 8


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"flavor": {"name": "honest", "served_name": "Decisio-12b-tuned-honest"}}, "flavor.name must be one of"),
        ({"flavor": {"name": "commit", "served_name": "Decisio-12b-tuned"}}, "ending in -commit"),
        ({"flavor": {"name": "commit", "served_name": "bad name-commit"}}, "ending in -commit"),
        ({"flavor": {"name": "commit", "served_name": "x-commit", "extra": 1}}, "unknown keys"),
        ({"flavor": {"name": "commit", "served_name": "x-commit"}}, "flavor commit"),  # calibrated defaults
        ({"flavor": {"name": "read-enhanced", "served_name": "x-read-enhanced"}}, "flavor read-enhanced"),
        ({"layout": "plain"}, "header layout"),
        ({"layout": None}, "header layout"),
        ({"think": {"factor": 1.0}}, "does not think"),
        ({"read_twice_under": 2000}, "flavor calibrated"),
        ({"read_twice_max_questions": 4}, "limits defaults.read_twice_under"),
        ({"read_twice_under": -1}, "read_twice_under must be 0 or a number"),
        ({"read_twice_under": True}, "read_twice_under must be 0 or a number"),
    ],
)
def test_t2_every_malformation_is_named(over, message):
    with pytest.raises(ValueError, match=message):
        build(**over)


def test_t2_a_one_file_cannot_carry_a_flavor(tmp_path):
    c = build()
    one = hub.export_decision_config(FAM, "0.30.0")
    d = repo_dir(tmp_path, {**one, "flavor": c["flavor"]})
    with pytest.raises(ValueError, match=f"has flavor: that is {hub.SCHEMA_TUNED}"):
        hub.open_repository(str(d))


@pytest.mark.parametrize("name", list(NAMES))
def test_t3_the_server_is_named_by_the_repository(tmp_path, name):
    d = repo_dir(tmp_path, build(name))
    a = args_for(str(d))
    fam, fmt = resolve_base(a)
    assert a.served_name == NAMES[name]
    assert a.flavor_report["name"] == name and a.flavor_report["served_name"] == NAMES[name]
    assert a.flavor_report["overrides"] == {} and a.flavor_report["pending"] == []
    assert bool(a.noul_commit) is (name == "commit")
    assert fmt.read_twice_under == (2000 if name == "read-enhanced" else 0)
    assert (a.temperature_noul, a.temperature_score, a.temperature_choice) == (3.314, 2.1109, 1.8222)
    assert a.choice_bands == BANDS
    assert hub.open_repository(str(d)).facts()["flavor"] == {"name": name, "served_name": NAMES[name]}
    assert flavors.siblings(a.flavor_report) == [n for k, n in NAMES.items() if k != name]


def test_t3_a_flag_that_differs_is_an_override(tmp_path):
    d = repo_dir(tmp_path, build("commit"))
    a = args_for(str(d), served_name="mine", temperature_noul=2.5, layout="plain", noul_commit=False)
    resolve_base(a)
    assert a.served_name == "mine" and a.temperature_noul == 2.5
    o = a.flavor_report["overrides"]
    assert o["served_name"] == {"flavor": NAMES["commit"], "served": "mine"}
    assert o["temperature_noul"] == {"flavor": 3.314, "served": 2.5}
    assert o["layout"]["served"] == "plain" and o["noul_commit"]["served"] is False
    assert flavors.header_value(a.flavor_report) == "commit+overrides"


def test_t3_a_flag_equal_to_the_flavor_is_no_override_and_read_twice_follows_the_flag(tmp_path):
    d = repo_dir(tmp_path, build("read-enhanced"))
    a = args_for(str(d), temperature_noul=3.314, read_twice_under=1000)
    _, fmt = resolve_base(a)
    assert "temperature_noul" not in a.flavor_report["overrides"]
    assert a.flavor_report["overrides"]["read_twice_under"] == {"flavor": 2000, "served": 1000}
    assert fmt.read_twice_under == 1000


def test_t3_flavor_still_refuses_a_tuned_repository_and_a_repository_with_no_flavor_is_unchanged(tmp_path):
    d = repo_dir(tmp_path, build("commit"))
    with pytest.raises(ValueError, match="tuned"):
        resolve_base(args_for(str(d), flavor="commit"))
    plain = repo_dir(tmp_path, build(flavor=None), name="plain")
    a = args_for(str(plain))
    resolve_base(a)
    assert a.flavor_report is None and a.served_name == FAM.served_name


def test_t4_the_command_writes_a_file_check_accepts(tmp_path, capsys):
    temps = tmp_path / "A.json"
    temps.write_text(
        json.dumps(
            {
                "schema": "decisio.adapter/1",
                "temperatures": TEMPS,
                "choice_bands": BANDS,
                "layout": "header",
                "fit": {"method": "m"},
            }
        )
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "model-00001-of-00006.safetensors").write_bytes(b"w")
    (weights / "config.json").write_text("{}")
    report = tmp_path / "MERGE_BF16.json"
    report.write_text(
        json.dumps(
            {
                "kind": "fp32 merge to bf16",
                "rounding": "stochastic",
                "seed": 0,
                "adapter_sha256": hashlib.sha256(b"adapter weights").hexdigest(),
                "adapter_scale": 2.0,
                "adapted_tensors": 328,
                "dropped_k_eq_v_pairs": 8,
                "null": False,
            }
        )
    )
    out = tmp_path / "decision_config.json"
    base = [
        "tuned-config", "--base", "gemma-4-12b", "--temperatures", str(temps), "--adapter-dir", str(adapter),
        "--adapter-repository", "tachara-ai/adapter", "--adapter-revision", "abc", "--merge-report", str(report),
        "--weights-dir", str(weights), "--out", str(out), "--layout", "header", "--decisio-version", "0.30.0",
    ]  # fmt: skip
    ok = [*base, "--no-noul-commit", "--flavor", "calibrated", "--served-name", NAMES["calibrated"]]
    assert hub.main(ok) == 0
    assert json.loads(out.read_text())["flavor"] == {"name": "calibrated", "served_name": NAMES["calibrated"]}
    assert hub.main(["check", str(out), "--weights-dir", str(weights)]) == 0
    twice = [*base, "--no-noul-commit", "--flavor", "read-enhanced", "--served-name", NAMES["read-enhanced"]]
    assert hub.main([*twice, "--read-twice-under", "2000", "--read-twice-max-questions", "8"]) == 0
    d = json.loads(out.read_text())["defaults"]
    assert (d["read_twice_under"], d["read_twice_max_questions"]) == (2000, 8)
    capsys.readouterr()
    assert hub.main(twice) == 1 and "flavor read-enhanced" in capsys.readouterr().out  # no read twice given
    assert hub.main([*base, "--no-noul-commit", "--flavor", "commit", "--served-name", NAMES["commit"]]) == 1
