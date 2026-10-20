# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The `/2` schema of `decision_config.json`: a tuned model's repository (decisio.hub; item 3 of the board path).

A1  a /2 file is built from a base's profile, a temperature file, the commit setting, the adapter's and the merge's
    facts and the merged files' digests; it carries the profile unchanged, a weights section that says the tensors were
    modified by a merge, and `tuned` and `defaults`
A2  an older decisio refuses it by its schema (its reader's condition, and the real 0.19.0 reader when run by hand), and
    this decisio refuses a /1 file that carries `tuned` or `defaults`, and a /3
A3  every malformation is named by its path: temperatures and bands as an adapter file's, the commit setting stated,
    the adapter's and the files' digests, the precision, a merge report of another adapter or of a null merge
A4  the defaults digest covers `tuned` and `defaults` and nothing else; a repository read from a directory carries the
    defaults and the digest in its facts; /1 repositories are unchanged
A5  the profile of a /2 file is still compared with the installed one (a repository cannot change how the base is read)
A6  the commands: `tuned-config` writes a file `check` accepts, refuses a missing commit decision, a merge report of
    another adapter and a null merge; `check --weights-dir` finds a changed file

  uv run pytest -q tests/unit/test_hub_tuned.py
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from decisio import hub
from decisio.families import BASES

FAM = BASES["gemma-4-31b"]
ADAPTER_SHA = hashlib.sha256(b"adapter weights").hexdigest()
CONFIG_SHA = hashlib.sha256(b"adapter config").hexdigest()
TEMPS = {
    "schema": "decisio.adapter/1",
    "temperatures": {"choice": 1.5885, "noul": 4.1932, "score": 6.2068},
    "choice_bands": {"2-10": 2.4454, "11-26": 1.7014, "27-255": 1.2678},
    "fit": {"data": "a natural calibration split", "n": {"choice": 1500}},
}
BASIS = {"yes_no_board_count_accuracy_with_commit": [0.8129, 0.8258], "f": 0.8065, "without_commit": 0.645}


def build(**over):
    kw = {
        "temperatures": dict(TEMPS["temperatures"]),
        "choice_bands": dict(TEMPS["choice_bands"]),
        "noul_commit": True,
        "noul_commit_basis": BASIS,
        "fit": {"method": "minimum log loss", "data": "a natural calibration split"},
        "adapter": {
            "repository": "tachara-ai/adapter",
            "revision": "abc",
            "weights_sha256": ADAPTER_SHA,
            "config_sha256": CONFIG_SHA,
            "rank": 16,
            "alpha": 32,
        },
        "merge": {"precision": "fp32", "output": "fp8", "adapter_sha256": ADAPTER_SHA, "adapted_tensors": 840},
        "files": {"model-00001-of-00004.safetensors": hashlib.sha256(b"w").hexdigest()},
        "decisio_version": "0.20.0",
    }
    kw.update(over)
    return hub.tuned_config(FAM, **kw)


def write_repo(tmp_path, config, name="tuned"):
    d = tmp_path / name
    d.mkdir()
    (d / hub.FILE).write_text(json.dumps(config))
    (d / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    return d


def test_a1_the_file_is_the_profile_plus_what_the_merge_changes():
    c = build()
    assert c["schema"] == "decisio.decision_config/2" and c["base"] == "gemma-4-31b"
    assert hub.check_profile(c, FAM) == []  # the profile sections are the installed base's, unchanged
    assert c["weights"]["modified"] is True and c["weights"]["quantized_on_load"] is False
    assert "merged" in c["weights"]["method"]
    assert c["tuned"]["base"] == {"repository": FAM.model, "revision": FAM.revision}
    assert c["defaults"]["temperatures"] == TEMPS["temperatures"] and c["defaults"]["choice_bands"]["11-26"] == 1.7014
    assert c["defaults"]["noul_commit"] is True and c["defaults"]["noul_commit_basis"] == BASIS
    plain = build(choice_bands=None, noul_commit=False, noul_commit_basis=None)
    assert "choice_bands" not in plain["defaults"] and plain["defaults"]["noul_commit"] is False


def test_a2_an_older_decisio_and_a_mislabelled_file_are_refused(tmp_path):
    c = build()
    assert c["schema"] != hub.SCHEMA  # the /1 reader's condition (`config.get("schema") != SCHEMA`) refuses it
    assert hub.SCHEMA == "decisio.decision_config/1"  # and /1 is what every earlier release reads
    d = write_repo(tmp_path, {**c, "schema": "decisio.decision_config/3"})
    with pytest.raises(ValueError, match="decision_config/3.*reads decisio.decision_config/1 and"):
        hub.open_repository(str(d))
    one = hub.export_decision_config(FAM, "0.20.0")
    for key in ("tuned", "defaults"):
        d = write_repo(tmp_path, {**one, key: c[key]}, name=f"one-{key}")
        with pytest.raises(ValueError, match=f"has {key}: that is {hub.SCHEMA_TUNED}"):
            hub.open_repository(str(d))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["defaults"]["temperatures"].update(noul=-1.0), "noul temperature must be a positive number"),
        (lambda c: c["defaults"]["temperatures"].update(ranking=2.0), "unknown question type 'ranking'"),
        (lambda c: c["defaults"]["choice_bands"].update({"5-30": 2.0}), "overlap"),
        (lambda c: c["defaults"]["choice_bands"].update({"2-300": 2.0}), "2 <= lo <= hi <= 255"),
        (lambda c: c["defaults"].update(noul_commit="yes"), "noul_commit must be true or false"),
        (lambda c: c["defaults"].pop("noul_commit"), "noul_commit must be true or false"),
        (lambda c: c["defaults"].pop("noul_commit_basis"), "noul_commit_basis must say why"),
        (lambda c: c["defaults"].update(extra=1), "unknown keys"),
        (lambda c: c["defaults"]["fit"].pop("data"), "defaults.fit.data"),
        (lambda c: c["tuned"]["adapter"].update(weights_sha256="abc"), "tuned.adapter.weights_sha256 must be a sha256"),
        (lambda c: c["tuned"]["adapter"].pop("revision"), "tuned.adapter.revision"),
        (lambda c: c["tuned"]["merge"].update(precision="bf16"), "must be 'fp32'"),
        (lambda c: c["tuned"]["merge"].update(adapted_tensors=0), "adapted_tensors"),
        (lambda c: c["tuned"]["merge"].update(adapter_sha256="0" * 64), "merge report is of another adapter"),
        (lambda c: c["tuned"].update(files={}), "tuned.files must list"),
        (lambda c: c["tuned"]["files"].update(x="nothex"), "tuned.files['x'] must be a sha256"),
        (lambda c: c["weights"].update(modified=False), "weights.modified must be true"),
        (lambda c: c.update(base="gemma-4-12b"), "tuned repositories are made for gemma-4-31b"),
    ],
)
def test_a3_a_malformed_file_is_named_by_its_path(tmp_path, mutate, message):
    c = build()
    mutate(c)
    with pytest.raises(ValueError, match=re.escape(message)):
        hub.check_sections(c)
    with pytest.raises(ValueError, match=re.escape(message)):
        hub.open_repository(str(write_repo(tmp_path, c)))


def test_a4_the_digest_covers_what_the_repository_changes_and_the_facts_carry_it(tmp_path):
    c = build()
    repo = hub.open_repository(str(write_repo(tmp_path, c)))
    assert repo.tuned and repo.defaults == c["defaults"] and repo.defaults_sha256 == hub.defaults_sha256(c)
    facts = repo.facts()
    assert facts["schema"] == hub.SCHEMA_TUNED and facts["defaults"] == c["defaults"] and facts["tuned"] == c["tuned"]
    assert facts["defaults_sha256"] == repo.defaults_sha256
    same = build(decisio_version="0.20.1")  # the version it was exported for is not part of the digest
    assert hub.defaults_sha256(same) == hub.defaults_sha256(c)
    for other in (
        build(temperatures={**TEMPS["temperatures"], "noul": 4.2}),
        build(choice_bands={**TEMPS["choice_bands"], "11-26": 1.8}),
        build(noul_commit=False, noul_commit_basis=None),
        build(files={"model-00001-of-00004.safetensors": hashlib.sha256(b"other").hexdigest()}),
    ):
        assert hub.defaults_sha256(other) != hub.defaults_sha256(c)
    one = hub.open_repository(str(write_repo(tmp_path, hub.export_decision_config(FAM, "0.20.0"), name="one")))
    assert not one.tuned and one.defaults is None and one.defaults_sha256 is None
    assert one.facts()["schema"] == hub.SCHEMA and "defaults" not in one.facts()


def test_a5_the_profile_is_still_compared(tmp_path):
    c = build()
    c["temperatures"]["global"] = 1.0  # a repository cannot change how the base is read
    assert hub.check_profile(c, FAM) == ["temperatures.global"]
    repo = hub.open_repository(str(write_repo(tmp_path, c)))
    with pytest.raises(ValueError, match="temperatures.global"):
        hub.verify(repo, FAM)


def test_a6_the_commands(tmp_path, capsys):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "model-00001-of-00004.safetensors").write_bytes(b"w")
    (merged / "config.json").write_text("{}")
    (merged / hub.FILE).write_text("not part of the digests")
    temps = tmp_path / "decisio_adapter.json"
    temps.write_text(json.dumps(TEMPS))
    report = tmp_path / "MERGE_FP8.json"
    report.write_text(
        json.dumps({"adapter_sha256": ADAPTER_SHA, "adapter_scale": 2.0, "adapted_tensors": 840, "groups": 60})
    )
    out = tmp_path / "out.json"
    argv = [
        "tuned-config", "--base", "gemma-4-31b", "--temperatures", str(temps), "--adapter-dir", str(adapter),
        "--adapter-repository", "tachara-ai/adapter", "--adapter-revision", "abc", "--merge-report", str(report),
        "--weights-dir", str(merged), "--out", str(out), "--decisio-version", "0.20.0",
    ]  # fmt: skip
    with pytest.raises(SystemExit):  # the commit setting must be stated
        hub.main(argv)
    assert "--noul-commit" in capsys.readouterr().err
    assert hub.main([*argv, "--noul-commit", "--noul-commit-basis", "with the commit rule 0.81 against 0.80"]) == 0
    written = json.loads(out.read_text())
    assert written["defaults"]["temperatures"] == TEMPS["temperatures"] and written["defaults"]["noul_commit"] is True
    assert written["tuned"]["adapter"]["weights_sha256"] == ADAPTER_SHA and written["tuned"]["adapter"]["rank"] == 16
    assert set(written["tuned"]["files"]) == {"model-00001-of-00004.safetensors", "config.json"}
    assert written["defaults"]["fit"]["temperature_file_sha256"] == hashlib.sha256(temps.read_bytes()).hexdigest()
    capsys.readouterr()
    assert hub.main(["check", str(out), "--weights-dir", str(merged)]) == 0
    assert "agrees" in capsys.readouterr().out
    (merged / "model-00001-of-00004.safetensors").write_bytes(b"changed")
    assert hub.main(["check", str(out), "--weights-dir", str(merged)]) == 1
    assert "model-00001-of-00004.safetensors" in capsys.readouterr().out
    # a merge report of another adapter, and of a null merge
    report.write_text(json.dumps({"adapter_sha256": "0" * 64, "adapted_tensors": 840}))
    assert hub.main([*argv, "--no-noul-commit"]) == 1 and "another adapter" in capsys.readouterr().out
    report.write_text(json.dumps({"null": True, "adapter_sha256": ADAPTER_SHA, "adapted_tensors": 840}))
    assert hub.main([*argv, "--no-noul-commit"]) == 1 and "null merge" in capsys.readouterr().out
    assert Path(out).exists()
