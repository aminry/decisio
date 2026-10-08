# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The Hugging Face model repositories' `decision_config.json` and the loading path for `--base owner/name[@revision]`
(decisio.hub; plan in RLCD experiments/2026-10-07_sd_hf_model_repos/PLAN.md). No model, no network.

F1  the task fingerprint is what the server gave its task store before it was extracted: a registered task keeps
    matching
T1  the checked-in files in hub/ are what the profiles export; a profile change fails here until they are exported
    again (and a new revision of each repository is made for them)
T2  the profile section is compared: a changed temperature, class or serving default is named by its path
R1  what names a repository (owner/name, a directory) and what does not (a base's key, a typo)
R2  a file is read from a directory or fetched at a revision (the pinned one, the one given, else the current commit);
    a file of another schema, of an unknown base or missing is refused
S1  resolve_base on a repository: the base and its settings from the profile, the repository's weights as the
    checkpoint, a stored FP8 checkpoint not quantized again on load, tasks named for the source checkpoint; refused with
    --model, with a file that differs from the installed profile (naming the keys and the release), or with a
    checkpoint of another base
C1  the command line: export writes the files in hub/, check compares a file
E1  the server's main() on a repository: its task store is named for the source checkpoint (a task fitted under Google's
    weights serves on the copy), and the engine carries the repository for /health

    uv run pytest -q tests/unit/test_hub.py
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from decisio import hub
from decisio.families import BASES
from decisio.names import task_fingerprint
from decisio.serve.vllm_engine import resolve_base

HUB = Path(__file__).resolve().parents[2] / "hub"

# decisio 0.9.0's task fingerprints for each base on its defaults (taken from the server's own construction before it
# was moved to decisio.names.task_fingerprint)
FINGERPRINTS = {
    "qwen3.6-35b-a3b": '{"desnake_labels": true, "hide_index_keys": true, "model": "Qwen3.6-35B-A3B-FP8", "pad_to": "block", "pad_where": "front", "prompt_format": {"slot": "prefill", "tail": "spaced", "variants": "single"}, "served_name": "decisio-qwen3.6-35b-a3b-letters"}',  # noqa: E501
    "gemma-4-12b": '{"base": "gemma-4-12b", "desnake_labels": true, "hide_index_keys": true, "model": "gemma-4-12B-it", "pad_to": "none", "pad_where": "front", "prompt_format": {"slot": "template", "system_prompt": true, "tail": "spaced", "variants": "summed"}, "served_name": "decisio-gemma-4-12b-it-letters"}',  # noqa: E501
    "gemma-4-31b": '{"base": "gemma-4-31b", "desnake_labels": true, "hide_index_keys": true, "model": "gemma-4-31B-it", "pad_to": "none", "pad_where": "front", "prompt_format": {"slot": "template", "system_prompt": true, "tail": "spaced", "variants": "summed"}, "served_name": "decisio-gemma-4-31b-it-letters"}',  # noqa: E501
}
MODEL_TYPE = {"qwen3.6-35b-a3b": "qwen3_5_moe", "gemma-4-12b": "gemma4_unified", "gemma-4-31b": "gemma4"}


def repo_dir(tmp_path, base, mutate=None, name="repo"):
    """A directory that looks like a repository for a base: its file and a config.json with the base's model type."""
    d = tmp_path / name
    d.mkdir()
    config = hub.export_decision_config(BASES[base], "0.9.0")
    if mutate:
        mutate(config)
    (d / hub.FILE).write_text(json.dumps(config))
    (d / "config.json").write_text(json.dumps({"model_type": MODEL_TYPE[base]}))
    return d


def args_for(base, **kw):
    """The command line resolve_base reads, with nothing given but the base."""
    a = dict(
        backend="vllm",
        base=base,
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


@pytest.mark.parametrize("base", sorted(BASES))
def test_f1_the_fingerprint_is_what_the_server_always_gave(base):
    assert json.loads(FINGERPRINTS[base]) == hub.profile_of(BASES[base])["prompt"]["fingerprint"]
    fam = BASES[base]
    fp = task_fingerprint(
        served_name=fam.served_name,
        model=os.path.basename(fam.model),
        pad_to=fam.pad_to,
        pad_where="front",
        hide_index_keys=True,
        desnake_labels=True,
        prompt_format=json.loads(FINGERPRINTS[base])["prompt_format"],
        base=None if base == hub.QWEN else base,
    )
    assert fp == FINGERPRINTS[base]


def test_f1_the_optional_keys_enter_only_when_not_the_default():
    plain = dict(
        served_name="s", model="m", pad_to="none", pad_where="front", hide_index_keys=True, desnake_labels=True
    )
    assert set(json.loads(task_fingerprint(**plain))) == set(plain)
    more = json.loads(
        task_fingerprint(**plain, pad_policy="none", prompt_format={"tail": "compact"}, base="gemma-4-12b")
    )
    assert (
        more["pad_policy"] == "none" and more["prompt_format"] == {"tail": "compact"} and more["base"] == "gemma-4-12b"
    )
    assert "pad_policy" not in json.loads(task_fingerprint(**plain, pad_policy="always"))


@pytest.mark.parametrize("base", sorted(BASES))
def test_t1_the_files_in_hub_are_what_the_profiles_export(base):
    golden = json.loads((HUB / f"decision_config.{base}.json").read_text())
    now = hub.export_decision_config(BASES[base], golden["decisio"], golden["weights"]["quantization"])
    assert now == golden, (
        f"the {base} profile no longer exports hub/decision_config.{base}.json: export it again "
        "(python -m decisio.hub export --all --out hub --decisio-version <release>) and make a new revision of the "
        f"repository {hub.REPOSITORIES[base]}; differs at {hub.differences(golden, now)}"
    )
    assert hub.check_profile(golden, BASES[base]) == []


def test_t1_the_weights_of_each_repository():
    w = {b: hub.weights_of(b) for b in BASES}
    assert w["gemma-4-31b"]["modified"] and w["gemma-4-31b"]["dtype"] == "fp8_e4m3"
    assert "nothing was trained" in w["gemma-4-31b"]["method"]
    assert not w["gemma-4-12b"]["modified"] and w["gemma-4-12b"]["dtype"] == "bfloat16"
    assert not w[hub.QWEN]["modified"] and w[hub.QWEN]["quantization"] == "fp8"
    assert all(not v["quantized_on_load"] for v in w.values())  # a stored checkpoint is not quantized again
    assert hub.weights_of("gemma-4-31b", "fp8")["quantization"] == "fp8"  # the identity gate's fallback format


def test_t2_a_changed_profile_is_named_by_path():
    config = hub.export_decision_config(BASES["gemma-4-12b"], "0.9.0")
    assert hub.check_profile(config, BASES["gemma-4-12b"]) == []
    config["temperatures"]["choice"] = 1.0
    config["serving"]["register_boundary"] = "before"
    config["classes"]["hidden-readout"] = "Other"
    config["prompt"]["format"]["tail"] = "compact"
    assert hub.check_profile(config, BASES["gemma-4-12b"]) == [
        "classes.hidden-readout",
        "prompt.format.tail",
        "temperatures.choice",
        "serving.register_boundary",
    ]  # in the order of hub.PROFILE_KEYS
    # the repository's own facts are not compared, nor the release
    other = hub.export_decision_config(BASES["gemma-4-12b"], "9.9.9")
    other["source"]["revision"] = "x" * 40
    other["weights"]["modified"] = True
    assert hub.check_profile(other, BASES["gemma-4-12b"]) == []
    assert hub.differences({"a": [1, 2]}, {"a": [1, 3]}) == ["a"] and hub.differences({"a": 1}, {"b": 1}) == ["a", "b"]


def test_r1_what_names_a_repository(tmp_path):
    assert not hub.is_repository("gemma-4-31b") and not hub.is_repository("gemma-4-3lb")
    assert hub.is_repository("aminry/decisio-gemma-4-31b") and hub.is_repository("aminry/decisio-gemma-4-31b@abc")
    assert hub.is_repository(str(tmp_path))
    assert not hub.is_repository("nonsense")


def test_r2_a_directory_is_read(tmp_path):
    d = repo_dir(tmp_path, "gemma-4-12b")
    repo = hub.open_repository(str(d))
    assert (repo.base, repo.model, repo.revision) == ("gemma-4-12b", str(d), None)
    assert repo.facts()["exported_for_decisio"] == "0.9.0" and repo.facts()["weights"]["modified"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.update(schema="decisio.decision_config/2"), "schema"),
        (lambda c: c.update(base="gemma-9"), "names the base"),
    ],
)
def test_r2_a_file_this_decisio_cannot_read_is_refused(tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        hub.open_repository(str(repo_dir(tmp_path, "gemma-4-12b", mutate)))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no readable"):
        hub.open_repository(str(empty))


def test_r2_a_remote_repository_is_fetched_at_its_revision(tmp_path, monkeypatch):
    d = repo_dir(tmp_path, "gemma-4-31b")
    fetched = []

    def fake_download(name, filename, revision=None):
        fetched.append((name, filename, revision))
        return str(d / filename)

    class FakeApi:
        def model_info(self, name):
            return SimpleNamespace(sha="c" * 40)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    name = hub.REPOSITORIES["gemma-4-31b"]
    assert hub.open_repository(f"{name}@abc123").revision == "abc123"
    assert hub.open_repository(name, "from-flag").revision == "from-flag"
    assert hub.open_repository(f"{name}@abc123", "from-flag").revision == "abc123"  # the @ is the more specific
    monkeypatch.delitem(hub.PINNED_REVISIONS, name)
    assert hub.open_repository(name).revision == "c" * 40  # none given and none pinned: the current commit
    monkeypatch.setitem(hub.PINNED_REVISIONS, name, "d" * 40)
    assert hub.open_repository(name).revision == "d" * 40  # the release's pin
    assert [f[:2] for f in fetched] == [(name, hub.FILE)] * 5 and fetched[-1][2] == "d" * 40


@pytest.mark.parametrize("base", sorted(BASES))
def test_s1_a_repository_is_served_under_its_base(tmp_path, monkeypatch, base):
    d = repo_dir(tmp_path, base)
    a = args_for(str(d))
    fam, fmt = resolve_base(a)
    assert fam.key == base and a.base == base and a.model == str(d)
    assert a.served_name == BASES[base].served_name and a.temperature == BASES[base].temperature
    assert fam.quantization is None  # a stored checkpoint is not quantized again on load (the 31B's profile says fp8)
    assert a.repository["name"] == str(d) and a.repository["exported_for_decisio"] == "0.9.0"
    # tasks fitted under the source checkpoint serve on the repository, and the other way round
    assert a.model_identity == os.path.basename(BASES[base].model)
    monkeypatch.delitem(hub.PINNED_REVISIONS, hub.REPOSITORIES[base], raising=False)
    plain = args_for(base)
    resolve_base(plain)
    assert plain.repository is None and not hasattr(plain, "model_identity")


def test_s1_only_the_31b_changes_its_load_quantization(tmp_path):
    assert BASES["gemma-4-31b"].quantization == "fp8"
    assert (
        hub.serve_family(BASES["gemma-4-31b"], hub.open_repository(str(repo_dir(tmp_path, "gemma-4-31b")))).quantization
        is None
    )
    twelve = hub.open_repository(str(repo_dir(tmp_path, "gemma-4-12b", name="b")))
    assert hub.serve_family(BASES["gemma-4-12b"], twelve) is BASES["gemma-4-12b"]


def test_s1_a_file_that_differs_is_refused_with_the_keys_and_the_release(tmp_path):
    d = repo_dir(tmp_path, "gemma-4-12b", lambda c: c["temperatures"].update(choice=9.9))
    with pytest.raises(ValueError, match=r"exported for decisio 0\.9\.0.*differs.*at: temperatures\.choice"):
        resolve_base(args_for(str(d)))


def test_s1_refusals(tmp_path):
    d = repo_dir(tmp_path, "gemma-4-12b")
    with pytest.raises(ValueError, match="do not combine it with --model"):
        resolve_base(args_for(str(d), model="/some/checkpoint"))
    with pytest.raises(ValueError, match="neither a base"):
        resolve_base(args_for("nonsense"))
    wrong = repo_dir(tmp_path, "gemma-4-12b", name="w")
    (wrong / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))  # a checkpoint of another base
    with pytest.raises(ValueError, match="is a qwen3.6-35b-a3b checkpoint"):
        resolve_base(args_for(str(wrong)))
    mlx = repo_dir(tmp_path, "gemma-4-31b", name="m")
    with pytest.raises(ValueError, match="gemma-4-31b is served on vLLM only"):
        resolve_base(args_for(str(mlx), backend="mlx"))


def test_c1_the_command_line(tmp_path, capsys):
    assert hub.main(["export", "--all", "--out", str(tmp_path), "--decisio-version", "0.10.0"]) == 0
    for base in BASES:
        assert (tmp_path / f"decision_config.{base}.json").read_text() == (
            HUB / f"decision_config.{base}.json"
        ).read_text()
    good = str(tmp_path / "decision_config.gemma-4-12b.json")
    assert hub.main(["check", good]) == 0 and "agrees" in capsys.readouterr().out
    changed = json.loads(Path(good).read_text())
    changed["temperatures"]["global"] = 1.0
    Path(good).write_text(json.dumps(changed))
    assert hub.main(["check", good]) == 1 and "temperatures.global" in capsys.readouterr().out
    assert hub.main(["export", "--base", "gemma-4-12b", "--decisio-version", "0.10.0"]) == 0
    assert json.loads(capsys.readouterr().out)["base"] == "gemma-4-12b"


def test_e1_main_serves_a_repository(monkeypatch, tmp_path):
    import sys

    import uvicorn

    import decisio.serve.hf_letters as hf
    from decisio.serve import vllm_engine

    class Engine:
        adapters = {}

        def __init__(self, model, *a, **kw):
            self.pad_unit, self.model_name = None, model

        def facts(self):
            return {}

    served = {}
    monkeypatch.setattr(hf, "HFLettersEngine", Engine)
    monkeypatch.setattr(vllm_engine, "make_app", lambda engine, so: served.update(engine=engine, so=so) or object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)

    def start(*argv):
        monkeypatch.setattr(sys, "argv", ["decisio", "--backend", "hf", "--model-class", "view", *argv])
        vllm_engine.main()
        return served["engine"], served["so"].task_store.fingerprint

    d = repo_dir(tmp_path, "gemma-4-12b")
    engine, fingerprint = start("--base", str(d))
    assert engine.model_name == str(d) and engine.family.key == "gemma-4-12b"
    assert engine.repository["name"] == str(d) and engine.repository["weights"]["modified"] is False
    assert fingerprint == FINGERPRINTS["gemma-4-12b"]  # as for --base gemma-4-12b: its tasks serve on the copy
    assert start("--base", "gemma-4-12b")[1] == fingerprint  # (no network: the checkpoint id is never opened)
    assert start("--base", "gemma-4-12b")[0].repository is None
