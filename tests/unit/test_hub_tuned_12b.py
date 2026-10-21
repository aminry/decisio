# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A tuned 12B repository: the `/2` schema for the base that is stored in bf16, and everything the 31B's has
(defaults served, bands, the commit setting, over_limit, think), read from a repository directory.

C1  the file: bf16 weights (not FP8), the merge facts of `hub_bf16` (output, stochastic rounding, seed), the 12B's
    profile unchanged; a 12B with an FP8 merge (named as gemma-4-12b-fp8's), a 31B with a bf16 merge, a wrong weights
    dtype and a missing seed are refused (the 12B in FP8: test_hub_tuned_12b_fp8.py)
C2  `tuned-config --base gemma-4-12b` writes a file `check` accepts, from a real (tiny) merge report
C3  served from a directory: the family is the 12B, the repository's temperatures, bands, commit setting, over_limit
    and think reach the arguments, a flag beats them, the identity is the repository's and differs from the 31B's with
    the same defaults, and the context ceiling is still the 12B's
C4  an older decisio refuses it by its schema; the defaults digest follows over_limit and think

  uv run pytest -q tests/unit/test_hub_tuned_12b.py
"""

import hashlib
import json

import pytest
from test_hub import args_for
from test_hub_bf16_merge import adapter, targets
from test_hub_fp8 import checkpoint
from test_hub_tuned import ADAPTER_SHA, BASIS, CONFIG_SHA, TEMPS, build, write_repo

from decisio import hub, hub_bf16
from decisio.families import BASES
from decisio.serve.vllm_engine import resolve_base

FAM = BASES["gemma-4-12b"]
MERGE = {
    "precision": "fp32",
    "output": "bf16",
    "rounding": "stochastic",
    "seed": 7,
    "adapter_sha256": ADAPTER_SHA,
    "adapted_tensors": 410,
}


def build12(**over):
    kw = {
        "temperatures": dict(TEMPS["temperatures"]),
        "choice_bands": dict(TEMPS["choice_bands"]),
        "noul_commit": True,
        "noul_commit_basis": BASIS,
        "fit": {"method": "minimum log loss", "data": "a natural calibration split"},
        "adapter": {
            "repository": "tachara-ai/adapter-12b",
            "revision": "abc",
            "weights_sha256": ADAPTER_SHA,
            "config_sha256": CONFIG_SHA,
            "rank": 16,
            "alpha": 32,
        },
        "merge": dict(MERGE),
        "files": {"model.safetensors": hashlib.sha256(b"w").hexdigest()},
        "decisio_version": "0.21.0",
        "over_limit": "truncate",
        "think": {"factor": 0.79},
    }
    kw.update(over)
    return hub.tuned_config(FAM, **kw)


def test_c1_the_file_is_bf16_with_the_rounding_stated():
    c = build12()
    assert c["schema"] == hub.SCHEMA_TUNED and c["base"] == "gemma-4-12b"
    assert hub.check_profile(c, FAM) == []
    assert c["weights"] == {**c["weights"], "modified": True, "dtype": "bfloat16", "quantization": None}
    assert "stochastically" in c["weights"]["method"] and c["tuned"]["merge"]["seed"] == 7
    assert c["defaults"]["over_limit"] == "truncate" and c["defaults"]["think"] == {"factor": 0.79}
    assert hub.TUNED_BASES == ("gemma-4-31b", "gemma-4-12b", "gemma-4-12b-fp8")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m.update(output="fp8 compressed-tensors"), "a tuned 12B in FP8 is the base gemma-4-12b-fp8"),
        (lambda m: m.update(rounding="nearest"), "must say output 'bf16' and rounding 'stochastic'"),
        (lambda m: m.pop("seed"), "tuned.merge.seed must be the rounding's seed"),
        (lambda m: m.update(seed=True), "tuned.merge.seed must be the rounding's seed"),
        (lambda m: m.update(precision="bf16"), "must be 'fp32'"),
    ],
)
def test_c1_a_12b_merge_that_is_not_the_stochastic_bf16_one_is_refused(mutate, message):
    merge = dict(MERGE)
    mutate(merge)
    with pytest.raises(ValueError, match=message):
        build12(merge=merge)


def test_c1_the_two_bases_do_not_take_each_others_merges():
    with pytest.raises(ValueError, match="must be fp8"):
        build(merge={"precision": "fp32", "output": "bf16", "adapter_sha256": ADAPTER_SHA, "adapted_tensors": 840})
    c = build12()
    c["weights"]["dtype"] = "fp8_e4m3"
    with pytest.raises(ValueError, match="a tuned gemma-4-12b is stored as bfloat16"):
        hub.check_sections(c)
    c = build()
    c["weights"]["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="a tuned gemma-4-31b is stored as fp8_e4m3"):
        hub.check_sections(c)


def test_c2_tuned_config_writes_a_file_check_accepts_from_a_real_merge_report(tmp_path):
    base, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=3, scale_alpha=0.02)
    out = tmp_path / "merged"
    hub_bf16.merge(base, ad, out, seed=11)
    (tmp_path / "temps.json").write_text(json.dumps(TEMPS))
    rc = hub.main(
        [
            "tuned-config", "--base", "gemma-4-12b", "--temperatures", str(tmp_path / "temps.json"), "--noul-commit",
            "--noul-commit-basis", json.dumps(BASIS), "--think-factor", "0.79", "--over-limit", "truncate",
            "--adapter-dir", str(ad), "--adapter-repository", "o/a", "--adapter-revision", "0" * 40,
            "--merge-report", str(out / "MERGE_BF16.json"), "--weights-dir", str(out), "--out", str(out / hub.FILE),
        ]
    )  # fmt: skip
    assert rc == 0
    config = json.loads((out / hub.FILE).read_text())
    assert config["base"] == "gemma-4-12b" and config["tuned"]["merge"]["seed"] == 11
    assert config["defaults"]["think"] == {"factor": 0.79} and config["defaults"]["over_limit"] == "truncate"
    assert hub.main(["check", str(out / hub.FILE), "--weights-dir", str(out)]) == 0
    (out / "tokenizer.json").write_text("changed")
    assert hub.main(["check", str(out / hub.FILE), "--weights-dir", str(out)]) == 1  # a changed file is found


def write12(tmp_path, config, name):
    d = write_repo(tmp_path, config, name=name)
    (d / "config.json").write_text(json.dumps({"model_type": "gemma4_unified"}))  # the 12B's model type
    return d


def tuned12(tmp_path, config=None, **kw):
    d = write12(tmp_path, config or build12(), name=kw.pop("name", "t12"))
    args = args_for(str(d), **kw)
    return d, args, resolve_base(args)


def test_c3_served_from_a_directory_at_the_repositorys_defaults(tmp_path):
    d, args, (fam, _) = tuned12(tmp_path)
    assert fam.key == "gemma-4-12b" and fam.context_ceiling == 65536
    t = TEMPS["temperatures"]
    assert (args.temperature_choice, args.temperature_noul, args.temperature_score) == (
        t["choice"],
        t["noul"],
        t["score"],
    )
    assert args.choice_bands == TEMPS["choice_bands"] and args.noul_commit is True
    assert (
        args.over_limit == "truncate" and args.think is True and args.think_factor == 0.79 and args.think_budget == 512
    )
    assert args.repository_defaults["think"] == {"factor": 0.79}
    assert args.repository["defaults_sha256"] == hub.defaults_sha256(build12())
    # the identity is the repository's own, and differs from a 31B repository with the same defaults
    d31 = write_repo(tmp_path, build(), name="t31")
    args31 = args_for(str(d31))
    resolve_base(args31)
    assert args.model_identity.startswith("t12@") and args.model_identity != args31.model_identity


def test_c3_a_flag_beats_the_repository(tmp_path):
    _, args, _ = tuned12(tmp_path, over_limit="refuse", think=False, temperature_choice=3.0)
    assert args.over_limit == "refuse" and args.think is False and args.choice_bands == {}
    assert args.repository_defaults["over_limit_set_by"] == "--over-limit refuse"
    _, args, _ = tuned12(tmp_path, name="t12b", think_factor=0.6, think_budget=128)
    assert (args.think, args.think_factor, args.think_budget) == (True, 0.6, 128)


def test_c4_an_older_decisio_refuses_it_and_the_digest_follows_the_new_keys(tmp_path):
    c = build12()
    assert c["schema"] != hub.SCHEMA
    plain = build12(over_limit=None, think=None)
    assert "over_limit" not in plain["defaults"] and "think" not in plain["defaults"]
    assert (
        len({hub.defaults_sha256(x) for x in (c, plain, build12(think={"factor": 0.8}), build12(over_limit="refuse"))})
        == 4
    )


def test_c5_a_tuned_repositorys_file_says_nothing_about_data(tmp_path):
    """The fit tools write `data` (what the items were), `n` (how many) and `detail` into a temperature file; none of it
    reaches the repository's file, for either base, through the function or the command."""
    fit = {
        "method": "minimum log loss",
        "objective": "log loss",
        "data": "a natural calibration split of public sources",
        "n": {"choice": 1500},
        "detail": {"sources": ["x", "y"]},
        "kappa_on_raw_readout": {"choice": 0.6},
    }
    for make, extra in ((build12, {}), (build, {})):
        c = make(fit=dict(fit), **extra)
        assert set(c["defaults"]["fit"]) == {"method", "objective", "kappa_on_raw_readout"}
        text = json.dumps(c).lower()
        assert not any(w in text for w in ("calibration split", "public sources", '"n":', "sources"))
    # a file with fit.data (0.20.0 and 0.21.x wrote it) is still read
    old = build12()
    old["defaults"]["fit"]["data"] = "a note"
    hub.check_defaults(old["defaults"])
    # the command: a temperature file whose fit carries data, n and detail
    temps = json.loads(json.dumps(TEMPS))
    temps["fit"] = {**fit, "data": "a natural calibration split of public sources"}
    (tmp_path / "t.json").write_text(json.dumps(temps))
    base, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=3, scale_alpha=0.02)
    out = tmp_path / "m"
    hub_bf16.merge(base, ad, out)
    assert (
        hub.main(
            [
                "tuned-config",
                "--base",
                "gemma-4-12b",
                "--temperatures",
                str(tmp_path / "t.json"),
                "--noul-commit",
                "--noul-commit-basis",
                json.dumps(BASIS),
                "--adapter-dir",
                str(ad),
                "--adapter-repository",
                "o/a",
                "--adapter-revision",
                "0" * 40,
                "--merge-report",
                str(out / "MERGE_BF16.json"),
                "--weights-dir",
                str(out),
                "--out",
                str(out / hub.FILE),
            ]
        )  # fmt: skip
        == 0
    )
    written = json.loads((out / hub.FILE).read_text())["defaults"]["fit"]
    assert (
        set(written) <= set(hub.FIT_KEYS) and "data" not in written and "n" not in written and "detail" not in written
    )
