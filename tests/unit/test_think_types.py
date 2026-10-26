# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Thinking on some question types only, the long-state ceiling as a repository default, and the commit-think flavor
of a tuned repository (decisio.serve.think, decisio.hub, decisio.flavors):

Y1  the type list: checked, canonical order, refused when empty, unknown or repeated; `defaults.think.types`
Y2  a server with a list thinks on those types only; the others stay at one read and say why; /health, the response and
    the fingerprint carry the list only when there is one
Y3  the flags: `--think-types choice,score` reaches the arguments, a repository's list is the default, a flag beats it
Y4  `defaults.over_limit_tokens`: validated, written by tuned-config, the digest follows it, the repository's value is
    the server's ceiling and `--over-limit-tokens` beats it
Y5  the commit-think flavor: it must think and commit; the other flavors still must not think; the served name
    ends in it; the flavor object carries the thinking settings, so a flag that differs is an override and one that
    agrees is not

  uv run pytest -q tests/unit/test_think_types.py
"""

import hashlib
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_hub import args_for
from test_over_limit import LimitEngine, long_message
from test_over_limit import plain_rows as _plain_rows  # noqa: F401  (the row stand-ins, autouse there)
from test_think import ask, make

from decisio import flavors as F
from decisio import hub
from decisio.serve import think as T
from decisio.serve import vllm_engine
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore
from decisio.serve.vllm_engine import resolve_base

TUNED = ("tuned_config",)


def test_y1_the_type_list_is_checked():
    assert T.check_types(["score", "choice"], "x") == ["choice", "score"]
    assert T.check_settings({"factor": 0.5, "budget": 256, "types": ["choice"]}, "d") == {
        "factor": 0.5,
        "budget": 256,
        "types": ["choice"],
    }
    for bad in ([], "choice", ["choice", "choice"], ["choice", "ranking"], None, 3):
        with pytest.raises(ValueError, match="types"):
            T.check_settings({"factor": 0.5, "types": bad}, "d")


def test_y2_only_the_listed_types_think():
    client, so, eng = make(think=True, factor=1.0, think_types=["choice"])
    r = ask(client, "0:0.5:x:1", qtype="noul")  # a yes/no that is unsure under factor 1.0 would think without the list
    assert r.json()["thinking"]["per_question"]["q"] == {"thought": False, "why": T.NOT_THINKING_TYPE}
    assert eng.calls == []
    r = ask(client, "0:0.5:x:1", qtype="choice")
    assert r.json()["thinking"]["per_question"]["q"]["thought"] is True
    assert r.json()["thinking"]["types"] == ["choice"]
    plain, so2, _ = make(think=True)
    assert "types" not in ask(plain, "0:0.5:x:1").json()["thinking"]
    assert so.think_facts()["types"] == ["choice"] and "types" not in so2.think_facts()
    assert so.think_facts() != so2.think_facts()  # the fingerprint follows the list, and only when there is one


def test_y2_a_type_outside_the_list_is_refused_at_construction():
    with pytest.raises(ValueError, match="types"):
        make(think=True, think_types=["ranking"])


def test_y3_the_flag_and_the_repository_default(tmp_path):
    a = vllm_engine.build_parser().parse_args(["--think", "--think-types", "choice,score"])
    assert a.think_types == "choice,score"
    from test_hub_tuned import build, write_repo

    d = write_repo(tmp_path, build(think={"factor": 0.5, "budget": 256, "types": ["choice"]}), name="t")
    args = args_for(str(d))
    resolve_base(args)
    assert args.think_types == ["choice"] and args.repository_defaults["think"]["types"] == ["choice"]
    args = args_for(str(d), think_types="choice,score")  # a flag beats the repository
    resolve_base(args)
    assert args.think_types == ["choice", "score"]
    plain = write_repo(tmp_path, build(think={"factor": 0.5}), name="p")
    args = args_for(str(plain))
    resolve_base(args)
    assert getattr(args, "think_types", None) is None
    bad = args_for(str(d), think_types="choice,ranking")
    with pytest.raises(SystemExit, match="types"):
        resolve_base(bad)


FAM12 = hub.BASES["gemma-4-12b"]


def build12(**over):
    import hashlib

    from test_hub_tuned import ADAPTER_SHA, BASIS, CONFIG_SHA, TEMPS

    MERGE = {
        "precision": "fp32",
        "output": "bf16",
        "rounding": "stochastic",
        "seed": 0,
        "adapter_sha256": ADAPTER_SHA,
        "adapted_tensors": 328,
    }

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
        "files": {"model-00001-of-00006.safetensors": hashlib.sha256(b"w").hexdigest()},
        "decisio_version": "0.34.0",
        "layout": "header",
    }
    kw.update(over)
    return hub.tuned_config(FAM12, **kw)


THINK = {"factor": 0.5, "budget": 256, "types": ["choice"]}
FLAVOR = {"name": "commit-think", "served_name": "Decisio-12b-tuned-commit-think"}


def test_y4_over_limit_tokens_is_a_validated_repository_default():
    c = build12(over_limit="truncate", over_limit_tokens=45000)
    assert c["defaults"]["over_limit_tokens"] == 45000
    assert hub.defaults_sha256(c) != hub.defaults_sha256(build12(over_limit="truncate"))
    for bad in (0, 100, 2_000_000, "45000", True, 4.5e4):
        c["defaults"]["over_limit_tokens"] = bad
        with pytest.raises(ValueError, match="over_limit_tokens"):
            hub.check_defaults(c["defaults"])


def repo_dir(tmp_path, **over):
    from test_hub_tuned import write_repo

    d = write_repo(tmp_path, build12(**over), name="r")
    (d / "config.json").write_text(json.dumps({"model_type": "gemma4_unified"}))
    return d


def test_y4_the_repositorys_ceiling_is_the_servers_and_the_flag_beats_it(tmp_path):
    d = repo_dir(tmp_path, over_limit="truncate", over_limit_tokens=45000)
    args = args_for(str(d))
    resolve_base(args)
    assert (args.over_limit, args.over_limit_tokens) == ("truncate", 45000)
    assert args.repository_defaults["over_limit_tokens"] == 45000
    args = args_for(str(d), over_limit_tokens=30000)
    resolve_base(args)
    assert args.over_limit_tokens == 30000
    assert args.repository_defaults["over_limit_tokens_set_by"] == "--over-limit-tokens 30000"


def test_y4_the_command_writes_it(tmp_path, capsys):
    from test_hub_tuned import ADAPTER_SHA, TEMPS

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    temps = tmp_path / "decisio_adapter.json"
    temps.write_text(json.dumps(TEMPS))
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "model-00001-of-00006.safetensors").write_bytes(b"w")
    (weights / "config.json").write_text("{}")
    report = tmp_path / "MERGE_BF16.json"
    report.write_text(
        json.dumps(
            {
                "rounding": "stochastic",
                "seed": 0,
                "adapter_sha256": hashlib.sha256(b"adapter weights").hexdigest(),
                "adapted_tensors": 328,
                "adapter_scale": 2.0,
                "null": False,
            }
        )
    )
    out = tmp_path / "decision_config.json"
    argv = [
        "tuned-config", "--base", "gemma-4-12b", "--temperatures", str(temps), "--adapter-dir", str(adapter),
        "--adapter-revision", "abc", "--merge-report", str(report), "--weights-dir", str(weights), "--out", str(out),
        "--decisio-version", "0.34.0", "--noul-commit", "--noul-commit-basis", "because", "--layout", "header",
        "--flavor", "commit-think", "--served-name", "Decisio-12b-tuned-commit-think",
        "--over-limit", "truncate", "--over-limit-tokens", "45000", "--think-factor", "0.5", "--think-budget", "256",
        "--think-types", "choice",
    ]  # fmt: skip
    assert hub.main(argv) == 0
    defaults = json.loads(out.read_text())["defaults"]
    assert defaults["over_limit_tokens"] == 45000 and defaults["think"] == THINK
    assert hub.main(["check", str(out), "--weights-dir", str(weights)]) == 0
    assert ADAPTER_SHA != ""
    i = argv.index("--think-factor")
    with pytest.raises(SystemExit):  # --think-types without --think-factor
        hub.main(argv[:i] + argv[i + 2 :])


def test_y5_the_commit_think_flavor_is_checked():
    c = build12(flavor=FLAVOR, think=THINK, over_limit="truncate", over_limit_tokens=45000)
    hub.check_sections(c)
    with pytest.raises(ValueError, match="commit-think flavor thinks"):
        build12(flavor=FLAVOR)  # no think
    with pytest.raises(ValueError, match="needs True and off"):
        build12(flavor=FLAVOR, think=THINK, noul_commit=False, noul_commit_basis=None)
    with pytest.raises(ValueError, match="does not think"):
        build12(flavor={"name": "commit", "served_name": "Decisio-12b-tuned-commit"}, think=THINK)
    with pytest.raises(ValueError, match="ending in -commit-think"):
        build12(flavor={"name": "commit-think", "served_name": "Decisio-12b-tuned-think"}, think=THINK)


def test_y5_the_flavor_object_carries_the_thinking_settings(tmp_path):
    d = repo_dir(tmp_path, flavor=FLAVOR, think=THINK, over_limit="truncate", over_limit_tokens=45000)
    repo = hub.open_repository(str(d))
    f = F.from_repository(repo)
    assert (f.name, f.served_name, f.noul_commit, f.think, f.think_factor, f.think_budget, f.think_types) == (
        "commit-think", "Decisio-12b-tuned-commit-think", True, True, 0.5, 256, ["choice"]
    )  # fmt: skip
    args = args_for(str(d))
    resolve_base(args)
    assert args.flavor_report["name"] == "commit-think" and args.flavor_report["overrides"] == {}
    assert (args.think, args.think_factor, args.think_budget, args.think_types) == (True, 0.5, 256, ["choice"])
    assert args.served_name == "Decisio-12b-tuned-commit-think"
    assert F.siblings(args.flavor_report) == [f"Decisio-12b-tuned-{n}" for n in F.FLAVOR_NAMES]
    # a flag that agrees is no override; one that differs is
    same = args_for(str(d), think_factor=0.5, think_types=["choice"])
    resolve_base(same)
    assert same.flavor_report["overrides"] == {}
    other = args_for(str(d), think_factor=0.6)
    resolve_base(other)
    assert other.flavor_report["overrides"]["think_factor"] == {"flavor": 0.5, "served": 0.6}


# ---- a cut state is answered by the single pass ------------------------------------------------------------------


def test_y6_a_truncated_state_is_answered_by_the_single_pass_and_says_so():
    """Check (a) of the guard: when the server cut the state to its ceiling, no thinking read is made whatever the first
    read's confidence: `thinking` is absent (null), the answer is the single pass's, and /health counts it."""

    class ThinkLimit(LimitEngine):
        def __init__(self, limit):
            super().__init__(limit)
            self.calls = []

        def can_think(self):
            return True

        def think(self, state, questions, adapter=None, budget=512, subset=None):
            subset = list(range(len(questions))) if subset is None else subset
            self.calls.append(subset)
            logps = [np.log(np.full(len(questions[i]["options"]), 1 / 3)) for i in subset]
            return logps, [10] * len(subset), {}

    def server(limit, think=True):
        eng = ThinkLimit(limit)
        so = SystemOne(
            eng, "decisio-test", task_store=TaskStore("fp"), over_limit="truncate", think=think, think_factor=1.0,
            think_types=["choice"],
        )  # fmt: skip
        app = FastAPI()
        add_routes(app, so)
        return TestClient(app), so, eng

    body = {
        "state": {"message": long_message(top=1, conf=0.5)},  # unsure at factor 1.0 (0.5 < 0.75)
        "model": "m",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {"option_0": "a", "option_1": "b", "option_2": "c"},
            }
        },
    }
    client, so, eng = server(limit=200)
    r = client.post("/v1/systemone", json=body).json()
    assert r["state_truncated"]["part"] == "start" and r.get("thinking") is None
    assert eng.calls == []  # no thinking read was made
    plain, _, _ = server(limit=200, think=False)
    off = plain.post("/v1/systemone", json=body).json()
    assert r["answers"] == off["answers"]  # the single pass's answer, exactly
    health = so.thought
    assert health["skipped_truncated"] == 1 and health["requests"] == 0
    # a request that asks for thinking itself is skipped the same way
    off_server, so2, eng2 = server(limit=200, think=False)
    asked = off_server.post("/v1/systemone", json={**body, "think": True}).json()
    assert asked.get("thinking") is None and eng2.calls == []
    # the same unsure question on a state that fits is read again after reasoning
    fits = {**body, "state": {"message": "1:0.5:x w0 w1"}}
    client3, so3, eng3 = server(limit=10_000)
    r3 = client3.post("/v1/systemone", json=fits).json()
    assert (
        "state_truncated" not in r3 and r3["thinking"]["per_question"]["q"]["thought"] is True and eng3.calls == [[0]]
    )
    assert "skipped_truncated" not in so3.thought  # absent until there is one: a server without it is what it was
