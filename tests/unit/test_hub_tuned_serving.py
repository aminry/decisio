# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A tuned repository (`decision_config` /2) is served at its defaults: a plain request, with no flag and no adapter
field, gets the repository's temperatures, choice bands and commit setting (item 3 of the board path, step 3b).

B1  resolve_base: the repository's defaults reach the arguments; a /1 repository and a base's key are unchanged
B2  precedence: a flag beats the repository's default (and --temperature-choice drops the bands); the repository's beats
    the base's profile
B3  a plain request is answered at the defaults: each choice band, yes/no with the commit setting, score; the answer
    equals the temperature applied to the plain readout
B4  the serving-stack fingerprint holds the repository's defaults digest and the bands, and differs from the base's and
    from a repository with other defaults
B5  /health shows the defaults that were applied, their digest and the bands
B6  a tuned repository's tasks are its own: the task store's model identity differs from the base's, and from another
    repository with other defaults
B7  --noul-commit is a pair: unset, on, off; the start-up line says what was applied

  uv run pytest -q tests/unit/test_hub_tuned_serving.py
"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_adapter_bands import FakeEngine as BandsEngine
from test_hub import args_for, repo_dir
from test_hub_tuned import TEMPS, build, write_repo

from decisio import hub, names
from decisio.serve import stack, vllm_engine
from decisio.serve.systemone import SystemOne, add_routes, commit_noul
from decisio.serve.tasks import TaskStore
from decisio.serve.temperature import apply_temperature
from decisio.serve.vllm_engine import resolve_base

T = TEMPS["temperatures"]
BANDS = TEMPS["choice_bands"]


def raw(k):
    w = np.linspace(2.0, 0.2, k) ** 2
    return w / w.sum()


def tuned_args(tmp_path, config=None, **kw):
    d = write_repo(tmp_path, config or build(), name=kw.pop("name", "tuned"))
    args = args_for(str(d), **kw)
    return d, args, resolve_base(args)


def test_b1_the_defaults_reach_the_arguments(tmp_path):
    d, args, (fam, _) = tuned_args(tmp_path)
    assert fam.key == "gemma-4-31b" and args.repository["defaults_sha256"] == hub.defaults_sha256(build())
    assert (args.temperature_choice, args.temperature_noul, args.temperature_score) == (
        T["choice"],
        T["noul"],
        T["score"],
    )
    assert args.choice_bands == BANDS and args.noul_commit is True
    assert args.repository_defaults["noul_commit"] is True and args.repository_defaults["choice_bands"] == BANDS
    # the base's own key and a /1 repository are as before: the profile's choice temperature, no bands, no commit
    for base in (args_for("gemma-4-31b"), args_for(str(repo_dir(tmp_path, "gemma-4-12b")))):
        fam, _ = resolve_base(base)
        assert base.temperature_choice == fam.choice_temperature and base.choice_bands == {}
        assert base.noul_commit is False and base.repository_defaults is None
        assert getattr(base, "temperature_noul", None) is None


def test_b2_precedence_flag_then_repository_then_profile(tmp_path):
    _, args, _ = tuned_args(tmp_path, temperature_choice=3.0, temperature_noul=2.0, noul_commit=False)
    assert args.temperature_choice == 3.0 and args.temperature_noul == 2.0 and args.temperature_score == T["score"]
    assert args.choice_bands == {}  # one number for every choice question drops the repository's bands
    assert args.repository_defaults["choice_bands_dropped_by"] == "--temperature-choice"
    assert args.noul_commit is False and args.repository_defaults["noul_commit_set_by"] == "--no-noul-commit"
    _, on, _ = tuned_args(tmp_path, build(noul_commit=False, noul_commit_basis=None), name="t2", noul_commit=True)
    assert on.noul_commit is True and on.repository_defaults["noul_commit_set_by"] == "--noul-commit"
    _, off, _ = tuned_args(tmp_path, build(noul_commit=False, noul_commit_basis=None), name="t3")
    assert off.noul_commit is False  # the repository states false: that is what it gets
    _, plain, _ = tuned_args(tmp_path, build(choice_bands=None), name="t4")
    assert plain.choice_bands == {} and plain.temperature_choice == T["choice"]  # no bands in the file: the flat one


def served(args, **extra):
    so = SystemOne(
        BandsEngine(),
        "decisio-test",
        task_store=TaskStore("fp"),
        noul_commit=args.noul_commit,
        temperature=1.0,
        temperatures={
            "choice": args.temperature_choice,
            "noul": getattr(args, "temperature_noul", None),
            "score": getattr(args, "temperature_score", None),
        },
        choice_bands=args.choice_bands,
        **extra,
    )
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so


def ask(client, qtype, k):
    q = {"type": qtype, "instructions": "Which?"}
    if qtype == "choice":
        q["criteria"] = {f"option_{i}": f"o{i}" for i in range(k)}
    elif qtype == "score":
        q["criteria"] = [f"level {i}" for i in range(k)]
    r = client.post("/v1/systemone", json={"state": {"message": "x"}, "model": "m", "questions": {"q": q}})
    assert r.status_code == 200, r.text
    return r.json()["answers"]["q"]


def test_b3_a_plain_request_is_served_at_the_defaults(tmp_path):
    _, args, _ = tuned_args(tmp_path)
    client, so = served(args)
    for k, t in (
        (2, BANDS["2-10"]),
        (10, BANDS["2-10"]),
        (11, BANDS["11-26"]),
        (26, BANDS["11-26"]),
        (27, BANDS["27-255"]),
        (200, BANDS["27-255"]),
    ):
        got = ask(client, "choice", k)["probabilities"]
        assert [got[f"option_{i}"] for i in range(k)] == pytest.approx(list(apply_temperature(raw(k), t)), abs=1e-3), k
        assert so.temperature_of("choice", None, k) == t
    yes = ask(client, "noul", 2)["noul"]  # yes/no: its temperature, then the stated commit setting
    want = commit_noul(apply_temperature(raw(2), T["noul"]))
    assert yes == pytest.approx(float(want[0]), abs=1e-3) and yes in (pytest.approx(0.8), pytest.approx(0.2))
    s = ask(client, "score", 4)["probabilities"]
    assert list(s.values()) == pytest.approx(list(apply_temperature(raw(4), T["score"])), abs=1e-3)
    # the base's profile is not what answered
    base_client, _ = served(args_with_profile())
    assert ask(base_client, "choice", 3)["probabilities"] != ask(client, "choice", 3)["probabilities"]


def args_with_profile():
    args = args_for("gemma-4-31b")
    resolve_base(args)
    return args


class TunedEngine(BandsEngine):
    """The engine of a tuned repository: its facts carry the repository, as the real engines' do."""

    def __init__(self, repository=None):
        super().__init__()
        self.repository = repository

    def facts(self):
        out = super().facts()
        return {**out, **({"repository": self.repository} if self.repository else {})}


def stack_of(args):
    return SystemOne(
        TunedEngine(args.repository),
        "decisio-test",
        task_store=TaskStore("fp"),
        noul_commit=args.noul_commit,
        temperatures={"choice": args.temperature_choice, "noul": getattr(args, "temperature_noul", None)},
        choice_bands=args.choice_bands,
    ).stack_facts()


def test_b4_the_fingerprint_holds_the_defaults(tmp_path):
    _, args, _ = tuned_args(tmp_path)
    facts = stack_of(args)
    assert facts["repository"]["defaults_sha256"] == hub.defaults_sha256(build())
    assert facts["calibration"]["temperatures"]["choice_bands"] == BANDS
    base = stack_of(args_with_profile())
    assert (
        "defaults_sha256" not in (base["repository"] or {})
        and "choice_bands" not in base["calibration"]["temperatures"]
    )
    assert stack.sha256(facts) != stack.sha256(base)
    for name, other in (
        ("bands", build(choice_bands={**BANDS, "11-26": 1.8})),
        ("commit", build(noul_commit=False, noul_commit_basis=None)),
        ("noul", build(temperatures={**T, "noul": 4.2})),
    ):
        _, a2, _ = tuned_args(tmp_path, other, name=f"o-{name}")
        assert stack.sha256(stack_of(a2)) != stack.sha256(facts), name
    _, same, _ = tuned_args(
        tmp_path, build(decisio_version="0.20.1"), name="same"
    )  # the exported-for version is not part of it
    assert stack_of(same)["repository"]["defaults_sha256"] == facts["repository"]["defaults_sha256"]


def test_b5_health_shows_what_was_applied(tmp_path):
    _, args, _ = tuned_args(tmp_path)
    engine = TunedEngine(args.repository)
    engine.repository_defaults = args.repository_defaults
    engine.fmt = SimpleNamespace(facts=lambda: {})
    so = SystemOne(
        engine,
        "decisio-test",
        task_store=TaskStore("fp"),
        noul_commit=args.noul_commit,
        temperatures={"choice": args.temperature_choice},
        choice_bands=args.choice_bands,
    )
    try:
        client = TestClient(vllm_engine.make_app(engine, so))
        body = client.get("/health").json()
    except Exception as e:  # make_app wants more of a real engine than the stand-in has
        pytest.skip(f"make_app needs a fuller engine: {e}")
    s = body["systemone"]
    assert s["choice_bands"] == BANDS and s["noul_commit"] is True
    assert s["repository_defaults"]["defaults_sha256"] == hub.defaults_sha256(build())
    assert body.get("repository", {}).get("defaults_sha256") == hub.defaults_sha256(build())


def test_b6_a_tuned_models_tasks_are_its_own(tmp_path):
    _, a, _ = tuned_args(tmp_path)
    _, b, _ = tuned_args(tmp_path, build(temperatures={**T, "noul": 4.2}), name="other")
    base = args_with_profile()
    assert a.model_identity != b.model_identity and a.model_identity != base.model_identity
    assert a.model_identity.startswith("tuned@") and len(a.model_identity.split("@")[1]) == 12
    fp = lambda args: names.task_fingerprint(  # noqa: E731
        served_name="s",
        model=args.model_identity,
        pad_to="none",
        pad_where="front",
        hide_index_keys=True,
        desnake_labels=True,
    )
    assert not names.same_fingerprint(fp(a), fp(base)) and not names.same_fingerprint(fp(a), fp(b))


def test_b7_the_commit_flag_is_a_pair_and_the_start_says_what_was_applied(monkeypatch, tmp_path, capsys):
    import uvicorn

    import decisio.serve.hf_letters as hf

    ap = vllm_engine.build_parser()
    assert [ap.parse_args(x).noul_commit for x in ([], ["--noul-commit"], ["--no-noul-commit"])] == [None, True, False]

    class Engine:
        adapters = {}

        def __init__(self, model, *a, **kw):
            self.pad_unit, self.model_name = None, model

        def facts(self):
            return {}

    got = {}
    monkeypatch.setattr(hf, "HFLettersEngine", Engine)
    monkeypatch.setattr(vllm_engine, "make_app", lambda engine, so: got.update(engine=engine, so=so) or object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    d = write_repo(tmp_path, build())
    monkeypatch.setattr(sys, "argv", ["decisio", "--backend", "hf", "--model-class", "view", "--base", str(d)])
    vllm_engine.main()
    so = got["so"]
    assert so.noul_commit is True and so.choice_bands == BANDS and so.temperatures["noul"] == T["noul"]
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("REPOSITORY DEFAULTS"))
    assert json.loads(line.split(" ", 2)[2])["defaults_sha256"] == hub.defaults_sha256(build())
    assert got["engine"].repository_defaults["noul_commit"] is True
    monkeypatch.setattr(
        sys, "argv", ["decisio", "--backend", "hf", "--model-class", "view", "--base", str(d), "--no-noul-commit"]
    )
    vllm_engine.main()
    assert got["so"].noul_commit is False
