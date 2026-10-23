# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Flavors: named configurations of the same weights (docs/design/flavors.md).

F1  the names: nine served names, capitalised, the 31B without a precision and the 12B with one; unknown names, bases
    without flavors and a flavor whose defining value is missing are refused
F2  `--flavor` through `resolve_base`: the flavor's settings reach the arguments and the prompt format, every flavor on
    the header layout, the server is named for it, pending values are listed
F3  overrides: a flag wins on any flavor and is recorded (not refused); the header and the fingerprint say so
F4  a server with no flavor is what it was: no flavor in the report, the fingerprint, the headers or the model it echoes
F5  the server: /v1/models and the response name the flavor in capitals, a request's model matches in any case, another
    flavor's name is refused, the header and /health carry the flavor, certificates separate per flavor
F6  the flag is in the parser and refused with a tuned repository

  uv run pytest -q tests/unit/test_flavors.py
"""

import pytest
from test_hub import args_for
from test_hub_tuned import build, write_repo
from test_think import ask, make

from decisio import flavors
from decisio.families import BASES
from decisio.serve import stack, vllm_engine
from decisio.serve.vllm_engine import resolve_base

NAMES31 = ["Decisio-31b-calibrated", "Decisio-31b-thinking-commit", "Decisio-31b-read-enhanced"]
NAMES12 = ["Decisio-12b-calibrated", "Decisio-12b-thinking-commit", "Decisio-12b-read-enhanced"]
NAMES12FP8 = ["Decisio-12b-fp8-calibrated", "Decisio-12b-fp8-thinking-commit", "Decisio-12b-fp8-read-enhanced"]
SERVING = [("gemma-4-31b", "thinking-commit"), ("gemma-4-31b", "read-enhanced")] + [
    (b, f) for b in ("gemma-4-12b", "gemma-4-12b-fp8") for f in ("thinking-commit", "read-enhanced")
]


# ---- F1 -------------------------------------------------------------------------------------------------------------


def test_f1_the_nine_served_names():
    assert flavors.served_names("gemma-4-31b") == NAMES31
    assert flavors.served_names("gemma-4-12b") == NAMES12
    assert flavors.served_names("gemma-4-12b-fp8") == NAMES12FP8
    assert flavors.served_names("qwen3.6-35b-a3b") == []
    assert flavors.FLAVOR_NAMES == ("calibrated", "thinking-commit", "read-enhanced")


def test_f1_the_31b_names_carry_no_precision_and_the_12b_names_do():
    assert not any("fp8" in n.lower() for n in NAMES31)
    assert all("fp8" in n for n in NAMES12FP8) and not any("fp8" in n for n in NAMES12)


def test_f1_refusals():
    with pytest.raises(flavors.FlavorError, match="not one of"):
        flavors.flavor_of("gemma-4-31b", "honest")
    with pytest.raises(flavors.FlavorError, match="no flavors"):
        flavors.flavor_of("qwen3.6-35b-a3b", "read-enhanced")
    with pytest.raises(flavors.FlavorError, match="not fitted yet"):
        flavors.flavor_of("gemma-4-31b", "calibrated")  # its defining value does not exist yet


def test_f1_every_flavor_uses_the_header_layout():
    for base in flavors.FLAVORS:
        for f in flavors.FLAVORS[base].values():
            assert f.layout == "header"


# ---- F2 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("base, name", SERVING)
def test_f2_the_flavor_reaches_the_arguments_and_the_format(base, name):
    args = args_for(base, flavor=name)
    fam, fmt = resolve_base(args)
    f = flavors.FLAVORS[base][name]
    assert args.served_name == f.served_name == f"{flavors.STEMS[base]}-{name}"
    assert fmt.layout == "header"
    assert args.layout == "header"
    if name == "thinking-commit":
        assert args.noul_commit is True and args.think is True and fmt.read_twice_under == 0
    else:
        assert not args.noul_commit and not args.think and fmt.read_twice_under == flavors.READ_TWICE_N
    r = args.flavor_report
    assert r["name"] == name and r["served_name"] == args.served_name and r["overrides"] == {} and r["pending"]
    assert flavors.header_value(r) == f"{name}+pending"


def test_f2_the_thinking_defaults_are_the_servers_own_when_the_flavor_names_none():
    args = args_for("gemma-4-31b", flavor="thinking-commit")
    resolve_base(args)
    assert (args.think_factor, args.think_budget) == (1.0, 512)  # resolve fills what the flavor left


def test_f2_the_base_profile_fills_the_temperatures_the_flavor_does_not_set():
    args = args_for("gemma-4-31b", flavor="read-enhanced")
    fam, _ = resolve_base(args)
    assert (args.temperature, args.temperature_choice) == (fam.temperature, fam.choice_temperature)


def test_f2_read_enhanced_and_thinking_commit_differ_only_where_they_should():
    a, b = args_for("gemma-4-31b", flavor="read-enhanced"), args_for("gemma-4-31b", flavor="thinking-commit")
    resolve_base(a)
    resolve_base(b)
    assert a.read_twice_under != b.read_twice_under and a.noul_commit != b.noul_commit and a.think != b.think
    assert a.layout == b.layout == "header"


# ---- F3 -------------------------------------------------------------------------------------------------------------


def test_f3_a_flag_wins_and_is_recorded_not_refused():
    args = args_for("gemma-4-31b", flavor="thinking-commit", noul_commit=False, layout="plain", temperature_noul=3.5)
    _, fmt = resolve_base(args)
    assert args.noul_commit is False and fmt.layout == "plain" and args.temperature_noul == 3.5
    o = args.flavor_report["overrides"]
    assert o["noul_commit"] == {"flavor": True, "served": False}
    assert o["layout"] == {"flavor": "header", "served": "plain"}
    assert o["temperature_noul"] == {"flavor": None, "served": 3.5}
    assert flavors.header_value(args.flavor_report) == "thinking-commit+overrides+pending"


def test_f3_a_flag_that_equals_the_flavor_is_not_an_override():
    args = args_for("gemma-4-31b", flavor="thinking-commit", noul_commit=True, layout="header", think=True)
    resolve_base(args)
    assert args.flavor_report["overrides"] == {}


def test_f3_the_server_name_can_be_overridden_and_says_so():
    args = args_for("gemma-4-31b", flavor="read-enhanced", served_name="my-di")
    resolve_base(args)
    assert args.served_name == "my-di"
    assert args.flavor_report["overrides"]["served_name"] == {"flavor": "Decisio-31b-read-enhanced", "served": "my-di"}


def test_f3_read_twice_can_be_turned_up_or_off_on_read_enhanced():
    args = args_for("gemma-4-31b", flavor="read-enhanced", read_twice_under=3000, read_twice_max_questions=8)
    _, fmt = resolve_base(args)
    assert (fmt.read_twice_under, fmt.read_twice_max_questions) == (3000, 8)
    assert set(args.flavor_report["overrides"]) == {"read_twice_under", "read_twice_max_questions"}


def test_f3_overrides_separate_the_fingerprint():
    a, b = (
        args_for("gemma-4-31b", flavor="thinking-commit"),
        args_for("gemma-4-31b", flavor="thinking-commit", noul_commit=False),
    )
    resolve_base(a)
    resolve_base(b)
    assert flavors.fingerprint_facts(a.flavor_report) != flavors.fingerprint_facts(b.flavor_report)


# ---- F4 -------------------------------------------------------------------------------------------------------------


def test_f4_a_server_with_no_flavor_is_what_it_was():
    args = args_for("gemma-4-31b")
    fam, fmt = resolve_base(args)
    assert args.flavor_report is None and args.served_name == BASES["gemma-4-31b"].served_name
    client, so, _ = make()
    assert so.flavor is None and "flavor" not in so.stack_facts()["serving"]
    r = ask(client, "0:0.9:x:1")
    assert r.status_code == 200 and "x-decisio-flavor" not in r.headers
    assert r.json()["model"] == "m"  # the request's own model string, as before


# ---- F5 -------------------------------------------------------------------------------------------------------------


def flavored(name="read-enhanced", **flags):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from test_think import ThinkEngine

    from decisio.serve.systemone import SystemOne, add_routes
    from decisio.serve.tasks import TaskStore

    args = args_for("gemma-4-31b", flavor=name, **flags)
    resolve_base(args)
    so = SystemOne(ThinkEngine(), args.served_name, task_store=TaskStore("fp"), flavor=args.flavor_report)
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so


def test_f5_models_and_the_response_name_the_flavor_in_capitals():
    client, so = flavored("read-enhanced")
    assert client.get("/v1/models").json()["models"][0]["name"] == "Decisio-31b-read-enhanced"
    for sent in (
        "decisio-31b-read-enhanced",
        "DECISIO-31B-READ-ENHANCED",
        "Decisio-31b-read-enhanced",
        "anything",
        None,
    ):
        body = {"state": {"message": "0:0.9:x:1"}, "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
        if sent:
            body["model"] = sent
        r = client.post("/v1/systemone", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "Decisio-31b-read-enhanced"
        assert r.headers["x-decisio-flavor"] == "read-enhanced+pending"


def test_f5_another_flavors_name_is_refused_with_the_served_name():
    client, _ = flavored("read-enhanced")
    body = {"state": {"message": "0:0.9:x:1"}, "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
    for other in ("Decisio-31b-calibrated", "decisio-31b-thinking-commit"):
        r = client.post("/v1/systemone", json={**body, "model": other})
        assert r.status_code == 422 and "Decisio-31b-read-enhanced" in r.text
    ok = client.post("/v1/systemone", json={**body, "model": "Decisio-12b-calibrated"})  # another base's: not a sibling
    assert ok.status_code == 200


def test_f5_the_header_says_overrides():
    client, _ = flavored("read-enhanced", noul_commit=True)
    body = {"state": {"message": "0:0.9:x:1"}, "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
    assert client.post("/v1/systemone", json=body).headers["x-decisio-flavor"] == "read-enhanced+overrides+pending"


def test_f5_certificates_separate_per_flavor_and_per_override():
    _, a = flavored("read-enhanced")
    _, b = flavored("thinking-commit")
    _, c = flavored("read-enhanced", noul_commit=True)
    _, plain, _ = make()
    diff = stack.differences
    assert any("flavor" in d for d in diff(a.stack_facts(), b.stack_facts()))
    assert any("flavor" in d for d in diff(a.stack_facts(), c.stack_facts()))
    assert any("flavor" in d for d in diff(a.stack_facts(), plain.stack_facts()))


def test_f5_health_carries_the_flavor():
    from fastapi.testclient import TestClient

    from decisio.serve.reference_docs import _DocEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore
    from decisio.serve.vllm_engine import make_app

    args = args_for("gemma-4-31b", flavor="thinking-commit", layout="plain")
    resolve_base(args)
    engine = _DocEngine()
    so = SystemOne(engine, args.served_name, task_store=TaskStore("fp"), flavor=args.flavor_report)
    h = TestClient(make_app(engine, so)).get("/health").json()
    f = h["systemone"]["flavor"]
    assert f["name"] == "thinking-commit" and f["served_name"] == "Decisio-31b-thinking-commit"
    assert "layout" in f["overrides"] and f["pending"]


# ---- F6 -------------------------------------------------------------------------------------------------------------


def test_f6_the_flag_is_in_the_parser():
    p = vllm_engine.build_parser()
    assert p.parse_args(["--flavor", "calibrated"]).flavor == "calibrated"
    assert p.parse_args([]).flavor is None
    with pytest.raises(SystemExit):
        p.parse_args(["--flavor", "honest"])


def test_f6_a_tuned_repository_is_refused_for_now(tmp_path):
    d = write_repo(tmp_path, build(), name="t")
    with pytest.raises(ValueError, match="tuned"):
        resolve_base(args_for(str(d), flavor="read-enhanced"))


def test_f6_the_qwen_base_has_no_flavors():
    with pytest.raises(ValueError, match="no flavors"):
        resolve_base(args_for("qwen3.6-35b-a3b", flavor="read-enhanced"))


def test_f6_calibrated_refuses_to_start_until_its_values_exist():
    with pytest.raises(ValueError, match="calibrated temperatures are not fitted"):
        resolve_base(args_for("gemma-4-12b-fp8", flavor="calibrated"))
