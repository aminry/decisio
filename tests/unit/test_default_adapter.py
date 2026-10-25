# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""`--default-adapter NAME`: a loaded adapter answers a request that names none.

D1  a request with no adapter field, or null, is answered by the default adapter at its own temperatures; the response
    and the x-decisio-adapter header name it
D2  a request naming another loaded adapter gets that one, "none" gets the base, an unknown name is still a 404
D3  a server with no default answers as before; "none" there is the base too
D4  /health carries the default and the serving-stack fingerprint holds it only when set, so a certificate made under
    another default (or none) is suspended
D5  the flag: in the parser, refused when the name is not a loaded adapter, an adapter cannot be called "none"

  uv run pytest -q tests/unit/test_default_adapter.py
"""

import pytest
from test_adapter_bands import make, raw

from decisio.serve import stack, vllm_engine
from decisio.serve.temperature import apply_temperature

TEMPS = {"metask": {"choice": 2.0}, "plain": {"choice": 3.0}}


def body(adapter="<absent>", k=4):
    b = {
        "state": {"message": "x"},
        "model": "m",
        "questions": {
            "c": {"type": "choice", "instructions": "Which?", "criteria": {f"option_{i}": f"o{i}" for i in range(k)}}
        },
    }
    if adapter != "<absent>":
        b["adapter"] = adapter
    return b


def probs(r, k=4):
    return [r.json()["answers"]["c"]["probabilities"][f"option_{i}"] for i in range(k)]


# ---- D1, D2, D3 -----------------------------------------------------------------------------------------------------


def server(default="metask"):
    return make(temperatures={"choice": 5.0}, adapter_temperatures=TEMPS, default_adapter=default)


@pytest.mark.parametrize("field", ["<absent>", None])
def test_d1_a_request_naming_no_adapter_is_answered_by_the_default(field):
    client, _ = server()
    r = client.post("/v1/systemone", json=body(field))
    assert r.status_code == 200, r.text
    assert probs(r) == pytest.approx(list(apply_temperature(raw(4), 2.0)), abs=1e-3)  # the adapter's temperature
    assert r.json()["adapter"] == "metask" and r.headers["x-decisio-adapter"] == "metask"


def test_d2_another_adapter_none_and_an_unknown_name():
    client, _ = server()
    other = client.post("/v1/systemone", json=body("plain"))
    assert other.json()["adapter"] == "plain" and probs(other) == pytest.approx(
        list(apply_temperature(raw(4), 3.0)), abs=1e-3
    )
    base = client.post("/v1/systemone", json=body("none"))
    assert base.status_code == 200 and "adapter" not in base.json() and "x-decisio-adapter" not in base.headers
    assert probs(base) == pytest.approx(list(apply_temperature(raw(4), 5.0)), abs=1e-3)  # the base's temperature
    unknown = client.post("/v1/systemone", json=body("nope"))
    assert unknown.status_code == 404 and "metask" in unknown.text
    assert client.post("/v1/systemone", json=body(3)).status_code == 422


def test_d3_a_server_with_no_default_is_unchanged():
    client, so = make(temperatures={"choice": 5.0}, adapter_temperatures=TEMPS)
    r = client.post("/v1/systemone", json=body())
    assert "adapter" not in r.json() and probs(r) == pytest.approx(list(apply_temperature(raw(4), 5.0)), abs=1e-3)
    assert so.default_adapter is None
    assert "adapter" not in client.post("/v1/systemone", json=body("none")).json()
    assert client.post("/v1/systemone", json=body("metask")).json()["adapter"] == "metask"


# ---- D4 -------------------------------------------------------------------------------------------------------------


def test_d4_the_default_is_in_health_and_in_the_fingerprint_only_when_set():
    _, a = server("metask")
    _, b = server("plain")
    _, none = make(temperatures={"choice": 5.0}, adapter_temperatures=TEMPS)
    assert a.stack_facts()["serving"]["default_adapter"] == "metask"
    assert "default_adapter" not in none.stack_facts()["serving"]  # no default, no key, as before
    assert any(d.endswith("default_adapter") for d in stack.differences(a.stack_facts(), b.stack_facts()))
    assert any(d.endswith("default_adapter") for d in stack.differences(a.stack_facts(), none.stack_facts()))


def test_d4_health_carries_the_default():
    from fastapi.testclient import TestClient

    from decisio.serve.reference_docs import _DocEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore
    from decisio.serve.vllm_engine import make_app

    engine = _DocEngine()
    so = SystemOne(engine, "decisio-test", task_store=TaskStore("fp"), default_adapter="metask")
    assert TestClient(make_app(engine, so)).get("/health").json()["systemone"]["default_adapter"] == "metask"
    so2 = SystemOne(engine, "decisio-test", task_store=TaskStore("fp"))
    assert "default_adapter" not in TestClient(make_app(engine, so2)).get("/health").json()["systemone"]


# ---- D5 -------------------------------------------------------------------------------------------------------------


def test_d5_the_flag_is_in_the_parser_and_its_refusals(monkeypatch, capsys):
    p = vllm_engine.build_parser()
    assert p.parse_args(["--default-adapter", "metask"]).default_adapter == "metask"
    assert p.parse_args([]).default_adapter is None
    for argv, words in (
        (["--adapter", "a=/x", "--default-adapter", "b"], "not a loaded adapter"),
        (["--default-adapter", "a"], "not a loaded adapter"),
        (["--adapter", "none=/x"], 'cannot be named "none"'),
    ):
        monkeypatch.setattr("sys.argv", ["vllm_engine", *argv])
        with pytest.raises(SystemExit):
            vllm_engine.main()
        assert words in capsys.readouterr().err
