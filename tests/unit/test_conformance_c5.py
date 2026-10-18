# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Gate C5 of `decisio.serve.systemone_conformance`: the wire against System One's published OpenAPI document, and the
422 for a choice over System One's limit.

W1  C5 passes on the snapshot of System One's OpenAPI 0.2.0: the differences are exactly the documented ones
W2  a difference that is not documented fails C5 (System One's limit changes under us)
W3  a documented difference that has gone away fails C5 (the docs would be stale)
W4  a choice of 255 options is accepted and one of 256 is a 422 `too_long` with System One's message and location
W5  the documented differences hold on the real route: score levels past 10 and a request without `model` are answered,
    a `box` question is a 422, an `assets` field is accepted and ignored
W6  a response with System One's fields has no response differences, and a missing field is named
W7  the choice cap is the letter readout's cap

  uv run pytest -q tests/unit/test_conformance_c5.py
"""

import copy
import json
import zlib
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.readout.letters import MAX_LABELS
from decisio.serve import systemone_conformance as sc
from decisio.serve.systemone import MAX_CHOICE_OPTIONS, SystemOne, add_routes
from decisio.serve.tasks import TaskStore

SNAPSHOT = json.loads((Path(__file__).parents[2] / "benchmarks" / "system_one_wire_0.2.0.json").read_text())


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [zlib.crc32(w.encode()) % 50000 for w in str(text).split()]


class FakeEngine:
    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"
    adapters = {}

    def facts(self):
        return {"engine": "fake", "base": "fake-base"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None, images=None):
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            p = np.full(k, 0.3 / (k - 1))
            p[0] = 0.7
            out.append(p)
        return out, {}


def client():
    app = FastAPI()
    add_routes(app, SystemOne(FakeEngine(), "decisio-test", task_store=TaskStore("fp")))
    return TestClient(app)


def choice(n):
    return {
        "state": "s",
        "model": "m",
        "questions": {"q": {"type": "choice", "criteria": {f"t{i}": None for i in range(n)}}},
    }


def score(n):
    return {"state": "s", "model": "m", "questions": {"q": {"type": "score", "criteria": [f"l{i}" for i in range(n)]}}}


def test_w1_c5_passes_on_the_snapshot_with_exactly_the_documented_differences():
    ok, detail = sc.gate_c5(SNAPSHOT)
    assert ok, detail
    assert sorted(detail["differences"]) == sorted(sc.DOCUMENTED_DIFFERENCES)
    assert SNAPSHOT["version"] == "0.2.0"


def test_w2_an_undocumented_difference_fails():
    snap = copy.deepcopy(SNAPSHOT)
    snap["question_types"]["choice"]["criteria"]["maxProperties"] = 100  # System One lowers its cap
    ok, detail = sc.gate_c5(snap)
    assert not ok and any("choice criteria maxProperties" in d for d in detail["undocumented"])
    snap = copy.deepcopy(SNAPSHOT)
    snap["request"]["properties"].append("tools")  # System One adds a request field decisio does not read
    ok, detail = sc.gate_c5(snap)
    assert not ok and detail["undocumented"] == ["request field `tools`: System One has it, decisio does not read it"]


def test_w3_a_documented_difference_that_has_gone_away_fails():
    snap = copy.deepcopy(SNAPSHOT)
    snap["question_types"]["score"]["criteria"].pop("maxItems")  # System One drops its cap of 10
    ok, detail = sc.gate_c5(snap)
    assert not ok and detail["documented_but_gone"] == ["score criteria maxItems: System One 10, decisio none"]
    snap = copy.deepcopy(SNAPSHOT)
    snap["request"]["required"].remove("model")  # System One makes model optional
    ok, detail = sc.gate_c5(snap)
    assert not ok and "request field `model`: required by System One, optional here" in detail["documented_but_gone"]


def test_w4_255_choices_answer_and_256_is_a_422_like_system_ones():
    c = client()
    assert c.post("/v1/systemone", json=choice(255)).status_code == 200
    r = c.post("/v1/systemone", json=choice(256))
    assert r.status_code == 422
    err = r.json()["detail"][0]
    assert err["type"] == "too_long"
    assert err["loc"][-2:] == ["choice", "criteria"] and err["loc"][:3] == ["body", "questions", "q"]
    assert "at most 255 items" in err["msg"] and "not 256" in err["msg"]


def test_w5_the_documented_differences_hold_on_the_route():
    c = client()
    assert c.post("/v1/systemone", json=score(11)).status_code == 200  # System One refuses more than 10
    no_model = choice(3)
    no_model.pop("model")
    assert c.post("/v1/systemone", json=no_model).status_code == 200  # System One requires model
    box = {"state": "s", "model": "m", "questions": {"q": {"type": "box", "instructions": "where"}}}
    assert c.post("/v1/systemone", json=box).status_code == 422
    assets = {**choice(3), "assets": {"abc": {"type": "image", "format": "png", "data_base64": "AAAA"}}}
    assert c.post("/v1/systemone", json=assets).status_code == 200  # accepted, not read


def test_w6_response_fields_against_system_ones():
    c = client()
    body = {
        "state": "s",
        "model": "m",
        "questions": {
            "n": {"type": "noul", "instructions": "Is it?"},
            "c": {"type": "choice", "criteria": {"a": None, "b": None}},
            "s": {"type": "score", "criteria": ["x", "y", "z"]},
        },
    }
    resp = c.post("/v1/systemone", json=body).json()
    assert sc.response_differences(SNAPSHOT, resp) == []
    ok, detail = sc.gate_c5(SNAPSHOT, [resp])
    assert ok, detail
    broken = copy.deepcopy(resp)
    del broken["usage"]["output_tokens"]
    del broken["answers"]["c"]["confidence"]
    assert sc.response_differences(SNAPSHOT, broken) == [
        "usage field `output_tokens` missing",
        "choice answer `c`: field `confidence` missing",
    ]


def test_w7_the_choice_cap_is_the_letter_readouts():
    assert MAX_CHOICE_OPTIONS == MAX_LABELS == 255
