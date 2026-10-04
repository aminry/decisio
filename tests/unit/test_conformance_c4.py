# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Gate C4 of `decisio.serve.systemone_conformance` against a stand-in server, with a temperature per question type.

K1  the served temperature of a type: the type's own from /health, else the global one
K2  C4 passes when each item's raw readout, at the temperature its type is served at, equals what /v1/systemone sends
    (the Qwen base serves choice questions at 1.370 and the rest at 1.506; one temperature for all failed C4 on choice)
K3  C4 still fails when a served answer differs from the raw readout at its temperature

  uv run pytest -q tests/unit/test_conformance_c4.py
"""

import numpy as np
import pytest

from decisio.serve import systemone_conformance as sc
from decisio.serve.temperature import apply_temperature

HEALTH = {"temperature": 1.506, "temperatures": {"choice": 1.37}, "noul_rendering": "words"}
RAW = {"noul": [0.8, 0.2], "choice": [0.6, 0.3, 0.1]}


def items():
    teams, outage = ["billing", "technical", "sales"], "Is this an outage?"
    pairs = [
        ({"type": "noul", "instructions": outage}, {"kind": "noul", "instructions": outage}),
        (
            {"type": "choice", "instructions": "Which team?", "criteria": dict.fromkeys(teams)},
            {"kind": "choice", "instructions": "Which team?", "options": teams},
        ),
    ]
    return [
        {
            "task": "x",
            "i": i,
            "label": 0,
            "answer": {"state": "s", "questions": [a]},
            "systemone": {"state": "s", "questions": {"q": q}},
        }
        for i, (q, a) in enumerate(pairs)
    ]


class Response:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


def server(temperature_of):
    """A stand-in for httpx.Client: /v1/answer gives the raw readout, /v1/systemone the readout at
    temperature_of(type)."""

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return Response({"systemone": HEALTH})

        def post(self, url, json):
            if url.endswith("/v1/answer"):
                kind = json["questions"][0]["kind"]
                opts = json["questions"][0].get("options") or ["yes", "no"]
                return Response({"answers": [{"probs": RAW[kind], "options": opts}]})
            q = json["questions"]["q"]
            p = apply_temperature(np.array(RAW[q["type"]]), temperature_of(q["type"]))
            if q["type"] == "noul":
                return Response({"answers": {"q": {"type": "noul", "noul": float(p[0])}}})
            probs = dict(zip(q["criteria"], map(float, p)))
            return Response({"answers": {"q": {"type": "choice", "probabilities": probs}}})

    return Client


def test_k1_served_temperature():
    assert sc.served_temperature(HEALTH, "choice") == 1.37
    assert sc.served_temperature(HEALTH, "noul") == sc.served_temperature(HEALTH, "score") == 1.506
    assert sc.served_temperature({}, "choice") == 1.0


def test_k2_c4_applies_each_types_temperature(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "Client", server(lambda t: sc.served_temperature(HEALTH, t)))
    ok, detail = sc.gate_c4("http://stand-in", items())
    assert ok, detail
    assert detail["max_abs_delta_p"] == 0.0 and detail["temperatures"] == {"noul": 1.506, "choice": 1.37}


@pytest.mark.parametrize("wrong", [1.506, 1.0])
def test_k3_c4_fails_on_a_different_answer(monkeypatch, wrong):
    import httpx

    monkeypatch.setattr(httpx, "Client", server(lambda t: wrong if t == "choice" else 1.506))
    ok, detail = sc.gate_c4("http://stand-in", items())
    assert not ok and detail["max_abs_delta_p"] > 0
