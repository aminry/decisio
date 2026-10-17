# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Adapter serving: an adapter carries its own temperatures per question type, and its identity and temperatures are
part of the serving-stack fingerprint.

A1  an adapter file is validated (schema, question types, positive finite numbers), read from the adapter's folder or
    from `--adapter-temperatures`, which wins; a name that was not loaded is refused
A2  `fit` recovers the temperature a synthetic adapter was calibrated at (log loss), and the score fit takes the
    flattest temperature within the tolerance of the best expected-position error
A3  a request naming the adapter is served at its temperature for the type, a type it has none for keeps the base's,
    and a request without the adapter is served at the base's
A4  yes/no stays honest: an adapter's yes/no answer is not committed, `--noul-commit` still commits it, and the answer
    keeps the most probable option
A5  the fingerprint holds each adapter's weights hash and temperatures; a server with no adapters has neither key
A6  an adapter's identity is the hash of its weights and of its configuration

  uv run pytest -q tests/unit/test_adapter_serving.py
"""

import json
import zlib

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.serve import adapter_temperatures as at
from decisio.serve import stack
from decisio.serve.adapters import identity
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore
from decisio.serve.temperature import apply_temperature

CRIT = {"option_0": "billing", "option_1": "shipping", "option_2": "access"}
RAW = {"choice": [0.6, 0.3, 0.1], "noul": [0.7, 0.3], "score": [0.1, 0.2, 0.3, 0.4]}


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        text = str(text)
        if len(text.strip()) <= 2 and text.strip().isalpha():
            return [1000 + sum(ord(c) * 53**i for i, c in enumerate(text.strip()))]
        return [zlib.crc32(w.encode()) % 50000 for w in text.split()]


class FakeEngine:
    """Answers every question with a fixed raw distribution of its kind, and says which adapters it holds."""

    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"

    def __init__(self, identity=None):
        self.adapters = {n: object() for n in ("metask", "plain")}
        self.adapter_identity = identity or {}

    def facts(self):
        out = {"engine": "fake", "base": "fake-base", "adapters": sorted(self.adapters)}
        if self.adapter_identity:
            out["adapter_identity"] = self.adapter_identity
        return out

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None, images=None):
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            raw = {3: RAW["choice"], 2: RAW["noul"]}.get(k) or RAW["score"]
            out.append(np.asarray(raw[:k], float))
        return out, {}


QUESTIONS = {
    "c": {"type": "choice", "instructions": "Which queue?", "criteria": CRIT},
    "y": {"type": "noul", "instructions": "Is it billing?"},
    "s": {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high", "max"]},
}


def make(**kw):
    eng = FakeEngine(kw.pop("identity", None))
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"), **kw)
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so


def ask(client, adapter=None, **extra):
    body = {"state": {"message": "x"}, "model": "m", "questions": QUESTIONS, **extra}
    if adapter is not None:
        body["adapter"] = adapter
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    a = r.json()["answers"]
    return {
        "c": [a["c"]["probabilities"][k] for k in CRIT],
        "y": [a["y"]["noul"], 1 - a["y"]["noul"]],
        "s": list(a["s"]["probabilities"].values()),
        "raw": a,
    }


# ---- A1 -------------------------------------------------------------------------------------------------------------


def write(path, temps, schema=at.SCHEMA):
    path.write_text(json.dumps({"schema": schema, "temperatures": temps}))
    return path


def test_a1_a_file_is_validated(tmp_path):
    ok = at.read(write(tmp_path / "a.json", {"choice": 2.1, "noul": 3, "score": 1.2}))
    assert ok["temperatures"] == {"choice": 2.1, "noul": 3.0, "score": 1.2}
    for temps, why in (
        ({"choice": 0}, "positive"),
        ({"choice": -1.0}, "positive"),
        ({"choice": True}, "positive"),
        ({"choice": float("inf")}, "positive"),
        ({"choice": "2"}, "positive"),
        ({"ranking": 2.0}, "unknown question type"),
        ({}, "non-empty"),
    ):
        with pytest.raises(at.AdapterFileError, match=why):
            at.read(write(tmp_path / "b.json", temps))
    with pytest.raises(at.AdapterFileError, match="not a decisio.adapter/1"):
        at.read(write(tmp_path / "c.json", {"choice": 2.0}, schema="other/1"))
    with pytest.raises(at.AdapterFileError, match="cannot read"):
        at.read(tmp_path / "missing.json")


def test_a1_the_folder_file_is_read_and_the_flag_wins(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "p").mkdir()
    write(tmp_path / "m" / at.SIDECAR, {"choice": 2.0})
    adapters = {"metask": str(tmp_path / "m"), "plain": str(tmp_path / "p")}
    got = at.load(adapters)
    assert set(got) == {"metask"} and got["metask"]["source"] == "sidecar"
    flag = write(tmp_path / "f.json", {"choice": 9.0})
    got = at.load(adapters, {"metask": str(flag), "plain": str(flag)})
    assert got["metask"]["temperatures"] == {"choice": 9.0} and got["metask"]["source"] == "flag"
    assert got["plain"]["source"] == "flag"
    with pytest.raises(at.AdapterFileError, match=r"names \['ghost'\]"):
        at.load(adapters, {"ghost": str(flag)})


# ---- A2 -------------------------------------------------------------------------------------------------------------


def synthetic(n, t_true, t_read, k, seed):
    """Readouts served at t_read from logits calibrated at t_true; the labels are drawn from that calibration."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        z = rng.normal(0, 4, k)
        p_true = np.exp(z / t_true) / np.exp(z / t_true).sum()
        served = np.exp(z / t_read) / np.exp(z / t_read).sum()
        rows.append((served, int(rng.choice(k, p=p_true))))
    return rows


def test_a2_the_log_loss_fit_recovers_the_calibrated_temperature():
    rows = synthetic(3000, t_true=2.5, t_read=5.0, k=4, seed=1)
    P, y = [r[0] for r in rows], [r[1] for r in rows]
    kappa = at.fit_log_loss(P, y)
    assert 5.0 / kappa == pytest.approx(2.5, rel=0.1)
    doc = at.fit(
        [{"type": "choice", "probs": p.tolist(), "gold": g} for p, g in rows], {"choice": 5.0}, data="synthetic"
    )
    assert doc["schema"] == at.SCHEMA and doc["temperatures"]["choice"] == pytest.approx(2.5, rel=0.1)
    assert doc["fit"]["objective"] == {"choice": at.LOG_LOSS} and doc["fit"]["n"] == {"choice": 3000}


def test_a2_the_score_fit_takes_the_flattest_temperature_within_the_tolerance():
    rows = synthetic(600, t_true=3.0, t_read=6.0, k=5, seed=2)
    P, y = [r[0] for r in rows], [r[1] for r in rows]
    kappa, cc, best = at.fit_expected_position(P, y, tolerance=0.5)
    assert cc >= best - 0.5 - 1e-9 and best == max(at.position_cc(P, y, float(k)) for k in at.GRID)
    smaller = at.GRID[at.GRID < kappa]
    assert all(at.position_cc(P, y, float(k)) < best - 0.5 for k in smaller)  # nothing flatter qualifies
    tight, *_ = at.fit_expected_position(P, y, tolerance=0.0)
    assert tight >= kappa  # a tighter tolerance never gives a flatter temperature
    rows_ = [{"type": "score", "probs": p.tolist(), "gold": g} for p, g in rows]
    ll = at.fit(rows_, {"score": 6.0})
    ep = at.fit(rows_, {"score": 6.0}, score_objective=at.EXPECTED_POSITION)
    assert ll["fit"]["kappa_on_raw_readout"]["score"] > 0
    assert ll["fit"]["objective"] == {"score": at.LOG_LOSS} and ep["fit"]["objective"] == {
        "score": at.EXPECTED_POSITION
    }
    assert ep["temperatures"]["score"] <= ll["temperatures"]["score"]  # the error fit is the sharper one
    with pytest.raises(ValueError, match="score objective"):
        at.fit(rows_, {"score": 6.0}, score_objective="mean")
    with pytest.raises(ValueError, match="unknown question type"):
        at.fit([{"type": "ranking", "probs": [0.5, 0.5], "gold": 0}])


# ---- A3, A4 ---------------------------------------------------------------------------------------------------------


def test_a3_an_adapter_is_served_at_its_own_temperatures():
    base_t = {"choice": 4.0, "noul": 5.0, "score": 5.0}
    client, so = make(temperatures=base_t, adapter_temperatures={"metask": {"choice": 2.0, "noul": 3.0}})
    plain = ask(client)
    for q in ("c", "y", "s"):  # no adapter: the base's
        assert plain[q] == pytest.approx(
            list(
                apply_temperature(
                    RAW[{"c": "choice", "y": "noul", "s": "score"}[q]],
                    base_t[{"c": "choice", "y": "noul", "s": "score"}[q]],
                )
            )
        )
    own = ask(client, "metask")
    assert own["c"] == pytest.approx(list(apply_temperature(RAW["choice"], 2.0)))
    assert own["y"] == pytest.approx(list(apply_temperature(RAW["noul"], 3.0)))
    assert own["s"] == pytest.approx(list(apply_temperature(RAW["score"], 5.0)))  # none listed for score: the base's
    other = ask(client, "plain")  # an adapter with no file at all is served at the base's
    assert other["c"] == pytest.approx(plain["c"]) and other["y"] == pytest.approx(plain["y"])
    assert so.temperature_of("choice", "metask") == 2.0 and so.temperature_of("score", "metask") == 5.0
    assert so.temperature_of("choice") == 4.0 and so.temperature_of("choice", "plain") == 4.0


def test_a4_yes_no_is_not_committed_by_an_adapter():
    client, so = make(temperatures={"noul": 5.0}, adapter_temperatures={"metask": {"noul": 3.0}})
    p = ask(client, "metask")["y"][0]
    assert p == pytest.approx(apply_temperature(RAW["noul"], 3.0)[0])
    assert so.noul_commit is False
    committed, so2 = make(noul_commit=True, temperatures={"noul": 5.0}, adapter_temperatures={"metask": {"noul": 3.0}})
    q = ask(committed, "metask")["y"][0]
    assert q != pytest.approx(p) and so2.noul_commit is True  # the operator's choice still commits
    assert (q >= 0.5) == (p >= 0.5)  # and never changes the answer


# ---- A5, A6 ---------------------------------------------------------------------------------------------------------

ID = {"metask": {"weights_sha256": "a" * 64, "weights_bytes": 10, "config_sha256": "b" * 64}}


def digest(so):
    return stack.sha256(so.stack_facts())


def test_a5_the_fingerprint_holds_the_adapters_identity_and_temperatures():
    _, none = make()
    facts = none.stack_facts()
    assert "adapter_identity" not in facts["engine"] and "adapters" not in facts["calibration"]["temperatures"]
    _, a = make(identity=ID, adapter_temperatures={"metask": {"choice": 2.0}})
    facts = a.stack_facts()
    assert facts["engine"]["adapter_identity"] == ID
    assert facts["calibration"]["temperatures"]["adapters"] == {"metask": {"choice": 2.0}}
    base = digest(a)
    _, t = make(identity=ID, adapter_temperatures={"metask": {"choice": 2.1}})
    assert digest(t) != base and "calibration.temperatures.adapters.metask.choice" in stack.differences(
        a.stack_facts(), t.stack_facts()
    )
    other = {"metask": {**ID["metask"], "weights_sha256": "c" * 64}}
    _, w = make(identity=other, adapter_temperatures={"metask": {"choice": 2.0}})
    assert digest(w) != base and "engine.adapter_identity.metask.weights_sha256" in stack.differences(
        a.stack_facts(), w.stack_facts()
    )
    _, same = make(identity=ID, adapter_temperatures={"metask": {"choice": 2.0}})
    assert digest(same) == base


def test_a6_an_adapters_identity_is_its_weights_and_configuration(tmp_path):
    def folder(name, weights=b"w" * 100, cfg=b'{"r": 16}'):
        d = tmp_path / name
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(weights)
        if cfg is not None:
            (d / "adapter_config.json").write_bytes(cfg)
        return d

    a, same, other_w, other_c, no_cfg = (
        identity(folder("a")),
        identity(folder("b")),
        identity(folder("c", weights=b"x" * 100)),
        identity(folder("d", cfg=b'{"r": 32}')),
        identity(folder("e", cfg=None)),
    )
    assert a == same and a["weights_bytes"] == 100 and len(a["weights_sha256"]) == 64
    assert other_w["weights_sha256"] != a["weights_sha256"] and other_c["config_sha256"] != a["config_sha256"]
    assert "config_sha256" not in no_cfg
