# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""An adapter's choice temperature by the question's number of options (`choice_bands` in the adapter file).

An adapter that is calibrated on questions with few options is over-confident on questions with many, and the one choice
temperature of a flat file cannot follow it (a diagnosis of a tuned model). The file can carry a choice temperature
per band of
option count (2-10, 11-26, 27-255 by default), and the fit tools fit them.

B1  the file's `choice_bands` are validated (lo-hi keys within 2 to 255, no overlap, positive finite numbers); a file of
    bands alone is valid; a file without bands reads as before
B2  a choice question is served at the temperature of the band its option count is in, then at the adapter's flat
    choice temperature, then at the base's; yes/no and score are not banded; a request without the adapter is unchanged
B3  the bands are in /health's `adapter_temperatures` and in the serving-stack fingerprint, so a changed band is a
    different stack; a file without bands leaves the fingerprint as it was
B4  `fit` fits each band of option count on its own rows, writes the all-choice temperature beside them, leaves a band
    with too few rows to it, and writes no bands when the rows fall in one band; `none` turns them off
B5  `fit-split` does the same on the choice pool, reports each band's held-out figures and swap-validated ECE, and the
    CLI takes `--choice-bands` and `--min-band-items`

  uv run pytest -q tests/unit/test_adapter_bands.py
"""

import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.serve import adapter_temperatures as at
from decisio.serve import split_fit as sf
from decisio.serve import stack
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore
from decisio.serve.temperature import apply_temperature

BANDS = {"2-10": 2.0, "11-26": 3.0, "27-255": 4.0}


def write(path, temps=None, bands=None):
    doc = {"schema": at.SCHEMA, "temperatures": temps or {}}
    if bands is not None:
        doc["choice_bands"] = bands
    path.write_text(json.dumps(doc))
    return path


# ---- B1 -------------------------------------------------------------------------------------------------------------


def test_b1_the_bands_are_validated(tmp_path):
    ok = at.read(write(tmp_path / "a.json", {"choice": 2.5, "noul": 3}, BANDS))
    assert ok["choice_bands"] == BANDS and ok["temperatures"] == {"choice": 2.5, "noul": 3.0}
    assert at.read(write(tmp_path / "b.json", {}, {"2-255": 1.5}))["choice_bands"] == {"2-255": 1.5}  # bands alone
    assert at.read(write(tmp_path / "c.json", {"choice": 2.5}))["choice_bands"] == {}  # no bands: as before
    for bands, why in (
        ({"2-10": 2.0, "10-20": 3.0}, "overlap"),
        ({"2-10": 2.0, "5-6": 3.0}, "overlap"),
        ({"1-10": 2.0}, "2 <= lo <= hi <= 255"),
        ({"2-256": 2.0}, "2 <= lo <= hi <= 255"),
        ({"20-10": 2.0}, "2 <= lo <= hi <= 255"),
        ({"two-ten": 2.0}, "lo-hi"),
        ({"7": 2.0}, "lo-hi"),
        ({"2-10": 0}, "positive"),
        ({"2-10": -1.0}, "positive"),
        ({"2-10": True}, "positive"),
        ({"2-10": "2"}, "positive"),
        ({"2-10": float("inf")}, "positive"),
        ({}, "non-empty"),
        ([2.0], "non-empty|object"),
    ):
        with pytest.raises(at.AdapterFileError, match=why):
            at.read(write(tmp_path / "d.json", {"choice": 2.5}, bands))
    with pytest.raises(at.AdapterFileError, match="non-empty"):  # neither types nor bands
        at.read(write(tmp_path / "e.json", {}))
    assert (
        at.parse_bands("none") == ()
        and at.parse_bands("") == ()
        and at.parse_bands("11-26, 2-10")
        == (
            (2, 10),
            (11, 26),
        )
    )
    assert at.band_of(at.DEFAULT_BANDS, 2) == (2, 10) and at.band_of(at.DEFAULT_BANDS, 11) == (11, 26)
    assert at.band_of(at.DEFAULT_BANDS, 255) == (27, 255) and at.band_of(at.DEFAULT_BANDS, 256) is None
    assert at.band_of(((11, 26),), 5) is None


def test_b1_the_loader_carries_the_bands_to_the_server(tmp_path):
    (tmp_path / "m").mkdir()
    write(tmp_path / "m" / at.SIDECAR, {"choice": 2.5}, BANDS)
    got = at.load({"metask": str(tmp_path / "m")})["metask"]
    assert at.serving_temperatures(got) == {"choice": 2.5, "choice_bands": BANDS}
    (tmp_path / "p").mkdir()
    write(tmp_path / "p" / at.SIDECAR, {"choice": 2.5})
    flat = at.load({"plain": str(tmp_path / "p")})["plain"]
    assert at.serving_temperatures(flat) == {"choice": 2.5}  # no empty key for a file without bands


# ---- B2, B3 ---------------------------------------------------------------------------------------------------------


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [1000 + sum(ord(c) * 53**i for i, c in enumerate(str(text)[:6]))]


class FakeEngine:
    """Answers a question of k options with a fixed raw distribution of k options."""

    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"

    def __init__(self):
        self.adapters = {n: object() for n in ("metask", "plain")}
        self.adapter_identity = {}

    def facts(self):
        return {"engine": "fake", "base": "fake-base", "adapters": sorted(self.adapters)}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None, images=None):
        return [raw(len(q["options"]) if q.get("options") else 2) for q in questions], {}


def raw(k):
    w = np.linspace(2.0, 0.2, k) ** 2
    return w / w.sum()


def make(**kw):
    so = SystemOne(FakeEngine(), "decisio-test", task_store=TaskStore("fp"), **kw)
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so


def ask_choice(client, k, adapter=None):
    body = {
        "state": {"message": "x"},
        "model": "m",
        "questions": {
            "c": {"type": "choice", "instructions": "Which?", "criteria": {f"option_{i}": f"o{i}" for i in range(k)}}
        },
    }
    if adapter:
        body["adapter"] = adapter
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    return [r.json()["answers"]["c"]["probabilities"][f"option_{i}"] for i in range(k)]


def test_b2_a_choice_is_served_at_the_temperature_of_its_option_count():
    base_t = {"choice": 5.0}
    client, so = make(temperatures=base_t, adapter_temperatures={"metask": {"choice": 2.5, "choice_bands": BANDS}})
    for k, t in ((2, 2.0), (3, 2.0), (10, 2.0), (11, 3.0), (26, 3.0), (27, 4.0), (60, 4.0), (255, 4.0)):
        assert so.temperature_of("choice", "metask", k) == t
        assert ask_choice(client, k, "metask") == pytest.approx(list(apply_temperature(raw(k), t)), abs=1e-3), k
        assert ask_choice(client, k) == pytest.approx(list(apply_temperature(raw(k), 5.0)), abs=1e-3)  # no adapter
    # no option count (a caller that does not know it) and the unbanded paths: the flat choice temperature
    assert so.temperature_of("choice", "metask") == 2.5 and so.temperature_of("choice") == 5.0
    assert so.temperature_of("choice", "plain", 12) == 5.0 and so.temperature_of("noul", "metask", 12) == so.temperature


def test_b2_a_count_in_no_band_falls_to_the_flat_temperature_then_the_base():
    spec = {"choice": 2.5, "choice_bands": {"11-26": 3.0}}
    client, so = make(temperatures={"choice": 5.0}, adapter_temperatures={"metask": spec})
    assert so.temperature_of("choice", "metask", 5) == 2.5 and so.temperature_of("choice", "metask", 12) == 3.0
    assert ask_choice(client, 5, "metask") == pytest.approx(list(apply_temperature(raw(5), 2.5)), abs=1e-3)
    _, only = make(temperatures={"choice": 5.0}, adapter_temperatures={"metask": {"choice_bands": {"11-26": 3.0}}})
    assert only.temperature_of("choice", "metask", 5) == 5.0 and only.temperature_of("choice", "metask", 20) == 3.0


def test_b2_yes_no_and_score_are_not_banded():
    client, so = make(
        temperatures={"noul": 5.0, "score": 6.0},
        adapter_temperatures={"metask": {"noul": 3.0, "choice_bands": BANDS}},
    )
    assert so.temperature_of("noul", "metask", 30) == 3.0 and so.temperature_of("score", "metask", 30) == 6.0
    body = {
        "state": {"message": "x"},
        "model": "m",
        "adapter": "metask",
        "questions": {"s": {"type": "score", "instructions": "How bad?", "criteria": [f"l{i}" for i in range(12)]}},
    }
    r = client.post("/v1/systemone", json=body)  # a 12-level score is not a choice with 12 options
    assert r.status_code == 200
    got = list(r.json()["answers"]["s"]["probabilities"].values())
    assert got == pytest.approx(list(apply_temperature(raw(12), 6.0)), abs=1e-3)


def test_b3_the_bands_are_in_health_and_in_the_fingerprint():
    def digest(so):
        return stack.sha256(so.stack_facts())

    _, flat = make(adapter_temperatures={"metask": {"choice": 2.5}})
    _, a = make(adapter_temperatures={"metask": {"choice": 2.5, "choice_bands": BANDS}})
    assert a.adapter_temperatures == {"metask": {"choice": 2.5, "choice_bands": BANDS}}  # what /health shows
    assert a.stack_facts()["calibration"]["temperatures"]["adapters"] == a.adapter_temperatures
    assert digest(a) != digest(flat)
    _, same = make(adapter_temperatures={"metask": {"choice": 2.5, "choice_bands": dict(BANDS)}})
    assert digest(same) == digest(a)
    _, moved = make(adapter_temperatures={"metask": {"choice": 2.5, "choice_bands": {**BANDS, "11-26": 3.1}}})
    assert digest(moved) != digest(a)
    assert "calibration.temperatures.adapters.metask.choice_bands.11-26" in stack.differences(
        a.stack_facts(), moved.stack_facts()
    )
    _, none = make()
    assert "adapters" not in none.stack_facts()["calibration"]["temperatures"]  # no adapter, no key, as before


# ---- B4 -------------------------------------------------------------------------------------------------------------


def synthetic(n, t_true, t_read, k, seed):
    """Readouts served at t_read from logits calibrated at t_true; the labels are drawn from that calibration."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        z = rng.normal(0, 4, k)
        p_true = np.exp(z / t_true) / np.exp(z / t_true).sum()
        served = np.exp(z / t_read) / np.exp(z / t_read).sum()
        rows.append({"type": "choice", "probs": served.tolist(), "gold": int(rng.choice(k, p=p_true))})
    return rows


def banded_rows(seed=0):
    return (
        synthetic(600, 2.0, 5.0, 4, seed)
        + synthetic(400, 3.0, 5.0, 15, seed + 1)
        + synthetic(300, 4.5, 5.0, 40, seed + 2)
    )


def test_b4_each_band_is_fitted_on_its_own_rows():
    doc = at.fit(banded_rows(), {"choice": 5.0}, data="synthetic")
    bands = doc["choice_bands"]
    assert set(bands) == {"2-10", "11-26", "27-255"}
    assert bands["2-10"] == pytest.approx(2.0, rel=0.15)
    assert bands["11-26"] == pytest.approx(3.0, rel=0.15)
    assert bands["27-255"] == pytest.approx(4.5, rel=0.15)
    assert bands["2-10"] < doc["temperatures"]["choice"] < bands["27-255"]  # the all-choice fit lies between them
    d = doc["fit"]["detail"]["choice"]
    assert d["bands"]["11-26"]["n"] == 400 and d["not_fitted_under_min_items"] == {} and d["min_band_items"] == 20
    assert doc["fit"]["n"]["choice"] == 1300
    assert (
        at.serving_temperatures({"temperatures": doc["temperatures"], "choice_bands": bands})["choice_bands"] == bands
    )


def test_b4_the_file_round_trips_and_serves(tmp_path):
    doc = at.fit(banded_rows(1), {"choice": 5.0})
    path = tmp_path / "decisio_adapter.json"
    path.write_text(json.dumps(doc))
    got = at.read(path)
    assert got["choice_bands"] == doc["choice_bands"] and got["temperatures"] == doc["temperatures"]
    _, so = make(adapter_temperatures={"metask": at.serving_temperatures(got)})
    assert so.temperature_of("choice", "metask", 4) == doc["choice_bands"]["2-10"]
    assert so.temperature_of("choice", "metask", 40) == doc["choice_bands"]["27-255"]


def test_b4_a_thin_band_keeps_the_all_choice_temperature_and_one_band_is_no_bands():
    rows = synthetic(600, 2.0, 5.0, 4, 3) + synthetic(8, 4.0, 5.0, 40, 4)  # 8 rows in 27-255
    doc = at.fit(rows, {"choice": 5.0})
    assert set(doc["choice_bands"]) == {"2-10"}  # 11-26 has none, 27-255 has 8 < 20
    assert doc["fit"]["detail"]["choice"]["not_fitted_under_min_items"] == {"27-255": 8}
    assert at.fit(rows, {"choice": 5.0}, min_band_items=5)["choice_bands"].keys() == {"2-10", "27-255"}
    one = at.fit(synthetic(300, 2.0, 5.0, 4, 5), {"choice": 5.0})
    assert "choice_bands" not in one and "choice" not in one["fit"]["detail"]  # every row in 2-10: the flat fit is it
    off = at.fit(banded_rows(2), {"choice": 5.0}, bands="none")
    assert "choice_bands" not in off
    custom = at.fit(banded_rows(2), {"choice": 5.0}, bands="2-20,21-255")
    assert set(custom["choice_bands"]) == {"2-20", "21-255"}
    with pytest.raises(ValueError, match="overlap"):
        at.fit(banded_rows(2), {"choice": 5.0}, bands="2-20,20-255")


def test_b4_ragged_option_counts_inside_a_band_are_fitted_together():
    rows = synthetic(30, 3.0, 5.0, 12, 6) + synthetic(30, 3.0, 5.0, 20, 7) + synthetic(60, 2.0, 5.0, 4, 8)
    doc = at.fit(rows, {"choice": 5.0})
    assert doc["fit"]["detail"]["choice"]["bands"]["11-26"]["n"] == 60
    assert doc["choice_bands"]["11-26"] == pytest.approx(3.0, rel=0.3)


# ---- B5 -------------------------------------------------------------------------------------------------------------


def pool(prefix, groups, seed):
    """Reads and keys of choice items: groups = [(n, k, t_true)], served at decisio's 4.672."""
    reads, keys, n = {}, {}, 0
    rng = np.random.default_rng(seed)
    for count, k, t_true in groups:
        for _ in range(count):
            z = rng.normal(0, 4, k)
            raw_p = np.exp(z) / np.exp(z).sum()
            gold = int(rng.choice(k, p=apply_temperature(raw_p, t_true)))
            i = f"{prefix}-{n}"
            reads[i] = {"probs": apply_temperature(raw_p, 4.672).tolist(), "p_plain": raw_p.tolist()}
            keys[i] = {"gold": gold, "type": "choice"}
            n += 1
    return reads, keys


def other_types(seed=9):
    rng = np.random.default_rng(seed)
    reads, keys = {}, {}
    for t, k in (("noul", 2), ("score", 5)):
        for n in range(80):
            z = rng.normal(0, 4, k)
            raw_p = np.exp(z) / np.exp(z).sum()
            reads[f"{t}-{n}"] = {"probs": apply_temperature(raw_p, 5.252).tolist(), "p_plain": raw_p.tolist()}
            keys[f"{t}-{n}"] = {"gold": int(rng.choice(k, p=apply_temperature(raw_p, 3.0))), "type": t}
    return reads, keys


def test_b5_fit_split_fits_the_bands_on_the_choice_pool():
    H, hk = pool("h", [(300, 4, 2.0)], 1)
    D, dk = pool("d", [(160, 4, 2.0), (120, 15, 3.0), (100, 40, 4.5)], 2)
    extra, extra_keys = other_types()
    D.update(extra), dk.update(extra_keys)
    doc = sf.fit_split(H, hk, D, dk, variant="dev")
    assert set(doc["choice_bands"]) == {"2-10", "11-26", "27-255"}
    assert doc["choice_bands"]["2-10"] < doc["choice_bands"]["11-26"] < doc["choice_bands"]["27-255"]
    assert doc["temperatures"]["choice"] > 0 and doc["fit"]["n"]["choice"] == 380
    ref = {
        b: sf.fit_kappa(
            [D[i]["probs"] for i in D if dk[i]["type"] == "choice" and lo <= len(D[i]["probs"]) <= hi],
            [dk[i]["gold"] for i in D if dk[i]["type"] == "choice" and lo <= len(D[i]["probs"]) <= hi],
        )
        for b, (lo, hi) in {"2-10": (2, 10), "11-26": (11, 26), "27-255": (27, 255)}.items()
    }
    for b, kb in ref.items():  # the served temperature over the band's own kappa
        assert doc["choice_bands"][b] == round(4.672 / kb, 3)
    assert doc["fit"]["choice_bands"] == {"not_fitted_under_min_items": {}, "min_band_items": 20}
    sv = doc["fit"]["swap_validated_dev"]["choice"]
    assert set(sv["bands"]) == {"2-10", "11-26", "27-255"} and sv["bands"]["11-26"]["n"] == 120
    for b in ("2-10", "11-26"):  # served at 4.672 against calibrated at 2.0 and 3.0: scaled within the band is better
        assert sv["bands"][b]["ece10_swap_validated"] < sv["bands"][b]["ece10_unscaled"], b


def test_b5_the_halves_report_each_bands_held_out_figures():
    H, hk = pool("h", [(100, 4, 2.0)], 1)
    D, dk = pool("d", [(160, 4, 2.0), (120, 15, 3.0), (100, 40, 4.5)], 3)
    extra, extra_keys = other_types()
    D.update(extra), dk.update(extra_keys)
    doc = sf.fit_split(H, hk, D, dk, variant="devhalf0")
    bands = doc["fit"]["checks"]["choice"]["bands"]
    assert set(bands) == {"2-10", "11-26", "27-255"}
    for b, e in bands.items():
        assert e["served_at"] == 4.672 and e["n_fit"] > 0 and e["heldout"]["n"] > 0, b
        assert e["heldout"]["ece_fit"] <= e["heldout"]["ece_kappa1"] + 0.03, b
    assert sum(e["n_fit"] + e["heldout"]["n"] for e in bands.values()) == 380


def test_b5_the_h_pool_is_one_band_so_the_file_has_none_and_a_thin_band_is_left_out():
    H, hk = pool("h", [(300, 4, 2.0)], 1)
    D, dk = pool("d", [(160, 4, 2.0), (120, 15, 3.0), (6, 40, 4.5)], 3)
    extra, extra_keys = other_types()
    D.update(extra), dk.update(extra_keys)
    doc = sf.fit_split(H, hk, D, dk, variant="half0")  # choice from H: option counts 4 only
    assert "choice_bands" not in doc and "choice_bands" not in doc["fit"]
    thin = sf.fit_split(H, hk, D, dk, variant="dev")
    assert set(thin["choice_bands"]) == {"2-10", "11-26"}
    assert thin["fit"]["choice_bands"]["not_fitted_under_min_items"] == {"27-255": 6}
    assert thin["fit"]["swap_validated_dev"]["choice"]["bands"]["27-255"]["not_scored"]
    off = sf.fit_split(H, hk, D, dk, variant="dev", bands="none")
    assert "choice_bands" not in off and "bands" not in off["fit"]["swap_validated_dev"]["choice"]


def test_b5_the_cli_takes_the_bands(tmp_path, capsys):
    import gzip

    H, hk = pool("h", [(100, 4, 2.0)], 1)
    D, dk = pool("d", [(160, 4, 2.0), (120, 15, 3.0), (100, 40, 4.5)], 3)
    extra, extra_keys = other_types()
    D.update(extra), dk.update(extra_keys)

    def dump(reads, name):
        with gzip.open(tmp_path / name, "wt") as fh:
            for i, r in reads.items():
                fh.write(json.dumps({"id": i, **r}) + "\n")

    dump(H, "h.jsonl.gz"), dump(D, "d.jsonl.gz")
    (tmp_path / "hk.json").write_text(json.dumps(hk)), (tmp_path / "dk.json").write_text(json.dumps(dk))
    base = [
        "fit-split",
        "--h",
        str(tmp_path / "h.jsonl.gz"),
        "--h-keys",
        str(tmp_path / "hk.json"),
        "--dev",
        str(tmp_path / "d.jsonl.gz"),
        "--dev-keys",
        str(tmp_path / "dk.json"),
        "--variant",
        "dev",
    ]
    out = tmp_path / "out.json"
    assert at.main(base + ["--out", str(out)]) == 0
    assert set(json.loads(out.read_text())["choice_bands"]) == {"2-10", "11-26", "27-255"}
    assert set(at.read(out)["choice_bands"]) == {"2-10", "11-26", "27-255"}
    assert at.main(base + ["--out", str(out), "--choice-bands", "none"]) == 0
    assert "choice_bands" not in json.loads(out.read_text())
    assert at.main(base + ["--out", str(out), "--choice-bands", "2-20,21-255", "--min-band-items", "50"]) == 0
    assert set(json.loads(out.read_text())["choice_bands"]) == {"2-20", "21-255"}
    with pytest.raises(SystemExit):
        at.main(base + ["--out", str(out), "--choice-bands", "2-20,10-255"])
    assert "overlap" in capsys.readouterr().err
    # `fit` takes the same options
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps(r) for r in banded_rows(4)))
    o2 = tmp_path / "o2.json"
    assert at.main(["fit", "--reads", str(rows), "--read-at", "choice=5.0", "--out", str(o2)]) == 0
    assert set(json.loads(o2.read_text())["choice_bands"]) == {"2-10", "11-26", "27-255"}
    assert (
        at.main(["fit", "--reads", str(rows), "--read-at", "choice=5.0", "--out", str(o2), "--choice-bands", "none"])
        == 0
    )
    assert "choice_bands" not in json.loads(o2.read_text())
