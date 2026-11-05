# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The fast-table latency benchmark (`decisio.bench.table_latency`) end to end on a server with the fast-table tests'
fake engine and encoder (virtual time): no model, no encoder, no network.

B1  it registers the two arms (the table's task and the model's), times every cell (arm x length x concurrency) on
    each arm's own requests, and reports the medians, the 95th percentiles, the `task` stage and the table's share
B2  the arms run one after the other on their own requests (a choice task is matched by its options, so the two tasks
    cannot be registered at once): the table answers only in `table`, and nothing is left registered
B3  a long cell's states are padded to the length asked, an object state keeping its fields
B4  a probe file too short for the cells is refused before anything is timed
B5  --kinds: a table arm per student kind, each on its own requests, its cells and its task named by the kind, the
    registration's fast_table_report kept; the probe is sized for every arm; an unknown or repeated kind is refused

    uv run pytest -q tests/unit/test_table_latency_bench.py
"""

import time

import pytest
import test_fast_table_task as FTT
from test_fast_table_task import virtual_time  # noqa: F401  (autouse: the virtual clock, the fake encoder)

from decisio.bench import table_latency as TL


def poster(client):
    def post(url, path, body, timeout=600.0):
        t = time.perf_counter()
        r = client.post(path, json=body)
        return r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.json(), (time.perf_counter() - t) * 1000

    return post


def deleter(client):
    return lambda url, path: client.delete(path).status_code


def probe(n, seed=7):
    return [e["request"] for e in FTT.examples(n, seed)]


def test_b1_b2_every_cell_on_its_own_requests():
    client, so, _ = FTT.make()
    rep = TL.run(
        "",
        FTT.examples(800),
        probe(160),
        concurrency=(1, 4),
        lengths=("short", "long"),
        per_cell=20,
        long_chars=3000,
        certify={"alpha": 0.1, "delta": 0.1},
        post=poster(client),
        delete=deleter(client),
    )
    assert rep["tasks"]["table"]["certified"] and rep["tasks"]["model"]["certified"]
    assert rep["tasks"]["table"]["fast_table"]["state"] == "active"
    cells = {(c["arm"], c["length"], c["concurrency"]): c for c in rep["cells"]}
    assert len(cells) == 8
    for (arm, _, _), c in cells.items():
        assert c["requests"] == 20 and c["errors"] == 0
        assert c["server_ms_p50"] is not None and c["task_ms_p50"] is not None
        assert c["certified_share"] > 0
        if arm == "model":
            assert c["table_share"] == 0  # its task has no table
    # the table answers short states; the fake encoder's bag of words is diluted by a long state's filler, so there it
    # may defer every one (decisio reads them, which the cell's task stage then shows)
    assert all(cells[("table", "short", k)]["table_share"] > 0 for k in (1, 4))
    assert client.get("/v1/tasks").json()["tasks"] == []  # both arms' tasks were deleted


def test_b3_long_states_are_padded_and_keep_their_fields():
    req = probe(1)[0]
    long = TL.lengthen(req, 4000)  # a text state: filler after its text
    assert long["state"].startswith(req["state"]) and len(long["state"]) >= 4000
    obj = TL.lengthen({**req, "state": {"message": "hi", "id": 7}}, 500)  # an object state: a notes field
    assert obj["state"]["message"] == "hi" and obj["state"]["id"] == 7 and len(str(obj["state"])) >= 500


def test_b4_a_short_probe_file_is_refused():
    client, so, _ = FTT.make()
    with pytest.raises(SystemExit, match="the probe file has"):
        TL.run(
            "",
            FTT.examples(800),
            probe(10),
            per_cell=20,
            post=poster(client),
            delete=deleter(client),
            certify={"alpha": 0.1, "delta": 0.1},
        )


def test_b5_one_table_arm_per_kind():
    client, so, _ = FTT.make()
    post, seen = poster(client), []

    def recording(url, path, body, timeout=600.0):
        if path == "/v1/tasks":
            seen.append(body.get("fast_table"))
        return post(url, path, body, timeout)

    with pytest.raises(SystemExit, match="the cells need 60"):  # three arms of one cell of 20
        TL.run(
            "",
            FTT.examples(800),
            probe(59),
            kinds=("static", "encoder"),
            concurrency=(1,),
            lengths=("short",),
            per_cell=20,
            post=recording,
            delete=deleter(client),
            certify={"alpha": 0.1, "delta": 0.1},
        )
    rep = TL.run(
        "",
        FTT.examples(800),
        probe(60),
        kinds=("static", "encoder"),
        concurrency=(1,),
        lengths=("short",),
        per_cell=20,
        certify={"alpha": 0.1, "delta": 0.1},
        post=recording,
        delete=deleter(client),
    )
    assert seen == [{"kind": "static"}, {"kind": "encoder"}, None]
    assert set(rep["tasks"]) == {"table:static", "table:encoder", "model"} and rep["kinds"] == ["static", "encoder"]
    for k in ("static", "encoder"):
        t = rep["tasks"][f"table:{k}"]
        assert t["kind"] == k and t["fast_table_report"]["kinds"][0]["kind"] == k
    assert "kind" not in rep["tasks"]["model"]
    assert [(c["arm"], c["kind"]) for c in rep["cells"]] == [("table", "static"), ("table", "encoder"), ("model", None)]
    assert all(c["requests"] == 20 and c["errors"] == 0 for c in rep["cells"])
    assert client.get("/v1/tasks").json()["tasks"] == []
    assert TL.kind_request("static:multilingual") == {"kind": "static", "static_model": "multilingual"}
    for bad in (("hashed",), ("static", "static"), ()):
        with pytest.raises(SystemExit, match="kind"):
            TL.run("", FTT.examples(10), probe(10), kinds=bad, post=recording, delete=deleter(client))
