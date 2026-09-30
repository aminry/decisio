# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""JevBench v1.5 open-set scoring (`decisio.bench.jevbench_v15`), against the frozen method's rules worked by hand.

J1  per-item rules: choice (invalid is wrong, chance 1 / options), noul (0.20 / 0.80 thresholds, between is an
    abstention counted wrong), score (expected position, chance-corrected nMAE, invalid takes the largest error)
J2  aggregation: tier weights 10/20/30/40 renormalised over tiers with items, type weights A (thirds) and
    B (50/25/25) renormalised over supported types, per-tier values not clipped

  uv run pytest -q tests/unit/test_jevbench_v15.py
"""

import pytest

from decisio.bench.jevbench_v15 import cell_cc, item_score, score_runs


def task(i, qtype, labels, expected):
    return {"id": i, "labels": labels, "expected": expected, "question": {"type": qtype}}


def row(i, probs, predicted=None, valid=True):
    return {"task_id": i, "valid": valid, "probs": probs, "predicted": predicted}


def test_j1_item_rules():
    c = task("c", "choice", ["a", "b", "c", "d"], "b")
    assert item_score(c, row("c", {"a": 0.1, "b": 0.7, "c": 0.1, "d": 0.1}, "b")) == {
        "type": "choice",
        "correct": True,
        "chance": 0.25,
        "valid": True,
    }
    assert (
        item_score(c, row("c", None, None, valid=False))["correct"] is False and item_score(c, None)["valid"] is False
    )
    n = task("n", "noul", ["no", "yes"], "yes")
    for p, correct, abst in (
        (0.80, True, False),
        (0.95, True, False),
        (0.79, False, True),
        (0.5, False, True),
        (0.21, False, True),
        (0.20, False, False),
    ):
        s = item_score(n, row("n", {"yes": p, "no": 1 - p}))
        assert (s["correct"], s["abstained"]) == (correct, abst), p
    assert item_score(task("n", "noul", ["no", "yes"], "no"), row("n", {"yes": 0.2, "no": 0.8}))["correct"] is True
    s = task("s", "score", ["0", "1", "2", "3"], 0)
    got = item_score(s, row("s", {"0": 0.5, "1": 0.5, "2": 0, "3": 0}))
    assert got["nmae"] == pytest.approx(0.5 / 3) and got["nmae_chance"] == pytest.approx((0 + 1 + 2 + 3) / 4 / 3)
    assert item_score(s, None)["nmae"] == 1.0  # invalid: the largest error
    assert item_score(task("s", "score", ["0", "1", "2", "3", "4"], 2), None)["nmae"] == 0.5


def test_j2_aggregation():
    # choice: easy 2 of 2 right (4 options), hard 1 of 2 right (2 and 4 options: chance 0.375)
    tasks_e = [
        task("e1", "choice", list("abcd"), "a"),
        task("e2", "choice", list("abcd"), "b"),
        task("en", "noul", ["no", "yes"], "yes"),
    ]
    rows_e = [row("e1", {"a": 1}, "a"), row("e2", {"b": 1}, "b"), row("en", {"yes": 0.9, "no": 0.1})]
    tasks_h = [
        task("h1", "choice", list("ab"), "a"),
        task("h2", "choice", list("abcd"), "a"),
        task("hn1", "noul", ["no", "yes"], "no"),
        task("hn2", "noul", ["no", "yes"], "no"),
        task("hs", "score", ["0", "1", "2"], 2),
    ]
    rows_h = [
        row("h1", {"a": 1}, "a"),
        row("h2", {"b": 1}, "b"),
        row("hn1", {"yes": 0.6, "no": 0.4}),
        row("hn2", {"yes": 0.9, "no": 0.1}),
        row("hs", {"0": 0, "1": 1.0, "2": 0}),
    ]
    r = score_runs([("easy", tasks_e, rows_e), ("hard", tasks_h, rows_h)])
    ch = r["per_type"]["choice"]
    assert ch["tiers"]["easy"] == pytest.approx(100.0) and ch["tiers"]["hard"] == pytest.approx(
        100 * (0.5 - 0.375) / 0.625
    )
    assert ch["cc"] == pytest.approx((0.1 * 100 + 0.4 * 20) / 0.5)  # renormalised over easy + hard
    no = r["per_type"]["noul"]
    assert no["tiers"] == {"easy": pytest.approx(100.0), "hard": pytest.approx(-100.0)}  # not clipped
    assert no["abstention_rate"] == pytest.approx(1 / 3)
    sc = r["per_type"]["score"]
    assert sc["tiers"] == {"hard": pytest.approx(100 * (1 - 0.5 / 0.5))} and sc["cc"] == pytest.approx(0.0)
    assert r["I_open_A"] == pytest.approx((ch["cc"] + no["cc"] + sc["cc"]) / 3)
    assert r["I_open_B"] == pytest.approx(0.5 * ch["cc"] + 0.25 * no["cc"] + 0.25 * sc["cc"])
    assert r["tiers_missing"] == ["judge", "standard"] and r["n_items"] == 8 and r["invalid"] == 0
    # an unsupported type is left out and the weights renormalise
    r2 = score_runs([("easy", tasks_e[:2], rows_e[:2])])
    assert r2["I_open_A"] == pytest.approx(100.0) and r2["I_open_B"] == pytest.approx(100.0)
    assert cell_cc("score", [{"nmae": 0.0, "nmae_chance": 0.5}]) == 100.0
