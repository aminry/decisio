# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""decisio.bench.pair_check: two servers' answers compared with the prefix cache in the same state per request (no
model, no network).

P1  two servers that differ only by cache residency (a fresh read differs from a cached one) compare equal on the
    second reads
P2  a real difference between the servers (another probability on a cached read) fails the comparison, and so does a
    different choice
P3  a request the first server's table answered is not compared

    uv run pytest -q tests/unit/test_pair_check.py
"""

import collections

from decisio.bench import pair_check

KEYS = ["a", "b", "c"]


class FakeServer:
    """A server whose read is perturbed when the prompt is not resident in a small LRU cache; `shift` is a real,
    residency-independent difference."""

    def __init__(self, capacity=4, shift=0.0, table=(), flip=False):
        self.cap, self.lru, self.shift, self.table, self.flip = (
            capacity,
            collections.OrderedDict(),
            shift,
            set(table),
            flip,
        )

    def answer(self, state):
        if state in self.table:
            return {"answered_by": "table", "choice": "a", "probabilities": {"a": 0.9, "b": 0.05, "c": 0.05}}
        fresh = state not in self.lru
        self.lru[state] = True
        self.lru.move_to_end(state)
        while len(self.lru) > self.cap:
            self.lru.popitem(last=False)
        top = 0.6 + self.shift + (0.02 if fresh else 0.0)
        p = {"a": top, "b": (1 - top) * 0.6, "c": (1 - top) * 0.4}
        return {"answered_by": "decisio", "choice": "b" if self.flip else "a", "probabilities": p}


def poster(servers):
    def post(url, path, body, timeout=0):
        return 200, {}, {"answers": {"q": servers[url].answer(body["state"])}}, 1.0

    return post


def requests(n):
    q = {"type": "choice", "criteria": {k: k for k in KEYS}}
    return [{"state": f"s{i}", "questions": {"q": q}} for i in range(n)]


def test_p1_residency_alone_does_not_fail_the_second_reads():
    # A's table answers 15 of 20, so only its 5 deferred rows are read on both; each is read twice back to back, and
    # the second reads (both cached) are compared
    table = {f"s{i}" for i in range(20) if i % 4}
    servers = {"A": FakeServer(4, table=table), "B": FakeServer(4)}
    rep = pair_check.compare(poster(servers), "A", "B", requests(20), tol=0.01)
    assert rep["ok"] and rep["compared"] == 5 and rep["max_dp_second_reads"] == 0.0 and rep["choice_different"] == 0


def test_p2_a_real_difference_fails():
    servers = {"A": FakeServer(40), "B": FakeServer(40, shift=0.03)}
    rep = pair_check.compare(poster(servers), "A", "B", requests(6), tol=0.01)
    assert not rep["ok"] and rep["max_dp_second_reads"] > 0.01
    servers = {"A": FakeServer(40), "B": FakeServer(40, flip=True)}
    assert not pair_check.compare(poster(servers), "A", "B", requests(6))["ok"]


def test_p3_table_answers_are_not_compared():
    servers = {"A": FakeServer(40, table={"s0", "s1"}), "B": FakeServer(40)}
    assert pair_check.compare(poster(servers), "A", "B", requests(5))["compared"] == 3
