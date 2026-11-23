# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Compare the answers of two decisio servers of one build on the same requests, fairly with respect to the prefix
cache.

A read answered from vLLM's prefix cache is not bit-equal to a fresh read (batch-invariant mode is off by default),
and which prompts are resident depends on each server's request history and on the KV cache's capacity. Two servers
that have not read the same requests in the same order (a server whose fast table answers most of them, say) hold
different prompts, so comparing "the next read" on each compares a cached read with a fresh one: the difference
measures residency. `compare` puts both servers in the same state per request: each request the first server
answered with the model (not the fast table) is read twice back to back on both, and the second reads are compared
(both cached, on an identical prefix). The first reads' difference is reported, not judged.

    from decisio.bench import pair_check
    report = pair_check.compare(post, url_a, url_b, requests, tol=0.01)

`post(url, path, body)` has `decisio.bench.table_latency.http_post`'s signature: (status, headers, json, wall_ms).
"""

from __future__ import annotations


def _answer(post, url: str, request: dict):
    status, _, body, _ = post(url, "/v1/systemone", request)
    if status != 200:
        raise RuntimeError(f"{url}: status {status}: {str(body)[:200]}")
    name = next(iter(request["questions"]))
    return body["answers"][name]


def _probs(ans: dict) -> dict:
    return ans.get("probabilities") or {}


def _dp(a: dict, b: dict) -> float:
    pa, pb = _probs(a), _probs(b)
    return max((abs(pa[k] - pb[k]) for k in pa if k in pb), default=0.0)


def compare(post, url_a: str, url_b: str, requests: list[dict], tol: float = 0.01) -> dict:
    """The comparison of `url_a` and `url_b` on `requests` (one question each). `ok`: every compared request has the
    same choice on the second reads and their largest |dp| over the options is at most `tol`."""
    compared = choice_diff = 0
    second, first, own = [], [], []
    for r in requests:
        a1 = _answer(post, url_a, r)
        if a1.get("answered_by") == "table":
            continue  # the first server's fast table answered: no model read to compare
        b1 = _answer(post, url_b, r)
        a2 = _answer(post, url_a, r)
        b2 = _answer(post, url_b, r)
        compared += 1
        second.append(_dp(a2, b2))
        first.append(_dp(a1, b1))
        own.append(_dp(a1, a2))
        choice_diff += int(a2.get("choice") != b2.get("choice"))
    worst = max(second, default=0.0)
    return {
        "compared": compared,
        "choice_different": choice_diff,
        "max_dp_second_reads": worst,
        "max_dp_first_reads": max(first, default=0.0),
        "max_dp_first_vs_second_same_server": max(own, default=0.0),
        "tol": tol,
        "ok": choice_diff == 0 and worst <= tol,
    }
