# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Run by test_second_question_cached.py on a card: on a base's served profile, a later, different question about a
state reads the state from the prefix cache, also with the cache filled near or past its pool (part B).

The server is built as `decisio-serve --base <base> --model <checkpoint>` builds it (uvicorn left out). On the Gemma
bases a single question registers its state's boundary with a warm-up (decisio.families register_state_boundary);
without it a second question read the whole state again (on the 31B, 531 ms instead of 47 at 3,000 tokens, RLCD
experiments/2026-10-05_t7_card_e2_e3_e4), and vLLM's retention interval, the other way to keep it, lost even repeated
questions on the earliest states at 1.3 times the cache's pool (same record).

    python tests/gpu/second_question_cached.py <checkpoint> --base <base> [--tokens 3000] [--fill F] [--revisit 8]
        [--register-boundary after|before] [--follow 5]

PART A  a first question on a new state, the same question again, then a different question, which reads the state's
        whole hit units from the cache. With --register-boundary before (0.8.1) the first two answers are equal (a
        first read goes through the cache too); with after (the default) the first read is fresh and the warm-up follows
        the answer, so the first answer and its repeat may differ (reported as FRESH VS CACHED, not gated: on the Gemma
        4 12B up to 0.113, on the 31B 0.0, EVAL_CARD 6.5), and the repeat and a third ask are equal
PART B  distinct states filling `fill` times the pool vLLM reports (`cache_config.kv_cache_size_tokens`), one
        question each, each counted by the blocks its request occupies (its prompt rounded up to the cache block: on
        the Qwen base a 3,000-token state is padded to 3,168 and its question takes a fourth block of 1,056, so counting
        3,000 filled more than the pool and evicted the earliest states on the first card run of the 0.9 fill); then
        the `revisit` earliest states again, the same question first and then a different one, each reading the
        state's whole hit units from the cache (the repeat first: a miss on the different question
        would read the state again and let the repeat hit). The fill defaults to 1.3 on a base that registers its
        state boundaries (the Gemma bases, whose reported pool understates what their sliding-window cache holds) and
        to 0.9 on one that does not (the Qwen base, whose reported pool is its capacity, so above it the earliest
        states are evicted, as any cache evicts under overload)
        With --part-b report the verdict is printed and not gated: under --register-boundary after the Gemma 4 31B kept
        none of its 8 earliest states at 1.3 times its pool, where before kept all 8 (Lab 2's session of 2026-10-07; why
        its default is before); the 12B kept all 8 under both orders
PART C  (--register-boundary after) on `follow` new states each, a first question and then a different one at once
        (the registration still pending or running: the follow-up waits for it) and after a pause of a second (the
        registrar's thread has sent it): every follow-up reads the state's whole hit units from the cache, and its first
        question was answered without the warm-up (deferred). Reports the follow-ups' server times and how many sent a
        pending registration ahead of their questions
"""

import argparse
import json
import math
import random
import sys
import time

import numpy as np
import uvicorn

from decisio.serve import vllm_engine as sv

WORDS = "the order arrived late and the customer asked for a refund because the box was damaged".split()
FIRST = {
    "kind": "choice",
    "instructions": "Which team should handle this ticket?",
    "options": ["billing", "access", "bug", "other"],
}
SECOND = {"kind": "noul", "instructions": "Does the customer ask for a refund?"}


def served(base, model, register="after"):
    """The engine decisio's server builds for these flags."""
    got = {}
    make_app = sv.make_app
    sv.make_app = lambda engine, so=None, health=None: got.update(engine=engine) or make_app(engine, so, health)
    uvicorn.run = lambda app, **kw: None
    sys.argv = ["decisio", "--base", base, "--model", model, "--register-boundary", register]
    sv.main()
    return got["engine"]


def ticket(tok, n, seed):
    """A synthetic ticket of n tokens (the latency cells' words), different for each seed."""
    rng = random.Random(seed)
    text = f"Ticket {rng.randint(0, 10**9)}. "
    while len(tok.encode(text, add_special_tokens=False)) < n:
        text += " ".join(rng.choice(WORDS) for _ in range(20)) + ". "
    return text


def ask(eng, state, question):
    """(probabilities, cached tokens, whole hit units of the shared prefix, forward ms)."""
    probs, info = eng.answer(state, [question])
    hit = eng.cache_hit_unit
    return np.asarray(probs[0]), info["cached_tokens"][0], (info["shared_prefix_tokens"] // hit) * hit, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--base", required=True)
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--fill", type=float, default=None)
    ap.add_argument("--revisit", type=int, default=8)
    ap.add_argument("--register-boundary", default="after", choices=["after", "before"])
    ap.add_argument("--follow", type=int, default=5)
    ap.add_argument("--part-b", default="gate", choices=["gate", "report"])
    a = ap.parse_args()
    eng = served(a.base, a.model, a.register_boundary)
    facts = eng.facts()
    cfg = eng.llm.llm_engine.vllm_config.cache_config
    # the pool vLLM reports ("GPU KV cache size"); num_gpu_blocks x block_size counts every cache group's blocks, 5.25
    # times the reported pool on Gemma 4 12B (944,704 against 179,859 tokens), so part B once filled 6.8 times the pool
    pool = cfg.kv_cache_size_tokens
    fill = a.fill if a.fill is not None else (1.3 if facts.get("registers_state_boundary") else 0.9)
    keys = (
        "base",
        "engine_process",
        "block_size",
        "cache_hit_unit",
        "hash_unit",
        "pad_unit",
        "registers_state_boundary",
        "register_boundary",
    )
    print("CACHE", json.dumps({**{k: facts.get(k) for k in keys}, "pool_tokens": pool}), flush=True)

    after = facts.get("register_boundary") == "after"
    fresh_dp = []  # a first read against a cached read of the same question (after: the first read is fresh)

    state = ticket(eng.tok, a.tokens, 0)
    p1, c1, w1, i1 = ask(eng, state, FIRST)
    p2, c2, w2, i2 = ask(eng, state, FIRST)
    p3, c3, w3, i3 = ask(eng, state, SECOND)
    p4, c4, w4, _ = ask(eng, state, FIRST)
    same = np.array_equal(p1, p2)
    if after and facts.get("registers_state_boundary"):
        fresh_dp.append(float(np.abs(p1 - p2).max()))
        deferred = (i1.get("state_boundary") or {}).get("deferred") == 1
        a_ok = deferred and np.array_equal(p2, p4) and c2 >= w2 and c3 >= w3 and c4 >= w4
    else:
        a_ok = same and c3 >= w3 and c2 >= w2
    print(
        f"PART A {'PASS' if a_ok else 'FAIL'}: first read {c1} cached; the same question again {c2} of {w2}, answers "
        f"{'equal' if same else f'differ by {float(np.abs(p1 - p2).max()):.4g}'}; a different question {c3} of {w3}; "
        f"the first question a third time {c4} of {w4}, {'equal' if np.array_equal(p2, p4) else 'differs'} to the "
        f"second; state boundary {i1.get('state_boundary')}, {i2.get('state_boundary')}, {i3.get('state_boundary')}",
        flush=True,
    )

    c_ok = True
    if after and facts.get("registers_state_boundary"):
        lines, hits, total = [], 0, 0
        for pause in (0.0, 1.0):
            for k in range(a.follow):
                s = ticket(eng.tok, a.tokens, 500 + k + (100 if pause else 0))
                pf, _, _, first = ask(eng, s, FIRST)
                time.sleep(pause)
                _, c, w, info = ask(eng, s, SECOND)
                pr, _, _, _ = ask(eng, s, FIRST)
                fresh_dp.append(float(np.abs(pf - pr).max()))
                sb = info.get("state_boundary") or {}
                ok = c >= w and (first.get("state_boundary") or {}).get("deferred") == 1
                hits += ok
                total += 1
                lines.append(
                    f"pause {pause:g} s: follow-up {c} of {w} cached, {info['server_ms']:.1f} ms, "
                    f"registrations sent ahead {sb.get('ran_before', 0)} ({sb.get('ran_before_ms', 0.0):.1f} ms)"
                )
        c_ok = hits == total
        print(
            f"PART C {'PASS' if c_ok else 'FAIL'}: {hits} of {total} follow-ups read the state from the cache",
            flush=True,
        )
        for line in lines:
            print("  " + line, flush=True)
    if fresh_dp:
        print(
            f"FRESH VS CACHED: largest |dp| of a first read against its cached repeat {max(fresh_dp):.4g} over "
            f"{len(fresh_dp)} states (reported, not gated)",
            flush=True,
        )

    block = facts.get("block_size") or 1
    prompt = (i1.get("engine_prompt_tokens") or [round(i1["prompt_tokens_mean"])])[0]  # vLLM reports the first
    per_state = math.ceil(prompt / block) * block  # the blocks one request occupies
    n = math.ceil(fill * pool / per_state)
    states = [ticket(eng.tok, a.tokens, 1000 + k) for k in range(n)]
    for s in states:
        ask(eng, s, FIRST)
    hits_other = hits_same = 0
    for s in states[: a.revisit]:
        _, c, w, _ = ask(eng, s, FIRST)
        hits_same += c >= w
        _, c, w, _ = ask(eng, s, SECOND)
        hits_other += c >= w
    b_ok = hits_other == hits_same == a.revisit
    print(
        f"PART B {'PASS' if b_ok else 'FAIL'}{' (reported, not gated)' if a.part_b == 'report' else ''}: {n} states of "
        f"{a.tokens} tokens, {per_state} tokens of blocks each ({n * per_state}, {fill} times the pool of {pool}); of "
        f"the {a.revisit} earliest, a different question read the state from the cache on {hits_other}, the same "
        f"question on {hits_same}",
        flush=True,
    )
    if a.part_b == "report":
        b_ok = True
    passed = a_ok and b_ok and c_ok
    print(f"SECOND QUESTION {'CACHED PASS' if passed else 'FAIL'}: {a.base} {a.register_boundary}", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
