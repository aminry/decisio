# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A task's context far over the model's window is refused fast, before any model work (docs/tasks.md, "Task context").

W1  a context over the window is a 422 that gives the token counts, never the context's text; the engine reads nothing
    and the context is not tokenized whole
W2  a context that fits alone but not with the longest example's state is a 422 that names that example's tokens
W3  a context just under the window is accepted (the boundary is exact to the row), the examples read one at a time
W4  the dynamic-option registration and an import of a task record with a context follow the same rule
W5  a request whose state makes the row exceed the window is a 422 and reads nothing
W6  an engine without vLLM's configuration (MLX, HF) answers within its `--max-model-len` window
W7  the MLX prefix cache keeps a task context's entry once: states read after it never push it out

  uv run pytest -q tests/unit/test_context_window.py
"""

import types

from test_certified_answers import examples, post_task, wire
from test_task_context import CONTEXT, CtxEngine, ask, make

from decisio.readout.letters import fmt_state
from decisio.serve import vllm_engine

SECRET = "zq9x"
LIMIT = 200


def ctx(n):
    return " ".join([SECRET] * n)


class CountingTok:
    """The stand-in tokenizer, recording the longest text it was asked to encode."""

    def __init__(self, tok):
        self.tok, self.longest = tok, 0

    def encode(self, text, add_special_tokens=False):
        self.longest = max(self.longest, len(str(text)))
        return self.tok.encode(text, add_special_tokens=add_special_tokens)

    def __getattr__(self, name):
        return getattr(self.tok, name)


def with_limit(limit=LIMIT):
    eng = CtxEngine(limit=limit)
    eng.tok = CountingTok(eng.tok)
    return (*make(eng)[:2], eng)


def test_w1_a_context_far_over_the_window_is_refused_fast_before_any_model_work():
    client, so, eng = with_limit()
    r = post_task(client, {"id": "a", "examples": examples(120, 0.9), "context": ctx(50_000)})
    detail = r.json()["detail"]
    assert r.status_code == 422 and str(LIMIT) in detail and "more than" in detail and "tokens" in detail
    assert SECRET not in detail  # the context's text is never echoed
    assert eng.states == [] and so.task_store.by_id("a") is None  # no example was read
    assert eng.tok.longest <= 8 * LIMIT  # a huge context is counted on a bounded part, never tokenized whole


def test_w1_a_context_over_the_window_by_a_little_gives_its_exact_tokens():
    client, _, eng = with_limit()
    r = post_task(client, {"id": "a", "examples": examples(40, 0.9), "context": ctx(LIMIT + 20)})
    detail = r.json()["detail"]
    assert r.status_code == 422 and f"{LIMIT + 20} tokens" in detail and f"{LIMIT:,}" in detail
    assert SECRET not in detail and eng.states == []


def test_w2_the_longest_examples_state_with_the_context_over_the_window_names_its_tokens():
    client, so, eng = with_limit()
    body = {"id": "a", "examples": examples(40, 0.9), "context": ctx(60)}
    assert post_task(client, body).status_code == 200  # the context and short examples fit
    eng.states.clear()
    long_state = "0:0.9:x " + ctx(300)
    n_long = len(eng.tok.encode(fmt_state({"message": long_state})))
    body["examples"][17]["request"] = wire(long_state)
    r = post_task(client, {**body, "id": "b"})
    detail = r.json()["detail"]
    assert r.status_code == 422 and f"longest example's state is {n_long} tokens" in detail
    assert "60 tokens" in detail and str(LIMIT) in detail and SECRET not in detail
    assert eng.states == [] and so.task_store.by_id("b") is None


def test_w3_a_context_just_under_the_window_is_accepted_and_one_more_word_is_not():
    client, _, eng = with_limit()
    n = next(
        n
        for n in range(1, 4 * LIMIT)
        if post_task(client, {"id": "a", "examples": examples(40, 0.9), "context": ctx(n)}).status_code != 200
    )
    assert n > LIMIT - 60  # refused only once the row (context, question, template) reached the window
    eng.states.clear()
    r = post_task(client, {"id": "ok", "examples": examples(40, 0.9), "context": ctx(n - 1)})
    assert r.status_code == 200, r.text
    # the engine is asked one state at a time, each read with the context (never a batch of rows built up front)
    assert len(eng.states) == 40 and all(not isinstance(s, list) for s in eng.states)


def test_w4_a_dynamic_option_registration_follows_the_same_rule():
    client, _, eng = with_limit(400)
    body = {"id": "d", "options": "dynamic", "examples": examples(400, 0.96), "certify": {}}
    r = post_task(client, {**body, "context": ctx(5_000)})
    assert r.status_code == 422 and "5,000 tokens" in r.json()["detail"] and SECRET not in r.json()["detail"]
    assert eng.states == []
    long_ex = examples(400, 0.96)
    long_ex[3]["request"] = wire("0:0.9:x " + ctx(500))
    r = post_task(client, {**body, "examples": long_ex, "context": ctx(60)})
    assert r.status_code == 422 and "longest example's state" in r.json()["detail"] and eng.states == []


def test_w4_an_import_of_a_record_with_a_context_over_the_window_is_refused():
    source, _, _ = make()  # a server with no ceiling holds the task
    assert post_task(source, {"id": "a", "examples": examples(40, 0.9), "context": ctx(500)}).status_code == 200
    record = source.get("/v1/tasks?full=1").json()["tasks"][0]
    target, so, eng = with_limit()
    r = target.post("/v1/tasks/import", json={"tasks": [record]})
    assert r.status_code == 422 and "500 tokens" in r.json()["detail"] and SECRET not in r.json()["detail"]
    assert so.task_store.by_id("a") is None and eng.states == []
    record["context"] = {
        **record["context"],
        "text": "short",
        "sha256": __import__("hashlib").sha256(b"short").hexdigest(),
    }
    assert target.post("/v1/tasks/import", json={"tasks": [record]}).status_code == 200


def test_w5_a_request_whose_state_makes_the_row_exceed_the_window_is_refused_and_reads_nothing():
    client, _, eng = with_limit()
    assert post_task(client, {"id": "a", "examples": examples(40, 0.9), "context": CONTEXT}).status_code == 200
    reads = len(eng.states)
    r = ask(client, "0:0.9:x " + ctx(5_000))
    detail = r.json()["detail"]
    assert r.status_code == 422 and "the state" in detail and "the task's context" in detail and SECRET not in detail
    assert CONTEXT not in detail and len(eng.states) == reads


def test_w6_an_engine_without_vllm_configuration_answers_within_its_window():
    eng = vllm_engine.LettersEngine.__new__(vllm_engine.LettersEngine)
    assert eng.prompt_limit() is None  # no window set
    eng.context_window = 32_768
    assert eng.prompt_limit() == 32_767
    eng.over_limit_tokens = 5_000
    assert eng.prompt_limit() == 5_000  # --over-limit-tokens still wins


def test_w7_the_prefix_cache_keeps_a_context_entry_once(monkeypatch):
    from decisio.serve import mlx_engine

    sizes = {}
    monkeypatch.setattr(mlx_engine, "_cache_arrays", lambda cache: [types.SimpleNamespace(nbytes=sizes[id(cache)])])

    def entry(n):
        c = object()
        sizes[id(c)] = n
        return c

    cache = mlx_engine.PrefixCache(100)
    head = entry(60)
    cache.put([1, 2, 3], head)
    for k in range(20):  # twenty examples, each a context entry's size and a little more: none fits beside it
        cache.get([1, 2, 3], context=True)
        cache.put([1, 2, 3, 100 + k], entry(70), split=3, keep=[1, 2, 3])
    assert len(cache.entries) == 1 and cache.get([1, 2, 3]) is head and cache.bytes == 60
    cache.put([1, 2, 3, 7], entry(30), split=3, keep=[1, 2, 3])  # one that fits is kept, the context stays
    assert len(cache.entries) == 2 and cache.bytes == 90
    cache.put([1, 2, 3, 8], entry(30), split=3, keep=[1, 2, 3])  # the older state goes first, never the context
    assert len(cache.entries) == 2 and cache.get([1, 2, 3]) is head and cache.get([1, 2, 3, 7], 3) is None
    assert cache.bytes <= 100
