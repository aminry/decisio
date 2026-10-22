# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The engine's tokenizer, one caller at a time (decisio.serve.tokenizer_lock).

T1  calls through SerialTokenizer never overlap, from any number of threads; attributes, `len` and calling the
    tokenizer pass through; setting an attribute sets it on the tokenizer
T2  against a pool like vLLM 0.31.0's (`maybe_make_thread_pool`: a caller that finds no free copy deep-copies the
    tokenizer), 16 threads tokenizing at once make copies without the wrapper and none with it
T3  the vLLM engine wraps the tokenizer it is given

  uv run pytest -q tests/unit/test_tokenizer_lock.py
"""

import contextlib
import copy
import queue
import sys
import threading
import time
import types

from decisio.serve.tokenizer_lock import SerialTokenizer


class Slow:
    """A tokenizer whose encode takes a moment and records how many callers are inside it at once."""

    vocab_size = 7
    inside = 0
    most = 0
    guard = threading.Lock()

    def encode(self, text, add_special_tokens=False):
        with Slow.guard:
            Slow.inside += 1
            Slow.most = max(Slow.most, Slow.inside)
        time.sleep(0.002)
        with Slow.guard:
            Slow.inside -= 1
        return [len(text)]

    def __call__(self, text):
        return {"input_ids": self.encode(text)}

    def __len__(self):
        return 7


def hammer(fn, n=16, each=10):
    threads = [threading.Thread(target=lambda: [fn(f"text {i}") for i in range(each)]) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)


def test_t1_one_call_at_a_time_and_everything_else_passes_through():
    Slow.most = 0
    hammer(Slow().encode)
    assert Slow.most > 1  # the bare tokenizer is entered by several threads at once
    Slow.most = 0
    tok = SerialTokenizer(Slow())
    hammer(tok.encode)
    assert Slow.most == 1
    assert tok.encode("abc") == [3] and tok("abcd") == {"input_ids": [4]} and len(tok) == 7 and tok.vocab_size == 7
    tok.padding_side = "left"
    assert tok.wrapped.padding_side == "left" and tok.padding_side == "left"


def pooled(tokenizer):
    """vLLM 0.31.0's pool, as `maybe_make_thread_pool` builds it: one copy to start, a deep copy for any caller that
    finds none free; counts the copies made after the first."""
    og, pool, made = copy.copy(tokenizer), queue.Queue(), []
    pool.put(copy.deepcopy(og))

    @contextlib.contextmanager
    def borrow():
        try:
            tok = pool.get_nowait()
        except queue.Empty:
            tok = copy.deepcopy(og)
            made.append(1)
        try:
            yield tok
        finally:
            pool.put(tok)

    class Pool:
        def encode(self, *a, **k):
            with borrow() as tok:
                return tok.encode(*a, **k)

    return Pool(), made


def test_t2_concurrent_tokenizing_makes_no_copy_through_the_wrapper():
    bare, made = pooled(Slow())
    hammer(bare.encode)
    assert len(made) > 0  # what the stall was: copies of the whole tokenizer under concurrent callers
    wrapped, made = pooled(Slow())
    hammer(SerialTokenizer(wrapped).encode)
    assert made == []


def test_t3_the_engine_wraps_its_tokenizer(monkeypatch):
    from decisio.serve import vllm_engine

    class FakeLLM:
        def __init__(self, **kw):
            mc = types.SimpleNamespace(max_model_len=kw["max_model_len"])
            cc = types.SimpleNamespace(block_size=16, prefix_match_unit=None)
            self.llm_engine = types.SimpleNamespace(vllm_config=types.SimpleNamespace(model_config=mc, cache_config=cc))

        def get_tokenizer(self):
            return Slow()

        def collective_rpc(self, name):
            return [None]

    mod = types.ModuleType("vllm")
    mod.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setattr(vllm_engine.LettersEngine, "_warm_up", lambda self: None)
    from decisio.families import BASES

    e = vllm_engine.LettersEngine("stand-in", family=BASES["gemma-4-12b"], max_model_len=32768)
    assert isinstance(e.tok, SerialTokenizer) and isinstance(e.tok.wrapped, Slow)
