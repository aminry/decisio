# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The engine's tokenizer, one caller at a time.

vLLM 0.31.0 makes its Hugging Face tokenizer thread-safe with a pool of copies (`vllm/tokenizers/hf.py`,
`maybe_make_thread_pool`): a call that finds no free copy makes a new one with `copy.deepcopy` of the whole tokenizer,
which for Gemma 4 (262,144 tokens) is seconds of pure Python holding the interpreter lock, and the copy joins the pool.
decisio's request threads tokenize (the usage count, the over-limit cut, the question rows), so when many requests are
answered together (`--concurrency batched`: up to 16 at the end of one engine call) many threads tokenize at once and
each deep-copies the tokenizer; the engine's own thread then waits for the interpreter lock with the card idle. On an
RTX 5090 that stalled engine calls for seconds and made batched mode slower than serial (RLCD
experiments/2026-10-21_throughput/session2).

`SerialTokenizer` wraps the tokenizer so that its methods run one call at a time: the pool never finds every copy in
use, so it never copies. A call is about a millisecond of Rust (the state's tokenization, once per request), so the lock
costs nothing measurable; every other attribute passes through unchanged.
"""

from __future__ import annotations

import threading


class SerialTokenizer:
    def __init__(self, tokenizer):
        object.__setattr__(self, "_tok", tokenizer)
        object.__setattr__(self, "_lock", threading.Lock())

    @property
    def wrapped(self):
        return self._tok

    def __getattr__(self, name):
        attr = getattr(self._tok, name)
        if not callable(attr) or isinstance(attr, type):
            return attr
        lock = self._lock

        def call(*args, **kwargs):
            with lock:
                return attr(*args, **kwargs)

        call.__name__ = getattr(attr, "__name__", name)
        return call

    def __setattr__(self, name, value):
        setattr(self._tok, name, value)

    def __call__(self, *args, **kwargs):
        with self._lock:
            return self._tok(*args, **kwargs)

    def __len__(self):
        return len(self._tok)

    def __repr__(self):
        return f"SerialTokenizer({self._tok!r})"
