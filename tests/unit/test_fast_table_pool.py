# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The fast table's encoder pool (`decisio.serve.fast_table`: ENCODERS sessions, each used by one encode at a time).

P1  at most ENCODERS encodes run at once, a further one waits for the first that ends, and every caller gets its own
    texts' embeddings
P2  the pool is loaded once, its other encoders made from the first one's checked files (no second fetch or hash)
P3  the timing stack records the pool's size, and a replica with another size differs in the stack (so it re-times)
P4  [LOCAL ONLY: needs the pinned files in the local Hugging Face cache] the pool's embeddings, made concurrently, equal
    one encoder's made one at a time, bit for bit

  uv run pytest -q tests/unit/test_fast_table_pool.py
"""

import concurrent.futures as cf
import threading
import time

import numpy as np
import pytest

from decisio.serve import fast_table as FT


class Slow:
    """An encoder that takes 50 ms per call, counts the calls in flight and embeds a text as its length."""

    live = 0
    peak = 0
    lock = threading.Lock()
    files = ("model.onnx", "tokenizer.json")
    made = []

    def __init__(self):
        type(self).made.append(self)

    def encode(self, texts):
        with Slow.lock:
            Slow.live += 1
            Slow.peak = max(Slow.peak, Slow.live)
        time.sleep(0.05)
        with Slow.lock:
            Slow.live -= 1
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

    def another(self):
        return Slow()


@pytest.fixture
def slow_pool(monkeypatch):
    Slow.live = Slow.peak = 0
    Slow.made = []
    loads = []
    monkeypatch.setattr(FT, "_encoder", None)
    monkeypatch.setattr(FT, "_pool", None)
    monkeypatch.setattr(FT.OnnxEncoder, "load", classmethod(lambda cls, **k: loads.append(1) or Slow()))
    return loads


def test_p1_at_most_encoders_at_once_and_each_caller_gets_its_own(slow_pool):
    texts = [["x" * (i + 1)] for i in range(8)]
    with cf.ThreadPoolExecutor(8) as ex:
        out = list(ex.map(FT.embed, texts))
    assert Slow.peak == FT.ENCODERS == 2
    assert [float(o[0, 0]) for o in out] == [float(i + 1) for i in range(8)]


def test_p2_the_pool_is_loaded_once_from_the_first_encoders_files(slow_pool):
    FT.embed(["a"])
    FT.embed(["b"])
    assert slow_pool == [1] and len(Slow.made) == FT.ENCODERS  # one fetch and check, the rest from its files
    assert FT.encoder() is Slow.made[0]


def test_p3_the_timing_stack_records_the_pool(monkeypatch):
    a = FT.timing_stack({"engine": "fake"})
    assert a["encoders"] == FT.ENCODERS and a["threads"] == FT.THREADS
    monkeypatch.setattr(FT, "ENCODERS", 3)
    assert FT.stack_differences(a, FT.timing_stack({"engine": "fake"})) == ["encoders"]


def _files_cached():
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(FT.ENCODER, FT.ONNX_FILE, revision=FT.REVISION, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _files_cached(), reason="needs the pinned encoder files in the local Hugging Face cache")
def test_p4_the_pools_embeddings_equal_one_encoders(monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.setattr(FT, "_encoder", None)
    monkeypatch.setattr(FT, "_pool", None)
    one = FT.OnnxEncoder.load(local_files_only=True)
    rng = np.random.default_rng(0)
    words = "invoice refund parcel courier password login account the a not please order".split()
    texts = [" ".join(rng.choice(words, size=int(rng.integers(3, 200)))) for _ in range(24)]
    serial = np.vstack([one.encode([t]) for t in texts])
    monkeypatch.setattr(FT.OnnxEncoder, "load", classmethod(lambda cls, **k: FT.OnnxEncoder.from_files(*one.files)))
    with cf.ThreadPoolExecutor(8) as ex:
        pooled = np.vstack(list(ex.map(lambda t: FT.embed([t]), texts)))
    assert np.array_equal(serial, pooled)
