# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The cached-block counter of the GPU tier's Qwen fill (tests/gpu/pool_blocks.py).

P1  vLLM 0.30.0's pool: cached_block_hash_to_block grows with each state and cached_block_hashes_by_block stays empty
    (Lab 2's diag_blocks.py on a card: 6 blocks per padded Qwen state), and the counter follows the first
P2  an older pool that filled only block -> hashes is still counted
P3  a pool with neither map, or an empty cache everywhere, is not read as a count to divide by: it raises or gives 0
P4  the fill arithmetic the script does with it (floor(0.9 * 2329 / 6) = 349 states), and a zero growth is refused
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
from pool_blocks import cached_blocks, growth_per_state  # noqa: E402


def pool(hash_to_block=None, by_block=None):
    ns = SimpleNamespace()
    if hash_to_block is not None:
        ns.cached_block_hash_to_block = hash_to_block
    if by_block is not None:
        ns.cached_block_hashes_by_block = by_block
    return ns


def test_p1_the_030_pool_grows_in_the_hash_map_while_the_other_stays_empty():
    p = pool({i: i for i in range(4)}, {})  # a cache with four blocks at the start, as on the card
    before = cached_blocks(p)
    for state in range(4):
        for b in range(6):  # six blocks per padded Qwen state
            p.cached_block_hash_to_block[(state, b)] = b
    assert before == 4 and (cached_blocks(p) - before) / 4 == 6
    assert len(p.cached_block_hashes_by_block) == 0  # what the test read before, and divided by


def test_p2_an_older_pool_is_counted_through_the_other_map():
    p = pool({}, {i: [i] for i in range(10)})
    assert cached_blocks(p) == 10


def test_p3_a_pool_without_a_count_is_never_a_divisor():
    with pytest.raises(AttributeError, match="cached_block_hash_to_block"):
        cached_blocks(pool())
    assert cached_blocks(pool({}, {})) == 0


def test_p4_the_fill_arithmetic_and_the_zero_guard():
    kept = 6  # blocks per padded Qwen state, from Lab 2's diagnostic
    assert math.floor(0.9 * 2329 / kept) == 349
    assert growth_per_state(4, 28, 4) == 6
    for before, after in ((4, 4), (10, 4)):  # no growth, or a cache that shrank: refused before any division
        with pytest.raises(ValueError, match="did not grow"):
            growth_per_state(before, after, 4)
