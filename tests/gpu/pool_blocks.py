# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""How many blocks vLLM's prefix cache holds, read from its block pool (no vLLM import: unit-tested on the CPU).

vLLM 0.30.0 keeps them in `BlockPool.cached_block_hash_to_block` (hash -> block). `cached_block_hashes_by_block`
(block -> hashes), which this test read before, exists but stays empty in 0.30.0, so the Qwen fill of
second_question_cached.py (--fill-by blocks) divided by zero (Lab 2's diag_blocks.py on a card, RLCD
experiments/2026-10-08_lab2_gate_session: 6 new blocks per padded Qwen state in a pool of 2,329 usable blocks)."""

# the maps a vLLM release has used for it, in the order they are tried
COUNTERS = ("cached_block_hash_to_block", "cached_block_hashes_by_block")


def cached_blocks(pool):
    """The number of cached blocks of a pool: the first of COUNTERS that is not empty (an older vLLM filled the second,
    0.30.0 fills the first). Raises when the pool has none of them, or all are empty: an empty cache is a reading the
    caller must not divide by."""
    found = [getattr(pool, name) for name in COUNTERS if hasattr(pool, name)]
    if not found:
        raise AttributeError(f"the block pool has none of {COUNTERS}: this vLLM keeps its cached blocks elsewhere")
    sizes = [len(m) for m in found]
    return next((n for n in sizes if n), 0)


def growth_per_state(before, after, states):
    """Blocks the cache gained per new state; raises when it did not grow, so the fill is never divided by zero."""
    grown = (after - before) / states
    if grown <= 0:
        raise ValueError(f"the pool's cached blocks did not grow over {states} new states ({before} to {after})")
    return grown
