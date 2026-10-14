# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The slow tier's split across CI jobs (scripts/shard_slow_tier.py).

S1  every file lands in exactly one shard, whatever the number of shards
S2  the shards are balanced: the longest is within a third of the mean on the recorded files (3 shards)
S3  a file the table does not know still lands in a shard

  uv run pytest -q tests/unit/test_shard_slow_tier.py
"""

import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "scripts" / "shard_slow_tier.py"
spec = importlib.util.spec_from_file_location("shard_slow_tier", PATH)
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

FILES = sorted(st.SECONDS)


def test_s1_every_file_in_exactly_one_shard():
    for n in (1, 2, 3, 4, 7):
        got = [f for s in st.shards(FILES, n) for f in s]
        assert sorted(got) == FILES, n


def test_s2_the_shards_are_balanced():
    total = sum(st.SECONDS.values())
    loads = [sum(st.SECONDS[f] for f in s) for s in st.shards(FILES, 3)]
    assert max(loads) <= total / 3 * 4 / 3 + 1, loads  # the longest file alone is 246 of 960 seconds


def test_s3_an_unknown_file_lands_somewhere():
    files = [*FILES, "tests/unit/test_new_slow.py"]
    got = [f for s in st.shards(files, 3) for f in s]
    assert sorted(got) == sorted(files)
