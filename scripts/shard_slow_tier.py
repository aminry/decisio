# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Split the slow tier's test files across CI jobs so that each job takes about as long as the others.

The files come on standard input, one path per line (the paths `pytest --collect-only -q -m slow` collects, once each);
the files of shard INDEX of SHARDS go to standard output. Each file goes to the shard with the least time so far, the
longest first; a file not in SECONDS counts as the median of those that are, so a new file lands somewhere sensible
and the weights are refreshed from a run's job summary (the seconds column) whenever they drift.

  python3 scripts/shard_slow_tier.py SHARDS INDEX < files
"""

from __future__ import annotations

import statistics
import sys

# Seconds per file on a hosted ubuntu-latest runner: the nightly of 2026-10-14 (.github/workflows/gpu.yml).
SECONDS = {
    "tests/unit/test_abstention.py": 37,
    "tests/unit/test_boundary_after_response.py": 47,
    "tests/unit/test_head_modes.py": 95,
    "tests/unit/test_hidden_readout.py": 168,
    "tests/unit/test_image_route.py": 50,
    "tests/unit/test_names.py": 5,
    "tests/unit/test_tasks.py": 159,
    "tests/unit/test_tasks_walkthrough.py": 340,
    "tests/unit/test_temperature.py": 64,
    "tests/unit/test_ties.py": 5,
}


def shards(files: list[str], n: int) -> list[list[str]]:
    default = statistics.median(SECONDS.values())
    load, out = [0.0] * n, [[] for _ in range(n)]
    for f in sorted(files, key=lambda f: (-SECONDS.get(f, default), f)):
        i = load.index(min(load))
        load[i] += SECONDS.get(f, default)
        out[i].append(f)
    return out


def main(argv: list[str]) -> int:
    n, index = int(argv[1]), int(argv[2])
    if not 0 <= index < n:
        print(f"INDEX must be in 0..{n - 1}", file=sys.stderr)
        return 2
    files = [line.strip() for line in sys.stdin if line.strip()]
    for f in sorted(shards(files, n)[index]):
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
