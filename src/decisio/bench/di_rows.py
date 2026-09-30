# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The rows file of the four Decision Index 0.2.1 benchmarks the runs in runs/ measured, taken from the kit's rebuilt
suite directory: its selected rows of BANKING77, CLINC150+OOS and GPQA Diamond, then its added rows of MMLU-Pro, in
the suite's order. Checks the counts per benchmark and, by default, the uncompressed sha256 the runs used.

    python -m decisio.bench.di_rows <suite dir> <out rows.jsonl.gz> [--no-hash-check]
"""

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

FAMILIES = {"BANKING77": 3080, "CLINC150+OOS": 5500, "GPQA-Diamond": 198, "MMLU-Pro": 12032}
RUNS_SHA256 = "9b8537423f834be743373e3b033ea367b7c4e196b61be1b8480cddab0338ce80"  # uncompressed, runs/ 2026-09-27, -30


def rows(suite: Path) -> list[str]:
    out = []
    for name in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        with gzip.open(suite / name, "rt") as f:
            out += [line for line in f if json.loads(line)["family"] in FAMILIES]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-hash-check", action="store_true")
    a = ap.parse_args()
    got = rows(a.suite)
    counts = collections.Counter(json.loads(line)["family"] for line in got)
    if dict(counts) != FAMILIES:
        raise SystemExit(f"row counts {dict(counts)} differ from the runs' {FAMILIES}")
    digest = hashlib.sha256("".join(got).encode()).hexdigest()
    if digest != RUNS_SHA256 and not a.no_hash_check:
        raise SystemExit(f"rows sha256 {digest} differs from the runs' {RUNS_SHA256} (--no-hash-check to go on)")
    with gzip.GzipFile(a.out, "wb", mtime=0) as f:
        f.write("".join(got).encode())
    print(f"{sum(counts.values())} rows {dict(counts)} sha256 {digest} -> {a.out}")


if __name__ == "__main__":
    main()
