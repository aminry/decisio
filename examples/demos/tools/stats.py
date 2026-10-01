# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Bootstrap intervals, latency summaries and the run record's file list, for the demo measurement scripts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

BOOT = 10000


def bootstrap(values, stat=np.mean, n: int = BOOT, seed: int = 0) -> dict:
    """{"est", "lo", "hi"}: the statistic and its 95% percentile bootstrap interval over resampled items."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"est": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    draws = np.array([stat(v[i]) for i in idx]) if stat is not np.mean else v[idx].mean(axis=1)
    return {"est": float(stat(v)), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5))}


def latency_summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"n": 0}
    return {
        "n": len(v),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "mean": float(v.mean()),
    }


def git_state(repo) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()

    return {
        "repository": "https://github.com/aminry/decisio",
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--", "examples/demos", "src")),
    }


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_files_json(root) -> list:
    """files.json: every file under `root` (but itself) with its size and sha256, as the runs under runs/ carry."""
    root = Path(root)
    rows = [
        {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "files.json"
    ]
    (root / "files.json").write_text(json.dumps(rows, indent=1) + "\n")
    return rows
