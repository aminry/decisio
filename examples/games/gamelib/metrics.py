# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Shared by the metrics scripts: bootstrap intervals, latency summaries, calibration against the solver, player
specifications, and the run record (a manifest and a files.json with every file's sha256, as under `runs/`)."""

import gzip
import hashlib
import json
import math
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

from .hangman import ALPHABET
from .players import make_player

BOOT = 10000


def bootstrap(values, stat=np.mean, n=BOOT, seed=0):
    """{"est", "lo", "hi"}: the statistic and its 95% percentile bootstrap interval over resampled items."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"est": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    draws = np.array([stat(v[i]) for i in idx]) if stat is not np.mean else v[idx].mean(axis=1)
    return {"est": float(stat(v)), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5))}


def paired_bootstrap(a, b, n=BOOT, seed=0):
    """The mean of a - b over the same items, with its 95% paired bootstrap interval."""
    return bootstrap(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n=n, seed=seed)


def cluster_bootstrap(clusters, stat, n=BOOT, seed=0):
    """`stat(list of items)` over resampled clusters (a word's states stay together), 95% interval."""
    rng = np.random.default_rng(seed)
    keys = list(clusters)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), size=len(keys))
        draws.append(stat([item for i in pick for item in clusters[keys[i]]]))
    flat = [item for k in keys for item in clusters[k]]
    return {"est": float(stat(flat)), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5))}


def latency_summary(values):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"n": 0}
    return {
        "n": len(v),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "mean": float(v.mean()),
    }


# ---- calibration of letter probabilities against the solver ---------------------------------------------------------


def masked(probs, remaining):
    """The distribution over the unguessed letters only, renormalised, and the raw mass that sat on guessed ones."""
    total = sum(probs.values()) or 1.0
    kept = {c: probs.get(c, 0.0) for c in remaining}
    mass = sum(kept.values())
    wasted = 1.0 - mass / total
    return ({c: p / mass for c, p in kept.items()} if mass > 0 else {c: 1 / len(remaining) for c in remaining}), wasted


def ece(conf, hit, bins=10):
    conf, hit = np.asarray(conf, dtype=float), np.asarray(hit, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.sum() / n * abs(hit[in_bin].mean() - conf[in_bin].mean())
    return float(total)


def reliability(conf, hit, bins=10):
    """[(bin low, bin high, n, mean confidence, agreement rate)] for the bins that hold any state."""
    conf, hit = np.asarray(conf, dtype=float), np.asarray(hit, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            out.append(
                [round(lo, 2), round(hi, 2), int(in_bin.sum()), float(conf[in_bin].mean()), float(hit[in_bin].mean())]
            )
    return out


def calibration_stats(items):
    """Measures of a list of {"solver", "q" (masked distribution), "wasted"} against the solver's choice:
    agreement of the model's top letter with the solver's, mean probability of the solver's letter, its negative log
    likelihood, the Brier score of the distribution against the solver's one-hot choice, the calibration error of the
    top letter's confidence as a predictor of agreeing, and the raw probability mass that sat on guessed letters."""
    conf, hit, p_solver, brier, wasted = [], [], [], [], []
    for it in items:
        q, solver = it["q"], it["solver"]
        top = max(q, key=q.get)
        conf.append(q[top])
        hit.append(top == solver)
        p_solver.append(q[solver])
        brier.append(sum((p - (c == solver)) ** 2 for c, p in q.items()))
        wasted.append(it["wasted"])
    return {
        "n": len(items),
        "agreement": float(np.mean(hit)),
        "p_solver": float(np.mean(p_solver)),
        "nll": float(np.mean([-math.log(max(p, 1e-6)) for p in p_solver])),
        "brier": float(np.mean(brier)),
        "ece": ece(conf, hit),
        "confidence": float(np.mean(conf)),
        "mass_on_guessed": float(np.mean(wasted)),
    }


# ---- players -------------------------------------------------------------------------------------------------------


def parse_players(spec, kind, url=None, render=None, path=None):
    """[(arm name, player)] from "teacher,random,decisio,cygnet,..."; `url` and `path` go to every player."""
    names = [s.strip() for s in spec.split(",") if s.strip()]
    return [(n, make_player(kind, n, url, render=render, path=path)) for n in names]


def server_info(player):
    """What a decisio server says about itself (/health), for the record."""
    desc = player.describe()
    if desc.get("kind") != "decisio":
        return None
    try:
        with urllib.request.urlopen(desc["url"] + "/health", timeout=10) as r:
            info = json.loads(r.read())
    except OSError as e:
        return {"error": str(e)}
    keys = ("engine", "model", "mode", "block_size", "pad_unit", "adapters", "systemone", "head_engine")
    return {k: info.get(k) for k in keys}


# ---- the run record --------------------------------------------------------------------------------------------------


def git_state(repo):
    def run(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()

    return {
        "repository": "https://github.com/aminry/decisio",
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--", "examples/games", "src")),
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_files_json(root):
    """files.json: every file under `root` (but itself) with its size and sha256, as the runs under runs/ carry."""
    root = Path(root)
    rows = [
        {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "files.json"
    ]
    (root / "files.json").write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def gzip_file(path):
    """Compress `path` to path.gz and remove it."""
    path = Path(path)
    with open(path, "rb") as src, gzip.open(str(path) + ".gz", "wb", compresslevel=9) as dst:
        dst.writelines(src)
    path.unlink()


def gunzip_file(path):
    """Restore path from path.gz (a finished arm being resumed or extended), removing the .gz."""
    path = Path(path)
    gz = Path(str(path) + ".gz")
    if gz.exists() and not path.exists():
        with gzip.open(gz, "rb") as src, open(path, "wb") as dst:
            dst.writelines(src)
        gz.unlink()


def read_jsonl(path):
    path = Path(path)
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    gz = Path(str(path) + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt") as f:
            return [json.loads(line) for line in f if line.strip()]
    return []


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt_ci(d, scale=1.0, digits=3, pct=False):
    if d["est"] is None:
        return "n/a"
    f = f"{{:.{digits}f}}"
    if pct:
        return f"{d['est'] * 100:.1f}% [{d['lo'] * 100:.1f}, {d['hi'] * 100:.1f}]"
    return f"{f.format(d['est'] * scale)} [{f.format(d['lo'] * scale)}, {f.format(d['hi'] * scale)}]"


LETTERS = ALPHABET
