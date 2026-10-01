# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The run record of a metrics script: a directory with one resumable file of games per arm, a log, the tables, a
manifest and a files.json (as under `runs/`)."""

import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

from .metrics import git_state, gunzip_file, gzip_file, paired_bootstrap, read_jsonl, write_files_json
from .players import play


class LoadMonitor(threading.Thread):
    """Samples every process's CPU every `period` seconds and keeps the busiest one that is not this script, so that a
    latency measured on a shared machine says how idle it was (the rule: at most 25%). macOS's WindowServer, which draws
    the screen and sits near 25% on this Mac whatever runs, is not counted."""

    LIMIT = 25.0
    IGNORED = ("windowserver",)

    def __init__(self, period=10.0):
        super().__init__(daemon=True)
        self.period, self.stop_event = period, threading.Event()
        self.worst, self.process, self.samples = 0.0, "", 0
        self.load_start = os.getloadavg()

    def run(self):
        me = os.getpid()
        while not self.stop_event.is_set():
            ps = subprocess.run(["ps", "-A", "-o", "pid=,pcpu=,comm="], capture_output=True, text=True).stdout
            for line in ps.splitlines():
                parts = line.split(None, 2)
                if len(parts) < 3 or int(parts[0]) == me or any(x in parts[2].lower() for x in self.IGNORED):
                    continue
                if float(parts[1]) > self.worst:
                    self.worst, self.process = float(parts[1]), parts[2].strip()[-60:]
            self.samples += 1
            self.stop_event.wait(self.period)

    def report(self):
        self.stop_event.set()
        return {
            "max_other_process_cpu_pct": self.worst,
            "busiest_other_process": self.process,
            "idle_by_the_25pct_rule": self.worst <= self.LIMIT,
            "samples": self.samples,
            "loadavg_start": list(self.load_start),
            "loadavg_end": list(os.getloadavg()),
        }


class Run:
    def __init__(self, out, title, game, hardware=""):
        self.out, self.title, self.game, self.hardware = Path(out), title, game, hardware
        (self.out / "games").mkdir(parents=True, exist_ok=True)
        self.logf = open(self.out / "log.txt", "a")
        self.monitor = LoadMonitor() if platform.system() == "Darwin" else None
        if self.monitor:
            self.monitor.start()

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def arm_path(self, name):
        safe = name.replace(":", "_").replace("/", "_")
        return self.out / "games" / f"{safe}.jsonl"

    def play_arm(self, name, player, keys, make_game, row_for, teacher=None):
        """Play one game per key (a seed or a word) with `player`, appending a row to the arm's file as each ends; keys
        already in the file are skipped, so a rerun resumes. Returns every row of the arm."""
        path = self.arm_path(name)
        gunzip_file(path)
        done = {r["key"] for r in read_jsonl(path)}
        todo = [k for k in keys if k not in done]
        if done:
            self.log(f"{name}: {len(done)} games already written, {len(todo)} to play")
        t0 = time.perf_counter()
        with open(path, "a") as f:
            for i, key in enumerate(todo, 1):
                game = make_game(key)
                record = play(game, player, self.game, key, teacher=teacher)
                f.write(json.dumps({"key": key, **row_for(record, game)}) + "\n")
                f.flush()
                if i % 10 == 0 or i == len(todo):
                    self.log(f"{name}: {i}/{len(todo)} games, {time.perf_counter() - t0:.0f} s")
        return read_jsonl(path)

    def finish(self, result, markdown, manifest):
        """Write summary.json and summary.md, compress the games, write the manifest and files.json."""
        (self.out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
        (self.out / "summary.md").write_text(markdown + "\n")
        manifest = {
            "id": self.out.name,
            "title": self.title,
            "date": time.strftime("%Y-%m-%d"),
            "submitted": False,
            "typesafe_api_calls": 0,
            "hardware": self.hardware,
            "code": git_state(Path(__file__).resolve().parents[3]),
            "game": self.game,
            "command": " ".join(sys.argv),
            **manifest,
        }
        if self.monitor:
            manifest["machine_load"] = self.monitor.report()
        for f in (self.out / "games").glob("*.jsonl"):
            gzip_file(f)
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        self.logf.close()
        write_files_json(self.out)


def check_task_use(name, rows, expect_task, moves_of=lambda r: len(r["latency_ms"])):
    """A plain arm must have been answered by no task and a registered arm by the task on every move; a leftover task
    from an earlier session would otherwise make two arms identical without anyone noticing."""
    served = sum(r["task"] for r in rows)
    total = sum(moves_of(r) for r in rows)
    if (expect_task and served != total) or (not expect_task and served):
        want = "every move" if expect_task else "no move"
        raise RuntimeError(f"{name}: a task answered {served} of {total} moves; it should have answered {want}")


def paired(arms, key, metrics, pairs):
    """{"a - b": {metric: paired bootstrap, "games": n}} over the games two arms share (matched on row["key"]).
    `arms` maps a name to its rows, `metrics` a name to a function of a row, `pairs` is [(a, b)]."""
    out = {}
    for a, b in pairs:
        ra, rb = {r[key]: r for r in arms[a]}, {r[key]: r for r in arms[b]}
        common = sorted(set(ra) & set(rb), key=str)
        out[f"{a} - {b}"] = {"games": len(common)}
        for name, fn in metrics.items():
            out[f"{a} - {b}"][name] = paired_bootstrap([fn(ra[k]) for k in common], [fn(rb[k]) for k in common])
    return out


def reference_pairs(names, refs):
    """[(a, b)] for every arm against each reference arm present, without a pair and its mirror."""
    pairs = []
    for ref in refs:
        if ref in names:
            pairs += [
                (n, ref)
                for n in names
                if n != ref and (ref, n) not in pairs and not (n in refs and refs.index(n) < refs.index(ref))
            ]
    return pairs
