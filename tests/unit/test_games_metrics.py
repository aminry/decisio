# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The metrics scripts of examples/games, on players that need no server.

M1  bootstrap intervals contain the estimate, narrow with more items, and a paired interval of equal arms is zero
M2  calibration against the solver: a model that puts all its mass on the solver's letter scores perfectly, a uniform
    one agrees by chance, and mass on guessed letters is counted and masked out before the comparison
M3  metrics_maze.py writes a run record: summary, manifest, files.json whose sha256 are the files', games that resume
    (a second run adds games and plays none twice), and a registered arm refuses to pass off a leftover task
M4  metrics_hangman.py: the teacher wins nearly all words, random nearly none, the paired table and the calibration
    section are written
"""

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
GAMES = ROOT / "examples" / "games"
sys.path.insert(0, str(GAMES))

from gamelib.hangman import ALPHABET  # noqa: E402
from gamelib.metrics import (  # noqa: E402
    bootstrap,
    calibration_stats,
    ece,
    masked,
    paired_bootstrap,
    reliability,
)
from gamelib.runner import check_task_use  # noqa: E402


def run(script, *args):
    r = subprocess.run([sys.executable, GAMES / script, *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_m1_bootstrap():
    rng = np.random.default_rng(1)
    few, many = rng.normal(5, 2, 20), rng.normal(5, 2, 400)
    a, b = bootstrap(few, n=2000), bootstrap(many, n=2000)
    assert a["lo"] <= a["est"] <= a["hi"] and b["lo"] <= b["est"] <= b["hi"]
    assert b["hi"] - b["lo"] < a["hi"] - a["lo"]
    assert bootstrap(few, n=2000) == a  # seeded
    same = paired_bootstrap(many, many, n=500)
    assert same == {"est": 0.0, "lo": 0.0, "hi": 0.0}
    assert bootstrap([], n=10)["est"] is None


def test_m2_calibration_against_the_solver():
    remaining = list("abcde")
    sure = {"q": {c: (1.0 if c == "c" else 0.0) for c in remaining}, "solver": "c", "wasted": 0.0}
    flat = {"q": {c: 0.2 for c in remaining}, "solver": "c", "wasted": 0.0}
    perfect = calibration_stats([sure] * 10)
    assert perfect["agreement"] == 1.0 and perfect["nll"] < 1e-5 and perfect["brier"] == 0.0 and perfect["ece"] == 0.0
    chance = calibration_stats([flat] * 10)
    assert abs(chance["p_solver"] - 0.2) < 1e-9 and abs(chance["nll"] - np.log(5)) < 1e-9
    assert abs(chance["brier"] - (0.8**2 + 4 * 0.2**2)) < 1e-9
    probs = {c: 0.0 for c in ALPHABET}
    probs.update({"a": 0.5, "c": 0.3, "e": 0.2})  # a is already guessed
    q, wasted = masked(probs, ["b", "c", "d", "e"])
    assert abs(wasted - 0.5) < 1e-12 and abs(sum(q.values()) - 1) < 1e-12 and abs(q["c"] - 0.6) < 1e-12
    over = ece([0.9] * 10, [1] * 5 + [0] * 5)  # says 90%, right half the time
    assert abs(over - 0.4) < 1e-9
    assert reliability([0.9] * 10, [1] * 5 + [0] * 5) == [[0.8, 0.9, 10, 0.9, 0.5]]


def test_m3_maze_record_resumes(tmp_path):
    out = tmp_path / "run"
    run("metrics_maze.py", "--out", out, "--games", 6, "--players", "teacher,random", "--hardware", "test")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["arms"]["teacher"]["games"] == summary["arms"]["random"]["games"] == 6
    assert summary["arms"]["teacher"]["teacher_agreement"] == 1.0
    assert summary["arms"]["teacher"]["score"]["est"] > summary["arms"]["random"]["score"]["est"]
    assert "teacher - random" in summary["paired"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["id"] == "run" and manifest["hardware"] == "test" and manifest["typesafe_api_calls"] == 0
    files = json.loads((out / "files.json").read_text())
    assert {f["path"] for f in files} >= {"manifest.json", "summary.json", "summary.md", "games/teacher.jsonl.gz"}
    assert "files.json" not in {f["path"] for f in files}
    for f in files:
        assert hashlib.sha256((out / f["path"]).read_bytes()).hexdigest() == f["sha256"]
    run("metrics_maze.py", "--out", out, "--games", 9, "--players", "teacher,random")  # the same seeds, three more
    rows = [json.loads(line) for line in gzip.open(out / "games" / "teacher.jsonl.gz", "rt")]
    assert sorted(r["seed"] for r in rows) == list(range(9))
    assert json.loads((out / "summary.json").read_text())["arms"]["teacher"]["games"] == 9


def test_m3_a_leftover_task_is_refused():
    plain = [{"task": 3, "latency_ms": [1, 2, 3]}]
    with pytest.raises(RuntimeError, match="should have answered no move"):
        check_task_use("decisio", plain, expect_task=False)
    with pytest.raises(RuntimeError, match="every move"):
        check_task_use("decisio+task", [{"task": 2, "latency_ms": [1, 2, 3]}], expect_task=True)
    check_task_use("decisio+task", [{"task": 3, "latency_ms": [1, 2, 3]}], expect_task=True)
    check_task_use("decisio", [{"task": 0, "latency_ms": [1, 2, 3]}], expect_task=False)


def test_m4_hangman_record(tmp_path):
    out = tmp_path / "hang"
    run("metrics_hangman.py", "--out", out, "--words", 12, "--players", "teacher,random", "--hardware", "test")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["words"] == 12 and summary["teacher_states"] > 12
    assert summary["arms"]["teacher"]["win_rate"]["est"] >= 0.9 and summary["arms"]["random"]["win_rate"]["est"] <= 0.1
    assert summary["arms"]["teacher"]["teacher_agreement"] == 1.0
    assert "teacher - random" in summary["paired"] and "uniform" in summary["calibration"]
    states = json.loads((out / "teacher_states.json").read_text())
    assert len(states) == summary["teacher_states"] and {"word", "pattern", "wrong", "solver"} <= set(states[0])
    md = (out / "summary.md").read_text()
    assert "| teacher |" in md and "Paired differences" in md
