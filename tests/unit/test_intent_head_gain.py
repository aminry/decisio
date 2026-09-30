# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The single-engine intent-head intervals the guide cites are what benchmarks/intent_head_gain.py computes from the
record's per-item rows (runs/2026-09-30_plugin-verification/derived/intent_head_gains.json).

    uv run pytest -q tests/unit/test_intent_head_gain.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "2026-09-30_plugin-verification"


def test_stored_gains_are_reproduced():
    r = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "intent_head_gain.py"), str(RUN)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == json.loads((RUN / "derived" / "intent_head_gains.json").read_text())
    tasks = json.loads(r.stdout)["tasks"]
    assert (tasks["banking77"]["gain_points"], tasks["banking77"]["interval_95_points"]) == (10.7, [5.8, 15.8])
    assert (tasks["v3_clinc"]["gain_points"], tasks["v3_clinc"]["interval_95_points"]) == (7.3, [2.3, 13.0])
