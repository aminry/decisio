# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The restart check's comparison of two starts (runs/2026-10-14_certified-answers/restart_check.py).

D1  identical starts differ in nothing
D2  a certified flag that moves with a small change of the top probability is counted as noise, one that moves with a
    large change is not
D3  a certified answer whose choice moved is counted apart from an uncertified one

  uv run pytest -q tests/unit/test_restart_check_diff.py
"""

import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "runs" / "2026-10-14_certified-answers" / "restart_check.py"
spec = importlib.util.spec_from_file_location("restart_check", PATH)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

CERT = {"threshold": 0.9, "id": "t#1", "stack_sha256": "s"}


def start(answers):
    return {"certificate": dict(CERT), "answers": answers}


def a(top, certified, choice="x"):
    return {"top": top, "certified": certified, "choice": choice}


def test_d1_identical_starts_differ_in_nothing():
    x = start([a(0.95, True), a(0.5, False)])
    d = rc.diff(x, start([dict(i) for i in x["answers"]]))
    assert d["certified_flags_that_differ"] == d["choices_that_differ"] == d["certified_choices_that_differ"] == 0
    assert d["threshold_equal"] and d["certificate_id_equal"] and d["stack_digest_equal"]


def test_d2_a_flag_flip_is_noise_only_when_the_probability_moved_little():
    x = start([a(0.905, True), a(0.95, True)])
    y = start([a(0.895, False), a(0.80, False)])  # 0.01 and 0.15
    d = rc.diff(x, y)
    assert (
        d["certified_flags_that_differ"] == 2
        and d["flag_flips_within_noise"] == 1
        and d["flag_flips_beyond_noise"] == 1
    )


def test_d3_a_certified_choice_that_moved_is_counted_apart():
    x = start([a(0.95, True, "billing"), a(0.4, False, "billing")])
    y = start([a(0.95, True, "access"), a(0.4, False, "access")])
    d = rc.diff(x, y)
    assert d["choices_that_differ"] == 2 and d["certified_choices_that_differ"] == 1
