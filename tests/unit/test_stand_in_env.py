# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Every test that loads the CPU stand-in (Qwen3 0.6B base) reads it through `DECISIO_STAND_IN_MODEL`. CI fetches the
stand-in once at a pinned revision into a cached directory and points the variable at it; a test that names the model
on the Hub instead asks the Hub for its files at every load, follows a moving branch, and fails when the Hub rate-limits
the runner (a 429).

S1  the stand-in's Hub name appears in a test only as the variable's default.
"""

import re
from pathlib import Path

NAME = "Qwen/Qwen3-0.6B-Base"
READ = f'os.environ.get("DECISIO_STAND_IN_MODEL", "{NAME}")'
# names passed as plain strings to argument and fingerprint logic, never loaded
NOT_LOADED = {"test_gemma_base.py"}


def test_s1_stand_in_read_through_the_variable():
    here = Path(__file__).parent
    named = []
    for p in sorted(here.glob("*.py")):
        if p.name in NOT_LOADED or p.name == Path(__file__).name:
            continue
        text = p.read_text()
        bare = len(re.findall(re.escape(f'"{NAME}"'), text)) - text.count(READ)
        if bare:
            named.append(p.name)
    assert not named, f"read the stand-in through DECISIO_STAND_IN_MODEL ({READ}) in: {named}"
