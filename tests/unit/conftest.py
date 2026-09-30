# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The CPU tier. Tests that serve through the CPU stand-in load a small Hugging Face model (the `served` fixtures, the
tests marked in place) and are marked `slow`; `-m "not slow"` runs the rest in under a minute."""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "served" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.slow)
