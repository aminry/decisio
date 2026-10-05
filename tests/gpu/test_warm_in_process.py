# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""--engine-process in on a card (warm_in_process.py): a --multi-question warm request repeats exactly with the engine
in the server's process.

    DECISIO_VIEW=$DECISIO_VIEW uv run pytest -m gpu tests/gpu/test_warm_in_process.py
"""

import pytest
from gpu_tier import HERE, env_path, run_child

pytestmark = pytest.mark.gpu


def test_warm_repeats_exactly_in_process():
    out = run_child(HERE / "warm_in_process.py", env_path("DECISIO_VIEW"), "--repeats", "5")
    assert "WARM REPEATS PASS" in out, out[-4000:]
