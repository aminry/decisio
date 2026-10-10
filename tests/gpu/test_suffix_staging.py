# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The suffix-staging patch series' own tests (patches/vllm-0.31.0/suffix-staging, upstream's test file carried in
0001), against the vLLM installed in this environment after `bash patches/apply.sh`. The CUDA cases run the real
`RequestState` staging and prefill gather kernel against full staging, with stale data in the reused slot.

    uv run pytest -m gpu tests/gpu/test_suffix_staging.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
SERIES = Path(__file__).resolve().parents[2] / "patches" / "vllm-0.31.0" / "suffix-staging"


def test_series_unit_tests(tmp_path):
    pytest.importorskip("vllm.v1.worker.gpu.suffix_staging", reason="vLLM without the series (bash patches/apply.sh)")
    patch = next(SERIES.glob("0001-*.patch"))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "apply", "--include=tests/*", str(patch)], check=True)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/v1/worker/test_gpu_suffix_staging.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
