# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The letters engine's prompt construction, with the served model's tokenizer and no model (the CPU half of
tests/gpu/serving_gates.py): C1 separate-mode prompts equal the letters prompt in chat format and their label tokens;
C2 the state split depends on the state alone and padding inserts exactly enough tokens at the chosen place; C3 a
packed prompt's first turn equals its separate prompt and every read position is the final ':' of "Answer:".

    uv run pytest -q tests/unit/test_prompts.py        (downloads the tokenizer of Qwen/Qwen3.6-35B-A3B once)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
from serving_gates import cpu_gates  # noqa: E402

TOKENIZER = os.environ.get("DECISIO_TOKENIZER", "Qwen/Qwen3.6-35B-A3B")


def test_c1_c2_c3_prompt_construction(capsys):
    ok = cpu_gates(TOKENIZER)
    assert ok, capsys.readouterr().out
