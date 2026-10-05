# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Run by test_warm_in_process.py on a card: with the engine in the server's process (--engine-process in), a request
of several questions scored in one batch (--multi-question warm) returns the same probabilities on every repeat.

In vLLM's own arrangement the engine runs in a process of its own and receives a batch's requests one at a time, so
the questions do not always share an engine step and the answers can move between repeats; in the server's process
every question is added before the first step (vllm-project/vllm#59764, EVAL_CARD.md section 4).

    python tests/gpu/warm_in_process.py $DECISIO_VIEW [--repeats 5]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from decisio.serve import vllm_engine as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from test_prompts import QUESTIONS, STATE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--engine", default=json.dumps({"compilation_config": {"max_cudagraph_capture_size": 4096}}))
    a = ap.parse_args()
    assert sv.engine_process_guard("vllm", "in", os.environ)[0] == "in"
    eng = sv.LettersEngine(a.model, mode="separate", pad_to="block", pad_where="front", engine_kw=json.loads(a.engine))
    eng.multi_question = "warm"
    facts = eng.facts()
    print("ENGINE", facts)
    assert facts["engine_process"] == "in", facts
    answers = [eng.answer(STATE, QUESTIONS)[0] for _ in range(a.repeats)]
    first = [np.asarray(p) for p in answers[0]]
    same = all(all(np.array_equal(np.asarray(p), f) for p, f in zip(ans, first, strict=True)) for ans in answers[1:])
    worst = max(float(np.abs(np.asarray(p) - f).max()) for ans in answers[1:] for p, f in zip(ans, first, strict=True))
    print(
        f"WARM REPEATS {'PASS' if same else 'FAIL'}: {len(QUESTIONS)} questions, {a.repeats} repeats, max |dp| {worst}"
    )
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
