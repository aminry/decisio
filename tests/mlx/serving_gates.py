# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The serving gates of the MLX engine (decisio.serve.mlx_engine), the Mac counterpart of tests/gpu/serving_gates.py.

CPU part: C1, C2 and C3 of tests/unit/test_prompts.py, unchanged (the prompts are the served code's).

MLX part (`--model`, on Apple silicon):
  G1  the prefix path against the whole prompt in one pass: reported, not gated. The MLX engine scores every question
      from the state prefix, single-question requests included, so the whole-prompt path is never served; the
      difference is a chunking effect of bf16 arithmetic (what vLLM's G1 bounds for its cache), recorded for reference.
  G2  isolation, exact: 16 extra questions in a request leave every answer bit-identical, each question asked alone
      equals the same question inside the request bit for bit, and a repeated request is bit-identical.
  G3, G4 (packed mode, LoRA adapters): not applicable; the MLX engine serves neither.

    python tests/mlx/serving_gates.py --model mlx-community/Qwen3.6-35B-A3B-4bit [--tokenizer ...] [--out gates.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from test_prompts import QUESTIONS, STATE, _check, cpu_gates  # noqa: E402  (the CPU gates live in the unit tier)

FILLER = [{"kind": "noul", "instructions": f"Is the ticket id greater than {i}?"} for i in range(16)]


def _diffs(a, b):
    return [(float(np.abs(x - y).max()), int(np.argmax(x)) != int(np.argmax(y))) for x, y in zip(a, b)]


def mlx_gates(model, tokenizer, pad_to="block", pad_where="front"):
    from decisio.serve.mlx_engine import MLXLettersEngine

    eng = MLXLettersEngine(model, tokenizer=tokenizer, pad_to=pad_to, pad_where=pad_where)
    print("ENGINE", json.dumps(eng.facts()))
    served = eng.answer(STATE, QUESTIONS)[0]
    whole = []
    for q in QUESTIONS:  # the same prompt in one pass from an empty cache (prefix 0): not a served path
        rows, _ = eng._prepare_separate(STATE, [q])
        whole.append(eng.score_prompts(rows, prefix=0)[0][0])
    alone = [eng.answer(STATE, [q])[0][0] for q in QUESTIONS]
    filled = eng.answer(STATE, QUESTIONS + FILLER)[0][: len(QUESTIONS)]
    again = eng.answer(STATE, QUESTIONS)[0]
    d1 = _diffs(served, whole)
    record = {
        "facts": eng.facts(),
        "G1_max_abs_dp": max(v for v, _ in d1),
        "G1_flips": sum(f for _, f in d1),
        "G1_per_question": [v for v, _ in d1],
    }
    print(
        f"  G1 (reported) prefix path vs whole prompt: max|diff| {record['G1_max_abs_dp']:.2e}, flips "
        f"{record['G1_flips']}; per question " + ", ".join(f"{v:.1e}" for v, _ in d1)
    )
    ok = True
    for name, other in (("each question alone", alone), ("with 16 extra questions", filled), ("repeated", again)):
        same = all(np.array_equal(x, y) for x, y in zip(served, other))
        record[f"G2 {name}"] = {"bit_identical": same, "max_abs_dp": max(v for v, _ in _diffs(served, other))}
        ok &= _check(f"G2 {name}: bit-identical to the request's answers", same)
    record["G2_pass"] = bool(ok)
    print("  G3, G4: not applicable (no packed mode, no adapters on the MLX engine)")
    return ok, record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="an MLX conversion: run the MLX gates")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--out", default=None, help="write the MLX gates' record here (JSON)")
    args = ap.parse_args()
    cpu_ok = bool(cpu_gates(args.tokenizer))
    good = cpu_ok
    if args.model:
        mlx_ok, rec = mlx_gates(args.model, args.tokenizer)
        good &= mlx_ok
        if args.out:
            Path(args.out).write_text(json.dumps({**rec, "cpu_gates_pass": cpu_ok}, indent=1))
    sys.exit(0 if good else 1)
