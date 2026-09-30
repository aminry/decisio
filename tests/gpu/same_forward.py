# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The single-engine hidden readout's same-forward gate on a card (docs/design/hidden-state-readout.md, gate part 2):
for each question, the label probabilities recomputed from the hidden state recovered from the reserved columns
(`SingleEngineHidden.readout`) against the engine's own label probabilities for the same question
(`LettersEngine.answer`), within 1e-4 and the same option on every question. The questions are the latency grid's
77-option intent questions (BANKING77's intent names) over fresh support-ticket states.

    VLLM_USE_DEEP_GEMM=0 python tests/gpu/same_forward.py --model $DECISIO_MODEL [--n 60]
"""

import argparse
import json
import sys

import numpy as np

TOL = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from decisio.bench.latency import LABELS, questions, state_ids
    from decisio.serve.hidden_engine import SingleEngineHidden
    from decisio.serve.vllm_engine import SERVED_ENGINE, LettersEngine, engine_kwargs

    args = argparse.Namespace(engine=json.dumps(SERVED_ENGINE), model_class="hidden-readout")
    eng = LettersEngine(a.model, pad_to="block", pad_where="front", engine_kw=engine_kwargs(args))
    hidden = SingleEngineHidden(eng, a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    labels = json.load(open(LABELS))["banking77"]
    worst, same = 0.0, 0
    for i in range(a.n):
        state = tok.decode(state_ids(tok, 500 + 37 * i, i))
        q = questions("choice77", 1, labels, 10**6 + i)
        (p,), _ = eng.answer(state, q)  # the engine's own label probabilities
        ((lp, h),) = hidden.readout(state, q)  # recomputed from the recovered hidden state
        worst = max(worst, float(np.abs(p - np.exp(lp)).max()))
        same += int(np.argmax(p) == np.argmax(lp))
    ok = worst <= TOL and same == a.n
    print(
        f"SAME_FORWARD {'PASS' if ok else 'FAIL'}: {a.n} questions, max |dp| {worst:.2e} (tolerance {TOL:.0e}), "
        f"same option on {same} of {a.n}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    from decisio.serve.vllm_engine import run_or_die

    sys.exit(run_or_die(main))
