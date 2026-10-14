# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The single-engine hidden readout's same-forward gate on a card (docs/design/hidden-state-readout.md, gate part 2):
for each question, the label probabilities recomputed from the hidden state recovered from the reserved columns
(`SingleEngineHidden.readout`) against the engine's own label probabilities for the same question
(`LettersEngine.answer`) under like request histories, and the same option on every question. The questions are the
latency grid's 77-option intent questions (BANKING77's intent names) over fresh support-ticket states.

Like request histories: a first read of a state and a read of it from the prefix cache are different computations on
this stack (the row is prefilled in different pieces, so bf16 rounds differently; EVAL_CARD 6.5), and the check is
about the head, not the cache. So each question is read once plainly first, which caches its state, and the two reads
compared are both from the cache: the head's (`SingleEngineHidden.readout`) and the plain readout's again. The first
read's difference from the cached read is reported, not gated.

The tolerance follows how the base takes its label log-probabilities (decisio.families.Family.head_label_logprobs):
  recompute (Qwen)   recomputed from the hidden state: within 1e-4 (a card measured 3.1e-8)
  engine (Gemma)     the engine's own readout of the row, so the two reads are one computation: within 1e-6 (the
                     2026-10-04 cards measured 5.6e-17 on both bases)

    VLLM_USE_DEEP_GEMM=0 python tests/gpu/same_forward.py --model $DECISIO_MODEL [--n 60]
"""

import argparse
import json
import sys

import numpy as np

TOL = {"recompute": 1e-4, "engine": 1e-6}


def verdict(worst, same, n, source):
    """(ok, tolerance): the largest difference within the base's tolerance, and the same option on every question."""
    tol = TOL[source]
    return worst <= tol and same == n, tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from decisio.bench.latency import LABELS, questions, state_ids
    from decisio.serve.hidden_engine import SingleEngineHidden
    from decisio.serve.vllm_engine import SERVED_ENGINE, LettersEngine, engine_kwargs

    args = argparse.Namespace(engine=json.dumps(SERVED_ENGINE), model_class="hidden-readout", model=a.model)
    eng = LettersEngine(a.model, pad_to="block", pad_where="front", engine_kw=engine_kwargs(args))
    hidden = SingleEngineHidden(eng, a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    labels = json.load(open(LABELS))["banking77"]
    worst, same, first = 0.0, 0, 0.0
    for i in range(a.n):
        state = tok.decode(state_ids(tok, 500 + 37 * i, i))
        q = questions("choice77", 1, labels, 10**6 + i)
        (p_first,), _ = eng.answer(state, q)  # a first read: caches the state; reported, not gated
        ((lp, h),) = hidden.readout(state, q)  # the head's label log-probabilities, from the cache
        (p,), _ = eng.answer(state, q)  # the engine's own label probabilities, from the cache
        worst = max(worst, float(np.abs(p - np.exp(lp)).max()))
        first = max(first, float(np.abs(p_first - p).max()))
        same += int(np.argmax(p) == np.argmax(lp))
    source = hidden.label_source
    ok, tol = verdict(worst, same, a.n, source)
    print(
        f"SAME_FORWARD {'PASS' if ok else 'FAIL'}: {a.n} questions, label source {source}, max |dp| {worst:.2e} "
        f"(tolerance {tol:.0e}), same option on {same} of {a.n}; "
        f"a first read against the cached read: max |dp| {first:.2e}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    from decisio.serve.vllm_engine import run_or_die

    sys.exit(run_or_die(main))
