# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The serving gates of the letters engine (decisio.serve.vllm_engine), run by test_serving_gates.py.

CPU part (runs anywhere with the tokenizer; no vLLM needed; also tests/unit/test_prompts.py):
  C1  a separate-mode prompt is token-identical to the letters prompt in chat format (readout.letters_prompt);
  C2  the state split lands just past the state's blank line, depends on the state alone, and padding
      inserts exactly enough tokens there for the shared prefix to be a multiple of the unit, leaving
      the question tokens untouched;
  C3  a pack's first turn is token-identical to that question's separate prompt, piecewise encoding
      equals whole-string encoding, and each read position is the final ":" of an "Answer:".

GPU part (`--gpu MODEL`, on the vLLM box):
  G1  cache hit versus miss: a question answered after its state was cached equals the same question
      with the prefix cache reset (kernel noise only);
  G2  isolation: adding unrelated questions to a request does not move an answer;
  G3  packed versus separate: a pack's first question equals its separate answer;
  G4  LoRA: with an adapter the answers move (it is applied, not silently ignored), and without one
      they equal the base engine's.

    python tests/gpu/serving_gates.py                                              # CPU gates
    python tests/gpu/serving_gates.py --gpu /models/Qwen3.6-35B-A3B-FP8-text [--adapter name=/path]   # G1, G2, G4
    python tests/gpu/serving_gates.py --gpu /models/Qwen3.6-35B-A3B-FP8-text --part packed           # G3, after that
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from decisio.readout.letters import chat_wrap, fmt_state, letters_prompt
from decisio.serve import vllm_engine as sv

STATE = ("Ticket 8842. Customer writes: our API integration started returning 500 errors on every "
         "request about 20 minutes ago, and we cannot process customer orders until this is fixed. "
         "Account tier: enterprise. Previous tickets this month: 3. Last deploy: 40 minutes ago.")
QUESTIONS = [
    {"kind": "choice", "instructions": "Which team should handle this?",
     "options": ["billing", "technical", "sales", "account"]},
    {"kind": "noul", "instructions": "Is the customer reporting an outage?"},
    {"kind": "score", "instructions": "How urgent is this?", "options": ["low", "medium", "high", "critical"]},
    {"kind": "noul", "instructions": "Is this urgent?"},
]


class _Tokonly(sv.LettersEngine):
    """The engine's prompt logic with a tokenizer and no model."""
    def __init__(self, tok, pad_unit=None, pad_where="between"):
        self.tok, self.pad_unit, self.pad_token, self.pad_where = tok, pad_unit, sv.PAD_TOKEN, pad_where


def cpu_gates(model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)            # noqa: E731
    ok = True

    eng = _Tokonly(tok)
    rows, P = eng._prepare_separate(STATE, QUESTIONS)
    for q, (ids, lab) in zip(QUESTIONS, rows):
        kind = q["kind"]
        opts = ["yes", "no"] if kind == "noul" else q["options"]
        prompt, cands = letters_prompt(tok, kind, fmt_state(STATE), q["instructions"], opts)
        ref = enc(chat_wrap(tok, prompt, "chat"))
        ok &= _check(f"C1 {kind:6s} prompt identical to eval", ids == ref)
        ok &= _check(f"C1 {kind:6s} labels", lab == [enc(c)[0] for c in cands])
    n_state = len(enc("<|im_start|>user\n" + STATE))
    ok &= _check(f"C2 split {P} just past the state's blank line (state alone encodes to {n_state})",
                 P in (n_state, n_state + 1) and tok.decode(rows[0][0][:P]).endswith(STATE + "\n\n"))
    other, P2 = eng._prepare_separate(STATE, QUESTIONS[::-1][:2])
    ok &= _check("C2 split depends on the state alone, not the question mix", P2 == P)

    for unit in (16, 544, 1056):
        padded, Pp = _Tokonly(tok, unit)._prepare_separate(STATE, QUESTIONS)
        k = Pp - P
        same = all(pi[:P] == ri[:P] and pi[P:Pp] == [sv.PAD_TOKEN] * k and pi[Pp:] == ri[P:]
                   for (pi, _), (ri, _) in zip(padded, rows))
        ok &= _check(f"C2 pad to {unit}: prefix {Pp} = multiple, {k} pads at the split, rest unchanged",
                     Pp % unit == 0 and 0 <= k < unit and same)

    head = len(enc(sv.USER_HEAD))
    for where, at in (("front", 0), ("user", head)):
        padded, Pp = _Tokonly(tok, 1056, where)._prepare_separate(STATE, QUESTIONS)
        k = Pp - P
        same = all(pi[at:at + k] == [sv.PAD_TOKEN] * k and pi[:at] + pi[at + k:] == ri
                   for (pi, _), (ri, _) in zip(padded, rows))
        shared = all(pi[:Pp] == padded[0][0][:Pp] for pi, _ in padded)
        ok &= _check(f"C2 pad {where}: prefix {Pp} = 1056 multiple, {k} pads at position {at}, prompt otherwise "
                     f"unchanged, prefix shared", Pp % 1056 == 0 and same and shared)

    turns = [(f"{STATE}\n\n{sv.question_text(tok, QUESTIONS[0])[0]}", rows[0][1])] + \
            [(sv.question_text(tok, q)[0], lab) for q, (_, lab) in zip(QUESTIONS[1:], rows[1:])]
    ids, pos = [], []
    for j, (content, _) in enumerate(turns):
        ids += enc((sv.TURN_SEP if j else "") + sv.user_turn(tok, content))
        pos.append(len(ids) - 1)
    whole = enc(sv.user_turn(tok, turns[0][0]) + "".join(sv.TURN_SEP + sv.user_turn(tok, c) for c, _ in turns[1:]))
    ok &= _check("C3 pack: piecewise encoding = whole-string encoding", ids == whole)
    ok &= _check("C3 pack: first turn = separate prompt", ids[:pos[0] + 1] == rows[0][0])
    colon = enc("Answer:")[-1]
    ok &= _check("C3 pack: every read position is the final ':' of 'Answer:'", all(ids[p] == colon for p in pos))
    print("CPU GATES", "PASS" if ok else "FAIL")
    return ok


def gpu_gates(model, adapters, part, handoff, tol_g1=2e-2, tol_g2=2e-2, engine_kw=None, pad_to="block",
              pad_where="between"):
    """Two processes, because vLLM does not reliably release a card between engines in one process:
    `separate` runs G1, G2, G4 and saves its answers to `handoff`; `packed` runs G3 against them."""
    ok = True
    # bf16 kernels are deterministic for a fixed batch but not batch-invariant: the no-cache batch-shape
    # floor measured on vLLM 0.29-0.30 is 1.3-1.8e-2 and up to 3.2e-2, see --part noise. G1 and G2 take
    # their tolerances from the command line; zero top-answer flips is required regardless.
    tol = 2e-2
    if part == "separate":
        eng = sv.LettersEngine(model, mode="separate", pad_to=None if pad_to == "none" else pad_to,
                               pad_where=pad_where, adapters=adapters or None, engine_kw=engine_kw)
        print("ENGINE", eng.facts())
        a, info = eng.answer(STATE, QUESTIONS)
        # padded: the whole shared prefix; unpadded: down to the last match-unit boundary inside it
        P = info["shared_prefix_tokens"]
        expect = P - 1 if eng.pad_unit else (P // eng.match_unit) * eng.match_unit
        ok &= _check(f"G1 prefix served from cache ({info['cached_tokens_mean']:.0f} of "
                     f"{info['prompt_tokens_mean']:.0f} tokens per question, shared prefix {P}, expected >= {expect})",
                     info["cached_tokens_mean"] >= expect)
        # same batch, no cache reads: every question computes the prefix itself, so the only difference
        # from `a` is where the prefix's state came from (a cache reset alone is not cold: vLLM runs one
        # request first and the other three hit its prefix, 792 of 1,089 tokens cached)
        rows, _ = eng._prepare_separate(STATE, QUESTIONS)
        cold_batch, cold_info = eng.score_prompts(rows, skip_cache=True)
        ok &= _check(f"G1 cold batch really cold ({cold_info['cached_tokens_mean']:.0f} cached tokens per question)",
                     cold_info["cached_tokens_mean"] == 0)
        eng.llm.reset_prefix_cache()
        b = [eng.answer(STATE, [q])[0][0] for q in QUESTIONS]                  # one at a time, cold
        d1 = max(float(np.abs(x - y).max()) for x, y in zip(a, b))
        filler = [{"kind": "noul", "instructions": f"Is the ticket id greater than {i}?"} for i in range(16)]
        c, _ = eng.answer(STATE, QUESTIONS + filler)
        d2 = max(float(np.abs(x - y).max()) for x, y in zip(a, c[:len(QUESTIONS)]))
        d1b = _diffs(a, cold_batch)
        m1b = max(v for v, _ in d1b)
        flips = any(f for _, f in d1b + _diffs(a, b) + _diffs(a, c[:len(QUESTIONS)]))
        ok &= _check(f"G1 cached vs cold, same batch (cache effect alone) max|diff| {m1b:.2e} (tol {tol_g1:.1e})",
                     m1b < tol_g1)
        print(f"  (G1 cached batch vs cold one-at-a-time, cache + batch shape: max|diff| {d1:.2e}; informational)")
        ok &= _check(f"G2 isolation with 16 extra questions max|diff| {d2:.2e} (tol {tol_g2:.1e})", d2 < tol_g2)
        ok &= _check("G1/G2 no top answer changed", not flips)
        for name, y in (("G1 per question", b), ("G2 per question", c[:len(QUESTIONS)])):
            print(f"  ({name}: " + ", ".join(f"{v:.1e}{' FLIP' if f else ''}" for v, f in _diffs(a, y)) + ")")
        for name in (adapters or {}):
            la, _ = eng.answer(STATE, QUESTIONS, adapter=name)
            moved = max(float(np.abs(x - y).max()) for x, y in zip(a, la))
            base_again, _ = eng.answer(STATE, QUESTIONS)
            back = max(float(np.abs(x - y).max()) for x, y in zip(a, base_again))
            # applied = the adapter moves answers by more than the isolation tolerance (twice this card's
            # measured batch noise), not a fixed 0.1: one adapter moved them 0.12-0.18 on the RTX PRO 6000
            # but 0.092 on the H200, where the base returns bit-exactly after it (a no-op moves nothing)
            ok &= _check(f"G4 adapter {name}: answers move by {moved:.2e} (applied, > {tol_g2:.1e})", moved > tol_g2)
            # the base must return bit-exactly (not within 0.02: the H200 bf16 path returned 1.58e-2, the
            # first-request effect, which a tolerance would have hidden)
            ok &= _check(f"G4 base after adapter max|diff| {back:.2e} (bit-exact)", back == 0.0)
        Path(handoff).write_text(json.dumps([[float(v) for v in x] for x in a]))
    elif part == "noise":
        ok = noise_floor(model)
    else:
        a = [np.array(x) for x in json.loads(Path(handoff).read_text())]
        pk = sv.LettersEngine(model, mode="packed", pack=len(QUESTIONS))
        print("ENGINE", pk.facts())
        p, _ = pk.answer(STATE, QUESTIONS)
        d3 = float(np.abs(p[0] - a[0]).max())
        ok &= _check(f"G3 packed first question vs separate max|diff| {d3:.2e}", d3 < tol)
        later = max(float(np.abs(x - y).max()) for x, y in zip(a[1:], p[1:]))
        print(f"  (packed later questions vs separate: max|diff| {later:.2e}; cross-question contamination, reported)")
    print(f"GPU GATES {part}", "PASS" if ok else "FAIL")
    return ok


def _diffs(a, b):
    """Per question: max |delta p| and whether the top answer changed."""
    return [(float(np.abs(x - y).max()), int(np.argmax(x)) != int(np.argmax(y))) for x, y in zip(a, b)]


def noise_floor(model):
    """What the bf16 kernels alone do to an answer, with prefix caching off and no padding, so the
    cached-vs-cold (G1) and isolation (G2) differences can be judged against a measured floor rather
    than a guessed tolerance. Reports only; the tolerance decision is the owner's."""
    eng = sv.LettersEngine(model, mode="separate", engine_kw={"enable_prefix_caching": False})
    print("ENGINE", eng.facts())
    filler = [{"kind": "noul", "instructions": f"Is the ticket id greater than {i}?"} for i in range(16)]
    one = [eng.answer(STATE, [q])[0][0] for q in QUESTIONS]
    again = [eng.answer(STATE, [q])[0][0] for q in QUESTIONS]
    batch = eng.answer(STATE, QUESTIONS)[0]
    with_f = eng.answer(STATE, QUESTIONS + filler)[0][:len(QUESTIONS)]
    for name, x, y in (("N1 one-at-a-time, repeated (determinism)", one, again),
                       ("N2 one-at-a-time vs batched (batch shape)", one, batch),
                       ("N3 batched vs batched + 16 extra questions", batch, with_f)):
        d = _diffs(x, y)
        print(f"  {name}: max|diff| {max(v for v, _ in d):.2e}; per question "
              + ", ".join(f"{v:.1e}{' FLIP' if f else ''}" for v, f in d))
    print("NOISE_DONE")
    return True


def _check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    return bool(cond)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--gpu", default=None, help="model path: run the GPU gates")
    ap.add_argument("--adapter", action="append", default=[])
    ap.add_argument("--part", default="separate", choices=["separate", "packed", "noise"])
    ap.add_argument("--handoff", default="/tmp/decisio_serving_gates_separate.json")
    ap.add_argument("--engine", default="{}", help="extra LLM(...) keyword arguments as JSON (separate part)")
    ap.add_argument("--pad-to", default="block", help="'block', a token count, or 'none' (separate part)")
    ap.add_argument("--pad-where", default="between", choices=sv.PAD_PLACES, help="pad placement (separate part)")
    ap.add_argument("--tol-g1", type=float, default=2e-2, help="cache effect alone (same batch), max |delta p|")
    ap.add_argument("--tol-g2", type=float, default=2e-2, help="isolation, max |delta p|")
    args = ap.parse_args()
    def _main():
        good = cpu_gates(args.gpu or args.tokenizer)
        if args.gpu:
            good &= gpu_gates(args.gpu, dict(a.split("=", 1) for a in args.adapter), args.part, args.handoff,
                              args.tol_g1, args.tol_g2, json.loads(args.engine), args.pad_to, args.pad_where)
        return good
    sys.exit(0 if sv.run_or_die(_main) else 1)
