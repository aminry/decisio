# Lab 2's card session of 2026-10-15 (S12): the parts that concern decisio's public documents

Run by Lab 2 on one RTX PRO 6000 Blackwell Workstation Edition (600 W enforced, driver 595.104.02, AMD EPYC 9534 host), 3.60 card hours; first-contact gate passed (`first_contact.json`).
Stack: decisio 0.12.0 (`f5e40e0f`) on vLLM 0.31.0; parts X and X5 ran on the head of #179 (`X/commit.txt`).
Lab 1's parts of the same session (L, I, J of the teaching study) are Lab 1's and not in this record.
Item-level files that name a private request set are not here: the 12B control's answers, the long-context rows, the certified-answers server logs and answers; the W2 verdict keeps its aggregates and the counts of near-tie and out-of-range items (`W2/summary.json`).
`manifest.json` lists every file with its sha256.

| Part | What was run | Result | Files |
| --- | --- | --- | --- |
| C | the default 12B, two fresh starts, 200 single-question requests | equal on 200 of 200, fingerprints identical | `C/` |
| W2 | the 12B, `--multi-question sequential` (three starts) against `warm` (two), suite, JevBench public items, travel steps | **fail**: suite largest difference 0.1931 against the bound 0.1866, warm accuracy 0.7307 against 0.7329 to 0.7343; JevBench passes (0.0540 against 0.0645); travel 0.2039 against 0.2016 with one choice outside the sequential range; speed 731 ms warm against 1,598 ms sequential; warm repeats itself exactly | `W2/` |
| K2 | the 12B, certified answers across restarts at alpha 0.15 | **holds**: threshold 0.9912, 166 certified, accuracy of the 200 questions 0.83, 10.2% of the certified wrong; five restart comparisons (two re-registrations, three imports) with equal threshold, certificate id and stack digest, every import `ok`, no choice or flag or top probability differing; the alpha 0.25 fallback was not needed | `K2/` |
| H | the head's reading history, 31B then 12B, 40 items, three registrations | 31B: `h` bitwise equal on 40 of 40, the label log-probabilities equal on 39 and different on one (by 0.73), offline head against live head at most 0.0013 with no answer changed, one head digest and two calibration digests; 12B: all equal, one head digest and one calibration digest | `H/` |
| F | the stored FP8 31B (`tachara-ai/decisio-gemma-4-31b` at `6673a7f4`) against Google's bf16 at `842da379` quantized on load, one start each, the same 200 requests | read below | `F/` |
| X | `--max-model-len` at 65,536 and 131,072 on the 12B, at 65,536 on the 31B, each with a smoke accuracy check on 30 items (the same items short and at length) | 12B starts at both lengths (154 s and 156 s; 57.53 GiB of KV at 131,072 and 65,536 alike, 333,114 tokens at 65,536); 12B at 60,000 tokens: 23 right against 26 short, first question 11.0 s, second 0.29 s; at 120,000: 19 right against 26 short, first 23.2 s, second 0.57 s; 31B at 65,536 with the model's own KV dtype and the default share: 48.05 GiB of KV, 102,255 tokens, at 60,000: 24 right against 25 short, first 23.0 s, second 0.29 s | `X/` |
| X5 | the 31B at `--max-model-len` 98,304, the model's own KV dtype, the check at 85,000 tokens | starts (48.05 GiB, 141,865 tokens, concurrency 1.44 at 98,304); 23 right against 25 short, first question 40.8 s, second 0.40 s | `X5/` |
| P | 50 questions about a 3,000-token document against one plain read of the same 4,843 tokens, the 12B, 10 documents | median 1,598 ms sequential and 731 ms warm against 461 ms for the plain read (ratios 3.47 and 1.59) | `P/` |
| J (score comparison) | Lab 2's comparison of the stored FP8 against the committed on-load 31B record and against F's on-load arm | see below | `J/` |

## F, read tensor by tensor

`F/f_a.fp.json` and `F/f_b.fp.json` are each engine's fingerprint: for every tensor its dtype, shape and sha256.
The script that reads them and the answers (the private RLCD repository's `experiments/2026-10-15_serve_s12_f_analysis/f_analysis.py`) is not copied here; its output is `F/analysis_f_analysis.json`.

- All 1,145 tensors have the same bytes in the two engines, including all 240 FP8 weights and every scale tensor.
- `compare-fp` lists 420 tensors as different because it compares shapes too: the 240 `weight_scale` tensors are 0-d in the stored engine and one element in the on-load engine, and the 180 attention `_q_scale`, `_k_scale` and `_v_scale` tensors are the other way round.
  Their sha256 are equal, so the verdict file's "DIFFERENT" is a difference of shape and not of value.
- The two engines' configurations in `F/f_a.log` and `F/f_b.log` are identical except the model path, the revision and `quantization` (compressed-tensors against fp8); both selected `CutlassFP8ScaledMMLinearKernel`.
- The answers are not equal: of 200, none exactly (median difference in a probability 0.0036, 90th percentile 0.052, largest 0.371), and 4 top answers differ (`mmlu_pro_text:39`, `mmlu_pro_text:41`, `banking77:89`, `banking77:93`).
- On the suite's gold, 162 of 200 are right in each arm (0.810); the expected calibration error is 0.075 stored and 0.080 on load (10 equal-width bins; `J/score_compare.json`).
- No same-arm control was run for the 31B in this session (the control of part C is the 12B). The stored 31B's own start-to-start repeat was measured on 2026-10-13, on another box: two starts equal on 200 of 200 on vLLM 0.31.0 (`EVAL_CARD.md` section 7.5), so a difference of this size between two arms is not that variation unless the on-load arm differs; an on-load control on this stack was not run.
  On vLLM 0.30.0 the same two arms were bit for bit equal on 2,175 answers (the gate G2 of 2026-10-08, `EVAL_CARD.md` section 7).

## J, the score comparison

`J/score_compare.json` compares the stored FP8's JevBench run of this session (v0.12.0) with the committed on-load record `runs/2026-10-04_gemma-4-31b/jevbench_g31/` (decisio `249b56d`, vLLM 0.30.0) and F's two arms.
Accuracy is 213 of 231 on both, tier by tier (1.000, 1.000, 0.838); Lab 2 reports the stored FP8's JevBench run bit-identical to Lab 1's frozen stored-FP8 run of 2026-10-13 (that run is not in this record).
The expected calibration error and the score type of the v1.5 reading differ from the on-load record's, which is older software, so those gaps are not attributed to the checkpoint.

## Not in this record

The cause of the answer differences in F is not isolated.
A restart control of the 31B and an on-load JevBench on the same stack are the two cheapest next steps.
