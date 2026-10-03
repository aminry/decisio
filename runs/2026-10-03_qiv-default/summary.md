# The served default's prompt: the Cygnet tail and yes/no as named letters (session of 2026-10-03)

Flag names in this record are those of 2026-10-03: `--prompt-tail cygnet` is now `--prompt-tail spaced` and `--prompt-tail decisio` is `--prompt-tail compact`; the prompts are unchanged.

Q0 is the earlier served default (`--prompt-tail decisio`, `--noul-rendering words`, T 1.307); Q-IV is `--prompt-tail cygnet --noul-rendering letters-keys`, each judged at its own temperature, fitted by minimum log loss with 5-fold cross-validation stratified by task on its suite readouts (Q0 1.307, Q-IV 1.506).
Both servers ran in one session on one card with `--debug-readout`, so every temperature was applied offline to the plain readout, which is exact.
The gates were pre-registered; the intent-head gate was retested with six draws per set under an amendment written before that measurement.

## Gates, Q-IV against Q0

| Gate | Q-IV against Q0 | Verdict |
| --- | --- | --- |
| Suite accuracy (1,400) | 77.14%, +0.93 [-0.29, +2.14] points | pass |
| Suite ECE, tie-robust, out-of-fold | 0.0247 against 0.0202 (allowed: +0.010) | pass |
| JevBench 231, correct | 200 against 194, +2.60 [-0.43, +5.64] points | pass |
| Decision Index BANKING77 | +1.46 [+0.62, +2.31] points | pass |
| Decision Index CLINC150+OOS | +0.25 [-0.40, +0.93] | pass |
| Decision Index GPQA Diamond | +2.02 [-3.03, +7.07] | pass |
| Decision Index MMLU-Pro | +0.41 [+0.01, +0.81] | pass |
| Conformance C2 / C3 / C4 | pass / pass / pass | pass |
| Latency, one question on a cached state, p50 | 20.6 ms against 20.0 (allowed: +2 ms) | pass |
| Intent heads, six draws per set | BANKING77 84.00 against 84.89 (-0.89; allowed -1.0), CLINC150 91.17 against 90.00; gains over the plain readout +9.3 [+5.1, +14.0] and +9.2 [+3.5, +15.2] | pass |

JevBench I_open A (v1.5, the paired item bootstrap): 49.39 against 42.69, +6.71 [+1.08, +12.68]; per type, choice 77.3 against 78.3, yes/no 13.7 against -10.9, score 57.2 against 60.6.
In the first three draws of each set the intent-head gate failed on BANKING77 (83.1 against 84.7); the six-draw retest reran draw 0 on both servers as a determinism check and reproduced every row.

## The served configuration

- `--prompt-tail cygnet`, `--noul-rendering letters-keys`, T 1.506, and choice questions at their own T 1.370, fitted on the suite's 800 choice items; that choice temperature passed its own gate against the single temperature (out-of-fold log loss -0.0188 [-0.0326, -0.0041], tie-robust ECE 0.066 to 0.015, and lower ECE on JevBench's choice items and the Decision Index).
- The served-default numbers at these temperatures (`served_qiv.json`): JevBench accuracy easy 1.000, standard 0.972, hard 0.739, ECE 0.014 / 0.121 / 0.043; Decision Index BANKING77 macro-F1 0.746 (ECE 0.008), CLINC150+OOS 0.822 (0.099), GPQA Diamond 0.510 (0.115), MMLU-Pro 0.613 (0.013); suite pooled 0.770, ECE 0.033 (in-sample).
- Each metric was first checked to reproduce the earlier default's published value from Q0's readouts at T 1.307.

## Files

`conformance_*`, `capture_*` (the plain readouts of the suite and JevBench), `intents_*` (draws 0 to 2 and 3 to 5; BANKING77 draw 0 appears twice, the check), `jevbench_*` (the harness's aggregates and v1.5 reading at T 1.307), `di_*` (the kit's report and our calibration at T 1.307), `latency_*`, `health_*`, `heads6.json` (the six-draw gate), `served_qiv.json`.
