# The MLX backend re-gated at 6 bits under the current served default (2026-10-03)

The served default changed in #54 (the spaced layout, yes/no as named letters, T 1.506 with choice T 1.370), so the MLX gates of `runs/2026-10-02_mlx-backend`, measured under the earlier default, were run again at 6 bits on the same Mac.
Reference: the FP8 record of the new default, `runs/2026-10-03_qiv-default`.
Pre-registered before any measurement (the RLCD repository, `experiments/2026-10-02_t2_mlx_backend/PREREG.md`, addenda E, F and G).
Every gate ran on decisio `62dfe1f`; #56's renames were then checked on `4f5c704`, and the latency was measured on `4f5c704`.

## Gates

| Gate | MLX 6-bit | FP8 | Verdict |
| --- | --- | --- | --- |
| S1: suite accuracy (1,400 items) | 0.7721 | 0.7700 | pass: +0.2 [-1.0, +1.4] points |
| S2: top-answer flips | 102 of 1,400 | | counted |
| S3: pooled ECE (allowed: within 0.01) | 0.0451 | 0.0327 | **fail:** +0.0125 [-0.004, +0.026] |
| Brier (reported) | 0.318 | 0.313 | +0.005 [-0.004, +0.014] |
| H: BANKING77 intent head, six draws (of 150) | 124 / 124 / 125 / 127 / 128 / 129, mean 0.841 | 126 / 126 / 122 / 127 / 126 / 129, mean 0.840 | pass: +0.1 [-1.8, +2.0] points |
| H: CLINC150 intent head, six draws (of 100) | 91 / 91 / 91 / 91 / 92 / 91, mean 0.912 | 88 / 92 / 91 / 91 / 91 / 94, mean 0.912 | pass: 0.0 [-3.0, +2.8] points |
| C2 / C3 / C4 conformance | pass / pass (0 of 1,400 differ) / pass (largest probability difference 0.0) | | pass |
| G1: #56's renames, `/health` | equal to the gate server's except the renamed tail (`spaced`) and the removed system field | | pass |
| G2: #56's renames, 50 suite items | 50 of 50 bit-identical to the gate run | | pass |

The intent heads' BANKING77 draw 0 was asked twice and repeated bit for bit.
Conformance ran with T 1.307 for every type, as the FP8 record's conformance server did, because the conformance tool reads only the global temperature.

## Why the ECE fail changes no setting

The FP8 ECE is in-sample: the served temperatures were fitted on those FP8 suite readouts, and MLX is scored at temperatures fitted to another stack.
Fitted the same way on the MLX readouts, MLX would take choice T 1.472 and yes/no T 1.646; at those its suite ECE is 0.0328 out-of-fold, against FP8's own 0.0384 out-of-fold.
That candidate was then judged on JevBench's 231 public items, which neither fit saw (pre-registered as addendum F, `6bit_qiv/gate_f.json`).
There it makes log loss worse (+0.0033 [-0.0019, +0.0079] against the served temperatures), so it fails, and the served temperatures stay.
At the served temperatures MLX is already calibrated as well as FP8 on those items: ECE 0.033 against 0.042, accuracy 0.861 against 0.870.
The suite's largest ECE rises are SciFact clarified (0.070 to 0.118), SciFact (0.105 to 0.131) and MMLU-Pro x350 (0.022 to 0.042).

## Latency (decisio `4f5c704`, Apple M5 Pro, 64 GB)

| Server time, p50 | Earlier default (`runs/2026-10-02_mlx-backend`) | Current default |
| --- | --- | --- |
| One question, cold | 230 ms | 266 ms |
| One question on a state seen before (prefix cache hit) | 85 ms | 111 ms |
| 100 questions sharing a state, per question | 95 ms | 130 ms |
| 10 questions sharing a state, per question | 89 ms | 126 ms |
| 100 questions on an 8,000-token state, per question | 143 ms | 179 ms |

- The drivers and run counts are the earlier record's: `latency.py` on a server with the prefix cache off (three passes; the table shows the last), and the prefix cache's 1-question cell on a fresh default server (30 requests, every hit's answers equal to its miss's).
- The Mac was not idle by the drivers' rule: other processes peaked at 111% to 250% CPU (indexing, a browser, another job). The three passes agree within 1% for one question and within 6% for 100.
- The earlier figures were measured under the earlier served prompt; how much of the rise is the current prompt and how much is load was not separated.
- A first attempt stopped after its first pass (`latency/latency_interrupted_pass1.log`, 269 ms for one question); the run above is the second.
- An earlier latency pass during the gate run, under heavier load (up to 915% CPU), is in `6bit_qiv/latency.log` and is not used.

## Files

- `6bit_qiv/`: the gate run. The suite answers (`suite.json.gz`), the intent heads (`intents.json.gz`, draws 0 to 2; `intents6.json.gz`, BANKING77 draw 0 again and draws 3 to 5), conformance, the scored gates (`report.json`), the held-out check (`gate_f.json`, `capture_jevbench.json.gz`), the exploratory refit (`diag_temperature.json`), each server's `/health` (A: every flag the default; B2: conformance; C: the prefix cache off), and logs.
- `check_56/`: #56's check (G1, G2).
- `latency/`: the latency above.
