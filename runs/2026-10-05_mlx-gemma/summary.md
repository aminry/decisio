# The Gemma base on MLX: the gates at 6, 4 and 8 bits on a Mac (2026-10-04 and 2026-10-05)

The Gemma base (`--base gemma-4-12b`) served by `--backend mlx` from the mlx-community conversions of `google/gemma-4-12B-it`, gated against the base's vLLM bf16 record `runs/2026-10-04_gemma-base`, on one Apple M5 Pro (64 GB).
Pre-registered before any measurement (the RLCD repository, `experiments/2026-10-03_t2_gemma_mac/PREREG.md`), with the shape of the Qwen re-gate (`runs/2026-10-03_mlx-regate`).
The gates ran on the branch at `e7d20fc` (v0.4.0 plus this change); the 50-item checks were repeated on `cf4a8b6`, the same code rebased onto main.

## Gates

| Gate | 6-bit | 4-bit | 8-bit | vLLM bf16 |
| --- | --- | --- | --- | --- |
| S1: suite accuracy (1,400 items) | 0.7357, +0.1 [-0.7, +0.9] points: pass | 0.7093, -2.6 [-4.1, -1.0]: **fail** | 0.7336, -0.1 [-0.6, +0.3]: pass | 0.7350 |
| S2: top-answer flips | 56 | 214 | 23 | |
| S3: pooled ECE (allowed: within 0.01) | 0.0286: pass | 0.0180: **fail** (-0.0108) | 0.0286: pass | 0.0288 |
| Brier (reported) | 0.362, +0.000 [-0.004, +0.005] | 0.398, +0.036 [+0.023, +0.050] | 0.362, -0.000 [-0.002, +0.002] | 0.362 |
| H: BANKING77 intent head, six draws (of 150) | 128 / 128 / 126 / 128 / 124 / 127, mean 0.846: pass, +1.3 [-0.8, +3.6] | not run | not run | 127 / 124 / 125 / 126 / 121 / 126, mean 0.832 |
| H: CLINC150 intent head, six draws (of 100) | 90 / 89 / 92 / 91 / 90 / 89, mean 0.902: pass, -0.7 [-2.0, +0.5] | not run | not run | 91 / 90 / 91 / 91 / 91 / 91, mean 0.908 |
| The heads' gain over the plain readout (points) | +11.9 / +5.2 | | | +8.6 / +5.8 |
| C2 / C3 / C4 conformance | pass / pass (0 of 1,400 differ) / pass (largest probability difference 0.0) | not run | not run | pass |
| B1: 50 suite items on a fresh server | 50 of 50 bit-identical to the gate run (on `e7d20fc` and on `cf4a8b6`) | | | |
| B2: the Qwen base's 6-bit path on the same code | 50 of 50 bit-identical to `runs/2026-10-03_mlx-regate` (on `e7d20fc` and on `cf4a8b6`) | | | |

BANKING77's draw 0 was asked twice and repeated bit for bit.
Scoring as the Qwen re-gate: the reference's served answers rebuilt from its plain readouts at T 3.592 (reproducing its 0.735 and 0.029), yes/no as [P(yes), P(no)], argmax, ECE over 10 equal-mass bins, paired bootstrap intervals (4,000 draws, seeded from the data).

**The default:** 6-bit, which passes every gate.
- 4-bit fails suite accuracy, so there is no 4-bit option. 6-bit runs on a 32 GB Mac itself (below).
- 8-bit matches 6-bit on the suite at more memory.

**Deviation from the pre-registration:** the 4-bit run was stopped after gate S. With S1 failed, the pre-registered rule already excluded 4-bit as the 32 GB option, and its heads, conformance and latency (about 10 hours) could not change that. 8-bit was pre-registered for gate S only.

## Memory (6-bit, in process, `6bit/memory.json`)

- Weights: 9.7 GB.
- Peak: 11.9 GB at 8,056 tokens of state, 12.8 GB at 16,383, and 15.1 GB at 32,687 (one question), with no swap growth.
- The memory driver is the Qwen record's (`runs/2026-10-02_mlx-backend`, 30.7 GB at 32,761 tokens for the Qwen base at 6 bits).

## Latency (6-bit, server medians)

The Mac was not quiet within the 60 minutes the pre-registration allowed, so the measurement went ahead with each pass's load recorded.
- **`latency.py`** (the prefix cache off): foreign processes peaked at 100%, 124% and 121% CPU over its three passes, and the passes agree within 4%. Pass 3:
  - one question on a new state: 392 ms;
  - per question, 10 sharing a state: 179 ms; 100 sharing a state: 163 ms; 100 on an 8,000-token state: 285 ms.
- **The cached-state cell** (a fresh default server, 30 states asked twice) met the idle rule: 392 ms on first sight and 156 ms when the state was seen before, every hit equal to its miss.
- **Track 7's driver** (`6bit/latency_t7.json`, the reference's own), against the reference card's:
  - one question on a cached state: 165 ms (26.7);
  - a new state, one question: 554 / 1,299 / 3,585 ms at 300 / 1,000 / 3,000 tokens (39 / 102 / 293);
  - four questions: 1,025 / 1,771 / 4,117 ms (131 / 196 / 392).
- **Against the Qwen base on this Mac** (`runs/2026-10-03_mlx-regate/6bit_qiv/latency_t7.json`, also under load): one question on a new 1,000-token state takes 1,299 ms on the Gemma base and 704 ms on the Qwen base, a dense 12B model against about 3B active parameters.
- **Registration:** an intent task takes about twice the Qwen base's time on this Mac: BANKING77 22 minutes and CLINC150 57 minutes per draw.

## Files

- `6bit/`: the suite answers, the intent heads (`intents.json.gz` draws 0 to 2; `intents6.json.gz` BANKING77 draw 0 again and draws 3 to 5), conformance, the scored gates (`report.json`), latency, memory, each server's `/health` (A: every flag the default; L: the latency server; C: the prefix cache off), and logs.
- `4bit/`, `8bit/`: the suite answers and the scored gate S.
- `check_b_e7d20fc/`, `check_b_cf4a8b6/`: the 50-item checks.
