# Where vLLM's engine runs (session of 2026-10-05)

One RTX PRO 6000 Blackwell Workstation Edition, vLLM 0.30.0, decisio v0.6.0 servers; every checkpoint at its pinned revision.
"In-process" is the engine in the server's process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, what `--engine-process in` sets); "separate" is vLLM's own arrangement, the engine in a process of its own.
The browser-agent demo's travel steps are published as aggregates only (`manifest.json`); everything else is computed from this folder's files.

## The issue's script (vllm-project/vllm#59764), stock vLLM, 20 prompts of 1,400 random tokens

| checkpoint | engine | alone_repeat | alone_vs_batch2 | batch2_repeat | batches with both rows equal to the lone result | batches with the two rows equal to each other |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP8 | separate | 0.0 | 0.105 | 0.145 | 34 of 40 | 40 of 40 |
| FP8 | in-process | 0.0 | 0.198 | 0.0 | 0 of 40 | 40 of 40 |
| bf16 (`max_num_seqs` 256) | separate | 0.0 | 0.149 | 0.140 | 16 of 40 | 40 of 40 |
| bf16 (`max_num_seqs` 256) | in-process | 0.0 | 0.149 | 0.0 | 0 of 40 | 40 of 40 |

The run-to-run difference is batch composition: the separate engine receives a batch's requests one at a time and starts a step with whatever has arrived, so the two copies sometimes share a step and sometimes do not.
On this card two identical rows in a shared step are bit-identical to each other; forward hooks on all 602 per-token module outputs of the in-process FP8 engine (eager, no compilation, prefix caching off) found no op where they diverge (`hooks.json`).

## --multi-question warm with the engine in-process, on 272 four-question requests (5 repeats each)

| | Qwen3.6-35B-A3B FP8 | Gemma 4 31B | Gemma 4 12B |
| --- | --- | --- | --- |
| an identical request repeats exactly | 272 of 272 | 272 of 272 | 68 of 272: the first answer differs from repeats 2 to 5, which agree (max \|dp\| 0.046) |
| with 4 concurrent clients sending different requests | 272 of 272 identical to the first answer | 272 of 272 | follows the prefix cache's history |
| against each question alone: requests differing (operation, click target, select target) | 272, 272, 271 | 0, 0, 6 (at most 0.0046) | 272, 272, 272 (cached against cached) |
| choices changed against each question alone (same order) | 21, 25, 11 | 0, 0, 0 | 3, 5, 0 |

The same server on Qwen with the engine separate repeated exactly on 22 of 272 requests (choices changed on 52, max |dp| 0.404).
On Gemma 4 12B the movement left in-process is the base's own: a state read for the first time and the same state read from the prefix cache can differ (EVAL_CARD.md section 6.5).

Accuracy against the demo's acceptable actions (paired over steps, 95% bootstrap interval, points, batched minus alone):

| | operation (272 steps) | click target (200) | select target (108) |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | 86.8% against 85.3%, +1.5 [-1.1, +4.0] | 49.0% against 52.5%, -3.5 [-7.5, +0.0] | 100% both |
| Gemma 4 31B | 91.2% both | 71.5% both | 100% both |
| Gemma 4 12B (cached answers) | 98.2% both, +0.0 [-1.1, +1.1] | 68.5% against 67.0%, +1.5 [+0.0, +3.5] | 100% both |

## Latency (server time, median of 20 per cell, ms)

| cell | Qwen: warm in-process / sequential separate / sequential in-process | Gemma 4 31B | Gemma 4 12B |
| --- | --- | --- | --- |
| one question, cached 1,000-token state | 18.8 / 19.6 / 18.8 | 36.9 / 44.8 / 36.7 | 27.9 / 34.8 / 27.4 |
| four questions, new 300-token state | 56.0 / 103.7 / 99.9 | 120.9 / 192.1 / 180.6 | 90.3 / 140.8 / 128.8 |
| four questions, new 1,000-token state | 57.3 / 104.8 / 101.0 | 209.4 / 289.8 / 278.7 | 147.9 / 205.6 / 197.4 |
| four questions, new 3,000-token state | 108.4 / 162.1 / 154.7 | 544.1 / 623.9 / 612.1 | 318.9 / 395.2 / 383.5 |

## The served default in-process against separate

On every base the 272 four-question answers and their 1,088 questions sent alone were bit-identical between the two arrangements, and in-process was faster in most cells (Qwen -7.4 to +0.5 ms per cell, Gemma 4 31B -11.8 to -0.4, Gemma 4 12B -12.1 to +1.3; `rules/`).
On Qwen and Gemma 4 31B every four-question answer also equalled its questions sent alone; on Gemma 4 12B 104 of 272 did, the same on both arrangements, because the four questions were each state's first read and the lone questions cached reads (section 6.5).
